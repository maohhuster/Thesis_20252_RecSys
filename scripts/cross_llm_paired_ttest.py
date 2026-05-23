#!/usr/bin/env python3
"""cross-LLM analysis — paired t-test of GPT-4o-mini vs Claude features for M4 and M7.

Reads results_cross_llm/<config>_<llm>/seed-N/results.json for all 20 cells, then for
each (config, metric) computes:
  - 5-seed mean ± std for Claude and GPT
  - Δ = GPT − Claude (absolute and percentage)
  - Paired t-test (per-seed pairing) p-value
  - 95% CI on the paired difference

Outputs results_cross_llm/cross_llm_summary.json + a Markdown table for the paper appendix.

Usage:
    python3 scripts/cross_llm_paired_ttest.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.stats as st

REPO = Path(__file__).resolve().parent.parent
RES_ROOT = REPO / "code" / "benchmark" / "results"  # ML-20M base convention
SUMMARY_DIR = REPO / "code" / "benchmark" / "results_cross_llm"  # group-summary umbrella
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = SUMMARY_DIR / "cross_llm_summary.json"
MD_OUT  = SUMMARY_DIR / "cross_llm_paper_table.md"
ENCODER = "bge-large-en-v1.5"

CONFIGS = ["m4", "m7"]
LLMS    = ["claude", "gpt"]
SEEDS   = [42, 123, 456, 789, 2026]
METRICS = ["NDCG@10", "Recall@10", "MRR"]


# Claude per-seed M4/M7 values are the paper-canonical numbers — the *exact*
# values reported in the main results table (bge-large-en-v1.5 encoder),
# released here as scripts/_claude_m4_m7_per_seed.json. No re-training required,
# which avoids numerical drift between the paper table and the cross-LLM rerun.
CLAUDE_FROM_XLSX = REPO / "scripts" / "_claude_m4_m7_per_seed.json"


def load_cell(config: str, llm: str, seed: int) -> dict | None:
    """Load test_metrics for one (config, llm, seed) cell.

    For llm='claude': read from the static JSON extracted from the paper's xlsx.
    For llm='gpt':    read from the cross-LLM retrain results.
    """
    if llm == "claude":
        if not CLAUDE_FROM_XLSX.exists():
            return None
        d = json.loads(CLAUDE_FROM_XLSX.read_text())
        m = d.get(config, {}).get(str(seed))
        return {"test_metrics": m} if m else None
    p = RES_ROOT / f"{config}_{llm}" / ENCODER / f"seed-{seed}" / "results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def aggregate():
    out = {"protocol": ("Paired GPT-4o-mini vs Claude downstream comparison: "
                        "M4/M7 retrained from scratch with each LLM's profile features "
                        "(bge-large-en-v1.5, identical 10,381-item ID set), 5 seeds "
                        "[42,123,456,789,2026]; per-seed pairing for the t-test."),
           "configs": {}}

    missing = []
    for cfg in CONFIGS:
        cfg_summary = {"per_metric": {}}
        for metric in METRICS:
            claude_vals, gpt_vals = [], []
            paired_seeds = []
            for s in SEEDS:
                c = load_cell(cfg, "claude", s)
                g = load_cell(cfg, "gpt", s)
                if c is None or g is None:
                    missing.append((cfg, s, "claude" if c is None else "gpt"))
                    continue
                cm, gm = c["test_metrics"].get(metric), g["test_metrics"].get(metric)
                if cm is None or gm is None:
                    continue
                claude_vals.append(cm); gpt_vals.append(gm); paired_seeds.append(s)
            if len(claude_vals) < 2:
                cfg_summary["per_metric"][metric] = {"error": f"insufficient seeds ({len(claude_vals)})"}
                continue
            cv = np.array(claude_vals); gv = np.array(gpt_vals)
            diff = gv - cv
            t, pval = st.ttest_rel(gv, cv)
            ci_lo, ci_hi = st.t.interval(0.95, len(diff)-1, loc=diff.mean(), scale=st.sem(diff))
            cfg_summary["per_metric"][metric] = {
                "n_seeds": len(claude_vals),
                "claude_mean": float(cv.mean()), "claude_std": float(cv.std(ddof=1)),
                "gpt_mean":    float(gv.mean()), "gpt_std":    float(gv.std(ddof=1)),
                "diff_mean":   float(diff.mean()),
                "diff_pct_of_claude": float(diff.mean() / cv.mean() * 100),
                "paired_t":    float(t),
                "p_value":     float(pval),
                "ci95_lower":  float(ci_lo),
                "ci95_upper":  float(ci_hi),
                "claude_per_seed": cv.tolist(),
                "gpt_per_seed":    gv.tolist(),
                "paired_seeds": paired_seeds,
            }
        out["configs"][cfg] = cfg_summary

    out["missing_cells"] = missing
    out["completeness"] = f"{20 - len(set((c, s) for c, s, _ in missing))}/20 cells with both LLMs"
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(out, indent=2))
    return out


def write_markdown_table(summary: dict):
    lines = []
    lines.append("# cross-LLM downstream NDCG comparison\n")
    lines.append(f"_{summary['protocol']}_\n")
    lines.append(f"**Completeness:** {summary['completeness']}\n")
    if summary["missing_cells"]:
        lines.append(f"_Missing: {summary['missing_cells']}_\n")

    for cfg in CONFIGS:
        if cfg not in summary["configs"]:
            continue
        lines.append(f"\n## {cfg.upper()} — LightGCN-SF + LLM features\n")
        lines.append("| Metric | Claude (5-seed mean ± std) | GPT-4o-mini | Δ (abs) | Δ (%) | paired-t p | 95% CI |")
        lines.append("|---|---|---|---|---|---|---|")
        for metric in METRICS:
            r = summary["configs"][cfg]["per_metric"].get(metric)
            if r is None or "error" in r:
                continue
            sig = ""
            if r["p_value"] < 0.001: sig = " ***"
            elif r["p_value"] < 0.01: sig = " **"
            elif r["p_value"] < 0.05: sig = " *"
            elif r["p_value"] < 0.1:  sig = " ."
            lines.append(
                f"| {metric} "
                f"| {r['claude_mean']:.4f} ± {r['claude_std']:.4f} "
                f"| {r['gpt_mean']:.4f} ± {r['gpt_std']:.4f} "
                f"| {r['diff_mean']:+.4f} "
                f"| {r['diff_pct_of_claude']:+.2f}% "
                f"| {r['p_value']:.3f}{sig} "
                f"| [{r['ci95_lower']:+.4f}, {r['ci95_upper']:+.4f}] |"
            )

    lines.append("\n_Significance: \\*\\*\\* p<0.001, \\*\\* p<0.01, \\* p<0.05, . p<0.1 (paired t-test, GPT vs Claude)._\n")
    MD_OUT.write_text("\n".join(lines))
    print(f"\n=== Wrote {MD_OUT} ===\n")
    print("\n".join(lines))


if __name__ == "__main__":
    summary = aggregate()
    write_markdown_table(summary)
