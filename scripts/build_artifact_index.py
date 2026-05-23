#!/usr/bin/env python3
"""Single source of truth for CRITICAL (Tier-3+) checkpoints & results.

Storage is intentionally heterogeneous (RLMRec native .pth vs project .pt
copies vs retune-bundle JSON vs per-seed JSON). Hand-maintained maps rot
and mislead. This generator PROBES the real locations, then emits ONE
unified, regenerable view:

  reproducibility/artifacts/<method>/<datapoint>/checkpoints/seed-<s>.<ext>  -> symlink
  reproducibility/artifacts/<method>/<datapoint>/result.json                 -> symlink
  reproducibility/artifact_index.json   (machine manifest: paths, sha256, status, value)
  REPRODUCTION_INDEX.md                 (human table at repo root)

Originals are never moved or modified (RLMRec/upstream + submission
bundles + SHA256SUMS keep working). Re-run any time: idempotent.

    python3 scripts/build_artifact_index.py
"""
from __future__ import annotations
import hashlib, json, glob, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
B = REPO / "code" / "benchmark"
RL = B / "external" / "RLMRec" / "encoder" / "checkpoint"
SEEDS = [42, 123, 456, 789, 2026]
ART = REPO / "reproducibility" / "artifacts"
INDEX_JSON = REPO / "reproducibility" / "artifact_index.json"

# Per (method,datapoint): ordered ckpt-location candidates (first existing wins),
# ordered result candidates, the eval command, and whether JSON-only by design.
SPEC = {
 ("R1-gene","ML-20M"): dict(
   ckpts=[f"{B}/checkpoints/r1/bge-large-en-v1.5/seed-{{s}}/best_model.pt",
          f"{B}/checkpoints/r1/bge-large-en-v1.5/seed-{{s}}/best_model.pt",
          f"{RL}/lightgcn_gene/lightgcn_gene-ml20m-{{s}}.pth"],
   results=[f"{B}/results/r1_metrics.json"],
   eval="python3 scripts/eval_ml20m_r1.py --device mps"),
 ("R1-gene","sub163"): dict(
   ckpts=[f"{B}/checkpoints_ml20m_sub163/r1/bge-large-en-v1.5/seed-{{s}}/best_model.pt",
          f"{RL}/lightgcn_gene/lightgcn_gene-ml20m_sub163_ours-{{s}}.pth"],  # project .pt is authoritative; RLMRec-native .pth = legacy fallback (dir now absent)
   results=[f"{B}/results_ml20m_sub163/r1_ml20m_sub163_metrics.json"],
   eval="python3 scripts/eval_ml20m_sub163_r1.py --device mps"),
 ("R1-gene","ML-1M"): dict(
   ckpts=[f"{B}/checkpoints_ml1m/r1/bge-large-en-v1.5/seed-{{s}}/best_model.pt",
          f"{B}/checkpoints_ml1m/r1/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results_ml1m/r1_ml1m_metrics.json"],
   eval="python3 scripts/eval_ml1m_r1.py"),
 ("R1-gene","Amazon"): dict(
   ckpts=[f"{B}/checkpoints_amazon/r1/bge-large-en-v1.5/seed-{{s}}/best_model.pt",
          f"{B}/checkpoints_amazon/r1/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results_amazon/r1_amazon_metrics.json"],
   eval="python3 scripts/eval_amazon_r1.py"),
 ("R1-plus","ML-20M"): dict(
   ckpts=[f"{B}/checkpoints/r1plus/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results/r1plus_metrics.json"],
   eval="python3 scripts/eval_ml20m_r1plus.py --device mps"),
 ("R1-plus","sub163"): dict(
   ckpts=[f"{B}/checkpoints_ml20m_sub163/r1plus/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results_ml20m_sub163/r1plus_ml20m_sub163_metrics.json"],
   eval="python3 scripts/eval_ml20m_sub163_r1plus.py --device mps"),
 ("R1-plus","ML-1M"): dict(
   ckpts=[f"{B}/checkpoints_ml1m/r1plus/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results_ml1m/r1plus_ml1m_metrics.json"],
   eval="python3 scripts/eval_ml1m_r1plus.py"),
 ("R1-plus","Amazon"): dict(
   ckpts=[f"{B}/checkpoints_amazon/r1plus/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results_amazon/r1plus_amazon_metrics.json"],
   eval="python3 scripts/eval_amazon_r1plus.py"),
 ("R2","ML-20M"): dict(
   ckpts=[f"{B}/checkpoints/r2/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results/r2_metrics.json"],
   eval="python3 scripts/eval_ml20m_r2.py --device mps"),
 ("R2","ML-1M"): dict(
   ckpts=[f"{B}/checkpoints_ml1m/r2/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results_ml1m/r2_ml1m_metrics.json", f"{REPO}/code/benchmark/hparams/r2/grid_selection.json"],
   ndcg_block="ml1m",
   eval="python3 scripts/run_r2_retune_sweep.py  (ml1m block)"),
 ("R2","Amazon"): dict(
   ckpts=[f"{B}/checkpoints_amazon/r2/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results_amazon/r2_amazon_metrics.json", f"{REPO}/code/benchmark/hparams/r2/grid_selection.json"],
   ndcg_block="amazon",
   eval="python3 scripts/run_r2_retune_sweep.py  (amazon block)"),
 ("R3","ML-20M"): dict(
   ckpts=[f"{B}/checkpoints/r3/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{REPO}/code/benchmark/hparams/r3/grid_selection.json"],
   ndcg_block="ml20m",
   eval="python3 scripts/eval_r3_ml20m_checkpoints.py --device mps"),
 ("R3","ML-1M"): dict(
   ckpts=[f"{B}/checkpoints_ml1m/r3/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results_ml1m/r3_ml1m_metrics.json", f"{REPO}/code/benchmark/hparams/r3/grid_selection.json"],
   ndcg_block="ml1m",
   eval="python3 scripts/run_r3_retune_sweep.py  (ml1m block)"),
 ("R3","Amazon"): dict(
   ckpts=[f"{B}/checkpoints_amazon/r3/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results_amazon/r3_amazon_metrics.json", f"{REPO}/code/benchmark/hparams/r3/grid_selection.json"],
   ndcg_block="amazon",
   eval="python3 scripts/run_r3_retune_sweep.py  (amazon block)"),
 ("SASRec","ML-20M"): dict(
   ckpts=[f"{B}/checkpoints/sasrec_pmixer/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results/sasrec_pmixer/bge-large-en-v1.5/ml20m_test_5seeds.json"],
   eval="python3 scripts/run_ml20m_sasrec_pmixer.py"),
 ("SASRec","sub163"): dict(
   ckpts=[f"{B}/checkpoints_ml20m_sub163/sasrec_pmixer/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results_ml20m_sub163/sasrec_pmixer/bge-large-en-v1.5/seed-{{s}}/results.json"],
   eval="python3 scripts/run_ml20m_sub163_sasrec_pmixer.py", per_seed_result=True),
 ("SASRec","ML-1M"): dict(
   ckpts=[f"{B}/checkpoints_ml1m/sasrec_pmixer/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results_ml1m/sasrec_pmixer/bge-large-en-v1.5/seed-{{s}}/results.json"],
   eval="python3 scripts/run_ml1m_sasrec_pmixer.py", per_seed_result=True),
 ("SASRec","Amazon"): dict(
   ckpts=[f"{B}/checkpoints_amazon/sasrec_pmixer/bge-large-en-v1.5/seed-{{s}}/best_model.pt"],
   results=[f"{B}/results_amazon/sasrec_pmixer/bge-large-en-v1.5/seed-{{s}}/results.json"],
   eval="python3 scripts/run_amazon_sasrec_pmixer.py", per_seed_result=True),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def first_ckpt_pattern(cands):
    for pat in cands:
        if any(os.path.exists(pat.format(s=s)) for s in SEEDS):
            return pat
    return None


def first_result(cands, per_seed):
    for c in cands:
        if per_seed:
            if any(os.path.exists(c.format(s=s)) for s in SEEDS):
                return c, True
        elif os.path.exists(c):
            return c, False
    return None, False


def ndcg_mean(res_path, block=None):
    try:
        d = json.load(open(res_path))
        # main-table aggregate (results/<method>_metrics.json)
        for k in ("summary", "summary_test"):
            if k in d and "NDCG@10" in d[k]:
                v = d[k]["NDCG@10"]
                return round(v["mean"] if isinstance(v, dict) else v, 4)
        # unified hparams manifest (hparams/<method>/grid_selection.json)
        if "datasets" in d:
            for blk in ((block,) if block else ("ml20m", "ml1m", "amazon")):
                ds = d["datasets"].get(blk, {})
                for key in ("r3_test_metrics", "retuned_test_metrics"):
                    if key in ds and "NDCG@10" in ds[key]:
                        return round(ds[key]["NDCG@10"]["mean"], 4)
        # legacy retune summary schema (kept for back-compat with old runs)
        for blk in ((block,) if block else ("ml20m", "ml1m", "amazon")):
            if blk in d and "r3_test_metrics" in d.get(blk, {}):
                return round(d[blk]["r3_test_metrics"]["NDCG@10"]["mean"], 4)
            if blk in d and "retuned_test_metrics" in d.get(blk, {}):
                return round(d[blk]["retuned_test_metrics"]["NDCG@10"]["mean"], 4)
    except Exception:
        pass
    return None


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except Exception:
        return str(p)


def main():
    if ART.exists():
        import shutil
        shutil.rmtree(ART)
    index = {"_doc": "Single source of truth for critical Tier-3+ checkpoints/results. "
                     "Regenerate: python3 scripts/build_artifact_index.py. Originals never moved.",
             "seeds": SEEDS, "entries": []}
    md = ["# Reproduction Index — critical checkpoints & results",
          "",
          "> Auto-generated by `scripts/build_artifact_index.py` (re-run any time; idempotent).",
          "> `reproducibility/artifacts/<method>/<datapoint>/` is a unified **symlink** view — real files stay",
          "> in their tool-native locations (RLMRec native `.pth`, project `.pt`, retune bundles).",
          "",
          "## Conventions",
          "- ML-20M = base dataset → unsuffixed `code/benchmark/{checkpoints,results}/`; "
          "others suffixed `_ml20m_sub163` / `_ml1m` / `_amazon`.",
          "- All methods (M0–M9 / R1-gene / R1-plus / R2 / R3 / SASRec) use project "
          "`.pt` checkpoints at `code/benchmark/checkpoints*/<m>/bge-large-en-v1.5/"
          "seed-*/best_model.pt` (+ `training_state.pt`). The legacy RLMRec-native "
          "`encoder/checkpoint/*.pth` directory is no longer used.",
          "- R3 has full per-seed checkpoints on ML-20M / ML-1M / Amazon (5 distinct "
          "seeds each, integrity-checked). Per-cell retune robustness numbers live in "
          "`code/benchmark/hparams/r3/`.",
          "- Retune/grid bundles live under `code/benchmark/`: `code/benchmark/hparams/r2/`, `code/benchmark/hparams/r3/`.",
          "",
          "| Method | Datapoint | Checkpoints (real location) | Seeds | Result JSON | NDCG@10 | Verify |",
          "|---|---|---|---|---|---|---|"]

    for (method, dp), spec in SPEC.items():
        slug = f"{method}/{dp}".replace("/", "_").replace(" ", "")
        d = ART / method / dp.replace("/", "-")
        (d / "checkpoints").mkdir(parents=True, exist_ok=True)
        ck_pat = None if spec.get("json_only") else first_ckpt_pattern(spec["ckpts"])
        seed_files, shas = [], []
        if ck_pat:
            for s in SEEDS:
                src = Path(ck_pat.format(s=s))
                if src.exists():
                    ext = src.suffix
                    ln = d / "checkpoints" / f"seed-{s}{ext}"
                    if ln.exists() or ln.is_symlink():
                        ln.unlink()
                    ln.symlink_to(os.path.relpath(src, ln.parent))
                    seed_files.append(rel(src))
                    shas.append(sha256(src))
        n = len(seed_files)
        backfill = (len(set(shas)) != len(shas)) if shas else False
        res_pat, per_seed = first_result(spec["results"], spec.get("per_seed_result", False))
        res_show = "—"
        ndcg = None
        if res_pat:
            if per_seed:
                # link the seed-42 result as representative; index records the dir
                rp = Path(res_pat.format(s=42))
                rl_ = d / "result-seed-42.json"
                if rl_.exists() or rl_.is_symlink():
                    rl_.unlink()
                rl_.symlink_to(os.path.relpath(rp, d))
                res_show = rel(res_pat.format(s="<seed>"))
            else:
                rp = Path(res_pat)
                rl_ = d / "result.json"
                if rl_.exists() or rl_.is_symlink():
                    rl_.unlink()
                rl_.symlink_to(os.path.relpath(rp, d))
                res_show = rel(res_pat)
                ndcg = ndcg_mean(rp, spec.get("ndcg_block"))
        ck_show = "JSON-only (by design)" if spec.get("json_only") else (
            rel(ck_pat.format(s="<seed>")) if ck_pat else "**MISSING**")
        status = ("json-only" if spec.get("json_only")
                  else "BACKFILL?" if backfill
                  else f"{n}/5 ckpt" if n else "no-ckpt")
        index["entries"].append(dict(
            method=method, datapoint=dp, checkpoint_pattern=ck_show,
            n_seed_ckpts=n, ckpt_sha256=shas, duplicate_backfill=backfill,
            result=res_show, ndcg10=ndcg, eval_cmd=spec["eval"],
            json_only=bool(spec.get("json_only")),
            artifacts_view=str(d.relative_to(REPO))))
        md.append(f"| {method} | {dp} | `{ck_show}` | {n if not spec.get('json_only') else '—'}"
                  f" | `{res_show}` | {f'{ndcg:.4f}' if ndcg is not None else '—'} | {status} |")

    ART.mkdir(parents=True, exist_ok=True)
    INDEX_JSON.write_text(json.dumps(index, indent=2))
    (REPO / "REPRODUCTION_INDEX.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {INDEX_JSON.relative_to(REPO)} ({len(index['entries'])} entries) + REPRODUCTION_INDEX.md")
    miss = [e for e in index["entries"] if e["checkpoint_pattern"] == "**MISSING**"]
    bf = [e for e in index["entries"] if e["duplicate_backfill"]]
    print(f"MISSING ckpt: {[(e['method'],e['datapoint']) for e in miss] or 'none'}")
    print(f"BACKFILL flag: {[(e['method'],e['datapoint']) for e in bf] or 'none'}")


if __name__ == "__main__":
    main()
