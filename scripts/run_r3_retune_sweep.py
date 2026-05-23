#!/usr/bin/env python3
"""replacer triangulation: orchestrate the R3 (HypernetReplacer) retune sweep.

Mirrors run_r2_retune_sweep.py exactly so R2-vs-R3 is a controlled comparison:
    - Same 4-cell pre-registered hparam grid
    - Same 5 seeds [42, 123, 456, 789, 2026]
    - Same datasets (ML-20M, ML-1M, Amazon-Books-2018)
    - Same train_model entrypoint, same patience, same num_epochs cap

Pre-registered before seeing R3 results.

Run unattended:
    nohup python3 scripts/run_r3_retune_sweep.py > /tmp/r3_sweep.log 2>&1 &
    tail -f /tmp/r3_sweep.log
"""
from __future__ import annotations

import json, subprocess, sys, time
from pathlib import Path

REPO    = Path(__file__).resolve().parent.parent
HARNESS = REPO / "scripts" / "run_r3_retune.py"
GRID_DIR  = REPO / "code" / "benchmark" / "hparams" / "r3" / "_per_cell_runs"
RERUN_DIR = REPO / "code" / "benchmark" / "hparams" / "r3" / "winner_5seed"
SUMMARY   = REPO / "code" / "benchmark" / "hparams" / "r3" / "r3_retune_summary.json"

GRID_DIR.mkdir(parents=True, exist_ok=True)
RERUN_DIR.mkdir(parents=True, exist_ok=True)

# Pre-registered grid — identical to R2 retune for controlled comparison
GRID = [
    {"lr": 3e-4, "weight_decay": 1e-5},
    {"lr": 3e-4, "weight_decay": 1e-4},
    {"lr": 1e-3, "weight_decay": 1e-5},
    {"lr": 1e-3, "weight_decay": 1e-4},
]
DATASETS = ["ml20m", "ml1m", "amazon"]
SEEDS    = [42, 123, 456, 789, 2026]
DEVICE   = "auto"                # picks cuda → mps → cpu

# R2 retuned numbers for direct R3-vs-R2 delta
R2_RETUNED = {
    "ml1m":   {"NDCG@10": (0.1680, 0.0053), "Recall@10": (0.0447, 0.0050), "MRR": (0.3147, 0.0047)},
    "amazon": {"NDCG@10": (0.0439, 0.0013), "Recall@10": (0.0585, 0.0019), "MRR": (0.0796, 0.0017)},
}
# M7 (injection) for "did R3 also collapse?" comparison
M7_NUMBERS = {
    "ml1m":   {"NDCG@10": 0.1661},
    "amazon": {"NDCG@10": 0.0563},
}


def fmt(cell): return f"lr{cell['lr']}_wd{cell['weight_decay']}"


def run_cell(dataset, lr, weight_decay, seed, exp_name, out_path):
    if out_path.exists():
        return True
    cmd = [
        sys.executable, str(HARNESS),
        "--dataset", dataset, "--seed", str(seed),
        "--lr", str(lr), "--weight_decay", str(weight_decay),
        "--device", DEVICE, "--exp_name", exp_name, "--out", str(out_path),
    ]
    print(f"  → {' '.join(cmd[2:])}")
    t0 = time.time()
    r = subprocess.run(cmd)
    dt = time.time() - t0
    if r.returncode != 0 or not out_path.exists():
        print(f"     FAILED after {dt/60:.1f} min")
        return False
    d = json.loads(out_path.read_text())
    val = (d.get("best_val_metrics") or {}).get("NDCG@10")
    test = (d.get("test_metrics") or {}).get("NDCG@10")
    print(f"     done in {dt/60:.1f} min — val NDCG@10={val:.4f}, test NDCG@10={test:.4f}")
    return True


def stage_grid_selection():
    print("\n" + "=" * 70)
    print("STAGE 1 — R3 grid selection (seed=42 only)")
    print("=" * 70)
    winners = {}
    for ds in DATASETS:
        print(f"\n[{ds}] grid sweep ({len(GRID)} cells)")
        for cell in GRID:
            exp = f"r3_retune_{ds}_{fmt(cell)}_seed42"
            out = GRID_DIR / f"{ds}_{fmt(cell)}_seed42.json"
            run_cell(ds, cell["lr"], cell["weight_decay"], 42, exp, out)
        candidates = []
        for cell in GRID:
            f = GRID_DIR / f"{ds}_{fmt(cell)}_seed42.json"
            if f.exists():
                d = json.loads(f.read_text())
                candidates.append(((d.get("best_val_metrics") or {}).get("NDCG@10"),
                                   cell["lr"], cell["weight_decay"]))
        if not candidates:
            print(f"[{ds}] ERROR: no grid results")
            return {}
        candidates.sort(reverse=True)
        print(f"\n[{ds}] grid ranking (val NDCG@10):")
        for ndcg, lr, wd in candidates:
            print(f"     lr={lr}  wd={wd}  →  {ndcg:.4f}")
        best_ndcg, best_lr, best_wd = candidates[0]
        winners[ds] = {"lr": best_lr, "weight_decay": best_wd, "val_ndcg": best_ndcg}
        print(f"[{ds}] WINNER: lr={best_lr}, wd={best_wd}")
    return winners


def stage_winner_rerun(winners):
    print("\n" + "=" * 70)
    print("STAGE 2 — 5-seed rerun of R3 winners")
    print("=" * 70)
    for ds, hp in winners.items():
        print(f"\n[{ds}] rerun: lr={hp['lr']}, wd={hp['weight_decay']}, 5 seeds")
        for seed in SEEDS:
            out = RERUN_DIR / f"{ds}_winner_seed{seed}.json"
            if out.exists():
                print(f"  [seed={seed}] skip (exists)")
                continue
            if seed == 42:
                src = GRID_DIR / f"{ds}_{fmt(hp)}_seed42.json"
                if src.exists():
                    out.write_text(src.read_text())
                    d = json.loads(src.read_text())
                    test = (d.get("test_metrics") or {}).get("NDCG@10")
                    print(f"  [seed=42] reused (test NDCG@10={test:.4f})")
                    continue
            exp = f"r3_retune_{ds}_{fmt(hp)}_seed{seed}"
            run_cell(ds, hp["lr"], hp["weight_decay"], seed, exp, out)


def stage_aggregate():
    print("\n" + "=" * 70)
    print("STAGE 3 — aggregate + triangulation decision")
    print("=" * 70)
    import numpy as np
    summary = {
        "protocol": ("R3 (HypernetReplacer) retune for replacer triangulation. "
                     "Pre-registered 4-cell grid identical to R2 retune; 5 seeds at winner. "
                     "Goal: test whether R2's Amazon collapse is paradigm-class behaviour "
                     "(R3 also collapses) or instantiation-specific (R3 ties M7)."),
        "device": DEVICE,
        "datasets": DATASETS,
        "grid": GRID,
        "seeds": SEEDS,
    }
    for ds in DATASETS:
        files = sorted(RERUN_DIR.glob(f"{ds}_winner_seed*.json"))
        rows = [json.loads(f.read_text()) for f in files]
        if not rows:
            continue
        test_metrics = {}
        for k in ("NDCG@10", "Recall@10", "MRR"):
            vals = [(r.get("test_metrics") or {}).get(k) for r in rows]
            vals = [v for v in vals if v is not None]
            if vals:
                test_metrics[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "values": [float(v) for v in vals],
                }

        r3_ndcg = test_metrics.get("NDCG@10", {}).get("mean")
        r2_ndcg = R2_RETUNED[ds]["NDCG@10"][0]
        m7_ndcg = M7_NUMBERS[ds]["NDCG@10"]

        summary[ds] = {
            "winner_lr": rows[0]["lr"],
            "winner_weight_decay": rows[0]["weight_decay"],
            "n_seeds": len(rows),
            "r3_test_metrics": test_metrics,
            "r2_retuned_test_metrics": {k: {"mean": R2_RETUNED[ds][k][0], "std": R2_RETUNED[ds][k][1]}
                                         for k in R2_RETUNED[ds]},
            "m7_NDCG10": m7_ndcg,
            "deltas_NDCG10": {
                "r3_vs_r2_pct": (r3_ndcg - r2_ndcg) / r2_ndcg * 100 if r3_ndcg else None,
                "r3_vs_m7_pct": (r3_ndcg - m7_ndcg) / m7_ndcg * 100 if r3_ndcg else None,
                "r2_vs_m7_pct": (r2_ndcg - m7_ndcg) / m7_ndcg * 100,
            },
            "wall_time_s_total": sum(r.get("wall_time_s", 0) for r in rows),
        }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def stage_decision(summary):
    print("\n" + "=" * 70)
    print("TRIANGULATION DECISION — does R3 also collapse on sparse-per-user data?")
    print("=" * 70)
    for ds in DATASETS:
        if ds not in summary:
            continue
        d = summary[ds]
        r3_vs_m7 = d["deltas_NDCG10"]["r3_vs_m7_pct"]
        r2_vs_m7 = d["deltas_NDCG10"]["r2_vs_m7_pct"]
        r3_vs_r2 = d["deltas_NDCG10"]["r3_vs_r2_pct"]
        print(f"\n[{ds}] R3 vs M7 = {r3_vs_m7:+.1f}%  |  R2 vs M7 = {r2_vs_m7:+.1f}%  "
              f"|  R3 vs R2 = {r3_vs_r2:+.1f}%")

    # Overall decision based on Amazon outcome
    if "amazon" in summary:
        r3_amz = summary["amazon"]["deltas_NDCG10"]["r3_vs_m7_pct"]
        if r3_amz <= -15.0:
            print("\n  → OUTCOME 1: Both R2 and R3 collapse on Amazon. "
                  "Replacer-class claim STRONGLY corroborated. Two structurally-different "
                  "replacers fail at sparse-per-user density. Paper substantially strengthened.")
        elif r3_amz <= -5.0:
            print("\n  → OUTCOME 2: R3 partially collapses but less than R2. "
                  "Replacer-class claim partially corroborated; magnitude of collapse "
                  "is instantiation-dependent. Honest reporting required.")
        elif r3_amz < 5.0:
            print("\n  → OUTCOME 3: R3 ties M7 on Amazon while R2 collapses. "
                  "Replacer-class claim FALSIFIED — R2's collapse was MoE-instantiation-specific. "
                  "Paper rewrites: 'replacer' ≠ failure-class; specifically MoE-on-LightGCN-BPR fails.")
        else:
            print("\n  → OUTCOME 4: R3 BEATS M7 on Amazon. Strong falsification. "
                  "The pure-content replacer is actually the right choice for sparse data. "
                  "Major paper restructure.")


if __name__ == "__main__":
    t0 = time.time()
    print(f"R3 (HypernetReplacer) retune sweep — replacer triangulation")
    print(f"REPO: {REPO}")
    print(f"GRID: {GRID}")
    print(f"DATASETS: {DATASETS}")
    print(f"SEEDS: {SEEDS}")

    winners = stage_grid_selection()
    if not winners:
        sys.exit(1)
    stage_winner_rerun(winners)
    summary = stage_aggregate()
    stage_decision(summary)

    print(f"\n=== Total wall time: {(time.time()-t0)/60:.1f} min ===")
