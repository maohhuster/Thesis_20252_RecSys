#!/usr/bin/env python3
"""Orchestrate the Option-A R2 retune sweep on Mac MPS.

Sweep:
    4 hand-picked grid cells × seed=42 → winner per dataset
    Winner × 5 seeds (seed=42 reused from grid)
    Datasets: ml1m, amazon

Outputs per cell:
    code/benchmark/hparams/r2/_per_cell_runs/<ds>_lr<lr>_wd<wd>_seed42.json
    code/benchmark/hparams/r2/winner_5seed/<ds>_winner_seed<n>.json
Aggregate:
    code/benchmark/hparams/r2/r2_retune_summary.json

Idempotent: skips cells whose JSON already exists. Can be killed and resumed.

Run unattended:
    nohup python3 scripts/run_r2_retune_sweep.py > /tmp/r2_sweep.log 2>&1 &
    tail -f /tmp/r2_sweep.log
"""
from __future__ import annotations

import json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "scripts" / "run_r2_retune.py"
GRID_DIR = REPO / "code" / "benchmark" / "hparams" / "r2" / "_per_cell_runs"
RERUN_DIR = REPO / "code" / "benchmark" / "hparams" / "r2" / "winner_5seed"
SUMMARY = REPO / "code" / "benchmark" / "hparams" / "r2" / "r2_retune_summary.json"

GRID_DIR.mkdir(parents=True, exist_ok=True)
RERUN_DIR.mkdir(parents=True, exist_ok=True)

# Pre-registered hand-picked grid (small + sensible neighbourhood of ML-20M winner)
GRID = [
    {"lr": 3e-4, "weight_decay": 1e-5},
    {"lr": 3e-4, "weight_decay": 1e-4},
    {"lr": 1e-3, "weight_decay": 1e-5},  # ← ML-20M winner — included as control
    {"lr": 1e-3, "weight_decay": 1e-4},
]
DATASETS = ["ml1m", "amazon"]
SEEDS = [42, 123, 456, 789, 2026]
DEVICE = "mps"

# Original paper R2 numbers (ML-20M-default hparams), for delta comparison
ORIGINAL = {
    "ml1m":   {"NDCG@10": (0.1677, 0.0046), "Recall@10": (0.0438, 0.0030), "MRR": (0.3140, 0.0033)},
    "amazon": {"NDCG@10": (0.0401, 0.0045), "Recall@10": (0.0538, 0.0059), "MRR": (0.0731, 0.0075)},
}


def fmt_lr_wd(cell: dict) -> str:
    return f"lr{cell['lr']}_wd{cell['weight_decay']}"


def run_cell(dataset: str, lr: float, weight_decay: float, seed: int,
             exp_name: str, out_path: Path) -> bool:
    """Run one cell. Returns True if cell completed (or was already done)."""
    if out_path.exists():
        return True
    cmd = [
        sys.executable, str(HARNESS),
        "--dataset", dataset,
        "--seed", str(seed),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--device", DEVICE,
        "--exp_name", exp_name,
        "--out", str(out_path),
    ]
    print(f"  → {' '.join(cmd[2:])}")
    t0 = time.time()
    r = subprocess.run(cmd)
    dt = time.time() - t0
    ok = r.returncode == 0 and out_path.exists()
    if ok:
        d = json.loads(out_path.read_text())
        ndcg = (d.get("test_metrics") or {}).get("NDCG@10")
        val_ndcg = (d.get("best_val_metrics") or {}).get("NDCG@10")
        print(f"     done in {dt/60:.1f} min — val NDCG@10={val_ndcg:.4f}, "
              f"test NDCG@10={ndcg:.4f}")
    else:
        print(f"     FAILED after {dt/60:.1f} min (rc={r.returncode})")
    return ok


def stage_grid_selection() -> dict:
    """Run 4 grid cells × seed=42 per dataset. Returns winners."""
    print("\n" + "=" * 70)
    print("STAGE 1 — grid selection (seed=42 only)")
    print("=" * 70)
    winners = {}
    for ds in DATASETS:
        print(f"\n[{ds}] grid sweep ({len(GRID)} cells)")
        for cell in GRID:
            exp = f"r2_retune_{ds}_{fmt_lr_wd(cell)}_seed42"
            out_path = GRID_DIR / f"{ds}_{fmt_lr_wd(cell)}_seed42.json"
            run_cell(ds, cell["lr"], cell["weight_decay"], 42, exp, out_path)

        # Pick winner by val NDCG@10
        candidates = []
        for cell in GRID:
            f = GRID_DIR / f"{ds}_{fmt_lr_wd(cell)}_seed42.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            val_ndcg = (d.get("best_val_metrics") or {}).get("NDCG@10")
            candidates.append((val_ndcg, cell["lr"], cell["weight_decay"]))
        if not candidates:
            print(f"[{ds}] ERROR: no grid results — abort")
            return {}
        candidates.sort(reverse=True)
        print(f"\n[{ds}] grid ranking (val NDCG@10):")
        for ndcg, lr, wd in candidates:
            print(f"     lr={lr}  wd={wd}  →  {ndcg:.4f}")
        best_ndcg, best_lr, best_wd = candidates[0]
        winners[ds] = {"lr": best_lr, "weight_decay": best_wd, "val_ndcg": best_ndcg}
        print(f"[{ds}] WINNER: lr={best_lr}, wd={best_wd}")
    return winners


def stage_winner_rerun(winners: dict) -> None:
    """Rerun each winner across 5 seeds (seed=42 reused from grid)."""
    print("\n" + "=" * 70)
    print("STAGE 2 — 5-seed rerun of winners")
    print("=" * 70)
    for ds, hp in winners.items():
        print(f"\n[{ds}] rerun: lr={hp['lr']}, wd={hp['weight_decay']}, 5 seeds")
        for seed in SEEDS:
            out_path = RERUN_DIR / f"{ds}_winner_seed{seed}.json"
            if out_path.exists():
                print(f"  [seed={seed}] skip (exists)")
                continue
            if seed == 42:
                # Reuse grid-selection result
                src = GRID_DIR / f"{ds}_{fmt_lr_wd(hp)}_seed42.json"
                if src.exists():
                    out_path.write_text(src.read_text())
                    d = json.loads(src.read_text())
                    ndcg = (d.get("test_metrics") or {}).get("NDCG@10")
                    print(f"  [seed=42] reused from grid (test NDCG@10={ndcg:.4f})")
                    continue
            exp = f"r2_retune_{ds}_{fmt_lr_wd(hp)}_seed{seed}"
            run_cell(ds, hp["lr"], hp["weight_decay"], seed, exp, out_path)


def stage_aggregate() -> dict:
    """Compute per-dataset 5-seed mean±std of test metrics, write summary JSON."""
    print("\n" + "=" * 70)
    print("STAGE 3 — aggregate")
    print("=" * 70)
    import numpy as np
    summary = {
        "protocol": ("R2 retuned with 4-cell hand-picked grid (lr ∈ {3e-4, 1e-3}, "
                     "weight_decay ∈ {1e-5, 1e-4}, n_experts=4 fixed); "
                     "winner reran across 5 seeds [42, 123, 456, 789, 2026]; "
                     "pre-registered before seeing results."),
        "device": DEVICE,
    }
    for ds in DATASETS:
        files = sorted(RERUN_DIR.glob(f"{ds}_winner_seed*.json"))
        rows = [json.loads(f.read_text()) for f in files]
        if not rows:
            print(f"[{ds}] no winner files — skip")
            continue

        test_metrics = {}
        for k in ("NDCG@10", "Recall@10", "MRR"):
            vals = [(r.get("test_metrics") or {}).get(k) for r in rows]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            test_metrics[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": [float(v) for v in vals],
            }

        # Compare to original paper R2 numbers
        original = ORIGINAL[ds]
        deltas = {}
        for k, (orig_mean, orig_std) in original.items():
            new_mean = test_metrics.get(k, {}).get("mean")
            if new_mean is None:
                continue
            deltas[k] = {
                "abs": new_mean - orig_mean,
                "pct": (new_mean - orig_mean) / orig_mean * 100,
            }

        summary[ds] = {
            "winner_lr": rows[0]["lr"],
            "winner_weight_decay": rows[0]["weight_decay"],
            "n_seeds": len(rows),
            "retuned_test_metrics": test_metrics,
            "original_paper_test_metrics": {
                k: {"mean": v[0], "std": v[1]} for k, v in original.items()
            },
            "delta_vs_original": deltas,
            "wall_time_s_total": sum(r.get("wall_time_s", 0) for r in rows),
        }

    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {SUMMARY}")
    print(json.dumps(summary, indent=2))
    return summary


def stage_decision(summary: dict) -> None:
    """Print which outcome bucket the result falls into."""
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)
    for ds in DATASETS:
        if ds not in summary:
            continue
        d = summary[ds]
        delta_pct = d.get("delta_vs_original", {}).get("NDCG@10", {}).get("pct")
        new_std = d["retuned_test_metrics"].get("NDCG@10", {}).get("std")
        old_std = ORIGINAL[ds]["NDCG@10"][1]
        print(f"\n[{ds}] retuned vs original: ΔNDCG@10 = {delta_pct:+.1f}%")
        print(f"  std: {old_std:.4f} → {new_std:.4f}")
        if abs(delta_pct) < 2.0 and new_std <= old_std * 1.2:
            print("  → Outcome 1: framework strengthened. R2 anomaly was paradigm-class behaviour.")
        elif ds == "ml1m" and abs(delta_pct) < 5.0 and new_std < old_std * 0.5:
            print("  → Outcome 2: ML-1M std blowup was hparam noise; rewrite App. E.13.")
        elif delta_pct > 10.0:
            print("  → Outcome 3-4: framework's class-level claim weakened/falsified for replacer.")
        else:
            print("  → Mixed: report both numbers, conservative interpretation.")


if __name__ == "__main__":
    t0 = time.time()
    print(f"R2 retune sweep — Path A (Mac MPS)")
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
