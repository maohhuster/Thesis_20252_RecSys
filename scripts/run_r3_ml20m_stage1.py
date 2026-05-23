#!/usr/bin/env python3
"""Stage-1-only orchestrator: R3 grid selection on ML-20M (4 cells × seed=42).

Path Z (Best-Paper triangulation): adds ML-20M as the 3rd density datapoint
for R3, completing the cross-replacer × cross-density matrix.

Each cell runs to completion (200 epochs cap with patience=20). Idempotent:
re-running skips cells whose JSON already exists.

Run unattended:
    nohup python3 scripts/run_r3_ml20m_stage1.py > /tmp/r3_ml20m_stage1.log 2>&1 &
    tail -f /tmp/r3_ml20m_stage1.log
"""
from __future__ import annotations

import json, subprocess, sys, time
from pathlib import Path

REPO    = Path(__file__).resolve().parent.parent
HARNESS = REPO / "scripts" / "run_r3_retune.py"
GRID_DIR = REPO / "code" / "benchmark" / "hparams" / "r3" / "_per_cell_runs"
GRID_DIR.mkdir(parents=True, exist_ok=True)

# Same pre-registered grid as R3 ML-1M / Amazon (and R2 retune)
GRID = [
    {"lr": 3e-4, "weight_decay": 1e-5},
    {"lr": 3e-4, "weight_decay": 1e-4},
    {"lr": 1e-3, "weight_decay": 1e-5},
    {"lr": 1e-3, "weight_decay": 1e-4},
]
DEVICE = "auto"
DATASET = "ml20m"
# Uniform protocol with all other ML-20M sweeps (R1, R2 ML-20M grid; R2/R3
# retune on ML-1M and Amazon-Books): 200-epoch cap, patience=20, eval-every=5.
# This matches the LightGCN-SF training settings used throughout the paper
# (config.py NUM_EPOCHS=200, PATIENCE=20) and §4.3 of the paper. Early stopping
# typically fires well before the cap; the 200-epoch budget ensures slow-convergence
# cells are not artificially clipped.
NUM_EPOCHS_STAGE1 = 200
PATIENCE_STAGE1 = 20


def fmt(c): return f"lr{c['lr']}_wd{c['weight_decay']}"


def run_cell(cell):
    out_path = GRID_DIR / f"{DATASET}_{fmt(cell)}_seed42.json"
    if out_path.exists():
        d = json.loads(out_path.read_text())
        ndcg = (d.get("test_metrics") or {}).get("NDCG@10")
        print(f"  [skip] {out_path.name} (test NDCG@10={ndcg:.4f})")
        return True
    exp = f"r3_retune_{DATASET}_{fmt(cell)}_seed42"
    cmd = [
        sys.executable, str(HARNESS),
        "--dataset", DATASET, "--seed", "42",
        "--lr", str(cell["lr"]), "--weight_decay", str(cell["weight_decay"]),
        "--num_epochs", str(NUM_EPOCHS_STAGE1),
        "--patience", str(PATIENCE_STAGE1),
        "--device", DEVICE, "--exp_name", exp, "--out", str(out_path),
    ]
    print(f"\n  → {' '.join(cmd[2:])}")
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


if __name__ == "__main__":
    t0 = time.time()
    print(f"R3 ML-20M Stage-1 grid selection (Path Z)")
    print(f"Grid: {GRID}")
    print(f"Output: {GRID_DIR}/")
    print(f"Estimated wall time on Mac MPS: 2–6 hours total (cell-1 timing will calibrate)")

    for i, cell in enumerate(GRID, 1):
        print(f"\n{'='*70}\nCell {i}/{len(GRID)}: {fmt(cell)}\n{'='*70}")
        ok = run_cell(cell)
        if not ok:
            print(f"\nABORT after cell {i}")
            sys.exit(1)

    # Pick winner
    candidates = []
    for cell in GRID:
        f = GRID_DIR / f"{DATASET}_{fmt(cell)}_seed42.json"
        d = json.loads(f.read_text())
        val = (d.get("best_val_metrics") or {}).get("NDCG@10")
        test = (d.get("test_metrics") or {}).get("NDCG@10")
        candidates.append((val, cell["lr"], cell["weight_decay"], test))
    candidates.sort(reverse=True)
    print(f"\n{'='*70}\nML-20M grid ranking (val NDCG@10):\n{'='*70}")
    for val, lr, wd, test in candidates:
        print(f"  lr={lr}  wd={wd}  →  val={val:.4f}  test={test:.4f}")
    best = candidates[0]
    print(f"\nWINNER: lr={best[1]}, wd={best[2]} (val NDCG@10={best[0]:.4f}, test NDCG@10={best[3]:.4f})")
    print(f"\nFor R2 ML-20M comparison: 0.1145 ± 0.0014")
    print(f"For M7 ML-20M comparison: 0.1175")
    print(f"\nNext step: 4-seed winner rerun (123, 456, 789, 2026) — see Path Z plan")
    print(f"\n=== Total Stage-1 wall time: {(time.time()-t0)/60:.1f} min ===")
