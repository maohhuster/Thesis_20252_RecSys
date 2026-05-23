#!/usr/bin/env python3
"""Regeneration harness: R2 (KAR) native 18-point selection grid on ML-20M.

This is the grid that produced the R2 ML-20M *main-results-table* (Table 4)
winner — n_experts=4, lr=1e-3, wd=1e-5 — which is then inherited unchanged to
ML-1M and Amazon-Books in the headline cross-density tables (same protocol as
the M-configs). The original sweep's per-cell validation NDCG@10 was not
persisted (only the winner survives, hardcoded as KAR_N_EXPERTS=4 in
code/benchmark/config.py). This script makes that selection reproducible.

Grid (18 cells, seed=42):
    n_experts    ∈ {2, 4, 8}
    lr           ∈ {5e-4, 1e-3, 5e-3}
    weight_decay ∈ {1e-5, 1e-4}

Each cell runs to completion (200-epoch cap, patience=20) — the uniform
LightGCN-SF protocol used throughout the paper (config.py NUM_EPOCHS=200,
PATIENCE=20). Idempotent: re-running skips cells whose JSON already exists.

Run unattended:
    nohup python3 scripts/run_r2_ml20m_grid.py > /tmp/r2_ml20m_grid.log 2>&1 &
    tail -f /tmp/r2_ml20m_grid.log

After all 18 cells complete, this prints the val-NDCG@10 ranking and the
winner; repopulate code/benchmark/hparams/r2/grid_selection.json from the per-cell
JSONs in code/benchmark/hparams/r2/_per_cell_runs/.
"""
from __future__ import annotations

import itertools, json, subprocess, sys, time
from pathlib import Path

REPO     = Path(__file__).resolve().parent.parent
HARNESS  = REPO / "scripts" / "run_r2_retune.py"
GRID_DIR = REPO / "code" / "benchmark" / "hparams" / "r2" / "_per_cell_runs"
GRID_DIR.mkdir(parents=True, exist_ok=True)

N_EXPERTS    = [2, 4, 8]
LR           = [5e-4, 1e-3, 5e-3]
WEIGHT_DECAY = [1e-5, 1e-4]
GRID = [
    {"n_experts": ne, "lr": lr, "weight_decay": wd}
    for ne, lr, wd in itertools.product(N_EXPERTS, LR, WEIGHT_DECAY)
]

DEVICE  = "auto"
DATASET = "ml20m"
# Uniform protocol with every other ML-20M sweep (R1; R3 ML-20M grid; R2/R3
# retune on ML-1M and Amazon-Books): 200-epoch cap, patience=20, eval-every=5.
NUM_EPOCHS = 200
PATIENCE   = 20

# Native main-table winner (config.py KAR_N_EXPERTS=4).
WINNER = {"n_experts": 4, "lr": 1e-3, "weight_decay": 1e-5}


def fmt(c): return f"ne{c['n_experts']}_lr{c['lr']}_wd{c['weight_decay']}"


def run_cell(cell):
    out_path = GRID_DIR / f"{DATASET}_{fmt(cell)}_seed42.json"
    if out_path.exists():
        d = json.loads(out_path.read_text())
        ndcg = (d.get("test_metrics") or {}).get("NDCG@10")
        print(f"  [skip] {out_path.name} (test NDCG@10={ndcg})")
        return True
    exp = f"r2_ml20m_grid_{fmt(cell)}_seed42"
    cmd = [
        sys.executable, str(HARNESS),
        "--dataset", DATASET, "--seed", "42",
        "--n_experts", str(cell["n_experts"]),
        "--lr", str(cell["lr"]), "--weight_decay", str(cell["weight_decay"]),
        "--num_epochs", str(NUM_EPOCHS),
        "--patience", str(PATIENCE),
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
    val  = (d.get("best_val_metrics") or {}).get("NDCG@10")
    test = (d.get("test_metrics") or {}).get("NDCG@10")
    print(f"     done in {dt/60:.1f} min — val NDCG@10={val}, test NDCG@10={test}")
    return True


if __name__ == "__main__":
    t0 = time.time()
    print("R2 ML-20M native 18-point selection grid (Table-4 winner regeneration)")
    print(f"Grid: {len(GRID)} cells")
    print(f"Output: {GRID_DIR}/")
    print("Estimated wall time: ML-20M is large; budget many hours "
          "(cell-1 timing will calibrate).")

    for i, cell in enumerate(GRID, 1):
        print(f"\n{'='*70}\nCell {i}/{len(GRID)}: {fmt(cell)}\n{'='*70}")
        if not run_cell(cell):
            print(f"\nABORT after cell {i}")
            sys.exit(1)

    candidates = []
    for cell in GRID:
        f = GRID_DIR / f"{DATASET}_{fmt(cell)}_seed42.json"
        d = json.loads(f.read_text())
        val  = (d.get("best_val_metrics") or {}).get("NDCG@10")
        test = (d.get("test_metrics") or {}).get("NDCG@10")
        candidates.append((val, cell["n_experts"], cell["lr"], cell["weight_decay"], test))
    candidates.sort(reverse=True)
    print(f"\n{'='*70}\nML-20M 18-point grid ranking (val NDCG@10):\n{'='*70}")
    for val, ne, lr, wd, test in candidates:
        print(f"  n_experts={ne}  lr={lr}  wd={wd}  →  val={val}  test={test}")
    best = candidates[0]
    print(f"\nWINNER: n_experts={best[1]}, lr={best[2]}, wd={best[3]} "
          f"(val NDCG@10={best[0]}, test NDCG@10={best[4]})")
    print(f"Expected (config.py KAR_N_EXPERTS=4): {WINNER}")
    print(f"\n=== Total wall time: {(time.time()-t0)/60:.1f} min ===")
