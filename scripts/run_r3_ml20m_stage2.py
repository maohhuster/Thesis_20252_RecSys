#!/usr/bin/env python3
"""Stage-2 orchestrator: R3 ML-20M 5-seed winner rerun.

Stage 1 (run_r3_ml20m_stage1.py) ran the 4-cell hparam grid on seed=42 at
the full 200-epoch / patience-20 budget to identify the winner. Stage 2
reruns the winner across 5 seeds [42, 123, 456, 789, 2026] at the same
200/20 budget — protocol-matched to R3 ML-1M / Amazon Stage 2 and to the
R2 retune sweep, satisfying §4.3 of the paper.

Note: seed=42 may be reused from Stage 1 since the budgets are identical
(both stages run at 200/20); for safety the orchestrator does not reuse it
by default, which matches the convention used for R2 / R3 retune sweeps.

Pre-registered winner from Stage 1: lr=3e-4, wd=1e-5, n_experts=4,
hypernet hidden=256.

Run unattended on server:
    nohup python3 scripts/run_r3_ml20m_stage2.py > /tmp/r3_ml20m_stage2.log 2>&1 &
    tail -f /tmp/r3_ml20m_stage2.log
"""
from __future__ import annotations

import json, subprocess, sys, time
from pathlib import Path

REPO     = Path(__file__).resolve().parent.parent
HARNESS  = REPO / "scripts" / "run_r3_retune.py"
RERUN_DIR = REPO / "code" / "benchmark" / "hparams" / "r3" / "winner_5seed"
SUMMARY  = REPO / "code" / "benchmark" / "hparams" / "r3" / "ml20m_summary.json"

RERUN_DIR.mkdir(parents=True, exist_ok=True)

# Stage-1 winner — fill these in from Stage 1 if different.
WINNER_LR = 3e-4
WINNER_WD = 1e-5

DATASET = "ml20m"
DEVICE  = "auto"
SEEDS   = [42, 123, 456, 789, 2026]

# Full publication-quality budget (matches R3 ML-1M / Amazon Stage 2 and R2 retune)
NUM_EPOCHS = 200
PATIENCE   = 20

# Reference numbers for cross-paradigm context (NDCG@10 on ML-20M test)
REFERENCE = {
    "R2_NDCG10":   {"mean": 0.1145, "std": 0.0014, "source": "main paper Table 4 (ML-20M-tuned)"},
    "M7_NDCG10":   {"mean": 0.1175, "std": 0.0004, "source": "main paper Table 4"},
    "M4_NDCG10":   {"mean": 0.1173, "std": 0.0007, "source": "main paper Table 4"},
    "M1_NDCG10":   {"mean": 0.1139, "std": 0.0007, "source": "main paper Table 4"},
}


def fmt_hp() -> str:
    return f"lr{WINNER_LR}_wd{WINNER_WD}"


def run_seed(seed: int) -> bool:
    out_path = RERUN_DIR / f"{DATASET}_winner_seed{seed}.json"
    if out_path.exists():
        d = json.loads(out_path.read_text())
        ndcg = (d.get("test_metrics") or {}).get("NDCG@10")
        print(f"  [skip] seed={seed} (test NDCG@10={ndcg:.4f})")
        return True
    exp = f"r3_retune_{DATASET}_{fmt_hp()}_seed{seed}"
    cmd = [
        sys.executable, str(HARNESS),
        "--dataset", DATASET, "--seed", str(seed),
        "--lr", str(WINNER_LR), "--weight_decay", str(WINNER_WD),
        "--num_epochs", str(NUM_EPOCHS),
        "--patience", str(PATIENCE),
        "--device", DEVICE, "--exp_name", exp, "--out", str(out_path),
    ]
    print(f"\n  → seed={seed}, lr={WINNER_LR}, wd={WINNER_WD}")
    t0 = time.time()
    r = subprocess.run(cmd)
    dt = time.time() - t0
    if r.returncode != 0 or not out_path.exists():
        print(f"     FAILED after {dt/60:.1f} min")
        return False
    d = json.loads(out_path.read_text())
    val = (d.get("best_val_metrics") or {}).get("NDCG@10")
    test = (d.get("test_metrics") or {}).get("NDCG@10")
    best_ep = d.get("best_epoch")
    print(f"     done in {dt/60:.1f} min — best_epoch={best_ep}, val NDCG@10={val:.4f}, test NDCG@10={test:.4f}")
    return True


def aggregate() -> dict:
    """Compute 5-seed mean ± std + cross-paradigm context."""
    import numpy as np
    files = sorted(RERUN_DIR.glob(f"{DATASET}_winner_seed*.json"))
    rows = [json.loads(f.read_text()) for f in files]
    if not rows:
        return {}
    metrics = {}
    for k in ("NDCG@10", "Recall@10", "MRR"):
        vals = [(r.get("test_metrics") or {}).get(k) for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            metrics[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": [float(v) for v in vals],
            }
    r3_ndcg = metrics.get("NDCG@10", {}).get("mean")
    r2_ndcg = REFERENCE["R2_NDCG10"]["mean"]
    m7_ndcg = REFERENCE["M7_NDCG10"]["mean"]

    summary = {
        "protocol": (
            "R3 (HypernetReplacer) ML-20M, 5-seed winner rerun. "
            "Stage 1 ran the 4-cell grid on seed=42 at 200/20 budget; "
            f"Stage 2 (this run) reruns the winner across 5 seeds at the same 200/20 budget, "
            f"matching the LightGCN-SF training settings used throughout the paper (§4.3). "
            f"Winner from Stage 1: lr={WINNER_LR}, wd={WINNER_WD}, hidden=256, n_experts=4."
        ),
        "dataset": DATASET,
        "winner_lr": WINNER_LR,
        "winner_weight_decay": WINNER_WD,
        "n_seeds": len(rows),
        "seeds": [r.get("seed") for r in rows],
        "r3_test_metrics": metrics,
        "reference_paradigms_NDCG10": REFERENCE,
        "deltas_NDCG10": {
            "r3_vs_r2_pct": (r3_ndcg - r2_ndcg) / r2_ndcg * 100 if r3_ndcg else None,
            "r3_vs_m7_pct": (r3_ndcg - m7_ndcg) / m7_ndcg * 100 if r3_ndcg else None,
            "r2_vs_m7_pct": (r2_ndcg - m7_ndcg) / m7_ndcg * 100,
        },
        "wall_time_s_total": sum(r.get("wall_time_s", 0) for r in rows),
        "best_epochs": [r.get("best_epoch") for r in rows],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    return summary


def decision(summary: dict) -> None:
    """Print Path Z (3-density triangulation) interpretation."""
    if not summary:
        return
    r3_ndcg = summary["r3_test_metrics"].get("NDCG@10", {}).get("mean")
    r2_ndcg = REFERENCE["R2_NDCG10"]["mean"]
    m7_ndcg = REFERENCE["M7_NDCG10"]["mean"]
    r3_vs_m7 = summary["deltas_NDCG10"]["r3_vs_m7_pct"]
    r3_vs_r2 = summary["deltas_NDCG10"]["r3_vs_r2_pct"]

    print(f"\n{'='*70}")
    print("R3 ML-20M result vs reference paradigms")
    print(f"{'='*70}")
    print(f"R3 NDCG@10 = {r3_ndcg:.4f}")
    print(f"  vs M7={m7_ndcg:.4f}: {r3_vs_m7:+.1f}%")
    print(f"  vs R2={r2_ndcg:.4f}: {r3_vs_r2:+.1f}%")

    # Interpret in framework terms
    if abs(r3_vs_m7) <= 2.0:
        print("\n→ R3 ties M7 on ML-20M (within ±2%). Framework prediction confirmed: "
              "no replacer collapses on dense per-item density.")
    elif r3_vs_m7 < -5.0:
        print("\n→ R3 underperforms M7 on dense ML-20M. UNEXPECTED — investigate "
              "before paper claims; may indicate hparam mistuning or implementation issue.")
    else:
        print("\n→ R3 within ~5% of M7. Consistent with framework's coarse prediction "
              "(replacers don't collapse on dense data).")


if __name__ == "__main__":
    t0 = time.time()
    print(f"R3 ML-20M Stage-2 5-seed winner rerun (Path Z)")
    print(f"Winner from Stage 1: lr={WINNER_LR}, wd={WINNER_WD}")
    print(f"Budget: num_epochs={NUM_EPOCHS}, patience={PATIENCE}")
    print(f"Seeds: {SEEDS}")
    print(f"Output: {RERUN_DIR}/")

    for i, seed in enumerate(SEEDS, 1):
        print(f"\n{'='*70}\nSeed {i}/{len(SEEDS)}: {seed}\n{'='*70}")
        ok = run_seed(seed)
        if not ok:
            print(f"\nABORT after seed {seed}")
            sys.exit(1)

    summary = aggregate()
    print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    print(json.dumps(summary, indent=2))
    decision(summary)
    print(f"\n=== Total Stage-2 wall time: {(time.time()-t0)/60:.1f} min ===")
