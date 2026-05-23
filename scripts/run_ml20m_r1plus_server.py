#!/usr/bin/env python3
"""5-seed orchestrator for R1-plus on full ML-20M (server, CUDA-default).

Closes the dense-end cell of the regularizer-class density triangulation
(R1-gene already at -2.2% vs M7 here; R1-plus expected to land in the
same n.s. tie/loss regime — a tie corroborates the framework's monotone
prediction; a structurally-different result would falsify the
regularizer-class claim at the dense end).

Idempotent: re-running skips seeds whose marker JSON + checkpoint already exist.
After each seed completes the orchestrator auto-runs eval_ml20m_r1plus.py for
that seed and merges test-set NDCG@10/Recall@10/MRR into the seed's marker
JSON, so progress is visible incrementally without waiting for all 5 seeds.
nohup-friendly. CUDA by default; override with --device mps for Apple.

Usage on the server:
    nohup python3 scripts/run_ml20m_r1plus_server.py > /tmp/r1plus_ml20m.log 2>&1 &
    tail -f /tmp/r1plus_ml20m.log

Quick smoke test (1 seed, will run to convergence):
    python3 scripts/run_ml20m_r1plus_server.py --seeds 42

Pre-flight checks:
    - data/ml20m/{itm,usr}_emb_np.pkl present (Claude+bge embeddings)
    - lightgcn_plus.yml > model.ml20m block present (layer_num=3, kd_weight=1e-2)
    - lightgcn_plus.yml > train.patience=10 (bumped from upstream 5)
    - GPU has >= 24GB VRAM recommended (A100/L4/A40); fall back to batch=2048 if OOM
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RLMREC_DIR = REPO / "code" / "benchmark" / "external" / "RLMRec"
DATA_DIR = RLMREC_DIR / "data" / "ml20m"
RESULTS_DIR = REPO / "code" / "benchmark" / "results" / "r1plus" / "ml20m"
LOG_DIR = REPO / "code" / "benchmark" / "results" / "logs"
CKPT_BASE = REPO / "code" / "benchmark" / "checkpoints" / "r1plus" / "bge-large-en-v1.5"
SEEDS = [42, 123, 456, 789, 2026]


def preflight() -> None:
    missing = []
    for f in ("itm_emb_np.pkl", "usr_emb_np.pkl", "trn_mat.pkl", "val_mat.pkl", "tst_mat.pkl"):
        if not (DATA_DIR / f).exists():
            missing.append(str(DATA_DIR / f))
    if missing:
        print("ERROR: missing prepared ML-20M data files:")
        for m in missing:
            print(f"  - {m}")
        print("Run: cd code/benchmark/external && python3 prepare_rlmrec_data.py --item-features profile")
        sys.exit(1)
    yml = RLMREC_DIR / "encoder" / "config" / "modelconf" / "lightgcn_plus.yml"
    txt = yml.read_text()
    if "patience: 10" not in txt:
        print(f"WARNING: {yml} does not have `patience: 10`. Current value:")
        for ln in txt.splitlines():
            if "patience" in ln:
                print("  ", ln)
        print("Either bump it or accept upstream patience=5.")


def run_seed(seed: int, device: str, cuda: str, batch_size: int | None) -> dict:
    marker = RESULTS_DIR / f"seed-{seed}-marker.json"
    ckpt = CKPT_BASE / f"seed-{seed}" / "best_model.pt"

    if marker.exists() and ckpt.exists():
        print(f"[seed={seed}] SKIP — marker + checkpoint already present")
        return {"seed": seed, "status": "skipped"}

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (CKPT_BASE / f"seed-{seed}").mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"r1plus-ml20m-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    cmd = [
        "python3", "encoder/train_encoder.py",
        "--model", "lightgcn_plus",
        "--dataset", "ml20m",
        "--device", device,
        "--cuda", cuda,
        "--seed", str(seed),
    ]

    print(f"\n[seed={seed}] start  device={device} cuda={cuda} log={log_path}")
    start = time.time()
    with log_path.open("w") as logf:
        proc = subprocess.run(
            cmd, cwd=RLMREC_DIR, stdout=logf, stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed_min = (time.time() - start) / 60.0
    print(f"[seed={seed}] done   exit={proc.returncode}  wall={elapsed_min:.1f} min")

    # Move native checkpoint to project location
    native_ckpt = RLMREC_DIR / "encoder" / "checkpoint" / "lightgcn_plus" / f"lightgcn_plus-ml20m-{seed}.pth"
    if native_ckpt.exists():
        ckpt.write_bytes(native_ckpt.read_bytes())
        print(f"[seed={seed}] checkpoint → {ckpt}")
    else:
        print(f"[seed={seed}] WARNING: native checkpoint not found at {native_ckpt}")

    marker_payload = {
        "experiment": f"r1plus_ml20m_seed{seed}",
        "status": "complete" if proc.returncode == 0 else f"exit_{proc.returncode}",
        "seed": seed,
        "device": device,
        "wall_minutes": round(elapsed_min, 1),
        "log": str(log_path),
        "checkpoint": str(ckpt),
    }
    marker.write_text(json.dumps(marker_payload, indent=2))

    # Auto-evaluate this seed immediately so per-seed test-set metrics are visible
    # before all 5 seeds finish. Failures here do not abort the overall run.
    if proc.returncode == 0 and ckpt.exists():
        eval_script = REPO / "scripts" / "eval_ml20m_r1plus.py"
        if eval_script.exists():
            print(f"[seed={seed}] running incremental eval ...")
            try:
                eval_proc = subprocess.run(
                    ["python3", str(eval_script),
                     "--device", device, "--seeds", str(seed),
                     "--layer-num", "3"],
                    cwd=REPO, check=False, timeout=3600,
                )
                if eval_proc.returncode != 0:
                    print(f"[seed={seed}] eval returned exit={eval_proc.returncode} "
                          f"(non-fatal; metrics will be filled by a later --seeds run)")
                else:
                    # Re-load the marker since eval merged test_metrics into it
                    marker_payload = json.loads(marker.read_text())
                    tm = marker_payload.get("test_metrics", {})
                    if tm:
                        print(f"[seed={seed}] test  NDCG@10={tm.get('NDCG@10', '?'):.4f}  "
                              f"Recall@10={tm.get('Recall@10', '?'):.4f}  "
                              f"MRR={tm.get('MRR', '?'):.4f}")
            except subprocess.TimeoutExpired:
                print(f"[seed={seed}] eval timed out after 1 h; run manually later")
            except Exception as e:
                print(f"[seed={seed}] eval raised {type(e).__name__}: {e} (non-fatal)")

    return marker_payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--cuda", default="0", help="CUDA device index (ignored if --device != cuda)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="Seeds to run (default: all 5)")
    ap.add_argument("--batch-size", type=int, default=None, help="Override batch size (passes through env; YAML controls actual)")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    if not args.skip_preflight:
        preflight()

    print(f"═══ R1-plus ML-20M orchestrator ═══")
    print(f"  device : {args.device}{' (cuda='+args.cuda+')' if args.device == 'cuda' else ''}")
    print(f"  seeds  : {args.seeds}")
    print(f"  data   : {DATA_DIR}")
    print(f"  results: {RESULTS_DIR}")
    print(f"  ckpts  : {CKPT_BASE}")

    overall_start = time.time()
    summary = []
    for seed in args.seeds:
        summary.append(run_seed(seed, args.device, args.cuda, args.batch_size))

    total_min = (time.time() - overall_start) / 60.0
    print(f"\n═══ All done — {total_min:.1f} min total ═══")
    for s in summary:
        if s["status"] == "skipped":
            print(f"  seed={s['seed']:5d}  SKIPPED")
        else:
            print(f"  seed={s['seed']:5d}  {s['status']:<10s} {s.get('wall_minutes', '?')} min")
            # Show per-seed metrics if eval succeeded
            marker_path = RESULTS_DIR / f"seed-{s['seed']}-marker.json"
            if marker_path.exists():
                m = json.loads(marker_path.read_text()).get("test_metrics", {})
                if m:
                    print(f"                  NDCG@10={m.get('NDCG@10', '?'):.4f}  "
                          f"Recall@10={m.get('Recall@10', '?'):.4f}  "
                          f"MRR={m.get('MRR', '?'):.4f}")

    print(f"\nFor 5-seed aggregate (mean ± std), run:")
    print(f"  python3 scripts/eval_ml20m_r1plus.py --device {args.device}")
    print(f"  → produces code/benchmark/results/r1plus_metrics.json")


if __name__ == "__main__":
    main()
