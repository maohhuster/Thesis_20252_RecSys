#!/usr/bin/env python3
"""R3 (HypernetReplacer) retune harness for ML-1M and Amazon-Books-2018.

replacer triangulation: second replacer instantiation, structurally distant from R2.

Pre-registered 4-cell grid:
    lr           ∈ {3e-4, 1e-3}
    weight_decay ∈ {1e-5, 1e-4}
    hypernet hidden = 256 (fixed)

Each invocation runs ONE (dataset, lr, weight_decay, seed) cell and writes a JSON.

Smoke test (one minimal run on Mac, ~1 min on MPS):
    python3 scripts/run_r3_retune.py \
        --dataset ml1m --seed 42 --lr 1e-3 --weight_decay 1e-4 \
        --exp_name smoke --num_epochs 2 --patience 5 --device mps \
        --out /tmp/r3_smoke.json
"""
from __future__ import annotations

import argparse, json, os, random, shutil, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "code" / "benchmark"))

import numpy as np
import torch

from config import EMBED_DIM, LIGHTGCN_LAYERS, NUM_EPOCHS, PATIENCE
from data.dataset import InteractionData
from features.loader import FeatureLoader
from models.hypernet_replacer import HypernetReplacer
from train import train_model


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ml1m", "amazon", "ml20m"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--weight_decay", type=float, required=True)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--exp_name", required=True)
    p.add_argument("--out", required=True, help="JSON output path")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = pick_device(args.device)
    print(f"[r3_retune] device={device}, dataset={args.dataset}, seed={args.seed}, "
          f"lr={args.lr}, wd={args.weight_decay}, hidden={args.hidden}, exp={args.exp_name}")

    if args.dataset == "ml1m":
        data_dir = REPO / "code" / "benchmark" / "data" / "processed_ml1m"
        emb_dir  = REPO / "code" / "embedding_generator" / "output_ml1m" / "bge-large-en-v1.5"
    elif args.dataset == "amazon":
        data_dir = REPO / "code" / "benchmark" / "data" / "processed_amazon"
        emb_dir  = REPO / "code" / "embedding_generator" / "output_amazon" / "bge-large-en-v1.5"
    else:  # ml20m
        data_dir = REPO / "code" / "benchmark" / "data" / "processed"
        emb_dir  = REPO / "code" / "embedding_generator" / "output" / "bge-large-en-v1.5"

    data = InteractionData(data_dir=data_dir)
    fl   = FeatureLoader(data_dir=data_dir, embedding_dir=emb_dir)
    feat = fl.get_combined_tensor(["profile", "mood"], device=device)
    print(f"[r3_retune] n_users={data.n_users}, n_items={data.n_items}, feat_dim={feat.shape[1]}")

    model = HypernetReplacer(
        data.n_users, data.n_items, EMBED_DIM, LIGHTGCN_LAYERS,
        feature_dim=feat.shape[1], hidden=args.hidden,
    ).to(device)
    model.set_adj(data.get_norm_adj().to(device))
    model.set_features(feat)

    out_path     = Path(args.out)
    scratch_root = REPO / "code" / "benchmark" / "hparams" / "r3" / "_scratch"
    ckpt_root    = scratch_root / "checkpoints"
    res_root     = scratch_root / "results"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    res_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = train_model(
        model=model,
        interaction_data=data,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        patience=args.patience,
        eval_every=args.eval_every,
        experiment_name=args.exp_name,
        resume=False,
        checkpoint_dir=ckpt_root,
        results_dir=res_root,
        seed=args.seed,
    )
    wall_s = time.time() - t0

    # Persist the winner checkpoint into the project M-config layout (NOT only
    # _scratch, which gets cleaned). Base ml20m -> checkpoints/; non-base ->
    # checkpoints_<ds>/  — same convention as M0–M9 / R1 / R1-plus / R2 / SASRec.
    _proj = {"ml1m": "checkpoints_ml1m", "amazon": "checkpoints_amazon",
             "ml20m": "checkpoints"}[args.dataset]
    canon = (REPO / "code" / "benchmark" / _proj / "r3" /
             "bge-large-en-v1.5" / f"seed-{args.seed}")
    canon.mkdir(parents=True, exist_ok=True)
    _exp_dir = ckpt_root / args.exp_name
    for fn in ("best_model.pt", "training_state.pt"):
        src = _exp_dir / fn
        if src.exists():
            shutil.copy2(src, canon / fn)
    ckpt_path = canon / "best_model.pt"
    print(f"[r3_retune] checkpoint -> {ckpt_path.relative_to(REPO)}"
          f" ({'OK' if ckpt_path.exists() else 'MISSING — train_model produced no best_model.pt'})")

    out = {
        "model": "HypernetReplacer (R3)",
        "dataset": args.dataset,
        "seed": args.seed,
        "checkpoint": str(ckpt_path.relative_to(REPO)),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden": args.hidden,
        "exp_name": args.exp_name,
        "num_epochs_requested": args.num_epochs,
        "patience": args.patience,
        "device": device,
        "best_val_metrics": result.get("best_val_metrics") or result.get("best_metrics"),
        "test_metrics": result.get("test_metrics"),
        "best_epoch": result.get("best_epoch"),
        "wall_time_s": wall_s,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
