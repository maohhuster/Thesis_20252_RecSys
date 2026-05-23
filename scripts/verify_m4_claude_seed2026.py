#!/usr/bin/env python3
"""Verify Claude M4 seed-2026 metrics by loading the saved best_model.pt and
re-evaluating the ML-20M test set. Compares to the xlsx-extracted values.

Used to check whether the recorded MRR=0.2307 (the seed-2026 outlier) is real
or a typo.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "code" / "benchmark"
sys.path.insert(0, str(BENCH))

import numpy as np
import torch

from config import EMBED_DIM, LIGHTGCN_LAYERS
from data.dataset import InteractionData
from features.loader import FeatureLoader
from models.lightgcn import LightGCNSF
from evaluate import evaluate_model

DATA_DIR = BENCH / "data" / "processed"
EMB_DIR  = REPO / "code" / "embedding_generator" / "output" / "bge-large-en-v1.5"
CKPT     = BENCH / "checkpoints" / "m4" / "bge-large-en-v1.5" / "seed-2026" / "best_model.pt"

# Extracted xlsx values for comparison
XLSX_RECORDED = {"NDCG@10": 0.1185, "Recall@10": 0.0499, "MRR": 0.2307}


def main():
    if not CKPT.exists():
        print(f"ERROR: missing {CKPT}"); sys.exit(2)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[verify] device = {device}")
    print(f"[verify] ckpt   = {CKPT} ({CKPT.stat().st_size/1e6:.1f} MB)")

    # Build M4 = LightGCN-SF + profile (1024-d) — same architecture as paper
    data = InteractionData(data_dir=DATA_DIR)
    fl   = FeatureLoader(data_dir=DATA_DIR, embedding_dir=EMB_DIR)
    feat = fl.get_combined_tensor(["profile"], device=device)
    print(f"[verify] n_users={data.n_users}, n_items={data.n_items}, feat_dim={feat.shape[1]}")

    model = LightGCNSF(data.n_users, data.n_items, EMBED_DIM, LIGHTGCN_LAYERS,
                       feature_dim=feat.shape[1]).to(device)
    model.set_adj(data.get_norm_adj().to(device))
    model.set_features(feat)

    sd = torch.load(CKPT, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    print(f"[verify] best_model.pt loaded ({len(sd)} tensors)")

    # Run val + test
    t0 = time.time()
    val_metrics = evaluate_model(model, data, split="val", device=device)
    test_metrics = evaluate_model(model, data, split="test", device=device)
    wall = time.time() - t0
    print(f"[verify] eval done in {wall:.1f}s\n")

    print("=" * 70)
    print("M4 CLAUDE seed-2026 — best_model.pt re-evaluation")
    print("=" * 70)
    print(f"\nXLSX-recorded (paper Table 4 source-of-truth):")
    for k, v in XLSX_RECORDED.items():
        print(f"  {k:12s} = {v:.4f}")

    print(f"\nFreshly-evaluated from saved checkpoint:")
    for k in ("NDCG@10", "Recall@10", "MRR"):
        v = test_metrics.get(k)
        if v is None: continue
        recorded = XLSX_RECORDED.get(k)
        diff = v - recorded if recorded is not None else None
        flag = ""
        if diff is not None and abs(diff) > 0.001:
            flag = "  ⚠ DIFFERS FROM XLSX"
        print(f"  {k:12s} = {v:.4f}   (xlsx: {recorded:.4f}, diff: {diff:+.4f}){flag}")

    print(f"\nValidation metrics for context:")
    for k in ("NDCG@10", "Recall@10", "MRR"):
        v = val_metrics.get(k)
        if v is not None:
            print(f"  val {k:12s} = {v:.4f}")

    # Save for the record
    out = {
        "config": "M4 (LightGCN-SF + LLM profile, bge-large-en-v1.5, Claude)",
        "seed": 2026,
        "checkpoint": str(CKPT),
        "device": device,
        "xlsx_recorded": XLSX_RECORDED,
        "freshly_evaluated_test": {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))},
        "freshly_evaluated_val":  {k: float(v) for k, v in val_metrics.items()  if isinstance(v, (int, float))},
        "wall_time_s": round(wall, 1),
    }
    out_path = REPO / "code" / "benchmark" / "results_cross_llm" / "verify_m4_claude_seed2026.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
