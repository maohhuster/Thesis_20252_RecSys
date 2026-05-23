#!/usr/bin/env python3
"""Evaluate R2 (KAR-MoE) ML-20M checkpoints on the test set.

Faithful re-evaluation that REUSES the project's exact construction
(run_experiment.build_model + models.kar.KAR + FeatureLoader +
InteractionData + evaluate.evaluate_model). The ONLY difference from the
standard run_experiment/train resume path is that checkpoints are loaded
with an explicit `map_location` — the committed R2 checkpoints were saved
on CUDA and train.py:211 omits map_location, so the canonical resume path
crashes on CPU/MPS machines. This script is read-only (no checkpoint or
training-state writes).

Usage:
    python3 scripts/eval_ml20m_r2.py --device mps
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "code" / "benchmark"
sys.path.insert(0, str(BENCH))
from config import FEATURE_CONFIGS, DATA_DIR, EMBEDDING_DIR  # noqa: E402
from data.dataset import InteractionData  # noqa: E402
from features.loader import FeatureLoader  # noqa: E402
from evaluate import evaluate_model  # noqa: E402
from run_experiment import build_model  # noqa: E402

CK_DIR = BENCH / "checkpoints" / "r2" / "bge-large-en-v1.5"   # ML-20M base (unsuffixed)
RESULTS_DIR = BENCH / "results"
SEEDS = [42, 123, 456, 789, 2026]
FEATS = "llm_prof_mood"   # R2 = (kar, llm_prof_mood) per config.CONFIG_NAME_MAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    dev = args.device

    data = InteractionData(data_dir=DATA_DIR)
    norm_adj = data.get_norm_adj().to(dev)
    fl = FeatureLoader(data_dir=DATA_DIR, embedding_dir=EMBEDDING_DIR)
    fnames = FEATURE_CONFIGS[FEATS]
    fdim = fl.get_feature_dim(fnames)
    model = build_model("kar", data.n_users, data.n_items, fdim, norm_adj)
    model.set_features(fl.get_combined_tensor(fnames, device=dev))
    model.to(dev)
    print(f"n_users={data.n_users} n_items={data.n_items} feat_dim={fdim}")

    per_seed = {}
    for s in args.seeds:
        ck = CK_DIR / f"seed-{s}" / "best_model.pt"
        sd = torch.load(ck, map_location=dev, weights_only=True)  # the fix vs train.py:211
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            m = evaluate_model(model, data, split="test", device=dev)
        per_seed[str(s)] = {k: float(v) for k, v in m.items()}
        print(f"seed {s}: NDCG@10={m['NDCG@10']:.4f} Recall@10={m['Recall@10']:.4f} MRR={m['MRR']:.4f}")

    keys = ("NDCG@10", "Recall@10", "MRR")
    summary = {}
    for k in keys:
        v = np.array([per_seed[str(s)][k] for s in args.seeds])
        summary[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                      "values": [float(x) for x in v]}
    print("\n=== R2 (KAR-MoE) on ML-20M ===")
    for k in keys:
        print(f"  {k:10s} {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")

    out = RESULTS_DIR / "r2_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"layer_num": 3, "n_experts": 4, "features": FEATS,
         "seeds_evaluated": args.seeds, "per_seed": per_seed, "summary": summary,
         "_provenance": ("Re-evaluation of code/benchmark/checkpoints/r2/bge-large-en-v1.5 "
                         "via project build_model('kar')+evaluate_model; map_location fix vs "
                         "train.py:211 (CUDA-saved ckpts). Read-only; checkpoints unmodified.")},
        indent=2))
    print(f"Saved {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
