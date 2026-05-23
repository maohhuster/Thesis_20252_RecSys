#!/usr/bin/env python3
"""Evaluate the 5 R3 ML-20M winner checkpoints on the test split.

Loads each best_model.pt from code/benchmark/checkpoints/r3/bge-large-en-v1.5/seed-{S}/
and writes:
  - code/benchmark/results/r3_metrics.json — the Tier-3 aggregate, SAME schema as
    results/{r1,r1plus,r2}_metrics.json (seeds_evaluated / per_seed / summary).
  - code/benchmark/hparams/r3/winner_5seed/ml20m_winner_seed{S}.json — per-seed detail in
    R3's established triangulation home (same schema as the ml1m_/amazon_ files;
    see code/benchmark/hparams/r3/README.md for why R3 lives there).

Eval-only — does not train. Reuses the canonical InteractionData + FeatureLoader +
HypernetReplacer + evaluate_model pipeline so metrics are byte-equivalent to what
the training-loop test-eval would have produced.

The grid winner hparams (lr, wd) are inferred from training_state.pt → optimizer_state_dict.
"""
from __future__ import annotations

import json, random, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "code" / "benchmark"))

from config import EMBED_DIM, LIGHTGCN_LAYERS
from data.dataset import InteractionData
from features.loader import FeatureLoader
from models.hypernet_replacer import HypernetReplacer
from evaluate import evaluate_model


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


CKPT_ROOT   = REPO / "code" / "benchmark" / "checkpoints" / "r3" / "bge-large-en-v1.5"
RESULTS_DIR = REPO / "code" / "benchmark" / "results"
# Per-seed detail lives in the established, documented R3 retune home (repo-root
# code/benchmark/hparams/r3/ — R3 is triangulation-only, no main-results-table entry; see
# code/benchmark/hparams/r3/README.md). This script only ADDS the Tier-3 aggregate
# results/r3_metrics.json (schema-parallel to results/{r1,r1plus,r2}_metrics.json).
OUT_ROOT    = REPO / "code" / "benchmark" / "hparams" / "r3" / "winner_5seed"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2026]
HIDDEN = 256
# Authoritative R3 ML-20M 4-cell-grid winner hparams (r3_retune_summary.json >
# ml20m.winner_*). Note: the optimizer_state_dict serialised inside training_state.pt records
# weight_decay=0 for an extra non-tuned PyTorch rescorer param group (this
# group is not part of the 4-cell grid search space); we write the documented
# winner value so the per-seed JSONs stay consistent with the summary + README.
WINNER_LR = 3e-4
WINNER_WD = 1e-5


def main() -> None:
    device = pick_device()
    print(f"[eval] device={device}")

    # Build data + features once (shared across all seeds)
    data_dir = REPO / "code" / "benchmark" / "data" / "processed"
    emb_dir  = REPO / "code" / "embedding_generator" / "output" / "bge-large-en-v1.5"

    print(f"[eval] loading data from {data_dir.relative_to(REPO)}")
    data = InteractionData(data_dir=data_dir)
    fl   = FeatureLoader(data_dir=data_dir, embedding_dir=emb_dir)
    feat = fl.get_combined_tensor(["profile", "mood"], device=device)
    norm_adj = data.get_norm_adj().to(device)
    print(f"[eval] n_users={data.n_users}, n_items={data.n_items}, feat_dim={feat.shape[1]}")

    per_seed: dict[str, dict] = {}
    for seed in SEEDS:
        ckpt_dir = CKPT_ROOT / f"seed-{seed}"
        best_pt  = ckpt_dir / "best_model.pt"
        train_pt = ckpt_dir / "training_state.pt"
        if not best_pt.exists():
            print(f"[eval] seed={seed}: best_model.pt MISSING, skipping")
            continue

        # Reproducibility
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        # Build fresh model and load weights
        model = HypernetReplacer(
            data.n_users, data.n_items, EMBED_DIM, LIGHTGCN_LAYERS,
            feature_dim=feat.shape[1], hidden=HIDDEN,
        ).to(device)
        model.set_adj(norm_adj)
        model.set_features(feat)
        state = torch.load(best_pt, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.eval()

        # Recover hparams + epoch metadata from training_state.pt (best-effort)
        lr, wd, best_epoch = None, None, None
        best_val_metrics = None
        if train_pt.exists():
            ts = torch.load(train_pt, map_location="cpu", weights_only=False)
            best_epoch = ts.get("best_epoch")
            best_val_metrics = ts.get("best_metrics")
            opt = ts.get("optimizer_state_dict", {})
            for g in opt.get("param_groups", []):
                if "lr" in g and lr is None:
                    lr = g["lr"]
                if "weight_decay" in g and wd is None:
                    wd = g["weight_decay"]

        # The optimizer state records wd=0 for an extra non-tuned rescorer param group (not in the
        # 4-cell grid). Persist the documented authoritative winner hparams so
        # the per-seed JSONs stay consistent with r3_retune_summary.json + README.
        lr, wd = WINNER_LR, WINNER_WD

        # Run test-split full-ranking eval
        t0 = time.time()
        test_metrics = evaluate_model(model, data, split="test", device=device)
        wall_s = time.time() - t0
        print(f"[eval] seed={seed}: NDCG@10={test_metrics['NDCG@10']:.4f}  "
              f"Recall@10={test_metrics['Recall@10']:.4f}  MRR={test_metrics['MRR']:.4f}  "
              f"({wall_s:.1f}s, lr={lr}, wd={wd}, best_epoch={best_epoch})")

        out = {
            "model": "HypernetReplacer (R3)",
            "dataset": "ml20m",
            "seed": seed,
            "lr": lr,
            "weight_decay": wd,
            "hidden": HIDDEN,
            "exp_name": f"r3_ml20m_seed{seed}_eval_only",
            "num_epochs_requested": None,
            "patience": None,
            "device": device,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "best_epoch": best_epoch,
            "wall_time_s": wall_s,
            "_note": "eval-only re-evaluation of pre-trained checkpoint",
        }
        out_path = OUT_ROOT / f"ml20m_winner_seed{seed}.json"
        out_path.write_text(json.dumps(out, indent=2, default=str))
        print(f"[eval] wrote {out_path.relative_to(REPO)}")

        per_seed[str(seed)] = {k: float(v) for k, v in test_metrics.items()}

    # Tier-3 aggregate — SAME schema as results/{r1,r1plus,r2}_metrics.json
    evaluated = [s for s in SEEDS if str(s) in per_seed]
    keys = ("NDCG@10", "Recall@10", "MRR")
    summary = {}
    for k in keys:
        v = np.array([per_seed[str(s)][k] for s in evaluated])
        summary[k] = {"mean": float(v.mean()),
                      "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "values": [float(x) for x in v]}
    print("\n=== R3 (HypernetReplacer) on ML-20M ===")
    for k in keys:
        print(f"  {k:10s} {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")

    agg = RESULTS_DIR / "r3_metrics.json"
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_text(json.dumps(
        {"layer_num": LIGHTGCN_LAYERS, "hidden": HIDDEN, "features": "llm_prof_mood",
         "seeds_evaluated": evaluated, "per_seed": per_seed, "summary": summary,
         "_provenance": ("Eval-only re-evaluation of "
                         "code/benchmark/checkpoints/r3/bge-large-en-v1.5 via project "
                         "HypernetReplacer+evaluate_model. Read-only; checkpoints unmodified. "
                         "Per-seed detail in results/r3/ml20m/.")},
        indent=2, default=str))
    print(f"[eval] wrote {agg.relative_to(REPO)}")


if __name__ == "__main__":
    main()
