#!/usr/bin/env python3
"""5-seed cold-start eval for M1/M4/M7/R1/R2 on ML-1M.

Mirror of `run_cold_start_amazon.py` (which already covers R1) with paths
pinned to ML-1M outputs. R1 uses our compute_metrics-comparable evaluator
that matches the rest of the paper.

Bucket boundaries match ML-20M and Amazon-Books exactly:
  cold:   <10 train interactions
  medium: 10-49
  warm:   >=50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import csr_matrix

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "code" / "benchmark"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / "scripts"))

from config import EMBED_DIM, LIGHTGCN_LAYERS  # noqa: E402
from data.dataset import InteractionData  # noqa: E402
from features.loader import FeatureLoader  # noqa: E402
from models.lightgcn import LightGCN, LightGCNSF  # noqa: E402
from models.kar import KAR  # noqa: E402
from eval_amazon_r1 import build_norm_adj as build_r1_adj, propagate as r1_propagate  # noqa: E402

DATA_DIR = BENCH / "data" / "processed_ml1m"
EMBEDDING_DIR = (
    REPO / "code" / "embedding_generator" / "output_ml1m" / "bge-large-en-v1.5"
)
CHECKPOINT_DIR = BENCH / "checkpoints_ml1m"
RESULTS_DIR = BENCH / "results_ml1m"
SEEDS = [42, 123, 456, 789, 2026]
TOP_K = [10, 20, 50]
K_VALUES = [10, 50, 100, 500, 1000]
R1_LAYER_NUM = 3   # matches the ml20m / ml1m_ours yaml block

CONFIGS_LIGHTGCN = [
    ("M1", "lightgcn",     "none"),
    ("M4", "lightgcn_sf",  "llm_profile"),
    ("M7", "lightgcn_sf",  "llm_prof_mood"),
    ("R2", "kar",          "llm_prof_mood"),
]

BUCKETS = {
    "cold":   (0, 10),
    "medium": (10, 50),
    "warm":   (50, float("inf")),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("cold_start_ml1m")


def _pick_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_lightgcn_model(name, n_users, n_items, feat_dim, norm_adj, device):
    if name == "lightgcn":
        m = LightGCN(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS).to(device)
    elif name == "kar":
        m = KAR(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS,
                feature_dim=feat_dim, n_experts=4).to(device)
    else:
        m = LightGCNSF(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS, feat_dim).to(device)
    m.set_adj(norm_adj)
    return m


def _features(name, key, fl, device):
    if name == "lightgcn":
        return None, 0
    parts = {"llm_profile": ["profile"], "llm_prof_mood": ["profile", "mood"]}[key]
    t = fl.get_combined_tensor(parts, device=device)
    return t, t.shape[1]


def _bucket_map(data):
    counts = defaultdict(int)
    for items in data.train_user_items.values():
        for it in items:
            counts[it] += 1
    out = {}
    for it in range(data.n_items):
        c = counts.get(it, 0)
        for name, (lo, hi) in BUCKETS.items():
            if lo <= c < hi:
                out[it] = name
                break
    return out


@torch.no_grad()
def _gt_ranks_lightgcn(model, data, device, batch=256):
    model.eval()
    triples = []
    eval_users = sorted(data.test_user_items.keys())
    n_items = data.n_items
    for start in range(0, len(eval_users), batch):
        bu = eval_users[start:start + batch]
        scores = model.predict(torch.LongTensor(bu).to(device)).cpu().numpy()
        for i, uid in enumerate(bu):
            gt = data.test_user_items.get(uid, set())
            if not gt:
                continue
            s = scores[i].copy()
            for t in data.train_user_items.get(uid, set()):
                s[t] = -np.inf
            order = np.argsort(-s)
            rank_of = np.empty(n_items, dtype=np.int32)
            rank_of[order] = np.arange(n_items)
            for g in gt:
                triples.append((uid, g, int(rank_of[g])))
    return triples


@torch.no_grad()
def _gt_ranks_r1(seed, data, adj_r1, device):
    ck = CHECKPOINT_DIR / "r1" / "bge-large-en-v1.5" / f"seed-{seed}" / "best_model.pt"
    if not ck.exists():
        return None
    sd = torch.load(ck, map_location=device, weights_only=False)
    user_emb = sd["user_embeds"].to(device)
    item_emb = sd["item_embeds"].to(device)
    user_final, item_final = r1_propagate(adj_r1, user_emb, item_emb, R1_LAYER_NUM)
    triples = []
    eval_users = sorted(data.test_user_items.keys())
    n_items = data.n_items
    bs = 256
    for start in range(0, len(eval_users), bs):
        bu = eval_users[start:start + bs]
        ur = user_final[torch.LongTensor(bu).to(device)]
        scores = (ur @ item_final.T).cpu().numpy()
        for i, uid in enumerate(bu):
            gt = data.test_user_items.get(uid, set())
            if not gt:
                continue
            s = scores[i].copy()
            for t in data.train_user_items.get(uid, set()):
                s[t] = -np.inf
            order = np.argsort(-s)
            rank_of = np.empty(n_items, dtype=np.int32)
            rank_of[order] = np.arange(n_items)
            for g in gt:
                triples.append((uid, g, int(rank_of[g])))
    return triples


def _bucket_metrics(triples, bucket_map):
    user_gt = defaultdict(list)
    for uid, it, r in triples:
        user_gt[uid].append((it, r))
    out = {}
    for bname in BUCKETS:
        per_user = defaultdict(list)
        ranks_in_bucket = []
        for uid, pairs in user_gt.items():
            gt_b = [(it, r) for it, r in pairs if bucket_map.get(it) == bname]
            if not gt_b:
                continue
            ranks_b = [r for _, r in gt_b]
            ranks_in_bucket.extend(ranks_b)
            for K in K_VALUES:
                hits = [r for r in ranks_b if r < K]
                per_user[f"Recall@{K}"].append(len(hits) / len(gt_b))
                dcg = sum(1.0 / np.log2(r + 2) for r in hits)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_b), K)))
                per_user[f"NDCG@{K}"].append(dcg / idcg if idcg > 0 else 0.0)
            per_user["MRR_full"].append(1.0 / (min(ranks_b) + 1))
        per_bucket = {k: float(np.mean(v)) for k, v in per_user.items()}
        per_bucket["n_users"] = len(per_user.get("Recall@10", []))
        per_bucket["n_gt_items"] = len(ranks_in_bucket)
        if ranks_in_bucket:
            arr = np.asarray(ranks_in_bucket) + 1
            per_bucket["rank_median"] = float(np.median(arr))
            per_bucket["rank_mean"] = float(arr.mean())
        out[bname] = per_bucket
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = _pick_device(args.device)
    encoder = "bge-large-en-v1.5"
    logger.info(f"Device: {device}")

    data = InteractionData(data_dir=DATA_DIR)
    fl = FeatureLoader(data_dir=DATA_DIR, embedding_dir=EMBEDDING_DIR)
    norm_adj = data.get_norm_adj().to(device)
    bucket_map = _bucket_map(data)
    counts = {b: sum(1 for v in bucket_map.values() if v == b) for b in BUCKETS}
    logger.info(f"Item buckets: {counts}")

    # Build R1's adjacency (different normalization than LightGCN's)
    adj_r1 = build_r1_adj(data.n_users, data.n_items, data.train_user_items, device)

    all_results = {}
    for seed in SEEDS:
        # LightGCN-family configs
        for label, model_name, feat_key in CONFIGS_LIGHTGCN:
            ck = CHECKPOINT_DIR / label.lower() / encoder / f"seed-{seed}" / "best_model.pt"
            if not ck.exists():
                logger.warning(f"[{label} seed={seed}] missing {ck}")
                continue
            logger.info(f"[{label} seed={seed}] {ck.relative_to(REPO)}")
            feat, fdim = _features(model_name, feat_key, fl, device)
            model = _build_lightgcn_model(model_name, data.n_users, data.n_items, fdim, norm_adj, device)
            try:
                state = torch.load(ck, map_location=device, weights_only=False)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                model.load_state_dict(state)
            except RuntimeError as e:
                logger.warning(f"  skip — {str(e)[:80]}")
                continue
            if feat is not None:
                model.set_features(feat)
            triples = _gt_ranks_lightgcn(model, data, device)
            all_results.setdefault(label, {})[seed] = _bucket_metrics(triples, bucket_map)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        # R1
        logger.info(f"[R1 seed={seed}] (RLMRec lightgcn_gene)")
        triples = _gt_ranks_r1(seed, data, adj_r1, device)
        if triples is not None:
            all_results.setdefault("R1", {})[seed] = _bucket_metrics(triples, bucket_map)

    # Aggregate
    summary = {}
    for label in list(all_results.keys()):
        per_seed = all_results[label]
        if not per_seed:
            continue
        seeds_done = sorted(per_seed.keys())
        summary[label] = {"seeds": seeds_done, "buckets": {}}
        for bname in BUCKETS:
            metric_names = sorted(set(per_seed[seeds_done[0]][bname].keys()))
            agg = {}
            for met in metric_names:
                vals = [per_seed[s][bname][met] for s in seeds_done]
                if met in ("n_users", "n_gt_items"):
                    agg[met] = vals[0]
                else:
                    agg[met] = {"mean": float(np.mean(vals)),
                                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                "values": [float(v) for v in vals]}
            summary[label]["buckets"][bname] = agg

    out_path = RESULTS_DIR / "cold_start_5seeds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "dataset": "ml-1m",
        "encoder": encoder,
        "seeds": SEEDS,
        "k_values": K_VALUES,
        "buckets": {k: list(v) for k, v in BUCKETS.items()},
        "bucket_item_counts": counts,
        "configs": summary,
    }, indent=2, default=str))
    logger.info(f"Saved {out_path.relative_to(REPO)}")

    # Pretty print
    print()
    print("=" * 100)
    print("ML-1M cold-start 5-seed summary")
    print("=" * 100)
    print(f"{'Cfg':<5} {'Bucket':<7} {'rank_med':>9} {'R@10':>8} {'R@100':>9} {'R@1000':>9} {'NDCG@10':>9} {'MRR':>8} {'n_users':>8}")
    for label in ("M1", "M4", "M7", "R1", "R2"):
        if label not in summary:
            continue
        for bname in BUCKETS:
            b = summary[label]["buckets"][bname]
            rm = b['rank_median']['mean']
            print(f"{label:<5} {bname:<7} {rm:>9.0f} "
                  f"{b['Recall@10']['mean']:>8.4f} {b['Recall@100']['mean']:>9.4f} "
                  f"{b['Recall@1000']['mean']:>9.4f} {b['NDCG@10']['mean']:>9.4f} "
                  f"{b['MRR_full']['mean']:>8.4f} {b['n_users']:>8d}")
        print()


if __name__ == "__main__":
    main()
