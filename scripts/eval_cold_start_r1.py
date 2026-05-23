#!/usr/bin/env python3
"""Bucket-stratified cold-start eval for R1 (RLMRec-gene) on Amazon-Books.

R1 cannot be plugged into `run_cold_start_amazon.py` directly because it
uses RLMRec's checkpoint format and 32-d embeddings with a different
propagation rule (sum over layers vs.~mean). This script mirrors the
bucket logic of `run_cold_start_amazon.py` but plugs in R1's inference.

Reads:
    code/benchmark/results_amazon/cold_start_5seeds.json  (M1/M4/M7/R2)
Writes:
    code/benchmark/results_amazon/cold_start_5seeds.json  (adds R1)

Usage:
    python3 scripts/eval_cold_start_r1.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "code" / "benchmark"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / "scripts"))
from data.dataset import InteractionData  # noqa: E402
from eval_amazon_r1 import build_norm_adj, propagate, CK_DIR, SEEDS  # noqa: E402

DATA_DIR = BENCH / "data" / "processed_amazon"
RESULTS_JSON = BENCH / "results_amazon" / "cold_start_5seeds.json"
K_VALUES = [10, 50, 100, 500, 1000]
LAYER_NUM = 2

BUCKETS = {
    "cold":   (0, 10),
    "medium": (10, 50),
    "warm":   (50, float("inf")),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("r1_cold")


def _bucket_map(data) -> dict:
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
def _gt_ranks_r1(seed, data, adj, device="cpu"):
    ck = CK_DIR / f"seed-{seed}" / "best_model.pt"
    sd = torch.load(ck, map_location=device, weights_only=False)
    user_emb = sd["user_embeds"].to(device)
    item_emb = sd["item_embeds"].to(device)
    user_final, item_final = propagate(adj, user_emb, item_emb, LAYER_NUM)

    triples = []
    eval_users = sorted(data.test_user_items.keys())
    n_items = data.n_items
    bs = 256
    for start in range(0, len(eval_users), bs):
        bu = eval_users[start: start + bs]
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
    logger.info(f"Loading data from {DATA_DIR}")
    data = InteractionData(data_dir=DATA_DIR)
    bucket_map = _bucket_map(data)
    adj = build_norm_adj(data.n_users, data.n_items, data.train_user_items, "cpu")

    per_seed = {}
    for seed in SEEDS:
        logger.info(f"R1 seed={seed} ...")
        triples = _gt_ranks_r1(seed, data, adj, device="cpu")
        per_seed[seed] = _bucket_metrics(triples, bucket_map)

    # Aggregate across seeds — same shape as run_cold_start_amazon.py
    seeds_done = sorted(per_seed.keys())
    summary_buckets = {}
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
        summary_buckets[bname] = agg

    # Merge into existing cold_start_5seeds.json
    existing = json.loads(RESULTS_JSON.read_text())
    existing["configs"]["R1"] = {"seeds": seeds_done, "buckets": summary_buckets}
    RESULTS_JSON.write_text(json.dumps(existing, indent=2, default=str))
    logger.info(f"Merged R1 into {RESULTS_JSON.relative_to(REPO)}")

    # Pretty print R1 row
    print("\n=== R1 (RLMRec-gene) bucket-stratified on Amazon-Books-2018 ===")
    print(f"{'Bucket':<7} {'rank_med':>10} {'R@10':>8} {'R@100':>9} "
          f"{'R@1000':>10} {'NDCG@10':>9} {'MRR':>8} {'n_users':>8}")
    for bname in BUCKETS:
        b = summary_buckets[bname]
        print(f"{bname:<7} {b['rank_median']['mean']:>10.0f} "
              f"{b['Recall@10']['mean']:>8.4f} {b['Recall@100']['mean']:>9.4f} "
              f"{b['Recall@1000']['mean']:>10.4f} {b['NDCG@10']['mean']:>9.4f} "
              f"{b['MRR_full']['mean']:>8.4f} {b['n_users']:>8d}")


if __name__ == "__main__":
    main()
