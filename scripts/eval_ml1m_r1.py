#!/usr/bin/env python3
"""Re-evaluate R1 (RLMRec-gene) on ML-1M with our `compute_metrics`.

The numbers RLMRec's training-time evaluator reports use a different MRR
definition (sum of reciprocal ranks of all top-K hits) than our
`compute_metrics` (reciprocal rank of the FIRST hit only). NDCG@10 and
Recall@10 use the same formula in both, but MRR diverges substantially
when users have multiple test items.

This script loads R1's RLMRec checkpoints (.pth) and runs OUR full-ranking
evaluator so the resulting NDCG@10 / Recall@10 / MRR are directly comparable
to M1 / M4 / M7 / R2 on ML-1M (all of which use compute_metrics).

Mirror of `scripts/eval_amazon_r1.py`, paths swapped to ML-1M.

Usage on Colab:
    cd llm-movielens
    python3 scripts/eval_ml1m_r1.py --device cuda
    # or with subset of seeds:
    python3 scripts/eval_ml1m_r1.py --device cuda --seeds 42 123
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
from data.dataset import InteractionData  # noqa: E402
from evaluate import compute_metrics  # noqa: E402

DATA_DIR = BENCH / "data" / "processed_ml1m"
CK_DIR = BENCH / "checkpoints_ml1m" / "r1" / "bge-large-en-v1.5"
RESULTS_DIR = BENCH / "results_ml1m"
SEEDS_DEFAULT = [42, 123, 456, 789, 2026]
TOP_K = [10, 20, 50]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("eval_r1_ml1m")


def build_norm_adj(n_users: int, n_items: int, train_user_items: dict, device: str):
    """RLMRec-style symmetric-normalized bipartite adjacency."""
    rows, cols = [], []
    for u, items in train_user_items.items():
        for it in items:
            rows.append(u); cols.append(it)
    inter = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                       shape=(n_users, n_items))
    a = csr_matrix((n_users, n_users))
    b = csr_matrix((n_items, n_items))
    bipart = sp.vstack([sp.hstack([a, inter]), sp.hstack([inter.T, b])]).tocsr()
    bipart = (bipart != 0).astype(np.float32)
    deg = np.array(bipart.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg))
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = sp.diags(d_inv_sqrt)
    norm = D @ bipart @ D
    coo = norm.tocoo()
    idx = torch.LongTensor(np.vstack([coo.row, coo.col]))
    val = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(idx, val, coo.shape, device=device).coalesce()


def propagate(adj, user_emb, item_emb, layer_num: int):
    """RLMRec lightgcn_gene propagation: stack, propagate, sum over layers."""
    embeds = torch.cat([user_emb, item_emb], dim=0)
    embeds_list = [embeds]
    for _ in range(layer_num):
        embeds = torch.sparse.mm(adj, embeds_list[-1])
        embeds_list.append(embeds)
    final = sum(embeds_list)
    return final[: user_emb.shape[0]], final[user_emb.shape[0]:]


@torch.no_grad()
def evaluate_seed(seed: int, layer_num: int, device: str, data, adj):
    ck = CK_DIR / f"seed-{seed}" / "best_model.pt"
    if not ck.exists():
        # Fallback: maybe checkpoints are in checkpoints_amazon-style layout
        alt = BENCH / "checkpoints_ml1m" / "r1" / "bge-large-en-v1.5" / f"seed-{seed}" / "best_model.pt"
        if alt.exists():
            ck = alt
        else:
            logger.warning(f"  no checkpoint at {ck} or {alt}")
            return None
    sd = torch.load(ck, map_location=device, weights_only=False)
    user_emb = sd["user_embeds"].to(device)
    item_emb = sd["item_embeds"].to(device)
    user_final, item_final = propagate(adj, user_emb, item_emb, layer_num)

    eval_users = sorted(data.test_user_items.keys())
    all_metrics = defaultdict(list)
    bs = 256
    for start in range(0, len(eval_users), bs):
        bu = eval_users[start: start + bs]
        ur = user_final[torch.LongTensor(bu).to(device)]
        scores = (ur @ item_final.T).cpu().numpy()
        for i, uid in enumerate(bu):
            gt = data.test_user_items.get(uid, set())
            if not gt:
                continue
            train_items = data.train_user_items.get(uid, set())
            m = compute_metrics(scores[i], gt, train_items, data.n_items, TOP_K)
            for k, v in m.items():
                all_metrics[k].append(v)
    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-num", type=int, default=3,
                    help="Matches the upstream `ml20m` block we copied into "
                         "the ml1m_ours: YAML block (layer_num=3).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    args = ap.parse_args()

    logger.info(f"Device: {args.device} | layer_num: {args.layer_num}")
    data = InteractionData(data_dir=DATA_DIR)
    logger.info(f"n_users={data.n_users}, n_items={data.n_items}")
    adj = build_norm_adj(data.n_users, data.n_items, data.train_user_items, args.device)
    logger.info(f"adj nnz={adj._nnz()}")

    per_seed = {}
    for seed in args.seeds:
        logger.info(f"R1 seed={seed} ...")
        m = evaluate_seed(seed, args.layer_num, args.device, data, adj)
        if m is None:
            continue
        per_seed[seed] = m
        logger.info(f"  NDCG@10={m['NDCG@10']:.4f}  "
                    f"Recall@10={m['Recall@10']:.4f}  MRR={m['MRR']:.4f}")

    if not per_seed:
        logger.error("No checkpoints found. Place under "
                     "code/benchmark/checkpoints_ml1m/r1/ml1m_ours/seed-N/best_model.pt")
        return

    # Aggregate
    keys = ("NDCG@10", "Recall@10", "MRR")
    summary = {}
    for k in keys:
        vals = [per_seed[s][k] for s in per_seed]
        summary[k] = {"mean": float(np.mean(vals)),
                      "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                      "values": [float(v) for v in vals]}

    print("\n=== R1 (RLMRec-gene) on ML-1M, our compute_metrics ===")
    for k in keys:
        s = summary[k]
        print(f"  {k:10s} {s['mean']:.4f} ± {s['std']:.4f}")

    out_path = RESULTS_DIR / "r1_ml1m_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"layer_num": args.layer_num, "evaluator": "compute_metrics",
         "per_seed": per_seed, "summary": summary},
        indent=2, default=str))
    logger.info(f"Saved {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
