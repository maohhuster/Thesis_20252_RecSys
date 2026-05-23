#!/usr/bin/env python3
"""Evaluate R1 (RLMRec-gene) checkpoints on subsampled-ML-20M (same-domain density control).

Faithful sibling of eval_ml20m_sub163_r1plus.py — identical RLMRec
LightGCN-gene inference (symmetric-normalized bipartite adjacency, sum
over `layer_num` layers) + our `evaluate.compute_metrics` full-ranking,
for direct comparability with M1/M4/M7. R1-gene and R1-plus share the
inference path; only the checkpoint source differs.

R1-gene checkpoints live in RLMRec's NATIVE checkpoint dir (.pth),
NOT under code/benchmark/checkpoints_ml20m_sub163/ (which holds no r1/).

Usage:
    python3 scripts/eval_ml20m_sub163_r1.py --device mps
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

DATA_DIR = BENCH / "data" / "processed_ml20m_sub163"  # subsampled-ML-20M data
# RLMRec native checkpoint dir (source of truth for R1-gene); .pth files
CK_DIR = BENCH / "external" / "RLMRec" / "encoder" / "checkpoint" / "lightgcn_gene"
CK_NAME = "lightgcn_gene-ml20m_sub163_ours-{seed}.pth"
RESULTS_DIR = BENCH / "results_ml20m_sub163"
SEEDS = [42, 123, 456, 789, 2026]
TOP_K = [10, 20, 50]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("eval_r1_sub163")


def build_norm_adj(n_users: int, n_items: int, train_user_items: dict, device: str):
    """RLMRec-style symmetric-normalized bipartite adjacency over (users + items)."""
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
    """RLMRec lightgcn_gene: stack, propagate, sum over layers (incl. layer 0)."""
    embeds = torch.cat([user_emb, item_emb], dim=0)
    embeds_list = [embeds]
    for _ in range(layer_num):
        embeds = torch.sparse.mm(adj, embeds_list[-1])
        embeds_list.append(embeds)
    final = sum(embeds_list)
    return final[: user_emb.shape[0]], final[user_emb.shape[0]:]


@torch.no_grad()
def evaluate_seed(seed: int, layer_num: int, device: str, data, adj):
    ck = CK_DIR / CK_NAME.format(seed=seed)
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
                    help="Matches the upstream lightgcn_gene.yml > model.ml20m_sub163_ours "
                         "(same-domain density control derived from ML-20M).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    logger.info(f"Device: {args.device} | layer_num: {args.layer_num} | seeds: {args.seeds}")
    data = InteractionData(data_dir=DATA_DIR)
    logger.info(f"n_users={data.n_users}, n_items={data.n_items}")
    adj = build_norm_adj(data.n_users, data.n_items, data.train_user_items, args.device)
    logger.info(f"adj nnz={adj._nnz()}")

    per_seed = {}
    for seed in args.seeds:
        m = evaluate_seed(seed, args.layer_num, args.device, data, adj)
        per_seed[seed] = m
        logger.info(f"seed={seed}  NDCG@10={m['NDCG@10']:.4f}  "
                    f"Recall@10={m['Recall@10']:.4f}  MRR={m['MRR']:.4f}")

    keys = ("NDCG@10", "Recall@10", "MRR")
    summary = {}
    if len(args.seeds) >= 2:
        for k in keys:
            vals = [per_seed[s][k] for s in args.seeds]
            summary[k] = {"mean": float(np.mean(vals)),
                          "std": float(np.std(vals, ddof=1)),
                          "values": [float(v) for v in vals]}
    else:
        for k in keys:
            summary[k] = {"single_seed": per_seed[args.seeds[0]][k]}

    print(f"\n=== R1 (RLMRec-gene) on subsampled-ML-20M (seeds {args.seeds}) ===")
    for k in keys:
        s = summary[k]
        if "mean" in s:
            print(f"  {k:10s} {s['mean']:.4f} ± {s['std']:.4f}")
        else:
            print(f"  {k:10s} {s['single_seed']:.4f}  (single seed)")

    marker_dir = RESULTS_DIR / "r1" / "ml20m_sub163_ours"
    for seed, m in per_seed.items():
        marker_path = marker_dir / f"seed-{seed}-marker.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            marker["test_metrics"] = m
            marker["layer_num"] = args.layer_num
            marker_path.write_text(json.dumps(marker, indent=2))
            logger.info(f"  → metrics merged into {marker_path.relative_to(REPO)}")

    out_path = RESULTS_DIR / "r1_ml20m_sub163_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"layer_num": args.layer_num, "seeds_evaluated": args.seeds,
         "per_seed": per_seed, "summary": summary,
         "_provenance": ("Re-evaluation of RLMRec native checkpoints "
                         "external/RLMRec/encoder/checkpoint/lightgcn_gene/"
                         "lightgcn_gene-ml20m_sub163_ours-<seed>.pth (read-only). "
                         "Same-domain density control; full-ranking compute_metrics.")},
        indent=2, default=str))
    logger.info(f"Saved {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
