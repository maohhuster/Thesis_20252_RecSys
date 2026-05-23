#!/usr/bin/env python3
"""Run pmixer's exact SASRec model on Amazon-Books-2018 with FULL-RANKING eval.

This script imports `SASRec` and the FFN block from
`code/benchmark/external/sasrec_pmixer/python/model.py` unchanged, uses
pmixer's published Amazon hyperparameters (Result_Norm.md):

    embed_dim    = 50
    num_blocks   = 2
    num_heads    = 1
    dropout      = 0.5      ← Amazon default; ML-1M would use 0.2
    maxlen       = 50       ← Amazon default; ML-1M would use 200
    lr           = 1e-3
    batch_size   = 128
    num_epochs   = up to 1000
    l2_emb       = 0.0
    optimizer    = Adam, betas=(0.9, 0.98)   (matches Kang & McAuley original paper)
    norm_first   = False    ← pmixer default = Post-LN (original SASRec design)

But the **evaluator** uses our `evaluate.compute_metrics` for full-ranking
NDCG@10 / Recall@10 / MRR over all items, on our temporal val/test splits —
matching the protocol used by every other config in the paper (M1/M4/M7/R1/R2).
This gives architecturally-faithful pmixer SASRec numbers that are directly
comparable with our other Tier-1/2/3 results.

Per-position full-sequence BCE loss exactly as pmixer's main.py.

Usage:
    python3 scripts/run_amazon_sasrec_pmixer.py --seed 42 --device mps
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "code" / "benchmark"
PMIXER = BENCH / "external" / "sasrec_pmixer" / "python"
DATA_DIR = BENCH / "data" / "processed_amazon"
RESULTS_DIR = BENCH / "results_amazon"
CHECKPOINT_DIR = BENCH / "checkpoints_amazon"

# Add pmixer's python dir to import path so `from model import SASRec` works
sys.path.insert(0, str(PMIXER))
sys.path.insert(0, str(BENCH))
from model import SASRec  # noqa: E402   ← pmixer's exact code, no edits
from evaluate import compute_metrics  # noqa: E402   ← our full-ranking metric helper

# pmixer's published Amazon hyperparameters (Result_Norm.md)
HPARAMS = SimpleNamespace(
    hidden_units=50,
    num_blocks=2,
    num_heads=1,
    dropout_rate=0.5,
    maxlen=50,
    lr=1e-3,
    batch_size=128,
    num_epochs=1000,
    l2_emb=0.0,
    norm_first=False,        # pmixer default; original SASRec Post-LN
)
TOP_K = [10, 20, 50]
EVAL_EVERY = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sasrec_pmixer")


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING — use our temporal splits, not pmixer's leave-one-out
# ─────────────────────────────────────────────────────────────────────────────

def load_amazon_data():
    """Load processed_amazon train/val/test as user-chronological histories
    in pmixer's 1-indexed item space.

    Our IDs are 0-indexed in CSVs; pmixer expects 1..n_items (id 0 = padding).
    We shift by +1 throughout.
    """
    def _load(name):
        df = pd.read_csv(DATA_DIR / f"{name}.csv").sort_values(["userId", "timestamp"])
        return df

    train = _load("train")
    val = _load("val")
    test = _load("test")

    n_users = int(max(train["userId"].max(), val["userId"].max(), test["userId"].max())) + 1
    n_items = int(max(train["movieId"].max(), val["movieId"].max(), test["movieId"].max())) + 1

    # Per-user training history (1-indexed items)
    user_train: dict[int, list[int]] = defaultdict(list)
    for u, i in zip(train["userId"].values, train["movieId"].values):
        user_train[int(u) + 1].append(int(i) + 1)

    # Per-user val / test ground-truth sets (1-indexed)
    val_gt: dict[int, set] = defaultdict(set)
    for u, i in zip(val["userId"].values, val["movieId"].values):
        val_gt[int(u) + 1].add(int(i) + 1)
    test_gt: dict[int, set] = defaultdict(set)
    for u, i in zip(test["userId"].values, test["movieId"].values):
        test_gt[int(u) + 1].add(int(i) + 1)

    return user_train, val_gt, test_gt, n_users, n_items


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLER — same logic as pmixer.utils.sample_function, no multiprocessing
# ─────────────────────────────────────────────────────────────────────────────

def random_neq(low, high, s):
    t = np.random.randint(low, high)
    while t in s:
        t = np.random.randint(low, high)
    return t


def sample_one(uid, user_train, n_items, maxlen):
    """Build one training example (seq, pos, neg) for a user."""
    seq = np.zeros(maxlen, dtype=np.int32)
    pos = np.zeros(maxlen, dtype=np.int32)
    neg = np.zeros(maxlen, dtype=np.int32)
    history = user_train[uid]
    nxt = history[-1]
    idx = maxlen - 1
    ts = set(history)
    for i in reversed(history[:-1]):
        seq[idx] = i
        pos[idx] = nxt
        neg[idx] = random_neq(1, n_items + 1, ts)
        nxt = i
        idx -= 1
        if idx == -1:
            break
    return seq, pos, neg


def make_batches(user_train, n_items, maxlen, batch_size, seed_offset=0):
    """Generator yielding (u, seq, pos, neg) numpy arrays — one epoch."""
    valid_uids = [u for u, h in user_train.items() if len(h) >= 2]
    rng = np.random.default_rng(seed_offset)
    rng.shuffle(valid_uids)
    for start in range(0, len(valid_uids) - batch_size + 1, batch_size):
        batch_uids = valid_uids[start:start + batch_size]
        seqs, poses, negs = [], [], []
        for u in batch_uids:
            s, p, n = sample_one(u, user_train, n_items, maxlen)
            seqs.append(s); poses.append(p); negs.append(n)
        yield (np.asarray(batch_uids, dtype=np.int32),
               np.asarray(seqs), np.asarray(poses), np.asarray(negs))


# ─────────────────────────────────────────────────────────────────────────────
# FULL-RANKING EVALUATOR — uses our compute_metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def full_ranking_eval(model, user_train, eval_gt, n_items, maxlen, device,
                      eval_users_subset=None):
    """Score every (user, item) pair, mask train items, compute NDCG/Recall/MRR.

    Returns dict of mean metrics over users with ≥1 GT item in `eval_gt`.
    """
    model.eval()
    metrics = defaultdict(list)
    eval_uids = sorted(eval_gt.keys())
    if eval_users_subset is not None:
        eval_uids = eval_uids[:eval_users_subset]

    bs = 256
    # Precompute item indices once (1..n_items, pmixer's 1-indexed convention)
    all_items = np.arange(1, n_items + 1, dtype=np.int64)

    for start in range(0, len(eval_uids), bs):
        bu = eval_uids[start:start + bs]
        # Build left-padded sequences from train history
        seqs_np = np.zeros((len(bu), maxlen), dtype=np.int32)
        for j, u in enumerate(bu):
            hist = user_train.get(u, [])
            tail = hist[-maxlen:]
            seqs_np[j, maxlen - len(tail):] = tail

        # pmixer's predict() expects (user_ids, log_seqs, item_indices) and
        # returns (batch, n_items) raw logits
        logits = model.predict(np.asarray(bu),
                               seqs_np,
                               all_items).cpu().numpy()  # (batch, n_items)

        for j, u in enumerate(bu):
            gt = eval_gt[u]                            # 1-indexed item ids
            if not gt:
                continue
            # Convert to 0-indexed bench item ids for compute_metrics
            scores = logits[j]                         # (n_items,) indexed by 0..n_items-1 ↔ pmixer item 1..n_items
            train_items_0idx = {i - 1 for i in user_train.get(u, [])}
            gt_0idx = {i - 1 for i in gt}
            m = compute_metrics(scores, gt_0idx, train_items_0idx, n_items, TOP_K)
            for k, v in m.items():
                metrics[k].append(v)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = pick_device(args.device)
    logger.info(f"Device: {device}")
    set_seed(args.seed)

    user_train, val_gt, test_gt, n_users, n_items = load_amazon_data()
    avg_len = np.mean([len(h) for h in user_train.values()])
    logger.info(f"Loaded: n_users={n_users}, n_items={n_items}, "
                f"train_users={len(user_train)}, avg_seq={avg_len:.1f}")
    logger.info(f"  val users with GT: {len(val_gt)}, test users with GT: {len(test_gt)}")

    # Build pmixer SASRec with their args namespace
    hp = SimpleNamespace(**vars(HPARAMS))
    hp.device = device
    if args.epochs is not None:
        hp.num_epochs = args.epochs
    model = SASRec(n_users, n_items, hp).to(device)

    # Match pmixer's init exactly (xavier_normal_)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except ValueError:
            pass
    # Zero out the padding row of pos_emb and item_emb (matches pmixer main.py)
    with torch.no_grad():
        model.pos_emb.weight.data[0].zero_()
        model.item_emb.weight.data[0].zero_()

    # Optimizer with Kang & McAuley original-paper betas
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, betas=(0.9, 0.98))

    exp_name = f"sasrec_pmixer/bge-large-en-v1.5/seed-{args.seed}"
    ck_dir = CHECKPOINT_DIR / exp_name
    ck_dir.mkdir(parents=True, exist_ok=True)
    res_dir = RESULTS_DIR / exp_name
    res_dir.mkdir(parents=True, exist_ok=True)

    best_ndcg = 0.0
    best_metrics = {}
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, hp.num_epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()
        for u, seq, pos, neg in make_batches(user_train, n_items, hp.maxlen,
                                              hp.batch_size, seed_offset=args.seed * 1000 + epoch):
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_t = torch.from_numpy(pos).to(device)
            indices = (pos_t != 0)
            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)
            loss = F.binary_cross_entropy_with_logits(pos_logits[indices], pos_labels[indices])
            loss += F.binary_cross_entropy_with_logits(neg_logits[indices], neg_labels[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_time = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % EVAL_EVERY == 0 or epoch == hp.num_epochs:
            t1 = time.time()
            val_metrics = full_ranking_eval(model, user_train, val_gt, n_items,
                                             hp.maxlen, device)
            eval_time = time.time() - t1
            logger.info(f"epoch {epoch:3d}  loss={avg_loss:.4f}  "
                        f"NDCG@10={val_metrics.get('NDCG@10', 0):.4f}  "
                        f"R@10={val_metrics.get('Recall@10', 0):.4f}  "
                        f"MRR={val_metrics.get('MRR', 0):.4f}  "
                        f"({train_time:.0f}s train + {eval_time:.0f}s eval)")
            ndcg10 = val_metrics.get("NDCG@10", 0)
            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                best_metrics = val_metrics
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), ck_dir / "best_model.pt")
            else:
                patience_counter += 1
            history.append({"epoch": epoch, "loss": avg_loss, **val_metrics})
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (best: {best_epoch})")
                break
        else:
            logger.info(f"epoch {epoch:3d}  loss={avg_loss:.4f}  ({train_time:.0f}s)")

    # Final test eval with best checkpoint
    if best_epoch > 0:
        model.load_state_dict(torch.load(ck_dir / "best_model.pt", map_location=device))
    test_metrics = full_ranking_eval(model, user_train, test_gt, n_items,
                                      hp.maxlen, device)
    logger.info(f"=== Test ===  NDCG@10={test_metrics['NDCG@10']:.4f}  "
                f"R@10={test_metrics['Recall@10']:.4f}  MRR={test_metrics['MRR']:.4f}")

    out = {
        "experiment": exp_name,
        "config": {**vars(HPARAMS), "seed": args.seed,
                   "best_epoch": best_epoch,
                   "model_source": "pmixer/SASRec.pytorch (vendored, unmodified)",
                   "evaluator": "our full-ranking compute_metrics",
                   "split": "temporal (processed_amazon)"},
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
    }
    (res_dir / "results.json").write_text(json.dumps(out, indent=2, default=str))
    logger.info(f"Saved results to {res_dir / 'results.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override num_epochs (default: pmixer's 1000)")
    p.add_argument("--patience", type=int, default=20,
                   help="Eval-rounds without improvement before early stop")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
