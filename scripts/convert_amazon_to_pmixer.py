#!/usr/bin/env python3
"""Convert our Amazon-Books-2018 splits to pmixer/SASRec.pytorch's data format.

pmixer format: a single text file `data/{name}.txt` with `user_id item_id`
per line, sorted chronologically per user, ids 1-indexed.
pmixer's data_partition() then splits leave-one-out: last item per user = test,
second-last = val, rest = train.

We combine our train+val+test (which were temporally split), sort per user by
timestamp, and emit the canonical pmixer file. This restores per-user
chronological order so pmixer's leave-one-out + sampled-NDCG@10 protocol can
run faithfully.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "code" / "benchmark" / "data" / "processed_amazon"
OUT = REPO / "code" / "benchmark" / "external" / "sasrec_pmixer" / "python" / "data" / "amazon_books.txt"


def main():
    dfs = [pd.read_csv(DATA_DIR / f"{s}.csv") for s in ("train", "val", "test")]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["userId", "timestamp"], kind="mergesort")

    # 1-index ids (pmixer's data_partition expects 1-indexed; id 0 = padding)
    df["userId"] = df["userId"].astype(int) + 1
    df["movieId"] = df["movieId"].astype(int) + 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for u, i in zip(df["userId"].values, df["movieId"].values):
            f.write(f"{u} {i}\n")

    print(f"Wrote {len(df):,} interactions to {OUT.relative_to(REPO)}")
    print(f"  users: {df['userId'].nunique():,} (1..{df['userId'].max()})")
    print(f"  items: {df['movieId'].nunique():,} (1..{df['movieId'].max()})")
    print(f"  avg seq/user: {len(df) / df['userId'].nunique():.1f}")


if __name__ == "__main__":
    main()
