#!/usr/bin/env python3
"""Prepare Amazon-Books data + our LLM embeddings in RLMRec's expected format.

Reads:
    code/benchmark/data/processed_amazon/{train,val,test}.csv
    code/benchmark/data/processed_amazon/{stats,item_map}.json
    code/embedding_generator/output_amazon/bge-large-en-v1.5/profile_embeddings.npy
    code/embedding_generator/output_amazon/bge-large-en-v1.5/movie_id_index.json

Writes:
    code/benchmark/external/RLMRec/data/amazon_ours/
        trn_mat.pkl, val_mat.pkl, tst_mat.pkl   (csr sparse interaction matrices)
        itm_emb_np.pkl                          (n_items=9332, 1024)  ← our LLM features
        usr_emb_np.pkl                          (n_users=11000, 1024) ← mean of liked items
        itm_prf.pkl, usr_prf.pkl                (dummy stubs required by RLMRec loader)

The 43 metadata-impoverished items receive zero-vector content features
(consistent with the cross-domain missing-items policy documented in
output_amazon/README.md and HUGGINGFACE_CARD.md §7).

Why a separate `amazon_ours` namespace (rather than overwriting RLMRec's
pristine `data/amazon/`):
    - Preserves RLMRec's published Amazon data for any future reference run
    - The new YAML block in `lightgcn_gene.yml` (added by --patch-yaml) will
      reuse the upstream `amazon:` hyperparameters since the dataset density
      profile is identical — the published RLMRec authors' Amazon hparams
      apply unchanged. This is the strongest possible position.

Usage:
    python scripts/prepare_rlmrec_amazon.py
    python scripts/prepare_rlmrec_amazon.py --patch-yaml   # also append amazon_ours: block
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "code" / "benchmark" / "data" / "processed_amazon"
EMB_DIR = REPO / "code" / "embedding_generator" / "output_amazon" / "bge-large-en-v1.5"
RLMREC_ROOT = REPO / "code" / "benchmark" / "external" / "RLMRec"
OUT_DIR = RLMREC_ROOT / "data" / "amazon_ours"
YAML_PATH = RLMREC_ROOT / "encoder" / "config" / "modelconf" / "lightgcn_gene.yml"
DATA_HANDLER_PATH = (
    RLMREC_ROOT / "encoder" / "data_utils" / "data_handler_general_cf.py"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("prep_rlmrec_amazon")


def build_sparse(csv_path: Path, n_users: int, n_items: int) -> csr_matrix:
    df = pd.read_csv(csv_path)
    return csr_matrix(
        (np.ones(len(df), dtype=np.float32),
         (df["userId"].values, df["movieId"].values)),
        shape=(n_users, n_items),
    )


def align_item_emb(profile_emb: np.ndarray, ids: list, n_items: int) -> np.ndarray:
    """Map (n_present, dim) profile rows to (n_items, dim) zero-padded matrix.

    `ids[i]` is the iid that owns row i of `profile_emb` (the 9,289 successful
    profiles). Missing iids stay as all-zero rows.
    """
    aligned = np.zeros((n_items, profile_emb.shape[1]), dtype=np.float32)
    for row, iid in enumerate(ids):
        aligned[int(iid)] = profile_emb[row]
    return aligned


def build_user_emb(train_csv: Path, item_emb: np.ndarray, n_users: int) -> np.ndarray:
    """User vector = unit-normalized mean of their liked-item embeddings.

    Mirrors the convention used by `prepare_rlmrec_data.py` for ML-20M, so
    the cross-domain comparison is methodologically identical.
    """
    df = pd.read_csv(train_csv)
    dim = item_emb.shape[1]
    out = np.zeros((n_users, dim), dtype=np.float32)
    for uid, group in df.groupby("userId"):
        out[uid] = item_emb[group["movieId"].values].mean(axis=0)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (out / norms).astype(np.float32)


# Hyperparameter block we will append to lightgcn_gene.yml.
# Values match upstream `amazon:` block exactly — these are the original
# RLMRec authors' published Amazon-Book hyperparameters, which apply
# unchanged here because the density profile is identical.
AMAZON_OURS_YAML_BLOCK = """\
  # for amazon_ours (cross-domain Amazon-Books with our LLM embeddings)
  # — values copied verbatim from upstream `amazon:` block (authors' published
  #   hyperparameters for the same dataset/density regime)
  amazon_ours:
    layer_num: 2
    reg_weight: 1.0e-7
    mask_ratio: 0.15
    recon_weight: 0.1
    re_temperature: 0.5
"""


def maybe_patch_yaml(yaml_path: Path) -> None:
    text = yaml_path.read_text()
    if "amazon_ours:" in text:
        logger.info(f"  YAML already has amazon_ours block — skipping patch")
        return
    if not text.endswith("\n"):
        text += "\n"
    text += AMAZON_OURS_YAML_BLOCK
    yaml_path.write_text(text)
    logger.info(f"  Patched {yaml_path.relative_to(REPO)} (+amazon_ours block)")


def maybe_patch_data_handler(handler_path: Path) -> None:
    """Add an `amazon_ours` branch to the dataset-name dispatch in
    DataHandlerGeneralCF.__init__. Without this, --dataset amazon_ours
    falls through to `raise NotImplementedError`."""
    text = handler_path.read_text()
    if "'amazon_ours'" in text or '"amazon_ours"' in text:
        logger.info(f"  data handler already supports amazon_ours — skipping patch")
        return
    needle = "elif configs['data']['name'] == 'ml20m':\n            predir = './data/ml20m/'"
    if needle not in text:
        logger.warning(
            f"  expected ml20m branch not found in {handler_path.name}; "
            f"data handler patch SKIPPED. Run will fail with NotImplementedError."
        )
        return
    addition = (
        "elif configs['data']['name'] == 'ml20m':\n"
        "            predir = './data/ml20m/'\n"
        "        elif configs['data']['name'] == 'amazon_ours':\n"
        "            predir = './data/amazon_ours/'"
    )
    text = text.replace(needle, addition, 1)
    handler_path.write_text(text)
    try:
        rel = handler_path.relative_to(REPO)
    except ValueError:
        rel = handler_path
    logger.info(f"  Patched {rel} (+amazon_ours branch)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-yaml", action="store_true",
                   help="Append `amazon_ours:` block to lightgcn_gene.yml.")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = json.loads((DATA_DIR / "stats.json").read_text())
    n_users = stats["n_users"]
    n_items = stats["n_items"]
    logger.info(f"n_users={n_users}, n_items={n_items}")

    logger.info("Building sparse interaction matrices...")
    trn = build_sparse(DATA_DIR / "train.csv", n_users, n_items)
    val = build_sparse(DATA_DIR / "val.csv", n_users, n_items)
    tst = build_sparse(DATA_DIR / "test.csv", n_users, n_items)
    logger.info(f"  train.nnz={trn.nnz}, val.nnz={val.nnz}, test.nnz={tst.nnz}")
    pickle.dump(trn, (OUT_DIR / "trn_mat.pkl").open("wb"))
    pickle.dump(val, (OUT_DIR / "val_mat.pkl").open("wb"))
    pickle.dump(tst, (OUT_DIR / "tst_mat.pkl").open("wb"))

    logger.info("Aligning item embeddings (zero-vector for 43 missing-metadata items)...")
    profile_emb = np.load(EMB_DIR / "profile_embeddings.npy")
    ids = json.loads((EMB_DIR / "movie_id_index.json").read_text())
    item_emb = align_item_emb(profile_emb, ids, n_items)
    n_zero = int((np.linalg.norm(item_emb, axis=1) == 0).sum())
    logger.info(f"  item_emb shape={item_emb.shape}, zero-rows={n_zero}")
    pickle.dump(item_emb, (OUT_DIR / "itm_emb_np.pkl").open("wb"))

    logger.info("Building user embeddings (mean-pool of liked items, unit-normalized)...")
    usr_emb = build_user_emb(DATA_DIR / "train.csv", item_emb, n_users)
    logger.info(f"  usr_emb shape={usr_emb.shape}")
    pickle.dump(usr_emb, (OUT_DIR / "usr_emb_np.pkl").open("wb"))

    logger.info("Writing dummy profile pickles (loader requires presence)...")
    pickle.dump([{"profile": "", "reasoning": ""} for _ in range(n_items)],
                (OUT_DIR / "itm_prf.pkl").open("wb"))
    pickle.dump([{"profile": "", "reasoning": ""} for _ in range(n_users)],
                (OUT_DIR / "usr_prf.pkl").open("wb"))

    if args.patch_yaml:
        maybe_patch_yaml(YAML_PATH)
        maybe_patch_data_handler(DATA_HANDLER_PATH)
    else:
        logger.info("(Use --patch-yaml to append amazon_ours: block to "
                    "encoder/config/modelconf/lightgcn_gene.yml AND register "
                    "amazon_ours in data_handler_general_cf.py)")

    logger.info(f"Done. Artifacts in {OUT_DIR.relative_to(REPO)}/")
    for fp in sorted(OUT_DIR.iterdir()):
        logger.info(f"  {fp.name}  ({fp.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
