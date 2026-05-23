#!/usr/bin/env python3
"""Prepare ML-1M data + our LLM embeddings in RLMRec's expected format.

Mirrors `scripts/prepare_rlmrec_amazon.py`, swapping paths to ML-1M.

Reads:
    code/benchmark/data/processed_ml1m/{train,val,test}.csv
    code/benchmark/data/processed_ml1m/{stats,item_map}.json
    code/embedding_generator/output_ml1m/bge-large-en-v1.5/profile_embeddings.npy
    code/embedding_generator/output_ml1m/bge-large-en-v1.5/movie_id_index.json

Writes:
    code/benchmark/external/RLMRec/data/ml1m_ours/
        trn_mat.pkl, val_mat.pkl, tst_mat.pkl   (csr sparse interaction matrices)
        itm_emb_np.pkl                          (n_items, 1024)  ← our LLM features
        usr_emb_np.pkl                          (n_users, 1024)  ← mean of liked items
        itm_prf.pkl, usr_prf.pkl                (dummy stubs)

Hyperparameter block appended to lightgcn_gene.yml (--patch-yaml):
    Copies the upstream `ml20m` block — same domain (movies), so the
    authors' density-driven adjustments still apply at first approximation;
    the cross-density framework predicts that R1's gain over M7 lies between
    ML-20M's −2.3% and Amazon's +29.5%.

Also patches `data_handler_general_cf.py` to register `ml1m_ours`.

Usage:
    python scripts/prepare_rlmrec_ml1m.py
    python scripts/prepare_rlmrec_ml1m.py --patch-yaml
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
DATA_DIR = REPO / "code" / "benchmark" / "data" / "processed_ml1m"
EMB_DIR = REPO / "code" / "embedding_generator" / "output_ml1m" / "bge-large-en-v1.5"
RLMREC_ROOT = REPO / "code" / "benchmark" / "external" / "RLMRec"
OUT_DIR = RLMREC_ROOT / "data" / "ml1m_ours"
YAML_PATH = RLMREC_ROOT / "encoder" / "config" / "modelconf" / "lightgcn_gene.yml"
DATA_HANDLER_PATH = (
    RLMREC_ROOT / "encoder" / "data_utils" / "data_handler_general_cf.py"
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prep_rlmrec_ml1m")


def build_sparse(csv_path: Path, n_users: int, n_items: int) -> csr_matrix:
    df = pd.read_csv(csv_path)
    return csr_matrix(
        (np.ones(len(df), dtype=np.float32),
         (df["userId"].values, df["movieId"].values)),
        shape=(n_users, n_items),
    )


def align_item_emb(profile_emb: np.ndarray, ids: list, n_items: int) -> np.ndarray:
    """Map (n_present, dim) profile rows to (n_items, dim) zero-padded.
    For ML-1M our embeddings already align row-i to bench item id i (built
    that way in build_ml1m_embeddings.py), so this is mostly a copy."""
    aligned = np.zeros((n_items, profile_emb.shape[1]), dtype=np.float32)
    for row, iid in enumerate(ids):
        aligned[int(iid)] = profile_emb[row]
    return aligned


def build_user_emb(train_csv: Path, item_emb: np.ndarray, n_users: int) -> np.ndarray:
    df = pd.read_csv(train_csv)
    dim = item_emb.shape[1]
    out = np.zeros((n_users, dim), dtype=np.float32)
    for uid, group in df.groupby("userId"):
        out[uid] = item_emb[group["movieId"].values].mean(axis=0)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (out / norms).astype(np.float32)


# Upstream `ml20m` block (the local-edit ml20m hparams from the existing YAML)
# applied verbatim — same domain (movies), the density-driven adjustments
# transfer reasonably as a starting point.
ML1M_OURS_YAML_BLOCK = """\
  # for ml1m_ours (ML-1M with our LLM embeddings, third density datapoint)
  # — values copied from `ml20m:` block (same domain, similar density profile)
  ml1m_ours:
    layer_num: 3
    reg_weight: 1.0e-7
    mask_ratio: 0.1
    recon_weight: 0.1
    re_temperature: 0.2
"""


def maybe_patch_yaml(yaml_path: Path) -> None:
    text = yaml_path.read_text()
    if "ml1m_ours:" in text:
        logger.info("  YAML already has ml1m_ours block — skipping")
        return
    if not text.endswith("\n"):
        text += "\n"
    text += ML1M_OURS_YAML_BLOCK
    yaml_path.write_text(text)
    logger.info(f"  Patched {yaml_path.relative_to(REPO)} (+ml1m_ours block)")


def maybe_patch_data_handler(handler_path: Path) -> None:
    """Add ml1m_ours branch to DataHandlerGeneralCF.__init__ dispatch."""
    text = handler_path.read_text()
    if "'ml1m_ours'" in text or '"ml1m_ours"' in text:
        logger.info("  data handler already supports ml1m_ours — skipping")
        return
    # Anchor on either the ml20m branch or the amazon_ours branch (whichever
    # exists last in the dispatch chain we've previously installed).
    candidates = [
        "elif configs['data']['name'] == 'amazon_ours':\n            predir = './data/amazon_ours/'",
        "elif configs['data']['name'] == 'ml20m':\n            predir = './data/ml20m/'",
    ]
    for needle in candidates:
        if needle in text:
            addition = needle + (
                "\n        elif configs['data']['name'] == 'ml1m_ours':\n"
                "            predir = './data/ml1m_ours/'"
            )
            text = text.replace(needle, addition, 1)
            handler_path.write_text(text)
            try:
                rel = handler_path.relative_to(REPO)
            except ValueError:
                rel = handler_path
            logger.info(f"  Patched {rel} (+ml1m_ours branch)")
            return
    logger.warning(f"  could not find anchor in {handler_path.name}; data handler "
                   f"patch SKIPPED — run will fail with NotImplementedError.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-yaml", action="store_true",
                   help="Append ml1m_ours: block to lightgcn_gene.yml AND "
                        "register ml1m_ours in data_handler_general_cf.py")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = json.loads((DATA_DIR / "stats.json").read_text())
    n_users = stats["n_users"]
    n_items = stats["n_items"]
    logger.info(f"n_users={n_users}, n_items={n_items}")

    logger.info("Building sparse matrices...")
    trn = build_sparse(DATA_DIR / "train.csv", n_users, n_items)
    val = build_sparse(DATA_DIR / "val.csv", n_users, n_items)
    tst = build_sparse(DATA_DIR / "test.csv", n_users, n_items)
    logger.info(f"  train.nnz={trn.nnz}, val.nnz={val.nnz}, test.nnz={tst.nnz}")
    pickle.dump(trn, (OUT_DIR / "trn_mat.pkl").open("wb"))
    pickle.dump(val, (OUT_DIR / "val_mat.pkl").open("wb"))
    pickle.dump(tst, (OUT_DIR / "tst_mat.pkl").open("wb"))

    logger.info("Aligning item embeddings...")
    profile_emb = np.load(EMB_DIR / "profile_embeddings.npy")
    ids = json.loads((EMB_DIR / "movie_id_index.json").read_text())
    item_emb = align_item_emb(profile_emb, ids, n_items)
    n_zero = int((np.linalg.norm(item_emb, axis=1) == 0).sum())
    logger.info(f"  item_emb shape={item_emb.shape}, zero-rows={n_zero}")
    pickle.dump(item_emb, (OUT_DIR / "itm_emb_np.pkl").open("wb"))

    logger.info("Building user embeddings (mean-pool, unit-normalized)...")
    usr_emb = build_user_emb(DATA_DIR / "train.csv", item_emb, n_users)
    pickle.dump(usr_emb, (OUT_DIR / "usr_emb_np.pkl").open("wb"))

    logger.info("Writing dummy profile pickles...")
    pickle.dump([{"profile": "", "reasoning": ""} for _ in range(n_items)],
                (OUT_DIR / "itm_prf.pkl").open("wb"))
    pickle.dump([{"profile": "", "reasoning": ""} for _ in range(n_users)],
                (OUT_DIR / "usr_prf.pkl").open("wb"))

    if args.patch_yaml:
        maybe_patch_yaml(YAML_PATH)
        maybe_patch_data_handler(DATA_HANDLER_PATH)
    else:
        logger.info("(Use --patch-yaml to register ml1m_ours in YAML + data handler)")

    logger.info(f"Done. Artifacts in {OUT_DIR.relative_to(REPO)}/")
    for fp in sorted(OUT_DIR.iterdir()):
        logger.info(f"  {fp.name}  ({fp.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
