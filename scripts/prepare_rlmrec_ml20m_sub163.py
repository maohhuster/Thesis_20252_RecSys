#!/usr/bin/env python3
"""Prepare subsampled ML-20M (~225 int/item) data + ML-20M LLM embeddings in
RLMRec's expected format.

Mirrors `scripts/prepare_rlmrec_ml1m.py`. Embeddings come from the parent ML-20M
release (`code/embedding_generator/output/`); subsampled item IDs map back to
ML-20M original movieIDs via `subsampled_to_ml20m_movieId.json`.

Usage:
    python3 scripts/prepare_rlmrec_ml20m_sub163.py
    python3 scripts/prepare_rlmrec_ml20m_sub163.py --patch-yaml
"""
from __future__ import annotations
import argparse, json, logging, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "code" / "benchmark" / "data" / "processed_ml20m_sub163"
EMB_DIR  = REPO / "code" / "embedding_generator" / "output"  # parent ML-20M release
RLMREC_ROOT = REPO / "code" / "benchmark" / "external" / "RLMRec"
OUT_DIR  = RLMREC_ROOT / "data" / "ml20m_sub163_ours"
YAML_PATH = RLMREC_ROOT / "encoder" / "config" / "modelconf" / "lightgcn_gene.yml"
DATA_HANDLER_PATH = RLMREC_ROOT / "encoder" / "data_utils" / "data_handler_general_cf.py"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_sparse(csv: Path, n_users: int, n_items: int) -> csr_matrix:
    df = pd.read_csv(csv)
    return csr_matrix(
        (np.ones(len(df), dtype=np.float32), (df["userId"].values, df["movieId"].values)),
        shape=(n_users, n_items),
    )


def align_item_emb(emb: np.ndarray, ml20m_id_index: list, sub_to_ml20m: dict, n_items: int) -> np.ndarray:
    """Map ML-20M genome embedding rows to subsampled-contiguous item IDs.
    sub_to_ml20m: {str(sub_contiguous_id): ml20m_orig_movieId}."""
    ml20m_to_emb_row = {int(mid): row for row, mid in enumerate(ml20m_id_index)}
    aligned = np.zeros((n_items, emb.shape[1]), dtype=np.float32)
    matched = 0
    for sub_id_str, ml20m_orig in sub_to_ml20m.items():
        row = ml20m_to_emb_row.get(int(ml20m_orig))
        if row is not None:
            aligned[int(sub_id_str)] = emb[row]
            matched += 1
    logger.info(f"  aligned {matched}/{n_items} items to ML-20M embedding rows")
    return aligned


def build_user_emb(train_csv: Path, item_emb: np.ndarray, n_users: int) -> np.ndarray:
    df = pd.read_csv(train_csv)
    out = np.zeros((n_users, item_emb.shape[1]), dtype=np.float32)
    for uid, group in df.groupby("userId"):
        out[uid] = item_emb[group["movieId"].values].mean(axis=0)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (out / norms).astype(np.float32)


# Use ml20m hparams as starting point — same domain (movies), lower density
# than full ML-20M. Predicted by the framework: at ~225 int/item the R1-over-M7
# gain should sit between ML-20M's −2.3% and ML-1M's +17%.
ML20M_SUB163_OURS_YAML_BLOCK = """\
  # for ml20m_sub163_ours (subsampled ML-20M ~225 int/item — same-domain density control)
  ml20m_sub163_ours:
    layer_num: 3
    reg_weight: 1.0e-7
    mask_ratio: 0.1
    recon_weight: 0.1
    re_temperature: 0.2
"""


def maybe_patch_yaml(yaml_path: Path) -> None:
    text = yaml_path.read_text()
    if "ml20m_sub163_ours:" in text:
        logger.info("  YAML already has ml20m_sub163_ours block — skipping")
        return
    if not text.endswith("\n"):
        text += "\n"
    text += ML20M_SUB163_OURS_YAML_BLOCK
    yaml_path.write_text(text)
    logger.info(f"  Patched {yaml_path.relative_to(REPO)} (+ml20m_sub163_ours block)")


def maybe_patch_data_handler(handler_path: Path) -> None:
    text = handler_path.read_text()
    if "'ml20m_sub163_ours'" in text or '"ml20m_sub163_ours"' in text:
        logger.info("  data handler already supports ml20m_sub163_ours — skipping")
        return
    candidates = [
        "elif configs['data']['name'] == 'ml1m_ours':\n            predir = './data/ml1m_ours/'",
        "elif configs['data']['name'] == 'amazon_ours':\n            predir = './data/amazon_ours/'",
        "elif configs['data']['name'] == 'ml20m':\n            predir = './data/ml20m/'",
    ]
    for needle in candidates:
        if needle in text:
            text = text.replace(needle, needle + (
                "\n        elif configs['data']['name'] == 'ml20m_sub163_ours':\n"
                "            predir = './data/ml20m_sub163_ours/'"
            ), 1)
            handler_path.write_text(text)
            logger.info(f"  Patched {handler_path.name} (+ml20m_sub163_ours branch)")
            return
    logger.warning(f"  could not find anchor in {handler_path.name}; data handler patch SKIPPED")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-yaml", action="store_true")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = json.loads((DATA_DIR / "stats.json").read_text())
    n_users, n_items = stats["n_users"], stats["n_items"]
    logger.info(f"n_users={n_users}, n_items={n_items}")

    logger.info("Building sparse matrices...")
    trn = build_sparse(DATA_DIR / "train.csv", n_users, n_items)
    val = build_sparse(DATA_DIR / "val.csv", n_users, n_items)
    tst = build_sparse(DATA_DIR / "test.csv", n_users, n_items)
    logger.info(f"  train.nnz={trn.nnz}, val.nnz={val.nnz}, test.nnz={tst.nnz}")
    pickle.dump(trn, (OUT_DIR / "trn_mat.pkl").open("wb"))
    pickle.dump(val, (OUT_DIR / "val_mat.pkl").open("wb"))
    pickle.dump(tst, (OUT_DIR / "tst_mat.pkl").open("wb"))

    logger.info("Aligning item embeddings (ML-20M release)...")
    profile_emb = np.load(EMB_DIR / "profile_embeddings.npy")
    ml20m_ids = json.loads((EMB_DIR / "movie_id_index.json").read_text())
    sub_to_ml20m = json.loads((DATA_DIR / "subsampled_to_ml20m_movieId.json").read_text())
    item_emb = align_item_emb(profile_emb, ml20m_ids, sub_to_ml20m, n_items)
    pickle.dump(item_emb, (OUT_DIR / "itm_emb_np.pkl").open("wb"))

    logger.info("Building user embeddings (mean-pool, unit-normalized)...")
    usr_emb = build_user_emb(DATA_DIR / "train.csv", item_emb, n_users)
    pickle.dump(usr_emb, (OUT_DIR / "usr_emb_np.pkl").open("wb"))

    pickle.dump([{"profile": "", "reasoning": ""} for _ in range(n_items)],
                (OUT_DIR / "itm_prf.pkl").open("wb"))
    pickle.dump([{"profile": "", "reasoning": ""} for _ in range(n_users)],
                (OUT_DIR / "usr_prf.pkl").open("wb"))

    if args.patch_yaml:
        maybe_patch_yaml(YAML_PATH)
        maybe_patch_data_handler(DATA_HANDLER_PATH)

    logger.info(f"Done. Artifacts in {OUT_DIR.relative_to(REPO)}/")
    for fp in sorted(OUT_DIR.iterdir()):
        logger.info(f"  {fp.name}  ({fp.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
