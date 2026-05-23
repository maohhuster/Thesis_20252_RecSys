# KAR Upstream Sanity Check

Vendor the official KAR repo and reproduce its published ML-1M-CTR result. This validates that our R2 (KAR-style MoE replacer) adaptation in `code/benchmark/models/kar.py` is methodologically faithful, not buggy.

## What this directory is for

The paper claims R2 adapts KAR's HEA module from CTR to BPR ranking. A reviewer can reasonably ask "is your adaptation correct, or did you misread the original?". This sanity check answers that question by running the **original** KAR repo on its **native** CTR domain and matching the published AUC.

## Upstream repo

- **URL**: https://github.com/YunjiaXi/Open-World-Knowledge-Augmented-Recommendation
- **Paper**: Xi et al. 2024, "Towards Open-World Recommendation with Knowledge Augmentation from LLMs" (arXiv:2306.10933)
- **Last verified**: 2026-04-23 (108 stars, active maintenance)
- **License**: Not specified upstream — treat as research-use-only.

## Reproduction target

KAR's published numbers on ML-1M (CTR, DIN backbone, from arXiv:2306.10933):

| Config | AUC | LogLoss |
|---|---|---|
| DIN baseline | 0.7878 | 0.5364 |
| DIN + KAR (HEA, augment=True) | **0.8143** | **0.5096** |
| Δ AUC | **+0.0265** | |

Match within ±0.005 AUC = sanity check passes.

## Setup (run on a GPU machine — Colab Pro / lab server)

```bash
# 1. Clone the upstream repo
cd /path/to/repo/code/benchmark/external/KAR_upstream
git clone https://github.com/YunjiaXi/Open-World-Knowledge-Augmented-Recommendation.git upstream
cd upstream

# 2. Install dependencies (their requirements.txt is light)
pip install "transformers>=4.22.2" "torch>=1.10.0" scikit-learn numpy tqdm

# 3. Download ML-1M raw data
mkdir -p data/ml-1m/raw_data
cd data/ml-1m/raw_data
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip && mv ml-1m/* . && rm -rf ml-1m ml-1m.zip
cd ../../..

# 4. Preprocess (creates data/ml-1m/proc_data/)
cd preprocess
python preprocess_ml-1m.py
python generate_data_and_prompt.py --dataset ml-1m
cd ..

# 5. Knowledge encoding (the upstream's pre-generated item.klg / user.klg are in data/ml-1m/knowledge/ — already in repo)
cd knowledge_encoding
python lm_encoding.py --dataset ml-1m
cd ..

# 6. Run CTR with augment=True (KAR enabled)
cd RS
python run_ctr.py  # default args target ml-1m + KAR HEA
```

Wall time on a T4: ~2h for the full bs/lr grid (5×3×5 = 75 configs); ~30min for a single best-known config.

## Quick-mode (single best-known config)

If you don't want to run the full grid, edit `RS/run_ctr.py` to a single point:

```python
# Replace the for-loops at the bottom with:
batch_size = 1024
lr = '5e-4'
export_num = 2
specific_export_num = 5
# (rest of the script unchanged — it already has aug_prefix='bert_avg', augment=True)
```

This single run is the published-default config and should reproduce AUC ≈ 0.814 within seed noise.

## Capturing the result

After the run completes, KAR writes a CSV log under `RS/model/ml-1m/ctr/DIN/.../` with per-epoch val/test AUC. Take the test AUC of the best-val-epoch.

```bash
# After the run:
python3 - <<'EOF'
import json, glob, os
from pathlib import Path
log_dir = Path("RS/model/ml-1m/ctr/DIN")
# Find the most recent run
latest = max(log_dir.iterdir(), key=os.path.getmtime)
log_csv = latest / "log.csv"  # path may need adjustment per upstream
import csv
with open(log_csv) as f:
    rows = list(csv.DictReader(f))
best = max(rows, key=lambda r: float(r.get("val_auc", 0)))
print(json.dumps({"best_val_epoch": int(best["epoch"]),
                  "test_auc": float(best["test_auc"]),
                  "test_logloss": float(best["test_logloss"]),
                  "config_dir": str(latest)}, indent=2))
EOF
```

Paste the output into `code/benchmark/external/KAR_upstream/sanity_result.json`. The paper's appendix subsection (App. E.x, "KAR Sanity Check on Native CTR Domain") cites this file.

## What "passing" the sanity check means

- AUC within ±0.005 of upstream 0.8143 → sanity check passes; the paper appendix can cite a clean reproduction.
- AUC within ±0.005–0.020 → minor drift, likely from a different transformer or BERT version. Cite both numbers and explain.
- AUC > 0.020 off → something is wrong. Investigate before claiming reproduction.

The R2 adaptation in our paper does not depend on this number being exactly right — it depends on us understanding KAR correctly. ±0.020 AUC drift on a re-run is normal across hardware/PyTorch versions and does not invalidate the adaptation claim.
