#!/usr/bin/env bash
# Fix-4 sanity check: clone upstream KAR repo, run published ML-1M-CTR config,
# capture AUC, compare to paper's Table 2 (DIN+KAR=0.8143, baseline DIN=0.7878).
#
# Run this on a GPU machine (Colab Pro / lab server). On a T4 the full pipeline
# (preprocess + lm_encoding + 1 CTR config 20 epochs) takes ~2 hours.
#
# Usage:
#   bash scripts/run_kar_sanity_check.sh
#
# Output:
#   code/benchmark/external/KAR_upstream/sanity_result.json — final AUC + LogLoss
#   code/benchmark/external/KAR_upstream/upstream/ — vendored KAR clone (gitignored)

set -e
set -o pipefail

REPO=$(cd "$(dirname "$0")"/.. && pwd)
DEST="$REPO/code/benchmark/external/KAR_upstream/upstream"
RESULT="$REPO/code/benchmark/external/KAR_upstream/sanity_result.json"
UPSTREAM_URL="https://github.com/YunjiaXi/Open-World-Knowledge-Augmented-Recommendation.git"

echo "=== KAR upstream sanity check ==="
echo "Destination: $DEST"

# 1. Clone upstream
if [ ! -d "$DEST" ]; then
    git clone --depth 1 "$UPSTREAM_URL" "$DEST"
else
    echo "[skip] upstream already cloned at $DEST"
fi

cd "$DEST"
UPSTREAM_SHA=$(git rev-parse HEAD)
echo "Upstream commit: $UPSTREAM_SHA"

# 2. Dependencies
pip install -q "transformers>=4.22.2" "torch>=1.10.0" scikit-learn numpy tqdm pandas

# 3. Fetch ML-1M raw data
mkdir -p data/ml-1m/raw_data
if [ ! -f data/ml-1m/raw_data/ratings.dat ]; then
    cd data/ml-1m/raw_data
    wget -q https://files.grouplens.org/datasets/movielens/ml-1m.zip
    unzip -q ml-1m.zip
    mv ml-1m/*.dat . 2>/dev/null || true
    rm -rf ml-1m ml-1m.zip
    cd "$DEST"
fi

# 4. Preprocess
if [ ! -d data/ml-1m/proc_data ]; then
    cd preprocess
    python preprocess_ml-1m.py
    python generate_data_and_prompt.py --dataset ml-1m || python generate_data_and_prompt.py
    cd "$DEST"
fi

# 5. LM encoding (uses upstream's pre-released item.klg / user.klg)
if [ ! -f data/ml-1m/proc_data/bert_avg_item.npy ]; then
    cd knowledge_encoding
    python lm_encoding.py --dataset ml-1m || python lm_encoding.py
    cd "$DEST"
fi

# 6. Run CTR — single config (the published-default winner) for fast sanity check
#    To run the full grid, edit RS/run_ctr.py instead.
cd RS

# Patch run_ctr.py loop to a single config (idempotent: only patches if not already)
python - <<'PYEOF'
import re, pathlib
p = pathlib.Path("run_ctr.py")
src = p.read_text()
if "# SANITY_CHECK_PATCH" not in src:
    # Replace the nested for-loops with a single config
    new_src = src.replace(
        "for batch_size in [256, 512, 2048, 128, 1024,]:",
        "# SANITY_CHECK_PATCH: collapse grid to published-best config\nfor batch_size in [1024]:",
    ).replace(
        "for lr in ['1e-4', '5e-4', '1e-3']:",
        "for lr in ['5e-4']:",
    ).replace(
        "for specific_export_num in [2, 3, 4, 5, 6]:",
        "for specific_export_num in [5]:",
    )
    p.write_text(new_src)
    print("Patched run_ctr.py to single-config sanity-check mode.")
else:
    print("run_ctr.py already patched.")
PYEOF

python -u run_ctr.py 2>&1 | tee run_ctr.log
cd "$DEST"

# 7. Extract AUC + LogLoss from the run's log
python - <<PYEOF
import json, glob, os, csv
from pathlib import Path

upstream = Path("$DEST")
result_path = Path("$RESULT")

# KAR upstream writes per-run logs under RS/model/ml-1m/ctr/DIN/<run_dir>/
run_root = upstream / "RS" / "model" / "ml-1m" / "ctr" / "DIN"
if not run_root.exists():
    print(f"ERROR: no run output at {run_root}")
    raise SystemExit(1)

latest = max((p for p in run_root.iterdir() if p.is_dir()), key=os.path.getmtime)
log_candidates = list(latest.glob("*.log")) + list(latest.glob("*.csv")) + list(latest.glob("*.txt"))
print(f"Latest run dir: {latest}")
print(f"Log files: {[str(p) for p in log_candidates]}")

# Try to find AUC in the run log
import re
text = ""
for f in log_candidates:
    text += f.read_text(errors="ignore") + "\n"
text += (upstream / "RS" / "run_ctr.log").read_text(errors="ignore") if (upstream / "RS" / "run_ctr.log").exists() else ""

aucs = re.findall(r"test[_ ]?auc[\s:=]+([\d.]+)", text, re.IGNORECASE)
loglosses = re.findall(r"test[_ ]?log[_]?loss[\s:=]+([\d.]+)", text, re.IGNORECASE)

best_auc = max(map(float, aucs)) if aucs else None
best_ll = min(map(float, loglosses)) if loglosses else None

published = {"DIN_KAR_AUC": 0.8143, "DIN_KAR_LogLoss": 0.5096,
             "DIN_baseline_AUC": 0.7878, "DIN_baseline_LogLoss": 0.5364}

result = {
    "upstream_repo": "YunjiaXi/Open-World-Knowledge-Augmented-Recommendation",
    "upstream_commit": "$UPSTREAM_SHA",
    "config": "DIN + KAR (HEA, augment=True), bs=1024, lr=5e-4, export_num=2, specific_export_num=5, epoch=20",
    "dataset": "ml-1m (native CTR)",
    "our_run_test_AUC": best_auc,
    "our_run_test_LogLoss": best_ll,
    "published_DIN_KAR": published,
    "auc_delta_vs_published": (best_auc - published["DIN_KAR_AUC"]) if best_auc else None,
    "sanity_check_passed": (best_auc is not None and abs(best_auc - 0.8143) < 0.020),
    "run_dir": str(latest),
}

result_path.parent.mkdir(parents=True, exist_ok=True)
result_path.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
PYEOF

echo ""
echo "=== Done ==="
echo "Result: $RESULT"
