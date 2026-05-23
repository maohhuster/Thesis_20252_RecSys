#!/bin/bash
# Run RLMRec experiments on ML-20M
#
# Resume support:
#   - Data preparation is skipped if all files exist
#   - Each model variant checks for saved results before running
#   - Safe to re-run after interruption
#
# Usage:
#   bash run_rlmrec.sh              # run all 3 variants
#   bash run_rlmrec.sh plus         # run only contrastive variant
#   bash run_rlmrec.sh gene         # run only generative variant
#   bash run_rlmrec.sh backbone     # run only backbone (no LLM features)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/RLMRec"

DEVICE="${DEVICE:-cpu}"  # set DEVICE=cuda for GPU
RESULTS_DIR="$SCRIPT_DIR/../results"

# Step 1: Prepare data (skips if already done)
echo "=== Checking RLMRec data ==="
cd "$SCRIPT_DIR"
python3 prepare_rlmrec_data.py --item-features profile
cd "$SCRIPT_DIR/RLMRec"

run_variant() {
    local model=$1 label=$2
    local result_dir="${RESULTS_DIR}/rlmrec_${model}__ml20m"
    if [ -f "${result_dir}/results.json" ]; then
        echo "=== RLMRec: ${label} — SKIPPED (already complete) ==="
    else
        echo "=== RLMRec: ${label} ==="
        python3 encoder/train_encoder.py --model "$model" --dataset ml20m --device "$DEVICE"
        # Save results marker
        mkdir -p "$result_dir"
        echo "{\"experiment\": \"rlmrec_${model}__ml20m\", \"status\": \"complete\"}" > "${result_dir}/results.json"
    fi
}

VARIANT="${1:-all}"

if [ "$VARIANT" = "all" ] || [ "$VARIANT" = "backbone" ]; then
    run_variant lightgcn "LightGCN backbone (ML-20M)"
fi

if [ "$VARIANT" = "all" ] || [ "$VARIANT" = "plus" ]; then
    run_variant lightgcn_plus "LightGCN+ contrastive (ML-20M)"
fi

if [ "$VARIANT" = "all" ] || [ "$VARIANT" = "gene" ]; then
    run_variant lightgcn_gene "LightGCN-gene generative (ML-20M)"
fi

echo "=== Done ==="
