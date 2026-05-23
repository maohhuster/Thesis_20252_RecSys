#!/usr/bin/env bash
# Run R1 (RLMRec-gene) on ML-1M with our LLM embeddings.
#
# Steps:
#   1. Prepare RLMRec data under RLMRec/data/ml1m_ours/ (idempotent).
#   2. Patch lightgcn_gene.yml (+ml1m_ours block) and data_handler_general_cf.py
#      (+ml1m_ours branch). Both patches are idempotent.
#   3. Invoke RLMRec's upstream encoder trainer with --dataset ml1m_ours.
#
# Usage:
#   bash scripts/run_ml1m_r1.sh                     # default (CPU)
#   DEVICE=cuda bash scripts/run_ml1m_r1.sh         # GPU
#   DEVICE=cuda CUDA=0 bash scripts/run_ml1m_r1.sh  # pin to GPU 0
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEVICE="${DEVICE:-cpu}"
CUDA="${CUDA:-0}"
RESULTS_DIR="code/benchmark/results_ml1m/r1/ml1m_ours"
LOG_DIR="code/benchmark/results_ml1m/logs"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

echo "═══ R1 (RLMRec-gene) on ML-1M ═══"
echo "  device : $DEVICE"
[[ "$DEVICE" == "cuda" ]] && echo "  cuda   : $CUDA"

# Step 1+2
python3 scripts/prepare_rlmrec_ml1m.py --patch-yaml

# Step 3
echo ""
echo "── Launching RLMRec encoder (lightgcn_gene, --dataset ml1m_ours)..."
cd code/benchmark/external/RLMRec
LOG="$REPO/$LOG_DIR/r1-ml1m-$(date +%Y%m%d-%H%M%S).log"
python3 encoder/train_encoder.py \
    --model lightgcn_gene \
    --dataset ml1m_ours \
    --device "$DEVICE" \
    --cuda "$CUDA" \
    2>&1 | tee "$LOG"

cd "$REPO"
echo "{\"experiment\": \"r1_ml1m_ours\", \"status\": \"complete\", \"log\": \"$LOG\"}" \
    > "$RESULTS_DIR/results.json"
echo ""
echo "═══ Done ═══"
echo "  Log     : $LOG"
echo "  Marker  : $RESULTS_DIR/results.json"
