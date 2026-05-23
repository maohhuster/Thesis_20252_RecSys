#!/usr/bin/env bash
# Run R1-plus (RLMRec-plus, contrastive variant) on Amazon-Books with our LLM embeddings.
# Pre-registered regularizer-class triangulation: a second instantiation of the
# regularizer paradigm at the sparse-per-user density datapoint (where R1-gene shows +29.5% over M7).
#
# Mirrors scripts/run_amazon_r1.sh + scripts/run_ml1m_r1plus.sh; only --model + --dataset change.
#
# Usage:
#   bash scripts/run_amazon_r1plus.sh                     # default (CPU)
#   DEVICE=mps bash scripts/run_amazon_r1plus.sh          # Apple Silicon
#   DEVICE=cuda bash scripts/run_amazon_r1plus.sh         # GPU
#   SEED=123 bash scripts/run_amazon_r1plus.sh            # specific seed
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEVICE="${DEVICE:-cpu}"
CUDA="${CUDA:-0}"
SEED="${SEED:-42}"
RESULTS_DIR="code/benchmark/results_amazon/r1plus/amazon_ours"
LOG_DIR="code/benchmark/results_amazon/logs"
CKPT_DIR="code/benchmark/checkpoints_amazon/r1plus/bge-large-en-v1.5/seed-${SEED}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CKPT_DIR"

echo "═══ R1-plus (RLMRec-plus, contrastive) on Amazon-Books ═══"
echo "  device : $DEVICE"
echo "  seed   : $SEED"
[[ "$DEVICE" == "cuda" ]] && echo "  cuda   : $CUDA"
echo "  data   : reuses code/benchmark/external/RLMRec/data/amazon_ours/ (R1-gene preset)"

# Launch trainer
cd code/benchmark/external/RLMRec
LOG="$REPO/$LOG_DIR/r1plus-amazon-seed${SEED}-$(date +%Y%m%d-%H%M%S).log"

python3 encoder/train_encoder.py \
    --model lightgcn_plus \
    --dataset amazon_ours \
    --device "$DEVICE" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    2>&1 | tee "$LOG"

# Copy native checkpoint to project's standard location
NATIVE_CKPT="encoder/checkpoint/lightgcn_plus/lightgcn_plus-amazon_ours-${SEED}.pth"
PROJECT_CKPT="$REPO/$CKPT_DIR/best_model.pt"
if [ -f "$NATIVE_CKPT" ]; then
    cp "$NATIVE_CKPT" "$PROJECT_CKPT"
    echo "  → checkpoint saved to $PROJECT_CKPT"
fi

cd "$REPO"
echo "{\"experiment\": \"r1plus_amazon_ours_seed${SEED}\", \"status\": \"complete\", \"log\": \"$LOG\", \"checkpoint\": \"$CKPT_DIR/best_model.pt\"}" \
    > "$RESULTS_DIR/seed-${SEED}-marker.json"
echo ""
echo "═══ Done ═══"
echo "  Log     : $LOG"
echo "  Marker  : $RESULTS_DIR/seed-${SEED}-marker.json"
echo "  Ckpt    : $CKPT_DIR/best_model.pt"
echo ""
echo "  Next step: python3 scripts/eval_amazon_r1plus.py --device $DEVICE --seeds $SEED"
