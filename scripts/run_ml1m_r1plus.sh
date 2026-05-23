#!/usr/bin/env bash
# Run R1-plus (RLMRec-plus, contrastive variant) on ML-1M with our LLM embeddings.
# Pre-registered regularizer-class triangulation: a second instantiation of the
# regularizer paradigm at the intermediate-density datapoint (where R1-gene already
# shows +17% over M7).
#
# Mirrors scripts/run_ml1m_r1.sh exactly; only --model arg changes.
#
# Usage:
#   bash scripts/run_ml1m_r1plus.sh                     # default (CPU)
#   DEVICE=mps bash scripts/run_ml1m_r1plus.sh          # Apple Silicon
#   DEVICE=cuda bash scripts/run_ml1m_r1plus.sh         # GPU
#   SEED=123 bash scripts/run_ml1m_r1plus.sh            # specific seed
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEVICE="${DEVICE:-cpu}"
CUDA="${CUDA:-0}"
SEED="${SEED:-42}"
RESULTS_DIR="code/benchmark/results_ml1m/r1plus/ml1m_ours"
LOG_DIR="code/benchmark/results_ml1m/logs"
CKPT_DIR="code/benchmark/checkpoints_ml1m/r1plus/bge-large-en-v1.5/seed-${SEED}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CKPT_DIR"

echo "═══ R1-plus (RLMRec-plus, contrastive) on ML-1M ═══"
echo "  device : $DEVICE"
echo "  seed   : $SEED"
[[ "$DEVICE" == "cuda" ]] && echo "  cuda   : $CUDA"

# Step 1 — data prep is already done for ml1m_ours (same data as R1-gene)
echo "  data   : reuses code/benchmark/external/RLMRec/data/ml1m_ours/ (R1-gene preset)"

# Step 2 — launch trainer
cd code/benchmark/external/RLMRec
LOG="$REPO/$LOG_DIR/r1plus-ml1m-seed${SEED}-$(date +%Y%m%d-%H%M%S).log"

python3 encoder/train_encoder.py \
    --model lightgcn_plus \
    --dataset ml1m_ours \
    --device "$DEVICE" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    2>&1 | tee "$LOG"

# Step 3 — copy native checkpoint to the project's standard location
NATIVE_CKPT="encoder/checkpoint/lightgcn_plus/lightgcn_plus-ml1m_ours-${SEED}.pth"
PROJECT_CKPT="$REPO/$CKPT_DIR/best_model.pt"
if [ -f "$NATIVE_CKPT" ]; then
    cp "$NATIVE_CKPT" "$PROJECT_CKPT"
    echo "  → checkpoint saved to $PROJECT_CKPT"
fi

cd "$REPO"
echo "{\"experiment\": \"r1plus_ml1m_ours_seed${SEED}\", \"status\": \"complete\", \"log\": \"$LOG\", \"checkpoint\": \"$CKPT_DIR/best_model.pt\"}" \
    > "$RESULTS_DIR/seed-${SEED}-marker.json"
echo ""
echo "═══ Done ═══"
echo "  Log     : $LOG"
echo "  Marker  : $RESULTS_DIR/seed-${SEED}-marker.json"
echo "  Ckpt    : $CKPT_DIR/best_model.pt"
echo ""
echo "  Next step: python3 scripts/eval_ml1m_r1plus.py --device $DEVICE --seeds $SEED"
