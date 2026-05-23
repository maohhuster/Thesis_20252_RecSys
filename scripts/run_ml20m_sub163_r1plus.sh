#!/usr/bin/env bash
# Run R1-plus (RLMRec-plus, contrastive variant) on subsampled ML-20M with our LLM embeddings.
# Closes the fourth density point (same-domain control) in the regularizer-class triangulation.
# R1-gene subsampled-ML-20M already shows +3.7% vs M7 (n.s.); R1-plus expected to land in the
# expected slight loss-to-gain transition at this intermediate density. A non-tie at this datapoint would falsify
# the regularizer-class monotone-in-1/density prediction.
#
# Mirrors scripts/run_ml1m_r1plus.sh / run_amazon_r1plus.sh.
# Data dir reused from existing run_rlmrec.sh pipeline:
#   code/benchmark/external/RLMRec/data/ml20m_sub163_ours/  (Claude-Haiku-4.5 + bge-large-en-v1.5)
# YAML block: lightgcn_plus.yml > model.ml20m_sub163_ours  (already present)
#
# Usage:
#   bash scripts/run_ml20m_sub163_r1plus.sh                     # default (CPU)
#   DEVICE=mps bash scripts/run_ml20m_sub163_r1plus.sh          # Apple Silicon (recommended)
#   DEVICE=cuda bash scripts/run_ml20m_sub163_r1plus.sh         # GPU
#   SEED=123 bash scripts/run_ml20m_sub163_r1plus.sh            # specific seed
#
# Memory note: subsampled ML-20M has 17K users; RLMRec-plus uses contrastive distillation
# whose pair-construction memory can be larger than R1-gene. If the run OOMs,
# reduce batch_size in lightgcn_plus.yml from 4096 → 2048 → 1024 (linearly slower).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEVICE="${DEVICE:-cpu}"
CUDA="${CUDA:-0}"
SEED="${SEED:-42}"
RESULTS_DIR="code/benchmark/results_ml20m_sub163/r1plus/ml20m_sub163_ours"
LOG_DIR="code/benchmark/results_ml20m_sub163/logs"
CKPT_DIR="code/benchmark/checkpoints_ml20m_sub163/r1plus/bge-large-en-v1.5/seed-${SEED}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CKPT_DIR"

echo "═══ R1-plus (RLMRec-plus, contrastive) on subsampled-ML-20M (same-domain density control) ═══"
echo "  device : $DEVICE"
echo "  seed   : $SEED"
[[ "$DEVICE" == "cuda" ]] && echo "  cuda   : $CUDA"
echo "  data   : reuses code/benchmark/external/RLMRec/data/ml20m_sub163_ours/ (R1-gene preset, Claude+bge)"

# Sanity check that data is prepared
if [ ! -f "code/benchmark/external/RLMRec/data/ml20m_sub163_ours/itm_emb_np.pkl" ]; then
    echo "  WARNING: ml20m_sub163_ours data not prepared. Running prepare step..."
    cd code/benchmark/external
    python3 prepare_rlmrec_ml20m_sub163.py
    cd "$REPO"
fi

# Launch trainer
cd code/benchmark/external/RLMRec
LOG="$REPO/$LOG_DIR/r1plus-ml20m_sub163-seed${SEED}-$(date +%Y%m%d-%H%M%S).log"

python3 encoder/train_encoder.py \
    --model lightgcn_plus \
    --dataset ml20m_sub163_ours \
    --device "$DEVICE" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    2>&1 | tee "$LOG"

# Copy native checkpoint to project's standard location
NATIVE_CKPT="encoder/checkpoint/lightgcn_plus/lightgcn_plus-ml20m_sub163_ours-${SEED}.pth"
PROJECT_CKPT="$REPO/$CKPT_DIR/best_model.pt"
if [ -f "$NATIVE_CKPT" ]; then
    cp "$NATIVE_CKPT" "$PROJECT_CKPT"
    echo "  → checkpoint saved to $PROJECT_CKPT"
fi

cd "$REPO"
echo "{\"experiment\": \"r1plus_ml20m_sub163_seed${SEED}\", \"status\": \"complete\", \"log\": \"$LOG\", \"checkpoint\": \"$CKPT_DIR/best_model.pt\"}" \
    > "$RESULTS_DIR/seed-${SEED}-marker.json"
echo ""
echo "═══ Done ═══"
echo "  Log     : $LOG"
echo "  Marker  : $RESULTS_DIR/seed-${SEED}-marker.json"
echo "  Ckpt    : $CKPT_DIR/best_model.pt"
echo ""
echo "  Next step: python3 scripts/eval_ml20m_sub163_r1plus.py --device $DEVICE --seeds $SEED"
