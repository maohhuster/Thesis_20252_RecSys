#!/usr/bin/env bash
# Run R1-gene (RLMRec-gene, generative variant) on subsampled ML-20M with our LLM embeddings.
# Closes the R1-gene same-domain density control point (+3.7% vs M7, n.s.) so it is
# checkpoint-reproducible like its R1-plus / SASRec sub163 siblings (previously JSON-only).
#
# Faithful mirror of scripts/run_ml20m_sub163_r1plus.sh — only the model
# (lightgcn_gene vs lightgcn_plus) and the r1/ output paths differ.
# Data dir reused from the existing run_rlmrec.sh pipeline:
#   code/benchmark/external/RLMRec/data/ml20m_sub163_ours/  (Claude-Haiku-4.5 + bge-large-en-v1.5)
# YAML block: lightgcn_gene.yml > model.ml20m_sub163_ours  (already present)
#
# Usage:
#   bash scripts/run_ml20m_sub163_r1.sh                  # default (CPU)
#   DEVICE=mps bash scripts/run_ml20m_sub163_r1.sh       # Apple Silicon (recommended)
#   DEVICE=cuda bash scripts/run_ml20m_sub163_r1.sh      # GPU
#   SEED=123 bash scripts/run_ml20m_sub163_r1.sh         # specific seed
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEVICE="${DEVICE:-cpu}"
CUDA="${CUDA:-0}"
SEED="${SEED:-42}"
RESULTS_DIR="code/benchmark/results_ml20m_sub163/r1/ml20m_sub163_ours"
LOG_DIR="code/benchmark/results_ml20m_sub163/logs"
CKPT_DIR="code/benchmark/checkpoints_ml20m_sub163/r1/bge-large-en-v1.5/seed-${SEED}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CKPT_DIR"

echo "═══ R1-gene (RLMRec-gene, generative) on subsampled-ML-20M (same-domain density control) ═══"
echo "  device : $DEVICE"
echo "  seed   : $SEED"
[[ "$DEVICE" == "cuda" ]] && echo "  cuda   : $CUDA"
echo "  data   : reuses code/benchmark/external/RLMRec/data/ml20m_sub163_ours/ (Claude+bge)"

# Sanity check that data is prepared
if [ ! -f "code/benchmark/external/RLMRec/data/ml20m_sub163_ours/itm_emb_np.pkl" ]; then
    echo "  WARNING: ml20m_sub163_ours data not prepared. Running prepare step..."
    cd code/benchmark/external
    python3 prepare_rlmrec_ml20m_sub163.py
    cd "$REPO"
fi

# Launch trainer
cd code/benchmark/external/RLMRec
LOG="$REPO/$LOG_DIR/r1-ml20m_sub163-seed${SEED}-$(date +%Y%m%d-%H%M%S).log"

python3 encoder/train_encoder.py \
    --model lightgcn_gene \
    --dataset ml20m_sub163_ours \
    --device "$DEVICE" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    2>&1 | tee "$LOG"

# Copy native checkpoint to project's standard location
NATIVE_CKPT="encoder/checkpoint/lightgcn_gene/lightgcn_gene-ml20m_sub163_ours-${SEED}.pth"
PROJECT_CKPT="$REPO/$CKPT_DIR/best_model.pt"
if [ -f "$NATIVE_CKPT" ]; then
    cp "$NATIVE_CKPT" "$PROJECT_CKPT"
    echo "  → checkpoint saved to $PROJECT_CKPT"
fi

cd "$REPO"
echo "{\"experiment\": \"r1_ml20m_sub163_seed${SEED}\", \"status\": \"complete\", \"log\": \"$LOG\", \"checkpoint\": \"$CKPT_DIR/best_model.pt\"}" \
    > "$RESULTS_DIR/seed-${SEED}-marker.json"
echo ""
echo "═══ Done ═══"
echo "  Log     : $LOG"
echo "  Marker  : $RESULTS_DIR/seed-${SEED}-marker.json"
echo "  Ckpt    : $CKPT_DIR/best_model.pt"
