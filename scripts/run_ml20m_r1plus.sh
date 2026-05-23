#!/usr/bin/env bash
# Run R1-plus (RLMRec-plus, contrastive variant) on FULL ML-20M with our LLM embeddings.
# Closes the third density point in the regularizer-class triangulation.
# R1-gene ML-20M already shows -2.2% vs M7 (n.s.); R1-plus expected to land in the
# same dense-regime tie/loss range. A non-tie at this datapoint would falsify
# the regularizer-class monotone-in-1/density prediction.
#
# Mirrors scripts/run_ml1m_r1plus.sh / run_amazon_r1plus.sh.
# Data dir reused from existing run_rlmrec.sh pipeline:
#   code/benchmark/external/RLMRec/data/ml20m/  (Claude-Haiku-4.5 + bge-large-en-v1.5)
# YAML block: lightgcn_plus.yml > model.ml20m  (already present)
#
# Usage:
#   bash scripts/run_ml20m_r1plus.sh                     # default (CPU)
#   DEVICE=mps bash scripts/run_ml20m_r1plus.sh          # Apple Silicon (recommended)
#   DEVICE=cuda bash scripts/run_ml20m_r1plus.sh         # GPU
#   SEED=123 bash scripts/run_ml20m_r1plus.sh            # specific seed
#
# Memory note: ML-20M has 127K users; RLMRec-plus uses contrastive distillation
# whose pair-construction memory can be larger than R1-gene. If the run OOMs,
# reduce batch_size in lightgcn_plus.yml from 4096 → 2048 → 1024 (linearly slower).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEVICE="${DEVICE:-cpu}"
CUDA="${CUDA:-0}"
SEED="${SEED:-42}"
RESULTS_DIR="code/benchmark/results/r1plus/ml20m"
LOG_DIR="code/benchmark/results/logs"
CKPT_DIR="code/benchmark/checkpoints/r1plus/bge-large-en-v1.5/seed-${SEED}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CKPT_DIR"

echo "═══ R1-plus (RLMRec-plus, contrastive) on ML-20M ═══"
echo "  device : $DEVICE"
echo "  seed   : $SEED"
[[ "$DEVICE" == "cuda" ]] && echo "  cuda   : $CUDA"
echo "  data   : reuses code/benchmark/external/RLMRec/data/ml20m/ (R1-gene preset, Claude+bge)"

# Sanity check that data is prepared
if [ ! -f "code/benchmark/external/RLMRec/data/ml20m/itm_emb_np.pkl" ]; then
    echo "  WARNING: ml20m data not prepared. Running prepare step..."
    cd code/benchmark/external
    python3 prepare_rlmrec_data.py --item-features profile
    cd "$REPO"
fi

# Capacity-match to d=128 (shared LightGCN-SF backbone dimension; paper App. r1plus_triangulation).
# We do NOT modify the upstream YAML (which keeps embedding_size=32 for sparse-dataset runs).
# Instead, we generate a one-off override YAML for this ML-20M run only, and pass it to the
# configurator via --model-conf. The override is created in a tempdir and removed at exit.
cd code/benchmark/external/RLMRec
UPSTREAM_YAML="encoder/config/modelconf/lightgcn_plus.yml"
OVERRIDE_YAML="$(mktemp -t lightgcn_plus_ml20m_d128_XXXXXX).yml"
trap 'rm -f "$OVERRIDE_YAML"' EXIT
sed 's/^  embedding_size: 32$/  embedding_size: 128/' "$UPSTREAM_YAML" > "$OVERRIDE_YAML"
echo "  d=128 override config (capacity-match to shared backbone):"
echo "    upstream: $UPSTREAM_YAML  (unchanged; embedding_size: 32)"
echo "    override: $OVERRIDE_YAML  (embedding_size: 32 → 128)"

# Launch trainer
LOG="$REPO/$LOG_DIR/r1plus-ml20m-seed${SEED}-$(date +%Y%m%d-%H%M%S).log"

python3 encoder/train_encoder.py \
    --model lightgcn_plus \
    --model-conf "$OVERRIDE_YAML" \
    --dataset ml20m \
    --device "$DEVICE" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    2>&1 | tee "$LOG"

# Copy native checkpoint to project's standard location
NATIVE_CKPT="encoder/checkpoint/lightgcn_plus/lightgcn_plus-ml20m-${SEED}.pth"
PROJECT_CKPT="$REPO/$CKPT_DIR/best_model.pt"   # consolidated M-config-style naming (was .pth)
if [ -f "$NATIVE_CKPT" ]; then
    cp "$NATIVE_CKPT" "$PROJECT_CKPT"
    echo "  → checkpoint saved to $PROJECT_CKPT"
fi

cd "$REPO"
echo "{\"experiment\": \"r1plus_ml20m_seed${SEED}\", \"status\": \"complete\", \"log\": \"$LOG\", \"checkpoint\": \"$CKPT_DIR/best_model.pt\"}" \
    > "$RESULTS_DIR/seed-${SEED}-marker.json"
echo ""
echo "═══ Done ═══"
echo "  Log     : $LOG"
echo "  Marker  : $RESULTS_DIR/seed-${SEED}-marker.json"
echo "  Ckpt    : $CKPT_DIR/best_model.pt"
echo ""
echo "  Next step: python3 scripts/eval_ml20m_r1plus.py --device $DEVICE --seeds $SEED"
