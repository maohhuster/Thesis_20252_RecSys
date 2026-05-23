#!/usr/bin/env bash
# Run R1 (RLMRec-gene) on Amazon-Books with our LLM embeddings.
#
# Steps:
#   1. Prepare RLMRec data under RLMRec/data/amazon_ours/ (idempotent).
#   2. Patch lightgcn_gene.yml to add `amazon_ours:` block (idempotent).
#   3. Invoke RLMRec's upstream encoder trainer with --dataset amazon_ours.
#
# Hyperparameters: the appended `amazon_ours:` block copies the upstream
# `amazon:` values verbatim (RLMRec authors' published Amazon-Book hparams).
# This is the strongest position — no local tuning required because the
# original published config applies unchanged to a same-density dataset.
#
# Usage:
#   bash scripts/run_amazon_r1.sh                     # default (CPU)
#   DEVICE=cuda bash scripts/run_amazon_r1.sh         # GPU
#   DEVICE=cuda CUDA=0 bash scripts/run_amazon_r1.sh  # pin to GPU 0
#
# Output:
#   code/benchmark/external/RLMRec/encoder/log/      (RLMRec's own log dir)
#   code/benchmark/results_amazon/r1/amazon_ours/    (results marker)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEVICE="${DEVICE:-cpu}"
CUDA="${CUDA:-0}"
RESULTS_DIR="code/benchmark/results_amazon/r1/amazon_ours"
LOG_DIR="code/benchmark/results_amazon/logs"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

echo "═══ R1 (RLMRec-gene) on Amazon-Books ═══"
echo "  device : $DEVICE"
[[ "$DEVICE" == "cuda" ]] && echo "  cuda   : $CUDA"

# Step 1+2: prep data and YAML
python3 scripts/prepare_rlmrec_amazon.py --patch-yaml

# Step 3: run upstream encoder
echo ""
echo "── Launching RLMRec encoder (lightgcn_gene, --dataset amazon_ours)..."
cd code/benchmark/external/RLMRec
LOG="$REPO/$LOG_DIR/r1-amazon-$(date +%Y%m%d-%H%M%S).log"
python3 encoder/train_encoder.py \
    --model lightgcn_gene \
    --dataset amazon_ours \
    --device "$DEVICE" \
    --cuda "$CUDA" \
    2>&1 | tee "$LOG"

# Marker: RLMRec writes its own metrics to its log directory; we drop a
# completion marker for the aggregation script to discover.
cd "$REPO"
echo "{\"experiment\": \"r1_amazon_ours\", \"status\": \"complete\", \"log\": \"$LOG\"}" \
    > "$RESULTS_DIR/results.json"
echo ""
echo "═══ Done ═══"
echo "  Log     : $LOG"
echo "  Marker  : $RESULTS_DIR/results.json"
echo "  Metrics : grep 'recall@10\\|ndcg@10' '$LOG' | tail"
