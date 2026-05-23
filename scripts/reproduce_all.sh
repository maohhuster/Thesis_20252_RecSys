#!/usr/bin/env bash
# Full reproduction script for LLM-MovieLens benchmark
# Runs all 15 configurations × 5 seeds
#
# Usage:
#   bash scripts/reproduce_all.sh                    # Full run
#   bash scripts/reproduce_all.sh --tier 2           # Only Tier 2
#   bash scripts/reproduce_all.sh --config M4        # Single config
#   bash scripts/reproduce_all.sh --dry-run          # Print commands only

set -euo pipefail

SEEDS=(42 123 456 789 2026)
TIER1=(M0 M1 M1b M1c M1d)
TIER2=(M2 M3 M4 M5 M6 M7 M8 M9)
TIER3=(R2 R3)

TIER_FILTER=""
CONFIG_FILTER=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --tier) TIER_FILTER="$2"; shift 2 ;;
        --config) CONFIG_FILTER="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Select configs based on filters
CONFIGS=()
if [ -n "$CONFIG_FILTER" ]; then
    CONFIGS=("$CONFIG_FILTER")
elif [ -n "$TIER_FILTER" ]; then
    case $TIER_FILTER in
        1) CONFIGS=("${TIER1[@]}") ;;
        2) CONFIGS=("${TIER2[@]}") ;;
        3) CONFIGS=("${TIER3[@]}") ;;
        *) echo "Invalid tier: $TIER_FILTER (use 1, 2, or 3)" >&2; exit 1 ;;
    esac
else
    CONFIGS=("${TIER1[@]}" "${TIER2[@]}" "${TIER3[@]}")
fi

TOTAL=$((${#CONFIGS[@]} * ${#SEEDS[@]}))
CURRENT=0
FAILED=0

echo "=== LLM-MovieLens Benchmark Reproduction ==="
echo "Configs: ${CONFIGS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total experiments: ${TOTAL}"
echo ""

START_TIME=$(date +%s)

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        CMD="python -m llm_movielens.benchmark.run_experiment --config ${config} --seed ${seed}"

        echo "[${CURRENT}/${TOTAL}] ${config} seed=${seed}"

        if [ "$DRY_RUN" = true ]; then
            echo "  (dry run) ${CMD}"
        else
            if $CMD; then
                echo "  Done."
            else
                echo "  FAILED!" >&2
                FAILED=$((FAILED + 1))
            fi
        fi
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=== Summary ==="
echo "Completed: $((CURRENT - FAILED))/${TOTAL}"
echo "Failed: ${FAILED}"
echo "Time: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
