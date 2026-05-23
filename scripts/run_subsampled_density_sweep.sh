#!/usr/bin/env bash
# Launch the full subsampled-ML-20M density-control sweep:
#   M7 seed=42 (the missing seed-42 cell), then all configs × seeds {123,456,789,2026}
#
# Configurations:  M1 (LightGCN ID-only), M4 (+LLM profile), M7 (+profile+mood),
#                  R1 (RLMRec-gene), SASRec (pmixer reference)
# Seeds:           42 (only M7), 123, 456, 789, 2026 (all 5 configs)
#
# Total: 1 + 4 × 5 = 21 runs. Estimated wall time on Mac MPS: ~7 hours.
#
# Usage:
#   bash scripts/run_subsampled_density_sweep.sh        # blocks; prints progress
#   nohup bash scripts/run_subsampled_density_sweep.sh > /tmp/sweep.log 2>&1 &  # detached
#
# Each run writes its own log under code/benchmark/results_ml20m_sub163/logs/
# Aggregate metrics get written to results_ml20m_sub163/sweep_summary.csv at the end.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DATA_DIR="$REPO/code/benchmark/data/processed_ml20m_sub163"
RES_DIR="$REPO/code/benchmark/results_ml20m_sub163"
CKPT_DIR="$REPO/code/benchmark/checkpoints_ml20m_sub163"
LOG_DIR="$RES_DIR/logs"
mkdir -p "$LOG_DIR" "$RES_DIR" "$CKPT_DIR"

DEVICE="${DEVICE:-mps}"
SWEEP_LOG="$LOG_DIR/sweep_$(date +%Y%m%d-%H%M%S).log"
echo "===========================================================================" | tee -a "$SWEEP_LOG"
echo " Subsampled ML-20M density-control sweep " | tee -a "$SWEEP_LOG"
echo " Started: $(date)" | tee -a "$SWEEP_LOG"
echo " Device:  $DEVICE" | tee -a "$SWEEP_LOG"
echo " Log:     $SWEEP_LOG" | tee -a "$SWEEP_LOG"
echo "===========================================================================" | tee -a "$SWEEP_LOG"

# ---------- Step 0: wait for any in-flight R1 / SASRec / lightgcn runs ----------
echo "[$(date +%H:%M:%S)] Waiting for any in-flight runs to finish..." | tee -a "$SWEEP_LOG"
while pgrep -f "train_encoder.py|run_experiment.py|run_ml20m_sub163_sasrec_pmixer.py" >/dev/null 2>&1; do
    sleep 30
done
echo "[$(date +%H:%M:%S)] No in-flight runs detected. Proceeding." | tee -a "$SWEEP_LOG"

# ---------- Helper: run one config × seed cell ----------
run_M_cell() {
    # M1 / M4 / M7 — LightGCN-SF main pipeline
    local CONFIG="$1"  # "M1" | "M4" | "M7"
    local SEED="$2"
    local MODEL FEATURES
    case "$CONFIG" in
        M1) MODEL="lightgcn";    FEATURES="none" ;;
        M4) MODEL="lightgcn_sf"; FEATURES="llm_profile" ;;
        M7) MODEL="lightgcn_sf"; FEATURES="llm_prof_mood" ;;
        *)  echo "Unknown config: $CONFIG"; return 1 ;;
    esac

    local LOG="$LOG_DIR/${CONFIG}-seed${SEED}.log"
    echo "[$(date +%H:%M:%S)] === ${CONFIG} seed=${SEED} ===" | tee -a "$SWEEP_LOG"
    local START=$(date +%s)
    cd "$REPO/code/benchmark" && python3 run_experiment.py \
        --model "$MODEL" \
        --features "$FEATURES" \
        --seed "$SEED" \
        --data-dir "$DATA_DIR" \
        --results-dir "$RES_DIR" \
        --checkpoint-dir "$CKPT_DIR" \
        --device "$DEVICE" 2>&1 | tee "$LOG" | tail -2 >> "$SWEEP_LOG"
    local STATUS=$?
    cd "$REPO"
    local DUR=$(( $(date +%s) - START ))
    echo "[$(date +%H:%M:%S)] ${CONFIG} seed=${SEED} done (${DUR}s, exit=${STATUS})" | tee -a "$SWEEP_LOG"
}

run_R1_cell() {
    local SEED="$1"
    local LOG="$LOG_DIR/R1-seed${SEED}.log"
    echo "[$(date +%H:%M:%S)] === R1 seed=${SEED} ===" | tee -a "$SWEEP_LOG"
    local START=$(date +%s)
    cd "$REPO/code/benchmark/external/RLMRec" && python3 encoder/train_encoder.py \
        --model lightgcn_gene \
        --dataset ml20m_sub163_ours \
        --device "$DEVICE" \
        --seed "$SEED" 2>&1 | tee "$LOG" | tail -2 >> "$SWEEP_LOG"
    local STATUS=$?
    cd "$REPO"
    local DUR=$(( $(date +%s) - START ))
    echo "[$(date +%H:%M:%S)] R1 seed=${SEED} done (${DUR}s, exit=${STATUS})" | tee -a "$SWEEP_LOG"
}

run_SASRec_cell() {
    local SEED="$1"
    local LOG="$LOG_DIR/SASRec-seed${SEED}.log"
    echo "[$(date +%H:%M:%S)] === SASRec seed=${SEED} ===" | tee -a "$SWEEP_LOG"
    local START=$(date +%s)
    python3 scripts/run_ml20m_sub163_sasrec_pmixer.py --seed "$SEED" --device "$DEVICE" 2>&1 \
        | tee "$LOG" | tail -2 >> "$SWEEP_LOG"
    local STATUS=$?
    local DUR=$(( $(date +%s) - START ))
    echo "[$(date +%H:%M:%S)] SASRec seed=${SEED} done (${DUR}s, exit=${STATUS})" | tee -a "$SWEEP_LOG"
}

# ---------- Step 1: missing seed=42 cells (R1 + SASRec interrupted; M7 not yet run) ----------
run_M_cell M7 42
run_R1_cell 42
run_SASRec_cell 42

# ---------- Step 2: full sweep over remaining 4 seeds × 5 configs ----------
for SEED in 123 456 789 2026; do
    echo "[$(date +%H:%M:%S)] ───── seed=${SEED} ─────" | tee -a "$SWEEP_LOG"
    run_M_cell M1 "$SEED"
    run_M_cell M4 "$SEED"
    run_M_cell M7 "$SEED"
    run_R1_cell "$SEED"
    run_SASRec_cell "$SEED"
done

# ---------- Step 3: report ----------
echo "===========================================================================" | tee -a "$SWEEP_LOG"
echo " Sweep complete: $(date)" | tee -a "$SWEEP_LOG"
echo " All logs in $LOG_DIR/" | tee -a "$SWEEP_LOG"
echo " Aggregate sweep log: $SWEEP_LOG" | tee -a "$SWEEP_LOG"
echo "===========================================================================" | tee -a "$SWEEP_LOG"
