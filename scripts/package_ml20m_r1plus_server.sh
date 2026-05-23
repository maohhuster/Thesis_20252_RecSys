#!/usr/bin/env bash
# Package everything needed to run R1-plus on full ML-20M on a CUDA server.
#
# Output: ml20m_r1plus_server.tar.gz (~160 MB; mostly the prepared ml20m data)
#
# What's bundled:
#   1. Patched RLMRec repository (encoder code + plus-config with patience=10)
#   2. Prepared ML-20M data at data/ml20m/ (Claude-Haiku-4.5 + bge-large-en-v1.5)
#   3. Server orchestrator (scripts/run_ml20m_r1plus_server.py)
#   4. Eval script (scripts/eval_ml20m_r1plus.py)
#   5. Reference splits at code/benchmark/data/processed/ for evaluation
#   6. requirements.txt for the conda/venv environment
#
# On the server:
#   tar -xzf ml20m_r1plus_server.tar.gz && cd ml20m_r1plus_bundle
#   bash setup.sh                                   # creates venv, installs deps
#   nohup python3 scripts/run_ml20m_r1plus_server.py > /tmp/r1plus.log 2>&1 &
#   tail -f /tmp/r1plus.log
#
# Usage on local Mac:
#   bash scripts/package_ml20m_r1plus_server.sh

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="$REPO/ml20m_r1plus_bundle"
TARBALL="$REPO/ml20m_r1plus_server.tar.gz"

echo "═══ Packaging R1-plus ML-20M server bundle ═══"

rm -rf "$BUNDLE"
mkdir -p "$BUNDLE"

# 1. Patched RLMRec repo (encoder + config + trainer + data_utils + models)
#    Exclude only the data subdirs we don't need (amazon, ml1m_ours, etc.)
echo "  [1/5] Patched RLMRec repo (excluding non-ml20m datasets)"
mkdir -p "$BUNDLE/code/benchmark/external/RLMRec"
rsync -a \
    --exclude='data/amazon/' \
    --exclude='data/amazon_ours/' \
    --exclude='data/ml1m_ours/' \
    --exclude='data/ml20m_sub163_ours/' \
    --exclude='data/yelp/' \
    --exclude='data/steam/' \
    --exclude='encoder/checkpoint/' \
    --exclude='encoder/log/' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    "$REPO/code/benchmark/external/RLMRec/" \
    "$BUNDLE/code/benchmark/external/RLMRec/"

# 2. Prepared ML-20M data is included by the rsync above (data/ml20m/)
echo "  [2/5] Verifying ml20m data"
for f in itm_emb_np.pkl usr_emb_np.pkl trn_mat.pkl val_mat.pkl tst_mat.pkl; do
    test -f "$BUNDLE/code/benchmark/external/RLMRec/data/ml20m/$f" \
        || { echo "    MISSING: $f"; exit 1; }
done
echo "    OK"

# 3. Server orchestrator + eval script + helpers
echo "  [3/5] Scripts"
mkdir -p "$BUNDLE/scripts"
cp "$REPO/scripts/run_ml20m_r1plus_server.py" "$BUNDLE/scripts/"
cp "$REPO/scripts/eval_ml20m_r1plus.py"       "$BUNDLE/scripts/"
chmod +x "$BUNDLE/scripts"/*.py

# 4. Reference splits + benchmark code path that eval_ml20m_r1plus.py imports from
echo "  [4/5] Benchmark splits + eval support code"
mkdir -p "$BUNDLE/code/benchmark/data/processed"
cp "$REPO/code/benchmark/data/processed/train.csv" "$BUNDLE/code/benchmark/data/processed/" 2>/dev/null || true
cp "$REPO/code/benchmark/data/processed/val.csv"   "$BUNDLE/code/benchmark/data/processed/" 2>/dev/null || true
cp "$REPO/code/benchmark/data/processed/test.csv"  "$BUNDLE/code/benchmark/data/processed/" 2>/dev/null || true
cp "$REPO/code/benchmark/data/processed/stats.json" "$BUNDLE/code/benchmark/data/processed/" 2>/dev/null || true
cp "$REPO/code/benchmark/data/processed/item_map.json" "$BUNDLE/code/benchmark/data/processed/" 2>/dev/null || true

# Copy the support modules eval_ml20m_r1plus.py imports
mkdir -p "$BUNDLE/code/benchmark/data"
cp "$REPO/code/benchmark/data/dataset.py" "$BUNDLE/code/benchmark/data/" 2>/dev/null || true
cp "$REPO/code/benchmark/data/__init__.py" "$BUNDLE/code/benchmark/data/" 2>/dev/null || true
cp "$REPO/code/benchmark/evaluate.py" "$BUNDLE/code/benchmark/" 2>/dev/null || true
cp "$REPO/code/benchmark/config.py" "$BUNDLE/code/benchmark/" 2>/dev/null || true

# 5. requirements + setup script
echo "  [5/5] requirements + setup script"
if [ -f "$REPO/code/benchmark/requirements.txt" ]; then
    cp "$REPO/code/benchmark/requirements.txt" "$BUNDLE/"
else
    cat > "$BUNDLE/requirements.txt" <<'EOF'
# Minimal dependencies for R1-plus ML-20M server run
torch>=2.0
numpy>=1.24
scipy>=1.10
pandas>=2.0
pyyaml>=6.0
tqdm>=4.65
scikit-learn>=1.3
EOF
fi

cat > "$BUNDLE/setup.sh" <<'EOF'
#!/usr/bin/env bash
# One-shot env setup on the server.
# Assumes: Python 3.10+, CUDA toolkit + matching torch wheel.
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "Environment ready. Activate with: source .venv/bin/activate"
echo "Then launch: nohup python3 scripts/run_ml20m_r1plus_server.py > /tmp/r1plus.log 2>&1 &"
EOF
chmod +x "$BUNDLE/setup.sh"

# README for the server operator
cat > "$BUNDLE/README.md" <<'EOF'
# R1-plus on ML-20M — server bundle

## What this runs
RLMRec-plus (contrastive distillation) on the full LLM-MovieLens ML-20M dataset
(127K users × 9.9K items × 11.5M training interactions; Claude-Haiku-4.5 +
bge-large-en-v1.5 item/user embeddings already prepared).

This is the dense-end cell of the regularizer-class density triangulation.
Expected outcome: R1-plus lands within seed noise of R1-gene (which shows
-2.2% vs M7 at this density). 5 seeds: 42, 123, 456, 789, 2026.

## Quick start

```bash
bash setup.sh                                   # creates venv, installs deps
source .venv/bin/activate
nohup python3 scripts/run_ml20m_r1plus_server.py > /tmp/r1plus.log 2>&1 &
tail -f /tmp/r1plus.log
```

Smoke test (single seed only, will run to convergence):
```bash
python3 scripts/run_ml20m_r1plus_server.py --seeds 42
```

## Hardware

- Recommended: A100 (40 GB), L4 (24 GB), or A40 (48 GB)
- Minimum: 24 GB VRAM (RTX 3090, RTX 4090, V100-32G also work)
- If OOM: lower `train.batch_size` in `code/benchmark/external/RLMRec/encoder/config/modelconf/lightgcn_plus.yml` from 4096 → 2048 (linearly slower but ~half memory)

## Estimated wall time per seed

| GPU | est. min/seed | est. 5 seeds |
|---|---|---|
| A100 40GB | 30–60 min | 3–5 h |
| L4 24GB   | 60–120 min | 6–10 h |
| RTX 4090  | 45–90 min  | 4–7 h |

Patience bumped from upstream-default 5 to 10 in lightgcn_plus.yml `train.patience`
to give ML-20M's denser graph fair convergence; ML-1M and Amazon-Books
already converged at the old patience so re-running them is unnecessary.

## Output

After all 5 seeds complete:

```bash
python3 scripts/eval_ml20m_r1plus.py --device cuda
```

Results land in:
- `code/benchmark/results/r1plus/ml20m/seed-{42,123,456,789,2026}-marker.json`
- `code/benchmark/results/r1plus_metrics.json`  (5-seed aggregate)
- `code/benchmark/checkpoints/r1plus/bge-large-en-v1.5/seed-*/best_model.pt`

Send back the metrics JSON + log files; checkpoints can stay on the server.

## Idempotency

The orchestrator skips any seed whose marker JSON already exists. Safe to
re-launch after interruption — just re-run the same nohup command.
EOF

# Make the tarball
echo ""
echo "═══ Creating tarball ═══"
cd "$REPO"
tar -czf "$TARBALL" -C "$REPO" "ml20m_r1plus_bundle"

echo ""
echo "═══ Done ═══"
echo "  Bundle dir : $BUNDLE/"
echo "  Tarball    : $TARBALL ($(du -h "$TARBALL" | cut -f1))"
echo ""
echo "  Transfer to server:"
echo "    scp \"$TARBALL\" user@server:/path/to/work/"
echo ""
echo "  Or via rsync (resumable):"
echo "    rsync -avzP \"$TARBALL\" user@server:/path/to/work/"
