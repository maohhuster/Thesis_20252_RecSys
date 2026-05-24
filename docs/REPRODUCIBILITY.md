# Reproducibility Guide

Step-by-step instructions to reproduce all results from the paper.

## Hardware Requirements

| Stage | Minimum | Recommended | Time Estimate |
|-------|---------|-------------|---------------|
| Profile generation | CPU + internet | CPU + internet | ~52 min |
| Embedding generation | 16GB RAM | 32GB RAM + GPU | ~10 min |
| Benchmark (all 75 runs) | 1× GPU (8GB+) | 1× A100 (40GB) | ~24 hours |
| Single experiment | 1× GPU (8GB+) | 1× GPU (16GB) | ~20 min |

## Prerequisites

```bash
# Python 3.10+
python --version  # Should be 3.10+

# Clone and install
git clone https://github.com/anonyauthor4review-png/llm-movielens.git
cd llm-movielens
pip install -e ".[all]"
```

## Option A: Reproduce from Pre-computed Features (Recommended)

Skip stages 1-2 and use our pre-computed embeddings from HuggingFace.

```bash
# 1. Download MovieLens 20M and our features
make download-data

# 2. Preprocess into temporal splits (if not already done)
make preprocess

# 3. Run all experiments (16 model configs × 5 seeds)
make reproduce

# 4. Generate results table
make results-table
```

### Expected Results

After `make reproduce`, you should see results matching Table 4 in the paper (within ±0.002 due to hardware/CUDA non-determinism):

| Config | NDCG@10 (expected) |
|--------|-------------------|
| M0: BPR-MF | ~0.1137 |
| M4: +LLM Profile | ~0.1173 |
| M7: +Profile+Mood | ~0.1175 |
| R1: RLMRec-gene (regularizer) | ~0.1162 |
| R2: KAR-style MoE replacer | ~0.1145 |

## Option B: Full Pipeline Reproduction

Regenerate everything from scratch.

### Stage 1: Profile Generation

Requires API keys:
- `ANTHROPIC_API_KEY` — Claude API access
- `TMDB_API_KEY` — TMDb metadata access

```bash
# Set API keys
export ANTHROPIC_API_KEY="your-key-here"
export TMDB_API_KEY="your-key-here"

# Generate profiles (~$14 API cost, ~52 min)
make generate-profiles
```

**Cost:** ~$14 USD (Claude Haiku 4.5 with prompt caching)

### Stage 2: Embedding Generation

```bash
# Generate all embedding types (~10 min on GPU)
make generate-embeddings
make generate-bert-baseline
```

### Stage 3: Benchmark

```bash
# Full reproduction
make reproduce

# Or run individual tiers
make reproduce-tier1  # Pure CF baselines
make reproduce-tier2  # Content-augmented
make reproduce-tier3  # LLM-for-RecSys methods
```

## Verifying Results

```bash
# Run tests to validate data integrity
make test

# Compare your results against published values
python scripts/export_results_table.py
# Check results/main_results.csv
```

## Random Seeds

All experiments use 5 seeds: `42, 123, 456, 789, 2026` (see `SEEDS` in `code/benchmark/config.py`; `SEEDS[0] = 42` is the default single-run seed). Seeds control:
- PyTorch weight initialization
- Numpy random sampling (negative sampling)
- Data loader shuffling

Note: Full determinism requires `torch.use_deterministic_algorithms(True)`, which we do not enforce due to performance impact. Results may vary by ±0.002 across hardware.

## Known Issues

1. **CUDA non-determinism:** Results may differ slightly across GPU architectures
2. **TMDb API rate limits:** Profile generation may take longer if rate-limited
