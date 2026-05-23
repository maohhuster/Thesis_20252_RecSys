# LLM-MovieLens Benchmark — Makefile
# One-command reproduction for CIKM 2026 Full Research Track
#
# Usage:
#   make install          Install package + dependencies
#   make download-data    Download ML-20M + HuggingFace features
#   make reproduce        Full reproduction (all configs × 5 seeds)
#   make results-table    Generate LaTeX results table
#   make test             Run unit tests
#   make lint             Run linters

PYTHON := python3
SEEDS := 42 123 456 789 2026
TIER1_CONFIGS := M0 M1 M1b M1c M1d
TIER2_CONFIGS := M2 M3 M4 M5 M6 M7 M8 M9
TIER3_CONFIGS := R1 R1-plus R2 R3
ALL_CONFIGS := $(TIER1_CONFIGS) $(TIER2_CONFIGS) $(TIER3_CONFIGS)

.PHONY: install download-data preprocess reproduce reproduce-tier1 reproduce-tier2 reproduce-tier3 results-table cold-start test lint clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────────

install:  ## Install package in editable mode with dev dependencies
	pip install -e ".[dev]"

download-data:  ## Download MovieLens 20M and pre-computed features
	bash scripts/download_ml20m.sh
	huggingface-cli download anonyauthor4review/llm-movielens --local-dir data/

preprocess:  ## Preprocess ML-20M into temporal train/val/test splits
	$(PYTHON) -m benchmark.data.preprocess

# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE STAGES (for generating features from scratch)
# ──────────────────────────────────────────────────────────────────────────────

generate-profiles:  ## Stage 1: Generate LLM profiles (requires ANTHROPIC_API_KEY)
	$(PYTHON) -m profile_generator.main

generate-embeddings:  ## Stage 2: Generate embeddings from profiles
	$(PYTHON) -m embedding_generator.main

generate-bert-baseline:  ## Generate BERT title baseline embeddings
	$(PYTHON) -m benchmark.features.bert_baseline

# ──────────────────────────────────────────────────────────────────────────────
# BENCHMARK REPRODUCTION
# ──────────────────────────────────────────────────────────────────────────────

reproduce: reproduce-tier1 reproduce-tier2 reproduce-tier3 results-table cold-start  ## Full reproduction: all tiers + results

reproduce-tier1:  ## Run Tier 1: Pure CF baselines (M0, M1, M1b-d)
	@for config in $(TIER1_CONFIGS); do \
		for seed in $(SEEDS); do \
			echo "=== Running $$config seed=$$seed ==="; \
			$(PYTHON) -m benchmark.run_experiment --config $$config --seed $$seed; \
		done; \
	done

reproduce-tier2:  ## Run Tier 2: Content-augmented (M2-M9)
	@for config in $(TIER2_CONFIGS); do \
		for seed in $(SEEDS); do \
			echo "=== Running $$config seed=$$seed ==="; \
			$(PYTHON) -m benchmark.run_experiment --config $$config --seed $$seed; \
		done; \
	done

reproduce-tier3:  ## Run Tier 3: LLM-for-RecSys methods (R1 RLMRec-gene, R1-plus RLMRec-plus, R2 KAR-style, R3 HypernetReplacer)
	@for config in $(TIER3_CONFIGS); do \
		for seed in $(SEEDS); do \
			echo "=== Running $$config seed=$$seed ==="; \
			$(PYTHON) -m benchmark.run_experiment --config $$config --seed $$seed; \
		done; \
	done

run-%:  ## Run a single config with all seeds (e.g., make run-M4)
	@for seed in $(SEEDS); do \
		echo "=== Running $* seed=$$seed ==="; \
		$(PYTHON) -m benchmark.run_experiment --config $* --seed $$seed; \
	done

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

results-table:  ## Generate aggregated results CSV and LaTeX table
	$(PYTHON) scripts/export_results_table.py

cold-start:  ## Run cold-start evaluation on saved checkpoints
	$(PYTHON) scripts/run_cold_start_5seeds.py

# ──────────────────────────────────────────────────────────────────────────────
# QUALITY
# ──────────────────────────────────────────────────────────────────────────────

test:  ## Run unit tests
	pytest tests/ -v --tb=short

lint:  ## Run linters (ruff + mypy)
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

format:  ## Auto-format code
	ruff format src/ scripts/ tests/

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
