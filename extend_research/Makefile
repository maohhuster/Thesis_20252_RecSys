.PHONY: setup test lint run dts-v1 dts-v1b v2-cache v2-router clean

setup:
	python3 -m pip install --upgrade pip
	python3 -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

run:
	python3 scripts/run_experiment.py --config configs/default.json

dts-v1:
	python3 scripts/analyze_dts_v1.py --config configs/dts_v1.json

dts-v1b:
	python3 scripts/run_dts_v1b_routing.py --config configs/dts_v1b.json

v2-cache:
	python3 scripts/export_v2_candidate_cache.py --config configs/v2_candidate_cache.json

v2-router:
	python3 scripts/train_v2_router.py --config configs/v2_router.json

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
