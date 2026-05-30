.PHONY: setup test lint run clean

setup:
	python3 -m pip install --upgrade pip
	python3 -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

run:
	python3 scripts/run_experiment.py --config configs/default.json

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
