# Extend Research

This directory is a standalone workspace for follow-up experiments built on top of
the MovieLens recommendation benchmark.

## Structure

- `src/extend_research/`: reusable Python package code
- `configs/`: experiment configuration files
- `scripts/`: command-line entry points for setup and experiments
- `tests/`: automated tests
- `notebooks/`: exploratory notebooks
- `data/raw/`: source datasets, kept out of git by default
- `data/processed/`: derived datasets, kept out of git by default
- `results/`: experiment outputs, kept out of git by default
- `docs/`: project notes and research documentation

## Setup

```bash
cd extend_research
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

For a minimal install:

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 scripts/run_experiment.py --config configs/default.json
```

Run the DTS-v1 aggregate diagnostic:

```bash
python3 scripts/analyze_dts_v1.py --config configs/dts_v1.json
```

The tracked result summary is in `docs/DTS_V1_Initial_Results.md`.

Run the current best blend-router experiment:

```bash
make v2-2-router
```

The tracked result summary is in `docs/V2_2_Dataset_Global_Blend.md`.

Analyze where V2.2 helps:

```bash
make v2-2-segments
```

The tracked segment summary is in `docs/V2_2_Segment_Analysis.md`.

## Test

```bash
pytest
```
