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

Read the current experiment narrative and recommended reporting path:

```text
docs/Experiment_Summary.md
docs/Experiment_Overview_Architecture.md
docs/Final_Results_For_Report.md
docs/Next_Research_Strategy.md
docs/Cross_Domain_Readiness.md
```

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

Run paired user-level bootstrap significance for V2.2:

```bash
make v2-3-bootstrap
```

The tracked significance summary is in `docs/V2_3_Bootstrap_Significance.md`.

Estimate the remaining oracle headroom for a V3 router:

```bash
make v2-4-oracle
```

The tracked oracle-gap summary is in `docs/V2_4_Oracle_Gap.md`.

Run leave-one-seed-out global blend robustness:

```bash
make v2-5-loso
```

The tracked LOSO summary is in `docs/V2_5_LOSO_Global_Blend.md`.

Run the first conservative V3 action router:

```bash
make v3-router
```

The tracked V3 diagnostic summary is in `docs/V3_Action_Router_Initial_Result.md`.

Run the pairwise gain-router diagnostic:

```bash
make v3-1-router
```

The tracked V3.1 diagnostic summary is in `docs/V3_1_Pairwise_Gain_Router.md`.

Run deployable segment-level blend diagnostics:

```bash
make v4-segment-blend
```

The tracked V4 diagnostic summary is in `docs/V4_Segment_Level_Blend.md`.

Inspect Steam cross-domain readiness:

```bash
make cross-domain-steam-readiness
```

The tracked Steam smoke plan is in `docs/Cross_Domain_Steam_Smoke.md`.

Export Steam normalized tables and density regimes:

```bash
make cross-domain-steam-export
```

The tracked Steam density-design note is in `docs/Cross_Domain_Steam_Density_Design.md`.

## Test

```bash
pytest
```
