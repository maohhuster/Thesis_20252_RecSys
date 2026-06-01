# Cross-Domain Steam Smoke

Date: 2026-06-01

## Purpose

Steam is the first proposed cross-domain smoke test because it is a recommender
dataset with game metadata, user/item interactions, and a manageable item catalog.

This smoke step does not train M7/R1/R1-plus and does not run the full
cross-domain matrix. It only checks whether raw Steam files can be normalized into
the interaction/item format required by the current cache pipeline.

## Added Assets

- `configs/cross_domain_steam_readiness.json`
- `scripts/inspect_cross_domain_steam.py`
- `make cross-domain-steam-readiness`

The script accepts line-delimited JSON and Python-literal Steam records. It checks
for known UCSD/McAuley Steam file names under:

```text
data/raw/cross_domain/steam/
```

It writes generated reports to:

```text
results/cross_domain_readiness/steam/
```

Generated reports are not committed.

## Expected Raw Files

At minimum:

```text
data/raw/cross_domain/steam/steam_games.json.gz
data/raw/cross_domain/steam/australian_users_items.json.gz
```

Optional but useful:

```text
data/raw/cross_domain/steam/australian_user_reviews.json.gz
data/raw/cross_domain/steam/steam_reviews.json.gz
```

Source:

- UCSD / McAuley datasets: <https://cseweb.ucsd.edu/~jmcauley/datasets.html>
- Steam data directory: <https://mcauleylab.ucsd.edu/public_datasets/data/steam/>

## Current Local Status

Steam raw data has been downloaded locally under:

```text
data/raw/cross_domain/steam/
```

Downloaded files:

```text
australian_user_reviews.json.gz
australian_users_items.json.gz
steam_games.json.gz
```

These files are raw data and are intentionally kept out of git.

Latest smoke status: `ready_for_density_design`.

Observed smoke statistics:

| Metric | Value |
|---|---:|
| game metadata records scanned | 32,135 |
| unique item ids in metadata | 32,132 |
| metadata title coverage | 0.999938 |
| metadata text coverage | 0.999969 |
| sampled interactions | 1,000,000 |
| users in sampled interactions | 13,368 |
| items in sampled interactions | 8,865 |
| sampled density | 0.00843830 |

The current smoke is capped at 1,000,000 interactions by
`configs/cross_domain_steam_readiness.json`, so these are readiness statistics,
not final full-dataset statistics.

## Command

```bash
make cross-domain-steam-readiness
```

Equivalent direct command:

```bash
python3 scripts/inspect_cross_domain_steam.py --config configs/cross_domain_steam_readiness.json
```

## Success Criteria

Steam is ready for density design if the smoke report confirms:

- game metadata file found: **yes**;
- user/item interaction file found: **yes**;
- non-zero users/items/interactions: **yes**;
- metadata title coverage is high: **yes**;
- at least three plausible density regimes can be produced by k-core or
  interaction-per-item filtering: **likely**, based on k-core estimates.

## Next Step After Success

If Steam passes smoke:

1. define dense / medium / sparse subsets;
2. create temporal train/val/test split;
3. export normalized `interactions` and `items` tables;
4. only then build candidate score caches and run the cross-domain density matrix.

Recommended immediate next step:

> Implement Steam density-design/export script that turns the raw files into
> normalized `interactions` and `items` tables for three density regimes.
