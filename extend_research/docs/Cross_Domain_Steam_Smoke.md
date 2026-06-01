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

At the time this smoke plan was written, no Steam raw data was present locally.
Therefore the script is expected to report `missing_raw_data` until the raw files
are placed under `data/raw/cross_domain/steam/`.

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

- game metadata file found;
- user/item interaction file found;
- non-zero users/items/interactions;
- metadata title coverage is high;
- at least three plausible density regimes can be produced by k-core or
  interaction-per-item filtering.

## Next Step After Success

If Steam passes smoke:

1. define dense / medium / sparse subsets;
2. create temporal train/val/test split;
3. export normalized `interactions` and `items` tables;
4. only then build candidate score caches and run the cross-domain density matrix.

