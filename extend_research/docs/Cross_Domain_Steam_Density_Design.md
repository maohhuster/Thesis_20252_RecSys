# Cross-Domain Steam Density Design

Date: 2026-06-01

## Purpose

This document tracks the first normalized Steam export for the cross-domain
density-law direction.

The export converts raw Steam files into:

- normalized `interactions.csv`;
- normalized `items.csv`;
- three iterative k-core density regimes: sparse, medium, dense.

## Protocol Note

Steam `australian_users_items` contains playtime but no true interaction
timestamp. `australian_user_reviews` contains review dates, but it is much smaller
and only covers a subset of user-item pairs.

Therefore:

- playtime interactions are the main implicit-feedback source;
- review dates are attached only when the same `(user_id, item_id)` appears in the
  review file;
- the next matrix step must explicitly choose either a review-only temporal split
  or a random/user-level split for the playtime graph.

## Command

```bash
make cross-domain-steam-export
```

Equivalent direct command:

```bash
python3 scripts/export_cross_domain_steam.py --config configs/cross_domain_steam_export.json
```

## Output Locations

Generated normalized tables:

```text
data/processed/cross_domain/steam/
```

Generated run reports:

```text
results/cross_domain_density_design/steam/
```

Both directories are gitignored.

## Current Design

| Regime | User k-core | Item k-core |
|---|---:|---:|
| sparse | 5 | 5 |
| medium | 10 | 10 |
| dense | 20 | 20 |

The default config caps exported interactions at 2,000,000 for the first
manageable density-design pass.

## Export Result

Latest export status: **success**.

| Regime | Users | Items | Interactions | Density | Metadata Coverage | Title Coverage |
|---|---:|---:|---:|---:|---:|---:|
| base | 29,902 | 9,792 | 2,000,000 | 0.00683059 | 0.851307 | 1.000000 |
| sparse | 29,086 | 7,189 | 1,992,520 | 0.00952906 | 0.864376 | 1.000000 |
| medium | 27,593 | 6,089 | 1,974,542 | 0.01175226 | 0.866645 | 1.000000 |
| dense | 24,058 | 4,744 | 1,904,136 | 0.01668375 | 0.865304 | 1.000000 |

Interpretation:

- The three k-core regimes are separable by density, though the current first-pass
  design is still conservative.
- Steam is ready for the next matrix-smoke step.
- Full Steam metadata covers about 85-87% of retained items, but every retained
  item has a usable title because the exporter falls back to `item_name` from the
  user-item records.
- The next design pass may raise the dense regime to a stronger k-core if a wider
  density spread is needed.

## Next Decision

After export, inspect the density table and choose the split policy:

1. review-only temporal split, cleaner but smaller;
2. playtime random split, larger but not temporal;
3. hybrid split, using review timestamps where available and random split
   otherwise.

For the first matrix smoke run, option 2 is likely the fastest, but it must be
reported clearly as non-temporal.

Recommended immediate choice:

> Use the playtime random split for the first smoke run, because it preserves the
> largest graph and validates the cross-domain pipeline quickly. Treat temporal
> evaluation as a later review-subset extension.
