# V2.1 Refined Guarded Blend Router

Date: 2026-06-01

## Purpose

V2 showed that a soft blend over cached M7, R1, and R1-plus scores is more promising
than a hard density threshold router. V2.1 tests two follow-ups:

1. Refine the blend grid around the best V2 regions.
2. Add a validation-gain guard that falls back to the best fixed expert unless the
   blend improves validation NDCG@10 by a minimum margin.

This experiment still does not retrain M7, R1, or R1-plus. It only reranks cached
top-100 expert candidates.

## Design

Dataset-specific grids:

- ML-20M: M7-heavy region, `M7=0.60..1.00`, `R1=0.00..0.40`,
  `R1-plus=0.00..0.40`, step `0.05`.
- ML-1M: R1/R1-plus region, `M7=0.00`, `R1=0.35..1.00`,
  `R1-plus=0.00..0.65`, step `0.05`.
- Amazon: R1/R1-plus region, `M7=0.00`, `R1=0.40..0.80`,
  `R1-plus=0.20..0.60`, step `0.05`.

Guard thresholds tested:

- `0.0`
- `0.0005`
- `0.001`

The guard chooses the refined blend only if:

`blend_val_NDCG@10 - best_validation_expert_val_NDCG@10 >= threshold`

otherwise it falls back to the best validation expert.

## Summary

| Dataset | Best V2.1 Method | NDCG@10 | Best Expert | Delta |
|---|---|---:|---:|---:|
| Amazon | refined / guarded any threshold | 0.076538 | 0.072953 | +0.003585 |
| ML-20M | refined / guarded 0 | 0.119110 | 0.117462 | +0.001648 |
| ML-1M | refined / guarded 0 | 0.194701 | 0.194915 | -0.000214 |

## Comparison To V2

| Dataset | V2 Coarse Blend | V2.1 Refined Blend | Change |
|---|---:|---:|---:|
| Amazon | 0.076577 | 0.076538 | -0.000039 |
| ML-20M | 0.118593 | 0.119110 | +0.000516 |
| ML-1M | 0.194793 | 0.194701 | -0.000092 |

## Interpretation

### Amazon

Amazon remains the strongest positive case. All thresholds choose blend for every
seed because validation gains are large, around `0.0030..0.0048`.

The selected refined weights are still R1/R1-plus only:

- Seed 42: `(0.00, 0.60, 0.40)`
- Seed 123: `(0.00, 0.70, 0.30)`
- Seed 456: `(0.00, 0.70, 0.30)`
- Seed 789: `(0.00, 0.60, 0.40)`
- Seed 2026: `(0.00, 0.55, 0.45)`

The refined grid does not beat the V2 coarse mean, but the difference is tiny. The
main signal remains robust: a soft R1/R1-plus combination is clearly better than the
best fixed expert.

### ML-20M

ML-20M benefits from refinement. The best V2.1 mean is `0.119110`, improving over
the V2 coarse blend mean `0.118593` and the best fixed expert mean `0.117462`.

The strongest single seed is seed 123:

`M7=0.65, R1=0.20, R1-plus=0.15`, test delta `+0.003184` over M7.

The guard is not helpful here if the threshold is too conservative. A threshold of
`0.0005` throws away seed 2026, which actually has a test gain of `+0.001589`
despite only `+0.000280` validation gain. This says ML-20M has useful low-margin
blend signal that a strict guard can mistakenly suppress.

### ML-1M

ML-1M remains the uncomfortable dataset. Validation often prefers a blend, but test
NDCG@10 is slightly below the best fixed expert on average.

This is not a failure of the whole direction, but it is evidence that per-seed
validation tuning can overfit for ML-1M. The current guard thresholds do not solve
that, because the wrong ML-1M blends still have validation gains above `0.001`.

## Conclusion

V2.1 strengthens the story for ML-20M and preserves the Amazon gain, but it also
shows that per-seed validation-selected weights are not enough for ML-1M.

The next better design is V2.2:

1. Select one dataset-level global blend weight by aggregating validation metrics
   across all seeds.
2. Evaluate that single dataset-level weight on all test seeds.
3. Compare against per-seed refined weights.

This should reduce seed-level validation overfit and produce a cleaner paper claim:
"dataset-level soft expert composition" rather than "per-checkpoint hyperparameter
tuning."

## Output Files

- `extend_research/results/v2_1_router/v2_1_router_summary.csv`
- `extend_research/results/v2_1_router/v2_1_router_by_seed.csv`
- `extend_research/results/v2_1_router/V2_1_Router_Results.md`
