# V2 Candidate Blend Router Results

Date: 2026-05-31

## Purpose

V2 tests whether a lightweight router can improve over picking one fixed expert per
dataset. Instead of retraining M7, R1, or R1-plus, it reuses their cached top-100
candidate scores and learns/selects a soft convex blend:

`score = w_m7 * score_m7 + w_r1 * score_r1 + w_r1plus * score_r1plus`

Weights are selected on the validation split and evaluated once on the test split.
The current full run uses a coarse grid step of `0.2`.

## Data Coverage

Completed datasets:

- ML-20M: 5 seeds.
- ML-1M: 5 seeds.
- Amazon Books 2018: 5 seeds.

Skipped dataset:

- ML20M-sub163: checkpoint exists, but split/features/cache are still missing.

## Summary

| Dataset | Router NDCG@10 | Best Expert NDCG@10 | Delta | Main Signal |
|---|---:|---:|---:|---|
| Amazon | 0.076577 | 0.072953 | +0.003624 | Strong gain from blending R1 and R1-plus |
| ML-20M | 0.118593 | 0.117462 | +0.001131 | Small but useful gain over M7 |
| ML-1M | 0.194793 | 0.194915 | -0.000121 | Essentially tied with best fixed expert |

## Per-Dataset Interpretation

### Amazon

Amazon is the clearest positive result. All 5 seeds select the same blend:

`M7=0.0, R1=0.6, R1-plus=0.4`

This means the useful signal is not a hard switch between experts. R1 is the
strongest single expert, but R1-plus adds complementary ranking signal when blended
softly. This is the most promising dataset for the next V2 refinement.

### ML-20M

ML-20M mostly remains M7-dominant. Three seeds benefit from adding R1/R1-plus:

- Seed 42: `M7=0.8, R1=0.0, R1-plus=0.2`, delta `+0.001693`.
- Seed 123: `M7=0.8, R1=0.2, R1-plus=0.0`, delta `+0.002111`.
- Seed 456: `M7=0.6, R1=0.2, R1-plus=0.2`, delta `+0.001852`.

Two seeds collapse back to pure M7. This supports the earlier conclusion that ML-20M
has a strong M7 prior, but there is still measurable complementary signal from the
rerankers.

### ML-1M

ML-1M is near the ceiling of the best expert. The blend sometimes improves a seed,
but the mean is slightly below the best fixed expert. The selected weights are
always R1/R1-plus only, with M7 weight zero.

This means ML-1M is useful as a stability check, but it is not the best dataset for
claiming router gains unless the next model can improve seed consistency.

## Conclusion

The hard density threshold selector from V1b was too brittle. V2 soft blending is a
better direction because it can exploit complementary expert scores without making
an irreversible per-user or per-item expert choice.

The best next experiment is V2.1:

1. Refine Amazon blend grid around `(0.0, 0.6, 0.4)` with smaller step, such as
   `0.05`.
2. Refine ML-20M around M7-heavy regions, such as `(0.7-1.0, 0.0-0.3, 0.0-0.3)`.
3. Add a constrained fallback rule: only accept a blend if validation gain is above a
   small threshold; otherwise use the best fixed expert. This should protect ML-1M
   from small validation overfit.
4. After that, test a feature-conditioned router again, but use blend outputs as the
   baseline to beat.

## Output Files

- `extend_research/results/v2_router/v2_router_summary.csv`
- `extend_research/results/v2_router/v2_router_by_seed.csv`
- `extend_research/results/v2_router/V2_Router_Results.md`
