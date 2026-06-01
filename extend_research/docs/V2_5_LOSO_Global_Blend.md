# V2.5 Leave-One-Seed-Out Global Blend

Date: 2026-06-01

## Purpose

V2.2 selects one dataset-level blend weight using validation metrics averaged across
all seeds, then evaluates that weight on all test seeds. V2.5 checks whether this
choice is robust across seeds.

For each held-out seed:

1. remove that seed from weight selection;
2. select the global blend weight using validation caches from the remaining four
   seeds;
3. evaluate the selected weight on the held-out seed's test cache.

This is a stronger robustness check because the evaluated seed does not participate
in weight selection.

## Summary

| Dataset | LOSO NDCG@10 | Best Expert | Delta | V2.2 NDCG@10 | Difference vs V2.2 |
|---|---:|---:|---:|---:|---:|
| Amazon | 0.076577 | 0.072953 | +0.003624 | 0.076577 | +0.000000 |
| ML-20M | 0.119700 | 0.117462 | +0.002238 | 0.120126 | -0.000426 |
| ML-1M | 0.194945 | 0.194915 | +0.000030 | 0.195024 | -0.000079 |

## Interpretation

### Amazon

Amazon is perfectly stable under leave-one-seed-out selection. Every held-out run
selects the same weight:

`(M7=0.00, R1=0.60, R1-plus=0.40)`

The LOSO result exactly matches V2.2:

`0.076577`

This strongly supports the Amazon claim. The R1/R1-plus complementary blend is not
an artifact of one seed.

### ML-20M

ML-20M remains clearly positive:

`0.119700` vs best expert `0.117462`, delta `+0.002238`

The selected weights vary across held-out folds:

- `(0.70, 0.15, 0.15)`
- `(0.90, 0.00, 0.10)`
- `(0.95, 0.00, 0.05)`
- `(0.65, 0.20, 0.15)`

This means ML-20M's exact optimum is less stable than Amazon's, but the important
pattern is stable: M7 remains dominant, and a small amount of reranker signal helps.

### ML-1M

ML-1M remains neutral:

`0.194945` vs best expert `0.194915`, delta `+0.000030`

The selected weights are close variants of R1/R1-plus only:

- `(0.00, 0.60, 0.40)`
- `(0.00, 0.65, 0.35)`

This is consistent with previous experiments: ML-1M should be reported as stable but
not significantly improved.

## Research Implication

V2.5 strengthens the V2.2 story:

1. Amazon's blend is seed-stable.
2. ML-20M's gain survives leave-one-seed-out selection.
3. ML-1M remains neutral, consistent with bootstrap analysis.

Recommended wording:

> Leave-one-seed-out selection confirms that the global blend gains are not driven by
> selecting weights on the same random seed used for testing. Amazon is fully stable,
> while ML-20M remains positive with M7-dominant weights.

## Output Files

- `extend_research/results/v2_5_loso_global_blend/v2_5_loso_summary.csv`
- `extend_research/results/v2_5_loso_global_blend/v2_5_loso_by_seed.csv`
- `extend_research/results/v2_5_loso_global_blend/V2_5_LOSO_Global_Blend.md`
