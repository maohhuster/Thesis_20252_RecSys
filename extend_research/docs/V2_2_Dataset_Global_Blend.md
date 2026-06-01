# V2.2 Dataset-Level Global Blend

Date: 2026-06-01

## Purpose

V2.1 showed that per-seed validation-selected weights can overfit, especially on
ML-1M. V2.2 removes that degree of freedom:

- select one blend weight per dataset by averaging validation metrics across all
  available seeds;
- evaluate that same weight on every test seed.

This is a cleaner research claim than per-checkpoint tuning because the learned
object is a dataset-level expert composition.

## Design

V2.2 reuses the refined V2.1 grids:

- ML-20M: M7-heavy grid with step `0.05`.
- ML-1M: R1/R1-plus grid with step `0.05`.
- Amazon: R1/R1-plus grid with step `0.05`.

Selection objective:

`argmax_weight mean_seed(validation NDCG@10)`

Tie-breakers use validation Recall@10 and MRR.

## Selected Weights

| Dataset | Selected Weight `(M7, R1, R1-plus)` | Mean Val NDCG@10 |
|---|---|---:|
| ML-20M | `(0.70, 0.15, 0.15)` | 0.072707 |
| ML-1M | `(0.00, 0.60, 0.40)` | 0.136913 |
| Amazon | `(0.00, 0.60, 0.40)` | 0.077148 |

## Test Summary

| Dataset | Global Blend NDCG@10 | Best Expert NDCG@10 | Delta |
|---|---:|---:|---:|
| ML-20M | 0.120126 | 0.117462 | +0.002664 |
| ML-1M | 0.195024 | 0.194915 | +0.000110 |
| Amazon | 0.076577 | 0.072953 | +0.003624 |

## Comparison Across V2 Variants

| Dataset | V2 Coarse | V2.1 Per-Seed Refined | V2.2 Dataset Global | Best Current |
|---|---:|---:|---:|---|
| ML-20M | 0.118593 | 0.119110 | 0.120126 | V2.2 |
| ML-1M | 0.194793 | 0.194701 | 0.195024 | V2.2 |
| Amazon | 0.076577 | 0.076538 | 0.076577 | V2 / V2.2 tie |

## Interpretation

### ML-20M

This is the strongest improvement from V2.2. The global blend
`(M7=0.70, R1=0.15, R1-plus=0.15)` beats M7 on every seed:

- Seed 42: `+0.002866`
- Seed 123: `+0.003416`
- Seed 456: `+0.001433`
- Seed 789: `+0.003188`
- Seed 2026: `+0.002416`

This is a much better story than V1b and V2.1. It says ML-20M should not be routed
hard to M7 only; M7 should remain dominant, but both rerankers contribute useful
secondary signal.

### ML-1M

V2.2 fixes the main weakness of V2.1. A single dataset-level blend
`(0.00, 0.60, 0.40)` slightly beats the best fixed expert on average:

`0.195024` vs `0.194915`.

The gain is small, so ML-1M should be framed as stability evidence rather than the
headline result. Still, it is important: global selection avoids the negative mean
seen in V2.1.

### Amazon

Amazon remains robust. The same global blend `(0.00, 0.60, 0.40)` matches the V2
coarse result and beats the best expert by `+0.003624`.

This is still the cleanest dataset for the claim that R1 and R1-plus contain
complementary ranking signal.

## Current Research Position

The best current method is V2.2 dataset-level global blend.

Recommended claim:

> A lightweight dataset-level soft composition of M7, R1, and R1-plus consistently
> improves over the best fixed expert on available datasets, without retraining the
> base recommenders.

Recommended framing by dataset:

- ML-20M: main evidence for M7-dominant expert composition.
- Amazon: main evidence for complementary R1/R1-plus semantics.
- ML-1M: stability check with small positive gain.

## Next Experiment

The next step should not be another hand-tuned blend grid. The next useful step is
to explain *where* the global blend helps:

1. Segment users/items by item degree, user history length, and hit overlap between
   experts.
2. Compare global blend vs best expert inside each segment.
3. Identify whether gains come from cold items, warm items, sparse users, dense
   users, or expert-disagreement cases.

That analysis can turn the current result from "the blend works" into "why the
blend works."

## Output Files

- `extend_research/results/v2_2_router/v2_2_selected_weights.csv`
- `extend_research/results/v2_2_router/v2_2_router_summary.csv`
- `extend_research/results/v2_2_router/v2_2_router_by_seed.csv`
- `extend_research/results/v2_2_router/V2_2_Router_Results.md`
