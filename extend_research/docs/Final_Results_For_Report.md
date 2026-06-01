# Final Results For Report

Date: 2026-06-01

## Final Position

The final main method should be **V2.2 dataset-level global blend**.

The threshold-selector direction should be closed as a main method. It remains
useful as an exploratory baseline and motivation, but the evidence now supports
soft expert composition rather than hard expert selection.

Main message:

> A simple dataset-level soft blend of frozen M7, R1, and R1-plus expert scores
> provides the strongest deployable result. It improves ML-20M and Amazon over
> the best fixed expert, remains neutral on ML-1M, and does not require retraining
> the base recommenders.

## System Setup

The experiments use frozen expert checkpoints and cached expert scores:

```mermaid
flowchart LR
    A["Datasets<br/>ML-20M, ML-1M, Amazon"] --> B["Frozen experts<br/>M7, R1, R1-plus"]
    B --> C["Candidate score cache"]
    C --> D["Meta layer<br/>selector / router / blend"]
    D --> E["Evaluation<br/>NDCG@10, Recall@10, MRR"]
    E --> F["Analysis<br/>bootstrap, LOSO, segments, oracle gap"]
```

The base experts are not retrained. All V1/V2/V3/V4 experiments operate above the
cached expert-score layer.

## Main Result: V2.2 Global Blend

V2.2 selects one blend weight per dataset on validation caches and applies the
same weight to every test user:

`score(u, i) = w_M7 * s_M7(u, i) + w_R1 * s_R1(u, i) + w_R1plus * s_R1plus(u, i)`

| Dataset | Weight `(M7, R1, R1-plus)` | V2.2 NDCG@10 | Best Expert NDCG@10 | Delta |
|---|---|---:|---:|---:|
| ML-20M | `(0.70, 0.15, 0.15)` | 0.120126 | 0.117462 | +0.002664 |
| ML-1M | `(0.00, 0.60, 0.40)` | 0.195024 | 0.194915 | +0.000110 |
| Amazon | `(0.00, 0.60, 0.40)` | 0.076577 | 0.072953 | +0.003624 |

Interpretation:

- ML-20M benefits from an M7-dominant blend with small R1/R1-plus contributions.
- Amazon benefits from an R1/R1-plus blend.
- ML-1M should be treated as neutral, not as a strong improvement.

## Reliability Evidence

### V2.3 Bootstrap

| Dataset | Delta NDCG@10 | 95% CI | P(delta > 0) | Claim |
|---|---:|---|---:|---|
| ML-20M | +0.002664 | [0.001754, 0.003582] | 1.0000 | reliable gain |
| Amazon | +0.003624 | [0.003015, 0.004261] | 1.0000 | reliable gain |
| ML-1M | +0.000110 | [-0.001237, 0.001449] | 0.5688 | neutral |

### V2.5 Leave-One-Seed-Out

| Dataset | LOSO NDCG@10 | Best Expert | Delta | Difference vs V2.2 |
|---|---:|---:|---:|---:|
| Amazon | 0.076577 | 0.072953 | +0.003624 | +0.000000 |
| ML-20M | 0.119700 | 0.117462 | +0.002238 | -0.000426 |
| ML-1M | 0.194945 | 0.194915 | +0.000030 | -0.000079 |

Reliability conclusion:

- ML-20M and Amazon are the two datasets where the claim is strong.
- Amazon is especially stable: LOSO keeps the same global blend.
- ML-1M is approximately unchanged.

## Mechanism Evidence

V2.2 segment analysis explains where the gain comes from.

| Dataset | Segment | Delta NDCG@10 |
|---|---|---:|
| ML-20M | sparse users | +0.004819 |
| ML-20M | medium users | +0.002019 |
| ML-20M | dense users | +0.000455 |
| Amazon | sparse users | +0.004221 |
| Amazon | hot items | +0.005319 |
| Amazon | medium expert disagreement | +0.004093 |

Mechanism conclusion:

- ML-20M gains mainly from sparse-user ranking.
- Amazon gains from complementary R1/R1-plus signal.
- The improvement is better explained as soft composition than as hard threshold
  selection.

## Closed Direction: Threshold Selector

The DTS/V1 threshold-selector direction should be closed as the main method.

Reason:

- Hard threshold selection chooses one expert and discards useful secondary signal.
- It is sensitive to boundary choices and segment definitions.
- Later V2.2 results show that soft blending is more stable.
- V3/V4 diagnostics further show that adding adaptive decisions without stronger
  features can reduce performance.

How to write it in the report:

> Threshold-based selection was useful as an exploratory diagnostic: it showed
> that expert behavior varies across datasets and user/item regimes. However, the
> final method uses soft expert composition, because hard routing was less stable
> than a dataset-level blend.

## Adaptive Router Diagnostics

### Oracle Headroom

V2.4 shows that adaptive routing has theoretical room:

| Dataset | V2.2 Blend | Oracle All | Gap |
|---|---:|---:|---:|
| ML-20M | 0.120126 | 0.157996 | +0.037870 |
| ML-1M | 0.195024 | 0.245488 | +0.050463 |
| Amazon | 0.076577 | 0.107637 | +0.031060 |

But learned routers do not recover this oracle gap.

| Dataset | V2.2 Blend | V3 Action | V3.1 Gain | Best V4 Segment |
|---|---:|---:|---:|---:|
| ML-20M | 0.120126 | 0.119572 | 0.116966 | 0.119249 |
| ML-1M | 0.195024 | 0.188125 | 0.192643 | 0.193918 |
| Amazon | 0.076577 | 0.076529 | 0.076188 | 0.076611 |

Adaptive-router conclusion:

- The oracle gap is real but not yet deployable.
- V3/V3.1 per-user routers are too noisy.
- V4 segment-level blend is more conservative, but still does not consistently
  beat V2.2.
- These results should be reported as diagnostics and future work.

## Recommended Report Structure

1. Introduce the frozen expert setting: M7, R1, R1-plus are already trained.
2. Show why fixed expert choice is dataset-dependent.
3. Present threshold selector as an exploratory baseline, then close it.
4. Present V2.2 dataset-level global blend as the main method.
5. Report V2.2 main table.
6. Add bootstrap and LOSO robustness.
7. Add segment analysis as explanation.
8. Add oracle/V3/V4 as future-work diagnostics.

## Claims To Use

Strong claims:

- V2.2 significantly improves ML-20M and Amazon over the best fixed expert.
- V2.2 does not require retraining M7, R1, or R1-plus.
- Amazon's selected R1/R1-plus blend is robust across seeds.
- ML-20M benefits from M7-dominant blending, especially for sparse users.

Careful claims:

- ML-1M is neutral rather than significantly improved.
- Adaptive routing has oracle headroom but current learned routers do not recover
  it.
- V4 only gives a tiny Amazon gain and should not replace V2.2.

Avoid:

- claiming universal improvement across all datasets;
- claiming V3/V4 as the main method;
- claiming cold-item improvement unless the segment evidence is strengthened;
- suggesting that M7/R1/R1-plus were retrained for these experiments.

## Final Recommendation

Stop optimizing threshold selector. Keep it as motivation.

Report V2.2 as the final method, supported by V2.3, V2.5, and segment analysis.
Use V2.4/V3/V4 to explain why adaptive routing is promising but non-trivial.

