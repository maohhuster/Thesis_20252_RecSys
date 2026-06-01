# Extend Research Experiment Summary

Date: 2026-06-01

## Current Position

The strongest deployable method is **V2.2 dataset-level global blend**.

It learns one soft expert composition per dataset on validation caches and applies
that same composition to all users in the dataset:

`score(u, i) = w_M7 * s_M7(u, i) + w_R1 * s_R1(u, i) + w_R1plus * s_R1plus(u, i)`

It does not retrain M7, R1, or R1-plus. It only reuses cached expert scores.

## Main Result

| Dataset | Weight `(M7, R1, R1-plus)` | V2.2 NDCG@10 | Best Expert NDCG@10 | Delta |
|---|---|---:|---:|---:|
| ML-20M | `(0.70, 0.15, 0.15)` | 0.120126 | 0.117462 | +0.002664 |
| ML-1M | `(0.00, 0.60, 0.40)` | 0.195024 | 0.194915 | +0.000110 |
| Amazon | `(0.00, 0.60, 0.40)` | 0.076577 | 0.072953 | +0.003624 |

Interpretation:

- ML-20M is M7-dominant, but R1 and R1-plus add useful secondary signal.
- ML-1M is essentially R1/R1-plus only and should be treated as neutral.
- Amazon clearly benefits from the R1/R1-plus soft blend.

## Significance

V2.3 paired user-level bootstrap supports the main claim on ML-20M and Amazon:

| Dataset | Delta NDCG@10 | 95% CI | P(delta > 0) | Claim |
|---|---:|---|---:|---|
| ML-20M | +0.002664 | [0.001754, 0.003582] | 1.0000 | reliable gain |
| Amazon | +0.003624 | [0.003015, 0.004261] | 1.0000 | reliable gain |
| ML-1M | +0.000110 | [-0.001237, 0.001449] | 0.5688 | neutral |

Recommended wording:

> The dataset-level global blend yields statistically reliable improvements on
> ML-20M and Amazon, while remaining approximately neutral on ML-1M.

## Robustness

V2.5 leave-one-seed-out selection checks whether weights generalize across seeds.

| Dataset | LOSO NDCG@10 | Best Expert | Delta | Difference vs V2.2 |
|---|---:|---:|---:|---:|
| Amazon | 0.076577 | 0.072953 | +0.003624 | +0.000000 |
| ML-20M | 0.119700 | 0.117462 | +0.002238 | -0.000426 |
| ML-1M | 0.194945 | 0.194915 | +0.000030 | -0.000079 |

Interpretation:

- Amazon is fully stable: every held-out fold selects `(0.00, 0.60, 0.40)`.
- ML-20M remains positive even when the held-out seed is not used for weight
  selection.
- ML-1M remains neutral.

## Mechanism

V2.2 segment analysis explains where the global blend helps.

ML-20M:

| Segment | Delta NDCG@10 |
|---|---:|
| sparse users | +0.004819 |
| medium users | +0.002019 |
| dense users | +0.000455 |

Amazon:

| Segment | Delta NDCG@10 |
|---|---:|
| sparse users | +0.004221 |
| hot items | +0.005319 |
| medium expert disagreement | +0.004093 |

Interpretation:

- ML-20M gains mainly from sparse-user ranking.
- Amazon gains from complementary R1/R1-plus semantic signal, especially where
  experts disagree.

## Oracle And Router Diagnostics

V2.4 shows large oracle headroom:

| Dataset | V2.2 Blend | Oracle All | Gap |
|---|---:|---:|---:|
| ML-20M | 0.120126 | 0.157996 | +0.037870 |
| ML-1M | 0.195024 | 0.245488 | +0.050463 |
| Amazon | 0.076577 | 0.107637 | +0.031060 |

However, learned adaptive routers do not exploit that gap:

| Dataset | V2.2 Blend | V3.0 Action Router | V3.1 Gain Router | Best V4 Segment Blend |
|---|---:|---:|---:|---:|
| ML-20M | 0.120126 | 0.119572 | 0.116966 | 0.119249 |
| ML-1M | 0.195024 | 0.188125 | 0.192643 | 0.193918 |
| Amazon | 0.076577 | 0.076529 | 0.076188 | 0.076611 |

Interpretation:

- The oracle gap is real, but per-user oracle labels/gain labels are too noisy for
  the current feature set.
- V3.0, V3.1, and V4 should be treated as diagnostics/future work, not as main
  methods.
- V4 gives only a tiny Amazon gain over V2.2 and reduces ML-20M/ML-1M, so it does
  not replace the global blend.
- The best deployable method remains V2.2.

## Suggested Thesis Narrative

1. Start from the observation that fixed expert choice is dataset-dependent.
2. Propose dataset-level soft expert composition as a lightweight alternative to hard
   expert selection.
3. Show V2.2 improves ML-20M and Amazon without retraining the base recommenders.
4. Use bootstrap and leave-one-seed-out selection to support reliability.
5. Use segment analysis to explain the mechanism.
6. Use oracle and failed V3 routers to argue that adaptive routing is promising but
   non-trivial future work.

## Recommended Claims

Strong claims:

- V2.2 significantly improves ML-20M and Amazon over the best fixed expert.
- Amazon's R1/R1-plus blend is seed-stable.
- ML-20M benefits from M7-dominant blending, especially for sparse users.

Careful claims:

- ML-1M is neutral rather than significantly improved.
- Adaptive routing has oracle headroom but current supervised routers do not recover
  it.

Avoid claiming:

- universal significant improvement across all datasets;
- V3 router improvement;
- cold-item improvement, because current cold/warm segments are weak or sparse.

## Main Files

Core result docs:

- `docs/Final_Results_For_Report.md`
- `docs/V2_2_Dataset_Global_Blend.md`
- `docs/V2_3_Bootstrap_Significance.md`
- `docs/V2_5_LOSO_Global_Blend.md`
- `docs/V2_2_Segment_Analysis.md`

Diagnostic/future-work docs:

- `docs/V2_4_Oracle_Gap.md`
- `docs/V3_Action_Router_Initial_Result.md`
- `docs/V3_1_Pairwise_Gain_Router.md`
- `docs/V4_Segment_Level_Blend.md`

Recommended method to report:

- `make v2-2-router`
- `make v2-3-bootstrap`
- `make v2-5-loso`
- `make v2-2-segments`
