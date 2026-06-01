# V3.1 Pairwise Gain Router

Date: 2026-06-01

## Purpose

V3.0 failed because four-way oracle classification was too noisy. V3.1 tests a safer
design:

- keep V2.2 global blend as the default action;
- train one Ridge regressor per expert to predict:

`expert_NDCG@10 - global_blend_NDCG@10`

- switch from global blend to an expert only when the predicted gain exceeds a
validation-selected threshold.

This is a pairwise gain router, not a hard oracle classifier.

## Summary

| Dataset | V2.2 Blend | V3.1 Router | Delta vs V2.2 | Oracle All | Oracle Gap Recovered |
|---|---:|---:|---:|---:|---:|
| ML-20M | 0.120126 | 0.116966 | -0.003160 | 0.157996 | -0.0835 |
| ML-1M | 0.195024 | 0.192643 | -0.002381 | 0.245488 | -0.0469 |
| Amazon | 0.076577 | 0.076188 | -0.000389 | 0.107637 | -0.0126 |

## Interpretation

V3.1 also does not beat V2.2.

It improves over the V3.0 failure on ML-1M, but it is still worse than simply using
the dataset-level global blend. The most damaging case is ML-20M, where the gain
regressor switches away from global blend too often and loses about `0.00316`
NDCG@10.

## What This Teaches

The oracle gap is real, but validation-estimated per-user gain is not stable enough
with the current feature set. The model often predicts positive gain where the test
split does not support it.

This means the problem is not merely "use regression instead of classification."
The deeper issue is that the available user/candidate features do not reliably
predict held-out action gains.

## Current Best Method

The best deployable method remains:

**V2.2 dataset-level global blend**

Do not use V3.0 or V3.1 as main methods. They are diagnostic experiments showing
that adaptive routing is harder than the oracle gap suggests.

## Recommended Next Step

Stop adding increasingly flexible user-level routers for now. The next productive
step is to consolidate the research story:

1. Use V2.2 as the proposed method.
2. Use V2.3 bootstrap to support reliability on ML-20M and Amazon.
3. Use V2.2 segment analysis to explain where gains come from.
4. Use V2.4 oracle gap and V3 negative results to frame future work.

If another adaptive experiment is still needed, prefer a very coarse segment-level
policy rather than per-user learned routing.

## Output Files

- `extend_research/results/v3_1_pairwise_gain_router/v3_1_pairwise_gain_summary.csv`
- `extend_research/results/v3_1_pairwise_gain_router/v3_1_pairwise_gain_by_seed.csv`
- `extend_research/results/v3_1_pairwise_gain_router/v3_1_pairwise_gain_action_counts.csv`
- `extend_research/results/v3_1_pairwise_gain_router/V3_1_Pairwise_Gain_Router.md`
