# V3 Action Router Initial Result

Date: 2026-06-01

## Purpose

V2.4 showed a large oracle gap, so V3 tests whether a lightweight feature-based
router can recover part of that gap.

V3.0 is deliberately conservative:

- actions: `global_blend`, `m7`, `r1`, `r1plus`;
- train split: validation users;
- test split: test users;
- target label: per-user oracle best action on validation;
- model: multinomial logistic regression with balanced class weights;
- guard: default to `global_blend` unless prediction confidence passes a threshold
  selected on validation.

The baseline to beat is V2.2 global blend, not the best fixed expert.

## Summary

| Dataset | V2.2 Blend | V3 Router | Delta vs V2.2 | Oracle All | Oracle Gap Recovered |
|---|---:|---:|---:|---:|---:|
| ML-20M | 0.120126 | 0.119572 | -0.000553 | 0.157996 | -0.0147 |
| ML-1M | 0.195024 | 0.188125 | -0.006900 | 0.245488 | -0.1369 |
| Amazon | 0.076577 | 0.076529 | -0.000048 | 0.107637 | -0.0015 |

## Interpretation

V3.0 does not beat V2.2. This is a useful negative result.

The oracle gap is real, but the current feature/action classifier does not learn it
reliably from validation users. There are two likely causes:

1. Per-user oracle labels are noisy. Many users have tied or near-tied actions, and
   the validation label can be unstable.
2. The feature set is too weak to predict which action will win on held-out test
   users. This is especially visible on ML-1M, where the validation-trained router
   over-switches away from the global blend and loses badly.

## Dataset Notes

### ML-20M

The router is only slightly worse than V2.2:

`0.119572` vs `0.120126`

It mostly falls back to global blend, but the few switches it makes are not
consistently beneficial. V2.2 remains the better deployable method.

### Amazon

The router is effectively tied but still slightly worse:

`0.076529` vs `0.076577`

The guard is doing its job here: most users stay on global blend. This avoids a
large failure, but it also means V3.0 recovers none of the oracle gap.

### ML-1M

ML-1M is the clear failure case:

`0.188125` vs `0.195024`

The training labels are small and noisy, and the model switches too aggressively.
This reinforces the earlier finding that ML-1M is not a good headline dataset for
router gains unless the router is much better regularized.

## Current Best Method

The current best deployable method remains:

**V2.2 dataset-level global blend**

Recommended claim should not include V3.0 as an improvement. V3.0 should be treated
as diagnostic evidence that naive supervised action routing is not enough.

## Next Direction

The next V3 attempt should avoid hard per-user oracle classification. Better options:

1. **Pairwise gating against V2.2 only**
   - Learn whether switching from global blend to one expert is worth it.
   - This is safer than four-way classification.

2. **Expected-gain regression**
   - Predict `expert_metric - global_blend_metric` for each action.
   - Switch only if predicted gain is positive by a margin.

3. **Segment-level router**
   - Use the segment analysis directly:
     sparse users may use one policy, dense users another.
   - Fewer degrees of freedom, less validation overfit.

4. **Cross-seed training**
   - Train on validation users from multiple seeds together.
   - Evaluate seed by seed.
   - This may reduce noise compared with fitting one small classifier per seed.

The safest next experiment is **V3.1 pairwise gain router**: keep V2.2 as default
and learn only high-confidence switches that improve validation by a margin.

## Output Files

- `extend_research/results/v3_action_router/v3_action_router_summary.csv`
- `extend_research/results/v3_action_router/v3_action_router_by_seed.csv`
- `extend_research/results/v3_action_router/v3_action_router_action_counts.csv`
- `extend_research/results/v3_action_router/V3_Action_Router_Results.md`
