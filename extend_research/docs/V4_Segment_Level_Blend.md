# V4 Segment-Level Blend

Date: 2026-06-01

## Purpose

V4 tests whether a deployable segment-level blend can improve over the V2.2
dataset-level global blend.

Unlike the earlier segment analysis, V4 does **not** use target-popularity or
ground-truth target item information to choose a weight. The segment decision must
be available at ranking time, so V4 uses:

- `user_activity`: sparse / medium / dense user history;
- `expert_disagreement`: low / medium / high disagreement among expert rankings;
- `user_activity_x_expert_disagreement`: the 3x3 cross segment.

Segment thresholds are fitted on validation caches only, then applied unchanged to
test caches. Segments with too few validation users fall back to the V2.2 global
weight.

## Summary

| Dataset | Method | Seeds | NDCG@10 | V2.2 Global | Delta vs V2.2 | Best Expert | Delta vs Best |
|---|---|---:|---:|---:|---:|---:|---:|
| Amazon | `v4_expert_disagreement` | 5 | 0.076611 | 0.076577 | +0.000034 | 0.072953 | +0.003658 |
| Amazon | `v4_user_activity` | 5 | 0.076482 | 0.076577 | -0.000095 | 0.072953 | +0.003529 |
| Amazon | `v4_user_activity_x_expert_disagreement` | 5 | 0.076448 | 0.076577 | -0.000128 | 0.072953 | +0.003496 |
| ML-1M | `v4_expert_disagreement` | 5 | 0.193918 | 0.195024 | -0.001106 | 0.194915 | -0.000996 |
| ML-1M | `v4_user_activity` | 5 | 0.182044 | 0.195024 | -0.012980 | 0.194915 | -0.012870 |
| ML-1M | `v4_user_activity_x_expert_disagreement` | 5 | 0.179437 | 0.195024 | -0.015587 | 0.194915 | -0.015478 |
| ML-20M | `v4_expert_disagreement` | 5 | 0.118895 | 0.120126 | -0.001230 | 0.117462 | +0.001433 |
| ML-20M | `v4_user_activity` | 5 | 0.118708 | 0.120126 | -0.001418 | 0.117462 | +0.001246 |
| ML-20M | `v4_user_activity_x_expert_disagreement` | 5 | 0.119249 | 0.120126 | -0.000877 | 0.117462 | +0.001787 |

## Main Interpretation

V4 does **not** replace V2.2 as the main method.

- Amazon has a tiny positive gain with `expert_disagreement` segmentation
  (`+0.000034` NDCG@10 vs V2.2), but the margin is too small to claim as a robust
  improvement without a significance check.
- ML-20M remains above the best fixed expert, but all V4 segment strategies are
  below the V2.2 global blend.
- ML-1M becomes worse, especially when using `user_activity` or the 3x3 cross
  segment.

This suggests that the V2.2 global blend is already a strong regularized solution.
Segment-level selection adds degrees of freedom, but those extra degrees of freedom
mostly overfit validation segments instead of improving test ranking.

## Selected Segment Weights

### ML-20M

| Strategy | Segment | Val Users | Weights | Fallback |
|---|---|---:|---|---|
| `user_activity` | `dense` | 4490 | `[0.65, 0.15, 0.2]` | False |
| `user_activity` | `medium` | 4475 | `[1.0, 0.0, 0.0]` | False |
| `user_activity` | `sparse` | 4505 | `[0.9, 0.0, 0.1]` | False |
| `expert_disagreement` | `high` | 4490 | `[0.85, 0.0, 0.15]` | False |
| `expert_disagreement` | `low` | 4490 | `[0.65, 0.25, 0.1]` | False |
| `expert_disagreement` | `medium` | 4490 | `[1.0, 0.0, 0.0]` | False |

ML-20M segment weights often become more M7-heavy than V2.2. This helps stay above
the best expert, but loses the smoother benefit of the global `(0.70, 0.15, 0.15)`
blend.

### ML-1M

| Strategy | Segment | Val Users | Weights | Fallback |
|---|---|---:|---|---|
| `user_activity` | `dense` | 810 | `[1.0, 0.0, 0.0]` | False |
| `user_activity` | `medium` | 810 | `[0.0, 0.65, 0.35]` | False |
| `user_activity` | `sparse` | 815 | `[0.0, 0.6, 0.4]` | True |
| `expert_disagreement` | `high` | 812 | `[0.0, 0.65, 0.35]` | False |
| `expert_disagreement` | `low` | 812 | `[0.0, 0.65, 0.35]` | False |
| `expert_disagreement` | `medium` | 811 | `[0.0, 1.0, 0.0]` | False |

ML-1M is the clearest warning case. Segmenting by user activity selects pure M7
for dense users on validation, but that choice fails on test. This supports the
earlier conclusion that ML-1M should be treated as neutral, not as a dataset with a
stable routing gain.

### Amazon

| Strategy | Segment | Val Users | Weights | Fallback |
|---|---|---:|---|---|
| `user_activity` | `dense` | 15225 | `[0.0, 0.7, 0.3]` | False |
| `user_activity` | `medium` | 13085 | `[0.0, 0.6, 0.4]` | True |
| `user_activity` | `sparse` | 20975 | `[0.0, 0.6, 0.4]` | True |
| `expert_disagreement` | `high` | 16428 | `[0.0, 0.6, 0.4]` | True |
| `expert_disagreement` | `low` | 16429 | `[0.0, 0.6, 0.4]` | True |
| `expert_disagreement` | `medium` | 16428 | `[0.0, 0.7, 0.3]` | False |

Amazon is the only dataset where a segment-level variant slightly improves over
V2.2. The useful change is small: medium expert-disagreement users shift from
`(0.00, 0.60, 0.40)` to `(0.00, 0.70, 0.30)`.

## Conclusion

V4 is useful as a diagnostic, but not as the primary method.

Recommended thesis wording:

> We additionally tested deployable segment-level blend variants based on user
> activity and expert disagreement. These variants did not consistently improve
> over the dataset-level global blend. This suggests that the simple V2.2 blend
> acts as a strong regularized solution, while finer adaptive routing requires
> stronger features or more conservative regularization.

Recommended next step:

- Keep V2.2 as the main reported method.
- Use V4 to motivate why naive segment-level routing is not enough.
- If continuing adaptive routing, add regularization or constrain segment weights
  to stay close to the global V2.2 weight.

