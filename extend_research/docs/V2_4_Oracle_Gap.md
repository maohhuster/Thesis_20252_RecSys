# V2.4 Oracle Gap

Date: 2026-06-01

## Purpose

V2.2 established a strong dataset-level global blend, and V2.3 showed reliable gains
on ML-20M and Amazon. V2.4 asks whether there is still enough headroom to justify a
V3 per-user router.

This is an oracle analysis, so it is not deployable. It uses test labels to choose
the best method per user and should be interpreted only as an upper bound.

Compared methods:

- best fixed expert: best of M7, R1, R1-plus at seed level;
- V2.2 global blend: one dataset-level blend weight;
- oracle expert: per-user best of M7, R1, R1-plus;
- oracle all: per-user best of M7, R1, R1-plus, and V2.2 global blend.

## Summary

| Dataset | Best Fixed | V2.2 Blend | Oracle Expert | Oracle All | Blend Gain | Oracle-All Gap vs Blend |
|---|---:|---:|---:|---:|---:|---:|
| ML-20M | 0.117462 | 0.120126 | 0.155609 | 0.157996 | +0.002664 | +0.037870 |
| ML-1M | 0.194915 | 0.195024 | 0.243882 | 0.245488 | +0.000110 | +0.050463 |
| Amazon | 0.072953 | 0.076577 | 0.104959 | 0.107637 | +0.003624 | +0.031060 |

## Interpretation

The oracle gap is large on every dataset. This means V2.2 is a strong global
baseline, but there is still substantial per-user heterogeneity that a better router
could exploit.

The most important number is `Oracle-All Gap vs Blend`:

- ML-20M: `+0.037870`
- ML-1M: `+0.050463`
- Amazon: `+0.031060`

These gaps are much larger than the V2.2 gains over fixed experts, so V3 is
justified. The challenge is not whether there is room; the challenge is learning a
router that can capture even a small fraction of that oracle gap without overfitting.

## Oracle Selection Behavior

Oracle-all does not simply discard the global blend. It frequently selects it:

| Dataset | Typical Global-Blend Selection Share |
|---|---:|
| ML-20M | about 32-34% of users |
| ML-1M | about 22-29% of users |
| Amazon | about 45-46% of users |

This is useful. It says the global blend is not merely a compromise baseline; it is
the best per-user option for a large fraction of users, especially on Amazon. A V3
router should therefore choose among four actions:

1. M7
2. R1
3. R1-plus
4. V2.2 global blend

not just among the three base experts.

## Dataset-Specific Reading

### ML-20M

V2.2 already improves M7, but oracle-all reaches `0.157996`. The oracle often picks
M7, R1, R1-plus, and global blend in meaningful proportions. This supports a
feature-conditioned router, especially using sparse-user and expert-disagreement
features from V2.2 segment analysis.

### Amazon

Amazon has the cleanest current evidence: V2.2 is significant, and oracle-all still
adds a large upper-bound gap. Since oracle-all selects global blend for roughly 45%
of users, the blend itself should remain one of the candidate actions.

### ML-1M

ML-1M was neutral in V2.3, but V2.4 shows a large oracle gap. This means the problem
is not absence of signal. The problem is that a single dataset-level blend cannot
adapt across users. V3 may help ML-1M, but it needs strong regularization because
the dataset is smaller and validation overfit already appeared in V2.1.

## V3 Recommendation

The next experiment should be a conservative action router:

- actions: `m7`, `r1`, `r1plus`, `global_blend`;
- target: per-user best action on validation;
- model: multinomial logistic regression or shallow tree;
- features:
  - user history length;
  - expert disagreement;
  - score margins among experts;
  - overlap of expert top-10/top-100 candidate lists;
  - aggregate candidate popularity;
  - optionally dataset id if training a cross-dataset router.

The evaluation must compare against V2.2, not only against fixed experts.

Success criterion:

- beat V2.2 on ML-20M and Amazon;
- avoid hurting ML-1M;
- recover even 10-20% of the oracle-all gap.

## Output Files

- `extend_research/results/v2_4_oracle_gap/oracle_gap_summary.csv`
- `extend_research/results/v2_4_oracle_gap/oracle_gap_by_seed.csv`
- `extend_research/results/v2_4_oracle_gap/oracle_selection_counts.csv`
- `extend_research/results/v2_4_oracle_gap/V2_4_Oracle_Gap.md`
