# V2.2 Segment Analysis

Date: 2026-06-01

## Purpose

V2.2 showed that a dataset-level global blend improves over the best fixed expert on
ML-20M, ML-1M, and Amazon. This analysis asks where the improvement comes from.

Segments are computed on the test caches:

- user activity: sparse / medium / dense by train-history length tertiles;
- target popularity: cold / warm / hot by mean ground-truth item degree;
- expert disagreement: low / medium by average reciprocal-rank disagreement among
  M7, R1, and R1-plus.

All segment metrics use the exact V2.2 scoring rule, including `missing_score=-10.0`
for candidates absent from an expert top-100 list.

## Overall Check

The `all` segment matches the V2.2 router summary:

| Dataset | V2.2 Blend | Best Expert | Delta |
|---|---:|---:|---:|
| ML-20M | 0.120126 | 0.117462 | +0.002664 |
| ML-1M | 0.195024 | 0.194915 | +0.000110 |
| Amazon | 0.076577 | 0.072953 | +0.003624 |

## Key Findings

### ML-20M

The global blend helps most for sparse and medium-history users:

| Segment | Blend NDCG@10 | Best Expert | Delta |
|---|---:|---:|---:|
| sparse users | 0.175441 | 0.170622 | +0.004819 |
| medium users | 0.102451 | 0.100431 | +0.002019 |
| dense users | 0.082380 | 0.081924 | +0.000455 |

It also helps on hot target items:

| Segment | Blend NDCG@10 | Best Expert | Delta |
|---|---:|---:|---:|
| hot items | 0.129282 | 0.126303 | +0.002979 |
| warm items | 0.003650 | 0.006232 | -0.002582 |
| cold items | 0.000000 | 0.000000 | 0.000000 |

Interpretation: ML-20M should be framed as M7-dominant, but sparse-user ranking
benefits from a small amount of R1 and R1-plus signal. The current cold/warm item
segments have few users and low absolute NDCG, so they should not be overclaimed.

### Amazon

Amazon gain is broad across user activity, strongest for sparse users:

| Segment | Blend NDCG@10 | Best Expert | Delta |
|---|---:|---:|---:|
| sparse users | 0.075643 | 0.071422 | +0.004221 |
| dense users | 0.082456 | 0.078738 | +0.003719 |
| medium users | 0.071237 | 0.068950 | +0.002286 |

Amazon also gains most on hot target items:

| Segment | Blend NDCG@10 | Best Expert | Delta |
|---|---:|---:|---:|
| hot items | 0.101327 | 0.096007 | +0.005319 |
| warm items | 0.028814 | 0.028725 | +0.000089 |
| cold items | 0.011560 | 0.012504 | -0.000944 |

Expert disagreement matters:

| Segment | Blend NDCG@10 | Best Expert | Delta |
|---|---:|---:|---:|
| medium disagreement | 0.062078 | 0.057984 | +0.004093 |
| low disagreement | 0.105689 | 0.103052 | +0.002637 |

Interpretation: Amazon is the cleanest evidence that R1 and R1-plus contain
complementary semantic ranking signal. The blend is especially useful when experts
do not fully agree, and it improves sparse-user ranking without needing a hard
expert switch.

### ML-1M

ML-1M remains mostly a stability check. The global blend's overall gain is small,
and segment behavior is mixed.

It helps sparse users but hurts medium/dense users:

| Segment | Blend NDCG@10 | Best Expert | Delta |
|---|---:|---:|---:|
| sparse users | 0.236257 | 0.235689 | +0.000568 |
| medium users | 0.183784 | 0.184674 | -0.000891 |
| dense users | 0.164460 | 0.165945 | -0.001485 |

By item popularity, almost all evaluable users fall into the hot segment:

| Segment | Mean Users | Delta |
|---|---:|---:|
| hot items | 954.0 | +0.000048 |
| warm items | 25.0 | -0.001225 |
| cold items | 4.0 | 0.000000 |

Interpretation: ML-1M is close to saturated by R1/R1-plus. The global blend is not
bad, but its gain is too small to use as headline evidence. It is useful because it
does not break the method on a compact dataset.

## Research Implication

The V2.2 result is not just a raw ensemble win. The segment analysis suggests two
mechanisms:

1. On ML-20M, a mostly-M7 blend improves sparse-user ranking.
2. On Amazon, an R1/R1-plus blend improves ranking when semantic experts disagree,
   especially for sparse users and hot items.

This is enough to motivate a V3 router, but V3 should be targeted rather than blind:

- use user history length as a real feature;
- use expert disagreement as a real feature;
- treat target/item popularity carefully, because current gains come mostly from hot
  items and cold/warm segments are sparse or weak.

## Output Files

- `extend_research/results/v2_2_segments/segment_summary.csv`
- `extend_research/results/v2_2_segments/segment_by_seed.csv`
- `extend_research/results/v2_2_segments/V2_2_Segment_Analysis.md`
