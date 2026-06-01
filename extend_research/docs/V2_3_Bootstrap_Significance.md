# V2.3 Bootstrap Significance

Date: 2026-06-01

## Purpose

V2.2 showed that a dataset-level global blend improves mean NDCG@10 over the best
fixed expert. V2.3 tests whether those gains are reliable under paired user-level
bootstrap resampling.

Comparison:

- ML-20M: V2.2 global blend vs M7.
- ML-1M: V2.2 global blend vs the best fixed expert per seed.
- Amazon: V2.2 global blend vs the best fixed expert per seed.

Bootstrap setup:

- unit: test user;
- samples: `10,000`;
- confidence interval: percentile 95%;
- statistic: mean user-level delta.

## NDCG@10 Summary

| Dataset | Users | Mean Delta | 95% CI | P(delta > 0) | Interpretation |
|---|---:|---:|---|---:|---|
| ML-20M | 13,215 | +0.002664 | [0.001754, 0.003582] | 1.0000 | Reliable positive gain |
| ML-1M | 4,915 | +0.000110 | [-0.001237, 0.001449] | 0.5688 | Not significant / neutral |
| Amazon | 49,300 | +0.003624 | [0.003015, 0.004261] | 1.0000 | Reliable positive gain |

## Other Metrics

| Dataset | Metric | Mean Delta | 95% CI | P(delta > 0) |
|---|---|---:|---|---:|
| ML-20M | Recall@10 | +0.001193 | [0.000303, 0.002110] | 0.9952 |
| ML-20M | MRR | +0.004071 | [0.001603, 0.006479] | 0.9996 |
| Amazon | Recall@10 | +0.005015 | [0.003936, 0.006085] | 1.0000 |
| Amazon | HR@10 | +0.010751 | [0.008438, 0.013103] | 1.0000 |
| Amazon | MRR | +0.005562 | [0.004486, 0.006620] | 1.0000 |
| ML-1M | Recall@10 | +0.000107 | [-0.000866, 0.001070] | 0.5963 |
| ML-1M | MRR | +0.001565 | [-0.001800, 0.004931] | 0.8203 |

## Per-Seed Pattern

ML-20M is positive on all five seeds, but seed 456 has a wider interval:

- Seed 42: `+0.002866`, CI `[0.000823, 0.004931]`
- Seed 123: `+0.003416`, CI `[0.001403, 0.005435]`
- Seed 456: `+0.001433`, CI `[-0.000509, 0.003372]`
- Seed 789: `+0.003188`, CI `[0.001001, 0.005368]`
- Seed 2026: `+0.002416`, CI `[0.000306, 0.004507]`

Amazon is positive on every seed with all intervals fully above zero:

- Seed 42: `+0.003097`, CI `[0.001758, 0.004467]`
- Seed 123: `+0.003113`, CI `[0.001806, 0.004425]`
- Seed 456: `+0.003828`, CI `[0.002407, 0.005282]`
- Seed 789: `+0.004764`, CI `[0.003275, 0.006259]`
- Seed 2026: `+0.003319`, CI `[0.002002, 0.004657]`

ML-1M is mixed:

- three seeds are positive;
- two seeds are negative;
- every seed-level CI crosses zero.

## Conclusion

The reliable claims are:

1. V2.2 significantly improves ML-20M over M7 in aggregate.
2. V2.2 significantly improves Amazon over the best fixed expert in aggregate and
   on every seed.
3. ML-1M should be reported as neutral/stability evidence, not as a significant
   improvement.

Recommended paper/thesis wording:

> The dataset-level global blend yields statistically reliable gains on ML-20M and
> Amazon under paired user-level bootstrap, while remaining approximately neutral on
> ML-1M.

This is stronger and cleaner than claiming universal improvement across all
datasets.

## Output Files

- `extend_research/results/v2_3_bootstrap/bootstrap_summary.csv`
- `extend_research/results/v2_3_bootstrap/bootstrap_by_seed.csv`
- `extend_research/results/v2_3_bootstrap/V2_3_Bootstrap_Significance.md`
