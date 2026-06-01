# DTS-v1 Initial Results

**Stage:** V1a aggregate diagnostic  
**Generated from:** released aggregate JSON metrics under `code/benchmark/results*`  
**Status:** V1a aggregate diagnostic complete; V1b checkpoint calibration started.

---

## 1. What Was Run

The implemented V1a diagnostic reads aggregate NDCG@10 values for:

- `M7` as the injection expert
- `R1` as the generative regularizer expert
- `R1-plus` as the contrastive regularizer expert
- `R2` / `R3` as replacer diagnostics

It then evaluates density-based policies at the dataset level.

Run command:

```bash
cd extend_research
make dts-v1
```

Generated files:

```text
results/dts_v1/expert_table.csv
results/dts_v1/policy_summary.csv
results/dts_v1/policy_decisions.csv
results/dts_v1/ml20m_bucket_table.csv
results/dts_v1/dts_v1_report.json
results/dts_v1/DTS_V1_Results.md
```

---

## 2. Expert Winners

| Dataset | Density | M7 | R1 | R1-plus | R2 | R3 | Oracle |
|---|---:|---:|---:|---:|---:|---:|---|
| ML-20M | 1160 | 0.1175 | 0.1162 | 0.1110 | 0.1145 | 0.1099 | M7 |
| sub163 | 225 | 0.1076 | 0.1116 | 0.1193 | 0.1021 | 0.1034 | R1-plus |
| ML-1M | 163 | 0.1661 | 0.1943 | 0.1932 | 0.1680 | 0.1669 | R1 |
| Amazon | 13 | 0.0563 | 0.0729 | 0.0711 | 0.0439 | 0.0497 | R1 |

Interpretation:

```text
Dense endpoint    -> M7
Mid-density point -> R1-plus
Sparse endpoints  -> R1
```

This supports the broad density law:

```text
dense -> injection
sparse -> regularizer
```

It also shows item/catalog density alone is not enough to choose between `R1` and `R1-plus` in a principled way.

---

## 3. Best Aggregate Policy

The best aggregate policy after adding a two-threshold diagnostic is:

```text
DTS-2T-R1-R1plus-M7-T200-T500
```

Rule:

```text
density < 200       -> R1
200 <= density <500 -> R1-plus
density >= 500      -> M7
```

Aggregate result:

| Method | Mean NDCG@10 | Mean Regret vs Oracle | Win-rate vs Best Fixed |
|---|---:|---:|---:|
| DTS-2T-R1-R1plus-M7-T200-T500 | 0.125989 | 0.000000 | 1.00 |
| DTS-hard-R1-plus-T500 | 0.125283 | 0.000706 | 0.50 |
| DTS-hard-R1-T500 | 0.124053 | 0.001936 | 0.75 |
| R1-only | 0.123724 | 0.002265 | 0.50 |
| R1-plus-only | 0.123650 | 0.002339 | 0.25 |
| M7-only | 0.111875 | 0.014114 | 0.25 |

The two-threshold policy exactly matches the oracle on the four aggregate datapoints.

---

## 4. Important Caveat

This is **not yet evidence that DTS-v1 is a real method**.

Reason:

```text
The two-threshold policy was derived after seeing these four datapoints.
```

Therefore it is best interpreted as:

```text
an aggregate diagnostic showing what a density-aware policy would need to learn
```

not as:

```text
a validated selector that generalizes.
```

To become a real claim, the same rule must be tested on:

- validation/test reranking with checkpoint scores, or
- held-out domains not used to define the thresholds.

---

## 5. V1b Readiness

V1b requires one of:

```text
1. Checkpoints for M7/R1/R1-plus per seed and dataset.
2. Per-user/per-item score dumps for M7/R1/R1-plus.
```

Current local repo status after importing Downloads checkpoints:

```text
ML-20M local checkpoints are available for M7, R1, and R1-plus across all 5 seeds.
```

Manifest:

```text
configs/ml20m_checkpoint_paths.json
configs/ml20m_data_paths.json
```

Local checkpoint root:

```text
data/raw/checkpoints/ml20m_downloads/
```

Checkpoint count:

```text
M7      5 best_model.pt files
R1      5 best_model.pt files
R1-plus 5 best_model.pth files
```

Additional checkpoint import on 2026-05-31:

```text
data/raw/checkpoints/amazon_downloads/
data/raw/checkpoints/ml1m_downloads/
data/raw/checkpoints/ml20m_sub163_downloads/
```

New manifests:

```text
configs/amazon_checkpoint_paths.json
configs/ml1m_checkpoint_paths.json
configs/ml20m_sub163_checkpoint_paths.json
```

Verified counts:

```text
Amazon        M7/R1/R1-plus x 5 seeds = 15 checkpoints
ML-1M         M7/R1/R1-plus x 5 seeds = 15 checkpoints
ML20M-sub163  M7/R1/R1-plus x 5 seeds = 15 checkpoints
```

This means checkpoint coverage is now available for all four density datapoints. Actual reranking still requires matching split files and embedding feature files for each non-ML20M dataset to be present and calibrated.

Data availability check on 2026-05-31:

| Dataset | Checkpoints | Splits | Embeddings | Ready for reranking? | Notes |
|---|---|---|---|---|---|
| ML-20M | yes | yes | yes | yes | already calibrated for M7/R1-plus |
| Amazon | yes | yes | yes | almost | IDs are contiguous; synthesize identity maps; pad 43 missing embedding rows with zeros |
| ML-1M | yes | yes | yes | yes | maps and embeddings are present |
| ML20M-sub163 | yes | no | no | no | HF release does not include sub163 splits/embeddings |

Additional data manifests:

```text
configs/amazon_data_paths.json
configs/ml1m_data_paths.json
configs/ml20m_sub163_data_paths.json
```

Available-dataset V1b static reranking has been run for ML-1M and Amazon.

See:

```text
docs/DTS_V1b_Available_Datasets_Run.md
```

DTS-v1b validation-selected threshold routing has also been run on the three currently usable datasets.

See:

```text
docs/DTS_V1b_Routing_Results.md
```

Current conclusion:

```text
Dataset-level density is reproducible and strong, but item-degree-only routing is not yet a strong standalone method.
```

HF data downloaded locally:

```text
data/raw/hf/llm-movielens/benchmark_splits/ml20m/
data/raw/hf/llm-movielens/embeddings/ml20m/bge-large-en-v1.5/
```

Verified key shapes/counts:

```text
profile_embeddings.npy  (10381, 1024)
mood_vectors.npy        (10381, 10)
combined_features.npy   (10381, 1034)
train.csv               11,499,778 rows
val.csv                 49,668 rows
test.csv                67,466 rows
item_map.json           9,906 items
user_map.json           127,371 users
```

---

## 6. Next Technical Step

The next useful implementation task is one of:

1. Add a score-export interface for the local ML-20M checkpoints.
2. Add a `scores/` input format so V1b can run from precomputed expert scores.
3. Extend V1a to support held-out domain JSONs when new domain matrices are produced.

Recommended next step:

```text
Implement the V1b score-input contract first.
```

That keeps the reranking selector independent of how expert scores are produced.

---

## 7. V1b Calibration Update

See:

```text
docs/DTS_V1b_Checkpoint_Calibration.md
```

Current ML-20M checkpoint calibration:

| Expert | Status | 5-seed NDCG@10 |
|---|---|---:|
| M7 | calibrated | 0.117462 |
| R1-plus | calibrated, exactly matches paper aggregate | 0.110969 |
| R1 L2 | not calibrated vs paper-canonical R1 | 0.111087 |
| R1 L3 | closer but still below paper-canonical R1 | 0.114894 |

Decision:

```text
Run V1b primary routing with M7 + R1-plus first.
Keep R1 as a sensitivity experiment until the exact R1 checkpoint/protocol metadata is clarified.
```
