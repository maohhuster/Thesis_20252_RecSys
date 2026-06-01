# DTS-v1b Routing Results

**Date:** 2026-05-31  
**Stage:** validation-selected per-item density threshold reranking  
**Datasets run:** ML-20M, ML-1M, Amazon  
**Skipped:** ML20M-sub163, because split/features are still missing

---

## 1. What Was Run

DTS-v1b uses the already-trained expert checkpoints:

```text
M7, R1, R1-plus
```

No expert is retrained by bucket. For each seed and dataset:

```text
1. Load expert checkpoints.
2. Propagate embeddings through the train graph.
3. Compute item_degree from train.csv.
4. Evaluate fixed experts and density-threshold policies on validation.
5. Select the best validation policy.
6. Report the selected policy on test.
```

Score mixing detail:

```text
Each expert score vector is per-user z-score normalized before routing.
```

This is necessary because hard routing combines scores from different models item-by-item.

---

## 2. Validation-Selected Test Results

| Dataset | Seeds | Mean Test NDCG@10 | Std | Mean Test Recall@10 | Mean Test MRR | Selected Policies |
|---|---:|---:|---:|---:|---:|---|
| ML-20M | 5 | 0.117386 | 0.000415 | 0.049858 | 0.242659 | `{"DTS-2T-R1-R1plus-M7-T200-T500": 1, "DTS-hard-R1-T200": 1, "DTS-hard-R1-T500": 2, "DTS-hard-R1-plus-T200": 1}` |
| ML-1M | 5 | 0.194915 | 0.001264 | 0.060510 | 0.344751 | `{"R1-only": 4, "R1-plus-only": 1}` |
| Amazon | 5 | 0.072863 | 0.001026 | 0.095218 | 0.122201 | `{"R1-only": 5}` |

---

## 3. Fixed Expert Baselines

| Dataset | M7 | R1 | R1-plus | Best Fixed |
|---|---:|---:|---:|---|
| ML-20M | 0.117462 | 0.111087 | 0.110969 | M7 |
| ML-1M | 0.166112 | 0.194266 | 0.193157 | R1 |
| Amazon | 0.056335 | 0.072863 | 0.069523 | R1 |

Interpretation:

```text
The cross-dataset density law is reproduced:
dense ML-20M -> M7
sparser ML-1M/Amazon -> R1
```

---

## 4. Best Test Policies by Dataset

These are not the final protocol result because they look at test directly, but they help diagnose whether routing has headroom.

| Dataset | Best Test Policy | Mean Test NDCG@10 | Comment |
|---|---|---:|---|
| ML-20M | `DTS-hard-R1-plus-T100` / equivalent T100 variants | 0.117487 | Slightly above M7-only by ~0.000025 |
| ML-1M | `R1-only` | 0.194266 | Routing does not beat dataset-level expert choice |
| Amazon | `R1-only` / `DTS-hard-R1-T500` | 0.072863 | Routing collapses to R1 for almost all useful items |

---

## 5. Per-Seed Selection

| Dataset | Seed | Selected Policy | Val NDCG@10 | Test NDCG@10 |
|---|---:|---|---:|---:|
| ML-20M | 42 | DTS-hard-R1-plus-T200 | 0.071700 | 0.117669 |
| ML-20M | 123 | DTS-hard-R1-T500 | 0.073168 | 0.116814 |
| ML-20M | 456 | DTS-hard-R1-T500 | 0.071709 | 0.117827 |
| ML-20M | 789 | DTS-2T-R1-R1plus-M7-T200-T500 | 0.073434 | 0.116958 |
| ML-20M | 2026 | DTS-hard-R1-T200 | 0.073595 | 0.117663 |
| ML-1M | 42 | R1-only | 0.132995 | 0.193455 |
| ML-1M | 123 | R1-only | 0.136340 | 0.193400 |
| ML-1M | 456 | R1-only | 0.139362 | 0.196530 |
| ML-1M | 789 | R1-only | 0.136160 | 0.195477 |
| ML-1M | 2026 | R1-plus-only | 0.136099 | 0.195713 |
| Amazon | 42 | R1-only | 0.072083 | 0.073906 |
| Amazon | 123 | R1-only | 0.073597 | 0.073309 |
| Amazon | 456 | R1-only | 0.076248 | 0.073519 |
| Amazon | 789 | R1-only | 0.071998 | 0.071000 |
| Amazon | 2026 | R1-only | 0.072225 | 0.072583 |

---

## 6. Conclusion

Current DTS-v1b result is conservative:

```text
Item-degree threshold routing alone is not yet a strong method.
```

What it does show:

- The original density law is reproducible with local checkpoints.
- Dataset-level density is a strong signal for choosing the expert family.
- Per-item degree routing gives only tiny headroom on ML-20M and no benefit on ML-1M/Amazon.

What this means for the next design:

```text
V1b is a useful baseline, but the next selector should add more features than item_degree.
```

Useful next features:

- user history length;
- item degree;
- candidate rank from each expert;
- score margin / entropy per expert;
- profile-vs-ID agreement;
- dataset-level density as a global context feature.

Do not claim a full 4-datapoint routing result until ML20M-sub163 split/features are added.
