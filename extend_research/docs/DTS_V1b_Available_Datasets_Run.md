# DTS-v1b Available Dataset Run

**Date:** 2026-05-31  
**Stage:** Static expert reranking calibration for datasets with complete local data  
**Ran:** ML-1M and Amazon  
**Skipped:** ML20M-sub163 because split/features are still missing

---

## 1. Scope

The user supplied checkpoints for Amazon, ML-1M, and ML20M-sub163.

After checking data availability:

| Dataset | Checkpoints | Splits | Embeddings | Run status |
|---|---|---|---|---|
| ML-1M | yes | yes | yes | ran |
| Amazon | yes | yes | yes | ran |
| ML20M-sub163 | yes | no | no | skipped |

ML-20M had already been calibrated earlier.

---

## 2. Commands

Example command:

```bash
cd extend_research
python3 scripts/evaluate_ml20m_static_expert.py \
  --checkpoint-config configs/ml1m_checkpoint_paths.json \
  --data-config configs/ml1m_data_paths.json \
  --expert R1 \
  --seed 42 \
  --split test \
  --device cpu \
  --batch-size 512
```

Output root:

```text
results/dts_v1b/static_eval/
```

This result directory is local-only and ignored by git, so aggregate values are recorded below.

---

## 3. ML-1M Results

| Expert | Propagation | Mean NDCG@10 | Std | Mean Recall@10 | Mean MRR |
|---|---:|---:|---:|---:|---:|
| M7 | L3 | 0.166112 | 0.000371 | 0.043314 | 0.312786 |
| R1 | L3 | 0.194266 | 0.001499 | 0.060421 | 0.343562 |
| R1-plus | L3 | 0.193144 | 0.001785 | 0.059566 | 0.341272 |

Interpretation:

```text
ML-1M reproduces the density-law pattern: regularizers beat M7 clearly.
```

Best static expert:

```text
R1
```

---

## 4. Amazon Results

| Expert | Propagation | Mean NDCG@10 | Std | Mean Recall@10 | Mean MRR |
|---|---:|---:|---:|---:|---:|
| M7 | L3 | 0.056335 | 0.000268 | 0.075729 | 0.097486 |
| R1 | L2 | 0.072863 | 0.001026 | 0.095218 | 0.122201 |
| R1-plus | L2 | 0.069523 | 0.001186 | 0.092138 | 0.116412 |
| R1-plus | L3 sensitivity | 0.069061 | 0.000841 | 0.091492 | 0.116089 |

Interpretation:

```text
Amazon also reproduces the density-law pattern: regularizers beat M7 clearly.
```

Best static expert:

```text
R1
```

Note:

```text
The local Amazon R1-plus checkpoint family behaves closer to the L2 companion than the L3 canonical aggregate described in the original paper artifacts. For this local run, use L2 as the calibrated default unless a different R1-plus Amazon checkpoint set is supplied.
```

---

## 5. Current Four-Datapoint Status

| Dataset | Density regime | Static winner now available |
|---|---|---|
| ML-20M | dense | M7 |
| ML-1M | medium/sparse | R1 |
| Amazon | sparse | R1 |
| ML20M-sub163 | mid-density | pending data |

This is enough to continue implementation and evaluation on:

```text
ML-20M + ML-1M + Amazon
```

but not enough for:

```text
ML20M-sub163 reranking
```

until its split and feature alignment files are supplied.

---

## 6. Next Step

Use the calibrated datasets to run DTS-v1b routing in this order:

```text
1. ML-20M validation/test
2. ML-1M validation/test
3. Amazon validation/test
4. Add ML20M-sub163 later when files arrive
```

For the current local checkpoint set, default propagation depths are:

| Dataset | M7 | R1 | R1-plus |
|---|---:|---:|---:|
| ML-20M | 3 | 2 | 3 |
| ML-1M | 3 | 3 | 3 |
| Amazon | 3 | 2 | 2 |
