# DTS-v1b Checkpoint Calibration

**Stage:** V1b checkpoint calibration  
**Dataset:** ML-20M  
**Date:** 2026-05-31  
**Status:** M7 and R1-plus are ready for per-item routing; R1 needs checkpoint/protocol clarification.

---

## 1. Purpose

Before running a Density Threshold Selector on per-user reranking scores, each local checkpoint must first reproduce the paper-level static expert metrics.

This prevents a false routing result caused by a bad loader, wrong feature alignment, or wrong LightGCN propagation depth.

---

## 2. Data and Checkpoints

Data:

```text
extend_research/data/raw/hf/llm-movielens/benchmark_splits/ml20m/
extend_research/data/raw/hf/llm-movielens/embeddings/ml20m/bge-large-en-v1.5/
```

Checkpoints:

```text
extend_research/data/raw/checkpoints/ml20m_downloads/
```

Manifests:

```text
extend_research/configs/ml20m_data_paths.json
extend_research/configs/ml20m_checkpoint_paths.json
```

Evaluator:

```text
extend_research/scripts/evaluate_ml20m_static_expert.py
```

The evaluator loads checkpoint embeddings, applies the correct LightGCN graph propagation on the full train graph, and then uses the benchmark full-ranking evaluator.

---

## 3. Calibration Results

### M7

Command pattern:

```bash
cd extend_research
python3 scripts/evaluate_ml20m_static_expert.py --expert M7 --seed 42 --split test --device cpu --batch-size 512
```

5-seed result:

| Expert | Propagation | Mean NDCG@10 | Mean Recall@10 | Mean MRR | Paper Reference |
|---|---:|---:|---:|---:|---:|
| M7 | L3 | 0.117462 | 0.049990 | 0.242651 | NDCG@10 ≈ 0.1175 |

Conclusion:

```text
M7 checkpoint loading, profile+mood feature alignment, and graph propagation are calibrated.
```

### R1-plus

5-seed result:

| Expert | Propagation | Mean NDCG@10 | Mean Recall@10 | Mean MRR | Paper Reference |
|---|---:|---:|---:|---:|---:|
| R1-plus | L3 | 0.110969 | 0.047785 | 0.227590 | NDCG@10 = 0.110969 |

Conclusion:

```text
R1-plus exactly reproduces the paper aggregate. This checkpoint family is calibrated.
```

### R1

R1 was tested with both the documented d=128 winner depth and a depth sensitivity pass:

| Expert | Propagation | Mean NDCG@10 | Mean Recall@10 | Mean MRR | Paper Reference |
|---|---:|---:|---:|---:|---:|
| R1 | L2 | 0.111087 | 0.045615 | 0.232263 | 0.116184 |
| R1 | L3 | 0.114894 | 0.047874 | 0.234476 | 0.116184 |

Conclusion:

```text
The local R1 checkpoints do not reproduce the paper-canonical R1 result under the documented L2 protocol.
L3 is closer but still below the paper aggregate, so R1 should not be used as the primary V1b regularizer until the exact checkpoint/protocol metadata is clarified.
```

Likely causes:

- the local R1 files may not be the final paper-canonical d=128 winner checkpoints;
- the checkpoint files do not store `layer_num`, so propagation depth must be supplied externally;
- the paper R1 metadata says `layer_num=2`, while the local checkpoint behaves closer to an `L3` run.

---

## 4. V1b Decision

Proceed with V1b in two tracks:

| Track | Experts | Role |
|---|---|---|
| Primary | M7 + R1-plus | calibrated routing experiment |
| Sensitivity | M7 + R1(L3) + R1-plus | exploratory only, not a main claim |

Do not retrain M7/R1/R1-plus for bucket splits. These are full-train checkpoints, which is the correct protocol for DTS-v1.

---

## 5. Next Step

Implement per-item density routing on validation/test:

```text
1. Compute item_degree from train.csv.
2. Precompute final user/item embeddings for M7 and R1-plus per seed.
3. Tune threshold on validation only.
4. Report final selected threshold on test.
5. Add R1(L3) only as sensitivity.
```

Important implementation detail:

```text
Hard routing mixes scores from different experts, so scores must be calibrated before ranking.
Use per-user z-score normalization per expert before applying item-level routing.
```
