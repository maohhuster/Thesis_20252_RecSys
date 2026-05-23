# R1 (RLMRec-gene) — hyperparameter selection

This directory carries everything reviewers need to inspect or reproduce R1-gene's
hyperparameter selection across the four density datapoints.

## Files

| File | What |
|---|---|
| `grid_selection.json` | Unified manifest: per-dataset role, grid axes, all 54 d=128 cells with `val_recall20`, winner, test metrics. Includes a `ml20m_d32_archive` block preserving the prior d=32 selection. |
| `README.md` | This file. |

## Per-dataset summary (selection metric: validation Recall@20, RLMRec native)

| Datapoint | Selection protocol | Winner config | NDCG@10 (5-seed) |
|---|---|---|---|
| ML-20M (home, d=128) | 54-point grid: `layer_num × mask_ratio × recon_weight × re_temperature`; `reg_weight=1e-7` fixed | `layer_num=2, mask_ratio=0.05, recon_weight=0.10, re_temperature=0.5` | 0.1162 ± 0.0004 |
| Subsampled-ML-20M (d=32) | Upstream-canonical (no separate grid) | inherits the `ml20m_sub163_ours` block in `lightgcn_gene.yml` | (see `results_ml20m_sub163/r1_ml20m_sub163_metrics.json`) |
| ML-1M (d=32) | Upstream-canonical | inherits `ml1m_ours` block | (see `results_ml1m/r1_ml1m_metrics.json`) |
| Amazon-Books-2018 (d=32) | Upstream-canonical | inherits `amazon_ours` block | (see `results_amazon/r1_amazon_metrics.json`) |

## d=128 capacity-matched grid selection (May 2026)

The ML-20M grid was first run at RLMRec's upstream-default `embedding_size=32`
(winner: `ln=3, mr=0.10, rw=0.10, rt=0.2`; archived under
`datasets.ml20m_d32_archive`). To close the selection/deployment gap with the
shared M7/LightGCN-SF backbone (`d=128`), the full 54-point grid was re-run at
`d=128`. The new winner is **`ln=2, mr=0.05, rw=0.10, rt=0.5`** (val Recall@20
= 0.0527), structurally different from the d=32 winner:

- `layer_num`: **3 → 2** — the d=128 grid favours a shallower GNN once
  capacity is doubled.
- `re_temperature`: **0.2 → 0.5** — softer contrastive distribution suffices
  at higher capacity.
- `mask_ratio`: **0.10 → 0.05** — less aggressive masking.
- `recon_weight`: 0.10 (unchanged).

Reproduce: `python3 scripts/run_r1_ml20m_revalidate_d128.py` (winner + neighbours)
or `python3 scripts/run_r1_ml20m_single.py --layer_num <ln> ...` (single cell).

### d=128 5-seed test metrics

| Metric | Mean ± std | Per-seed values |
|---|---|---|
| NDCG@10  | 0.1162 ± 0.0004 | 0.1166, 0.1159, 0.1158, 0.1165, 0.1160 |
| Recall@10 | 0.0495 ± 0.0007 | 0.0507, 0.0490, 0.0487, 0.0495, 0.0495 |
| MRR      | 0.2366 ± 0.0018 | 0.2389, 0.2373, 0.2344, 0.2354, 0.2370 |

Paired-t vs M7 (mean 0.1175 ± 0.0004): Δ = −1.1% NDCG@10, p = 0.007 (**).
The capacity-match narrows R1's gap to M7 from −2.2% (d=32) to −1.1% (d=128)
while *preserving* the p<0.01 significance, without flipping the sign or the
qualitative conclusion (R1 still trails M7 on the dense ML-20M datapoint).

## Test metrics file pointers

Per-seed values live in `code/benchmark/results/r1_metrics.json` and
`code/benchmark/results_<dataset>/r1_<dataset>_metrics.json`. The
`grid_selection.json > datasets > <dataset> > test_metrics_file` field gives
the canonical path per datapoint.
