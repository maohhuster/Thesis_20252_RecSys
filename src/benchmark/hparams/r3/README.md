> _[LLM-MovieLens](../../../../README.md) · benchmark → hyperparameters → R3 (HypernetReplacer) — see the [root README](../../../../README.md) for the paper overview + repo map._

# R3 (HypernetReplacer) — hyperparameter selection

R3 is the second replacer-class instantiation used to triangulate R2; it is
*not* in the main-results table, so every datapoint's result lives here —
including the ML-20M one (this is the *only* asymmetry from R2's layout).

## Files

| File | What |
|---|---|
| `grid_selection.json` | Unified manifest: per-dataset 4-cell grids, winners, R3 test metrics, deltas vs. M7 and R2. |
| `<dataset>_winner_seed<N>.json` | Per-seed test metrics at the winner (15 files: 3 datasets × 5 seeds). |
| `README.md` | This file. |

## Per-dataset summary (selection metric: validation NDCG@10)

| Datapoint | Winner | NDCG@10 (5-seed) | Δ vs. M7 |
|---|---|---|---|
| ML-20M | `lr=3e-4, wd=1e-5` | 0.1099 ± 0.0045 | −6.5% (p=0.021) |
| ML-1M | `lr=3e-4, wd=1e-5` | 0.1669 ± 0.0004 | (within seed noise of M7) |
| Amazon-Books-2018 | `lr=1e-3, wd=1e-4` | 0.0496 ± 0.0003 | −11.7% |

Architecture (R3 = MLP from content to item embedding, no ID, no gating): see
`code/benchmark/models/hypernet_replacer.py` (~80 LOC).

## Subsampled-ML-20M

R3 is evaluated on the sub163 same-domain density control as part of the
replacer-class triangulation across all four datapoints (App. `r3_triangulation`):
R3 scores −3.9% vs. M7 (p<0.001, 5/5 same-sign). Per-seed metrics:
`results_ml20m_sub163/r3_ml20m_sub163_metrics.json`.
