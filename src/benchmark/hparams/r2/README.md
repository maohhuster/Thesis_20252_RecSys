> _[LLM-MovieLens](../../../../README.md) · benchmark → hyperparameters → R2 (KAR-MoE replacer) — see the [root README](../../../../README.md) for the paper overview + repo map._

# R2 (KAR-MoE replacer) — hyperparameter selection

## Files

| File | What |
|---|---|
| `grid_selection.json` | Unified manifest: ML-20M 18-pt grid + per-dataset 4-cell retunes, all winners, test-metrics pointers, deltas vs. original. |
| `<dataset>_winner_seed<N>.json` | Per-seed test metrics at the retune winner (10 files: 2 datasets × 5 seeds). |
| `README.md` | This file. |

## Per-dataset summary (selection metric: validation NDCG@10)

| Datapoint | Selection protocol | Winner |
|---|---|---|
| ML-20M (home) | 18-point grid: `n_experts × lr × weight_decay` | `n_experts=4, lr=1e-3, weight_decay=1e-5` |
| ML-1M (retune) | 4-cell: `lr ∈ {3e-4, 1e-3} × wd ∈ {1e-5, 1e-4}`, `n_experts=4` fixed | `lr=1e-3, wd=1e-4` |
| Amazon-Books-2018 (retune) | same 4-cell grid | `lr=1e-3, wd=1e-4` |

The ML-20M winner is inherited unchanged to ML-1M / Amazon-Books in the
headline cross-density tables (matching the M-config protocol); the per-dataset
retune is a robustness control inside the union of KAR's published per-task grids.

## Test metrics

- ML-20M: `code/benchmark/results/r2_metrics.json`
- ML-1M retune: see `grid_selection.json > datasets > ml1m > retuned_test_metrics`
  (per-seed in `ml1m_winner_seed<N>.json`)
- Amazon retune: same convention under `datasets > amazon`
