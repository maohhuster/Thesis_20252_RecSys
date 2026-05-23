# Amazon-Books-2018 test metrics

5-seed aggregate test metrics per Tier-3 / Tier-3+ method on Amazon-Books-2018 (sparse-density endpoint, 13 int/item).

| File | Method | Notes |
|---|---|---|
| `r1_amazon_metrics.json` | R1-gene (RLMRec-gene) | upstream RLMRec authors' Amazon-Book hparams: `layer_num=2`, $d{=}32$ |
| **`r1plus_amazon_metrics.json`** | **R1-plus (RLMRec-plus) — paper-canonical** | depth-controlled re-run: **`layer_num=3`** (matched to R1-gene's structurally-equivalent depth-2 generative reconstruction; paper App. `r1plus_triangulation`), $d{=}32$. **This is the file the paper's `+26.4%` chain endpoint reads from.** |
| `r1plus_amazon_metrics_l2.json` | R1-plus depth-ablation companion | alternate depth `layer_num=2` (RLMRec-plus's upstream `amazon_ours:` YAML block); shipped alongside the canonical run as the *control* arm of the depth-controlled re-run disclosed in paper App. `r1plus_triangulation`. NDCG@10 = 0.0695 vs. canonical `_l3 → unsuffixed` 0.0711 (ln=3 won by 2.3 pp, ln=3 selected). |
| `r2_amazon_metrics.json` | R2 (KAR-style MoE replacer) | per-dataset retune (4-cell pre-registered grid; see `hparams/r2/grid_selection.json`) |
| `r3_amazon_metrics.json` | R3 (HypernetReplacer) | per-dataset retune (4-cell pre-registered grid; see `hparams/r3/grid_selection.json`) |
| `sasrec_pmixer/` | SASRec (sequential, orthogonal axis) | per-seed result JSONs under `bge-large-en-v1.5/seed-*/results.json`; pmixer upstream defaults |
