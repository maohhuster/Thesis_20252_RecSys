# V2 Candidate Score Cache

**Date:** 2026-05-31  
**Purpose:** Build reusable top-K candidate/score caches for the next context-aware router.  
**Status:** Complete for ML-20M, ML-1M, and Amazon. ML20M-sub163 is still skipped.

---

## 1. What Was Exported

Command:

```bash
cd extend_research
python3 scripts/export_v2_candidate_cache.py --config configs/v2_candidate_cache.json
```

Output root:

```text
results/v2_candidate_cache/
```

Manifest:

```text
results/v2_candidate_cache/manifest.json
```

Cache coverage:

| Dataset | Seeds | Splits | Top-K | Files |
|---|---:|---|---:|---:|
| ML-20M | 5 | val, test | 100 | 10 |
| ML-1M | 5 | val, test | 100 | 10 |
| Amazon | 5 | val, test | 100 | 10 |
| ML20M-sub163 | - | - | - | skipped |

Total:

```text
30 top-100 NPZ files
~206 MB local disk
```

The cache is local-only under `results/` and ignored by git.

---

## 2. NPZ Schema

Each file:

```text
results/v2_candidate_cache/{dataset_slug}/seed-{seed}/{split}_top100.npz
```

contains:

| Key | Shape | Meaning |
|---|---|---|
| `users` | `(n_eval_users,)` | contiguous user IDs in this split |
| `user_history_len` | `(n_eval_users,)` | train history length for each eval user |
| `item_degree` | `(n_items,)` | train interaction count for each item |
| `gt_indptr` | `(n_eval_users + 1,)` | CSR pointer for held-out positives |
| `gt_items` | `(n_heldout_interactions,)` | held-out positive item IDs |
| `m7_top_items` | `(n_eval_users, 100)` | M7 top candidates |
| `m7_top_scores` | `(n_eval_users, 100)` | M7 normalized scores |
| `r1_top_items` | `(n_eval_users, 100)` | R1 top candidates |
| `r1_top_scores` | `(n_eval_users, 100)` | R1 normalized scores |
| `r1plus_top_items` | `(n_eval_users, 100)` | R1-plus top candidates |
| `r1plus_top_scores` | `(n_eval_users, 100)` | R1-plus normalized scores |
| `metadata_json` | scalar string | dataset, seed, split, top-K, expert list |

Score normalization:

```text
per-user z-score before train-history masking and top-K extraction
```

---

## 3. Verified Example Shapes

### ML-20M seed 42 test

```text
n_eval_users = 2643
n_items      = 9906
gt_items     = 67466
top arrays   = (2643, 100)
```

### ML-1M seed 42 test

```text
n_eval_users = 983
n_items      = 2807
gt_items     = 42982
top arrays   = (983, 100)
```

### Amazon seed 42 test

```text
n_eval_users = 9860
n_items      = 9332
gt_items     = 40106
top arrays   = (9860, 100)
```

---

## 4. How V2 Should Use This Cache

For each user:

```text
candidate_set = union(
    m7_top_items[user],
    r1_top_items[user],
    r1plus_top_items[user]
)
```

For each `(user, candidate_item)`, build features:

| Feature | Source |
|---|---|
| `log_item_degree` | `item_degree[item]` |
| `log_user_history_len` | `user_history_len[user]` |
| `m7_score`, `r1_score`, `r1plus_score` | expert top-score lookup, missing = low score |
| `m7_rank`, `r1_rank`, `r1plus_rank` | expert rank lookup, missing = `top_k + 1` |
| expert score margins | derived |
| label | item appears in `gt_items[gt_indptr[u]:gt_indptr[u+1]]` |

Recommended first V2 model:

```text
Logistic regression over cached candidate features.
```

Only move to MLP if logistic regression cannot improve over fixed experts and DTS-v1b.

---

## 5. Next Implementation Step

Build:

```text
scripts/train_v2_router.py
```

Initial scope:

```text
ML-1M and Amazon first, then ML-20M.
```

Reason:

```text
ML-1M/Amazon are smaller and show strong expert differences, so they are better for debugging router training.
```
