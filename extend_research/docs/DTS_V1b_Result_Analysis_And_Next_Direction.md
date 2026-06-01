# DTS-v1b Result Analysis and Next Direction

**Date:** 2026-05-31  
**Input results:** `docs/DTS_V1b_Routing_Results.md`  
**Main conclusion:** Dataset-level density is a strong signal, but item-degree-only routing is too weak as a standalone selector.

---

## 1. What the Results Say

### Static expert pattern is clear

| Dataset | Density regime | M7 | R1 | R1-plus | Winner |
|---|---|---:|---:|---:|---|
| ML-20M | dense | 0.117462 | 0.111087 | 0.110969 | M7 |
| ML-1M | medium/sparse | 0.166112 | 0.194266 | 0.193157 | R1 |
| Amazon | sparse | 0.056335 | 0.072863 | 0.069523 | R1 |

This reproduces the paper's central pattern:

```text
dense graph -> injection expert M7
sparse graph -> regularizer expert R1/R1-plus
```

So the original density law is real in the local run.

### Item-degree routing is not enough

DTS-v1b tried to route individual items by:

```text
item_degree < threshold -> regularizer
item_degree >= threshold -> M7
```

Validation-selected result:

| Dataset | Selected behavior | Interpretation |
|---|---|---|
| ML-20M | threshold policies selected, but unstable across seeds | small signal exists, but noisy |
| ML-1M | mostly R1-only | per-item routing adds no value beyond dataset-level choice |
| Amazon | always R1-only | graph is sparse enough that regularizer dominates globally |

The strongest test-only ML-20M threshold policy reaches about:

```text
0.117487 NDCG@10
```

while M7-only is:

```text
0.117462 NDCG@10
```

That is only `+0.000025`, too small to claim a meaningful method.

---

## 2. Why DTS-v1b Underperforms

### Reason 1: Item degree is a coarse feature

Item degree only describes item popularity. It does not know:

- whether the user has enough history;
- whether M7 and R1 disagree;
- whether the candidate item is ranked confidently by one expert;
- whether the item's LLM profile is informative;
- whether the graph neighborhood is semantically coherent.

So item degree can explain broad dataset regimes, but it cannot reliably choose the expert per user-item pair.

### Reason 2: Sparse datasets collapse to one global expert

On ML-1M and Amazon, validation learns:

```text
Use R1 almost everywhere.
```

This means the dataset-level choice is already strong. A per-item selector has little room unless it can identify the small subset where M7 is better.

### Reason 3: ML-20M has only tiny routing headroom

ML-20M is dense and M7 is already very strong. R1/R1-plus help only on small item regions, if at all. Hard threshold routing cannot capture that delicately.

### Reason 4: Hard routing is brittle

Hard routing makes a discrete decision:

```text
use exactly one expert score for this item
```

If the threshold is slightly wrong, it can replace good M7 scores with weaker R1 scores. A soft blend is likely safer.

---

## 3. What Not To Claim

Do not claim:

```text
DTS-v1b is a strong routing method.
```

Do not claim:

```text
item-degree routing beats fixed expert baselines.
```

Do not claim a full 4-datapoint result until ML20M-sub163 split/features are available.

Safe claim:

```text
V1b shows that dataset-level density is predictive, but item-degree alone is insufficient for per-item routing.
```

---

## 4. Best Next Direction

The right next step is **V2: context-aware soft router**, not more hard thresholds.

### V2 objective

Learn a lightweight selector:

```text
score_final(u, i) =
    w_M7(u, i)      * score_M7(u, i)
  + w_R1(u, i)      * score_R1(u, i)
  + w_R1plus(u, i)  * score_R1plus(u, i)
```

with:

```text
w_M7 + w_R1 + w_R1plus = 1
```

This is safer than hard routing because it can interpolate.

### V2 feature set

Use cheap features, no retraining experts:

| Feature | Why useful |
|---|---|
| `log_item_degree` | keeps density signal |
| `log_user_history_len` | captures user-side coldness |
| expert scores `s_M7, s_R1, s_R1plus` | lets router see current confidence |
| score margin between top experts | detects disagreement |
| item rank per expert | rank is more comparable than raw score |
| score entropy / top-k gap | confidence proxy |
| dataset density bucket | global prior |

Optional if cheap:

| Feature | Why useful |
|---|---|
| profile embedding norm / missing flag | detects weak content profile |
| item semantic-neighbor density | separates popularity from semantic support |

### V2 model

Start simple:

```text
Logistic regression / small MLP over user-item candidate features
```

Train on validation-style labels:

```text
positive = held-out item
negative = sampled/ranked non-relevant candidate
```

Do not train M7/R1/R1-plus again.

Only train the router.

---

## 5. Recommended Experiment Plan

### Step 1: Candidate cache

For each dataset/seed/expert:

```text
save top-100 or top-200 candidates per eval user
```

Do this for validation and test.

Reason:

```text
Full all-item scoring is expensive. Candidate cache makes V2 fast.
```

### Step 2: Build router dataset

For validation users:

```text
candidate = union(topK_M7, topK_R1, topK_R1plus)
label = 1 if candidate in held-out positives else 0
features = router features above
```

### Step 3: Train router

Train a simple model:

```text
LogReg -> MLP only if LogReg is insufficient
```

Tune only on validation.

### Step 4: Test once

Apply trained router to test candidates.

Report:

- NDCG@10;
- Recall@10;
- MRR;
- comparison vs M7/R1/R1-plus;
- comparison vs DTS-v1b hard routing.

### Step 5: Ablation

Run feature ablation:

| Variant | Purpose |
|---|---|
| degree only | compare directly to DTS-v1b |
| degree + user history | test user coldness |
| expert scores only | test confidence routing |
| all features | final router |

---

## 6. Immediate Next Task

Implement:

```text
V2 candidate/score cache
```

for the three currently usable datasets:

```text
ML-20M, ML-1M, Amazon
```

Do not wait for ML20M-sub163. It can be added later after split/features arrive.

The candidate cache is the infrastructure needed for any stronger selector.

Status update:

```text
Completed for ML-20M, ML-1M, and Amazon.
```

See:

```text
docs/V2_Candidate_Score_Cache.md
```
