# Thí nghiệm V1 — Density Threshold Selector

**Tên ngắn:** DTS-v1  
**Trạng thái:** Thiết kế thí nghiệm, chưa triển khai code  
**Mục tiêu:** Kiểm tra nhanh xem quy luật density-paradigm trong bài CIKM có thể khai thác thành một selector đơn giản hay không, trước khi đầu tư vào Meta-Router MLP.

---

## 1. Bối cảnh

Bài CIKM gốc đã thiết lập một quy luật:

> Lựa chọn paradigm tích hợp LLM content phụ thuộc vào density của dữ liệu.

Các kết quả chính:

| Regime | Quan sát chính |
|---|---|
| Dense, ví dụ ML-20M | Injection `M7` rất mạnh và ổn định |
| Sparse, ví dụ ML-1M / Amazon | Regularizer `R1` / `R1-plus` thắng `M7` rõ rệt |
| Replacer `R2` / `R3` | Không thắng `M7`, và sụp ở sparse |

Hạn chế của bài gốc: mới dừng ở mức **mô tả quy luật**, chưa có phương pháp tự động chọn paradigm theo density.

V1 kiểm tra phiên bản đơn giản nhất của ý tưởng này:

```text
Item sparse/cold -> dùng regularizer
Item dense/warm  -> dùng injection
```

---

## 2. Câu hỏi nghiên cứu

**RQ-V1:** Một selector chỉ dựa trên item density có thể giảm regret so với fixed paradigm không?

Nói cụ thể:

```text
DTS-v1 có tốt hơn hoặc ít nhất tiệm cận best-fixed expert trên từng dataset/bucket không?
```

Trong đó fixed expert là:

- `M7`: injection, profile + mood
- `R1`: regularizer generative
- `R1-plus`: regularizer contrastive

---

## 3. Giả thuyết

### H1 — Density routing hữu ích

```text
Selector theo item density giảm regret so với việc luôn dùng một expert cố định.
```

### H2 — Regularizer tốt hơn ở sparse bucket

```text
Ở cold/medium items, R1 hoặc R1-plus có thể tốt hơn M7.
```

### H3 — Injection tốt hơn ở warm bucket

```text
Ở warm items, M7 vẫn là expert an toàn nhất.
```

### Falsifier

Nếu DTS-v1 không thắng hoặc không tiệm cận best-fixed expert trên validation/test, thì:

```text
Quy luật density có thể đúng ở cấp dataset nhưng chưa đủ mạnh để routing ở cấp item.
```

Khi đó hướng Meta-Router vẫn có thể tiếp tục, nhưng cần thêm feature ngoài item density:

- user history length
- catalog-level density
- content uniqueness
- profile-vs-ID similarity
- degree skewness

---

## 4. Nguyên tắc protocol

Không train riêng model theo bucket ở V1.

Protocol chuẩn:

```text
1. Train M7 trên full train set.
2. Train R1 trên full train set.
3. Train R1-plus trên full train set.
4. Tính item density chỉ từ train set.
5. Tune selector bằng validation set.
6. Report cuối cùng trên test set.
```

Không được:

```text
Train M7 chỉ trên warm items.
Train R1-plus chỉ trên cold items.
Chọn threshold bằng test set.
Chọn expert theo kết quả test bucket.
```

Lý do:

- cold items vốn ít interaction, train riêng sẽ làm mất tín hiệu collaborative toàn cục
- graph CF cần toàn bộ user-item graph
- train riêng theo bucket sẽ không còn so sánh công bằng với paper gốc
- chọn rule bằng test sẽ leak kết quả

---

## 5. Expert pool

### V1 chính: 2 expert

| Expert | Model | Vai trò |
|---|---|---|
| Injection | `M7` | Expert cho dense/warm regime |
| Regularizer | `R1` hoặc `R1-plus` | Expert cho sparse/cold regime |

Ta chạy hai bản:

```text
DTS-v1-R1      = M7 + R1
DTS-v1-R1plus  = M7 + R1-plus
```

### Ablation phụ: 3 expert

Chỉ chạy nếu đã có score/checkpoint thuận tiện:

```text
DTS-v1-3expert = M7 + best(R1, R1-plus) + best(R2, R3)
```

Kỳ vọng: 3-expert không nhất thiết tốt hơn 2-expert, vì replacer không thắng `M7` trong paper gốc. Nếu không tốt hơn, loại `R2/R3` khỏi method chính và giữ làm ablation.

---

## 6. Density signal

Tính từ train interactions:

```text
item_degree_i = số train interactions của item i
```

Bucket mặc định:

| Bucket | Điều kiện |
|---|---|
| Cold | `item_degree < 10` |
| Medium | `10 <= item_degree < 50` |
| Warm | `item_degree >= 50` |

Ngoài bucket mặc định, V1 cần threshold sweep:

```text
T in {5, 10, 20, 50, 100}
```

Với 2-threshold policy:

```text
T1 in {5, 10, 20}
T2 in {25, 50, 100}
T1 < T2
```

---

## 7. Selector variants

### 7.1 Fixed expert baselines

```text
M7-only
R1-only
R1-plus-only
```

Đây là baseline bắt buộc. DTS-v1 phải so với các expert cố định này.

### 7.2 Hard threshold selector

Phiên bản đơn giản nhất:

```python
if item_degree < T:
    use regularizer
else:
    use M7
```

Với regularizer có hai lựa chọn:

```text
R1
R1-plus
```

Phiên bản 2-threshold:

```python
if item_degree < T1:
    use R1_plus
elif item_degree < T2:
    use R1
else:
    use M7
```

Chỉ giữ phiên bản 2-threshold nếu validation chứng minh tốt hơn rõ ràng.

### 7.3 Soft density blend

Thay vì chọn cứng một expert:

```text
score_final(u, i) =
    alpha_i * score_regularizer(u, i)
  + (1 - alpha_i) * score_M7(u, i)
```

Trong đó:

```text
alpha_i = sigmoid(a - b * log(item_degree_i + 1))
```

Ý nghĩa:

```text
item càng sparse -> alpha lớn -> dựa vào regularizer nhiều hơn
item càng warm   -> alpha nhỏ -> dựa vào M7 nhiều hơn
```

Ở V1, `a` và `b` không học bằng neural net. Chỉ grid search trên validation:

```text
a in {-2, -1, 0, 1, 2}
b in {0.25, 0.5, 1.0, 2.0}
```

### 7.4 Oracle upper bound

Oracle không phải method triển khai thật. Nó dùng để đo trần:

```text
Với mỗi dataset/bucket, chọn expert có validation NDCG@10 tốt nhất.
```

Report:

```text
regret_vs_oracle = oracle_metric - selector_metric
```

---

## 8. Score calibration

Khi blend score từ nhiều expert, phải chuẩn hóa scale.

### Cách chính: per-user z-score

```text
score_norm(u, i) =
    (score(u, i) - mean_i score(u, i)) / std_i score(u, i)
```

Sau đó mới hard select hoặc soft blend.

### Cách phụ: rank-based blending

Nếu score scale quá lệch:

```text
rank_score(u, i) = -rank(u, i)
```

Rồi blend rank score thay vì raw score.

V1 nên chạy cả hai nếu rẻ:

```text
DTS-zscore
DTS-rankblend
```

Nếu chỉ chọn một để bắt đầu, dùng z-score vì dễ giải thích và gần với score-based recommender hơn.

---

## 9. Dataset scope

V1 không mở domain mới.

Chỉ dùng datapoint có sẵn trong artifact:

| Datapoint | Density xấp xỉ | Vai trò |
|---|---:|---|
| ML-20M | 1,160 int/item | Dense endpoint |
| sub-ML-20M | 225 int/item | Same-domain density control |
| ML-1M | 163 int/item | Mid/sparse MovieLens |
| Amazon-Books | 13 int/item | Sparse endpoint |

Nếu chưa có đủ per-user/per-item scores cho tất cả expert, chia V1 thành hai mức.

### V1a — Aggregate diagnostic

Dùng result JSON hiện có để phân tích ở cấp dataset/bucket.

Output:

```text
Expert nào thắng ở dataset nào?
Expert nào thắng ở bucket nào nếu bucket result có sẵn?
Oracle bucket-level upper bound là bao nhiêu?
```

Không claim là selector thật, chỉ dùng để quyết định có làm V1b không.

### V1b — Actual reranking selector

Dùng checkpoint hoặc exported score để tạo ranking cuối:

```text
score_M7(u, all_items)
score_R1(u, all_items)
score_R1plus(u, all_items)
score_DTS(u, all_items)
```

Sau đó evaluate full ranking đúng protocol paper.

---

## 10. Metrics

### Overall metrics

Report trên toàn test set:

```text
NDCG@10
Recall@10
MRR
```

### Bucket metrics

Report theo cold/medium/warm:

```text
Recall@100
Recall@1000
MRR_full
NDCG@100
NDCG@1000
```

Lý do: cold items thường không xuất hiện ở top-10, nên chỉ nhìn NDCG@10 có thể bỏ lỡ tín hiệu.

### Router-specific metrics

```text
regret_vs_oracle
win_rate_vs_M7
win_rate_vs_R1
win_rate_vs_R1plus
win_rate_vs_best_fixed
```

Trong đó:

```text
win_rate_vs_best_fixed =
    số dataset/bucket DTS-v1 >= best fixed expert
    / tổng số dataset/bucket
```

---

## 11. Validation và model selection

Tất cả lựa chọn sau phải chọn bằng validation:

- `R1` hay `R1-plus` là regularizer expert
- hard threshold hay soft blend
- threshold `T`, hoặc `T1/T2`
- z-score hay rank-based blend
- có dùng replacer trong 3-expert ablation không

Selection metric chính:

```text
validation NDCG@10
```

Nếu mục tiêu là cold-start, thêm tie-breaker:

```text
validation Recall@1000 trên cold/medium buckets
```

Nhưng tie-breaker phải được ghi trước khi chạy test.

---

## 12. Statistical protocol

Giữ nguyên seed set của paper:

```text
42, 123, 456, 789, 2026
```

Với mỗi seed:

```text
evaluate M7, R1, R1-plus, DTS-v1
```

So sánh:

```text
DTS-v1 vs M7
DTS-v1 vs R1
DTS-v1 vs R1-plus
DTS-v1 vs best-fixed
```

Test thống kê:

```text
paired t-test trên 5 seed
```

Report:

```text
mean ± std
delta %
p-value
significance band
```

---

## 13. Expected outcomes

### Thành công tối thiểu

```text
DTS-v1 giảm regret vs oracle so với M7-only.
DTS-v1 không thua best-fixed expert đáng kể.
```

### Thành công mạnh

```text
DTS-v1 thắng best-fixed expert trên nhiều dataset/bucket.
DTS-v1 cải thiện rõ ở sparse bucket mà không làm warm bucket giảm mạnh.
```

### Thất bại có ích

Nếu DTS-v1 không tốt:

```text
item_degree một mình không đủ để routing.
```

Kết luận tiếp theo:

```text
Meta-Router cần feature phong phú hơn: user density, catalog density,
degree skewness, content uniqueness, profile-vs-ID similarity.
```

---

## 14. Bảng output bắt buộc

### Table 1 — Overall test performance

| Method | NDCG@10 | Recall@10 | MRR |
|---|---:|---:|---:|
| M7-only | | | |
| R1-only | | | |
| R1-plus-only | | | |
| DTS-hard | | | |
| DTS-soft | | | |
| Oracle | | | |

### Table 2 — Bucket performance

| Method | Bucket | NDCG@100 | Recall@100 | NDCG@1000 | Recall@1000 | MRR_full |
|---|---|---:|---:|---:|---:|---:|
| M7-only | cold | | | | | |
| DTS-v1 | cold | | | | | |
| M7-only | medium | | | | | |
| DTS-v1 | medium | | | | | |
| M7-only | warm | | | | | |
| DTS-v1 | warm | | | | | |

### Table 3 — Threshold sensitivity

| T | Validation NDCG@10 | Test NDCG@10 | Cold Recall@1000 | Regret vs Oracle |
|---:|---:|---:|---:|---:|
| 5 | | | | |
| 10 | | | | |
| 20 | | | | |
| 50 | | | | |
| 100 | | | | |

### Table 4 — Expert choice summary

| Dataset | Bucket | Chosen Expert | Oracle Expert | Match? |
|---|---|---|---|---|
| ML-20M | cold | | | |
| ML-20M | warm | | | |
| Amazon | cold | | | |
| Amazon | warm | | | |

---

## 15. Files to implement

Planned implementation files:

```text
extend_research/configs/dts_v1.json

extend_research/src/extend_research/analysis/density_buckets.py
extend_research/src/extend_research/selectors/threshold_selector.py
extend_research/src/extend_research/selectors/score_calibration.py
extend_research/src/extend_research/evaluation/router_metrics.py

extend_research/scripts/analyze_dts_v1.py
extend_research/scripts/run_dts_v1.py
extend_research/scripts/export_dts_v1_tables.py

extend_research/results/dts_v1/
```

---

## 16. Minimal implementation order

### Step 1 — V1a diagnostic

```text
Read existing result JSON.
Create dataset-level expert table.
Create bucket table if bucket results are available.
Compute oracle upper bound at aggregate level.
```

### Step 2 — Score export check

```text
Check whether checkpoints can produce score_all(users, items).
If yes, implement actual selector.
If no, first implement evaluator/exporter for M7/R1/R1-plus scores.
```

### Step 3 — Hard selector

```text
Implement T sweep.
Tune T on validation.
Evaluate on test.
```

### Step 4 — Soft blend

```text
Implement alpha_i.
Grid search a, b on validation.
Evaluate on test.
```

### Step 5 — Report

```text
Export four required tables.
Write results summary in extend_research/docs/DTS_V1_Results.md.
```

---

## 17. Decision after V1

Go to Meta-Router MLP only if one of these is true:

```text
DTS-v1 beats best-fixed expert on at least one held-out density regime.
DTS-v1 reduces regret vs oracle consistently.
DTS-v1 improves sparse buckets without damaging warm buckets.
```

If none holds:

```text
Do not build MLP yet.
First add richer routing features and cross-domain data.
```

---

## 18. One-line summary

> DTS-v1 is a 2-expert density threshold selector: train `M7`, `R1`, and `R1-plus` on the full train set, compute item density from train interactions, tune hard/soft routing thresholds on validation, evaluate full-ranking on test, and report overall, bucket, threshold-sensitivity, and regret-vs-oracle tables.
