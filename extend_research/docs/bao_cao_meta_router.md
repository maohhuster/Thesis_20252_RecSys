# Báo cáo tóm tắt — Density-adaptive Meta-Router

**Hướng nghiên cứu ① từ kế hoạch follow-up của LLM-MovieLens (CIKM 2026)**

---

## 1. Bối cảnh và động lực

Bài báo gốc tại CIKM 2026 đã xác lập một quy luật quan trọng: trong các hệ thống gợi ý có sử dụng nội dung do LLM sinh ra, **lựa chọn paradigm tích hợp content phụ thuộc vào mật độ dữ liệu (density)**. Ba paradigm chính gồm:

- **Injection** — kết hợp ID embedding và content (M7)
- **Regularizer** — dùng content làm loss phụ để căn chỉnh (R1)
- **Replacer** — bỏ ID, chỉ dùng content (R2/R3)

Quy luật quan sát được trên 4 datapoint density:

| Dataset | Density (int/item) | Paradigm thắng |
|---|---|---|
| ML-20M | 1,160 | Injection ≈ Regularizer |
| sub-ML-20M | 225 | Regularizer (+3.7%) |
| ML-1M | 163 | Regularizer (+17.0%) |
| Amazon-Books | 13 | Regularizer (+29.5%) |

**Hạn chế của bài CIKM:** chỉ *mô tả* quy luật, chưa có phương pháp tự động khai thác. Hướng ① biến quy luật mô tả này thành một **selector học được**, áp dụng zero-shot trên các domain mới — chính là move biến công trình từ Best Resource Paper-competitive lên Best Full Research Paper-competitive.

---

## 2. Tổng quan cách hoạt động

Meta-router là một mạng nhỏ riêng biệt, đặt *bên trên* các paradigm CF chính. Nó **không phải là model CF** mà là một **policy** quyết định "với input này, dùng paradigm nào".

### Pipeline tổng thể

```
Catalog statistics + per-item statistics
              │
              ▼
       Meta-Router (MLP nhỏ)
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
   w_inj  w_reg  w_rep  (trọng số softmax)
       │      │      │
       ▼      ▼      ▼
      M7     R1     R2   (3 paradigm CF pre-computed)
       │      │      │
       └──────┼──────┘
              ▼
    Weighted combination
              ▼
       Final ranking
```

### Hai chế độ vận hành

| Chế độ | Mô tả | Vai trò |
|---|---|---|
| **Per-dataset routing** | Một bộ trọng số duy nhất cho cả catalog | Baseline đơn giản |
| **Per-item routing** | Trọng số riêng cho từng item | Claim chính — không paradigm cố định nào làm được |

**Per-item routing là hạt nhân mới:** trong cùng một catalog, item warm (nhiều interaction) đi qua injection, item cold (ít interaction) đi qua regularizer — một hỗn hợp có điều kiện theo density.

---

## 3. Đầu vào (Input features)

Có 3 nhóm feature, tổng ~16 chiều cho per-item routing.

### Nhóm A — Catalog-level (8 features, luôn dùng)

| Feature | Định nghĩa | Vai trò |
|---|---|---|
| `log_per_item_density` | log(interactions / items) | Predictor chính |
| `log_per_user_density` | log(interactions / users) | Predictor phụ |
| `log_n_items`, `log_n_users` | Log kích thước catalog | Catalog scale |
| `degree_gini_item`, `degree_gini_user` | Gini của phân bố degree | Đo lệch |
| `cold_fraction` | % item có <10 interaction | Sparse tail size |
| `density_ratio_p10_p90` | Tỷ số density percentile | Skewness |

### Nhóm B — Per-item (5 features, chỉ dùng per-item routing)

| Feature | Định nghĩa |
|---|---|
| `log_item_degree` | log(interaction của item này) |
| `item_age` | Thời gian từ first interaction |
| `item_degree_percentile` | Percentile trong catalog |
| `content_richness` | Độ dài profile / số theme |
| `content_uniqueness` | Distance trung bình tới k-NN |

### Nhóm C — Content-SNR (3 features)

| Feature | Định nghĩa |
|---|---|
| `nn_overlap_profile_vs_genome` | Overlap top-10 NN giữa profile và genome |
| `profile_vs_id_similarity` | Cosine giữa profile và ID embedding |
| `profile_dispersion` | Std của profile embeddings |

**Lưu ý kỹ thuật:** Các feature có range rộng (density, n_items, n_users) đều phải log-scale vì trải qua 2 bậc decade.

---

## 4. Kiến trúc mạng

### Thiết kế chính: MLP 2-layer nhỏ (~10K params)

```python
class MetaRouter(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, n_paradigms=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_paradigms),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, features):
        logits = self.net(features)
        return F.softmax(logits / self.temperature, dim=-1)
```

### Lý giải từng quyết định thiết kế

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| Architecture | MLP 2-layer, ~10K params | Đủ expressive, ít overfit khi training data nhỏ |
| Normalization | LayerNorm | Robust với batch size nhỏ |
| Activation | GELU | Smooth gradient, ổn định khi data ít |
| Regularization | Dropout 0.2 + weight decay 1e-4 | Chống overfit |
| Output | Softmax với learnable T | Cho phép hard switch hoặc soft blend |
| Số paradigm | 3 (có thể giảm 2) | Replacer có thể bỏ nếu luôn thua |

### Baseline so sánh — Linear router

Để chứng minh độ phức tạp là cần thiết, phải train song song một logistic regression đơn giản với cùng features. **Nếu MLP không vượt được linear, không cần MLP.** Đây là ablation bắt buộc.

---

## 5. Output và cách blend

### Hai chế độ output

**Hard switch (đơn giản, dễ giải thích):**

```python
paradigm_choice = weights.argmax(dim=-1)
# Item này chỉ dùng paradigm được chọn
```

**Soft blend (claim chính):**

```python
final_score = (weights[:, 0] * score_inj
             + weights[:, 1] * score_reg
             + weights[:, 2] * score_rep)
```

### Vấn đề calibration cross-paradigm

Scores từ M7, R1, R2 có thể ở **thang đo hoàn toàn khác nhau**. Hai giải pháp:

1. **Rank-based blending** — chuyển score thành rank, blend rank. Robust, đơn giản. *Đề xuất dùng.*
2. **Z-score normalization** — chuẩn hóa scores per-user. Mạnh hơn về lý thuyết nhưng phải calibrate cẩn thận.

---

## 6. Training — Phần khó nhất

Có 3 lựa chọn training, mỗi cái có trade-off riêng.

### Lựa chọn 1: Supervised với oracle labels *(đề xuất cho paper đầu)*

**Ý tưởng:** Với mỗi (dataset, item-bucket), biết paradigm nào tốt nhất → label oracle. Train router predict label đó.

```python
for (dataset, item_subset) in matrix_cells:
    features = extract_features(dataset, item_subset)
    
    ndcg_inj = evaluate(M7, dataset, item_subset)
    ndcg_reg = evaluate(R1, dataset, item_subset)
    ndcg_rep = evaluate(R2, dataset, item_subset)
    
    label = argmax([ndcg_inj, ndcg_reg, ndcg_rep])
    
    pred = router(features)
    loss = cross_entropy(pred, label)
```

**Trade-off:**
- ✓ Đơn giản, training nhanh
- ✓ Không cần backprop qua CF models (rất đắt)
- ✗ Cần chạy tất cả paradigm trên tất cả cell trước
- ✗ Training data ít — phải mở rộng qua Hướng ②

### Lựa chọn 2: Bilevel optimization end-to-end

Train router *cùng* với CF models, optimize NDCG cuối cùng qua surrogate loss (ApproxNDCG, ListNet, NeuralNDCG).

**Trade-off:**
- ✓ Train trực tiếp trên metric quan tâm
- ✗ Rất đắt, khó hội tụ, overfit cao

### Lựa chọn 3: REINFORCE — Policy gradient

Coi paradigm choice là action, NDCG là reward.

**Trade-off:**
- ✓ Cho phép discrete choice
- ✗ High variance, hội tụ chậm

### Mở rộng training data

Vì training data ban đầu rất ít, cần augmentation:

1. Chạy ②'s domain matrices → +3–4 dataset × 3 bucket = ~12 labels
2. Subsample mỗi dataset thành 4–5 density levels → +12 labels mỗi dataset
3. Tổng: ~80–100 oracle labels — đủ train MLP nhỏ

---

## 7. Inference — Hai scenarios triển khai

### Scenario A: Pre-compute scores, blend ở inference *(đề xuất chính)*

```python
# Offline: chạy mỗi paradigm 1 lần
M7_scores = M7.score_all(users, items)
R1_scores = R1.score_all(users, items)
R2_scores = R2.score_all(users, items)

# Online per user
weights = router(item_features)
final = (weights[:, 0:1] * M7_scores[user]
       + weights[:, 1:2] * R1_scores[user]
       + weights[:, 2:3] * R2_scores[user])
```

| Trade-off | Đánh giá |
|---|---|
| Inference speed | Nhanh |
| Memory | 3× scores cần lưu |
| Flexibility | Cho phép full blending |

### Scenario B: Hard routing, chỉ chạy paradigm được chọn

Phù hợp cho ablation "hard routing", memory thấp hơn nhưng mất khả năng blend.

---

## 8. Interpretability — Phần reviewer sẽ soi

Khi router là MLP, reviewer chắc chắn hỏi: "Mạng học cái gì? Có thật sự dựa vào density không?" Cần chuẩn bị 3 phân tích.

### Feature importance qua SHAP

```python
import shap
explainer = shap.DeepExplainer(router, background_features)
shap_values = explainer.shap_values(test_features)
```

**Expected:** `log_per_item_density` và `log_per_user_density` đứng top-2 importance.

### Sanity check: loại bỏ density feature

Train lại router không có density feature. **Expected:** performance sụp về paradigm cố định → confirm density là tín hiệu thật.

### Visualization decision boundary

Vẽ 2D plot với x-axis = log_per_item_density, y-axis = một feature khác, color = paradigm được chọn. Phải thấy phân vùng rõ ràng theo density.

---

## 9. Đánh giá — Bộ baseline đầy đủ

Để claim "selector ≥ best-fixed-paradigm trên mọi held-out domain", cần các baseline:

| Baseline | Vai trò |
|---|---|
| **Injection-always** (M7) | Paradigm cố định #1 |
| **Regularizer-always** (R1) | Paradigm cố định #2 |
| **Replacer-always** (R2/R3) | Paradigm cố định #3 |
| **Oracle** | Upper bound — không thể vượt |
| **Density-ablated router** | Phải sụp → chứng minh density là tín hiệu thật |
| **Linear router** | Phải bị MLP vượt → biện minh complexity |
| **Random router** | Sàn — phải vượt rõ ràng |
| **Threshold policy** (biến thể a) | Biện minh meta-router > rule-based |

### Metrics

- **NDCG@10, Recall@10, MRR** per domain — chất lượng cơ bản
- **Regret vs oracle** — khoảng cách tới upper bound
- **Win-rate vs best-fixed-paradigm trên held-out domains** — claim quan trọng nhất
- **Per-bucket lift** — chia item theo cold/medium/warm
- Protocol: 5 seed + paired t-test (giữ nguyên từ CIKM)

### Ablation bắt buộc

- Per-item routing **vs** per-dataset routing — chứng minh per-item tốt hơn
- Learned **vs** threshold — chứng minh học vượt heuristic
- Feature importance qua SHAP
- τ sensitivity / hyperparameter sweep
- Số paradigm: 3 vs 2 (bỏ replacer)

---

## 10. Defense trước critique chính

**Critique:** "Router của bạn quá đơn giản — chỉ là logistic regression trá hình. Sao không dùng threshold đơn giản?"

**Defense 3 lớp:**

| Lớp | Lập luận |
|---|---|
| **1** | Threshold cho hard switch, không cho trọng số blend liên tục. Per-item dynamic blending là điều threshold không match được. |
| **2** | Router học được nonlinear interactions (ví dụ: high density + low content-richness → injection chỉ thắng nhẹ) — không thể bằng ngưỡng đơn lẻ. |
| **3** | Router với nhiều feature generalize cross-domain tốt hơn threshold fit trên ML domains. Phải chứng minh empirically trên Yelp/Steam từ Hướng ②. |

---

## 11. Falsifier — Cách paper có thể "thất bại nhưng vẫn xuất bản"

**Claim chính:** Selector ≥ best-fixed-paradigm trên **mọi** domain held-out, zero-shot.

**Falsifier:** Nếu trên một domain held-out, selector thua best-fixed-paradigm → claim sai.

**Nhưng:** Vẫn xuất bản được dưới dạng **negative result có bằng chứng** ("quy luật không khai thác được"), backup bởi data từ ②.

Đây là điểm mạnh thiết kế — không bị rủi ro chạy 6 tháng rồi không publish được.

---

## 12. Checklist quyết định kỹ thuật

| Quyết định | Lựa chọn đề xuất |
|---|---|
| Architecture | MLP 2-layer, hidden=32, ~10K params |
| Activation | GELU |
| Normalization | LayerNorm |
| Dropout | 0.2 |
| Output | Softmax 3-way với learnable temperature |
| Số paradigm | 3 (có ablation 2) |
| Blending | Rank-based (Scenario A) |
| Training | Supervised với oracle labels |
| Loss | Cross-entropy với label smoothing |
| Data augmentation | Subsample density levels |
| Weight decay | 1e-4 |
| Validation | Leave-one-out trên domains |
| Seed | 5 seed + paired t-test |

---

## 13. Venue ceiling và lộ trình

| Yếu tố | Đánh giá |
|---|---|
| **Effort** | Medium (selector nhẹ; cost chính là matrix của ②) |
| **Risk** | "Trivial threshold critique" → giảm bằng per-item + zero-shot evidence |
| **Venue ceiling** | NeurIPS main / KDD / SIGIR / WWW |
| **Journal** | ACM TORS / TKDE |
| **Anchors peer-reviewed** | RLMRec, KAR, LightGCN |
| **Combo khuyến nghị** | ①+② cùng nhau (Full Research best-paper play) |

---

## 14. Tóm tắt một dòng

> Meta-router là một MLP 2-layer (~10K params), input ~16 catalog-level + per-item features (density log-scaled là chính), output softmax 3-way với learnable temperature, blend rank-based scores từ 3 paradigm pre-computed, train supervised với oracle labels từ matrix cells (mở rộng qua Hướng ②), validate bằng LOO cross-domain, defend bằng feature ablation chứng minh density là tín hiệu chính và per-item routing không thay thế được bằng threshold đơn lẻ.

---

## Tham chiếu

- **Bài CIKM gốc:** LLM-MovieLens: A Density-Paradigm Framework with Class-Level Triangulation for LLM-Augmented Collaborative Filtering (CIKM 2026)
- **README kế hoạch:** Hướng ① trong "Future directions — LLM-MovieLens follow-up work"
- **Peer-reviewed anchors:** RLMRec (WWW 2024), KAR (KDD 2024), LightGCN (SIGIR 2020), FACE (NeurIPS 2025)
