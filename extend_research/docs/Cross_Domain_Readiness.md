# Cross-Domain Readiness

Date: 2026-06-01

## Purpose

This document checks which new domains are realistic for the next research phase:

**cross-domain density-law replication + regularized zero-shot blend policy.**

The goal is not to start training immediately. The goal is to choose the first
domains where the existing `extend_research` pipeline can be adapted with the
least ambiguity.

## Decision

Start with:

1. **Steam**
2. **Yelp**

Keep as second wave:

3. **Goodreads**
4. **MIND**

Reason:

- Steam has direct recommender-system data, item metadata, manageable item count,
  and direct downloadable files.
- Yelp has strong review/business metadata and explicit ratings, but is larger and
  may require manual official download / terms handling.
- Goodreads is scientifically useful because it complements Amazon Books, but the
  complete graph is very large.
- MIND is important but protocol-shifted: it is news-impression recommendation,
  not the same user-item rating/review setup as the current MovieLens/Amazon cache
  pipeline.

## Local Readiness

Current local workspace status:

| Asset | Status |
|---|---|
| Current reusable score-cache pipeline | ready for ML-20M / ML-1M / Amazon-style caches |
| Cross-domain raw data | Steam downloaded locally; Yelp/Goodreads/MIND not downloaded |
| Cross-domain checkpoints | not available locally |
| Cross-domain candidate caches | not available locally |
| Existing result protocol | NDCG@10 / Recall@10 / MRR, 5 seeds, validation-selected blend |

Implication:

The next implementation phase should continue with Steam density design and
normalized table export, not with a new router.

## Dataset Readiness Table

| Domain | Readiness | First Use | Main Strength | Main Risk |
|---|---|---|---|---|
| Steam | High | first smoke domain | recommender-native interactions + game metadata | implicit-feedback interpretation and preprocessing |
| Yelp | High-Medium | first or second smoke domain | reviews, ratings, business metadata, timestamps | large JSON download and possible manual terms/download flow |
| Goodreads | Medium | second wave | huge book interaction graph, genre subsets, rich book metadata | complete data is very large; use subsets first |
| MIND | Medium-Low | later | strong news benchmark, impression logs, title/abstract/entities | protocol differs from rating/review recommendation |

## Domain Details

### 1. Steam

Recommended status: **first smoke test**.

Source:

- UCSD / McAuley recommender datasets: <https://cseweb.ucsd.edu/~jmcauley/datasets.html>
- Steam data directory: <https://mcauleylab.ucsd.edu/public_datasets/data/steam/>

Relevant source facts:

- UCSD lists Steam Video Game and Bundle Data with reviews, purchases, plays,
  recommends / likes, bundles, and pricing information.
- Reported scale: about 7.8M reviews, 2.6M users, 15.5K items, and 615 bundles.
- The public directory includes manageable files such as user/item data, bundle
  data, game metadata, and review files.

Why it fits:

- The domain is different from movies/books, so it helps salami-slicing defense.
- Item count is manageable compared with Goodreads.
- Metadata such as game title, genre, tags, pricing, and bundle membership can
  support content/profile construction.
- Interaction density can be sub-sampled into multiple density points.

Risks:

- Need to decide which behavior becomes the main positive signal: purchase, play,
  recommend, or review.
- Reviews/plays may need timestamp normalization.
- Rating scale may not match MovieLens/Amazon directly.

First adapter target:

```text
user_id, item_id, timestamp, interaction_value, item_title, item_metadata_text
```

Initial density points:

| Point | Sketch |
|---|---|
| dense | users/items after stronger k-core filtering |
| medium | moderate k-core or sampled interactions |
| sparse | lower item-degree / lower interaction-per-item subset |

First smoke run:

- one Steam subset;
- 3 seeds;
- M7/R1/R1-plus-style score cache if checkpoints/features are available;
- otherwise first produce a static expert/readiness baseline before profile
  generation.

### 2. Yelp

Recommended status: **first-wave domain, after Steam or in parallel**.

Source:

- Yelp Open Dataset: <https://business.yelp.com/data/resources/open-dataset/>

Relevant source facts:

- Yelp describes the dataset as an educational subset with reviews, businesses,
  photos, check-ins, and attributes.
- The current official page reports about 6.99M reviews and 150K businesses.
- The JSON download is several GB compressed and larger after extraction.

Why it fits:

- Explicit star ratings map naturally to recommender evaluation.
- Review text and business categories/attributes give strong content signal.
- It is a different domain family from movies/books.
- Timestamps support temporal split.

Risks:

- The official download may require accepting terms manually.
- The full JSON is large enough that preprocessing should be streaming.
- Business fields such as global `review_count` can leak popularity if used
  carelessly; compute train-only item degree instead.
- Yelp is location/category-heavy; need decide whether to use all businesses or a
  stable subset such as restaurants.

First adapter target:

```text
user_id, business_id, stars, date, business_name, categories, attributes, city/state
```

Initial density points:

| Point | Sketch |
|---|---|
| dense | popular categories / stronger k-core |
| medium | broad business subset with moderate k-core |
| sparse | lower k-core or tail-category subset |

First smoke run:

- parse `review.json` + `business.json`;
- create a restaurant-only or all-business subset;
- temporal split;
- compute train-only density features;
- verify candidate cache shape before profile generation.

### 3. Goodreads

Recommended status: **second wave**.

Sources:

- UCSD Book Graph: <https://sites.google.com/eng.ucsd.edu/ucsdbookgraph>
- UCSD / McAuley recommender datasets: <https://cseweb.ucsd.edu/~jmcauley/datasets.html>

Relevant source facts:

- UCSD Book Graph includes book metadata, user-book interactions, and detailed
  review texts.
- It reports more than 229M user-book interactions and more than 15M detailed
  reviews.
- The site recommends genre-wise subsets for experimentation because the complete
  interaction dataset is very large.

Why it fits:

- It complements Amazon Books and tests whether the book-domain result generalizes.
- It includes rich book metadata and reviews.
- Genre subsets can naturally provide multiple density regimes.

Risks:

- The complete dataset is large.
- It is closer to Amazon Books than Yelp/Steam, so it is less valuable as the first
  salami-slicing defense domain.
- Need care with duplicate/update versions and genre overlap.

First adapter target:

```text
user_id, book_id, rating_or_shelf_action, timestamp_if_available, title, authors, genres
```

Recommended use:

- do not start with complete 229M interactions;
- start with one or two genre subsets;
- use it after Steam/Yelp proves the pipeline works.

### 4. MIND

Recommended status: **later / separate protocol branch**.

Sources:

- Microsoft Learn MIND page: <https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news>
- MIND project page: <https://msnews.github.io/>

Relevant source facts:

- Microsoft describes MIND as a large-scale news recommendation dataset collected
  from anonymized Microsoft News behavior logs.
- It contains about 160K English news articles and more than 15M impression logs
  from about 1M users.
- `behaviors.tsv` contains histories and impression logs; `news.tsv` contains news
  title, abstract, category, subcategory, URL, and entity annotations.
- Microsoft notes that full article bodies are not included due to licensing, and
  article URLs may be expired.

Why it fits:

- It is a strong held-out domain because news has extreme sparsity and temporal
  churn.
- The impression format can evaluate ranking directly within displayed candidate
  sets.
- Title/abstract/category/entities are useful content fields.

Risks:

- The task protocol differs from full-catalog user-item recommendation.
- It may require a separate evaluator for impression ranking.
- Full article text is not reliably available.
- Candidate sets are impression-provided, not generated by the same pipeline as
  MovieLens/Amazon.

Recommended use:

- treat MIND as a later protocol-extension experiment;
- do not block the main cross-domain matrix on MIND.

## First Two Domains

Recommended first pair:

| Priority | Domain | Why |
|---:|---|---|
| 1 | Steam | direct recommender dataset, manageable item count, direct data files |
| 2 | Yelp | strong metadata/reviews, explicit ratings, very different domain |

Goodreads is useful but should follow after the first pair. MIND should be treated
as a separate protocol branch.

## Proposed Directory Layout

```text
extend_research/
  data/raw/cross_domain/
    steam/
    yelp/
    goodreads/
    mind/
  data/processed/cross_domain/
    steam/
    yelp/
  results/cross_domain_readiness/
  results/cross_domain_density_matrix/
```

Do not commit raw data or generated caches.

## Adapter Requirements

Each domain adapter should emit one normalized interaction table and one item
metadata table:

```text
interactions.parquet
  user_id
  item_id
  timestamp
  rating_or_value
  split_key_fields

items.parquet
  item_id
  title_or_name
  metadata_text
  category_fields
  raw_metadata_json
```

Minimum validation checks:

- no duplicate `(user_id, item_id, timestamp)` rows after normalization;
- item metadata coverage above 95 percent for retained interactions;
- temporal split leaves non-empty train/val/test;
- density statistics computed from train only;
- no use of full-dataset popularity fields as model features.

## First Implementation Plan

### Step 1: Steam adapter smoke test

Add:

- `configs/cross_domain_steam_readiness.json`
- `scripts/inspect_cross_domain_steam.py`
- `docs/Cross_Domain_Steam_Smoke.md`

Run:

1. download or point to Steam raw files;
2. parse a small subset first;
3. report users/items/interactions/density;
4. choose 3 density levels;
5. only then export candidate caches.

### Step 2: Yelp adapter smoke test

Add:

- `configs/cross_domain_yelp_readiness.json`
- `scripts/inspect_cross_domain_yelp.py`
- `docs/Cross_Domain_Yelp_Smoke.md`

Run:

1. parse `review.json` and `business.json` by streaming;
2. produce restaurant-only and all-business statistics;
3. choose the cleaner subset;
4. define temporal split;
5. verify train-only density features.

### Step 3: Cross-domain matrix v1

Only after Steam/Yelp adapters are stable:

- run 3 seeds first;
- use 3 density points per domain;
- expand to 5 seeds only if the first matrix is coherent.

## Stop / Go Criteria

Go to method development if:

- Steam and Yelp both produce clean train/val/test splits;
- at least 3 density levels are feasible per domain;
- baseline expert rankings vary with density;
- V2.2-style global blend remains competitive.

Stop or pivot if:

- preprocessing requires too much domain-specific repair;
- density points are not separable;
- no metadata coverage exists after filtering;
- current expert/checkpoint assumptions cannot transfer.

## Research Claim After Readiness

If Steam + Yelp are feasible, the next claim becomes:

> We test whether a density-conditioned expert-composition law, originally
> observed on movie/book recommendation, generalizes to service and game
> recommendation domains under a shared full-ranking protocol.

This is meaningfully stronger than continuing threshold selector on the current
three datasets.
