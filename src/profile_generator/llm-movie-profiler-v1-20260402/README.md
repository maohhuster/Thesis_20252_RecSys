# LLM Movie Profile Generator for MovieLens 20M

Generate **embedding-optimized** 80–120 word movie profiles for the **10,381 genome-covered movies** in MovieLens 20M using **Claude Haiku 4.5**, integrating top-30 genome tags by relevance score with crawled TMDb metadata. Each profile is designed to maximize downstream sentence-transformer embedding quality for collaborative filtering augmentation.

Embedding-optimized profiles with 10-axis mood vector.

## Architecture

```
┌───────────────────┐     ┌──────────────┐     ┌───────────────────────────┐
│   ML-20M Data     │     │   TMDb API   │     │  Claude Haiku 4.5         │
│                   │     │              │     │                           │
│ genome-scores.csv │     │ overview     │     │  System Prompt (cached)   │
│ genome-tags.csv   │────▶│ cast/crew    │────▶│  + 2 few-shot examples    │
│ movies.csv        │     │ keywords     │     │  + User Prompt (per movie)│
│ ratings.csv       │     │ runtime      │     │                           │
│ tags.csv          │     │ vote_average │     │  Prompt caching: ON       │
│ links.csv         │     └──────────────┘     │  (~90% input savings)     │
└───────────────────┘                          └─────────────┬─────────────┘
                                                             │
                        ┌────────────────────────────────────┘
                        ▼
              ┌───────────────────────────────┐
              │      JSON Output (per movie)  │
              │                               │
              │  profile     80–120 words     │ → sentence-transformer → embedding
              │  key_themes  3 strings        │ → categorical features
              │  mood_vector 10 float axes    │ → direct 10-dim numeric feature
              │  word_count  integer          │ → quality metric
              └───────────────────────────────┘
```

## Profile Design Principles

The profile text is **not** a movie summary. It is an **embedding-optimized semantic fingerprint** engineered for a sentence-transformer to produce maximally discriminative item vectors.

| Principle | Implementation | Why It Matters |
|---|---|---|
| Lead with thematic essence | Sentence 1 = movie's core identity | Sentence-transformers weight early tokens more heavily; the first sentence anchors the embedding |
| Exclude structured metadata | No cast, director, year, ratings, runtime | These are separate feature columns; repeating them wastes embedding capacity on redundant information |
| Consistent semantic flow | theme → tone → style → distinction → audience | Ensures analogous information occupies similar positions across all 10,381 embeddings |
| Tight word count (80–120) | Well within sentence-transformer token windows (256–512 tokens) | Denser text produces sharper, less diluted embeddings than padding to 200 words |
| Discriminative vocabulary | Bans: "great film", "well-made", "iconic", "masterpiece" | Generic praise carries zero embedding information; every word must distinguish this movie from others |

## Quick Start

```bash
# 1. Install dependencies
cd llm-movie-profiler
pip install -r requirements.txt

# 2. Download MovieLens 20M
mkdir -p data && cd data
wget https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip
cd ..

# 3. Set API keys in .env file (loaded automatically via python-dotenv)
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
echo 'TMDB_API_KEY=your-tmdb-api-key' >> .env   # Free at https://www.themoviedb.org/settings/api

# 4. Dry run — verify prompts without calling Claude
python main.py --dry-run --limit 3

# 5. Full run — all 10,381 genome-covered movies
python main.py

# 6. Resume after interruption
python main.py --resume

# 7. Process specific movies
python main.py --movie-ids 1 50 318 593 2571

# 8. Skip TMDb crawl (use cached metadata only)
python main.py --skip-tmdb --resume
```

## Output Format

Each movie produces a JSON entry in `output/movie_profiles.json`:

```json
{
  "1": {
    "movieId": 1,
    "title": "Toy Story",
    "profile": "A luminous exploration of possessiveness and loyalty within the secret emotional lives of childhood playthings. The narrative channels rivalry and insecurity through a buddy-film structure where a displaced cowboy doll confronts an oblivious space ranger, generating tension from identity crisis rather than external threat. Tonally it balances slapstick physical comedy with genuine existential anxiety about obsolescence and replacement. Its fantasy world-building operates through strict internal rules about toy consciousness, grounding whimsy in emotional realism. Pacing moves briskly through escalating misadventures externalizing inner jealousy. Sits at the intersection of family animation and psychological character study. Rewards audiences who appreciate layered storytelling beneath accessible surfaces.",
    "word_count": 103,
    "key_themes": ["jealousy", "identity", "friendship"],
    "mood_vector": {
      "dark_light": 0.6,
      "serious_playful": 0.4,
      "slow_fast": 0.3,
      "cerebral_visceral": -0.2,
      "realistic_fantastical": 0.7,
      "intimate_epic": -0.3,
      "conventional_experimental": -0.4,
      "emotional_detached": 0.6,
      "nostalgic_contemporary": 0.1,
      "predictable_subversive": 0.2
    }
  }
}
```

Note: the profile contains **no cast names, no director, no year, no ratings** — only thematic and tonal synthesis. Structured metadata is provided as separate features downstream.

## 10-Axis Mood Vector

Each movie receives a 10-dimensional continuous vector on a `[-1.0, 1.0]` scale. This is a **direct numeric feature** — no embedding needed.

| Axis | -1.0 (left pole) | +1.0 (right pole) | Discriminative purpose |
|---|---|---|---|
| `dark_light` | Bleak, grim | Bright, uplifting | Separates *Requiem for a Dream* from *Toy Story* |
| `serious_playful` | Grave, solemn | Comedic, absurd | Separates *Schindler's List* from *Airplane!* |
| `slow_fast` | Contemplative, still | Frenetic, breathless | Separates *2001* from *Mad Max: Fury Road* |
| `cerebral_visceral` | Intellectual, philosophical | Sensory, action-driven | Separates *Primer* from *John Wick* |
| `realistic_fantastical` | Gritty naturalism | Pure fantasy/surreal | Separates *Manchester by the Sea* from *Lord of the Rings* |
| `intimate_epic` | Small-scale, personal | Grand, sweeping | Separates *Before Sunrise* from *Lawrence of Arabia* |
| `conventional_experimental` | Mainstream, familiar | Avant-garde, unconventional | Separates *Marvel* films from *Mulholland Drive* |
| `emotional_detached` | Cold, analytical | Intensely emotional | Separates Kubrick from Spielberg |
| `nostalgic_contemporary` | Classic, retro sensibility | Modern, zeitgeist | Separates period nostalgia from contemporary urgency |
| `predictable_subversive` | Formulaic, safe | Expectation-defying | Separates standard sequels from genre-bending films |

## Prompt Documentation

All prompts are logged to `logs/prompts.jsonl` for full reproducibility.

### System Prompt (~1,327 tokens, cached after first call)

```
You produce embedding-optimized movie profiles for a recommendation system.
Each profile will be encoded by a sentence-transformer into a dense vector
used as an item feature in collaborative filtering. Your output quality
directly determines recommendation accuracy.

RULES — follow exactly:
1. The "profile" field is 80–120 words, one paragraph, no line breaks.
2. NEVER mention: actor/cast names, director names, release year, ratings,
   vote counts, runtime, box office, or awards. These exist as separate
   structured features. Your profile captures ONLY what requires semantic synthesis.
3. Sentence 1: the movie's thematic core and emotional identity.
4. Sentences 2–4: tonal texture, narrative approach, genre positioning,
   and what makes this film distinct from superficially similar ones.
5. Final sentence: the viewing experience or audience sensibility.
6. Every word must be discriminative. Ban: "a great film", "well-made",
   "entertaining", "must-see", "beloved", "iconic", "masterpiece".
7. Synthesize genome tags and plot into insight — never list tags or
   parrot the plot synopsis.

SEMANTIC FLOW (same order for every movie):
  thematic essence → emotional tone → narrative/visual style
  → genre distinction → audience experience

MOOD VECTOR — 10 axes, each float from -1.0 to 1.0:
  [dark_light, serious_playful, slow_fast, cerebral_visceral,
   realistic_fantastical, intimate_epic, conventional_experimental,
   emotional_detached, nostalgic_contemporary, predictable_subversive]

+ 2 few-shot examples (sci-fi and romcom) demonstrating exact output format
```

The full prompt text including few-shot examples is in `config/settings.py` → `SYSTEM_PROMPT`.

### User Prompt Template (per movie, ~306 tokens)

```
MOVIE: {title} ({year}) | {genres}

GENOME TAGS (top-30 by relevance):
   1. animation: 0.9876
   2. pixar: 0.9654
   ... (30 tags total)

PLOT: {overview}
KEYWORDS: {keywords}
DIRECTOR: {directors} | CAST: {cast} | RUNTIME: {runtime}min
SCORES: TMDb {vote_average}/10 ({vote_count}v) | ML {ml_avg_rating}/5 ({ml_rating_count}r)
USER TAGS: {user_tags_summary}

Generate the JSON profile. 80–120 words. No cast/director/year/ratings in profile text.
```

All metadata is provided so the LLM can *reason about it*, but the system prompt rules ensure only thematic/tonal synthesis appears in the profile text.

## Cost Estimate

| Component | Volume | Unit Cost | Est. Cost |
|---|---|---|---|
| TMDb API | 10,381 calls | Free (40 req/10s) | $0 |
| System prompt — cache write (1x) | ~1,327 tokens | $1.25/MTok | $0.002 |
| System prompt — cache reads (10,380x) | ~13.8M tokens | $0.10/MTok | $1.38 |
| User prompts | ~3.2M tokens | $1.00/MTok | $3.20 |
| Output (profiles + JSON) | ~2.6M tokens | $5.00/MTok | $13.00 |
| **Total (standard API with prompt caching)** | **10,381 movies** | | **~$14** |

Claude Haiku 4.5 pricing: $1/MTok input, $5/MTok output.
Prompt caching: cache writes at 1.25x base, cache reads at 0.1x base.
Estimated runtime at 200 RPM: **~52 minutes**.

### Why Standard API Instead of Batch API

The Anthropic Batch API offers a 50% discount on both input and output tokens, but it is **not cost-effective for this workload** due to prompt caching incompatibility.

**Prompt caching requires two conditions:**
1. The same cached content is sent in consecutive requests.
2. Requests hit the same server within the cache TTL (~5 minutes).

With the **standard API**, requests are sent sequentially from the client, and Anthropic routes them to the same infrastructure. The ~1,327-token system prompt gets cached after the first call and all subsequent 10,380 calls read from cache at 0.1x input price.

With the **Batch API**, Anthropic processes all requests **in parallel across many independent workers**. Each worker has no access to another worker's cache, so the system prompt is charged at full input price for every single request — resulting in 0 cache hits.

| API Mode | Input Cost | Output Cost | Total |
|---|---|---|---|
| Standard API (with prompt caching) | ~$4.58 (cache saves ~$9.22) | ~$13.00 | **~$14** |
| Batch API (no caching, 50% off) | ~$10.48 | ~$10.29 | **~$21** |

For this workload where a large system prompt (~1,327 tokens) is shared across all 10,381 requests, the standard API with prompt caching is **~33% cheaper** than the Batch API.

## Project Structure

```
llm-movie-profiler/
├── main.py                  # Entry point — orchestrates the full pipeline
├── config/
│   ├── __init__.py
│   └── settings.py          # All configuration: prompts, API params, mood axes, paths
├── data_loader.py           # ML-20M data loading & top-30 genome tag extraction
├── tmdb_crawler.py          # Async TMDb API crawler with disk caching
├── profile_generator.py     # Claude API client with prompt caching, validation, checkpointing
├── requirements.txt         # Pinned Python dependencies
├── data/
│   └── ml-20m/              # Place ML-20M CSV files here (6 files)
├── cache/
│   └── tmdb_metadata.json   # Cached TMDb responses (auto-generated)
├── output/
│   ├── movie_profiles.json          # Final output — all 10,381 profiles
│   └── movie_profiles_partial.json  # Checkpoint file (auto-generated)
└── logs/
    ├── prompts.jsonl         # Every prompt + response logged (reproducibility)
    └── run_*.log             # Timestamped execution logs
```

## Resume & Fault Tolerance

- **Prompt caching:** The ~1,327-token system prompt is cached on Anthropic's infrastructure after the first API call. All subsequent 10,380 calls read from cache at 0.1× input price.
- **Disk checkpointing:** Profiles are saved every 50 movies (configurable via `CHECKPOINT_EVERY`). Resume with `python main.py --resume`.
- **TMDb cache:** All TMDb API responses are cached to `cache/tmdb_metadata.json`. Re-runs skip previously fetched movies.
- **Retry with backoff:** Failed Claude API calls are retried up to 3 times. Rate limit errors trigger progressive wait (30s, 60s, 90s).
- **Output validation:** Every LLM response is parsed and validated against:
  - JSON schema (6 required fields)
  - Word count (80–120, with ±10/15 tolerance)
  - Mood vector completeness (all 10 axes present)
  - Mood value range (each axis within [-1.0, 1.0])
  - Failed validation triggers automatic retry with the same prompt.

## Known Issue: Malformed `movieId` for Number-in-Title Movies

During generation, 22 out of 10,381 movies consistently produced invalid JSON on all retry attempts. The root cause: **when a movie title contains a number, Haiku generates a slug-like string instead of the integer `movieId`**.

**Example — "Gone in 60 Seconds" (movieId: 26322):**

What the model **should** output:
```json
{"movieId": 26322, "title": "Gone in 60 Seconds", ...}
```

What the model **actually** output:
```json
{"movieId": 60seconds_1974, "title": "Gone in 60 Seconds", ...}
```

`60seconds_1974` is not valid JSON — it's a bare unquoted string mixing the title's number with the year. This causes `json.loads()` to fail with `Expecting ',' delimiter` at character 17.

**All 22 affected movies have numbers in their titles:**

| Movie ID | Title | Bad `movieId` output |
|---|---|---|
| 277 | Miracle on **34th** Street | `34street1994` |
| 2019 | **Seven** Samurai | `7samurai1954` |
| 2144 | **Sixteen** Candles | `16candles1984` |
| 4876 | **Thirteen** Ghosts | `13ghosts2001` |
| 26322 | Gone in **60** Seconds | `60seconds_1974` |
| 54997 | **3:10** to Yuma | `310toyuma2007` |
| 77846 | **12** Angry Men | `12_angry_men_1997` |
| ... | *(15 more movies)* | *(same pattern)* |

**Why retries didn't help:** The same movie + same prompt = same confusion. Haiku consistently made this mistake for these specific titles across all 3 retry attempts.

**Fix applied in `profile_generator.py`:** Since we already override `movieId` with the correct value after parsing, a regex pre-processor replaces whatever the model put in the `movieId` field with the correct integer *before* JSON parsing:

```python
text = re.sub(r'"movieId"\s*:\s*[^,}\]]+', f'"movieId": {movie_id}', text, count=1)
```

This turns `"movieId": 60seconds_1974` into `"movieId": 26322`, making the JSON valid. After applying this fix, all 22 movies succeeded on the first attempt.

## Downstream Usage

The generated `movie_profiles.json` provides **two complementary feature types** for recommendation models:

### Feature A: Profile Text → Dense Embedding (semantic)

```python
from sentence_transformers import SentenceTransformer
import json, numpy as np

# Load profiles
with open("output/movie_profiles.json") as f:
    profiles = json.load(f)

# Encode with sentence-transformer (default: bge-large-en-v1.5, 1024-dim)
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
texts = [profiles[mid]["profile"] for mid in sorted(profiles.keys())]
embeddings = model.encode(texts, show_progress_bar=True)  # (10381, 1024)

# Optional: reduce dimensionality via PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=128)
# embeddings = pca.fit_transform(embeddings)  # (10381, 128)
```

Use as item side features in LightGCN, DCN-V2, or two-tower models.

### Feature B: Mood Vector → Direct 10-Dim Numeric Feature

```python
# Extract mood vectors — ready-to-use, no embedding needed
mood_matrix = np.array([
    [profiles[mid]["mood_vector"][axis] for axis in [
        "dark_light", "serious_playful", "slow_fast", "cerebral_visceral",
        "realistic_fantastical", "intimate_epic", "conventional_experimental",
        "emotional_detached", "nostalgic_contemporary", "predictable_subversive",
    ]]
    for mid in sorted(profiles.keys())
])  # (10381, 10)
```

Usage in recommendation models:

| Method | How | When |
|---|---|---|
| **Direct concatenation** | Append 10-dim mood to item embeddings | Default — always include |
| **User preference matching** | Avg mood of user's liked items → cosine sim with candidates | Cold-start users, content-based fallback |
| **Cluster-gated routing** | K-means on mood vectors → per-cluster models | Heterogeneous user populations |
| **Interaction features** | Element-wise `|user_avg_mood - candidate_mood|` → ranking MLP input | Feature-crossing in DCN-V2 |

### Recommended Ablation Structure

| Experiment | Item Features | Added Dims | Expected Signal |
|---|---|---|---|
| Baseline: LightGCN (ID only) | Collaborative signals only | d | Baseline |
| + Genome raw (1128-dim PCA→128) | ID + genome vector | d + 128 | Known-strong content signal |
| + BERT(title+genre) | ID + BERT encoding | d + 1024 | Simpler text encoder baseline |
| + **LLM profile embedding** | ID + profile embedding | d + 1024 | Core contribution |
| + **LLM mood vector** | ID + 10-dim mood | d + 10 | Lightweight structured signal |
| + **Profile + mood combined** | ID + profile + mood | d + 1034 | Full LLM feature set |
| LLM profile only (no CF) | Profile embedding only | 1024 | Content-only floor |

The critical comparison is **LLM profile embedding vs. genome raw PCA** — this directly answers: *does having an LLM reason about genome tags produce a better representation than just feeding the 1,128-dim vector as-is?*

## Research Context

This pipeline targets a gap in the LLM-for-recommendation literature: to our knowledge, prior LLM-as-feature-engineer work (e.g., LLMRec, WSDM 2024; RLMRec, WWW 2024; A-LLMRec, KDD 2024) is evaluated on ML-1M, heavily filtered ML-10M subsets, or Amazon, and does not exploit MovieLens's genome-tag annotations with LLMs. We synthesise profiles from genome tags + TMDb metadata over the full genome-covered ML-20M catalogue. See the paper's related-work section for the complete positioning.
