"""
Configuration for Movie Embedding Generator.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
PROFILE_GENERATOR_ROOT = PROJECT_ROOT.parent / "profile_generator" / "llm-movie-profiler-v1-20260402"

# Input
PROFILES_JSON = PROFILE_GENERATOR_ROOT / "output" / "movie_profiles.json"
ML20M_DIR = PROFILE_GENERATOR_ROOT / "data" / "ml-20m"
GENOME_SCORES_CSV = ML20M_DIR / "genome-scores.csv"
GENOME_TAGS_CSV = ML20M_DIR / "genome-tags.csv"

# Output
OUTPUT_DIR = PROJECT_ROOT / "output" / "bge-large-en-v1.5"

# ──────────────────────────────────────────────────────────────────────────────
# SENTENCE TRANSFORMER MODELS
# ──────────────────────────────────────────────────────────────────────────────
# Model options (trade-off: quality vs speed vs dimensions):
#   Strong (recommended):
#     - "BAAI/bge-large-en-v1.5":          1024-dim, 335M params, MTEB ~64 (default)
#     - "intfloat/e5-large-v2":            1024-dim, 335M params, MTEB ~62
#     - "nomic-ai/nomic-embed-text-v1.5":  768-dim,  137M params, MTEB ~62 (Matryoshka)
#     - "Alibaba-NLP/gte-large-en-v1.5":   1024-dim, 434M params, MTEB ~65
#   Lightweight (fast, lower quality):
#     - "all-MiniLM-L6-v2":                384-dim, 22M params,  MTEB ~56
#     - "all-MiniLM-L12-v2":               384-dim, 33M params,  MTEB ~57
#     - "all-mpnet-base-v2":               768-dim, 110M params, MTEB ~58
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"

# Model registry with native dimensions (for reference/validation)
MODEL_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/e5-large-v2": 1024,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "Alibaba-NLP/gte-large-en-v1.5": 1024,
}

# ──────────────────────────────────────────────────────────────────────────────
# EMBEDDING SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
# PCA target dimensions for profile embeddings (None = no reduction, keep native dims)
# Use --pca-dims 128 or --pca-dims 256 to enable PCA reduction
PCA_DIMS = None

# Batch size for sentence-transformer encoding
ENCODE_BATCH_SIZE = 256

# ──────────────────────────────────────────────────────────────────────────────
# MOOD VECTOR AXES (must match profile_generator config)
# ──────────────────────────────────────────────────────────────────────────────
MOOD_AXES = [
    "dark_light",
    "serious_playful",
    "slow_fast",
    "cerebral_visceral",
    "realistic_fantastical",
    "intimate_epic",
    "conventional_experimental",
    "emotional_detached",
    "nostalgic_contemporary",
    "predictable_subversive",
]

# ──────────────────────────────────────────────────────────────────────────────
# GENOME TAG SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
GENOME_PCA_DIMS = 128  # PCA target for 1128-dim genome vectors

# ──────────────────────────────────────────────────────────────────────────────
# THEME FILTERING
# ──────────────────────────────────────────────────────────────────────────────
THEME_MIN_COUNT = 10  # Only keep themes appearing in >= 10 movies
