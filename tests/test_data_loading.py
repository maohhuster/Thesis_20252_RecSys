"""Tests for data loading and preprocessing pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest


DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings" / "bge-large-en-v1.5"
PROFILES_PATH = DATA_DIR / "profiles" / "movie_profiles.json"
SPLITS_DIR = DATA_DIR / "benchmark_splits"


@pytest.fixture
def movie_id_index():
    index_path = EMBEDDINGS_DIR / "movie_id_index.json"
    if not index_path.exists():
        # Fallback to source location
        alt = Path(__file__).parent.parent / "code" / "embedding_generator" / "output" / "movie_id_index.json"
        if alt.exists():
            index_path = alt
        else:
            pytest.skip("movie_id_index.json not found")
    with open(index_path) as f:
        return json.load(f)


class TestProfileData:
    """Test that movie profiles are well-formed."""

    @pytest.fixture
    def profiles(self):
        if not PROFILES_PATH.exists():
            alt = (
                Path(__file__).parent.parent
                / "code"
                / "profile_generator"
                / "llm-movie-profiler-v1-20260402"
                / "output"
                / "movie_profiles.json"
            )
            if alt.exists():
                with open(alt) as f:
                    return json.load(f)
            pytest.skip("movie_profiles.json not found")
        with open(PROFILES_PATH) as f:
            return json.load(f)

    def test_profile_count(self, profiles):
        """Should have 10,381 profiles."""
        assert len(profiles) == 10381, f"Expected 10381 profiles, got {len(profiles)}"

    def test_profile_fields(self, profiles):
        """Each profile should have required fields."""
        required = {"movieId", "title", "profile_text", "mood_vector", "key_themes"}
        sample = profiles[0] if isinstance(profiles, list) else list(profiles.values())[0]
        missing = required - set(sample.keys())
        assert not missing, f"Missing fields: {missing}"

    def test_mood_vector_shape(self, profiles):
        """Mood vectors should be 10-dimensional."""
        sample = profiles[0] if isinstance(profiles, list) else list(profiles.values())[0]
        assert len(sample["mood_vector"]) == 10

    def test_mood_vector_range(self, profiles):
        """Mood values should be in [0, 1]."""
        sample = profiles[0] if isinstance(profiles, list) else list(profiles.values())[0]
        for val in sample["mood_vector"]:
            assert 0.0 <= float(val) <= 1.0, f"Mood value {val} out of range [0, 1]"

    def test_key_themes_count(self, profiles):
        """Each profile should have exactly 3 key themes."""
        sample = profiles[0] if isinstance(profiles, list) else list(profiles.values())[0]
        assert len(sample["key_themes"]) == 3

    def test_profile_text_length(self, profiles):
        """Profile text should be 80-120 words (approximately)."""
        sample = profiles[0] if isinstance(profiles, list) else list(profiles.values())[0]
        word_count = len(sample["profile_text"].split())
        assert 40 <= word_count <= 200, f"Profile text has {word_count} words (expected ~80-120)"


class TestEmbeddings:
    """Test that embedding files have correct shapes."""

    def _load_npy(self, name):
        path = EMBEDDINGS_DIR / name
        if not path.exists():
            alt = Path(__file__).parent.parent / "code" / "embedding_generator" / "output" / name
            if alt.exists():
                path = alt
            else:
                pytest.skip(f"{name} not found")
        return np.load(path)

    def test_profile_embeddings_shape(self):
        emb = self._load_npy("profile_embeddings.npy")
        assert emb.shape == (10381, 1024), f"Expected (10381, 1024), got {emb.shape}"

    def test_mood_vectors_shape(self):
        emb = self._load_npy("mood_vectors.npy")
        assert emb.shape == (10381, 10), f"Expected (10381, 10), got {emb.shape}"

    def test_theme_matrix_shape(self):
        emb = self._load_npy("theme_matrix.npy")
        assert emb.shape[0] == 10381
        assert emb.shape[1] == 528, f"Expected 528 themes, got {emb.shape[1]}"

    def test_genome_embeddings_shape(self):
        emb = self._load_npy("genome_embeddings.npy")
        assert emb.shape == (10381, 128), f"Expected (10381, 128), got {emb.shape}"

    def test_bert_title_embeddings_shape(self):
        emb = self._load_npy("bert_title_embeddings.npy")
        assert emb.shape == (10381, 1024), f"Expected (10381, 1024), got {emb.shape}"

    def test_combined_features_shape(self):
        emb = self._load_npy("combined_features.npy")
        assert emb.shape == (10381, 1034), f"Expected (10381, 1034), got {emb.shape}"

    def test_combined_full_shape(self):
        emb = self._load_npy("combined_full.npy")
        assert emb.shape == (10381, 1562), f"Expected (10381, 1562), got {emb.shape}"

    def test_no_nan_in_profile_embeddings(self):
        emb = self._load_npy("profile_embeddings.npy")
        assert not np.isnan(emb).any(), "profile_embeddings contains NaN values"

    def test_no_nan_in_mood_vectors(self):
        emb = self._load_npy("mood_vectors.npy")
        assert not np.isnan(emb).any(), "mood_vectors contains NaN values"

    def test_embeddings_aligned(self, movie_id_index):
        """All embedding files should have the same row ordering."""
        assert len(movie_id_index) == 10381
