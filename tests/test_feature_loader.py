"""Tests for feature loading and mapping."""

import json
from pathlib import Path

import numpy as np
import pytest


class TestMovieIdIndex:
    """Verify movie ID index consistency."""

    @pytest.fixture
    def index_path(self):
        candidates = [
            Path(__file__).parent.parent / "data" / "embeddings" / "movie_id_index.json",
            Path(__file__).parent.parent / "code" / "embedding_generator" / "output" / "movie_id_index.json",
        ]
        for p in candidates:
            if p.exists():
                return p
        pytest.skip("movie_id_index.json not found")

    def test_index_count(self, index_path):
        with open(index_path) as f:
            index = json.load(f)
        assert len(index) == 10381

    def test_index_unique(self, index_path):
        with open(index_path) as f:
            index = json.load(f)
        ids = [entry["movieId"] if isinstance(entry, dict) else entry for entry in index]
        assert len(set(ids)) == len(ids), "Duplicate movieIds in index"


class TestThemeVocabulary:
    """Verify theme vocabulary consistency."""

    @pytest.fixture
    def vocab_path(self):
        candidates = [
            Path(__file__).parent.parent / "data" / "embeddings" / "theme_vocabulary.json",
            Path(__file__).parent.parent / "code" / "embedding_generator" / "output" / "theme_vocabulary.json",
        ]
        for p in candidates:
            if p.exists():
                return p
        pytest.skip("theme_vocabulary.json not found")

    def test_vocab_count(self, vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        assert len(vocab) == 528, f"Expected 528 themes, got {len(vocab)}"

    def test_vocab_unique(self, vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        assert len(set(vocab)) == len(vocab), "Duplicate themes in vocabulary"

    def test_theme_matrix_matches_vocab(self, vocab_path):
        """Theme matrix columns should match vocabulary size."""
        theme_candidates = [
            Path(__file__).parent.parent / "data" / "embeddings" / "bge-large-en-v1.5" / "theme_matrix.npy",
            Path(__file__).parent.parent / "code" / "embedding_generator" / "output" / "theme_matrix.npy",
        ]
        theme_path = None
        for p in theme_candidates:
            if p.exists():
                theme_path = p
                break
        if theme_path is None:
            pytest.skip("theme_matrix.npy not found")

        with open(vocab_path) as f:
            vocab = json.load(f)
        matrix = np.load(theme_path)
        assert matrix.shape[1] == len(vocab)
