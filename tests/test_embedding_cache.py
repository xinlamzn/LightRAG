"""Tests for the EmbeddingFunc embedding cache."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lightrag.utils import EmbeddingFunc


@pytest.fixture
def embedding_dim():
    return 4


@pytest.fixture
def make_embedding_func(embedding_dim):
    """Create an EmbeddingFunc with a mock underlying function."""

    def _make(model_name="test-model"):
        call_count = 0
        call_args_log = []

        async def mock_embed(texts, **kwargs):
            nonlocal call_count
            call_count += 1
            call_args_log.append(texts)
            # Return deterministic embeddings based on text content
            return np.array(
                [[float(hash(t) % 100) / 100.0] * embedding_dim for t in texts],
                dtype=np.float32,
            )

        ef = EmbeddingFunc(
            embedding_dim=embedding_dim,
            func=mock_embed,
            model_name=model_name,
        )
        return ef, lambda: call_count, call_args_log

    return _make


@pytest.fixture
def mock_cache():
    """Create a mock KV storage for the embedding cache."""
    cache = AsyncMock()
    cache.get_by_ids = AsyncMock(return_value=[])
    cache.upsert = AsyncMock()
    return cache


class TestCacheKeyFormat:
    def test_key_includes_model_name_dim_and_hash(self, make_embedding_func):
        ef, _, _ = make_embedding_func("my-model")
        key = ef._compute_embedding_cache_key("hello world")
        parts = key.split(":")
        assert parts[0] == "emb"
        assert parts[1] == "my-model"
        assert parts[2] == str(ef.embedding_dim)
        assert len(parts[3]) == 32  # MD5 hex digest

    def test_different_texts_different_keys(self, make_embedding_func):
        ef, _, _ = make_embedding_func()
        key1 = ef._compute_embedding_cache_key("text A")
        key2 = ef._compute_embedding_cache_key("text B")
        assert key1 != key2

    def test_same_text_same_key(self, make_embedding_func):
        ef, _, _ = make_embedding_func()
        key1 = ef._compute_embedding_cache_key("identical")
        key2 = ef._compute_embedding_cache_key("identical")
        assert key1 == key2

    def test_model_name_in_key(self, make_embedding_func):
        ef_a, _, _ = make_embedding_func("model-a")
        ef_b, _, _ = make_embedding_func("model-b")
        key_a = ef_a._compute_embedding_cache_key("same text")
        key_b = ef_b._compute_embedding_cache_key("same text")
        assert key_a != key_b
        assert "model-a" in key_a
        assert "model-b" in key_b

    def test_unknown_model_name(self, make_embedding_func, embedding_dim):
        async def dummy(texts, **kw):
            return np.zeros((len(texts), embedding_dim), dtype=np.float32)

        ef = EmbeddingFunc(embedding_dim=embedding_dim, func=dummy, model_name=None)
        key = ef._compute_embedding_cache_key("text")
        assert ":unknown:" in key


class TestAllCacheMiss:
    @pytest.mark.asyncio
    async def test_all_miss_calls_func_and_saves(
        self, make_embedding_func, mock_cache, embedding_dim
    ):
        ef, get_count, call_args = make_embedding_func()
        ef._embedding_cache_kv = mock_cache
        texts = ["alpha", "beta", "gamma"]
        mock_cache.get_by_ids.return_value = [None, None, None]

        result = await ef(texts)

        assert get_count() == 1
        assert call_args[0] == texts
        assert result.shape == (3, embedding_dim)
        mock_cache.upsert.assert_called_once()
        saved = mock_cache.upsert.call_args[0][0]
        assert len(saved) == 3


class TestAllCacheHit:
    @pytest.mark.asyncio
    async def test_all_hit_skips_func(
        self, make_embedding_func, mock_cache, embedding_dim
    ):
        ef, get_count, _ = make_embedding_func()
        ef._embedding_cache_kv = mock_cache
        texts = ["alpha", "beta"]

        cached_vecs = [
            {"embedding": [0.1] * embedding_dim},
            {"embedding": [0.2] * embedding_dim},
        ]
        mock_cache.get_by_ids.return_value = cached_vecs

        result = await ef(texts)

        assert get_count() == 0  # Real func NOT called
        assert result.shape == (2, embedding_dim)
        np.testing.assert_allclose(result[0], [0.1] * embedding_dim, atol=1e-6)
        np.testing.assert_allclose(result[1], [0.2] * embedding_dim, atol=1e-6)
        mock_cache.upsert.assert_not_called()


class TestPartialCacheHit:
    @pytest.mark.asyncio
    async def test_partial_hit_calls_func_for_misses_only(
        self, make_embedding_func, mock_cache, embedding_dim
    ):
        ef, get_count, call_args = make_embedding_func()
        ef._embedding_cache_kv = mock_cache
        texts = ["alpha", "beta", "gamma"]

        # alpha cached, beta miss, gamma cached
        mock_cache.get_by_ids.return_value = [
            {"embedding": [0.1] * embedding_dim},
            None,
            {"embedding": [0.3] * embedding_dim},
        ]

        result = await ef(texts)

        assert get_count() == 1
        # Only "beta" should have been passed to real func
        assert call_args[0] == ["beta"]
        assert result.shape == (3, embedding_dim)
        # First and third should be from cache
        np.testing.assert_allclose(result[0], [0.1] * embedding_dim, atol=1e-6)
        np.testing.assert_allclose(result[2], [0.3] * embedding_dim, atol=1e-6)
        # Upsert should only save the miss
        saved = mock_cache.upsert.call_args[0][0]
        assert len(saved) == 1


class TestCacheDisabled:
    @pytest.mark.asyncio
    async def test_no_cache_calls_func_directly(
        self, make_embedding_func, embedding_dim
    ):
        ef, get_count, call_args = make_embedding_func()
        # _embedding_cache_kv is None by default
        assert ef._embedding_cache_kv is None

        texts = ["alpha", "beta"]
        result = await ef(texts)

        assert get_count() == 1
        assert call_args[0] == texts
        assert result.shape == (2, embedding_dim)
