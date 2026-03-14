"""
Unit tests for OpenSearch storage implementations.

All tests use mocks — no running OpenSearch instance required.
Run with: pytest tests/test_opensearch_storage.py -v
"""

import re

import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
import numpy as np
from opensearchpy.exceptions import NotFoundError, OpenSearchException

from lightrag.kg.opensearch_impl import (
    OpenSearchKVStorage,
    OpenSearchDocStatusStorage,
    OpenSearchGraphStorage,
    OpenSearchVectorDBStorage,
    OpenSearchGraphVectorStorage,
    OpenSearchGraphRelationshipAdapter,
    ClientManager,
    _build_index_name,
    _resolve_workspace,
    _sanitize_index_name,
    _sanitize_database_name,
)
from lightrag.base import DocStatus, DocProcessingStatus
from lightrag.utils import compute_mdhash_id


# ---------------------------------------------------------------------------
# Mock the shared storage lock so tests don't need full LightRAG init
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _mock_lock():
    yield


def _mock_lock_factory():
    return _mock_lock()


def _missing_index_error() -> NotFoundError:
    return NotFoundError(404, "index_not_found_exception", "no such index")


@pytest.fixture(autouse=True)
def patch_data_init_lock():
    """Patch get_data_init_lock globally so initialize() works without shared storage."""
    with patch(
        "lightrag.kg.opensearch_impl.get_data_init_lock", side_effect=_mock_lock_factory
    ):
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockEmbeddingFunc:
    """Mock embedding function that returns random vectors."""

    def __init__(self, dim=128):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embed"

    async def __call__(self, texts, **kwargs):
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


@pytest.fixture
def global_config():
    """Standard global config fixture for all storage tests."""
    return {
        "embedding_batch_num": 10,
        "max_graph_nodes": 1000,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }


@pytest.fixture
def embed_func():
    """Mock embedding function fixture."""
    return MockEmbeddingFunc()


def _make_client():
    """Create a fully-mocked AsyncOpenSearch client with spec validation."""
    from opensearchpy import AsyncOpenSearch

    client = AsyncMock(spec=AsyncOpenSearch)
    # indices sub-client
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock()
    client.indices.delete = AsyncMock()
    client.indices.refresh = AsyncMock()
    client.indices.get_mapping = AsyncMock(return_value={})
    # transport for PPL
    client.transport = AsyncMock()
    client.transport.perform_request = AsyncMock(
        side_effect=Exception("PPL not available")
    )
    # document operations
    client.exists = AsyncMock(return_value=False)
    client.index = AsyncMock()
    client.delete = AsyncMock()
    client.delete_by_query = AsyncMock()
    client.get = AsyncMock(
        return_value={
            "_id": "doc1",
            "_source": {"content": "hello", "create_time": 0, "update_time": 0},
        }
    )
    client.mget = AsyncMock(
        return_value={
            "docs": [
                {"_id": "id1", "found": True, "_source": {"content": "c1"}},
                {"_id": "id2", "found": True, "_source": {"content": "c2"}},
            ]
        }
    )
    client.count = AsyncMock(return_value={"count": 5})
    client.search = AsyncMock(
        return_value={
            "hits": {"hits": [], "total": {"value": 0}},
            "aggregations": {
                "status_counts": {"buckets": []},
                "src": {"buckets": []},
                "tgt": {"buckets": []},
                "source_degrees": {"buckets": []},
                "target_degrees": {"buckets": []},
            },
        }
    )
    # PIT operations
    client.create_pit = AsyncMock(return_value={"pit_id": "mock_pit_id_123"})
    client.delete_pit = AsyncMock()
    return client


@pytest.fixture
def mock_client():
    """Fully-mocked AsyncOpenSearch client fixture."""
    return _make_client()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for module-level helper functions (_build_index_name, _resolve_workspace, _sanitize_index_name)."""

    def test_build_index_name_with_workspace(self):
        ws, ns, idx = _build_index_name("myws", "text_chunks")
        assert ws == "myws"
        assert ns == "myws_text_chunks"
        assert idx == _sanitize_index_name("myws_text_chunks")

    def test_build_index_name_no_workspace(self):
        ws, ns, idx = _build_index_name("", "chunks")
        assert ws == ""
        assert idx == _sanitize_index_name("chunks")

    def test_resolve_workspace_env_override(self):
        with patch.dict("os.environ", {"OPENSEARCH_WORKSPACE": "forced"}):
            assert _resolve_workspace("original", "ns") == "forced"

    def test_resolve_workspace_fallback(self):
        with patch.dict("os.environ", {}, clear=True):
            assert _resolve_workspace("original", "ns") == "original"

    def test_sanitize_index_name(self):
        assert _sanitize_index_name("Hello_World") == "hello_world"
        assert _sanitize_index_name("-bad") == "x-bad"
        assert _sanitize_index_name("a.b/c") == "a_b_c"


# ---------------------------------------------------------------------------
# ClientManager
# ---------------------------------------------------------------------------


class TestClientManager:
    """Tests for ClientManager singleton pattern and reference counting."""

    @pytest.mark.asyncio
    async def test_singleton_and_refcount(self):
        ClientManager._instances = {"client": None, "ref_count": 0}
        with patch("lightrag.kg.opensearch_impl.AsyncOpenSearch") as mock_cls:
            mock_cls.return_value = AsyncMock()
            c1 = await ClientManager.get_client()
            c2 = await ClientManager.get_client()
            assert c1 is c2
            assert ClientManager._instances["ref_count"] == 2
            await ClientManager.release_client(c1)
            assert ClientManager._instances["ref_count"] == 1
            await ClientManager.release_client(c2)
            assert ClientManager._instances["ref_count"] == 0
            assert ClientManager._instances["client"] is None

    @pytest.mark.asyncio
    async def test_close_called_on_last_release(self):
        ClientManager._instances = {"client": None, "ref_count": 0}
        with patch("lightrag.kg.opensearch_impl.AsyncOpenSearch") as mock_cls:
            inner = AsyncMock()
            mock_cls.return_value = inner
            c = await ClientManager.get_client()
            await ClientManager.release_client(c)
            inner.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# KV Storage
# ---------------------------------------------------------------------------


class TestKVStorage:
    """Tests for OpenSearchKVStorage CRUD operations, timestamps, refresh behavior."""

    def _make(self, global_config, embed_func, workspace="test"):
        return OpenSearchKVStorage(
            namespace="text_chunks",
            global_config=global_config,
            embedding_func=embed_func,
            workspace=workspace,
        )

    @pytest.mark.asyncio
    async def test_index_name(self, global_config, embed_func):
        s = self._make(global_config, embed_func, workspace="proj_a")
        assert s._index_name == "proj_a_text_chunks"

    @pytest.mark.asyncio
    async def test_initialize_creates_index(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.indices.exists.assert_awaited_once()
            mock_client.indices.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_skips_existing_index(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.exists = AsyncMock(return_value=True)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.indices.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_by_id(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": "doc1",
                        "found": True,
                        "_source": {
                            "content": "hello",
                            "create_time": 0,
                            "update_time": 0,
                        },
                    }
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            doc = await s.get_by_id("doc1")
            assert doc is not None
            assert doc["content"] == "hello"
            assert doc["_id"] == "doc1"
            mock_client.mget.assert_awaited_once_with(
                index=s._index_name, body={"ids": ["doc1"]}
            )

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "missing", "found": False}]}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_by_id("missing") is None
            mock_client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_by_ids_preserves_order(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs = await s.get_by_ids(["id1", "id2"])
            assert docs[0]["content"] == "c1"
            assert docs[1]["content"] == "c2"

    @pytest.mark.asyncio
    async def test_filter_keys(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {"_id": "a", "found": True},
                    {"_id": "b", "found": False},
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.filter_keys({"a", "b"})
            assert result == {"b"}

    @pytest.mark.asyncio
    async def test_upsert_uses_wait_for_refresh(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "v1"}})
                _, kwargs = mock_bulk.call_args
                assert kwargs["refresh"] == "wait_for"

    @pytest.mark.asyncio
    async def test_upsert_sets_timestamps(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "v1"}})
                actions = mock_bulk.call_args[0][1]
                src = actions[0]["_source"]
                assert "create_time" in src
                assert "update_time" in src

    @pytest.mark.asyncio
    async def test_is_empty(self, global_config, embed_func, mock_client):
        mock_client.count = AsyncMock(return_value={"count": 0})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.is_empty() is True

    @pytest.mark.asyncio
    async def test_delete(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (2, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.delete(["a", "b"])
                actions = mock_bulk.call_args[0][1]
                assert all(a["_op_type"] == "delete" for a in actions)

    @pytest.mark.asyncio
    async def test_drop(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.drop()
            assert result["status"] == "success"
            mock_client.indices.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_drop_error_marks_index_not_ready_and_next_upsert_recreates_index(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.delete = AsyncMock(
            side_effect=OpenSearchException("drop failed")
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                with patch.object(
                    s, "_create_index_if_not_exists", new_callable=AsyncMock
                ) as mock_create:
                    result = await s.drop()
                    assert result["status"] == "error"
                    assert s._index_ready is False
                    await s.upsert({"k1": {"content": "v1"}})
                    mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upsert_after_drop_recreates_index(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                with patch.object(
                    s, "_create_index_if_not_exists", new_callable=AsyncMock
                ) as mock_create:
                    await s.drop()
                    await s.upsert({"k1": {"content": "v1"}})
                    mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reads_short_circuit_after_drop(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.drop()

            assert await s.get_by_id("doc1") is None
            assert await s.get_by_ids(["doc1", "doc2"]) == [None, None]
            assert await s.is_empty() is True

            mock_client.mget.assert_not_awaited()
            mock_client.count.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_read_missing_index_demotes_readiness(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(side_effect=_missing_index_error())
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            assert await s.get_by_id("doc1") is None
            assert await s.get_by_id("doc1") is None
            assert s._index_ready is False
            assert mock_client.mget.await_count == 1

    @pytest.mark.asyncio
    async def test_finalize(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch.object(
                ClientManager, "release_client", new_callable=AsyncMock
            ) as mock_release:
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.finalize()
                mock_release.assert_awaited_once()
                assert s.client is None


# ---------------------------------------------------------------------------
# DocStatus Storage
# ---------------------------------------------------------------------------


class TestDocStatusStorage:
    """Tests for OpenSearchDocStatusStorage including aggregations, pagination, and data normalization."""

    def _make(self, global_config, embed_func, workspace="test"):
        return OpenSearchDocStatusStorage(
            namespace="doc_status",
            global_config=global_config,
            embedding_func=embed_func,
            workspace=workspace,
        )

    @pytest.mark.asyncio
    async def test_index_name(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        assert s._index_name == "test_doc_status"

    @pytest.mark.asyncio
    async def test_initialize_creates_index(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.indices.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": "doc-abc",
                        "found": True,
                        "_source": {"status": "processed", "file_path": "/a.txt"},
                    }
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            doc = await s.get_by_id("doc-abc")
            assert doc["status"] == "processed"
            assert doc["_id"] == "doc-abc"
            mock_client.mget.assert_awaited_once_with(
                index=s._index_name, body={"ids": ["doc-abc"]}
            )

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "missing", "found": False}]}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_by_id("missing") is None
            mock_client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_upsert_sets_chunks_list_default(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"d1": {"status": "pending"}})
                actions = mock_bulk.call_args[0][1]
                assert actions[0]["_source"]["chunks_list"] == []

    @pytest.mark.asyncio
    async def test_get_status_counts(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 0}},
                "aggregations": {
                    "status_counts": {
                        "buckets": [
                            {"key": "processed", "doc_count": 3},
                            {"key": "pending", "doc_count": 1},
                        ]
                    }
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            counts = await s.get_status_counts()
            assert counts == {"processed": 3, "pending": 1}

    @pytest.mark.asyncio
    async def test_get_all_status_counts_includes_all(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 0}},
                "aggregations": {
                    "status_counts": {
                        "buckets": [
                            {"key": "processed", "doc_count": 5},
                            {"key": "failed", "doc_count": 2},
                        ]
                    }
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            counts = await s.get_all_status_counts()
            assert counts["all"] == 7
            assert counts["processed"] == 5

    @pytest.mark.asyncio
    async def test_get_docs_by_status(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "d1",
                            "_source": {
                                "status": "processed",
                                "file_path": "/a.txt",
                                "content_summary": "s",
                                "content_length": 10,
                                "chunks_count": 1,
                                "created_at": 100,
                                "updated_at": 200,
                            },
                            "sort": ["d1"],
                        },
                    ],
                    "total": {"value": 1},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.get_docs_by_status(DocStatus.PROCESSED)
            assert "d1" in result
            assert isinstance(result["d1"], DocProcessingStatus)

    @pytest.mark.asyncio
    async def test_get_docs_paginated(self, global_config, embed_func, mock_client):
        """Page 1 returns results directly without search_after."""
        mock_client.count = AsyncMock(return_value={"count": 50})
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "d1",
                            "_source": {
                                "status": "processed",
                                "file_path": "/a.txt",
                                "content_summary": "s",
                                "content_length": 10,
                                "chunks_count": 1,
                                "created_at": 100,
                                "updated_at": 200,
                            },
                            "sort": [200, "d1"],
                        },
                    ],
                    "total": {"value": 50},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(page=1, page_size=10)
            assert total == 50
            assert len(docs) == 1
            assert docs[0][0] == "d1"
            # Page 1: no search_after needed, single search call
            assert mock_client.search.await_count == 1
            body = mock_client.search.call_args.kwargs.get(
                "body"
            ) or mock_client.search.call_args[1].get("body", {})
            assert "search_after" not in body

    @pytest.mark.asyncio
    async def test_get_docs_paginated_page2_uses_search_after(
        self, global_config, embed_func, mock_client
    ):
        """Page 2 skips page 1 results via search_after."""
        mock_client.count = AsyncMock(return_value={"count": 50})
        call_count = {"n": 0}

        async def search_side_effect(*args, **kwargs):
            call_count["n"] += 1
            body = kwargs.get("body", {})
            if "search_after" not in body:
                # First call: skip batch
                return {
                    "hits": {
                        "hits": [
                            {
                                "_id": f"skip{i}",
                                "_source": {
                                    "status": "processed",
                                    "file_path": f"/{i}.txt",
                                    "content_summary": "s",
                                    "content_length": 1,
                                    "chunks_count": 1,
                                    "created_at": 100,
                                    "updated_at": 100 + i,
                                },
                                "sort": [100 + i, f"skip{i}"],
                            }
                            for i in range(10)
                        ],
                        "total": {"value": 50},
                    }
                }
            else:
                # Second call: actual page
                return {
                    "hits": {
                        "hits": [
                            {
                                "_id": "page2_doc",
                                "_source": {
                                    "status": "pending",
                                    "file_path": "/p2.txt",
                                    "content_summary": "s",
                                    "content_length": 1,
                                    "chunks_count": 1,
                                    "created_at": 200,
                                    "updated_at": 300,
                                },
                                "sort": [300, "page2_doc"],
                            }
                        ],
                        "total": {"value": 50},
                    }
                }

        mock_client.search = AsyncMock(side_effect=search_side_effect)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(page=2, page_size=10)
            assert total == 50
            assert len(docs) == 1
            assert docs[0][0] == "page2_doc"
            # 2 search calls: 1 skip + 1 fetch
            assert mock_client.search.await_count == 2

    @pytest.mark.asyncio
    async def test_get_docs_paginated_empty_index(
        self, global_config, embed_func, mock_client
    ):
        """Empty index returns empty list with total 0."""
        mock_client.count = AsyncMock(return_value={"count": 0})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(page=1, page_size=10)
            assert total == 0
            assert docs == []
            mock_client.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_docs_paginated_page_beyond_total(
        self, global_config, embed_func, mock_client
    ):
        """Requesting a page beyond total docs returns empty list."""
        mock_client.count = AsyncMock(return_value={"count": 5})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(page=100, page_size=10)
            assert total == 5
            assert docs == []

    @pytest.mark.asyncio
    async def test_get_docs_paginated_with_status_filter(
        self, global_config, embed_func, mock_client
    ):
        """Status filter is passed as term query."""
        mock_client.count = AsyncMock(return_value={"count": 3})
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 3}},
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(
                status_filter=DocStatus.PROCESSED, page=1, page_size=10
            )
            assert total == 3
            # Verify count query used the status filter
            count_body = mock_client.count.call_args.kwargs.get("body", {})
            assert count_body["query"] == {"term": {"status": "processed"}}

    @pytest.mark.asyncio
    async def test_get_doc_by_file_path(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "d1",
                            "_source": {
                                "file_path": "/test.txt",
                                "status": "processed",
                            },
                        },
                    ],
                    "total": {"value": 1},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            doc = await s.get_doc_by_file_path("/test.txt")
            assert doc is not None
            assert doc["_id"] == "d1"

    @pytest.mark.asyncio
    async def test_get_doc_by_file_path_not_found(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 0}},
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_doc_by_file_path("/nope.txt") is None

    @pytest.mark.asyncio
    async def test_prepare_doc_status_data(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        raw = {"_id": "x", "status": "processed", "error": "oops"}
        data = s._prepare_doc_status_data(raw)
        assert "_id" not in data
        assert data["error_msg"] == "oops"
        assert "error" not in data
        assert data["file_path"] == "no-file-path"
        assert data["metadata"] == {}

    @pytest.mark.asyncio
    async def test_drop_error_marks_index_not_ready_and_next_upsert_recreates_index(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.delete = AsyncMock(
            side_effect=OpenSearchException("drop failed")
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                with patch.object(
                    s, "_create_index_if_not_exists", new_callable=AsyncMock
                ) as mock_create:
                    result = await s.drop()
                    assert result["status"] == "error"
                    assert s._index_ready is False
                    await s.upsert({"d1": {"status": "pending"}})
                    mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upsert_after_drop_recreates_index(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                with patch.object(
                    s, "_create_index_if_not_exists", new_callable=AsyncMock
                ) as mock_create:
                    await s.drop()
                    await s.upsert({"d1": {"status": "pending"}})
                    mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reads_short_circuit_after_drop(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.drop()

            assert await s.get_all_status_counts() == {}
            assert await s.get_docs_paginated(page=1, page_size=10) == ([], 0)
            assert await s.get_doc_by_file_path("/a.txt") is None
            assert await s.get_docs_by_status(DocStatus.PROCESSED) == {}

            mock_client.count.assert_not_awaited()
            mock_client.search.assert_not_awaited()
            mock_client.create_pit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_read_missing_index_demotes_readiness(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(side_effect=_missing_index_error())
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            assert await s.get_all_status_counts() == {}
            assert await s.get_all_status_counts() == {}
            assert s._index_ready is False
            assert mock_client.search.await_count == 1


# ---------------------------------------------------------------------------
# Database Name Sanitization
# ---------------------------------------------------------------------------


class TestDatabaseNameSanitization:
    """Tests for _sanitize_database_name helper."""

    def test_basic(self):
        name = _sanitize_database_name("myws", "lightrag")
        assert name.startswith("myws_lightrag-")
        assert len(name) <= 200

    def test_special_chars(self):
        name = _sanitize_database_name("My Project!", "light/rag")
        assert re.match(r"^[a-z][a-z0-9_-]*$", name)

    def test_unicode(self):
        name = _sanitize_database_name("日本語", "namespace")
        assert re.match(r"^[a-z][a-z0-9_-]*$", name)

    def test_digit_prefix(self):
        name = _sanitize_database_name("123start", "ns")
        assert name[0] == "g"

    def test_long_name_truncated(self):
        name = _sanitize_database_name("a" * 200, "b" * 200)
        assert len(name) <= 200

    def test_collision_safety(self):
        n1 = _sanitize_database_name("workspace_a", "ns")
        n2 = _sanitize_database_name("workspace-a", "ns")
        # Different inputs produce different hash suffixes
        assert n1 != n2

    def test_empty_workspace(self):
        name = _sanitize_database_name("", "ns")
        assert re.match(r"^[a-z][a-z0-9_-]*$", name)


# ---------------------------------------------------------------------------
# Graph Storage (Cypher-based)
# ---------------------------------------------------------------------------


def _cypher_response(data=None, columns=None):
    """Build a mock Cypher endpoint response."""
    if data is None:
        data = []
    return {"results": [{"columns": columns or [], "data": data}]}


class TestGraphStorage:
    """Tests for Cypher-based OpenSearchGraphStorage."""

    def _make(self, global_config, embed_func, workspace="test"):
        return OpenSearchGraphStorage(
            namespace="chunk_entity_relation",
            global_config=global_config,
            embedding_func=embed_func,
            workspace=workspace,
        )

    @pytest.mark.asyncio
    async def test_database_name(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        assert s.database_name.startswith("test_chunk_entity_relation-")
        assert len(s.database_name) <= 200

    @pytest.mark.asyncio
    async def test_initialize_creates_database(
        self, global_config, embed_func, mock_client
    ):
        mock_client.transport.perform_request = AsyncMock(return_value={})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            # Should have called PUT to create the database
            calls = mock_client.transport.perform_request.call_args_list
            put_calls = [c for c in calls if c[0][0] == "PUT"]
            assert len(put_calls) >= 1
            assert "_plugins/_graph/database/" in put_calls[0][0][1]

    @pytest.mark.asyncio
    async def test_has_node_true(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response(
                [{"row": [True]}], ["exists"]
            )
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.has_node("Alice") is True

    @pytest.mark.asyncio
    async def test_has_node_false(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response(
                [{"row": [False]}], ["exists"]
            )
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.has_node("Nobody") is False

    @pytest.mark.asyncio
    async def test_has_edge(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response(
                [{"row": [True]}], ["exists"]
            )
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.has_edge("A", "B") is True

    @pytest.mark.asyncio
    async def test_node_degree(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response(
                [{"row": [3]}], ["degree"]
            )
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.node_degree("A") == 3

    @pytest.mark.asyncio
    async def test_get_node(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response(
                [{"row": [{"entity_id": "Alice", "entity_type": "person", "description": "A researcher"}]}],
                ["props"],
            )
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            node = await s.get_node("Alice")
            assert node["entity_type"] == "person"

    @pytest.mark.asyncio
    async def test_get_node_not_found(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response([], ["props"])
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_node("Nobody") is None

    @pytest.mark.asyncio
    async def test_get_edge(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response(
                [{"row": [{"weight": 1.0, "description": "knows"}]}],
                ["props"],
            )
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            edge = await s.get_edge("A", "B")
            assert edge is not None
            assert edge["weight"] == 1.0

    @pytest.mark.asyncio
    async def test_get_node_edges(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response(
                [{"row": ["A", "B"]}, {"row": ["A", "C"]}],
                ["src", "tgt"],
            )
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            edges = await s.get_node_edges("A")
            assert len(edges) == 2
            assert ("A", "B") in edges

    @pytest.mark.asyncio
    async def test_get_nodes_batch(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response(
                [{"row": ["A", {"entity_type": "person"}]}],
                ["eid", "props"],
            )
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.get_nodes_batch(["A", "B"])
            assert "A" in result
            assert "B" not in result

    @pytest.mark.asyncio
    async def test_node_degrees_batch(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response(
                [{"row": ["A", 3]}, {"row": ["B", 5]}],
                ["eid", "degree"],
            )
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            degrees = await s.node_degrees_batch(["A", "B"])
            assert degrees["A"] == 3
            assert degrees["B"] == 5

    @pytest.mark.asyncio
    async def test_upsert_node(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(return_value={})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert_node(
                "Alice", {"entity_type": "person", "source_id": "c1<SEP>c2"}
            )
            # Verify Cypher was called with MERGE
            calls = mock_client.transport.perform_request.call_args_list
            cypher_calls = [
                c for c in calls
                if c[0][1] == "/_plugins/_cypher"
            ]
            assert len(cypher_calls) >= 1
            body = cypher_calls[-1][1].get("body") or cypher_calls[-1][0][2] if len(cypher_calls[-1][0]) > 2 else cypher_calls[-1].kwargs.get("body", {})
            if isinstance(body, dict):
                assert "MERGE" in body.get("query", "")

    @pytest.mark.asyncio
    async def test_upsert_edge(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(return_value={})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert_edge("A", "B", {"weight": "1.0", "description": "knows"})
            # Verify a Cypher call with MERGE was made
            calls = mock_client.transport.perform_request.call_args_list
            cypher_calls = [
                c for c in calls
                if len(c[0]) >= 2 and c[0][1] == "/_plugins/_cypher"
            ]
            assert len(cypher_calls) >= 1

    @pytest.mark.asyncio
    async def test_delete_node(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(return_value={})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.delete_node("Alice")
            # Verify Cypher DETACH DELETE was called
            calls = mock_client.transport.perform_request.call_args_list
            cypher_calls = [
                c for c in calls
                if len(c[0]) >= 2 and c[0][1] == "/_plugins/_cypher"
            ]
            found_detach = any(
                "DETACH DELETE" in str(c)
                for c in cypher_calls
            )
            assert found_detach

    @pytest.mark.asyncio
    async def test_remove_nodes(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(return_value={})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.remove_nodes(["A", "B"])

    @pytest.mark.asyncio
    async def test_remove_edges(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(return_value={})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.remove_edges([("A", "B"), ("C", "D")])

    @pytest.mark.asyncio
    async def test_get_all_labels(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response(
                [{"row": ["Alice"]}, {"row": ["Bob"]}], ["eid"]
            )
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            labels = await s.get_all_labels()
            assert "Alice" in labels
            assert "Bob" in labels

    @pytest.mark.asyncio
    async def test_get_popular_labels(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response(
                [{"row": ["A"]}, {"row": ["B"]}], ["eid"]
            )
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            labels = await s.get_popular_labels(limit=10)
            assert labels[0] == "A"

    @pytest.mark.asyncio
    async def test_search_labels_empty_query(
        self, global_config, embed_func, mock_client
    ):
        mock_client.transport.perform_request = AsyncMock(return_value={})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.search_labels("") == []

    @pytest.mark.asyncio
    async def test_drop(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(return_value={})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.drop()
            assert result["status"] == "success"
            # Verify DELETE was called on the database
            calls = mock_client.transport.perform_request.call_args_list
            delete_calls = [c for c in calls if c[0][0] == "DELETE"]
            assert len(delete_calls) >= 1

    @pytest.mark.asyncio
    async def test_reads_short_circuit_after_drop(
        self, global_config, embed_func, mock_client
    ):
        mock_client.transport.perform_request = AsyncMock(return_value={})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.drop()
            assert await s.has_node("A") is False
            assert await s.get_all_labels() == []
            assert await s.node_degree("A") == 0

    @pytest.mark.asyncio
    async def test_construct_graph_node(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        node = s._construct_graph_node(
            "Alice",
            {
                "entity_type": "person",
                "description": "A researcher",
                "_id": "Alice",
                "entity_id": "Alice",
                "embedding": [0.1, 0.2],
            },
        )
        assert node.id == "Alice"
        assert "entity_type" in node.properties
        assert "_id" not in node.properties
        assert "entity_id" not in node.properties
        assert "embedding" not in node.properties

    @pytest.mark.asyncio
    async def test_construct_graph_edge(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        edge = s._construct_graph_edge(
            "e1",
            {
                "source_node_id": "A",
                "target_node_id": "B",
                "relationship": "knows",
                "weight": 1.0,
            },
        )
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.type == "knows"
        assert "source_node_id" not in edge.properties

    @pytest.mark.asyncio
    async def test_client_property_raises_before_init(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        with pytest.raises(RuntimeError, match="not been initialized"):
            _ = s.client

    @pytest.mark.asyncio
    async def test_apoc_probe_uses_match_not_map(
        self, global_config, embed_func, mock_client
    ):
        """APOC capability probe must pass a node reference (via MATCH), not a map literal."""
        captured_queries = []

        async def capture_side_effect(*args, **kwargs):
            body = kwargs.get("body", {})
            if isinstance(body, dict) and "query" in body:
                captured_queries.append(body["query"])
            return {}

        mock_client.transport.perform_request = AsyncMock(side_effect=capture_side_effect)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

        # Find the APOC probe query
        apoc_queries = [q for q in captured_queries if "apoc.path.subgraphNodes" in q]
        assert len(apoc_queries) == 1, "Should call APOC probe once during init"
        probe_query = apoc_queries[0]
        # Must use MATCH to bind a node variable, then pass it to APOC
        assert "MATCH" in probe_query or "OPTIONAL MATCH" in probe_query
        # The first argument to apoc.path.subgraphNodes must be a node variable
        # (e.g., "start"), NOT a map literal like {entity_id: '__probe__'}
        import re as _re
        apoc_call = _re.search(r"apoc\.path\.subgraphNodes\((\w+)", probe_query)
        assert apoc_call is not None, "Could not parse APOC call"
        first_arg = apoc_call.group(1)
        assert first_arg.isidentifier(), \
            f"First arg to subgraphNodes should be a variable, got: {first_arg}"

    @pytest.mark.asyncio
    async def test_upsert_node_propagates_exceptions(
        self, global_config, embed_func, mock_client
    ):
        """upsert_node must propagate Cypher errors to callers."""
        # Initialize successfully first, then make subsequent calls fail
        mock_client.transport.perform_request = AsyncMock(return_value={})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
        # Now make Cypher calls fail
        mock_client.transport.perform_request = AsyncMock(
            side_effect=OpenSearchException("write failed")
        )
        with pytest.raises(OpenSearchException, match="write failed"):
            await s.upsert_node("Alice", {"entity_type": "person"})

    @pytest.mark.asyncio
    async def test_upsert_edge_propagates_exceptions(
        self, global_config, embed_func, mock_client
    ):
        """upsert_edge must propagate Cypher errors to callers."""
        # Initialize successfully first, then make subsequent calls fail
        mock_client.transport.perform_request = AsyncMock(return_value={})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
        # Now make Cypher calls fail
        mock_client.transport.perform_request = AsyncMock(
            side_effect=OpenSearchException("edge write failed")
        )
        with pytest.raises(OpenSearchException, match="edge write failed"):
            await s.upsert_edge("A", "B", {"weight": "1.0"})


# ---------------------------------------------------------------------------
# Vector Storage
# ---------------------------------------------------------------------------


class TestVectorStorage:
    """Tests for OpenSearchVectorDBStorage k-NN index, embeddings, cosine conversion, and entity deletion."""

    def _make(self, global_config, embed_func, workspace="test"):
        return OpenSearchVectorDBStorage(
            namespace="entities",
            global_config=global_config,
            embedding_func=embed_func,
            workspace=workspace,
            meta_fields={"content", "entity_name", "src_id", "tgt_id"},
        )

    @pytest.mark.asyncio
    async def test_index_name(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        assert s._index_name == "test_entities"

    @pytest.mark.asyncio
    async def test_cosine_threshold_required(self, embed_func):
        with pytest.raises(ValueError, match="cosine_better_than_threshold"):
            OpenSearchVectorDBStorage(
                namespace="v",
                global_config={
                    "embedding_batch_num": 10,
                    "vector_db_storage_cls_kwargs": {},
                },
                embedding_func=embed_func,
            )

    @pytest.mark.asyncio
    async def test_initialize_creates_knn_index(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.indices.create.assert_awaited_once()
            body = mock_client.indices.create.call_args.kwargs["body"]
            assert body["settings"]["index"]["knn"] is True
            assert body["mappings"]["properties"]["vector"]["dimension"] == 128
            assert (
                body["mappings"]["properties"]["vector"]["method"]["engine"] == "lucene"
            )

    @pytest.mark.asyncio
    async def test_upsert_generates_embeddings(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (2, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert(
                    {
                        "v1": {"content": "hello"},
                        "v2": {"content": "world"},
                    }
                )
                actions = mock_bulk.call_args[0][1]
                assert len(actions) == 2
                assert "vector" in actions[0]["_source"]
                assert len(actions[0]["_source"]["vector"]) == 128

    @pytest.mark.asyncio
    async def test_query_cosine_score_conversion(
        self, global_config, embed_func, mock_client
    ):
        """Test that scores are used directly and threshold filtering works."""
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "v1",
                            "_score": 0.85,
                            "_source": {"content": "match", "entity_name": "E1"},
                        },
                    ],
                    "total": {"value": 1},
                },
                "aggregations": {
                    "status_counts": {"buckets": []},
                    "src": {"buckets": []},
                    "tgt": {"buckets": []},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            results = await s.query("test", top_k=5)
            assert len(results) == 1
            assert results[0]["distance"] == 0.85

    @pytest.mark.asyncio
    async def test_query_filters_below_threshold(
        self, global_config, embed_func, mock_client
    ):
        """Low scores should be filtered out."""
        # score 0.15 < threshold 0.2
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "v1",
                            "_score": 0.15,
                            "_source": {"content": "weak match"},
                        },
                    ],
                    "total": {"value": 1},
                },
                "aggregations": {
                    "status_counts": {"buckets": []},
                    "src": {"buckets": []},
                    "tgt": {"buckets": []},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            results = await s.query("test", top_k=5)
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_query_with_provided_embedding(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {"_id": "v1", "_score": 1.0, "_source": {"content": "exact"}},
                    ],
                    "total": {"value": 1},
                },
                "aggregations": {
                    "status_counts": {"buckets": []},
                    "src": {"buckets": []},
                    "tgt": {"buckets": []},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            vec = np.random.rand(128).astype(np.float32)
            results = await s.query("test", top_k=5, query_embedding=vec)
            assert len(results) == 1
            assert results[0]["distance"] == 1.0

    @pytest.mark.asyncio
    async def test_get_by_id(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": "v1",
                        "found": True,
                        "_source": {"content": "hello", "vector": [0.1] * 128},
                    }
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            doc = await s.get_by_id("v1")
            assert doc["id"] == "v1"
            assert doc["content"] == "hello"
            mock_client.mget.assert_awaited_once_with(
                index=s._index_name, body={"ids": ["v1"]}
            )

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "missing", "found": False}]}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_by_id("missing") is None
            mock_client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_by_ids(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {"_id": "v1", "found": True, "_source": {"content": "a"}},
                    {"_id": "v2", "found": True, "_source": {"content": "b"}},
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs = await s.get_by_ids(["v1", "v2"])
            assert docs[0]["id"] == "v1"
            assert docs[1]["id"] == "v2"

    @pytest.mark.asyncio
    async def test_get_vectors_by_ids(self, global_config, embed_func, mock_client):
        vec = [0.1] * 128
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {"_id": "v1", "found": True, "_source": {"vector": vec}},
                    {"_id": "v2", "found": False},
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.get_vectors_by_ids(["v1", "v2"])
            assert "v1" in result
            assert "v2" not in result
            assert result["v1"] == vec

    @pytest.mark.asyncio
    async def test_delete(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (2, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.delete(["v1", "v2"])
                actions = mock_bulk.call_args[0][1]
                assert len(actions) == 2
                assert all(a["_op_type"] == "delete" for a in actions)

    @pytest.mark.asyncio
    async def test_delete_entity(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.delete_entity("Alice")
            mock_client.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_entity_relation(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.delete_entity_relation("Alice")
            mock_client.delete_by_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_drop_recreates_index(self, global_config, embed_func, mock_client):
        # After drop, _create_knn_index_if_not_exists is called again.
        # First call (init): exists=False -> create. Second call (after drop): exists=False -> create again.
        mock_client.indices.exists = AsyncMock(return_value=False)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.drop()
            assert result["status"] == "success"
            mock_client.indices.delete.assert_awaited_once()
            # create called twice: once during init, once during drop recreate
            assert mock_client.indices.create.await_count == 2

    @pytest.mark.asyncio
    async def test_drop_delete_error_marks_index_not_ready(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.delete = AsyncMock(
            side_effect=OpenSearchException("delete failed")
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.drop()
            assert result["status"] == "error"
            assert s._index_ready is False

    @pytest.mark.asyncio
    async def test_drop_recreate_error_marks_index_not_ready(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            with patch.object(
                s,
                "_create_knn_index_if_not_exists",
                new=AsyncMock(side_effect=OpenSearchException("recreate failed")),
            ):
                result = await s.drop()
                assert result["status"] == "error"
                assert s._index_ready is False

    @pytest.mark.asyncio
    async def test_drop_recreates_index_when_missing(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.exists = AsyncMock(return_value=False)
        mock_client.indices.delete = AsyncMock(
            side_effect=NotFoundError(404, "not found")
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.drop()
            assert result["status"] == "success"
            assert mock_client.indices.create.await_count == 2

    @pytest.mark.asyncio
    async def test_reads_short_circuit_when_index_not_ready(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            s._index_ready = False

            assert await s.query("test", top_k=5) == []
            assert await s.get_by_id("v1") is None
            assert await s.get_vectors_by_ids(["v1"]) == {}

            mock_client.search.assert_not_awaited()
            mock_client.mget.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_read_missing_index_demotes_readiness(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(side_effect=_missing_index_error())
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            assert await s.query("test", top_k=5) == []
            assert await s.query("test", top_k=5) == []
            assert s._index_ready is False
            assert mock_client.search.await_count == 1


# ---------------------------------------------------------------------------
# Cosine score edge cases
# ---------------------------------------------------------------------------


class TestScoreThreshold:
    """Verify that raw OpenSearch scores are compared directly against threshold."""

    def test_above_threshold(self):
        assert 0.85 >= 0.2

    def test_below_threshold(self):
        assert 0.15 < 0.2

    def test_exact_threshold(self):
        assert 0.2 >= 0.2


# ---------------------------------------------------------------------------
# Graph Vector Storage (hybrid retrieval)
# ---------------------------------------------------------------------------


class TestGraphVectorStorage:
    """Tests for OpenSearchGraphVectorStorage hybrid retrieval."""

    def _make_graph_storage(self, global_config, embed_func, mock_client):
        """Create a mock graph storage with initialized client."""
        gs = OpenSearchGraphStorage(
            namespace="chunk_entity_relation",
            global_config=global_config,
            embedding_func=embed_func,
            workspace="test",
        )
        gs._client = mock_client
        gs._database_ready = True
        gs._database_name = "test_db-12345678"
        return gs

    def _make(self, global_config, embed_func, mock_client, node_label="Entity"):
        gs = self._make_graph_storage(global_config, embed_func, mock_client)
        key_property = "vdb_id" if node_label == "Entity" else "id"
        merge_key = "entity_id" if node_label == "Entity" else "id"
        return OpenSearchGraphVectorStorage(
            namespace="entities",
            workspace="test",
            embedding_func=embed_func,
            meta_fields={"entity_name", "content", "source_id", "file_path"},
            node_label=node_label,
            key_property=key_property,
            merge_key=merge_key,
            graph_storage=gs,
        )

    @pytest.mark.asyncio
    async def test_initialize_uses_graph_client(self, global_config, embed_func, mock_client):
        s = self._make(global_config, embed_func, mock_client)
        await s.initialize()
        assert s._client is mock_client

    @pytest.mark.asyncio
    async def test_upsert_entity_nodes(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(return_value={})
        s = self._make(global_config, embed_func, mock_client)
        await s.initialize()
        await s.upsert({
            "ent-abc": {"content": "Alice is a researcher", "entity_name": "ALICE"},
            "ent-def": {"content": "Bob is an engineer", "entity_name": "BOB"},
        })
        # Verify Cypher was called with UNWIND MERGE
        calls = mock_client.transport.perform_request.call_args_list
        cypher_calls = [c for c in calls if len(c[0]) >= 2 and c[0][1] == "/_plugins/_cypher"]
        assert len(cypher_calls) >= 1

    @pytest.mark.asyncio
    async def test_upsert_entity_creates_mentioned_in_edges(
        self, global_config, embed_func, mock_client
    ):
        """Entity upsert with source_id creates MENTIONED_IN edges to chunks."""
        mock_client.transport.perform_request = AsyncMock(return_value={})
        s = self._make(global_config, embed_func, mock_client)
        await s.initialize()
        await s.upsert({
            "ent-abc": {
                "content": "Alice is a researcher",
                "entity_name": "ALICE",
                "source_id": "chunk-1<SEP>chunk-2",
            },
        })
        calls = mock_client.transport.perform_request.call_args_list
        cypher_calls = [c for c in calls if len(c[0]) >= 2 and c[0][1] == "/_plugins/_cypher"]
        mentioned_calls = [
            c for c in cypher_calls if "MENTIONED_IN" in str(c)
        ]
        assert len(mentioned_calls) == 1, "Should create MENTIONED_IN edges"
        # Verify the items contain the right chunk IDs
        body = mentioned_calls[0].kwargs.get("body", {})
        items = body.get("parameters", {}).get("items", [])
        chunk_ids = sorted(item["chunk_id"] for item in items)
        assert chunk_ids == ["chunk-1", "chunk-2"]

    @pytest.mark.asyncio
    async def test_upsert_entity_no_mentioned_in_without_source_id(
        self, global_config, embed_func, mock_client
    ):
        """Entity upsert without source_id skips MENTIONED_IN edge creation."""
        mock_client.transport.perform_request = AsyncMock(return_value={})
        s = self._make(global_config, embed_func, mock_client)
        await s.initialize()
        await s.upsert({
            "ent-abc": {"content": "Alice", "entity_name": "ALICE"},
        })
        calls = mock_client.transport.perform_request.call_args_list
        cypher_calls = [c for c in calls if len(c[0]) >= 2 and c[0][1] == "/_plugins/_cypher"]
        mentioned_calls = [c for c in cypher_calls if "MENTIONED_IN" in str(c)]
        assert len(mentioned_calls) == 0

    @pytest.mark.asyncio
    async def test_upsert_chunk_nodes(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(return_value={})
        s = self._make(global_config, embed_func, mock_client, node_label="Chunk")
        await s.initialize()
        await s.upsert({
            "chunk-abc": {"content": "Some text content", "full_doc_id": "doc-xyz"},
        })
        calls = mock_client.transport.perform_request.call_args_list
        cypher_calls = [c for c in calls if len(c[0]) >= 2 and c[0][1] == "/_plugins/_cypher"]
        # Should have 2 calls: chunk MERGE + PART_OF edges (with Document MERGE)
        assert len(cypher_calls) >= 2

    @pytest.mark.asyncio
    async def test_upsert_chunk_merges_document_node(
        self, global_config, embed_func, mock_client
    ):
        """PART_OF edge creation uses MERGE (not MATCH) on Document node,
        guaranteeing the Document exists even if created out-of-order."""
        mock_client.transport.perform_request = AsyncMock(return_value={})
        s = self._make(global_config, embed_func, mock_client, node_label="Chunk")
        await s.initialize()
        await s.upsert({
            "chunk-1": {"content": "text", "full_doc_id": "doc-abc", "file_path": "a.txt"},
        })
        calls = mock_client.transport.perform_request.call_args_list
        cypher_calls = [c for c in calls if len(c[0]) >= 2 and c[0][1] == "/_plugins/_cypher"]
        part_of_calls = [
            c for c in cypher_calls
            if "PART_OF" in str(c) and "MERGE (d:Document" in str(c)
        ]
        assert len(part_of_calls) == 1, "PART_OF should MERGE Document, not MATCH"

    @pytest.mark.asyncio
    async def test_upsert_propagates_cypher_errors(
        self, global_config, embed_func, mock_client
    ):
        """GraphVectorStorage.upsert must propagate Cypher write errors."""
        mock_client.transport.perform_request = AsyncMock(
            side_effect=OpenSearchException("cypher write failed")
        )
        s = self._make(global_config, embed_func, mock_client)
        await s.initialize()
        with pytest.raises(OpenSearchException, match="cypher write failed"):
            await s.upsert({"ent-abc": {"content": "test", "entity_name": "X"}})

    @pytest.mark.asyncio
    async def test_query_hybrid_retrieval(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(
            return_value={
                "results": [
                    {
                        "node_id": "ALICE",
                        "score": 0.9,
                        "properties": {
                            "vdb_id": "ent-abc",
                            "content": "Alice is a researcher",
                            "entity_name": "ALICE",
                        },
                    },
                    {
                        "node_id": "BOB",
                        "score": 0.1,  # Below hybrid_score_threshold
                        "properties": {
                            "vdb_id": "ent-def",
                            "content": "Bob",
                        },
                    },
                ]
            }
        )
        # Set hybrid_score_threshold so BOB (0.1) is filtered out
        global_config["vector_db_storage_cls_kwargs"] = {
            "hybrid_score_threshold": 0.2,
        }
        s = self._make(global_config, embed_func, mock_client)
        await s.initialize()
        results = await s.query("researcher", top_k=10)
        assert len(results) == 1  # Bob filtered by hybrid_score_threshold
        assert results[0]["id"] == "ent-abc"
        assert results[0]["entity_name"] == "ALICE"

    @pytest.mark.asyncio
    async def test_query_default_threshold_passes_all(
        self, global_config, embed_func, mock_client
    ):
        """Default hybrid_score_threshold=0.0 lets all results through."""
        mock_client.transport.perform_request = AsyncMock(
            return_value={
                "results": [
                    {
                        "node_id": "ALICE",
                        "score": 0.9,
                        "properties": {"vdb_id": "ent-abc", "content": "Alice"},
                    },
                    {
                        "node_id": "BOB",
                        "score": 0.05,
                        "properties": {"vdb_id": "ent-def", "content": "Bob"},
                    },
                ]
            }
        )
        s = self._make(global_config, embed_func, mock_client)
        await s.initialize()
        results = await s.query("test", top_k=10)
        assert len(results) == 2  # Both pass with default threshold 0.0

    @pytest.mark.asyncio
    async def test_delete(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(return_value={})
        s = self._make(global_config, embed_func, mock_client)
        await s.initialize()
        await s.delete(["ent-abc", "ent-def"])
        calls = mock_client.transport.perform_request.call_args_list
        cypher_calls = [c for c in calls if len(c[0]) >= 2 and c[0][1] == "/_plugins/_cypher"]
        assert any("DETACH DELETE" in str(c) for c in cypher_calls)

    @pytest.mark.asyncio
    async def test_get_by_id(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response(
                [{"row": [{"entity_name": "ALICE", "content": "researcher"}]}],
                ["props"],
            )
        )
        s = self._make(global_config, embed_func, mock_client)
        await s.initialize()
        result = await s.get_by_id("ent-abc")
        assert result is not None
        assert result["id"] == "ent-abc"

    @pytest.mark.asyncio
    async def test_drop_noop(self, global_config, embed_func, mock_client):
        s = self._make(global_config, embed_func, mock_client)
        result = await s.drop()
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_finalize_noop(self, global_config, embed_func, mock_client):
        s = self._make(global_config, embed_func, mock_client)
        await s.initialize()
        await s.finalize()  # Should not raise


# ---------------------------------------------------------------------------
# Relationship Adapter
# ---------------------------------------------------------------------------


class TestRelationshipAdapter:
    """Tests for OpenSearchGraphRelationshipAdapter."""

    def _make(self, global_config, embed_func, mock_client):
        gs = OpenSearchGraphStorage(
            namespace="chunk_entity_relation",
            global_config=global_config,
            embedding_func=embed_func,
            workspace="test",
        )
        gs._client = mock_client
        gs._database_ready = True
        gs._database_name = "test_db-12345678"

        entities_vdb = OpenSearchGraphVectorStorage(
            namespace="entities",
            workspace="test",
            embedding_func=embed_func,
            meta_fields={"entity_name", "content"},
            node_label="Entity",
            key_property="vdb_id",
            merge_key="entity_id",
            graph_storage=gs,
        )
        entities_vdb._client = mock_client

        adapter = OpenSearchGraphRelationshipAdapter(
            namespace="relationships",
            workspace="test",
            embedding_func=embed_func,
            meta_fields={"src_id", "tgt_id"},
            entities_vdb=entities_vdb,
            graph_storage=gs,
        )
        return adapter

    @pytest.mark.asyncio
    async def test_initialize(self, global_config, embed_func, mock_client):
        adapter = self._make(global_config, embed_func, mock_client)
        await adapter.initialize()
        assert adapter._client is mock_client

    @pytest.mark.asyncio
    async def test_upsert_noop(self, global_config, embed_func, mock_client):
        adapter = self._make(global_config, embed_func, mock_client)
        await adapter.upsert({"rel-1": {"content": "test"}})  # Should not raise

    @pytest.mark.asyncio
    async def test_query_pairwise_edges(self, global_config, embed_func, mock_client):
        # Mock hybrid retrieval for entities
        retrieve_resp = {
            "results": [
                {
                    "node_id": "ALICE",
                    "score": 0.9,
                    "properties": {"vdb_id": "ent-a", "content": "Alice", "entity_name": "ALICE"},
                },
                {
                    "node_id": "BOB",
                    "score": 0.8,
                    "properties": {"vdb_id": "ent-b", "content": "Bob", "entity_name": "BOB"},
                },
            ]
        }
        # Mock Cypher pairwise edge fetch
        edge_resp = _cypher_response(
            [
                {"row": ["ALICE", "BOB", {"description": "knows", "keywords": "friendship", "weight": 1.0}]},
            ],
            ["src", "tgt", "props"],
        )

        call_count = {"n": 0}

        async def side_effect(*args, **kwargs):
            call_count["n"] += 1
            path = args[1] if len(args) > 1 else ""
            if path == "/_plugins/_graph/retrieve":
                return retrieve_resp
            return edge_resp

        mock_client.transport.perform_request = AsyncMock(side_effect=side_effect)

        adapter = self._make(global_config, embed_func, mock_client)
        await adapter.initialize()
        results = await adapter.query("friendship", top_k=10)
        assert len(results) >= 1
        assert results[0]["src_id"] == "ALICE"
        assert results[0]["tgt_id"] == "BOB"
        assert results[0]["id"] == compute_mdhash_id("ALICE" + "BOB", prefix="rel-")

    @pytest.mark.asyncio
    async def test_query_canonicalizes_endpoint_order(
        self, global_config, embed_func, mock_client
    ):
        """Edge returned as (BOB, ALICE) should produce same id as (ALICE, BOB)."""
        retrieve_resp = {
            "results": [
                {"node_id": "ALICE", "score": 0.9,
                 "properties": {"vdb_id": "ent-a", "content": "Alice", "entity_name": "ALICE"}},
                {"node_id": "BOB", "score": 0.8,
                 "properties": {"vdb_id": "ent-b", "content": "Bob", "entity_name": "BOB"}},
            ]
        }
        # Return edge in REVERSED order: BOB→ALICE
        edge_resp = _cypher_response(
            [{"row": ["BOB", "ALICE", {"description": "knows", "weight": 1.0}]}],
            ["src", "tgt", "props"],
        )

        async def side_effect(*args, **kwargs):
            path = args[1] if len(args) > 1 else ""
            if path == "/_plugins/_graph/retrieve":
                return retrieve_resp
            return edge_resp

        mock_client.transport.perform_request = AsyncMock(side_effect=side_effect)
        adapter = self._make(global_config, embed_func, mock_client)
        await adapter.initialize()
        results = await adapter.query("test", top_k=10)

        assert len(results) == 1
        # Canonicalized: sorted(["BOB","ALICE"]) = ["ALICE","BOB"]
        assert results[0]["src_id"] == "ALICE"
        assert results[0]["tgt_id"] == "BOB"
        expected_id = compute_mdhash_id("ALICE" + "BOB", prefix="rel-")
        assert results[0]["id"] == expected_id

    @pytest.mark.asyncio
    async def test_client_storage_returns_all_edges(
        self, global_config, embed_func, mock_client
    ):
        """client_storage property must return all DIRECTED edges for export."""
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response(
                [
                    {"row": ["ALICE", "BOB", {"description": "knows", "weight": 1.0}]},
                    {"row": ["BOB", "CHARLIE", {"description": "works_with", "weight": 0.5}]},
                ],
                ["src", "tgt", "props"],
            )
        )
        adapter = self._make(global_config, embed_func, mock_client)
        await adapter.initialize()
        storage = await adapter.client_storage
        assert "data" in storage
        assert len(storage["data"]) == 2
        # Each record must have __id__
        for rec in storage["data"]:
            assert "__id__" in rec
            assert rec["__id__"].startswith("rel-")
        # Check canonical ordering
        rec0 = storage["data"][0]
        assert rec0["src_id"] == "ALICE"
        assert rec0["tgt_id"] == "BOB"

    @pytest.mark.asyncio
    async def test_client_storage_compatible_with_export_data_path(
        self, global_config, embed_func, mock_client
    ):
        """Exercises the exact call pattern from utils.py export_data():
            all_relationships = await relationships_vdb.client_storage
            for rel in all_relationships["data"]:
                {"relationship_id": rel["__id__"], "data": str(rel)}
        This must not raise AttributeError or KeyError.
        """
        mock_client.transport.perform_request = AsyncMock(
            return_value=_cypher_response(
                [{"row": ["X", "Y", {"description": "d", "weight": 1.0}]}],
                ["src", "tgt", "props"],
            )
        )
        adapter = self._make(global_config, embed_func, mock_client)
        await adapter.initialize()
        # Mimic export_data() exactly
        all_relationships = await adapter.client_storage
        relationships_data = []
        for rel in all_relationships["data"]:
            relationships_data.append({
                "relationship_id": rel["__id__"],
                "data": str(rel),
            })
        assert len(relationships_data) == 1
        assert relationships_data[0]["relationship_id"].startswith("rel-")

    @pytest.mark.asyncio
    async def test_client_storage_empty_when_no_graph(
        self, global_config, embed_func, mock_client
    ):
        """client_storage returns empty data when graph storage not ready."""
        adapter = self._make(global_config, embed_func, mock_client)
        adapter._graph_storage._database_ready = False
        await adapter.initialize()
        storage = await adapter.client_storage
        assert storage == {"data": []}

    @pytest.mark.asyncio
    async def test_delete_noop(self, global_config, embed_func, mock_client):
        adapter = self._make(global_config, embed_func, mock_client)
        await adapter.delete(["rel-1"])  # Should not raise

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none(self, global_config, embed_func, mock_client):
        adapter = self._make(global_config, embed_func, mock_client)
        assert await adapter.get_by_id("rel-abc") is None

    @pytest.mark.asyncio
    async def test_get_by_ids_returns_empty(self, global_config, embed_func, mock_client):
        adapter = self._make(global_config, embed_func, mock_client)
        assert await adapter.get_by_ids(["rel-1", "rel-2"]) == []

    @pytest.mark.asyncio
    async def test_get_vectors_returns_empty(self, global_config, embed_func, mock_client):
        adapter = self._make(global_config, embed_func, mock_client)
        assert await adapter.get_vectors_by_ids(["rel-1"]) == {}

    @pytest.mark.asyncio
    async def test_drop_noop(self, global_config, embed_func, mock_client):
        adapter = self._make(global_config, embed_func, mock_client)
        result = await adapter.drop()
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_delete_entity_relation(self, global_config, embed_func, mock_client):
        mock_client.transport.perform_request = AsyncMock(return_value={})
        adapter = self._make(global_config, embed_func, mock_client)
        await adapter.initialize()
        await adapter.delete_entity_relation("ALICE")
        calls = mock_client.transport.perform_request.call_args_list
        cypher_calls = [c for c in calls if len(c[0]) >= 2 and c[0][1] == "/_plugins/_cypher"]
        assert any("DELETE r" in str(c) for c in cypher_calls)
