"""
OpenSearch Storage Implementation for LightRAG

This module provides OpenSearch-based storage backends for LightRAG,
including KV storage, document status storage, graph storage, and vector storage.

Requirements:
    - opensearch-py >= 3.0.0
    - OpenSearch 3.x or higher with k-NN plugin enabled
"""

import hashlib
import json as json_module
import os
import re
import ssl as ssl_module
import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, Union, final
import numpy as np
import configparser

from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from ..utils import logger, compute_mdhash_id
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..constants import GRAPH_FIELD_SEP
from ..kg.shared_storage import get_data_init_lock

import pipmaster as pm

if not pm.is_installed("opensearch-py"):
    pm.install("opensearch-py")

from opensearchpy import AsyncOpenSearch, helpers
from opensearchpy.exceptions import OpenSearchException, NotFoundError, RequestError

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# Property schema for promoted fields in the graph plugin database.
# Promoted fields are stored as typed top-level OpenSearch fields instead
# of string key-value pairs in the ``properties`` flat_object.  This enables
# proper BM25 text analysis on ``content``/``description`` fields (improving
# hybrid retrieval quality), numeric sorting on ``weight``, and efficient
# exact-match indexing on keyword fields.
_GRAPH_PROPERTY_SCHEMA = {
    "nodes": {
        "entity_id": {"type": "keyword"},
        "entity_type": {"type": "keyword"},
        "vdb_id": {"type": "keyword"},
        "content": {"type": "text"},
        "description": {"type": "text"},
        "file_path": {"type": "keyword"},
        "full_doc_id": {"type": "keyword"},
        "tokens": {"type": "integer"},
        "chunk_order_index": {"type": "integer"},
    },
    "edges": {
        "weight": {"type": "float"},
        "description": {"type": "text"},
        "keywords": {"type": "text"},
    },
    "strict": False,
}


def _get_opensearch_env(key, fallback):
    cfg_key = key.replace("OPENSEARCH_", "").lower()
    return os.environ.get(key, config.get("opensearch", cfg_key, fallback=fallback))


def _sanitize_index_name(name: str) -> str:
    """Sanitize a string to be a valid OpenSearch index name."""
    sanitized = re.sub(r"[^a-z0-9_-]", "_", name.lower())
    if sanitized and sanitized[0] in "-_+":
        sanitized = "x" + sanitized
    return sanitized


class ClientManager:
    """Singleton manager for OpenSearch client connections."""

    _instances = {"client": None, "ref_count": 0}
    _lock = asyncio.Lock()

    @classmethod
    async def get_client(cls) -> AsyncOpenSearch:
        """Get or create a shared AsyncOpenSearch client with reference counting."""
        async with cls._lock:
            if cls._instances["client"] is None:
                hosts_str = _get_opensearch_env("OPENSEARCH_HOSTS", "localhost:9200")
                hosts = [h.strip() for h in hosts_str.split(",") if h.strip()]
                username = _get_opensearch_env("OPENSEARCH_USER", "admin")
                password = _get_opensearch_env("OPENSEARCH_PASSWORD", "admin")
                use_ssl = _get_opensearch_env("OPENSEARCH_USE_SSL", "true").lower() in (
                    "true",
                    "1",
                    "yes",
                )
                verify_certs = _get_opensearch_env(
                    "OPENSEARCH_VERIFY_CERTS", "false"
                ).lower() in ("true", "1", "yes")
                timeout = int(_get_opensearch_env("OPENSEARCH_TIMEOUT", "30"))
                max_retries = int(_get_opensearch_env("OPENSEARCH_MAX_RETRIES", "3"))

                ssl_context = None
                if use_ssl and not verify_certs:
                    ssl_context = ssl_module.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl_module.CERT_NONE

                client = AsyncOpenSearch(
                    hosts=hosts,
                    http_auth=(username, password) if username else None,
                    use_ssl=use_ssl,
                    verify_certs=verify_certs,
                    ssl_context=ssl_context,
                    ssl_show_warn=False,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_on_timeout=True,
                )
                cls._instances["client"] = client
                cls._instances["ref_count"] = 0
                logger.info(f"OpenSearch client connected to {hosts}")

            cls._instances["ref_count"] += 1
            return cls._instances["client"]

    @classmethod
    async def release_client(cls, client: AsyncOpenSearch):
        """Release a client reference. Closes the connection when ref count reaches 0."""
        async with cls._lock:
            if client is not None and client is cls._instances["client"]:
                cls._instances["ref_count"] -= 1
                if cls._instances["ref_count"] <= 0:
                    try:
                        await cls._instances["client"].close()
                    except Exception:
                        pass
                    cls._instances["client"] = None
                    cls._instances["ref_count"] = 0
                    logger.info("OpenSearch client connection closed")


def _resolve_workspace(workspace: str, namespace: str):
    """Resolve effective workspace from env or parameter."""
    opensearch_workspace = os.environ.get("OPENSEARCH_WORKSPACE")
    if opensearch_workspace and opensearch_workspace.strip():
        effective = opensearch_workspace.strip()
        logger.info(
            f"Using OPENSEARCH_WORKSPACE: '{effective}' (overriding '{workspace}/{namespace}')"
        )
        return effective
    return workspace


def _build_index_name(workspace: str, namespace: str) -> tuple[str, str, str]:
    """Build index name and return (effective_workspace, final_namespace, index_name)."""
    effective = _resolve_workspace(workspace, namespace)
    if effective:
        final_ns = f"{effective}_{namespace}"
    else:
        final_ns = namespace
        effective = ""
    index_name = _sanitize_index_name(final_ns)
    return effective, final_ns, index_name


def _sanitize_database_name(workspace: str, namespace: str) -> str:
    """Build a graph plugin database name from workspace + namespace.

    Steps:
    1. Concatenate ``{workspace}_{namespace}``, lowercase
    2. Replace ``[^a-z0-9_-]`` with ``_``
    3. Strip leading ``_`` / ``-``
    4. If starts with digit prepend ``g``
    5. Truncate to 180 chars
    6. Append ``-`` + first 8 chars of SHA-256 hash of the *original* unsanitized name
    7. Assert total length <= 200 and matches ``^[a-z][a-z0-9_-]*$``
    """
    raw = f"{workspace}_{namespace}"
    name = raw.lower()
    name = re.sub(r"[^a-z0-9_-]", "_", name)
    name = name.lstrip("_-")
    if not name:
        name = "g"
    if name[0].isdigit():
        name = "g" + name
    name = name[:180]
    suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    name = f"{name}-{suffix}"
    assert len(name) <= 200, f"Database name too long: {len(name)}"
    assert re.match(
        r"^[a-z][a-z0-9_-]*$", name
    ), f"Invalid database name: {name}"
    return name


async def _mget_optional_doc(
    client: AsyncOpenSearch, index_name: str, doc_id: str
) -> dict[str, Any] | None:
    """Fetch a single document via mget and return None when it is absent."""
    response = await client.mget(index=index_name, body={"ids": [doc_id]})
    docs = response.get("docs", [])
    if not docs:
        return None
    doc = docs[0]
    if not doc.get("found"):
        return None
    return doc


def _is_missing_index_error(exc: Exception) -> bool:
    """Return True when an OpenSearch exception means the target index is missing."""
    return "index_not_found_exception" in str(exc)


@final
@dataclass
class OpenSearchKVStorage(BaseKVStorage):
    """Key-Value storage using OpenSearch. Uses dynamic mapping to support varied schemas."""

    client: AsyncOpenSearch = field(default=None)
    _index_name: str = field(default="", init=False)
    _index_ready: bool = field(default=False, init=False)

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.__post_init__()

    def __post_init__(self):
        self.workspace, self.final_namespace, self._index_name = _build_index_name(
            self.workspace, self.namespace
        )

    async def initialize(self):
        """Initialize client connection and create index if needed."""
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            await self._create_index_if_not_exists()
            self._index_ready = True
            logger.debug(
                f"[{self.workspace}] OpenSearch KV storage initialized: {self._index_name}"
            )

    async def _ensure_index_ready(self):
        """Recreate the KV index after drop before the next write."""
        if self._index_ready:
            return
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            if not self._index_ready:
                await self._create_index_if_not_exists()
                self._index_ready = True

    def _mark_index_missing(self):
        """Mark the KV index as unavailable for subsequent read short-circuiting."""
        self._index_ready = False

    async def _create_index_if_not_exists(self):
        try:
            if not await self.client.indices.exists(index=self._index_name):
                # Use dynamic mapping so any namespace schema works
                body = {
                    "mappings": {"dynamic": True},
                    "settings": {
                        "index": {"number_of_shards": 1, "number_of_replicas": 0},
                    },
                }
                await self.client.indices.create(index=self._index_name, body=body)
                logger.info(f"[{self.workspace}] Created index: {self._index_name}")
        except RequestError as e:
            if "resource_already_exists_exception" not in str(e):
                raise
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error creating index: {e}")
            raise

    async def finalize(self):
        """Release the OpenSearch client connection."""
        if self.client is not None:
            await ClientManager.release_client(self.client)
            self.client = None

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get a document by its ID, or None if not found."""
        if not self._index_ready:
            return None
        try:
            response = await _mget_optional_doc(self.client, self._index_name, id)
            if response is None:
                return None
            doc = response["_source"]
            doc["_id"] = response["_id"]
            doc.setdefault("create_time", 0)
            doc.setdefault("update_time", 0)
            return doc
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return None
            logger.error(f"[{self.workspace}] Error getting document {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple documents by IDs, preserving input order."""
        if not self._index_ready:
            return [None] * len(ids)
        try:
            response = await self.client.mget(index=self._index_name, body={"ids": ids})
            doc_map = {}
            for doc in response["docs"]:
                if doc.get("found"):
                    data = doc["_source"]
                    data["_id"] = doc["_id"]
                    data.setdefault("create_time", 0)
                    data.setdefault("update_time", 0)
                    doc_map[doc["_id"]] = data
            return [doc_map.get(id) for id in ids]
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return [None] * len(ids)
            logger.error(f"[{self.workspace}] Error getting documents: {e}")
            return [None] * len(ids)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return the subset of keys that do not exist in storage."""
        if not self._index_ready:
            return keys
        try:
            response = await self.client.mget(
                index=self._index_name, body={"ids": list(keys)}, _source=False
            )
            existing_ids = {doc["_id"] for doc in response["docs"] if doc.get("found")}
            return keys - existing_ids
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return keys
            logger.error(f"[{self.workspace}] Error filtering keys: {e}")
            return keys

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update documents with automatic timestamping."""
        if not data:
            return
        await self._ensure_index_ready()
        logger.debug(
            f"[{self.workspace}] Upserting {len(data)} documents to {self.namespace}"
        )
        current_time = int(time.time())
        actions = []
        for doc_id, doc_data in data.items():
            doc_data["update_time"] = current_time
            doc_data.setdefault("create_time", current_time)
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self._index_name,
                    "_id": doc_id,
                    "_source": {k: v for k, v in doc_data.items() if k != "_id"},
                }
            )
        try:
            success, failed = await helpers.async_bulk(
                self.client, actions, raise_on_error=False, refresh="false"
            )
            if failed:
                logger.warning(
                    f"[{self.workspace}] {len(failed)} documents failed to upsert"
                )
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error upserting documents: {e}")
            raise

    async def index_done_callback(self) -> None:
        """Refresh index to make recently indexed documents searchable."""
        if not self._index_ready:
            return
        try:
            await self.client.indices.refresh(index=self._index_name)
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return
        except Exception:
            pass

    async def is_empty(self) -> bool:
        """Return True if the index contains no documents."""
        if not self._index_ready:
            return True
        try:
            response = await self.client.count(index=self._index_name)
            return response["count"] == 0
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
            return True

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by their IDs."""
        if not ids:
            return
        if not self._index_ready:
            return
        if isinstance(ids, set):
            ids = list(ids)
        try:
            actions = [
                {"_op_type": "delete", "_index": self._index_name, "_id": doc_id}
                for doc_id in ids
            ]
            success, _ = await helpers.async_bulk(
                self.client, actions, raise_on_error=False, refresh="wait_for"
            )
            logger.info(
                f"[{self.workspace}] Deleted {success} documents from {self.namespace}"
            )
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return
            logger.error(f"[{self.workspace}] Error deleting documents: {e}")

    async def drop(self) -> dict[str, str]:
        """Delete the entire index."""
        try:
            try:
                await self.client.indices.delete(index=self._index_name)
                logger.info(f"[{self.workspace}] Dropped index: {self._index_name}")
            except NotFoundError:
                logger.info(
                    f"[{self.workspace}] Index already missing during drop: {self._index_name}"
                )
            self._mark_index_missing()
            return {"status": "success", "message": f"Index {self._index_name} dropped"}
        except OpenSearchException as e:
            self._mark_index_missing()
            logger.error(f"[{self.workspace}] Error dropping index: {e}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            self._mark_index_missing()
            logger.error(f"[{self.workspace}] Unexpected error dropping index: {e}")
            return {"status": "error", "message": str(e)}


@final
@dataclass
class OpenSearchDocStatusStorage(DocStatusStorage):
    """Document status storage using OpenSearch."""

    client: AsyncOpenSearch = field(default=None)
    _index_name: str = field(default="", init=False)
    _index_ready: bool = field(default=False, init=False)

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.__post_init__()

    def __post_init__(self):
        self.workspace, self.final_namespace, self._index_name = _build_index_name(
            self.workspace, self.namespace
        )

    def _prepare_doc_status_data(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Normalize a raw OpenSearch document to DocProcessingStatus-compatible dict."""
        data = doc.copy()
        data.pop("_id", None)
        if "file_path" not in data:
            data["file_path"] = "no-file-path"
        data.setdefault("metadata", {})
        data.setdefault("error_msg", None)
        if "error" in data:
            if not data.get("error_msg"):
                data["error_msg"] = data.pop("error")
            else:
                data.pop("error", None)
        return data

    async def initialize(self):
        """Initialize client connection and create doc status index."""
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            await self._create_index_if_not_exists()
            self._index_ready = True
            logger.debug(
                f"[{self.workspace}] OpenSearch DocStatus storage initialized: {self._index_name}"
            )

    async def _ensure_index_ready(self):
        """Recreate the doc status index after drop before the next write."""
        if self._index_ready:
            return
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            if not self._index_ready:
                await self._create_index_if_not_exists()
                self._index_ready = True

    def _mark_index_missing(self):
        """Mark the doc status index as unavailable for subsequent read short-circuiting."""
        self._index_ready = False

    async def _create_index_if_not_exists(self):
        try:
            if not await self.client.indices.exists(index=self._index_name):
                body = {
                    "mappings": {
                        "dynamic": True,
                        "properties": {
                            "status": {"type": "keyword"},
                            "file_path": {"type": "keyword"},
                            "track_id": {"type": "keyword"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                        },
                    },
                    "settings": {
                        "index": {"number_of_shards": 1, "number_of_replicas": 0},
                    },
                }
                await self.client.indices.create(index=self._index_name, body=body)
                logger.info(
                    f"[{self.workspace}] Created doc status index: {self._index_name}"
                )
        except RequestError as e:
            if "resource_already_exists_exception" not in str(e):
                raise
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error creating doc status index: {e}")
            raise

    async def finalize(self):
        """Release the OpenSearch client connection."""
        if self.client is not None:
            await ClientManager.release_client(self.client)
            self.client = None

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        """Get a document status record by ID."""
        if not self._index_ready:
            return None
        try:
            response = await _mget_optional_doc(self.client, self._index_name, id)
            if response is None:
                return None
            doc = response["_source"]
            doc["_id"] = response["_id"]
            return doc
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return None
            logger.error(f"[{self.workspace}] Error getting doc status {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple document status records by IDs."""
        if not self._index_ready:
            return [None] * len(ids)
        try:
            response = await self.client.mget(index=self._index_name, body={"ids": ids})
            doc_map = {}
            for doc in response["docs"]:
                if doc.get("found"):
                    data = doc["_source"]
                    data["_id"] = doc["_id"]
                    doc_map[doc["_id"]] = data
            return [doc_map.get(id) for id in ids]
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return [None] * len(ids)
            logger.error(f"[{self.workspace}] Error getting doc statuses: {e}")
            return [None] * len(ids)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return the subset of keys that do not exist in storage."""
        if not self._index_ready:
            return keys
        try:
            response = await self.client.mget(
                index=self._index_name, body={"ids": list(keys)}, _source=False
            )
            existing_ids = {doc["_id"] for doc in response["docs"] if doc.get("found")}
            return keys - existing_ids
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return keys
            logger.error(f"[{self.workspace}] Error filtering keys: {e}")
            return keys

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update document status records."""
        if not data:
            return
        await self._ensure_index_ready()
        logger.debug(f"[{self.workspace}] Upserting {len(data)} doc statuses")
        actions = []
        for k, v in data.items():
            v.setdefault("chunks_list", [])
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self._index_name,
                    "_id": k,
                    "_source": {fk: fv for fk, fv in v.items() if fk != "_id"},
                }
            )
        try:
            await helpers.async_bulk(
                self.client, actions, raise_on_error=False, refresh="wait_for"
            )
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error upserting doc statuses: {e}")

    async def get_status_counts(self) -> dict[str, int]:
        """Get document counts grouped by status."""
        if not self._index_ready:
            return {}
        try:
            body = {
                "size": 0,
                "aggs": {"status_counts": {"terms": {"field": "status", "size": 100}}},
            }
            response = await self.client.search(index=self._index_name, body=body)
            return {
                bucket["key"]: bucket["doc_count"]
                for bucket in response["aggregations"]["status_counts"]["buckets"]
            }
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return {}
            logger.error(f"[{self.workspace}] Error getting status counts: {e}")
            return {}

    async def _search_all_docs(self, query: dict) -> dict[str, DocProcessingStatus]:
        """Fetch all documents matching a query using PIT + search_after."""
        if not self._index_ready:
            return {}
        result = {}
        batch_size = 10000
        try:
            pit = await self.client.create_pit(
                index=self._index_name, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                while True:
                    body = {
                        "query": query,
                        "size": batch_size,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                        "sort": [{"_shard_doc": "asc"}],
                    }
                    if search_after:
                        body["search_after"] = search_after
                    response = await self.client.search(body=body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    for hit in hits:
                        try:
                            data = self._prepare_doc_status_data(hit["_source"])
                            result[hit["_id"]] = DocProcessingStatus(**data)
                        except (KeyError, TypeError) as e:
                            logger.error(
                                f"[{self.workspace}] Error parsing doc {hit['_id']}: {e}"
                            )
                    search_after = hits[-1]["sort"]
                    if len(hits) < batch_size:
                        break
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return {}
            logger.error(f"[{self.workspace}] Error fetching docs: {e}")
        return result

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents matching a specific processing status."""
        return await self._search_all_docs({"term": {"status": status.value}})

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents matching a specific track ID."""
        return await self._search_all_docs({"term": {"track_id": track_id}})

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination using PIT + search_after."""
        if not self._index_ready:
            return [], 0
        page = max(1, page)
        page_size = max(10, min(200, page_size))
        if sort_field == "id":
            sort_field = "_id"
        if sort_field not in ("created_at", "updated_at", "_id", "file_path"):
            sort_field = "updated_at"
        sort_order = "asc" if sort_direction.lower() == "asc" else "desc"

        query = {"match_all": {}}
        if status_filter is not None:
            query = {"term": {"status": status_filter.value}}

        skip_count = (page - 1) * page_size

        try:
            count_resp = await self.client.count(
                index=self._index_name, body={"query": query}
            )
            total_count = count_resp.get("count", 0)
            if total_count == 0 or skip_count >= total_count:
                return [], total_count

            sort_clause = [{sort_field: {"order": sort_order}}, {"_shard_doc": "asc"}]

            pit = await self.client.create_pit(
                index=self._index_name, params={"keep_alive": "1m"}
            )
            pit_id = pit["pit_id"]
            try:
                search_after = None
                skipped = 0
                while skipped < skip_count:
                    batch = min(page_size, skip_count - skipped)
                    body = {
                        "query": query,
                        "sort": sort_clause,
                        "size": batch,
                        "pit": {"id": pit_id, "keep_alive": "1m"},
                    }
                    if search_after:
                        body["search_after"] = search_after
                    resp = await self.client.search(body=body)
                    hits = resp["hits"]["hits"]
                    if not hits:
                        return [], total_count
                    search_after = hits[-1]["sort"]
                    skipped += len(hits)

                body = {
                    "query": query,
                    "sort": sort_clause,
                    "size": page_size,
                    "pit": {"id": pit_id, "keep_alive": "1m"},
                }
                if search_after:
                    body["search_after"] = search_after
                response = await self.client.search(body=body)
            finally:
                try:
                    await self.client.delete_pit(body={"pit_id": [pit_id]})
                except Exception:
                    pass

            documents = []
            for hit in response["hits"]["hits"]:
                try:
                    data = self._prepare_doc_status_data(hit["_source"])
                    documents.append((hit["_id"], DocProcessingStatus(**data)))
                except (KeyError, TypeError) as e:
                    logger.error(
                        f"[{self.workspace}] Error parsing doc {hit['_id']}: {e}"
                    )
            return documents, total_count
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return [], 0
            logger.error(f"[{self.workspace}] Error in paginated query: {e}")
            return [], 0

    async def get_all_status_counts(self) -> dict[str, int]:
        """Get document counts for all statuses including an 'all' total."""
        if not self._index_ready:
            return {}
        try:
            body = {
                "size": 0,
                "aggs": {"status_counts": {"terms": {"field": "status", "size": 100}}},
            }
            response = await self.client.search(index=self._index_name, body=body)
            counts = {}
            total = 0
            for bucket in response["aggregations"]["status_counts"]["buckets"]:
                counts[bucket["key"]] = bucket["doc_count"]
                total += bucket["doc_count"]
            counts["all"] = total
            return counts
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return {}
            logger.error(f"[{self.workspace}] Error getting all status counts: {e}")
            return {}

    async def get_doc_by_file_path(self, file_path: str) -> Union[dict[str, Any], None]:
        """Find a document status record by its file_path field."""
        if not self._index_ready:
            return None
        try:
            body = {"query": {"term": {"file_path": file_path}}, "size": 1}
            response = await self.client.search(index=self._index_name, body=body)
            hits = response["hits"]["hits"]
            if hits:
                doc = hits[0]["_source"]
                doc["_id"] = hits[0]["_id"]
                return doc
            return None
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return None
            logger.error(f"[{self.workspace}] Error getting doc by file_path: {e}")
            return None

    async def index_done_callback(self) -> None:
        """Refresh index to make recently indexed documents searchable."""
        if not self._index_ready:
            return
        try:
            await self.client.indices.refresh(index=self._index_name)
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return
        except Exception:
            pass

    async def is_empty(self) -> bool:
        """Return True if the index contains no documents."""
        if not self._index_ready:
            return True
        try:
            response = await self.client.count(index=self._index_name)
            return response["count"] == 0
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
            return True

    async def delete(self, ids: list[str]) -> None:
        """Delete document status records by IDs."""
        if not ids:
            return
        if not self._index_ready:
            return
        if isinstance(ids, set):
            ids = list(ids)
        try:
            actions = [
                {"_op_type": "delete", "_index": self._index_name, "_id": doc_id}
                for doc_id in ids
            ]
            await helpers.async_bulk(
                self.client, actions, raise_on_error=False, refresh="wait_for"
            )
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return
            logger.error(f"[{self.workspace}] Error deleting doc statuses: {e}")

    async def drop(self) -> dict[str, str]:
        """Delete the entire doc status index."""
        try:
            try:
                await self.client.indices.delete(index=self._index_name)
                logger.info(
                    f"[{self.workspace}] Dropped doc status index: {self._index_name}"
                )
            except NotFoundError:
                logger.info(
                    f"[{self.workspace}] Doc status index already missing during drop: {self._index_name}"
                )
            self._mark_index_missing()
            return {"status": "success", "message": f"Index {self._index_name} dropped"}
        except OpenSearchException as e:
            self._mark_index_missing()
            logger.error(f"[{self.workspace}] Error dropping doc status index: {e}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            self._mark_index_missing()
            logger.error(
                f"[{self.workspace}] Unexpected error dropping doc status index: {e}"
            )
            return {"status": "error", "message": str(e)}


@final
@dataclass
class OpenSearchGraphStorage(BaseGraphStorage):
    """Graph storage using the OpenSearch graph plugin with Cypher queries.

    Uses a dedicated graph plugin database per workspace. All node/edge
    operations are performed via the ``POST _plugins/_cypher`` endpoint.
    Entity nodes use ``entity_id`` as the graph key, matching the Neo4J
    pattern. The database auto-manages two underlying indices:
    ``{database}-lpg-nodes`` and ``{database}-lpg-edges``.

    Traversal strategies:
    - APOC ``apoc.path.subgraphNodes`` (if available)
    - Variable-length Cypher paths (default fallback)
    """

    _client: AsyncOpenSearch = field(default=None, init=False)
    _database_name: str = field(default="", init=False)
    _database_ready: bool = field(default=False, init=False)
    _apoc_available: bool = field(default=False, init=False)

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.__post_init__()

    def __post_init__(self):
        effective = _resolve_workspace(self.workspace, self.namespace)
        self.workspace = effective
        self._database_name = _sanitize_database_name(effective, self.namespace)

    @property
    def database_name(self) -> str:
        """Public property: the graph plugin database name."""
        return self._database_name

    @property
    def client(self) -> AsyncOpenSearch:
        """Public property: the shared AsyncOpenSearch client."""
        if self._client is None:
            raise RuntimeError(
                "OpenSearchGraphStorage has not been initialized. "
                "Call initialize() first."
            )
        return self._client

    # --- Cypher execution ---

    async def _execute_cypher(
        self, query: str, params: dict | None = None, retries: int = 3
    ) -> dict:
        """Execute a Cypher query against the graph plugin endpoint.

        Args:
            query: Cypher query string
            params: Optional Cypher parameters
            retries: Number of retry attempts on transient failures

        Returns:
            Raw response dict from the ``_plugins/_cypher`` endpoint.
            The graph plugin returns either:
            - Write queries: ``{"stats": {"nodesCreated": N, ...}}``
            - Read queries: ``{"columns": ["a", "b"], "data": [{"a": 1, "b": 2}, ...]}``
        """
        import asyncio

        body: dict[str, Any] = {
            "query": query,
            "database": self._database_name,
        }
        if params:
            body["parameters"] = params
        last_exc = None
        for attempt in range(retries + 1):
            try:
                return await self._client.transport.perform_request(
                    "POST", "/_plugins/_cypher", body=body
                )
            except Exception as e:
                last_exc = e
                if attempt < retries:
                    wait = 2 ** attempt
                    logger.warning(
                        f"[{self.workspace}] Cypher query failed (attempt {attempt + 1}/{retries + 1}), "
                        f"retrying in {wait}s: {e}"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"[{self.workspace}] Cypher query failed after {retries + 1} attempts: {last_exc}\nQuery: {query}"
                    )
                    raise

    @staticmethod
    def _cypher_rows(resp: dict) -> list[list]:
        """Extract result rows from a Cypher response as positional lists.

        The graph plugin returns ``{"columns": ["a","b"], "data": [{"a":1,"b":2}]}``
        which this helper normalises to ``[[1, 2], ...]`` so callers can use
        positional indexing (``row[0]``, ``row[1]``).
        """
        columns = resp.get("columns", [])
        data = resp.get("data", [])
        if not columns or not data:
            return []
        return [[item.get(col) for col in columns] for item in data]

    # --- Database lifecycle ---

    async def _create_database_if_not_exists(self):
        """Create the graph plugin database with embedding config."""
        dim = self.embedding_func.embedding_dim
        db_body = {
            "embedding": {
                "dimension": dim,
                "field": "embedding",
                "engine": "faiss",
                "space_type": "cosinesimil",
            },
            "schema": _GRAPH_PROPERTY_SCHEMA,
        }
        try:
            await self._client.transport.perform_request(
                "PUT",
                f"/_plugins/_graph/database/{self._database_name}",
                body=db_body,
            )
            logger.info(
                f"[{self.workspace}] Created graph database: {self._database_name} (dim={dim})"
            )
        except Exception as e:
            # If database already exists, that's fine.  The graph plugin
            # returns HTTP 409 with body "Already exists" (or similar).
            err_lower = str(e).lower()
            if (
                "already exists" in err_lower
                or "already_exists" in err_lower
                or "resource_already_exists" in err_lower
                or isinstance(e, ConflictError)
            ):
                logger.debug(
                    f"[{self.workspace}] Graph database already exists: {self._database_name}"
                )
            else:
                raise

    async def _detect_apoc(self):
        """Check if APOC procedures are available.

        Uses a MATCH-then-CALL pattern with a non-existent node so the CALL
        processes zero rows, but the Cypher parser still validates the
        procedure signature. The start node argument must be a string node
        ID (not a map) — matching the graph plugin's expected shape.
        """
        try:
            await self._execute_cypher(
                "OPTIONAL MATCH (start:Entity {entity_id: '__apoc_probe__'}) "
                "WITH start WHERE start IS NOT NULL "
                "CALL apoc.path.subgraphNodes(start, "
                "{maxLevel: 0, relationshipFilter: 'DIRECTED'}) YIELD node "
                "RETURN node LIMIT 0"
            )
            self._apoc_available = True
            logger.info(f"[{self.workspace}] APOC procedures available")
        except Exception:
            self._apoc_available = False
            logger.info(
                f"[{self.workspace}] APOC not available, using variable-length Cypher paths"
            )

    async def initialize(self):
        """Initialize client, create graph database, detect APOC."""
        async with get_data_init_lock():
            if self._client is None:
                self._client = await ClientManager.get_client()
            await self._create_database_if_not_exists()
            self._database_ready = True
            await self._detect_apoc()
            logger.debug(
                f"[{self.workspace}] OpenSearch Graph storage initialized: "
                f"database={self._database_name} (APOC: {self._apoc_available})"
            )

    async def _ensure_database_ready(self):
        """Recreate graph database after drop before the next write."""
        if self._database_ready:
            return
        async with get_data_init_lock():
            if self._client is None:
                self._client = await ClientManager.get_client()
            if not self._database_ready:
                await self._create_database_if_not_exists()
                self._database_ready = True

    def _mark_database_missing(self):
        """Mark the graph database as unavailable."""
        self._database_ready = False

    async def finalize(self):
        """Release the OpenSearch client connection."""
        if self._client is not None:
            await ClientManager.release_client(self._client)
            self._client = None

    # --- Basic queries ---

    async def has_node(self, node_id: str) -> bool:
        """Check whether an Entity node exists in the graph."""
        if not self._database_ready:
            return False
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity {entity_id: $id}) RETURN count(n) > 0 AS exists",
                {"id": node_id},
            )
            rows = self._cypher_rows(resp)
            if rows:
                return bool(rows[0][0])
            return False
        except Exception:
            return False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check whether a DIRECTED edge exists between two Entity nodes."""
        if not self._database_ready:
            return False
        try:
            resp = await self._execute_cypher(
                "MATCH (a:Entity {entity_id: $src})-[r:DIRECTED]->(b:Entity {entity_id: $tgt}) "
                "RETURN count(r) > 0 AS exists",
                {"src": source_node_id, "tgt": target_node_id},
            )
            rows = self._cypher_rows(resp)
            if rows:
                return bool(rows[0][0])
            return False
        except Exception:
            return False

    async def node_degree(self, node_id: str) -> int:
        """Count the number of DIRECTED edges connected to an Entity node."""
        if not self._database_ready:
            return 0
        try:
            # Use direct MATCH instead of OPTIONAL MATCH — the graph plugin's
            # OPTIONAL MATCH scans ALL edges of the type (O(E)), while direct
            # MATCH uses the adjacency list and is O(degree).  Returns 0 rows
            # (not degree=0) when the node has no edges; we handle that below.
            resp = await self._execute_cypher(
                "MATCH (n:Entity {entity_id: $id})-[r:DIRECTED]-() "
                "RETURN count(r) AS degree",
                {"id": node_id},
            )
            rows = self._cypher_rows(resp)
            if rows:
                return int(rows[0][0])
            return 0
        except Exception:
            return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Sum of degrees of both endpoint nodes."""
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return src_degree + tgt_degree

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get an Entity node by its entity_id."""
        if not self._database_ready:
            return None
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity {entity_id: $id}) RETURN properties(n) AS props",
                {"id": node_id},
            )
            rows = self._cypher_rows(resp)
            if rows:
                props = rows[0][0]
                if props:
                    props.pop("embedding", None)
                    return props
            return None
        except Exception:
            return None

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get a DIRECTED edge between two Entity nodes."""
        if not self._database_ready:
            return None
        try:
            resp = await self._execute_cypher(
                "MATCH (a:Entity {entity_id: $src})-[r:DIRECTED]->(b:Entity {entity_id: $tgt}) "
                "RETURN properties(r) AS props LIMIT 1",
                {"src": source_node_id, "tgt": target_node_id},
            )
            rows = self._cypher_rows(resp)
            if rows:
                return rows[0][0]
            return None
        except Exception:
            return None

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all (source, target) edge tuples for DIRECTED edges connected to a node."""
        if not self._database_ready:
            return None
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity {entity_id: $id})-[r:DIRECTED]-(c:Entity) "
                "RETURN n.entity_id AS src, c.entity_id AS tgt",
                {"id": source_node_id},
            )
            rows = self._cypher_rows(resp)
            return [(r[0], r[1]) for r in rows]
        except Exception:
            return None

    # --- Batch operations ---

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Batch-fetch multiple Entity nodes by ID."""
        if not self._database_ready or not node_ids:
            return {}
        try:
            resp = await self._execute_cypher(
                "UNWIND $ids AS id "
                "MATCH (n:Entity {entity_id: id}) "
                "RETURN n.entity_id AS eid, properties(n) AS props",
                {"ids": node_ids},
            )
            result = {}
            for row in self._cypher_rows(resp):
                eid = row[0]
                props = row[1]
                if props:
                    props.pop("embedding", None)
                    result[eid] = props
            return result
        except Exception:
            return {}

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Batch-fetch DIRECTED edge counts for multiple Entity nodes.

        Uses MATCH ... WHERE IN instead of UNWIND to avoid the graph plugin's
        cross-join row limit (UNWIND size × total_edges < 1M).
        Uses direct MATCH instead of OPTIONAL MATCH to avoid O(E) edge scanning.
        """
        if not node_ids or not self._database_ready:
            return {}
        result = {}
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity)-[r:DIRECTED]-() "
                "WHERE n.entity_id IN $ids "
                "RETURN n.entity_id AS eid, count(r) AS degree",
                {"ids": node_ids},
            )
            for row in self._cypher_rows(resp):
                result[row[0]] = int(row[1])
        except Exception:
            # Fall back to individual queries
            for nid in node_ids:
                try:
                    resp = await self._execute_cypher(
                        "MATCH (n:Entity {entity_id: $id})-[r:DIRECTED]-() "
                        "RETURN count(r) AS degree",
                        {"id": nid},
                        retries=0,
                    )
                    rows = self._cypher_rows(resp)
                    result[nid] = int(rows[0][0]) if rows else 0
                except Exception:
                    result[nid] = 0
        # Nodes with 0 DIRECTED edges won't appear in results — fill them in.
        for nid in node_ids:
            if nid not in result:
                result[nid] = 0
        return result

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Batch-fetch DIRECTED edge tuples for multiple Entity nodes."""
        result = {nid: [] for nid in node_ids}
        if not self._database_ready or not node_ids:
            return result
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity)-[r:DIRECTED]-(c:Entity) "
                "WHERE n.entity_id IN $ids "
                "RETURN n.entity_id AS src, c.entity_id AS tgt",
                {"ids": node_ids},
            )
            for row in self._cypher_rows(resp):
                src, tgt = row[0], row[1]
                if src in result:
                    result[src].append((src, tgt))
                if tgt in result and tgt != src:
                    result[tgt].append((src, tgt))
        except Exception:
            for nid in node_ids:
                try:
                    resp = await self._execute_cypher(
                        "MATCH (n:Entity {entity_id: $id})-[r:DIRECTED]-(c:Entity) "
                        "RETURN n.entity_id AS src, c.entity_id AS tgt",
                        {"id": nid},
                        retries=0,
                    )
                    for row in self._cypher_rows(resp):
                        src, tgt = row[0], row[1]
                        if src in result:
                            result[src].append((src, tgt))
                        if tgt in result and tgt != src:
                            result[tgt].append((src, tgt))
                except Exception:
                    pass
        return result

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Batch-fetch edge properties for multiple (src, tgt) pairs.

        Uses a single WHERE IN query to fetch all edges between the entity set,
        then filters client-side.  This replaces N individual get_edge() calls.
        """
        if not pairs or not self._database_ready:
            return {}
        # Collect all unique entity IDs from the pairs.
        all_ids = list({p["src"] for p in pairs} | {p["tgt"] for p in pairs})
        wanted = {(p["src"], p["tgt"]) for p in pairs}

        result = {}
        try:
            resp = await self._execute_cypher(
                "MATCH (a:Entity)-[r:DIRECTED]-(b:Entity) "
                "WHERE a.entity_id IN $ids AND b.entity_id IN $ids "
                "RETURN a.entity_id, b.entity_id, properties(r)",
                {"ids": all_ids},
            )
            for row in self._cypher_rows(resp):
                src, tgt, props = row[0], row[1], row[2]
                key = (src, tgt)
                rev = (tgt, src)
                if key in wanted and key not in result:
                    result[key] = props if props else {}
                elif rev in wanted and rev not in result:
                    result[rev] = props if props else {}
        except Exception:
            # Fall back to individual queries
            for p in pairs:
                edge = await self.get_edge(p["src"], p["tgt"])
                if edge is not None:
                    result[(p["src"], p["tgt"])] = edge
        return result

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """Batch edge degree computation.

        Collects unique node IDs from all pairs, calls node_degrees_batch once,
        then sums src + tgt degrees for each pair.
        """
        if not edge_pairs:
            return {}
        all_ids = list({nid for pair in edge_pairs for nid in pair})
        degrees = await self.node_degrees_batch(all_ids)
        return {
            (src, tgt): degrees.get(src, 0) + degrees.get(tgt, 0)
            for src, tgt in edge_pairs
        }

    # --- Upsert operations ---

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert or update an Entity node via Cypher MERGE."""
        await self._ensure_database_ready()
        # Strip leading hyphens — the graph plugin's index scanner fails
        # when it encounters property values with leading hyphens (Bug 3).
        node_id = node_id.lstrip("-").strip()
        if not node_id:
            return
        props = {k: v for k, v in node_data.items() if k not in ("_id", "embedding")}
        props["entity_id"] = node_id
        if node_data.get("source_id", ""):
            props["source_ids"] = node_data["source_id"].split(GRAPH_FIELD_SEP)

        # Build dynamic label for entity_type.  The label clause is appended
        # with a comma inside ON CREATE/ON MATCH SET (a bare ``SET n:`label```
        # between ON clauses would be a Cypher parse error).
        entity_type = node_data.get("entity_type", "")
        label_clause = ""
        if entity_type:
            safe_type = re.sub(r"[^a-zA-Z0-9_]", "_", entity_type)
            label_clause = f", n:`{safe_type}`"

        # NOTE: The OpenSearch graph plugin ignores plain ``SET`` after
        # ``MERGE`` when the node already exists.  Use ``ON CREATE SET`` /
        # ``ON MATCH SET`` to ensure properties are written in both cases.
        await self._execute_cypher(
            f"MERGE (n:Entity {{entity_id: $id}}) "
            f"ON CREATE SET n += $props{label_clause} "
            f"ON MATCH SET n += $props{label_clause}",
            {"id": node_id, "props": props},
        )

        # Refresh the node index so subsequent MERGE calls (e.g. from the
        # vector storage upsert that follows) can find this node.  Without
        # this, concurrent async workers race on MERGE and create duplicates.
        try:
            node_index = f"{self._database_name}-lpg-nodes"
            await self._client.indices.refresh(index=node_index)
        except Exception:
            pass

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert or update a DIRECTED edge.

        Avoids Cypher ``MERGE`` on edges because the graph plugin scans ALL
        edges of the given type to check for duplicates, making it O(E).
        Instead we check existence first, then CREATE or SET.
        """
        await self._ensure_database_ready()
        # Strip leading hyphens (Bug 3 workaround).
        source_node_id = source_node_id.lstrip("-").strip()
        target_node_id = target_node_id.lstrip("-").strip()
        if not source_node_id or not target_node_id:
            return
        props = {k: v for k, v in edge_data.items() if k not in ("_id",)}
        if edge_data.get("source_id", ""):
            props["source_ids"] = edge_data["source_id"].split(GRAPH_FIELD_SEP)

        # Ensure weight is a float for the promoted field schema.
        if "weight" in props:
            try:
                props["weight"] = float(props["weight"])
            except (ValueError, TypeError):
                props["weight"] = 1.0
        else:
            props["weight"] = 1.0

        params = {"src": source_node_id, "tgt": target_node_id, "props": props}

        # Check if edge already exists (~0.2s vs 17s for MERGE).
        resp = await self._execute_cypher(
            "MATCH (s:Entity {entity_id: $src})-[r:DIRECTED]->(t:Entity {entity_id: $tgt}) "
            "RETURN count(r) AS cnt",
            params,
        )
        exists = resp.get("data", [{}])[0].get("cnt", 0) > 0

        if exists:
            await self._execute_cypher(
                "MATCH (s:Entity {entity_id: $src})-[r:DIRECTED]->(t:Entity {entity_id: $tgt}) "
                "SET r += $props",
                params,
            )
        else:
            # Graph plugin bug: CREATE ... SET r += $props in the same
            # statement silently drops the SET.  Split into two calls.
            await self._execute_cypher(
                "MATCH (s:Entity {entity_id: $src}), (t:Entity {entity_id: $tgt}) "
                "CREATE (s)-[r:DIRECTED]->(t)",
                params,
            )
            if props:
                await self._execute_cypher(
                    "MATCH (s:Entity {entity_id: $src})-[r:DIRECTED]->(t:Entity {entity_id: $tgt}) "
                    "SET r += $props",
                    params,
                )

    # --- Delete operations ---

    async def delete_node(self, node_id: str) -> None:
        """Delete an Entity node and all its connected edges (DETACH DELETE)."""
        try:
            await self._execute_cypher(
                "MATCH (n:Entity {entity_id: $id}) DETACH DELETE n",
                {"id": node_id},
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting node {node_id}: {e}")

    async def remove_nodes(self, nodes: list[str]) -> None:
        """Batch-delete multiple Entity nodes and their connected edges."""
        if not nodes:
            return
        logger.info(f"[{self.workspace}] Deleting {len(nodes)} nodes")
        try:
            await self._execute_cypher(
                "UNWIND $ids AS id "
                "MATCH (n:Entity {entity_id: id}) "
                "DETACH DELETE n",
                {"ids": nodes},
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error removing nodes: {e}")

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """Batch-delete multiple DIRECTED edges."""
        if not edges:
            return
        logger.info(f"[{self.workspace}] Deleting {len(edges)} edges")
        try:
            pairs = [{"src": s, "tgt": t} for s, t in edges]
            await self._execute_cypher(
                "UNWIND $pairs AS p "
                "MATCH (a:Entity {entity_id: p.src})-[r:DIRECTED]->(b:Entity {entity_id: p.tgt}) "
                "DELETE r",
                {"pairs": pairs},
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error removing edges: {e}")

    # --- Delete document cascade ---

    async def delete_document(self, doc_id: str) -> None:
        """Delete a Document node and cascade: PART_OF edges, orphaned Chunks, MENTIONED_IN edges."""
        try:
            # Find connected chunk IDs
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:PART_OF]->(d:Document {id: $doc_id}) "
                "RETURN c.id AS chunk_id",
                {"doc_id": doc_id},
            )
            chunk_ids = [
                r[0]
                for r in self._cypher_rows(resp)
                if r[0]
            ]

            if chunk_ids:
                # Delete MENTIONED_IN edges from entities to these chunks
                await self._execute_cypher(
                    "UNWIND $cids AS cid "
                    "MATCH (e:Entity)-[r:MENTIONED_IN]->(c:Chunk {id: cid}) "
                    "DELETE r",
                    {"cids": chunk_ids},
                )
                # Delete PART_OF edges and Chunk nodes
                await self._execute_cypher(
                    "UNWIND $cids AS cid "
                    "MATCH (c:Chunk {id: cid}) "
                    "DETACH DELETE c",
                    {"cids": chunk_ids},
                )

            # Delete Document node
            await self._execute_cypher(
                "MATCH (d:Document {id: $doc_id}) DETACH DELETE d",
                {"doc_id": doc_id},
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting document {doc_id}: {e}")

    # --- Query operations ---

    async def get_all_labels(self) -> list[str]:
        """Get all Entity node entity_ids sorted alphabetically."""
        if not self._database_ready:
            return []
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity) RETURN n.entity_id AS eid ORDER BY eid"
            )
            return [
                r[0]
                for r in self._cypher_rows(resp)
                if r[0]
            ]
        except Exception:
            return []

    def _construct_graph_node(self, node_id, node_data: dict) -> KnowledgeGraphNode:
        return KnowledgeGraphNode(
            id=node_id,
            labels=[node_id],
            properties={
                k: v
                for k, v in node_data.items()
                if k
                not in (
                    "_id",
                    "entity_id",
                    "source_ids",
                    "connected_edges",
                    "edge_count",
                    "embedding",
                )
            },
        )

    def _construct_graph_edge(self, edge_id: str, edge: dict) -> KnowledgeGraphEdge:
        return KnowledgeGraphEdge(
            id=edge_id,
            type=edge.get("relationship", ""),
            source=edge.get("source_node_id", edge.get("src", "")),
            target=edge.get("target_node_id", edge.get("tgt", "")),
            properties={
                k: v
                for k, v in edge.items()
                if k
                not in (
                    "_id",
                    "source_node_id",
                    "target_node_id",
                    "src",
                    "tgt",
                    "relationship",
                    "source_ids",
                )
            },
        )

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """Retrieve a subgraph via APOC (if available) or variable-length Cypher paths."""
        if not self._database_ready:
            return KnowledgeGraph()
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        result = KnowledgeGraph()
        start = time.perf_counter()

        try:
            if node_label == "*":
                result = await self._get_knowledge_graph_all(max_nodes)
            elif self._apoc_available:
                try:
                    result = await self._bfs_subgraph_apoc(
                        node_label, max_depth, max_nodes
                    )
                except Exception as e:
                    logger.warning(
                        f"[{self.workspace}] APOC traversal failed, falling back to VLP: {e}"
                    )
                    self._apoc_available = False
                    result = await self._bfs_subgraph_vlp(
                        node_label, max_depth, max_nodes
                    )
            else:
                result = await self._bfs_subgraph_vlp(
                    node_label, max_depth, max_nodes
                )

            duration = time.perf_counter() - start
            logger.info(
                f"[{self.workspace}] Subgraph query in {duration:.4f}s | "
                f"Nodes: {len(result.nodes)} | Edges: {len(result.edges)} | "
                f"Truncated: {result.is_truncated}"
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] Graph query failed: {e}")

        return result

    async def _get_knowledge_graph_all(self, max_nodes: int) -> KnowledgeGraph:
        """Get all Entity nodes (up to max_nodes, ranked by degree) and their edges."""
        result = KnowledgeGraph()
        if not self._database_ready:
            return result
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity) "
                "OPTIONAL MATCH (n)-[r:DIRECTED]-() "
                "WITH n, count(r) AS deg "
                "ORDER BY deg DESC LIMIT $max "
                "RETURN n.entity_id AS eid, properties(n) AS props",
                {"max": max_nodes},
            )
            rows = self._cypher_rows(resp)
            node_ids = []
            for row in rows:
                eid = row[0]
                props = row[1] or {}
                props.pop("embedding", None)
                node_ids.append(eid)
                result.nodes.append(self._construct_graph_node(eid, props))

            # Check if truncated
            count_resp = await self._execute_cypher(
                "MATCH (n:Entity) RETURN count(n) AS cnt"
            )
            count_rows = self._cypher_rows(count_resp)
            total = count_rows[0][0] if count_rows else 0
            result.is_truncated = total > max_nodes

            # Fetch edges between found nodes
            if node_ids:
                edge_resp = await self._execute_cypher(
                    "MATCH (a:Entity)-[r:DIRECTED]->(b:Entity) "
                    "WHERE a.entity_id IN $ids AND b.entity_id IN $ids "
                    "RETURN a.entity_id AS src, b.entity_id AS tgt, properties(r) AS props",
                    {"ids": node_ids},
                )
                for row in self._cypher_rows(edge_resp):
                    src, tgt, props = row[0], row[1], row[2] or {}
                    eid = f"{src}-{tgt}"
                    edge_data = {**props, "source_node_id": src, "target_node_id": tgt}
                    result.edges.append(self._construct_graph_edge(eid, edge_data))

        except Exception as e:
            logger.error(f"[{self.workspace}] Error in get_knowledge_graph_all: {e}")
        return result

    async def _bfs_subgraph_apoc(
        self, start_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """Subgraph traversal using APOC apoc.path.subgraphNodes."""
        result = KnowledgeGraph()
        resp = await self._execute_cypher(
            "MATCH (start:Entity {entity_id: $id}) "
            "CALL apoc.path.subgraphNodes(start, "
            "{maxLevel: $depth, relationshipFilter: 'DIRECTED'}) YIELD node "
            "WITH node LIMIT $max "
            "RETURN node.entity_id AS eid, properties(node) AS props",
            {"id": start_label, "depth": max_depth, "max": max_nodes},
        )
        rows = self._cypher_rows(resp)
        node_ids = []
        for row in rows:
            eid = row[0]
            props = row[1] or {}
            props.pop("embedding", None)
            if eid:
                node_ids.append(eid)
                result.nodes.append(self._construct_graph_node(eid, props))

        result.is_truncated = len(node_ids) >= max_nodes

        # Fetch edges between found nodes
        if node_ids:
            edge_resp = await self._execute_cypher(
                "MATCH (a:Entity)-[r:DIRECTED]->(b:Entity) "
                "WHERE a.entity_id IN $ids AND b.entity_id IN $ids "
                "RETURN a.entity_id AS src, b.entity_id AS tgt, properties(r) AS props",
                {"ids": node_ids},
            )
            for row in self._cypher_rows(edge_resp):
                src, tgt, props = row[0], row[1], row[2] or {}
                eid = f"{src}-{tgt}"
                edge_data = {**props, "source_node_id": src, "target_node_id": tgt}
                result.edges.append(self._construct_graph_edge(eid, edge_data))

        return result

    async def _bfs_subgraph_vlp(
        self, start_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """Subgraph traversal using variable-length Cypher paths."""
        result = KnowledgeGraph()

        resp = await self._execute_cypher(
            "MATCH (start:Entity {entity_id: $id}) "
            f"MATCH path = (start)-[:DIRECTED*1..{max_depth}]-(c:Entity) "
            "UNWIND nodes(path) AS n "
            "WITH DISTINCT n LIMIT $max "
            "RETURN n.entity_id AS eid, properties(n) AS props",
            {"id": start_label, "max": max_nodes},
        )
        rows = self._cypher_rows(resp)
        node_ids = []
        for row in rows:
            eid = row[0]
            props = row[1] or {}
            props.pop("embedding", None)
            if eid:
                node_ids.append(eid)
                result.nodes.append(self._construct_graph_node(eid, props))

        # Add start node if not already included
        if start_label not in node_ids:
            start_node = await self.get_node(start_label)
            if start_node:
                node_ids.insert(0, start_label)
                result.nodes.insert(
                    0, self._construct_graph_node(start_label, start_node)
                )

        result.is_truncated = len(node_ids) >= max_nodes

        # Fetch edges between found nodes
        if node_ids:
            edge_resp = await self._execute_cypher(
                "MATCH (a:Entity)-[r:DIRECTED]->(b:Entity) "
                "WHERE a.entity_id IN $ids AND b.entity_id IN $ids "
                "RETURN a.entity_id AS src, b.entity_id AS tgt, properties(r) AS props",
                {"ids": node_ids},
            )
            for row in self._cypher_rows(edge_resp):
                src, tgt, props = row[0], row[1], row[2] or {}
                eid = f"{src}-{tgt}"
                edge_data = {**props, "source_node_id": src, "target_node_id": tgt}
                result.edges.append(self._construct_graph_edge(eid, edge_data))

        return result

    async def get_all_nodes(self) -> list[dict]:
        """Get all Entity nodes with their properties."""
        if not self._database_ready:
            return []
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity) RETURN n.entity_id AS eid, properties(n) AS props"
            )
            nodes = []
            for row in self._cypher_rows(resp):
                props = row[1] or {}
                props.pop("embedding", None)
                props["id"] = row[0]
                nodes.append(props)
            return nodes
        except Exception:
            return []

    async def get_all_edges(self) -> list[dict]:
        """Get all DIRECTED edges."""
        if not self._database_ready:
            return []
        try:
            resp = await self._execute_cypher(
                "MATCH (a:Entity)-[r:DIRECTED]->(b:Entity) "
                "RETURN a.entity_id AS src, b.entity_id AS tgt, properties(r) AS props"
            )
            edges = []
            for row in self._cypher_rows(resp):
                props = row[2] or {}
                props["source"] = row[0]
                props["target"] = row[1]
                edges.append(props)
            return edges
        except Exception:
            return []

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get Entity node IDs ranked by DIRECTED edge degree."""
        if not self._database_ready:
            return []
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity)-[r:DIRECTED]-() "
                "WITH n.entity_id AS eid, count(r) AS deg "
                "ORDER BY deg DESC LIMIT $limit "
                "RETURN eid",
                {"limit": limit},
            )
            return [
                r[0]
                for r in self._cypher_rows(resp)
                if r[0]
            ]
        except Exception:
            return []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search Entity node labels with case-insensitive substring matching."""
        query = query.strip()
        if not query or not self._database_ready:
            return []
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity) "
                "WHERE toLower(n.entity_id) CONTAINS toLower($q) "
                "RETURN n.entity_id AS eid LIMIT $limit",
                {"q": query, "limit": limit},
            )
            return [
                r[0]
                for r in self._cypher_rows(resp)
                if r[0]
            ]
        except Exception:
            return []

    async def index_done_callback(self) -> None:
        """Refresh the graph database node index to make changes searchable."""
        if not self._database_ready:
            return
        try:
            node_index = f"{self._database_name}-lpg-nodes"
            await self._client.indices.refresh(index=node_index)
        except Exception:
            pass
        try:
            edge_index = f"{self._database_name}-lpg-edges"
            await self._client.indices.refresh(index=edge_index)
        except Exception:
            pass

    async def drop(self) -> dict[str, str]:
        """Delete the entire graph database."""
        try:
            await self._client.transport.perform_request(
                "DELETE",
                f"/_plugins/_graph/database/{self._database_name}",
            )
            logger.info(
                f"[{self.workspace}] Dropped graph database: {self._database_name}"
            )
            self._mark_database_missing()
            return {
                "status": "success",
                "message": f"Graph database {self._database_name} dropped",
            }
        except Exception as e:
            self._mark_database_missing()
            logger.error(
                f"[{self.workspace}] Error dropping graph database: {e}"
            )
            return {"status": "error", "message": str(e)}


@final
@dataclass
class OpenSearchVectorDBStorage(BaseVectorStorage):
    """Vector storage using OpenSearch k-NN plugin with corrected cosine score handling."""

    client: AsyncOpenSearch = field(default=None)
    _index_name: str = field(default="", init=False)
    _index_ready: bool = field(default=False, init=False)

    def __init__(
        self, namespace, global_config, embedding_func, workspace=None, meta_fields=None
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
            meta_fields=meta_fields or set(),
        )
        self.__post_init__()

    def __post_init__(self):
        self._validate_embedding_func()
        self.workspace, self.final_namespace, self._index_name = _build_index_name(
            self.workspace, self.namespace
        )
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold
        self._max_batch_size = self.global_config["embedding_batch_num"]

    async def initialize(self):
        """Initialize client and create k-NN vector index."""
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            await self._create_knn_index_if_not_exists()
            self._index_ready = True
            logger.debug(
                f"[{self.workspace}] OpenSearch Vector storage initialized: {self._index_name}"
            )

    async def _ensure_index_ready(self):
        """Recreate the vector index before the next write if it is missing."""
        if self._index_ready:
            return
        async with get_data_init_lock():
            if self.client is None:
                self.client = await ClientManager.get_client()
            if not self._index_ready:
                await self._create_knn_index_if_not_exists()
                self._index_ready = True

    def _mark_index_missing(self):
        """Mark the vector index as unavailable for subsequent read short-circuiting."""
        self._index_ready = False

    async def _create_knn_index_if_not_exists(self):
        try:
            if await self.client.indices.exists(index=self._index_name):
                # Validate existing index dimension
                try:
                    mapping = await self.client.indices.get_mapping(
                        index=self._index_name
                    )
                    existing_dim = (
                        mapping[self._index_name]["mappings"]["properties"]
                        .get("vector", {})
                        .get("dimension")
                    )
                    expected_dim = self.embedding_func.embedding_dim
                    if existing_dim is not None and existing_dim != expected_dim:
                        raise ValueError(
                            f"Vector dimension mismatch! Index '{self._index_name}' has "
                            f"dimension {existing_dim}, but current embedding model expects "
                            f"dimension {expected_dim}. Please drop the existing index or "
                            f"use an embedding model with matching dimensions."
                        )
                except (KeyError, TypeError):
                    logger.warning(
                        f"[{self.workspace}] Could not read vector mapping for index "
                        f"'{self._index_name}'; skipping dimension validation"
                    )
                return

            ef_construction = int(
                _get_opensearch_env("OPENSEARCH_KNN_EF_CONSTRUCTION", "200")
            )
            m = int(_get_opensearch_env("OPENSEARCH_KNN_M", "16"))
            ef_search = int(_get_opensearch_env("OPENSEARCH_KNN_EF_SEARCH", "100"))

            body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": ef_search,
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                    }
                },
                "mappings": {
                    "properties": {
                        "vector": {
                            "type": "knn_vector",
                            "dimension": self.embedding_func.embedding_dim,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "lucene",
                                "parameters": {
                                    "ef_construction": ef_construction,
                                    "m": m,
                                },
                            },
                        },
                        "content": {"type": "text"},
                        "entity_name": {"type": "keyword"},
                        "src_id": {"type": "keyword"},
                        "tgt_id": {"type": "keyword"},
                        "file_path": {"type": "keyword"},
                        "created_at": {"type": "long"},
                    },
                    "dynamic": True,
                },
            }
            await self.client.indices.create(index=self._index_name, body=body)
            logger.info(
                f"[{self.workspace}] Created k-NN index: {self._index_name} "
                f"(dim={self.embedding_func.embedding_dim})"
            )
        except RequestError as e:
            if "resource_already_exists_exception" not in str(e):
                logger.error(f"[{self.workspace}] Error creating k-NN index: {e}")
                raise
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error creating k-NN index: {e}")
            raise

    async def finalize(self):
        """Release the OpenSearch client connection."""
        if self.client is not None:
            await ClientManager.release_client(self.client)
            self.client = None

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Generate embeddings and upsert vectors in batches."""
        if not data:
            return
        await self._ensure_index_ready()
        logger.debug(
            f"[{self.workspace}] Upserting {len(data)} vectors to {self.namespace}"
        )
        current_time = int(time.time())

        list_data = [
            {
                "_id": k,
                "created_at": current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]

        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)

        for i, doc in enumerate(list_data):
            doc["vector"] = embeddings[i].tolist()

        actions = [
            {
                "_op_type": "index",
                "_index": self._index_name,
                "_id": doc["_id"],
                "_source": {k: v for k, v in doc.items() if k != "_id"},
            }
            for doc in list_data
        ]
        try:
            success, failed = await helpers.async_bulk(
                self.client, actions, raise_on_error=False, refresh="false"
            )
            if failed:
                logger.warning(
                    f"[{self.workspace}] {len(failed)} vectors failed to upsert"
                )
        except OpenSearchException as e:
            logger.error(f"[{self.workspace}] Error upserting vectors: {e}")
            raise

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """k-NN similarity search with cosine score conversion for lucene engine."""
        if not self._index_ready:
            return []
        if query_embedding is not None:
            query_vector = (
                query_embedding.tolist()
                if hasattr(query_embedding, "tolist")
                else list(query_embedding)
            )
        else:
            embedding = await self.embedding_func([query], _priority=5)
            query_vector = embedding[0].tolist()

        search_body = {
            "size": top_k,
            "query": {"knn": {"vector": {"vector": query_vector, "k": top_k}}},
            "_source": {"excludes": ["vector"]},
        }
        try:
            response = await self.client.search(
                index=self._index_name, body=search_body
            )
            results = []
            for hit in response["hits"]["hits"]:
                # OpenSearch k-NN with lucene engine and cosinesimil space type
                # returns scores that can be used directly as similarity measure.
                score = hit["_score"]

                if score >= self.cosine_better_than_threshold:
                    doc = hit["_source"]
                    doc["id"] = hit["_id"]
                    doc["distance"] = score
                    results.append(doc)
            logger.info(
                f"[{self.workspace}] Vector query on {self._index_name}: "
                f"top_k={top_k}, threshold={self.cosine_better_than_threshold}, "
                f"total_hits={len(response['hits']['hits'])}, "
                f"passed_filter={len(results)}, "
                f"score_range=[{min((h['_score'] for h in response['hits']['hits']), default=0):.4f}, "
                f"{max((h['_score'] for h in response['hits']['hits']), default=0):.4f}]"
            )
            return results
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return []
            logger.error(f"[{self.workspace}] Error querying vectors: {e}")
            return []

    async def index_done_callback(self) -> None:
        """Refresh index to make recently indexed vectors searchable."""
        if not self._index_ready:
            return
        try:
            await self.client.indices.refresh(index=self._index_name)
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return
        except Exception:
            pass

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get a vector document by ID."""
        if not self._index_ready:
            return None
        try:
            response = await _mget_optional_doc(self.client, self._index_name, id)
            if response is None:
                return None
            doc = response["_source"]
            doc["id"] = response["_id"]
            return doc
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return None
            logger.error(f"[{self.workspace}] Error getting vector {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector documents by IDs, preserving order."""
        if not ids:
            return []
        if not self._index_ready:
            return [None] * len(ids)
        try:
            response = await self.client.mget(index=self._index_name, body={"ids": ids})
            doc_map = {}
            for doc in response["docs"]:
                if doc.get("found"):
                    data = doc["_source"]
                    data["id"] = doc["_id"]
                    doc_map[doc["_id"]] = data
            return [doc_map.get(id) for id in ids]
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return [None] * len(ids)
            logger.error(f"[{self.workspace}] Error getting vectors by ids: {e}")
            return [None] * len(ids)

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get only the vector embeddings for given IDs."""
        if not ids:
            return {}
        if not self._index_ready:
            return {}
        try:
            response = await self.client.mget(
                index=self._index_name, body={"ids": ids}, _source_includes=["vector"]
            )
            result = {}
            for doc in response["docs"]:
                if doc.get("found") and "vector" in doc.get("_source", {}):
                    result[doc["_id"]] = doc["_source"]["vector"]
            return result
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return {}
            logger.error(f"[{self.workspace}] Error getting vectors: {e}")
            return {}

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by their IDs."""
        if not ids:
            return
        if not self._index_ready:
            return
        if isinstance(ids, set):
            ids = list(ids)
        try:
            actions = [
                {"_op_type": "delete", "_index": self._index_name, "_id": doc_id}
                for doc_id in ids
            ]
            result = await helpers.async_bulk(
                self.client, actions, raise_on_error=False, refresh="wait_for"
            )
            logger.debug(
                f"[{self.workspace}] Deleted {result[0]} vectors from {self.namespace}"
            )
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return
            logger.error(f"[{self.workspace}] Error deleting vectors: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity vector by computing its hash ID."""
        if not self._index_ready:
            return
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            try:
                await self.client.delete(
                    index=self._index_name, id=entity_id, refresh="wait_for"
                )
                logger.debug(f"[{self.workspace}] Deleted entity {entity_name}")
            except NotFoundError as e:
                if _is_missing_index_error(e):
                    self._mark_index_missing()
                    return
                logger.debug(f"[{self.workspace}] Entity {entity_name} not found")
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relation vectors where entity appears as src or tgt."""
        if not self._index_ready:
            return
        try:
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"src_id": entity_name}},
                            {"term": {"tgt_id": entity_name}},
                        ]
                    }
                }
            }
            await self.client.delete_by_query(
                index=self._index_name, body=body, refresh=True
            )
            logger.debug(
                f"[{self.workspace}] Deleted relations for entity {entity_name}"
            )
        except OpenSearchException as e:
            if _is_missing_index_error(e):
                self._mark_index_missing()
                return
            logger.error(
                f"[{self.workspace}] Error deleting relations for {entity_name}: {e}"
            )

    async def drop(self) -> dict[str, str]:
        """Delete and recreate the vector index."""
        try:
            try:
                await self.client.indices.delete(index=self._index_name)
                logger.info(
                    f"[{self.workspace}] Dropped vector index: {self._index_name}"
                )
            except NotFoundError:
                logger.info(
                    f"[{self.workspace}] Vector index already missing during drop: {self._index_name}"
                )
            # Recreate the index
            await self._create_knn_index_if_not_exists()
            self._index_ready = True
            logger.info(
                f"[{self.workspace}] Dropped and recreated vector index: {self._index_name}"
            )
            return {
                "status": "success",
                "message": f"Vector index {self._index_name} dropped and recreated",
            }
        except OpenSearchException as e:
            self._mark_index_missing()
            logger.error(f"[{self.workspace}] Error dropping vector index: {e}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            self._mark_index_missing()
            logger.error(
                f"[{self.workspace}] Unexpected error dropping vector index: {e}"
            )
            return {"status": "error", "message": str(e)}


@dataclass
class OpenSearchGraphVectorStorage(BaseVectorStorage):
    """Vector storage backed by graph plugin nodes with hybrid retrieval.

    Stores data as graph nodes (Entity or Chunk) and queries via the
    ``POST _plugins/_graph/retrieve`` hybrid retrieval endpoint. Configured
    with ``_node_label`` (e.g., "Entity" or "Chunk"), ``_key_property`` for
    lookups, and ``_merge_key`` for upserts.

    Not directly user-selectable — auto-created when the graph plugin is
    detected (see ``lightrag.py`` implicit wiring).
    """

    _graph_storage: OpenSearchGraphStorage = field(default=None, init=False)
    _client: AsyncOpenSearch = field(default=None, init=False)
    _node_label: str = field(default="Entity", init=False)
    _key_property: str = field(default="vdb_id", init=False)
    _merge_key: str = field(default="entity_id", init=False)

    def __init__(
        self,
        namespace,
        workspace,
        embedding_func,
        meta_fields=None,
        node_label="Entity",
        key_property="vdb_id",
        merge_key="entity_id",
        graph_storage=None,
        global_config=None,
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config or (graph_storage.global_config if graph_storage else {}),
            embedding_func=embedding_func,
            meta_fields=meta_fields or set(),
        )
        self._graph_storage = graph_storage
        self._node_label = node_label
        self._key_property = key_property
        self._merge_key = merge_key
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        # Hybrid retrieval returns a fused score (vector + text + graph) that
        # is NOT directly comparable to cosine similarity.  Use a dedicated
        # threshold (default 0.0 = pass-through) so that the cosine-specific
        # `cosine_better_than_threshold` is never mis-applied.
        self.hybrid_score_threshold = kwargs.get(
            "hybrid_score_threshold", 0.0
        )

    async def initialize(self):
        """Obtain client reference from graph storage (no ClientManager call)."""
        if self._graph_storage is not None:
            self._client = self._graph_storage.client
        logger.debug(
            f"[{self.workspace}] OpenSearchGraphVectorStorage initialized: "
            f"label={self._node_label}, key={self._key_property}"
        )

    async def finalize(self):
        """No-op: graph storage owns the client lifecycle."""
        pass

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Batch-embed and write nodes via Cypher + OpenSearch bulk embedding update.

        The graph plugin stores Cypher properties in a ``properties`` flat_object,
        which serialises arrays to strings.  The top-level ``embedding`` knn_vector
        field is separate and must be written via the OpenSearch document API.

        Flow:
        1. Cypher UNWIND MERGE — create/update node properties (no embedding).
        2. Cypher RETURN — collect the internal node UUIDs (= OpenSearch ``_id``).
        3. OpenSearch bulk script update — set the top-level ``embedding`` field.
        """
        if not data:
            return

        # Ensure graph database exists (may have been dropped).
        gs = self._graph_storage
        await gs._ensure_database_ready()

        logger.debug(
            f"[{self.workspace}] Upserting {len(data)} {self._node_label} nodes"
        )

        # Collect items and compute embeddings in batch
        items = []
        contents = []
        for doc_id, doc_data in data.items():
            items.append((doc_id, doc_data))
            contents.append(doc_data.get("content", ""))

        # Batch embedding computation
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)

        # Build Cypher items (no embedding — that goes via bulk update).
        # NOTE: Properties are flattened into each item dict (not nested as
        # ``item.props``) because the OpenSearch graph plugin's Cypher engine
        # silently drops nested map values in ``SET n += item.props``.
        cypher_items = []
        prop_keys: set[str] = set()
        for i, (doc_id, doc_data) in enumerate(items):
            props = {k: v for k, v in doc_data.items() if k in self.meta_fields}
            props["content"] = doc_data.get("content", "")

            # Ensure integer promoted fields are properly typed.
            for int_field in ("tokens", "chunk_order_index"):
                if int_field in props:
                    try:
                        props[int_field] = int(props[int_field])
                    except (ValueError, TypeError):
                        pass

            if self._node_label == "Entity":
                entity_name = doc_data.get("entity_name", doc_id)
                vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
                item = {"merge_key_val": entity_name, "vdb_id": vdb_id}
                item.update(props)
                cypher_items.append(item)
            else:
                # Chunk nodes: key = doc_id
                item = {"merge_key_val": doc_id}
                item.update(props)
                cypher_items.append(item)

            prop_keys.update(props.keys())

        # Build SET clause templates.  We use per-item non-UNWIND MERGE
        # because the graph plugin's UNWIND + MERGE always creates new nodes
        # (it lacks cross-row deduplication) and UNWIND + MATCH + SET silently
        # drops the SET.  Per-item MERGE with ON CREATE/ON MATCH SET is the
        # only reliable pattern.
        set_parts = [f"n.{k} = ${k}" for k in sorted(prop_keys)]

        # --- Step 1 + 2: per-item Cypher MERGE + collect UUIDs ---
        node_uuids = []  # parallel to cypher_items
        if self._node_label == "Entity":
            set_parts_entity = set_parts + ["n.vdb_id = $vdb_id"]
            set_clause = ", ".join(set_parts_entity)
            for item in cypher_items:
                params = {k: v for k, v in item.items() if k != "merge_key_val"}
                params["id"] = item["merge_key_val"]
                resp = await gs._execute_cypher(
                    "MERGE (n:Entity {entity_id: $id}) "
                    f"ON CREATE SET {set_clause} "
                    f"ON MATCH SET {set_clause} "
                    "RETURN n",
                    params,
                )
                for row in resp.get("data", []):
                    node_uuids.append(row.get("n", ""))

            # Create MENTIONED_IN edges from entities to source chunks.
            for _doc_id, doc_data in items:
                entity_name = doc_data.get("entity_name", _doc_id)
                source_id = doc_data.get("source_id", "")
                if source_id:
                    chunk_ids = [
                        cid.strip()
                        for cid in source_id.split(GRAPH_FIELD_SEP)
                        if cid.strip()
                    ]
                    for cid in chunk_ids:
                        p = {"ename": entity_name, "cid": cid}
                        resp = await gs._execute_cypher(
                            "MATCH (e:Entity {entity_id: $ename})-[r:MENTIONED_IN]->(c:Chunk {id: $cid}) "
                            "RETURN count(r) AS cnt", p,
                        )
                        if resp.get("data", [{}])[0].get("cnt", 0) == 0:
                            await gs._execute_cypher(
                                "MATCH (e:Entity {entity_id: $ename}), (c:Chunk {id: $cid}) "
                                "CREATE (e)-[:MENTIONED_IN]->(c)", p,
                            )
        elif self._node_label == "Chunk":
            set_clause = ", ".join(set_parts) if set_parts else "n.content = $content"
            for item in cypher_items:
                params = {k: v for k, v in item.items() if k != "merge_key_val"}
                params["id"] = item["merge_key_val"]
                resp = await gs._execute_cypher(
                    "MERGE (n:Chunk {id: $id}) "
                    f"ON CREATE SET {set_clause} "
                    f"ON MATCH SET {set_clause} "
                    "RETURN n",
                    params,
                )
                for row in resp.get("data", []):
                    node_uuids.append(row.get("n", ""))

            # Create PART_OF edges to Document nodes (per-item).
            for _doc_id, doc_data in items:
                full_doc_id = doc_data.get("full_doc_id")
                if full_doc_id:
                    p = {
                        "cid": _doc_id,
                        "did": full_doc_id,
                        "fp": doc_data.get("file_path", ""),
                    }
                    # Upsert Document node (MERGE on nodes is fast).
                    await gs._execute_cypher(
                        "MERGE (d:Document {id: $did}) "
                        "ON CREATE SET d.file_path = $fp "
                        "ON MATCH SET d.file_path = $fp",
                        p,
                    )
                    # Check-then-CREATE for PART_OF edge.
                    resp = await gs._execute_cypher(
                        "MATCH (c:Chunk {id: $cid})-[r:PART_OF]->(d:Document {id: $did}) "
                        "RETURN count(r) AS cnt", p,
                    )
                    if resp.get("data", [{}])[0].get("cnt", 0) == 0:
                        await gs._execute_cypher(
                            "MATCH (c:Chunk {id: $cid}), (d:Document {id: $did}) "
                            "CREATE (c)-[:PART_OF]->(d)", p,
                        )

        # --- Step 3: Bulk-update top-level embedding via OpenSearch ---
        # The graph plugin requires a top-level ``embedding`` field (knn_vector)
        # for hybrid retrieval, but Cypher SET only writes to ``properties.*``.
        # A direct index update is the only way to populate the promoted field.
        if node_uuids:
            node_index = f"{gs.database_name}-lpg-nodes"
            actions = []
            for idx, uuid in enumerate(node_uuids):
                if not uuid or idx >= len(embeddings):
                    continue
                actions.append({"update": {"_index": node_index, "_id": uuid}})
                actions.append({
                    "script": {
                        "source": (
                            "ctx._source.embedding = params.embedding; "
                            "if (ctx._source.properties == null) { ctx._source.properties = new HashMap(); } "
                            "ctx._source.properties.embedding = params.embedding"
                        ),
                        "params": {"embedding": embeddings[idx].tolist()},
                    }
                })
            if actions:
                body = "\n".join(
                    json_module.dumps(a) for a in actions
                ) + "\n"
                await self._client.bulk(body=body, refresh="false")

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Hybrid retrieval via ``POST _plugins/_graph/retrieve``."""
        gs = self._graph_storage
        await gs._ensure_database_ready()

        if query_embedding is not None:
            query_vector = (
                query_embedding.tolist()
                if hasattr(query_embedding, "tolist")
                else list(query_embedding)
            )
        else:
            embedding = await self.embedding_func([query], _priority=5)
            query_vector = embedding[0].tolist()
        retrieve_body = {
            "query_text": query,
            "query_vector": query_vector,
            "database": gs.database_name,
            "vector_field": "embedding",
            "seed_k": top_k * 2,
            "top_k": top_k,
            "hops": 2,
            "node_labels": [self._node_label],
            "weights": {"vector": 0.4, "text": 0.3, "graph": 0.3},
        }

        try:
            resp = await self._client.transport.perform_request(
                "POST", "/_plugins/_graph/retrieve", body=retrieve_body
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Hybrid retrieval failed for {self._node_label}: {e}"
            )
            return []

        results = []
        for r in resp.get("results", []):
            raw_props = r.get("properties", {})
            score = r.get("score", 0.0)
            if score < self.hybrid_score_threshold:
                continue

            # Hybrid retrieval returns property keys in the ``properties`` map.
            # Non-promoted fields have a "properties." prefix (e.g.,
            # "properties.source_id").  Promoted fields appear without the prefix
            # (e.g., "entity_id", "content").  ``removeprefix`` handles both.
            props = {}
            for k, v in raw_props.items():
                clean_key = k.removeprefix("properties.")
                if clean_key.startswith("__"):
                    continue  # skip internal keys like __labels
                props[clean_key] = v

            if self._node_label == "Entity":
                entity_name = props.get("entity_id", "")
                doc = {
                    "id": props.get("vdb_id", ""),
                    "entity_name": entity_name,
                    "distance": score,
                }
            else:
                doc = {
                    "id": props.get("id", r.get("node_id", "")),
                    "distance": score,
                }
            # Merge all properties
            for k, v in props.items():
                if k not in ("embedding", "vdb_id", "id"):
                    doc.setdefault(k, v)
            results.append(doc)

        logger.info(
            f"[{self.workspace}] Hybrid retrieval on {self._node_label}: "
            f"top_k={top_k}, results={len(results)}"
        )
        return results

    async def delete(self, ids: list[str]) -> None:
        """Delete nodes by their key_property IDs."""
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)
        gs = self._graph_storage
        try:
            await gs._execute_cypher(
                f"UNWIND $ids AS id "
                f"MATCH (n:{self._node_label} {{{self._key_property}: id}}) "
                f"DETACH DELETE n",
                {"ids": ids},
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error deleting {self._node_label} nodes: {e}"
            )

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by name (computes hash ID for vdb_id lookup)."""
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        await self.delete([entity_id])

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all DIRECTED edges for an entity."""
        gs = self._graph_storage
        try:
            await gs._execute_cypher(
                "MATCH (a:Entity {entity_id: $name})-[r:DIRECTED]-() DELETE r",
                {"name": entity_name},
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error deleting relations for {entity_name}: {e}"
            )

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get a node by its key_property."""
        gs = self._graph_storage
        try:
            resp = await gs._execute_cypher(
                f"MATCH (n:{self._node_label} {{{self._key_property}: $id}}) "
                f"RETURN properties(n) AS props",
                {"id": id},
            )
            rows = gs._cypher_rows(resp)
            if rows:
                props = rows[0][0]
                if props:
                    props.pop("embedding", None)
                    props["id"] = id
                    return props
            return None
        except Exception:
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple nodes by key_property IDs."""
        if not ids:
            return []
        gs = self._graph_storage
        try:
            resp = await gs._execute_cypher(
                f"UNWIND $ids AS id "
                f"MATCH (n:{self._node_label} {{{self._key_property}: id}}) "
                f"RETURN id, properties(n) AS props",
                {"ids": ids},
            )
            result_map = {}
            for row in gs._cypher_rows(resp):
                key = row[0]
                props = row[1] or {}
                props.pop("embedding", None)
                props["id"] = key
                result_map[key] = props
            return [result_map.get(id) for id in ids]
        except Exception:
            return [None] * len(ids)

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get embedding vectors by key_property IDs."""
        if not ids:
            return {}
        gs = self._graph_storage
        try:
            resp = await gs._execute_cypher(
                f"UNWIND $ids AS id "
                f"MATCH (n:{self._node_label} {{{self._key_property}: id}}) "
                f"RETURN id, n.embedding AS emb",
                {"ids": ids},
            )
            result = {}
            for row in gs._cypher_rows(resp):
                key = row[0]
                emb = row[1]
                if emb:
                    result[key] = [float(v) for v in emb]
            return result
        except Exception:
            return {}

    async def index_done_callback(self) -> None:
        """Refresh the node index to make newly written embeddings searchable."""
        if self._graph_storage and self._graph_storage._database_ready:
            try:
                node_index = f"{self._graph_storage.database_name}-lpg-nodes"
                await self._client.indices.refresh(index=node_index)
            except Exception:
                pass

    async def drop(self) -> dict[str, str]:
        """No-op: data is dropped when OpenSearchGraphStorage.drop() deletes the database."""
        return {"status": "success", "message": "data dropped"}


@dataclass
class OpenSearchGraphRelationshipAdapter(BaseVectorStorage):
    """Adapter implementing BaseVectorStorage for relationship retrieval.

    Relationships are stored as DIRECTED graph edges, not as vector embeddings.
    The ``query()`` method uses hybrid retrieval to find entities, then fetches
    pairwise DIRECTED edges between them. All other methods are no-ops or
    trivially implemented.

    Not directly user-selectable — auto-created when the graph plugin is
    detected (see ``lightrag.py`` implicit wiring).
    """

    _graph_storage: OpenSearchGraphStorage = field(default=None, init=False)
    _entities_vdb: OpenSearchGraphVectorStorage = field(default=None, init=False)
    _client: AsyncOpenSearch = field(default=None, init=False)
    _max_relationship_results: int = field(default=500, init=False)

    def __init__(
        self,
        namespace,
        workspace,
        embedding_func,
        meta_fields=None,
        entities_vdb=None,
        graph_storage=None,
        global_config=None,
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config or (graph_storage.global_config if graph_storage else {}),
            embedding_func=embedding_func,
            meta_fields=meta_fields or set(),
        )
        self._graph_storage = graph_storage
        self._entities_vdb = entities_vdb

    async def initialize(self):
        """Obtain client reference from graph storage (no ClientManager call)."""
        if self._graph_storage is not None:
            self._client = self._graph_storage.client
        logger.debug(
            f"[{self.workspace}] OpenSearchGraphRelationshipAdapter initialized"
        )

    async def finalize(self):
        """No-op: graph storage owns the client lifecycle."""
        pass

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """No-op: relationships stored as graph edges by OpenSearchGraphStorage.upsert_edge()."""
        pass

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Find entities via hybrid retrieval, then fetch pairwise DIRECTED edges.

        Returns results formatted as relationship vector store entries with
        ``rel-*`` hashed IDs for compatibility with existing callers.
        """
        # Step 1: Find relevant entities
        entity_results = await self._entities_vdb.query(
            query, top_k=top_k, query_embedding=query_embedding
        )
        if not entity_results:
            return []

        # Extract entity names (the entity_id / graph key)
        entity_ids = []
        for r in entity_results:
            ename = r.get("entity_name")
            if ename:
                entity_ids.append(ename)
        if not entity_ids:
            return []

        # Step 2: Pairwise edge fetch
        limit = min(top_k, self._max_relationship_results)
        gs = self._graph_storage
        all_rows = []
        try:
            resp = await gs._execute_cypher(
                "MATCH (a:Entity)-[r:DIRECTED]-(b:Entity) "
                "WHERE a.entity_id IN $entity_ids AND b.entity_id IN $entity_ids "
                "RETURN a.entity_id AS src, b.entity_id AS tgt, properties(r) AS props "
                "ORDER BY r.weight DESC "
                "LIMIT $limit",
                {"entity_ids": entity_ids, "limit": limit},
            )
            all_rows = gs._cypher_rows(resp)
        except Exception:
            # Full list failed — fall back to per-entity star queries
            for eid in entity_ids:
                try:
                    resp = await gs._execute_cypher(
                        "MATCH (a:Entity {entity_id: $eid})-[r:DIRECTED]-(b:Entity) "
                        "WHERE b.entity_id IN $entity_ids "
                        "RETURN a.entity_id AS src, b.entity_id AS tgt, properties(r) AS props "
                        "ORDER BY r.weight DESC LIMIT $limit",
                        {"eid": eid, "entity_ids": entity_ids, "limit": limit},
                        retries=0,
                    )
                    all_rows.extend(gs._cypher_rows(resp))
                except Exception:
                    pass

        # Step 3: Transform to relationship vector store format
        results = []
        seen = set()
        for row in all_rows:
            src = row[0]
            tgt = row[1]
            props = row[2] or {}

            # Canonicalize endpoint order so the same edge always
            # produces the same id, src_id, tgt_id regardless of which
            # direction Cypher returns the undirected match.
            canon_src, canon_tgt = sorted([src, tgt])
            edge_key = (canon_src, canon_tgt)
            if edge_key in seen:
                continue
            seen.add(edge_key)

            description = props.get("description", "")
            keywords = props.get("keywords", "")
            weight = props.get("weight", 1.0)

            rel_id = compute_mdhash_id(canon_src + canon_tgt, prefix="rel-")
            results.append({
                "id": rel_id,
                "src_id": canon_src,
                "tgt_id": canon_tgt,
                "content": f"{keywords}\t{canon_src}\n{canon_tgt}\n{description}",
                "description": description,
                "keywords": keywords,
                "weight": weight,
                "distance": weight if isinstance(weight, (int, float)) else 1.0,
                "source_id": props.get("source_id", ""),
                "file_path": props.get("file_path", ""),
            })

        logger.info(
            f"[{self.workspace}] Relationship query: "
            f"entities={len(entity_ids)}, edges={len(results)}"
        )
        return results

    async def delete(self, ids: list[str]) -> None:
        """No-op: edge deletion handled by graph storage remove_edges()."""
        pass

    async def delete_entity(self, entity_name: str) -> None:
        """No-op: entity deletion cascades edges in graph storage."""
        pass

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all DIRECTED edges for an entity."""
        gs = self._graph_storage
        try:
            await gs._execute_cypher(
                "MATCH (a:Entity {entity_id: $name})-[r:DIRECTED]-() DELETE r",
                {"name": entity_name},
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error deleting relations for {entity_name}: {e}"
            )

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Return None: hashed rel-* IDs are not directly reversible."""
        return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Return empty list: callers tolerate empty."""
        return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Return empty dict: no embeddings stored for relationships."""
        return {}

    async def index_done_callback(self) -> None:
        """No-op: graph database handles its own commit/refresh."""
        pass

    @property
    async def client_storage(self) -> dict[str, list[dict]]:
        """Return all DIRECTED edges for export/introspection.

        Returns the same ``{"data": [{"__id__": ..., ...}]}`` shape that
        ``NanoVectorDBStorage.client_storage`` and
        ``FaissVectorDBStorage.client_storage`` provide, so that
        ``export_data()`` in ``utils.py`` works unchanged.
        """
        gs = self._graph_storage
        if gs is None or not gs._database_ready:
            return {"data": []}
        try:
            resp = await gs._execute_cypher(
                "MATCH (a:Entity)-[r:DIRECTED]->(b:Entity) "
                "RETURN a.entity_id AS src, b.entity_id AS tgt, properties(r) AS props"
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] client_storage fetch failed: {e}")
            return {"data": []}

        records = []
        for row in gs._cypher_rows(resp):
            src = row[0]
            tgt = row[1]
            props = row[2] or {}
            canon_src, canon_tgt = sorted([src, tgt])
            rel_id = compute_mdhash_id(canon_src + canon_tgt, prefix="rel-")
            records.append({
                "__id__": rel_id,
                "src_id": canon_src,
                "tgt_id": canon_tgt,
                "description": props.get("description", ""),
                "keywords": props.get("keywords", ""),
                "weight": props.get("weight", 1.0),
                "source_id": props.get("source_id", ""),
            })
        return {"data": records}

    async def drop(self) -> dict[str, str]:
        """No-op: edges are dropped when OpenSearchGraphStorage.drop() deletes the database."""
        return {"status": "success", "message": "data dropped"}
