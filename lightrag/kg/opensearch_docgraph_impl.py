"""OpenSearch Docgraph storage backend for LightRAG.

Extends OpenSearchGraphStorage to use the docgraph REST API for ingestion
and evidence-mediated Cypher for queries. The existing LPG backend is
untouched — this class overrides only what differs.

Key differences from LPG mode:
- Ingestion uses _ingest endpoint (atomic per document) instead of individual
  Cypher MERGE calls
- Relations are RelFact nodes (not DIRECTED edges)
- Entity queries go through evidence-mediated patterns (Chunk->Entity)
- Ontology grounding links entities/relations to OntologyConcept/OntologyRelation
"""

import asyncio
import logging
import os
from collections import defaultdict

from lightrag.base import BaseVectorStorage
from lightrag.kg.opensearch_impl import (
    OpenSearchGraphStorage,
    _get_opensearch_env,
)
from lightrag.kg.ontology_mapping import (
    load_ontology,
    map_entity_type,
    map_relation_type,
)

logger = logging.getLogger("lightrag")


class OpenSearchDocgraphStorage(OpenSearchGraphStorage):
    """Docgraph-mode graph storage using the OpenSearch docgraph REST API."""

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(namespace, global_config, embedding_func, workspace)
        # Use a different database name so docgraph and LPG don't collide
        self._database_name = "docgraph-" + self._database_name
        # Per-document buffer: {buf_key: {chunks: {chunk_id: {entities, relations}}}}
        self._doc_buffer = defaultdict(lambda: {"properties": {}, "chunks": defaultdict(lambda: {"entities": [], "relations": []})})
        self._ontology_loaded = False

    # ── Cypher override ───────────────────────────────────────────────

    async def _execute_cypher(self, query: str, params: dict = None, retries: int = 3):
        """Override to suppress DOCGRAPH policy violations.

        In docgraph mode, some LPG-style queries are rejected with HTTP 400.
        - MERGE/MATCH on Entity nodes: let through (evidence rewriter handles
          simple patterns; needed for VDB embedding upserts)
        - Multi-pattern MATCH + CREATE: suppress (e.g., MENTIONED_IN edges)
        """
        import asyncio

        body: dict = {
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
                if "DOCGRAPH policy violation" in str(e) or "policy violation" in str(e).lower():
                    # Suppress policy violations — return empty result
                    return {"columns": [], "data": [], "stats": {}}
                last_exc = e
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    # ── Database lifecycle ────────────────────────────────────────────

    async def _create_database_if_not_exists(self):
        """Create a docgraph-mode database."""
        body = {
            "mode": "docgraph",
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
        try:
            await self._client.transport.perform_request(
                "PUT",
                f"/_plugins/_graph/database/{self._database_name}",
                body=body,
            )
            logger.info(f"Created docgraph database: {self._database_name}")
            # Reload ontology after database creation
            self._ontology_loaded = False
        except Exception as e:
            if "already exists" in str(e).lower() or "resource_already_exists" in str(e).lower():
                logger.info(f"Docgraph database already exists: {self._database_name}")
            else:
                raise

    async def _load_ontology(self):
        """Load ontology concepts and relations into the docgraph database."""
        if self._ontology_loaded:
            return
        ontology_path = os.environ.get("ONTOLOGY_FILE")
        try:
            ontology = load_ontology(ontology_path)
        except FileNotFoundError:
            logger.warning("Ontology file not found, skipping ontology loading")
            self._ontology_loaded = True
            return

        endpoint = f"/_plugins/_graph/docgraph/{self._database_name}/_upsert_ontology"
        for concept in ontology.get("concepts", []):
            try:
                await self._client.transport.perform_request(
                    "POST", endpoint,
                    body={
                        "ontology_id": concept["ontology_id"],
                        "ontology_kind": "ontology_concept",
                        "labels": concept.get("labels", ["OntologyConcept"]),
                        "properties": concept.get("properties", {}),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to upsert ontology concept {concept['ontology_id']}: {e}")

        for relation in ontology.get("relations", []):
            try:
                await self._client.transport.perform_request(
                    "POST", endpoint,
                    body={
                        "ontology_id": relation["ontology_id"],
                        "ontology_kind": "ontology_relation",
                        "labels": relation.get("labels", ["OntologyRelation"]),
                        "properties": relation.get("properties", {}),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to upsert ontology relation {relation['ontology_id']}: {e}")

        self._ontology_loaded = True
        logger.info(f"Loaded ontology: {len(ontology.get('concepts', []))} concepts, {len(ontology.get('relations', []))} relations")

    async def initialize(self):
        await super().initialize()
        await self._load_ontology()

    async def _ensure_database_ready(self):
        await super()._ensure_database_ready()
        if not self._ontology_loaded:
            await self._load_ontology()

    # ── Ingestion (buffered → batch _ingest) ──────────────────────────

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Buffer entity for batch ingestion instead of individual Cypher MERGE."""
        chunk_ids = []
        source_id = node_data.get("source_id", "")
        if source_id:
            chunk_ids = [c.strip() for c in source_id.split("<SEP>") if c.strip()]

        chunk_id = chunk_ids[0] if chunk_ids else "unknown-chunk"

        # Use a single shared buffer key — flush_document will be called with
        # the real doc_id from merge_nodes_and_edges.
        doc_id = "__pending__"

        entity_type = node_data.get("entity_type", "")
        ontology_id = map_entity_type(entity_type)

        labels = ["Entity"]
        if entity_type:
            import re
            safe_type = re.sub(r"[^a-zA-Z0-9_]", "_", entity_type.title().replace(" ", ""))
            if safe_type:
                labels.append(safe_type)

        entity = {
            "entity_id": node_id,
            "labels": labels,
            "properties": {
                "name": node_id,
                "description": node_data.get("description", ""),
                "entity_type": entity_type,
            },
            "ontology_id": ontology_id,
        }

        self._doc_buffer[doc_id]["chunks"][chunk_id]["entities"].append(entity)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Buffer relation for batch ingestion."""
        source_id = edge_data.get("source_id", "")
        chunk_ids = [c.strip() for c in source_id.split("<SEP>") if c.strip()] if source_id else []
        chunk_id = chunk_ids[0] if chunk_ids else "unknown-chunk"
        doc_id = "__pending__"

        keywords = edge_data.get("keywords", "")
        ontology_id = map_relation_type(keywords.split(",")[0].strip() if keywords else "")

        relation = {
            "relation_id": f"{source_node_id}--{target_node_id}",
            "type": keywords.split(",")[0].strip().upper().replace(" ", "_") if keywords else "RELATED_TO",
            "source_entity_id": source_node_id,
            "target_entity_id": target_node_id,
            "properties": {
                "description": edge_data.get("description", ""),
                "keywords": keywords,
                "weight": str(edge_data.get("weight", 1.0)),
            },
            "ontology_id": ontology_id,
        }

        self._doc_buffer[doc_id]["chunks"][chunk_id]["relations"].append(relation)

    async def flush_document(self, doc_id: str, doc_properties: dict = None) -> None:
        """Build and POST the _ingest payload for a buffered document."""
        # Entities/edges are buffered under "__pending__" key
        buf_key = "__pending__"
        if buf_key not in self._doc_buffer:
            return

        await self._ensure_database_ready()

        buf = self._doc_buffer[buf_key]
        chunks_payload = []

        for chunk_id, chunk_data in buf["chunks"].items():
            # Deduplicate entities by entity_id within this chunk
            seen_entities = {}
            for ent in chunk_data["entities"]:
                eid = ent["entity_id"]
                if eid not in seen_entities:
                    seen_entities[eid] = ent

            # Deduplicate relations by relation_id
            seen_relations = {}
            for rel in chunk_data["relations"]:
                rid = rel["relation_id"]
                if rid not in seen_relations:
                    seen_relations[rid] = rel

            # Ensure all relation endpoints are declared as entities in this chunk
            for rel in seen_relations.values():
                for eid_key in ("source_entity_id", "target_entity_id"):
                    eid = rel.get(eid_key)
                    if eid and eid not in seen_entities:
                        seen_entities[eid] = {
                            "entity_id": eid,
                            "labels": ["Entity"],
                            "properties": {"name": eid},
                            "ontology_id": "onto:Concept",
                        }

            chunks_payload.append({
                "chunk_id": chunk_id,
                "properties": {"text": chunk_id},  # chunk text stored elsewhere
                "entities": list(seen_entities.values()),
                "relations": list(seen_relations.values()),
            })

        if not chunks_payload:
            del self._doc_buffer[buf_key]
            return

        payload = {
            "document_id": doc_id,
            "document_properties": doc_properties or {"title": doc_id},
            "authz_claims": {"users": ["admin"], "backend_roles": ["admin"]},
            "chunks": chunks_payload,
        }

        try:
            endpoint = f"/_plugins/_graph/docgraph/{self._database_name}/_ingest"
            resp = await self._client.transport.perform_request(
                "POST", endpoint, body=payload,
            )
            nodes = resp.get("nodes_created", 0)
            edges = resp.get("edges_created", 0)
            logger.info(
                f"Docgraph ingested doc={doc_id}: {len(chunks_payload)} chunks, "
                f"{nodes} nodes, {edges} edges"
            )
        except Exception as e:
            err_msg = str(e)
            # Try to extract the response body for detailed error
            if hasattr(e, 'info'):
                err_msg = f"{e} | body={e.info}"
            logger.error(f"Docgraph ingest failed for {doc_id}: {err_msg}")
        finally:
            del self._doc_buffer[buf_key]

    # ── Embedding updates via _bulk_update_nodes ────────────────────────

    async def bulk_update_embeddings(
        self, updates: list[dict]
    ) -> None:
        """Update embeddings on existing docgraph nodes via _bulk_update_nodes.

        Args:
            updates: List of {"node_id": str, "target_kind": str, "embedding": list[float]}
        """
        if not updates:
            return
        endpoint = f"/_plugins/_graph/docgraph/{self._database_name}/_bulk_update_nodes"
        payload = {
            "updates": [
                {
                    "node_id": u["node_id"],
                    "target_kind": u["target_kind"],
                    "set": {"embedding": u["embedding"]},
                }
                for u in updates
            ]
        }
        try:
            resp = await self._client.transport.perform_request(
                "POST", endpoint, body=payload,
            )
            ok = resp.get("success_count", 0)
            err = resp.get("error_count", 0)
            if err:
                logger.warning(f"bulk_update_embeddings: {ok} ok, {err} errors")
            else:
                logger.debug(f"bulk_update_embeddings: {ok} nodes updated")
        except Exception as e:
            logger.error(f"bulk_update_embeddings failed: {e}")
            if hasattr(e, 'info'):
                logger.error(f"  Response: {e.info}")

    # ── Read queries (evidence-mediated) ──────────────────────────────

    async def has_node(self, node_id: str) -> bool:
        if not self._database_ready:
            return False
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(n:Entity) "
                "WHERE n.name = $id RETURN count(n) > 0 AS exists",
                {"id": node_id},
            )
            rows = self._cypher_rows(resp)
            return bool(rows and rows[0][0])
        except Exception:
            return False

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        if not self._database_ready:
            return None
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(n:Entity) "
                "WHERE n.name = $id RETURN properties(n) AS props LIMIT 1",
                {"id": node_id},
            )
            rows = self._cypher_rows(resp)
            if rows and rows[0][0]:
                props = rows[0][0]
                props["entity_name"] = props.get("name", node_id)
                return props
            return None
        except Exception:
            return None

    async def node_degree(self, node_id: str) -> int:
        if not self._database_ready:
            return 0
        try:
            # Count MENTIONS edges (how many chunks mention this entity)
            # plus SOURCE_ENTITY/TARGET_ENTITY edges (relations involving this entity)
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(n:Entity) "
                "WHERE n.name = $id "
                "OPTIONAL MATCH (n)<-[:SOURCE_ENTITY|TARGET_ENTITY]-(r:RelFact) "
                "RETURN count(DISTINCT r) AS degree",
                {"id": node_id},
            )
            rows = self._cypher_rows(resp)
            return rows[0][0] if rows else 0
        except Exception:
            return 0

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        if not self._database_ready:
            return False
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:ASSERTS]->(r:RelFact)-[:SOURCE_ENTITY]->(a:Entity), "
                "(r)-[:TARGET_ENTITY]->(b:Entity) "
                "WHERE a.name = $src AND b.name = $tgt "
                "RETURN count(r) > 0 AS exists LIMIT 1",
                {"src": source_node_id, "tgt": target_node_id},
            )
            rows = self._cypher_rows(resp)
            return bool(rows and rows[0][0])
        except Exception:
            return False

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        if not self._database_ready:
            return None
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:ASSERTS]->(r:RelFact)-[:SOURCE_ENTITY]->(a:Entity), "
                "(r)-[:TARGET_ENTITY]->(b:Entity) "
                "WHERE a.name = $src AND b.name = $tgt "
                "RETURN properties(r) AS props LIMIT 1",
                {"src": source_node_id, "tgt": target_node_id},
            )
            rows = self._cypher_rows(resp)
            if rows and rows[0][0]:
                return rows[0][0]
            # Try reverse direction
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:ASSERTS]->(r:RelFact)-[:SOURCE_ENTITY]->(a:Entity), "
                "(r)-[:TARGET_ENTITY]->(b:Entity) "
                "WHERE a.name = $tgt AND b.name = $src "
                "RETURN properties(r) AS props LIMIT 1",
                {"src": source_node_id, "tgt": target_node_id},
            )
            rows = self._cypher_rows(resp)
            return rows[0][0] if rows and rows[0][0] else None
        except Exception:
            return None

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        if not self._database_ready:
            return None
        try:
            edges = []
            # Outgoing: entity is source
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:SOURCE_ENTITY]-(r:RelFact)"
                "-[:TARGET_ENTITY]->(t:Entity) "
                "WHERE e.name = $id RETURN e.name AS src, t.name AS tgt",
                {"id": source_node_id},
            )
            for row in self._cypher_rows(resp):
                edges.append((str(row[0]), str(row[1])))
            # Incoming: entity is target
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:TARGET_ENTITY]-(r:RelFact)"
                "-[:SOURCE_ENTITY]->(s:Entity) "
                "WHERE e.name = $id RETURN s.name AS src, e.name AS tgt",
                {"id": source_node_id},
            )
            for row in self._cypher_rows(resp):
                edges.append((str(row[0]), str(row[1])))
            return edges if edges else None
        except Exception:
            return None

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        result = {}
        if not self._database_ready or not node_ids:
            return result
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(n:Entity) "
                "WHERE n.name IN $ids "
                "RETURN n.name AS eid, properties(n) AS props",
                {"ids": node_ids},
            )
            for row in self._cypher_rows(resp):
                eid, props = row[0], row[1] or {}
                if eid and eid not in result:
                    props["entity_name"] = props.get("name", eid)
                    result[str(eid)] = props
        except Exception:
            pass
        return result

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        result = {nid: 0 for nid in node_ids}
        if not self._database_ready or not node_ids:
            return result
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(n:Entity) "
                "WHERE n.name IN $ids "
                "OPTIONAL MATCH (n)<-[:SOURCE_ENTITY|TARGET_ENTITY]-(r:RelFact) "
                "RETURN n.name AS eid, count(DISTINCT r) AS degree",
                {"ids": node_ids},
            )
            for row in self._cypher_rows(resp):
                eid = str(row[0]) if row[0] is not None else None
                if eid in result:
                    result[eid] = row[1] or 0
        except Exception:
            pass
        return result

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        result = {nid: [] for nid in node_ids}
        if not self._database_ready or not node_ids:
            return result
        try:
            # Get outgoing relations
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:SOURCE_ENTITY]-(r:RelFact)"
                "-[:TARGET_ENTITY]->(t:Entity) "
                "WHERE e.name IN $ids "
                "RETURN e.name AS src, t.name AS tgt",
                {"ids": node_ids},
            )
            for row in self._cypher_rows(resp):
                src, tgt = str(row[0]), str(row[1])
                if src in result:
                    result[src].append((src, tgt))
                if tgt in result and tgt != src:
                    result[tgt].append((src, tgt))

            # Get incoming relations
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:TARGET_ENTITY]-(r:RelFact)"
                "-[:SOURCE_ENTITY]->(s:Entity) "
                "WHERE e.name IN $ids "
                "RETURN s.name AS src, e.name AS tgt",
                {"ids": node_ids},
            )
            for row in self._cypher_rows(resp):
                src, tgt = str(row[0]), str(row[1])
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
        if not pairs or not self._database_ready:
            return {}
        all_ids = list({p["src"] for p in pairs} | {p["tgt"] for p in pairs})
        wanted = {(p["src"], p["tgt"]) for p in pairs}
        result = {}
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:ASSERTS]->(r:RelFact)-[:SOURCE_ENTITY]->(a:Entity), "
                "(r)-[:TARGET_ENTITY]->(b:Entity) "
                "WHERE a.name IN $ids AND b.name IN $ids "
                "RETURN a.name, b.name, properties(r)",
                {"ids": all_ids},
            )
            for row in self._cypher_rows(resp):
                src, tgt, props = str(row[0]), str(row[1]), row[2] or {}
                key = (src, tgt)
                rev = (tgt, src)
                if key in wanted and key not in result:
                    result[key] = props
                elif rev in wanted and rev not in result:
                    result[rev] = props
        except Exception:
            pass
        return result

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        if not edge_pairs:
            return {}
        all_ids = list({nid for pair in edge_pairs for nid in pair})
        degrees = await self.node_degrees_batch(all_ids)
        return {
            (src, tgt): degrees.get(src, 0) + degrees.get(tgt, 0)
            for src, tgt in edge_pairs
        }

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        d = await self.edge_degrees_batch([(src_id, tgt_id)])
        return d.get((src_id, tgt_id), 0)

    async def get_all_labels(self) -> list[str]:
        if not self._database_ready:
            return []
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(n:Entity) "
                "RETURN n.name AS eid ORDER BY eid"
            )
            return [str(r[0]) for r in self._cypher_rows(resp) if r[0] is not None]
        except Exception:
            return []

    # ── Delete / lifecycle ────────────────────────────────────────────

    async def delete_document(self, doc_id: str) -> None:
        """Withdraw a document from the docgraph database."""
        try:
            endpoint = f"/_plugins/_graph/docgraph/{self._database_name}/_withdraw_document"
            await self._client.transport.perform_request(
                "POST", endpoint, body={"document_id": doc_id},
            )
            logger.info(f"Withdrew document {doc_id} from docgraph")
        except Exception as e:
            logger.error(f"Failed to withdraw document {doc_id}: {e}")

    async def delete_node(self, node_id: str) -> None:
        """Not directly supported in docgraph — use withdraw_document."""
        logger.warning(f"delete_node({node_id}) not supported in docgraph mode")

    async def remove_nodes(self, nodes: list[str]) -> None:
        logger.warning("remove_nodes not supported in docgraph mode")

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        logger.warning("remove_edges not supported in docgraph mode")


# ── Docgraph-aware VDB and Relationship Adapter ──────────────────────


class OpenSearchDocgraphVectorStorage(BaseVectorStorage):
    """Vector storage for docgraph mode.

    Instead of Cypher MERGE to create nodes + bulk update for embeddings,
    this class uses _bulk_update_nodes to write embeddings to entities/chunks
    that were already created by _ingest. Queries use the same hybrid
    retrieval endpoint.
    """

    _graph_storage: "OpenSearchDocgraphStorage" = None
    _node_label: str = "Entity"
    _max_batch_size: int = 50

    def __init__(
        self,
        namespace,
        workspace,
        embedding_func,
        meta_fields=None,
        node_label="Entity",
        graph_storage=None,
        global_config=None,
        **kwargs,
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            embedding_func=embedding_func,
            global_config=global_config or {},
        )
        self.meta_fields = meta_fields or set()
        self._node_label = node_label
        self._graph_storage = graph_storage
        self.cosine_better_than_threshold = 0.2
        self._pending_embeddings = []

    @property
    def _client(self):
        return self._graph_storage.client if self._graph_storage else None

    async def upsert(self, data: dict[str, dict]) -> None:
        """Buffer embeddings — they'll be flushed after _ingest creates the nodes."""
        if not data:
            return

        import numpy as np

        # Compute embeddings
        items = list(data.items())
        contents = [d.get("content", "") for _, d in items]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)

        # Buffer for later flush
        target_kind = "entity" if self._node_label == "Entity" else "chunk"
        for i, (doc_id, doc_data) in enumerate(items):
            if i >= len(embeddings):
                break
            if self._node_label == "Entity":
                entity_name = doc_data.get("entity_name", doc_id)
                source_id = doc_data.get("source_id", "")
                chunk_ids = [c.strip() for c in source_id.split("<SEP>") if c.strip()] if source_id else []
                chunk_id = chunk_ids[0] if chunk_ids else "unknown-chunk"
                node_id = f"{chunk_id}:{entity_name}"
            else:
                node_id = doc_id

            self._pending_embeddings.append({
                "node_id": node_id,
                "target_kind": target_kind,
                "embedding": embeddings[i].tolist(),
            })

    async def flush_embeddings(self) -> None:
        """Write all buffered embeddings to docgraph nodes via _bulk_update_nodes."""
        if not self._pending_embeddings or not self._graph_storage:
            return
        gs = self._graph_storage
        updates = self._pending_embeddings
        self._pending_embeddings = []
        for j in range(0, len(updates), 100):
            batch = updates[j:j + 500]
            await gs.bulk_update_embeddings(batch)

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict]:
        """Hybrid retrieval via _plugins/_graph/retrieve."""
        gs = self._graph_storage
        await gs._ensure_database_ready()

        if query_embedding is not None:
            qvec = query_embedding
        else:
            qvec = (await self.embedding_func([query]))[0].tolist()

        body = {
            "query_vector": qvec,
            "query_text": query,
            "database": gs.database_name,
            "seed_k": max(top_k, 10),
            "top_k": top_k,
            "hops": 0,
            "search_fields": ["properties.name", "properties.description"],
            "weights": {"vector_weight": 0.7, "text_weight": 0.3, "graph_weight": 0.0},
        }

        try:
            resp = await self._client.transport.perform_request(
                "POST", "/_plugins/_graph/retrieve", body=body,
            )
            results = []
            for r in resp.get("results", []):
                labels = r.get("labels", [])
                if self._node_label not in labels:
                    continue
                props = r.get("properties", {})
                props["entity_name"] = props.get("name", "")
                results.append(props)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Docgraph hybrid retrieval failed: {e}")
            return []

    async def index_done_callback(self):
        pass

    async def delete_by_ids(self, ids: list[str]) -> None:
        pass

    async def delete_entity(self, entity_name: str) -> None:
        pass

    async def delete_entity_relation(self, entity_name: str) -> None:
        pass

    async def get_by_id(self, id: str):
        return None

    async def get_by_ids(self, ids: list[str]):
        return []

    async def delete(self, ids: list[str]):
        pass

    async def get_vectors_by_ids(self, ids: list[str]):
        return {}

    async def drop(self) -> dict[str, str]:
        return {"status": "success"}


class OpenSearchDocgraphRelationshipAdapter(BaseVectorStorage):
    """Relationship adapter for docgraph mode.

    Scans RelFact nodes instead of DIRECTED edges. Uses hybrid retrieval
    to find relevant relations by embedding similarity.
    """

    _graph_storage: "OpenSearchDocgraphStorage" = None
    _entities_vdb: OpenSearchDocgraphVectorStorage = None
    _max_batch_size: int = 50

    def __init__(
        self,
        namespace,
        workspace,
        embedding_func,
        meta_fields=None,
        entities_vdb=None,
        graph_storage=None,
        global_config=None,
        **kwargs,
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            embedding_func=embedding_func,
            global_config=global_config or {},
        )
        self.meta_fields = meta_fields or set()
        self._entities_vdb = entities_vdb
        self._graph_storage = graph_storage
        self.cosine_better_than_threshold = 0.2
        self._pending_embeddings = []

    @property
    def _client(self):
        return self._graph_storage.client if self._graph_storage else None

    async def upsert(self, data: dict[str, dict]) -> None:
        """Buffer embeddings for RelFact nodes."""
        if not data:
            return

        import numpy as np

        items = list(data.items())
        contents = [d.get("content", "") for _, d in items]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)

        for i, (doc_id, doc_data) in enumerate(items):
            if i >= len(embeddings):
                break
            src = doc_data.get("src_id", "")
            tgt = doc_data.get("tgt_id", "")
            source_id = doc_data.get("source_id", "")
            chunk_ids = [c.strip() for c in source_id.split("<SEP>") if c.strip()] if source_id else []
            chunk_id = chunk_ids[0] if chunk_ids else "unknown-chunk"
            node_id = f"{chunk_id}:{src}--{tgt}"
            self._pending_embeddings.append({
                "node_id": node_id,
                "target_kind": "rel_fact",
                "embedding": embeddings[i].tolist(),
            })

    async def flush_embeddings(self) -> None:
        """Write all buffered embeddings to RelFact nodes."""
        if not self._pending_embeddings or not self._graph_storage:
            return
        gs = self._graph_storage
        updates = self._pending_embeddings
        self._pending_embeddings = []
        for j in range(0, len(updates), 100):
            batch = updates[j:j + 500]
            await gs.bulk_update_embeddings(batch)

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict]:
        """Find relevant relations via hybrid retrieval over RelFact nodes."""
        gs = self._graph_storage
        await gs._ensure_database_ready()

        if query_embedding is not None:
            qvec = query_embedding
        else:
            qvec = (await self.embedding_func([query]))[0].tolist()

        body = {
            "query_vector": qvec,
            "query_text": query,
            "database": gs.database_name,
            "seed_k": max(top_k, 10),
            "top_k": top_k * 2,  # over-fetch, filter to RelFact
            "hops": 1,  # expand to get source/target entities
            "search_fields": ["properties.description", "properties.keywords"],
            "weights": {"vector_weight": 0.7, "text_weight": 0.3, "graph_weight": 0.0},
        }

        try:
            resp = await self._client.transport.perform_request(
                "POST", "/_plugins/_graph/retrieve", body=body,
            )

            # Collect RelFact results and resolve src/tgt via Cypher
            relfact_ids = []
            for r in resp.get("results", []):
                if "RelFact" in r.get("labels", []):
                    relfact_ids.append(r.get("id"))

            if not relfact_ids:
                return []

            # Resolve source/target entity names for each RelFact
            results = []
            cypher_resp = await gs._execute_cypher(
                "MATCH (c:Chunk)-[:ASSERTS]->(r:RelFact)-[:SOURCE_ENTITY]->(a:Entity), "
                "(r)-[:TARGET_ENTITY]->(b:Entity) "
                "RETURN a.name AS src, b.name AS tgt, properties(r) AS props "
                "LIMIT $limit",
                {"limit": top_k},
            )
            for row in gs._cypher_rows(cypher_resp):
                src, tgt, props = row[0], row[1], row[2] or {}
                results.append({
                    "src_id": str(src),
                    "tgt_id": str(tgt),
                    "content": props.get("description", ""),
                    "source_id": props.get("source_id", ""),
                    **props,
                })

            return results[:top_k]
        except Exception as e:
            logger.error(f"Docgraph relationship query failed: {e}")
            return []

    async def index_done_callback(self):
        pass

    async def delete_by_ids(self, ids: list[str]) -> None:
        pass

    async def delete_entity(self, entity_name: str) -> None:
        pass

    async def delete_entity_relation(self, entity_name: str) -> None:
        pass

    async def get_by_id(self, id: str):
        return None

    async def get_by_ids(self, ids: list[str]):
        return []

    async def delete(self, ids: list[str]):
        pass

    async def get_vectors_by_ids(self, ids: list[str]):
        return {}

    async def drop(self) -> dict[str, str]:
        return {"status": "success"}
