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

import logging
import os
from collections import defaultdict

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
        # Per-document buffer: {doc_id: {chunk_id: {entities: [], relations: []}}}
        self._doc_buffer = defaultdict(lambda: {"properties": {}, "chunks": defaultdict(lambda: {"entities": [], "relations": []})})
        self._ontology_loaded = False

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

    # ── Ingestion (buffered → batch _ingest) ──────────────────────────

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Buffer entity for batch ingestion instead of individual Cypher MERGE."""
        chunk_ids = []
        source_id = node_data.get("source_id", "")
        if source_id:
            chunk_ids = [c.strip() for c in source_id.split("<SEP>") if c.strip()]

        # Determine which document/chunk this entity belongs to
        # Use the first chunk as the primary chunk for this entity
        chunk_id = chunk_ids[0] if chunk_ids else "unknown-chunk"

        # Infer doc_id from the global config or use a default
        doc_id = node_data.get("file_path", "unknown_source")

        entity_type = node_data.get("entity_type", "")
        ontology_id = map_entity_type(entity_type)

        # Build entity labels
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
        doc_id = edge_data.get("file_path", "unknown_source")

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
        if doc_id not in self._doc_buffer:
            return

        buf = self._doc_buffer[doc_id]
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

            chunks_payload.append({
                "chunk_id": chunk_id,
                "properties": {"text": chunk_id},  # chunk text stored elsewhere
                "entities": list(seen_entities.values()),
                "relations": list(seen_relations.values()),
            })

        if not chunks_payload:
            del self._doc_buffer[doc_id]
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
            logger.error(f"Docgraph ingest failed for {doc_id}: {e}")
            raise
        finally:
            del self._doc_buffer[doc_id]

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
