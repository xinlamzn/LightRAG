# LightRAG Docgraph Integration Plan

## Goal

Add a new `OpenSearchDocgraphStorage` backend that uses the docgraph REST API
and ontology grounding, while keeping the existing `OpenSearchGraphStorage`
(Cypher/LPG) fully functional.

## Background

The OpenSearch graph plugin now offers a **docgraph mode** with:
- Atomic document ingestion (`_ingest` endpoint)
- Evidence-mediated graph model (Document → Chunk → Entity / RelFact)
- Ontology grounding (entities and relations linked to ontology concepts)
- Built-in provenance tracking and access control
- Automatic evidence rewriter for Cypher queries

See `docs/design/docgraph-mode.md` for the full docgraph specification.

### Current LPG mode issues resolved by docgraph

| Issue | LPG Workaround | Docgraph Solution |
|-------|---------------|-------------------|
| Promoted fields return null via `n.field` | Use `entity_name` instead of `entity_id` | Docgraph manages its own property model |
| Edge MERGE is O(E) | Check-then-CREATE-or-SET | `_ingest` handles atomically |
| CREATE + SET silently drops SET | Split into two Cypher calls | Not applicable (REST API) |
| Leading-hyphen values poison index | Strip hyphens in application | `_ingest` handles sanitization |
| Individual upserts are slow | Batch workarounds | Single `_ingest` call per document |

## Design Principles

1. **Zero changes to existing LPG storage** — `OpenSearchGraphStorage` remains
   untouched. Users select backend via config.
2. **Inheritance over duplication** — `OpenSearchDocgraphStorage` extends
   `OpenSearchGraphStorage`, overriding only what differs.
3. **Evidence rewriter first** — rely on the docgraph engine's automatic
   `DocgraphEvidenceRewriter` for read queries. Only add explicit mediated
   patterns where the rewriter fails.
4. **Ontology mapping at ingestion time** — map LightRAG's free-form
   `entity_type` to ontology concept IDs during ingestion, not extraction.

## Data Model Mapping

### LPG → Docgraph

| LPG Concept | Docgraph Equivalent |
|-------------|-------------------|
| Entity node | Entity node (compound ID: `{chunk_id}:{entity_id}`) |
| DIRECTED edge | RelFact node + SOURCE_ENTITY / TARGET_ENTITY edges |
| Chunk node | Chunk node (with HAS_CHUNK from Document) |
| — | Document node (new, tracks provenance) |
| — | OntologyConcept / OntologyRelation nodes |

### Ontology

Uses `docs/ontologies/generic-knowledge-graph.json`:
- 11 entity concepts: Person, Organization, Location, Event, Concept, Artifact,
  Method, NaturalObject, Creature, Content, Data
- 15 relation types: RELATED_TO, WORKS_AT, LOCATED_IN, PART_OF, CREATED_BY,
  USES, KNOWS, PARTICIPATED_IN, OCCURRED_AT, STUDIES, PRODUCES, MEASURES,
  DERIVED_FROM, IMPACTS, SUPERSEDES

## Implementation Steps

### Step 1: Test evidence rewriter (prerequisite)

Before writing code, validate that the docgraph engine's automatic evidence
rewriter handles LightRAG's critical Cypher patterns. Test each pattern against
a docgraph database with sample data.

**Patterns to test:**

| Pattern | Used by |
|---------|---------|
| `MATCH (n:Entity {entity_id: $id}) RETURN properties(n)` | `get_node()` |
| `MATCH (n:Entity) WHERE n.entity_name IN $ids RETURN ...` | batch lookups |
| `MATCH (a:Entity)-[r:DIRECTED]->(b:Entity) WHERE ...` | edge queries |
| `MATCH (n:Entity {entity_id: $id})-[r:DIRECTED]-() RETURN count(r)` | `node_degree()` |
| `MATCH (n:Entity) RETURN count(n)` | `node_count()` |
| `MATCH (n:Entity) RETURN n.entity_name ORDER BY eid` | `get_all_labels()` |

**Outcome:** A list of patterns that work as-is vs patterns that need explicit
overrides with mediated queries.

**Note:** `DIRECTED` edges do not exist in docgraph — relations are stored as
`RelFact` nodes. Any query traversing `DIRECTED` edges will need to be
rewritten to the RelFact pattern regardless of the evidence rewriter.

### Step 2: Ontology mapping module

New file: `lightrag/kg/ontology_mapping.py`

```python
ENTITY_TYPE_TO_ONTOLOGY = {
    "person": "onto:Person",
    "organization": "onto:Organization",
    "location": "onto:Location",
    "event": "onto:Event",
    "concept": "onto:Concept",
    "artifact": "onto:Artifact",
    "method": "onto:Method",
    "naturalobject": "onto:NaturalObject",
    "creature": "onto:Creature",
    "content": "onto:Content",
    "data": "onto:Data",
}

RELATION_TYPE_TO_ONTOLOGY = {
    "works_at": "onto:WORKS_AT",
    "located_in": "onto:LOCATED_IN",
    "part_of": "onto:PART_OF",
    "created_by": "onto:CREATED_BY",
    "uses": "onto:USES",
    "knows": "onto:KNOWS",
    "participated_in": "onto:PARTICIPATED_IN",
    "occurred_at": "onto:OCCURRED_AT",
    "studies": "onto:STUDIES",
    "produces": "onto:PRODUCES",
    "measures": "onto:MEASURES",
    "derived_from": "onto:DERIVED_FROM",
    "impacts": "onto:IMPACTS",
    "supersedes": "onto:SUPERSEDES",
}
# Fallback: "onto:RELATED_TO"
```

Functions:
- `map_entity_type(entity_type: str) -> str` — returns ontology_id
- `map_relation_type(relation_type: str) -> str` — returns ontology_id with
  fuzzy matching and `onto:RELATED_TO` fallback
- `load_ontology(path: str) -> dict` — parse the ontology JSON

### Step 3: `OpenSearchDocgraphStorage` class

Add to `lightrag/kg/opensearch_impl.py` (or new file
`lightrag/kg/opensearch_docgraph_impl.py`).

Extends `OpenSearchGraphStorage`. Overrides:

#### Database lifecycle

| Method | Override behavior |
|--------|-----------------|
| `_create_database_if_not_exists()` | PUT with `"mode": "docgraph"` |
| `initialize()` | Call parent, then `_load_ontology()` |

#### Ingestion (buffered → batch)

The docgraph `_ingest` endpoint expects all entities and relations for a
document in a single request. LightRAG calls `upsert_node()` and
`upsert_edge()` individually per entity/relation. The docgraph class bridges
this gap with a buffer.

| Method | Override behavior |
|--------|-----------------|
| `upsert_node(node_id, node_data)` | Buffer entity into `_doc_buffer[doc_id][chunk_id]` |
| `upsert_edge(src, tgt, edge_data)` | Buffer relation into `_doc_buffer[doc_id][chunk_id]` |
| `flush_document(doc_id)` | **New.** Build `_ingest` payload from buffer, POST to docgraph, clear buffer |

The buffer structure:
```python
_doc_buffer = {
    "doc-001": {
        "doc_properties": {...},
        "chunks": {
            "chunk-001": {
                "properties": {"text": "..."},
                "entities": [...],
                "relations": [...]
            }
        }
    }
}
```

`flush_document()` builds the `_ingest` JSON:
- Maps each entity's `entity_type` → `ontology_id` via the mapping module
- Maps each relation's keywords → `ontology_id` via fuzzy matching
- Creates compound entity IDs: `{chunk_id}:{entity_id}`
- POSTs to `_plugins/_graph/docgraph/{db}/_ingest`

#### Read queries

Inherit all read methods from parent. Override only those where:
1. The evidence rewriter doesn't handle the pattern (determined in Step 1)
2. The query uses `DIRECTED` edges (must be rewritten to RelFact pattern)

**Guaranteed overrides** (DIRECTED → RelFact):

| Method | Rewrite |
|--------|---------|
| `get_edge(src, tgt)` | Query via `(r:RelFact)-[:SOURCE_ENTITY]->(a), (r)-[:TARGET_ENTITY]->(b)` |
| `get_edges_batch(pairs)` | Same pattern, batched |
| `get_node_edges(node_id)` | `(c:Chunk)-[:MENTIONS]->(e:Entity)<-[:SOURCE_ENTITY]-(r:RelFact)` |
| `edge_degree()` / `edge_degrees_batch()` | Count RelFacts instead of DIRECTED edges |
| `node_degree()` / `node_degrees_batch()` | Count MENTIONS edges from chunks |

**Conditional overrides** (depends on Step 1 results):

| Method | Override if rewriter fails |
|--------|--------------------------|
| `get_node(node_id)` | Add `(c:Chunk)-[:MENTIONS]->` prefix |
| `has_node(node_id)` | Same |
| `get_all_labels()` | Same |
| `node_count()` | Same |

#### Document lifecycle

| Method | Behavior |
|--------|---------|
| `withdraw_document(doc_id)` | **New.** POST to `_withdraw_document` |

### Step 4: Adapt `merge_nodes_and_edges()` in `operate.py`

Add a docgraph code path gated by storage type check:

```python
if hasattr(knowledge_graph_inst, 'flush_document'):
    # Docgraph path: buffer entities/edges, then flush
    ...
else:
    # Existing LPG path: unchanged
    ...
```

Changes in the docgraph path:
- Entity phase: call `upsert_node()` as before (buffered by override)
- Edge phase: call `upsert_edge()` as before (buffered by override)
- After Phase 2: call `knowledge_graph_inst.flush_document(doc_id)`

This minimizes changes to `operate.py` — the buffering is transparent.

### Step 5: Register storage type

In `lightrag/lightrag.py`, add to the storage registry:

```python
"OpenSearchDocgraphStorage": "lightrag.kg.opensearch_impl.OpenSearchDocgraphStorage"
```

### Step 6: Configuration and scripts

- Add `ONTOLOGY_FILE` env var (default: bundled JSON)
- Copy `generic-knowledge-graph.json` into `lightrag/data/`
- Update demo script with docgraph variant
- Update reproduce scripts to accept `--graph-storage` flag

## File Change Summary

| File | Change type | Est. lines |
|------|------------|-----------|
| `lightrag/kg/opensearch_impl.py` | Add `OpenSearchDocgraphStorage` class | ~250 new |
| `lightrag/kg/ontology_mapping.py` | New file | ~50 |
| `lightrag/data/generic-knowledge-graph.json` | Bundled ontology | copy |
| `lightrag/operate.py` | Add docgraph flush call | ~15 changed |
| `lightrag/lightrag.py` | Register storage type | ~3 |
| `examples/` | Docgraph demo variant | ~15 |

**Total: ~330 new lines, ~18 changed lines. Zero changes to existing LPG code.**

## Execution Order

1. **Step 1** — Test evidence rewriter (determines scope of Step 3 overrides)
2. **Step 2** — Ontology mapping module
3. **Step 3** — `OpenSearchDocgraphStorage` class
4. **Step 4** — `operate.py` adaptation
5. **Step 5–6** — Wiring, config, scripts
6. **Validation** — Run reproduce pipeline with docgraph backend, compare
   results against LPG baseline

## Risks

| Risk | Mitigation |
|------|-----------|
| Evidence rewriter doesn't handle all patterns | Step 1 identifies gaps early; add explicit overrides |
| Ontology mapping is lossy (free-form → 11 types) | `onto:Concept` as catch-all; RELATED_TO for unknown relations |
| Compound entity IDs break deduplication | Same entity in different chunks = different nodes (by design in docgraph) |
| Performance regression from mediated queries | Benchmark in Step 6; hybrid_retrieve may be faster than individual Cypher |
| Buffer memory for large documents | Flush per-chunk if document exceeds threshold |

## Success Criteria

1. Existing `OpenSearchGraphStorage` passes all current tests unchanged
2. `OpenSearchDocgraphStorage` completes the agriculture reproduce pipeline
3. All 4 query modes (naive, local, global, hybrid) return non-null results
4. Ontology grounding visible in graph (entities linked to OntologyConcept)
5. Evaluation scores comparable to LPG baseline
