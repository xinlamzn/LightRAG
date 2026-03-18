# OpenSearch Graph Plugin — Cypher Endpoint Bugs

**Plugin:** opensearch-graph 3.0.0-SNAPSHOT  
**OpenSearch:** 3.5.0  
**Endpoint:** `POST /_plugins/_cypher`

---

## Bug 1: `UNWIND` + `MATCH` returns 0 rows without a subsequent aggregation

### Description

When using `UNWIND` to iterate over a parameter list and `MATCH` nodes by property, the query returns 0 rows — even though the nodes exist and a direct `MATCH` with the same property value succeeds.

Adding an `OPTIONAL MATCH` followed by an aggregation function (e.g., `count()`) causes the query to return the correct results. This suggests the query planner or executor skips materializing the `MATCH` results when there is no downstream clause that forces evaluation.

### Reproduction

```bash
DB="chunk_entity_relation-c65fcac6"

# 1. Confirm the node exists
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (n:Entity {entity_id: \"Agripreneurship\"}) RETURN n.entity_id",
  "database": "'"$DB"'"
}'
# => {"columns":["n.entity_id"],"data":[{"n.entity_id":"Agripreneurship"}]}

# 2. UNWIND + MATCH — returns 0 rows (BUG)
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "UNWIND $ids AS id MATCH (n:Entity {entity_id: id}) RETURN n.entity_id AS eid",
  "database": "'"$DB"'",
  "parameters": {"ids": ["Agripreneurship"]}
}'
# => {"columns":["eid"],"data":[]}

# 3. Adding OPTIONAL MATCH + count() makes it work
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "UNWIND $ids AS id MATCH (n:Entity {entity_id: id}) OPTIONAL MATCH (n)-[r:DIRECTED]-() RETURN n.entity_id AS eid, count(r) AS degree",
  "database": "'"$DB"'",
  "parameters": {"ids": ["Agripreneurship"]}
}'
# => {"columns":["eid","degree"],"data":[{"eid":"Agripreneurship","degree":39}]}
```

### Expected behavior

Query #2 should return `{"columns":["eid"],"data":[{"eid":"Agripreneurship"}]}`.

### Notes

- `UNWIND $ids AS id RETURN id` works correctly (returns the unwound values).
- The issue is specifically in the binding of the `UNWIND` variable to a `MATCH` property filter when there is no aggregation in the `RETURN` clause.
- `MATCH (n:Entity) WHERE n.entity_id IN $ids RETURN n.entity_id` (using `WHERE IN` instead of `UNWIND`) works correctly without aggregation.

---

## Bug 2: `UNWIND` + `MATCH` silently truncates results to 15 rows

### Description

When using `UNWIND` to iterate over a parameter list and `MATCH` nodes, the Cypher endpoint silently truncates results to a maximum of **15 rows**, regardless of how many IDs are provided or how many nodes actually match. No error is returned — the response simply contains fewer rows than expected.

`UNWIND ... RETURN id` (without `MATCH`) correctly returns all rows. `MATCH ... WHERE n.entity_id IN $ids` (without `UNWIND`) also correctly returns all rows. The truncation only occurs when `UNWIND` is combined with `MATCH`.

Adding an explicit `LIMIT` clause makes the problem worse (e.g., `LIMIT 100` returns only 2 rows).

### Reproduction

```bash
DB="chunk_entity_relation-c65fcac6"

# Setup: get 30 entity IDs that exist
IDS=$(curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (n:Entity) RETURN n.entity_id LIMIT 30",
  "database": "'"$DB"'"
}' | jq '[.data[].["n.entity_id"]]')

# 1. UNWIND without MATCH — returns all 30 (OK)
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "UNWIND $ids AS id RETURN id",
  "database": "'"$DB"'",
  "parameters": {"ids": '"$IDS"'}
}'
# => 30 rows

# 2. UNWIND + MATCH — returns only 15 (BUG)
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "UNWIND $ids AS id MATCH (n:Entity {entity_id: id}) RETURN n.entity_id AS eid",
  "database": "'"$DB"'",
  "parameters": {"ids": '"$IDS"'}
}'
# => 15 rows (truncated, no error)

# 3. WHERE IN — returns all 30 (OK)
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (n:Entity) WHERE n.entity_id IN $ids RETURN n.entity_id AS eid",
  "database": "'"$DB"'",
  "parameters": {"ids": '"$IDS"'}
}'
# => 30 rows

# 4. UNWIND + MATCH + LIMIT 100 — returns only 2 (even worse)
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "UNWIND $ids AS id MATCH (n:Entity {entity_id: id}) RETURN n.entity_id AS eid LIMIT 100",
  "database": "'"$DB"'",
  "parameters": {"ids": '"$IDS"'}
}'
# => 2 rows
```

### Expected behavior

All queries should return 30 rows (one per matching entity).

### Scaling behavior

| IDs sent | `UNWIND + MATCH` rows | `WHERE IN` rows |
|----------|----------------------|-----------------|
| 5 | 5 | 5 |
| 10 | 10 | 10 |
| 15 | 15 | 15 |
| 20 | 15 | 20 |
| 30 | 15 | 30 |
| 50 | 15 | 50 |
| 67 | 15 | 67 |

### Notes

- The truncation is silent — no error or warning is returned.
- The 15-row cap applies regardless of what is in the `RETURN` clause (`properties(n)`, individual fields, or just `n.entity_id`).
- Adding `LIMIT N` where N > 15 paradoxically returns fewer rows (e.g., 2).
- Workaround: use `MATCH (n:Entity) WHERE n.entity_id IN $ids` instead of `UNWIND $ids AS id MATCH (n:Entity {entity_id: id})`.

---

## Bug 3: Index scan fails when ANY node has a property value with leading hyphens

### Description

When any node in the graph has a property value starting with hyphens (e.g., `entity_id: "----Shire Militia"`), the graph plugin's internal index scanner fails for ALL queries that require scanning the node index. This includes:

1. `UNWIND $ids AS id MATCH (n:Entity {entity_id: id})` — fails with `For input string: "-"`
2. `MATCH (n:Entity) RETURN n.entity_id LIMIT 1` — fails with `Failed to query the data store while scanning for matching nodes`
3. `MATCH (n:Entity) WHERE n.entity_id IN $ids RETURN ...` — same error
4. `MATCH (n:Entity {entity_id: "X"})-[r:DIRECTED]-(m) RETURN ...` — same error (neighbor scan hits the problematic node)

Only queries that avoid index scanning work: `MATCH (n:Entity {entity_id: "exact_value"}) RETURN ...` (direct lookup) and `MATCH (n:Entity) RETURN count(n)` (count aggregation).

The problematic node poisons the entire index — even after deleting it via `DETACH DELETE`, the tombstone document in the Lucene index continues to trigger the error until a `_forcemerge` + `_flush` purges it. Even then, the fix is fragile and can regress.

### Reproduction

```bash
DB="chunk_entity_relation-c65fcac6"

# Setup: create a node with leading hyphens in entity_id
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "CREATE (n:Entity {entity_id: \"----Bad Name\"})",
  "database": "'"$DB"'"
}'

# 1. Any scan query now fails
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (n:Entity) RETURN n.entity_id LIMIT 1",
  "database": "'"$DB"'"
}'
# => {"error":{"type":"Query execution error","reason":"Failed to query the data store while scanning for matching nodes."}}

# 2. WHERE IN also fails
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (n:Entity) WHERE n.entity_id IN $ids RETURN n.entity_id",
  "parameters": {"ids": ["SomeOtherEntity"]},
  "database": "'"$DB"'"
}'
# => same error

# 3. Edge traversal fails (neighbor scan hits the problematic node)
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (n:Entity {entity_id: \"GoodEntity\"})-[r:DIRECTED]-(m) RETURN m.entity_id LIMIT 1",
  "database": "'"$DB"'"
}'
# => same error

# 4. Direct lookup still works
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (n:Entity {entity_id: \"GoodEntity\"}) RETURN n.entity_id",
  "database": "'"$DB"'"
}'
# => works

# 5. Count aggregation still works
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (n:Entity) RETURN count(n)",
  "database": "'"$DB"'"
}'
# => works
```

### Expected behavior

All queries should succeed regardless of what property values exist in the graph. The hyphen character should be treated as a regular string character, not parsed as a numeric sign.

### Key observations

- A single node with leading hyphens in any property value poisons ALL index scan queries across the entire graph.
- The error persists even after deleting the problematic node — Lucene tombstones still trigger the scanner bug.
- `_forcemerge` + `_flush` can temporarily fix it by purging tombstones, but the fix is fragile.
- Direct property lookups (`{entity_id: "value"}`) and count aggregations bypass the scanner and work correctly.
- This is a critical data integrity issue: one bad entity name from LLM extraction can break the entire graph.

### Workaround

Strip leading hyphens from entity names before storing them in the graph. In `upsert_node()` and `upsert_edge()`, sanitize IDs with `node_id.lstrip("-").strip()`.

---

## Bug 4 (Performance): Edge `MERGE` is O(E) — scans all edges of the given type

### Description

Cypher `MERGE (s)-[r:TYPE]->(t)` between two already-matched nodes performs a full scan of ALL edges of type `TYPE` in the graph, rather than only checking edges between `s` and `t`. This makes edge MERGE O(E) where E is the total number of edges of that type, causing 15-30 second latencies with just 12K-18K edges.

Node `MERGE` is fast (~0.2s) because it uses the property index. Edge `MERGE` does not benefit from endpoint filtering.

### Reproduction

```bash
DB="chunk_entity_relation-c65fcac6"
# Assumes ~18K DIRECTED edges exist

# 1. MERGE edge — ~17s
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (s:Entity {entity_id: $src}), (t:Entity {entity_id: $tgt}) MERGE (s)-[r:DIRECTED]->(t) ON CREATE SET r += $props ON MATCH SET r += $props",
  "database": "'"$DB"'",
  "parameters": {"src": "Pollen", "tgt": "Colony", "props": {"description": "test"}}
}'
# => 17 seconds

# 2. MATCH edge + SET — ~0.17s (100x faster, same result)
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (s:Entity {entity_id: $src})-[r:DIRECTED]->(t:Entity {entity_id: $tgt}) SET r += $props",
  "database": "'"$DB"'",
  "parameters": {"src": "Pollen", "tgt": "Colony", "props": {"description": "test"}}
}'
# => 0.17 seconds

# 3. CREATE edge — ~0.09s
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (s:Entity {entity_id: $src}), (t:Entity {entity_id: $tgt}) CREATE (s)-[r:DIRECTED]->(t) SET r += $props",
  "database": "'"$DB"'",
  "parameters": {"src": "Pollen", "tgt": "Colony", "props": {"description": "test"}}
}'
# => 0.09 seconds
```

### Measurements (18K DIRECTED edges, 12K MENTIONED_IN edges)

| Operation | Time |
|---|---|
| `MERGE (s)-[r:DIRECTED]->(t)` | 17s |
| `MERGE (e)-[:MENTIONED_IN]->(c)` | 28s |
| `MATCH (s)-[r:DIRECTED]->(t) SET r += $props` | 0.17s |
| `MATCH (s), (t) CREATE (s)-[r:DIRECTED]->(t) SET r += $props` | 0.09s |
| `MATCH (s)-[r:DIRECTED]->(t) RETURN count(r)` | 0.28s |

### Expected behavior

Edge `MERGE` should only check edges between the two matched endpoint nodes, not scan all edges of that type. Expected time should be comparable to MATCH + SET (~0.2s).

### Workaround

Replace edge `MERGE` with a two-step check-then-create-or-update pattern:
1. `MATCH (s)-[r:TYPE]->(t) RETURN count(r) AS cnt` — check existence
2. If exists: `MATCH (s)-[r:TYPE]->(t) SET r += $props`
3. If not: `MATCH (s), (t) CREATE (s)-[r:TYPE]->(t) SET r += $props`

This reduces edge upsert from 17-28s to 0.12-0.16s (100x improvement).

---

## Bug 5: `CREATE` edge + `SET` in the same statement silently drops the `SET`

### Description

When creating an edge and setting its properties in a single Cypher statement (`CREATE (s)-[r:TYPE]->(t) SET r += $props`), the properties appear in the `RETURN` clause of the same query but are **never persisted** to the underlying index. Querying the edge afterward shows `"properties": {}`.

Splitting the operation into two separate Cypher calls — `CREATE` first, then `MATCH ... SET` — persists the properties correctly.

### Reproduction

```bash
DB="chunk_entity_relation-c65fcac6"

# Setup: ensure two nodes exist
# MATCH (a:Entity {entity_id: "TestNodeA"}), (b:Entity {entity_id: "TestNodeB"})

# 1. CREATE + SET in one statement — properties appear in RETURN
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (s:Entity {entity_id: $src}), (t:Entity {entity_id: $tgt}) CREATE (s)-[r:DIRECTED]->(t) SET r += $props RETURN r.weight, r.description",
  "database": "'"$DB"'",
  "parameters": {"src": "TestNodeA", "tgt": "TestNodeB", "props": {"weight": 5.0, "description": "test edge"}}
}'
# => {"columns":["r.weight","r.description"],"data":[{"r.weight":5.0,"r.description":"test edge"}]}
# Looks correct, but...

# 2. Query the edge — properties are EMPTY
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (s:Entity {entity_id: $src})-[r:DIRECTED]->(t:Entity {entity_id: $tgt}) RETURN r.weight, r.description, properties(r)",
  "database": "'"$DB"'",
  "parameters": {"src": "TestNodeA", "tgt": "TestNodeB"}
}'
# => {"columns":["r.weight","r.description","properties(r)"],"data":[{"r.weight":null,"r.description":null,"properties(r)":{}}]}

# 3. Workaround: CREATE first, then SET in a separate call
# Step A: CREATE
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (s:Entity {entity_id: $src}), (t:Entity {entity_id: $tgt}) CREATE (s)-[r:DIRECTED]->(t)",
  "database": "'"$DB"'",
  "parameters": {"src": "TestNodeC", "tgt": "TestNodeD"}
}'

# Step B: SET
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (s:Entity {entity_id: $src})-[r:DIRECTED]->(t:Entity {entity_id: $tgt}) SET r += $props",
  "database": "'"$DB"'",
  "parameters": {"src": "TestNodeC", "tgt": "TestNodeD", "props": {"weight": 5.0, "description": "test edge"}}
}'

# Step C: Verify — properties are persisted
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (s:Entity {entity_id: $src})-[r:DIRECTED]->(t:Entity {entity_id: $tgt}) RETURN r.weight, r.description",
  "database": "'"$DB"'",
  "parameters": {"src": "TestNodeC", "tgt": "TestNodeD"}
}'
# => {"columns":["r.weight","r.description"],"data":[{"r.weight":5.0,"r.description":"test edge"}]}
```

### Expected behavior

`CREATE (s)-[r:TYPE]->(t) SET r += $props` should persist the properties to the edge. The RETURN in the same statement should reflect what is actually stored.

### Notes

- `MATCH ... SET r += $props` on an existing edge works correctly and persists.
- Only the `CREATE ... SET` combination on edges fails to persist. The `SET` is silently dropped.
- The RETURN clause in the same statement misleadingly shows the properties as if they were set.
- This bug caused all ~4300 DIRECTED edges in a LightRAG knowledge graph to have empty properties (no weight, description, keywords, or source_id), producing 304 "missing weight" warnings during queries.

### Workaround

Split edge creation and property setting into two separate Cypher calls.

---

## Bug 6 (Performance): `UNWIND` + `MATCH` on edges produces a cross-join that hits a 1M row limit

### Description

`UNWIND $ids AS id MATCH (n:Entity {entity_id: id})-[r:DIRECTED]-()` internally produces a cross-join of `len(ids) × total_edges_of_type` before filtering. When this exceeds 1,000,000 rows, the query fails with:

```
{"error":{"type":"Query execution error","reason":"Cross join would produce N rows, exceeding maximum allowed size of 1000000"}}
```

This means the maximum number of IDs that can be used in a single UNWIND + edge MATCH query is `floor(1,000,000 / total_edges)`. With 5,018 DIRECTED edges (counted in both directions = ~10,036), the limit is ~99 IDs.

`MATCH ... WHERE n.entity_id IN $ids` does NOT have this cross-join behavior and works with any number of IDs.

### Reproduction

```bash
DB="chunk_entity_relation-c65fcac6"
# Assumes ~5,018 DIRECTED edges exist (10,036 bidirectional)

# 1. Get 200 entity IDs
IDS=$(curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (n:Entity) RETURN n.entity_id LIMIT 200",
  "database": "'"$DB"'"
}' | jq '[.data[].["n.entity_id"]]')

# 2. UNWIND + MATCH on edges — FAILS (200 × 10,036 = 2,007,200 > 1,000,000)
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "UNWIND $ids AS id MATCH (n:Entity {entity_id: id})-[r:DIRECTED]-() RETURN n.entity_id AS eid, count(r) AS degree",
  "database": "'"$DB"'",
  "parameters": {"ids": '"$IDS"'}
}'
# => {"error":{"type":"Query execution error","reason":"Cross join would produce 2007200 rows, exceeding maximum allowed size of 1000000"}}

# 3. WHERE IN — works with any number of IDs
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "MATCH (n:Entity)-[r:DIRECTED]-() WHERE n.entity_id IN $ids RETURN n.entity_id AS eid, count(r) AS degree",
  "database": "'"$DB"'",
  "parameters": {"ids": '"$IDS"'}
}'
# => Success: 176 results (in ~10s)

# 4. UNWIND with 50 IDs — works (50 × 10,036 = 501,800 < 1,000,000)
IDS50=$(echo "$IDS" | jq '.[0:50]')
curl -X POST '/_plugins/_cypher' -H 'Content-Type: application/json' -d '{
  "query": "UNWIND $ids AS id MATCH (n:Entity {entity_id: id})-[r:DIRECTED]-() RETURN n.entity_id AS eid, count(r) AS degree",
  "database": "'"$DB"'",
  "parameters": {"ids": '"$IDS50"'}
}'
# => Success: 49 results (in ~7s)
```

### Scaling behavior

| IDs | Cross-join size (×10,036) | Result |
|-----|--------------------------|--------|
| 50 | 501,800 | OK (~7s) |
| 80 | 802,880 | OK (~8s) |
| 100 | 1,003,600 | FAIL |
| 200 | 2,007,200 | FAIL |

### Expected behavior

`UNWIND` + `MATCH` on edges should filter by the unwound ID first, then traverse edges — not produce a full cross-join. The query should work with any number of IDs, similar to `WHERE IN`.

### Notes

- `UNWIND` + `MATCH` on nodes only (no edge traversal) also produces a cross-join (`len(ids) × total_nodes`), but this is less likely to hit the 1M limit since node counts are typically smaller than edge counts.
- The cross-join limit appears to be a hardcoded 1,000,000 in the graph plugin.
- As the graph grows, the maximum safe UNWIND batch size shrinks proportionally.

### Workaround

Use `MATCH (n:Entity)-[r:DIRECTED]-() WHERE n.entity_id IN $ids` instead of `UNWIND $ids AS id MATCH (n:Entity {entity_id: id})-[r:DIRECTED]-()`.
