# Running Demo and Reproduce Scripts

## Prerequisites

```bash
# 1. Set up environment
cd /workplace/xinl/LightRAG
source .env  # or: set -a && source .env && set +a

# 2. Ensure OpenSearch is reachable
curl -sk https://localhost:9200 -u admin:Admin@1234

# 3. Ensure AWS credentials are valid (for Bedrock)
aws sts get-caller-identity --region us-east-1

# 4. If using SSM tunnel to remote cluster
bash ~/workplace/opensearch/docker/tunnel.sh
```

## Demo Script

The demo script inserts Pride & Prejudice (`book.txt`) and runs all 4 query
modes. It drops and rebuilds the graph on each run.

### OpenSearchGraphStorage (existing LPG backend)

```bash
cd /workplace/xinl/LightRAG/examples
set -a && source ../.env && set +a

python lightrag_bedrock_opensearch_graph_demo.py
```

### OpenSearchDocgraphStorage (new docgraph backend)

```bash
cd /workplace/xinl/LightRAG/examples
set -a && source ../.env && set +a
export GRAPH_STORAGE=OpenSearchDocgraphStorage

python lightrag_bedrock_opensearch_graph_demo.py
```

## Reproduce Pipeline

The reproduce pipeline runs the full evaluation: insert documents (Step 1),
generate questions (Step 2), query all modes (Step 3), and evaluate (batch_eval).

### Step 1: Insert Documents

```bash
cd /workplace/xinl/LightRAG/reproduce
set -a && source ../.env && set +a

# LPG backend (default)
python Step_1.py -d agriculture

# Docgraph backend
GRAPH_STORAGE=OpenSearchDocgraphStorage python Step_1.py -d agriculture

# Options
python Step_1.py -d agriculture cs legal    # Multiple domains
python Step_1.py -d agriculture --max-contexts 5  # Limit contexts
```

### Step 2: Generate Questions

```bash
# Same for both backends (no graph interaction)
python Step_2.py -d agriculture
```

### Step 3: Query

```bash
# LPG backend
python Step_3.py -d agriculture -m hybrid

# Docgraph backend
GRAPH_STORAGE=OpenSearchDocgraphStorage python Step_3.py -d agriculture -m hybrid

# All modes
for mode in naive local global hybrid; do
  python Step_3.py -d agriculture -m $mode
done

# Options
python Step_3.py -d agriculture -m local --max-queries 10  # Limit queries
```

### Evaluation

```bash
python batch_eval.py \
  --queries ../datasets/questions/agriculture_questions.txt \
  --result1 agriculture_naive_result.json \
  --result2 agriculture_hybrid_result.json \
  --output agriculture_eval_naive_vs_hybrid.json
```

### Full Pipeline (all steps)

```bash
cd /workplace/xinl/LightRAG/reproduce
set -a && source ../.env && set +a

# --- LPG backend ---
nohup bash -c '
python Step_1.py -d agriculture 2>&1
python Step_2.py -d agriculture 2>&1
for mode in naive local global hybrid; do
  python Step_3.py -d agriculture -m $mode 2>&1
done
Q=../datasets/questions/agriculture_questions.txt
python batch_eval.py --queries "$Q" --result1 agriculture_naive_result.json --result2 agriculture_local_result.json --output agriculture_eval_naive_vs_local.json 2>&1
python batch_eval.py --queries "$Q" --result1 agriculture_naive_result.json --result2 agriculture_global_result.json --output agriculture_eval_naive_vs_global.json 2>&1
python batch_eval.py --queries "$Q" --result1 agriculture_naive_result.json --result2 agriculture_hybrid_result.json --output agriculture_eval_naive_vs_hybrid.json 2>&1
echo "ALL DONE"
' > /tmp/reproduce_full.log 2>&1 &

# --- Docgraph backend ---
export GRAPH_STORAGE=OpenSearchDocgraphStorage
nohup bash -c '
python Step_1.py -d agriculture 2>&1
python Step_2.py -d agriculture 2>&1
for mode in naive local global hybrid; do
  python Step_3.py -d agriculture -m $mode 2>&1
done
echo "ALL DONE"
' > /tmp/reproduce_docgraph.log 2>&1 &
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `us.anthropic.claude-opus-4-6-v1` | Bedrock LLM model |
| `EMBEDDING_MODEL` | `amazon.titan-embed-text-v2:0` | Bedrock embedding model |
| `EMBEDDING_DIM` | `1024` | Embedding dimension |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `OPENSEARCH_HOSTS` | `localhost:9200` | OpenSearch host(s) |
| `OPENSEARCH_USER` | `admin` | OpenSearch username |
| `OPENSEARCH_PASSWORD` | `Admin@1234` | OpenSearch password |
| `OPENSEARCH_USE_SSL` | `true` | Use HTTPS |
| `GRAPH_STORAGE` | `OpenSearchGraphStorage` | Graph backend (`OpenSearchGraphStorage` or `OpenSearchDocgraphStorage`) |
| `ONTOLOGY_FILE` | (bundled) | Path to ontology JSON (docgraph only) |
| `MAX_ASYNC` | `4` | Max concurrent LLM calls |
| `MAX_PARALLEL_INSERT` | `2` | Max concurrent document insertions |

## Monitoring Progress

```bash
# Watch log
tail -f /tmp/reproduce_full.log

# Check milestones
grep "=========" /tmp/reproduce_full.log

# Check for errors
grep -c "ERROR" /tmp/reproduce_full.log

# Check results
cd /workplace/xinl/LightRAG/reproduce && python3 -c "
import json
for mode in ['naive','local','global','hybrid']:
    with open(f'agriculture_{mode}_result.json') as f:
        data = json.load(f)
    nulls = sum(1 for r in data if r['result'] is None)
    avg = sum(len(r['result'] or '') for r in data) / len(data)
    print(f'{mode:8s}: avg={avg:.0f} chars, nulls={nulls}, queries={len(data)}')
"
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `security token expired` | Refresh AWS credentials (`ada credentials update`) |
| `Cannot connect to localhost:9200` | Restart tunnel: `bash ~/workplace/opensearch/docker/tunnel.sh` |
| `circuit_breaking_exception` | Reduce replicas: `curl -XPUT '.../lpg-nodes/_settings' -d '{"number_of_replicas":0}'` and force merge: `curl -XPOST '.../lpg-nodes/_forcemerge?max_num_segments=1'` |
| Step 1 skips documents | Clear doc_status: `curl -XPOST '.../doc_status/_delete_by_query' -d '{"query":{"match_all":{}}}'` |
| Stale cached results | Clear LLM cache: `curl -XPOST '.../llm_response_cache/_delete_by_query' -d '{"query":{"match_all":{}}}'` |
