"""
LightRAG Demo with OpenSearch Graph Plugin + Amazon Bedrock

This example demonstrates how to use LightRAG with:
- Amazon Bedrock (Claude Opus 4.6 for LLM + Titan Embedding v2 for embeddings)
- OpenSearch-backed storages for:
  - KV storage (OpenSearch indices)
  - Graph storage (OpenSearch Graph Plugin with Cypher queries)
  - Document status storage (OpenSearch indices)

Vector storage is provided implicitly by the graph plugin -- entity and
chunk embeddings are stored directly on graph nodes, and relationship
lookups go through the graph edges.  There is no need to set
``vector_storage`` explicitly; LightRAG detects ``OpenSearchGraphStorage``
and wires up ``OpenSearchGraphVectorStorage`` / ``OpenSearchGraphRelationshipAdapter``
automatically.

Prerequisites:
1. OpenSearch cluster running and accessible (3.x or higher with the
   Graph plugin and k-NN plugin enabled)
2. Required indices and the graph database will be auto-created by LightRAG
3. AWS credentials configured (via env vars, ~/.aws/credentials, or IAM role)
   with access to Bedrock in your chosen region
4. Set environment variables (example .env):

   # OpenSearch
   OPENSEARCH_HOSTS=localhost:9200
   OPENSEARCH_USER=admin
   OPENSEARCH_PASSWORD=your-password
   OPENSEARCH_USE_SSL=false
   OPENSEARCH_VERIFY_CERTS=false

   # AWS (optional if using IAM role or ~/.aws/credentials)
   AWS_ACCESS_KEY_ID=your-access-key
   AWS_SECRET_ACCESS_KEY=your-secret-key
   AWS_REGION=us-west-2

5. Prepare a text file to index (default: ./book.txt)

Usage:
    python examples/lightrag_bedrock_opensearch_graph_demo.py
"""

import os
import asyncio

from lightrag import LightRAG, QueryParam
from lightrag.llm.bedrock import bedrock_complete, bedrock_embed
from lightrag.utils import setup_logger, EmbeddingFunc


# --------------------------------------------------
# Logger
# --------------------------------------------------
setup_logger("lightrag", level="INFO")


# --------------------------------------------------
# Config
# --------------------------------------------------
WORKING_DIR = "./opensearch_bedrock_rag_storage"
BOOK_FILE = "./book.txt"

# Bedrock model IDs
LLM_MODEL = os.environ.get("BEDROCK_LLM_MODEL", "us.anthropic.claude-opus-4-6-v1")
EMBEDDING_MODEL = os.environ.get("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")

# Titan Embed v2 produces 1024-dimensional vectors
EMBEDDING_DIM = int(os.environ.get("BEDROCK_EMBEDDING_DIM", "1024"))
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("BEDROCK_EMBEDDING_MAX_TOKEN_SIZE", "8192"))
GRAPH_STORAGE = os.environ.get("GRAPH_STORAGE", "OpenSearchGraphStorage")

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# --------------------------------------------------
# Initialize RAG with OpenSearch storages + Bedrock
# --------------------------------------------------
async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        workspace="demo",
        # Bedrock LLM — llm_model_name is passed as the Bedrock model ID
        llm_model_func=bedrock_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={"max_tokens": 4096},
        # Bedrock Titan Embed v2
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
            func=bedrock_embed,
        ),
        # OpenSearch-backed storages
        kv_storage="OpenSearchKVStorage",
        doc_status_storage="OpenSearchDocStatusStorage",
        graph_storage=GRAPH_STORAGE,
        # vector_storage is intentionally omitted -- OpenSearchGraphStorage
        # provides implicit vector storage via the graph plugin (embeddings
        # are stored on graph nodes/edges, not in separate k-NN indices).
    )

    # REQUIRED: initialize all storage backends
    await rag.initialize_storages()

    # Clean previous data so the example is re-runnable.
    # Drop order matters: graph database first (this also removes the
    # entity/chunk/relationship vector data stored on graph nodes/edges),
    # then KV indices and doc status.
    await rag.chunk_entity_relation_graph.drop()
    for storage in [
        rag.full_docs,
        rag.text_chunks,
        rag.full_entities,
        rag.full_relations,
        rag.entity_chunks,
        rag.relation_chunks,
        rag.llm_response_cache,
        rag.doc_status,
    ]:
        await storage.drop()
    print("Cleared previous data.")

    return rag


# --------------------------------------------------
# Main
# --------------------------------------------------
async def main():
    rag = None
    try:
        print("Initializing LightRAG with OpenSearch + Bedrock...")
        print(f"  LLM:       {LLM_MODEL}")
        print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
        print(f"  Graph:     {GRAPH_STORAGE}")
        rag = await initialize_rag()

        if not os.path.exists(BOOK_FILE):
            raise FileNotFoundError(
                f"'{BOOK_FILE}' not found. Please provide a text file to index."
            )

        print(f"\nReading document: {BOOK_FILE}")
        with open(BOOK_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"Loaded document ({len(content)} characters)")

        print("\nInserting document into LightRAG (this may take some time)...")
        await rag.ainsert(content)
        print("Document indexed successfully!")

        print("\n" + "=" * 60)
        print("Running sample queries")
        print("=" * 60)

        query = "What are the top themes in this document?"

        for mode in ["naive", "local", "global", "hybrid"]:
            print(f"\n[{mode.upper()} MODE]")
            result = await rag.aquery(query, param=QueryParam(mode=mode))
            print(result)

        print("\nRAG system is ready for use!")

    except Exception as e:
        print("An error occurred:", e)
        import traceback

        traceback.print_exc()

    finally:
        if rag is not None:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
