"""
Step 1: Insert unique contexts into LightRAG using Bedrock (Claude Opus 4.6 + Titan Embed v2).

Uses the OpenSearch graph plugin for storage (graph + implicit vector storage).

Usage:
    python Step_1.py                          # Insert agriculture (default)
    python Step_1.py -d cs                    # Insert a specific domain
    python Step_1.py -d agriculture cs legal  # Insert multiple domains

Environment variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    OPENSEARCH_HOSTS, OPENSEARCH_USER, OPENSEARCH_PASSWORD
    BEDROCK_LLM_MODEL       (default: us.anthropic.claude-opus-4-6-v1)
    BEDROCK_EMBEDDING_MODEL  (default: amazon.titan-embed-text-v2:0)
    BEDROCK_EMBEDDING_DIM    (default: 1024)
"""

import os
import json
import time
import asyncio
import argparse

from lightrag import LightRAG
from lightrag.llm.bedrock import bedrock_complete, bedrock_embed
from lightrag.utils import EmbeddingFunc, setup_logger

setup_logger("lightrag", level="INFO")

# Bedrock model config
LLM_MODEL = os.environ.get("BEDROCK_LLM_MODEL", "us.anthropic.claude-opus-4-6-v1")
EMBEDDING_MODEL = os.environ.get(
    "BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"
)
EMBEDDING_DIM = int(os.environ.get("BEDROCK_EMBEDDING_DIM", "1024"))
EMBEDDING_MAX_TOKEN_SIZE = int(
    os.environ.get("BEDROCK_EMBEDDING_MAX_TOKEN_SIZE", "8192")
)


def insert_text(rag, file_path):
    with open(file_path, mode="r") as f:
        unique_contexts = json.load(f)

    print(f"Loaded {len(unique_contexts)} unique contexts from {file_path}")

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            print("Insertion complete.")
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")


async def initialize_rag(working_dir):
    os.makedirs(working_dir, exist_ok=True)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=bedrock_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={"max_tokens": 4096},
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
            func=bedrock_embed,
        ),
        # OpenSearch-backed storages
        kv_storage="OpenSearchKVStorage",
        doc_status_storage="OpenSearchDocStatusStorage",
        graph_storage="OpenSearchGraphStorage",
    )

    await rag.initialize_storages()
    return rag


def main():
    parser = argparse.ArgumentParser(description="Insert contexts into LightRAG")
    parser.add_argument(
        "-d",
        "--domains",
        nargs="+",
        default=["agriculture"],
        help="Domains to insert (default: agriculture)",
    )
    args = parser.parse_args()

    for cls in args.domains:
        print(f"\n{'='*60}")
        print(f"Processing domain: {cls}")
        print(f"  LLM:       {LLM_MODEL}")
        print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
        print(f"{'='*60}")

        working_dir = f"../{cls}"
        rag = asyncio.run(initialize_rag(working_dir))

        input_file = f"../datasets/unique_contexts/{cls}_unique_contexts.json"
        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            print("Run Step_0.py first to download and extract contexts.")
            continue

        insert_text(rag, input_file)


if __name__ == "__main__":
    main()
