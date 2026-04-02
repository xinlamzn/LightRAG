"""
Step 3: Query LightRAG using Bedrock (Claude Opus 4.6 + Titan Embed v2).

Runs all generated questions against the indexed LightRAG and saves results.

Usage:
    python Step_3.py                              # Query agriculture, hybrid mode
    python Step_3.py -d cs                        # Query a specific domain
    python Step_3.py -d agriculture -m local      # Use a specific query mode
    python Step_3.py -d agriculture cs legal -m hybrid

Environment variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    OPENSEARCH_HOSTS, OPENSEARCH_USER, OPENSEARCH_PASSWORD
    BEDROCK_LLM_MODEL       (default: us.anthropic.claude-opus-4-6-v1)
    BEDROCK_EMBEDDING_MODEL  (default: amazon.titan-embed-text-v2:0)
    BEDROCK_EMBEDDING_DIM    (default: 1024)
"""

import os
import re
import json
import asyncio
import argparse

from lightrag import LightRAG, QueryParam
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
GRAPH_STORAGE = os.environ.get("GRAPH_STORAGE", "OpenSearchGraphStorage")
    os.environ.get("BEDROCK_EMBEDDING_MAX_TOKEN_SIZE", "8192")
)


def extract_queries(file_path):
    with open(file_path, "r") as f:
        data = f.read()

    data = data.replace("**", "")
    queries = re.findall(r"- Question \d+: (.+)", data)
    return queries


async def process_query(query_text, rag_instance, query_param):
    try:
        result = await rag_instance.aquery(query_text, param=query_param)
        return {"query": query_text, "result": result}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}


async def run_queries_and_save_to_json(
    queries, rag_instance, query_param, output_file, error_file
):
    with (
        open(output_file, "w", encoding="utf-8") as result_file,
        open(error_file, "w", encoding="utf-8") as err_file,
    ):
        result_file.write("[\n")
        first_entry = True

        for i, query_text in enumerate(queries):
            print(f"  Query {i+1}/{len(queries)}: {query_text[:80]}...")
            result, error = await process_query(
                query_text, rag_instance, query_param
            )

            if result:
                if not first_entry:
                    result_file.write(",\n")
                json.dump(result, result_file, ensure_ascii=False, indent=4)
                first_entry = False
            elif error:
                json.dump(error, err_file, ensure_ascii=False, indent=4)
                err_file.write("\n")

        result_file.write("\n]")


async def initialize_rag(working_dir):
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
        graph_storage=GRAPH_STORAGE,
    )

    await rag.initialize_storages()
    return rag


async def main():
    parser = argparse.ArgumentParser(description="Query LightRAG")
    parser.add_argument(
        "-d",
        "--domains",
        nargs="+",
        default=["agriculture"],
        help="Domains to query (default: agriculture)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="hybrid",
        choices=["naive", "local", "global", "hybrid", "mix"],
        help="Query mode (default: hybrid)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Max queries to run per domain (0 = all, default: 0)",
    )
    args = parser.parse_args()

    for cls in args.domains:
        print(f"\n{'='*60}")
        print(f"Querying domain: {cls} (mode={args.mode})")
        print(f"  LLM:       {LLM_MODEL}")
        print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
        print(f"{'='*60}")

        working_dir = f"../{cls}"
        questions_file = f"../datasets/questions/{cls}_questions.txt"

        if not os.path.exists(questions_file):
            print(f"Questions file not found: {questions_file}")
            print("Run Step_2.py first to generate questions.")
            continue

        rag = await initialize_rag(working_dir)
        query_param = QueryParam(mode=args.mode)

        queries = extract_queries(questions_file)
        if args.max_queries > 0:
            queries = queries[: args.max_queries]
        print(f"Running {len(queries)} queries")

        result_file = f"{cls}_{args.mode}_result.json"
        error_file = f"{cls}_{args.mode}_errors.json"

        await run_queries_and_save_to_json(
            queries, rag, query_param, result_file, error_file
        )

        await rag.finalize_storages()
        print(f"Results saved to {result_file}")


if __name__ == "__main__":
    asyncio.run(main())
