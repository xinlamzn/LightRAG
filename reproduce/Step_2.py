"""
Step 2: Generate evaluation queries using Bedrock Claude Opus 4.6.

Reads unique contexts, extracts summaries, and uses Claude to generate
structured questions (5 users x 5 tasks x 5 questions = 125 questions).

Usage:
    python Step_2.py                          # Generate for agriculture (default)
    python Step_2.py -d cs                    # Generate for a specific domain
    python Step_2.py -d agriculture cs legal  # Generate for multiple domains

Environment variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    BEDROCK_LLM_MODEL  (default: us.anthropic.claude-opus-4-6-v1)
"""

import os
import json
import asyncio
import argparse

from lightrag.llm.bedrock import bedrock_complete_if_cache

LLM_MODEL = os.environ.get("BEDROCK_LLM_MODEL", "us.anthropic.claude-opus-4-6-v1")


def get_summary(context, tot_tokens=2000):
    """Extract representative tokens from beginning and end of context.

    Uses a simple character-based approximation (~4 chars per token)
    instead of requiring the transformers library.
    """
    chars_per_token = 4
    half_chars = (tot_tokens // 2) * chars_per_token
    offset = 1000 * chars_per_token

    start_text = context[offset : offset + half_chars]
    end_text = context[-(offset + half_chars) : -offset] if len(context) > offset else ""

    return start_text + end_text


async def generate_questions(cls, input_dir, output_dir):
    input_file = f"{input_dir}/{cls}_unique_contexts.json"
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    with open(input_file, mode="r") as f:
        unique_contexts = json.load(f)

    print(f"Loaded {len(unique_contexts)} unique contexts for {cls}")

    summaries = [get_summary(context) for context in unique_contexts]
    total_description = "\n\n".join(summaries)

    # Truncate if too long for context window
    max_chars = 100000
    if len(total_description) > max_chars:
        total_description = total_description[:max_chars]
        print(f"  Truncated description to {max_chars} chars")

    prompt = f"""
    Given the following description of a dataset:

    {total_description}

    Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

    Output the results in the following structure:
    - User 1: [user description]
        - Task 1: [task description]
            - Question 1:
            - Question 2:
            - Question 3:
            - Question 4:
            - Question 5:
        - Task 2: [task description]
            ...
        - Task 5: [task description]
    - User 2: [user description]
        ...
    - User 5: [user description]
        ...
    """

    print(f"Generating questions for {cls} using {LLM_MODEL}...")
    result = await bedrock_complete_if_cache(
        LLM_MODEL,
        prompt,
        max_tokens=4096,
    )

    os.makedirs(output_dir, exist_ok=True)
    file_path = f"{output_dir}/{cls}_questions.txt"
    with open(file_path, "w") as file:
        file.write(result)

    print(f"{cls}_questions written to {file_path}")


async def main():
    parser = argparse.ArgumentParser(description="Generate evaluation queries")
    parser.add_argument(
        "-d",
        "--domains",
        nargs="+",
        default=["agriculture"],
        help="Domains to generate questions for (default: agriculture)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../datasets/unique_contexts",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../datasets/questions",
    )
    args = parser.parse_args()

    for cls in args.domains:
        await generate_questions(cls, args.input_dir, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
