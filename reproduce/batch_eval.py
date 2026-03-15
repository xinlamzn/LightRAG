"""
Batch evaluation: Compare two RAG system results using Bedrock Claude Opus 4.6.

Evaluates answers on Comprehensiveness, Diversity, and Empowerment.

Usage:
    python batch_eval.py \\
        --queries ../datasets/questions/agriculture_questions.txt \\
        --result1 agriculture_hybrid_result.json \\
        --result2 agriculture_naive_result.json \\
        --output agriculture_eval.json

Environment variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    BEDROCK_LLM_MODEL  (default: us.anthropic.claude-opus-4-6-v1)
"""

import os
import re
import json
import asyncio
import argparse

from lightrag.llm.bedrock import bedrock_complete_if_cache

LLM_MODEL = os.environ.get("BEDROCK_LLM_MODEL", "us.anthropic.claude-opus-4-6-v1")


async def evaluate_pair(query, answer1, answer2, index):
    """Evaluate a single (query, answer1, answer2) triplet."""
    prompt = f"""
    You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

    - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
    - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
    - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

    For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

    Here is the question:
    {query}

    Here are the two answers:

    **Answer 1:**
    {answer1}

    **Answer 2:**
    {answer2}

    Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

    Output your evaluation in the following JSON format:

    {{
        "Comprehensiveness": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide explanation here]"
        }},
        "Diversity": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide explanation here]"
        }},
        "Empowerment": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide explanation here]"
        }},
        "Overall Winner": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
        }}
    }}
    """

    system_prompt = (
        "You are an expert tasked with evaluating two answers to the same question "
        "based on three criteria: Comprehensiveness, Diversity, and Empowerment. "
        "Always respond with valid JSON."
    )

    result = await bedrock_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        max_tokens=2048,
    )

    # Try to parse the JSON from the response
    try:
        # Find JSON block in response
        json_match = re.search(r"\{[\s\S]*\}", result)
        if json_match:
            evaluation = json.loads(json_match.group())
        else:
            evaluation = {"raw_response": result}
    except json.JSONDecodeError:
        evaluation = {"raw_response": result}

    return {"query": query, "evaluation": evaluation}


async def batch_eval(query_file, result1_file, result2_file, output_file_path):
    with open(query_file, "r") as f:
        data = f.read()
    data = data.replace("**", "")
    queries = re.findall(r"- Question \d+: (.+)", data)

    with open(result1_file, "r") as f:
        answers1 = json.load(f)
    answers1 = [i["result"] for i in answers1]

    with open(result2_file, "r") as f:
        answers2 = json.load(f)
    answers2 = [i["result"] for i in answers2]

    print(f"Evaluating {len(queries)} query pairs using {LLM_MODEL}...")

    results = []
    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2)):
        print(f"  Evaluating {i+1}/{len(queries)}...")
        result = await evaluate_pair(query, answer1, answer2, i)
        results.append(result)

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print summary
    wins = {"Answer 1": 0, "Answer 2": 0, "Tie": 0}
    for r in results:
        winner = r.get("evaluation", {}).get("Overall Winner", {}).get("Winner", "Tie")
        if "1" in winner:
            wins["Answer 1"] += 1
        elif "2" in winner:
            wins["Answer 2"] += 1
        else:
            wins["Tie"] += 1

    total = len(results)
    print(f"\nResults saved to {output_file_path}")
    print(f"\nOverall Winners:")
    print(f"  Answer 1: {wins['Answer 1']}/{total} ({100*wins['Answer 1']/total:.1f}%)")
    print(f"  Answer 2: {wins['Answer 2']}/{total} ({100*wins['Answer 2']/total:.1f}%)")
    print(f"  Tie:      {wins['Tie']}/{total} ({100*wins['Tie']/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate two RAG system results")
    parser.add_argument(
        "--queries",
        required=True,
        help="Path to questions file (e.g. ../datasets/questions/agriculture_questions.txt)",
    )
    parser.add_argument(
        "--result1",
        required=True,
        help="Path to first result JSON file",
    )
    parser.add_argument(
        "--result2",
        required=True,
        help="Path to second result JSON file",
    )
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="Output file for evaluation results (default: eval_results.json)",
    )
    args = parser.parse_args()

    asyncio.run(batch_eval(args.queries, args.result1, args.result2, args.output))


if __name__ == "__main__":
    main()
