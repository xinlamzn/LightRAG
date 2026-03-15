"""
Run all reproduce steps (Step_0 through Step_3) in sequence.

Modes:
    --test   Quick end-to-end verification with minimal data
             (2 contexts, ~4 questions, agriculture only)
    --full   Full benchmark run with all data
             (all contexts, 125 questions per domain)

Usage:
    python run_all.py --test                      # Quick smoke test
    python run_all.py --full                      # Full run, agriculture
    python run_all.py --full -d agriculture cs    # Full run, multiple domains
    python run_all.py --full -d agriculture -m hybrid naive  # Compare modes
    python run_all.py --test --skip-step0         # Skip download (already done)

Environment variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    OPENSEARCH_HOSTS, OPENSEARCH_USER, OPENSEARCH_PASSWORD
    BEDROCK_LLM_MODEL       (default: us.anthropic.claude-opus-4-6-v1)
    BEDROCK_EMBEDDING_MODEL  (default: amazon.titan-embed-text-v2:0)
    BEDROCK_EMBEDDING_DIM    (default: 1024)
"""

import os
import sys
import time
import argparse
import subprocess


def run_step(description, cmd):
    """Run a subprocess and stream its output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nFAILED: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)

    print(f"\nCompleted: {description} ({elapsed:.1f}s)")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run all LightRAG reproduce steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py --test                       # Quick smoke test
  python run_all.py --full                       # Full run, agriculture only
  python run_all.py --full -d agriculture cs     # Full run, two domains
  python run_all.py --full -m hybrid naive       # Run two query modes, then compare
  python run_all.py --test --skip-step0          # Skip dataset download
        """,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--test",
        action="store_true",
        help="Quick end-to-end test (2 contexts, ~4 questions, agriculture)",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Full benchmark run (all contexts, 125 questions)",
    )

    parser.add_argument(
        "-d",
        "--domains",
        nargs="+",
        default=None,
        help="Domains to process (default: agriculture for --test, agriculture for --full)",
    )
    parser.add_argument(
        "-m",
        "--modes",
        nargs="+",
        default=None,
        choices=["naive", "local", "global", "hybrid", "mix"],
        help="Query modes for Step 3 (default: hybrid)",
    )
    parser.add_argument(
        "--skip-step0",
        action="store_true",
        help="Skip dataset download (Step 0)",
    )
    parser.add_argument(
        "--skip-step1",
        action="store_true",
        help="Skip insertion (Step 1) — use if data is already indexed",
    )
    parser.add_argument(
        "--skip-step2",
        action="store_true",
        help="Skip question generation (Step 2) — use if questions already exist",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run batch_eval to compare first two query modes (requires --modes with 2+ modes)",
    )

    args = parser.parse_args()

    python = sys.executable
    domains = args.domains or ["agriculture"]
    modes = args.modes or ["hybrid"]
    total_start = time.time()
    timings = {}

    if args.test:
        max_contexts = 2
        test_flag = True
    else:
        max_contexts = 0
        test_flag = False

    print(f"\n{'#'*60}")
    print(f"  LightRAG Reproduce Pipeline")
    print(f"  Mode:    {'TEST' if args.test else 'FULL'}")
    print(f"  Domains: {', '.join(domains)}")
    print(f"  Modes:   {', '.join(modes)}")
    if max_contexts > 0:
        print(f"  Max ctx:  {max_contexts}")
    print(f"{'#'*60}")

    # Step 0: Download dataset
    if not args.skip_step0:
        cmd = [python, "Step_0.py", "-d"] + domains
        timings["Step 0 (download)"] = run_step("Step 0: Download dataset", cmd)
    else:
        print("\nSkipping Step 0 (--skip-step0)")

    # Step 1: Insert contexts
    if not args.skip_step1:
        cmd = [python, "Step_1.py", "-d"] + domains
        if max_contexts > 0:
            cmd += ["--max-contexts", str(max_contexts)]
        timings["Step 1 (insert)"] = run_step("Step 1: Insert contexts into LightRAG", cmd)
    else:
        print("\nSkipping Step 1 (--skip-step1)")

    # Step 2: Generate questions
    if not args.skip_step2:
        cmd = [python, "Step_2.py", "-d"] + domains
        if test_flag:
            cmd += ["--test"]
        timings["Step 2 (questions)"] = run_step("Step 2: Generate evaluation questions", cmd)
    else:
        print("\nSkipping Step 2 (--skip-step2)")

    # Step 3: Query for each mode
    for query_mode in modes:
        cmd = [python, "Step_3.py", "-d"] + domains + ["-m", query_mode]
        if test_flag:
            cmd += ["--max-queries", "4"]
        timings[f"Step 3 ({query_mode})"] = run_step(
            f"Step 3: Query LightRAG (mode={query_mode})", cmd
        )

    # Optional: batch_eval comparing first two modes
    if args.eval and len(modes) >= 2:
        for domain in domains:
            result1 = f"{domain}_{modes[0]}_result.json"
            result2 = f"{domain}_{modes[1]}_result.json"
            questions = f"../datasets/questions/{domain}_questions.txt"
            output = f"{domain}_{modes[0]}_vs_{modes[1]}_eval.json"
            cmd = [
                python,
                "batch_eval.py",
                "--queries", questions,
                "--result1", result1,
                "--result2", result2,
                "--output", output,
            ]
            timings[f"Eval ({domain}: {modes[0]} vs {modes[1]})"] = run_step(
                f"Batch eval: {domain} ({modes[0]} vs {modes[1]})", cmd
            )
    elif args.eval and len(modes) < 2:
        print("\nSkipping eval: need at least 2 query modes (--modes mode1 mode2)")

    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline Complete")
    print(f"{'='*60}")
    for step, elapsed in timings.items():
        print(f"  {step:40s} {elapsed:8.1f}s")
    print(f"  {'TOTAL':40s} {total_elapsed:8.1f}s")
    print()


if __name__ == "__main__":
    main()
