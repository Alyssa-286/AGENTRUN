"""
run_eval_finetuned.py — Run the benchmark against the remote fine-tuned Qwen 7B model.

Uses the exact same 23 held-out tasks from eval/tasks.py as run_eval.py,
but sends them to the Colab-hosted Qwen 2.5 7B + LoRA model.

Usage:
    python run_eval_finetuned.py

Requirements:
    - COLAB_SERVER_URL environment variable set (or pass via --url argument)
    - MCP server (mcp_server.py) must be available locally for tool execution
    - Colab server must be running with /health and /chat endpoints
"""

from __future__ import annotations

import os
import sys
# Ensure project root is on the path so we can import ft_agent and eval.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import asyncio
import json
import os
import time
from typing import Optional

from ft_agent import QwenRemoteClient
from eval.tasks import TASKS
from eval.harness import run_benchmark, TaskResult


def task_result_to_dict(result: TaskResult) -> dict:
    """Convert a TaskResult to a JSON-serializable dictionary."""
    return {
        "task_id": result.task.id,
        "category": result.task.category,
        "success": result.success,
        "failure_reasons": result.failure_reasons,
        "steps_taken": result.trace.steps_taken,
        "tool_calls": [
            {
                "tool_name": tc.tool_name,
                "arguments": tc.arguments,
                "is_error": tc.is_error,
                "result_preview": tc.result[:200] if len(tc.result) > 200 else tc.result,
            }
            for tc in result.trace.tool_calls
        ],
        "final_answer": result.trace.final_answer,
        "latency_seconds": result.trace.latency_seconds,
        "compute_latency_seconds": result.trace.compute_latency_seconds,
        "retry_wait_seconds": result.trace.retry_wait_seconds,
        "hit_max_steps": result.trace.hit_max_steps,
    }


def generate_markdown_report(results: list[TaskResult]) -> str:
    """Generate a human-readable markdown report."""
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed

    lines = [
        "# Fine-Tuned Qwen 7B Benchmark Report",
        "",
        f"**Overall: {passed}/{total} passed ({100 * passed / total:.1f}%)**",
        "",
        "| Task ID | Category | Result | Steps | Tool Calls | Latency (s) |",
        "|---|---|---|---|---|---|",
    ]

    for r in results:
        status = "✅ PASS" if r.success else "❌ FAIL"
        lines.append(
            f"| {r.task.id} | {r.task.category} | {status} | "
            f"{r.trace.steps_taken} | {len(r.trace.tool_calls)} | "
            f"{r.trace.latency_seconds:.2f} |"
        )

    # Category breakdown
    lines.extend(["", "## Category Breakdown", ""])
    categories = sorted(set(r.task.category for r in results))
    for cat in categories:
        cat_results = [r for r in results if r.task.category == cat]
        cat_passed = sum(1 for r in cat_results if r.success)
        lines.append(f"- **{cat}**: {cat_passed}/{len(cat_results)} passed")

    # Failure details
    if failed > 0:
        lines.extend(["", "## Failed Tasks", ""])
        for r in results:
            if not r.success:
                lines.append(f"- **{r.task.id}**: {', '.join(r.failure_reasons)}")

    return "\n".join(lines)


async def run_evaluation(
    colab_server_url: Optional[str] = None,
    max_steps: int = 3,
    verbose: bool = False,
    pace_delay: float = 1.0,
) -> list[TaskResult]:
    """Run the benchmark against the remote Qwen model."""
    # Get server URL from environment if not provided
    server_url = colab_server_url or os.environ.get("COLAB_SERVER_URL")
    if not server_url:
        raise RuntimeError(
            "COLAB_SERVER_URL not set. Either pass --url argument or set the environment variable."
        )

    print("=" * 60)
    print("Fine-Tuned Qwen 7B + LoRA Benchmark")
    print(f"Server: {server_url}")
    print("=" * 60)

    # Initialize remote agent
    agent = QwenRemoteClient(
        colab_server_url=server_url,
        mcp_server_script="mcp_server.py",
        max_steps=max_steps,
    )

    await agent.connect(verbose=True)

    try:
        # Run benchmark
        results = await run_benchmark(
            agent,
            TASKS,
            verbose=verbose,
            pace_delay_seconds=pace_delay,
        )
    finally:
        await agent.close()

    return results


def save_json_results(results: list[TaskResult], filepath: str) -> None:
    """Save results to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "Qwen2.5-7B-Instruct + LoRA (Fine-Tuned)",
                "base_model": "Qwen2.5-7B-Instruct",
                "adapter": "agentlab_qwen_lora",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_tasks": len(results),
                "passed": sum(1 for r in results if r.success),
                "results": [task_result_to_dict(r) for r in results],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"JSON results saved to {filepath}")


async def main():
    parser = argparse.ArgumentParser(description="Run benchmark against fine-tuned Qwen model")
    parser.add_argument(
        "--url",
        dest="server_url",
        help="Colab server URL (or set COLAB_SERVER_URL env var)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Maximum tool-call steps (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output during evaluation",
    )
    parser.add_argument(
        "--pace-delay",
        type=float,
        default=1.0,
        help="Delay between tasks in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--json-output",
        default="results/eval_results_finetuned.json",
        help="Output JSON file path (default: results/eval_results_finetuned.json)",
    )
    parser.add_argument(
        "--md-output",
        default="results/eval_report_finetuned.md",
        help="Output markdown file path (default: results/eval_report_finetuned.md)",
    )

    args = parser.parse_args()

    # Run evaluation
    results = await run_evaluation(
        colab_server_url=args.server_url,
        max_steps=args.max_steps,
        verbose=args.verbose,
        pace_delay=args.pace_delay,
    )

    # Print summary
    total = len(results)
    passed = sum(1 for r in results if r.success)
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Total: {total} tasks")
    print(f"Passed: {passed} ({100 * passed / total:.1f}%)")
    print(f"Failed: {total - passed} ({100 * (total - passed) / total:.1f}%)")

    # Save JSON results
    save_json_results(results, args.json_output)

    # Generate and save markdown report
    md_report = generate_markdown_report(results)
    with open(args.md_output, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"Markdown report saved to {args.md_output}")

    # Print category breakdown
    print("\n--- Category Breakdown ---")
    categories = sorted(set(r.task.category for r in results))
    for cat in categories:
        cat_results = [r for r in results if r.task.category == cat]
        cat_passed = sum(1 for r in cat_results if r.success)
        print(f"  {cat:26s} {cat_passed}/{len(cat_results)} passed")

    # Print failures
    if total - passed > 0:
        print("\n--- Failed Tasks ---")
        for r in results:
            if not r.success:
                print(f"  [{r.task.id}] {', '.join(r.failure_reasons)}")


if __name__ == "__main__":
    asyncio.run(main())
