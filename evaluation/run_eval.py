"""
run_eval.py — run the full benchmark and print a report.

Usage:
    python run_eval.py

This connects the agent to the MCP server exactly like main.py does, but
instead of an interactive chat loop, it runs every task in eval/tasks.py
automatically and scores each one.

Outputs:
    - eval_report.md          human-readable markdown report
    - eval_results_gemini.json machine-readable JSON results
"""

from __future__ import annotations

import asyncio
import json
import time

from agent import Agent
from eval.tasks import TASKS
from eval.harness import run_benchmark
from eval.report import print_report, results_to_markdown


def save_json_results(results: list, filepath: str = "eval_results_gemini.json") -> None:
    """Save benchmark results as machine-readable JSON."""
    json_results = {
        "model": "Gemini 3.5 Flash (Baseline)",
        "base_model": "gemini-3.5-flash",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tasks": len(results),
        "passed": sum(1 for r in results if r.success),
        "results": [
            {
                "task_id": r.task.id,
                "category": r.task.category,
                "success": r.success,
                "failure_reasons": r.failure_reasons,
                "steps_taken": r.trace.steps_taken,
                "tool_calls": [
                    {
                        "tool_name": tc.tool_name,
                        "arguments": tc.arguments,
                        "is_error": tc.is_error,
                        "result": tc.result,
                    }
                    for tc in r.trace.tool_calls
                ],
                "final_answer": r.trace.final_answer,
                "latency_seconds": r.trace.latency_seconds,
                "compute_latency_seconds": r.trace.compute_latency_seconds,
                "retry_wait_seconds": r.trace.retry_wait_seconds,
                "hit_max_steps": r.trace.hit_max_steps,
            }
            for r in results
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)


async def main():
    # No long-term memory during eval — we want each task scored on the
    # agent's raw capability, not on facts it happens to remember from a
    # previous benchmark run. This is important for reproducibility.
    agent = Agent(model_name="gemini-3.6-flash", max_steps=6, long_term_memory=None)
    await agent.connect(verbose=False)

    try:
        results = await run_benchmark(agent, TASKS, verbose=False)
    finally:
        await agent.close()

    # Print human-readable report to terminal
    print_report(results)

    # Save markdown report
    with open("eval_report.md", "w", encoding="utf-8") as f:
        f.write("# Agent Benchmark Report\n")
        f.write(results_to_markdown(results))
    print("\nMarkdown report written to eval_report.md")

    # Save machine-readable JSON results
    save_json_results(results)
    print("JSON results written to eval_results_gemini.json")


if __name__ == "__main__":
    asyncio.run(main())
