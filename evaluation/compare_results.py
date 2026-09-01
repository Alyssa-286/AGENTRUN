"""
compare_results.py — Comparative analysis of two benchmark result JSONs.

Compares machine-readable JSON results from two eval_results_*.json files
to produce a comprehensive comparative report.

Model labels are derived from each JSON file's own metadata (the "model"
field), not hardcoded. Pass --label-a / --label-b to override, or
--label-auto to use just the basename of the JSON file as the label.

Usage:
    python compare_results.py eval_results_base.json eval_results_v2.json
    python compare_results.py --gemini eval_results_base.json --qwen eval_results_v2.json
    python compare_results.py eval_results_base.json eval_results_v2.json --label-a "Base Qwen 7B" --label-b "LoRA v2"
    python compare_results.py  # uses default filenames

Outputs:
    - eval_comparison_report.md     detailed markdown comparison
    - Terminal summary               concise comparison table
"""

from __future__ import annotations

import os
import sys
# Ensure project root is on the path so we can import eval.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval.tasks import TASKS


def load_json_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_task_consistency(a_results: List[Dict], b_results: List[Dict], label_a: str, label_b: str) -> None:
    """Validate that both benchmark files contain the same tasks in the same order."""
    a_task_ids = [r["task_id"] for r in a_results]
    b_task_ids = [r["task_id"] for r in b_results]

    if a_task_ids != b_task_ids:
        raise ValueError(
            f"Task ID mismatch:\n"
            f"  {label_a} tasks: {a_task_ids}\n"
            f"  {label_b} tasks:   {b_task_ids}\n"
            f"Both benchmarks must use identical tasks in identical order."
        )

    print(f"Task sets validated - both benchmarks use identical {len(a_task_ids)} tasks in same order")


def compute_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate metrics from benchmark results."""
    total = len(results)
    passed = sum(1 for r in results if r["success"])

    # Compute average metrics
    avg_steps = sum(r["steps_taken"] for r in results) / total
    avg_compute_latency = sum(r["compute_latency_seconds"] for r in results) / total
    total_tool_calls = sum(len(r["tool_calls"]) for r in results)

    # Tool error statistics
    # A tool error is "expected" if the task passed (success=True) — meaning the
    # tool returning an error was the required behavior (e.g. adversarial_division_by_zero
    # requires the calculator to error). A tool error is "unexpected" if the task
    # FAILED because a tool errored when it shouldn't have.
    tool_error_total = sum(1 for r in results for tc in r["tool_calls"] if tc.get("is_error"))
    tool_error_expected = sum(
        1 for r in results if r["success"]
        for tc in r["tool_calls"] if tc.get("is_error")
    )
    tool_error_unexpected = sum(
        1 for r in results if not r["success"]
        for tc in r["tool_calls"] if tc.get("is_error")
    )

    # Failure mode categorization
    failure_categories = {}
    for r in results:
        if not r["success"]:
            for reason in r["failure_reasons"]:
                category = reason.split(":")[0] if ":" in reason else reason
                failure_categories[category] = failure_categories.get(category, 0) + 1

    return {
        "total_tasks": total,
        "passed": passed,
        "success_rate": 100.0 * passed / total,
        "avg_steps": avg_steps,
        "avg_compute_latency": avg_compute_latency,
        "total_tool_calls": total_tool_calls,
        "tool_error_count": tool_error_total,
        "tool_error_expected": tool_error_expected,
        "tool_error_unexpected": tool_error_unexpected,
        "failure_categories": failure_categories,
    }


def analyze_task_differences(a_results: List[Dict], b_results: List[Dict], label_a: str, label_b: str) -> List[Dict]:
    """Identify tasks where the two models disagree on success/failure."""
    differences = []

    for a, b in zip(a_results, b_results):
        a_success = a["success"]
        b_success = b["success"]

        if a_success != b_success:
            difference = {
                "task_id": a["task_id"],
                "category": a["category"],
                "a_result": "PASS" if a_success else "FAIL",
                "a_failure_reasons": a["failure_reasons"],
                "b_result": "PASS" if b_success else "FAIL",
                "b_failure_reasons": b["failure_reasons"],
                "a_steps": a["steps_taken"],
                "b_steps": b["steps_taken"],
                "a_latency": a["compute_latency_seconds"],
                "b_latency": b["compute_latency_seconds"],
            }
            differences.append(difference)

    return differences


def generate_comparison_markdown(
    a_metrics: Dict[str, Any],
    b_metrics: Dict[str, Any],
    label_a: str,
    label_b: str,
    task_differences: List[Dict],
    a_results: List[Dict],
    b_results: List[Dict],
) -> str:
    """Generate comprehensive comparison report."""
    delta = b_metrics['success_rate'] - a_metrics['success_rate']
    delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"

    lines = [
        "# Agent Benchmark Comparison",
        "",
        "## Executive Summary",
        "",
        f"| Metric | {label_a} | {label_b} | Delta |",
        "|---|---|---|---|",
        f"| **Overall Accuracy** | **{a_metrics['passed']}/{a_metrics['total_tasks']} ({a_metrics['success_rate']:.1f}%)** | **{b_metrics['passed']}/{b_metrics['total_tasks']} ({b_metrics['success_rate']:.1f}%)** | {delta_str} pp |",
        f"| **Avg Steps / Task** | {a_metrics['avg_steps']:.2f} | {b_metrics['avg_steps']:.2f} | — |",
        f"| **Avg Compute Latency** | {a_metrics['avg_compute_latency']:.2f}s | {b_metrics['avg_compute_latency']:.2f}s | — |",
        "",
        "---",
        "",
        "## Category-by-Category Breakdown",
        "",
        f"| Category | Tasks | {label_a} | {label_b} |",
        "|---|---|---|---|",
    ]

    categories = sorted(set(t.category for t in TASKS))
    for cat in categories:
        a_cat = [r for r in a_results if r["category"] == cat]
        b_cat = [r for r in b_results if r["category"] == cat]
        a_p = sum(1 for r in a_cat if r["success"])
        b_p = sum(1 for r in b_cat if r["success"])
        lines.append(f"| `{cat}` | {len(a_cat)} | {a_p}/{len(a_cat)} ({100*a_p/len(a_cat):.0f}%) | {b_p}/{len(b_cat)} ({100*b_p/len(b_cat):.0f}%) |")

    # Task-by-task matrix
    lines.extend([
        "",
        "---",
        "",
        "## Task-by-Task Detailed Matrix",
        "",
        f"| Task ID | Category | {label_a} | {label_b} |",
        "|---|---|---|---|",
    ])

    for a, b in zip(a_results, b_results):
        a_status = "PASS" if a["success"] else "FAIL"
        b_status = "PASS" if b["success"] else "FAIL"
        lines.append(
            f"| `{a['task_id']}` | `{a['category']}` | {a_status} | {b_status} |"
        )

    # Differences section
    if task_differences:
        lines.extend([
            "",
            "---",
            "",
            "## Task-Level Discrepancies",
            "",
            "The following tasks show performance differences between models:",
            "",
            f"| Task ID | Category | {label_a} | {label_b} |",
            "|---|---|---|---|",
        ])

        for diff in task_differences:
            lines.append(
                f"| `{diff['task_id']}` | `{diff['category']}` | {diff['a_result']} | {diff['b_result']} |"
            )

        lines.extend(["", f"### {label_a} — Failed Tasks", ""])
        for diff in task_differences:
            if diff["a_result"] == "FAIL":
                fr = diff["a_failure_reasons"]
                lines.append(f"- **{diff['task_id']}** (`{diff['category']}`): {', '.join(fr) if fr else 'no failure reasons recorded'}")

        lines.extend(["", f"### {label_b} — Failed Tasks", ""])
        for diff in task_differences:
            if diff["b_result"] == "FAIL":
                fr = diff["b_failure_reasons"]
                lines.append(f"- **{diff['task_id']}** (`{diff['category']}`): {', '.join(fr) if fr else 'no failure reasons recorded'}")

    # Failure taxonomy
    lines.extend([
        "",
        "---",
        "",
        "## Failure Mode Taxonomy",
        "",
        f"### {label_a}",
        "",
    ])

    if a_metrics["failure_categories"]:
        for category, count in sorted(a_metrics["failure_categories"].items(), key=lambda x: -x[1]):
            lines.append(f"- **{category}**: {count} task(s)")
    else:
        lines.append("- *None* (100% pass rate)")

    lines.extend([
        "",
        f"### {label_b}",
        "",
    ])

    if b_metrics["failure_categories"]:
        for category, count in sorted(b_metrics["failure_categories"].items(), key=lambda x: -x[1]):
            lines.append(f"- **{category}**: {count} task(s)")
    else:
        lines.append("- *None* (100% pass rate)")

    return "\n".join(lines)


def print_terminal_summary(
    a_metrics: Dict[str, Any],
    b_metrics: Dict[str, Any],
    label_a: str,
    label_b: str,
    task_differences: List[Dict],
) -> None:
    """Print concise comparison summary to terminal."""
    delta = b_metrics["success_rate"] - a_metrics["success_rate"]
    delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"

    print("\n" + "=" * 70)
    print("COMPARATIVE BENCHMARK RESULTS")
    print("=" * 70)

    print("\nOVERALL PERFORMANCE:")
    print(f"  {label_a}: {a_metrics['passed']}/{a_metrics['total_tasks']} ({a_metrics['success_rate']:.1f}%)")
    print(f"  {label_b}: {b_metrics['passed']}/{b_metrics['total_tasks']} ({b_metrics['success_rate']:.1f}%)")
    print(f"  Delta: {delta_str} percentage points  ({label_b} minus {label_a})")

    print("\nEFFICIENCY METRICS:")
    print(f"  Avg Steps/Task:      {label_a} {a_metrics['avg_steps']:.2f}  |  {label_b} {b_metrics['avg_steps']:.2f}")
    print(f"  Avg Compute Latency:  {label_a} {a_metrics['avg_compute_latency']:.2f}s  |  {label_b} {b_metrics['avg_compute_latency']:.2f}s")

    print("\nTASK DISCREPANCIES:")
    if task_differences:
        print(f"  Found {len(task_differences)} tasks with differing success:")
        for diff in task_differences:
            print(f"    - {diff['task_id']} (`{diff['category']}`): {label_a} {diff['a_result']} / {label_b} {diff['b_result']}")
    else:
        print("  No differences - both models achieved identical results")

    print("\n" + "=" * 70)
async def main():
    parser = argparse.ArgumentParser(
        description="Compare two benchmark result JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_results.py eval_results_base.json eval_results_v2.json
  python compare_results.py eval_results_base.json eval_results_v1.json --label-a "Base Qwen 7B" --label-b "LoRA v1"
  python compare_results.py eval_results_base.json eval_results_v2.json --label-auto
  python compare_results.py  # uses default filenames
        """
    )

    parser.add_argument(
        "baseline_json",
        nargs="?",
        default=None,
        help="Path to first results JSON (model A, the baseline)",
    )
    parser.add_argument(
        "comparison_json",
        nargs="?",
        default=None,
        help="Path to second results JSON (model B, the comparison)",
    )
    parser.add_argument(
        "--gemini",
        dest="gemini_json",
        default=None,
        help="[DEPRECATED] Use positional arg instead",
    )
    parser.add_argument(
        "--qwen",
        dest="qwen_json",
        default=None,
        help="[DEPRECATED] Use positional arg instead",
    )
    parser.add_argument(
        "--output",
        default="results/eval_comparison_report.md",
        help="Output markdown file path (default: results/eval_comparison_report.md)",
    )
    parser.add_argument(
        "--label-a",
        default=None,
        help="Override label for model A. If not set, uses the JSON 'model' field.",
    )
    parser.add_argument(
        "--label-b",
        default=None,
        help="Override label for model B. If not set, uses the JSON 'model' field.",
    )
    parser.add_argument(
        "--label-auto",
        action="store_true",
        help="Use just the basename of each JSON file as the label (e.g. 'eval_results_base' → 'Base'). Takes precedence over --label-a/--label-b.",
    )

    args = parser.parse_args()

    # Resolve file paths: positional args > --gemini/--qwen (deprecated) > defaults
    baseline_path = (
        args.baseline_json
        or args.gemini_json
        or "results/eval_results_base.json"
    )
    comparison_path = (
        args.comparison_json
        or args.qwen_json
        or "results/eval_results_v2.json"
    )

    # Load data
    print("Loading benchmark results...")
    a_data = load_json_results(baseline_path)
    b_data = load_json_results(comparison_path)

    a_results = a_data["results"]
    b_results = b_data["results"]

    # Resolve labels
    if args.label_auto:
        import os
        label_a = os.path.splitext(os.path.basename(baseline_path))[0]  # e.g. "eval_results_base"
        label_b = os.path.splitext(os.path.basename(comparison_path))[0]
    else:
        label_a = args.label_a if args.label_a else a_data["model"]
        label_b = args.label_b if args.label_b else b_data["model"]

    print(f"  Model A: {label_a}  ({baseline_path})")
    print(f"  Model B: {label_b}  ({comparison_path})")

    # Validate consistency
    validate_task_consistency(a_results, b_results, label_a, label_b)

    # Compute metrics
    a_metrics = compute_metrics(a_results)
    b_metrics = compute_metrics(b_results)

    # Analyze differences
    task_differences = analyze_task_differences(a_results, b_results, label_a, label_b)

    # Generate reports
    print(f"\nGenerating comparison report...")
    markdown_report = generate_comparison_markdown(
        a_metrics,
        b_metrics,
        label_a,
        label_b,
        task_differences,
        a_results,
        b_results,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    print(f"Comparison report written to {args.output}")
    print(f"Report length: {len(markdown_report)} characters")

    # Print terminal summary
    print_terminal_summary(a_metrics, b_metrics, label_a, label_b, task_differences)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())