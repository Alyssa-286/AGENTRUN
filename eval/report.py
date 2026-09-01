"""
eval/report.py — turns a list of TaskResult into diagnostic benchmark metrics:
overall success rate, raw vs. compute latency, rate-limit wait time,
step efficiency, and failure mode taxonomy.
"""

from __future__ import annotations

from collections import Counter
from eval.harness import TaskResult


def print_report(results: list[TaskResult]):
    total = len(results)
    if total == 0:
        print("No benchmark results to report.")
        return

    passed = sum(1 for r in results if r.success)
    failed = total - passed

    avg_latency = sum(r.trace.latency_seconds for r in results) / total
    avg_compute_latency = sum(r.trace.compute_latency_seconds for r in results) / total
    total_retry_wait = sum(r.trace.retry_wait_seconds for r in results)
    avg_steps = sum(r.trace.steps_taken for r in results) / total
    avg_tool_calls = sum(len(r.trace.tool_calls) for r in results) / total

    print("\n" + "=" * 60)
    print("BENCHMARK REPORT (Phase 4.2 - Measured Latency)")
    print("=" * 60)
    print(f"Total tasks:               {total}")
    print(f"Passed:                    {passed} ({100 * passed / total:.1f}%)")
    print(f"Failed:                    {failed} ({100 * failed / total:.1f}%)")
    print(f"Avg raw latency:           {avg_latency:.2f}s  (includes rate-limit waits)")
    print(f"Avg compute latency:       {avg_compute_latency:.2f}s  (excludes rate-limit waits — use THIS for comparisons)")
    print(f"Total rate-limit wait:     {total_retry_wait:.1f}s across all tasks")
    print(f"Avg steps/task:            {avg_steps:.1f}")
    print(f"Avg tool calls:            {avg_tool_calls:.1f}")

    # --- Breakdown by task category ---
    print("\n--- By category ---")
    categories = sorted(set(r.task.category for r in results))
    for cat in categories:
        cat_results = [r for r in results if r.task.category == cat]
        cat_passed = sum(1 for r in cat_results if r.success)
        print(f"  {cat:26s} {cat_passed}/{len(cat_results)} passed")

    # --- Failure mode breakdown (the diagnostic part) ---
    if failed > 0:
        print("\n--- Failure modes (why tasks failed) ---")
        failure_categories: Counter[str] = Counter()
        for r in results:
            for reason in r.failure_reasons:
                failure_categories[reason.split(":")[0]] += 1

        for reason, count in failure_categories.most_common():
            print(f"  {reason:30s} {count}")

        print("\n--- Failed task details ---")
        for r in results:
            if not r.success:
                print(f"  [{r.task.id}] {r.failure_reasons}")

    print("=" * 60)


def results_to_markdown(results: list[TaskResult]) -> str:
    """Generate a markdown table — separating compute latency from rate-limit pauses."""
    lines = [
        "| Task ID | Category | Result | Steps | Tool Calls | Compute Latency (s) | Retry Wait (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        status = "✅ PASS" if r.success else "❌ FAIL"
        lines.append(
            f"| {r.task.id} | {r.task.category} | {status} | "
            f"{r.trace.steps_taken} | {len(r.trace.tool_calls)} | "
            f"{r.trace.compute_latency_seconds:.2f} | {r.trace.retry_wait_seconds:.1f} |"
        )

    total = len(results)
    passed = sum(1 for r in results if r.success)
    summary = f"\n**Overall: {passed}/{total} passed ({100 * passed / total:.1f}%)**\n"

    return summary + "\n" + "\n".join(lines)
