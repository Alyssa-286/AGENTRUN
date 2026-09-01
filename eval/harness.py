"""
eval/harness.py — runs the task set against a live Agent and scores each result.

Scoring logic lives here, separate from the task DEFINITIONS (tasks.py) and
separate from agent internals (agent.py). This separation matters: you
could point this exact harness at a completely different agent
implementation (a fine-tuned model, a different framework) and get
directly comparable numbers, because the scoring rules don't know or care
how the agent produced its trace — only what the trace contains.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from eval.tasks import Task
from agent import Agent, RunTrace


@dataclass
class TaskResult:
    task: Task
    trace: RunTrace
    success: bool
    failure_reasons: list[str] = field(default_factory=list)  # empty if success


def _extract_numbers(text: str) -> list[float]:
    """Pull every number out of a string, handling commas (e.g. '305,989')
    and decimals. Used for numeric_check scoring."""
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    numbers = []
    for m in matches:
        cleaned = m.replace(",", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def score_task(task: Task, trace: RunTrace) -> TaskResult:
    """Apply every check the task defines. A task fails if ANY check fails;
    we collect ALL failing reasons (not just the first) so a report can
    show every problem in a single run, not just the first one hit."""
    failure_reasons: list[str] = []

    # --- Check 1: required tools were actually called ---
    missing_tools = task.required_tools - trace.tool_names_called
    if missing_tools:
        failure_reasons.append(f"missing_required_tools:{sorted(missing_tools)}")

    # --- Check 2: tool-level errors ---
    # Normally a tool error is a failure. But for tasks that deliberately
    # trigger one (e.g. division by zero) and want to see how the agent
    # HANDLES it, expect_tool_error flips this: no error occurring is the
    # failure, not the other way around.
    if task.expect_tool_error:
        if not trace.had_tool_error:
            failure_reasons.append("expected_tool_error_but_none_occurred")
    else:
        if trace.had_tool_error:
            errored_tools = [tc.tool_name for tc in trace.tool_calls if tc.is_error]
            failure_reasons.append(f"tool_execution_error:{errored_tools}")

    # --- Check 3: agent didn't get stuck in the step loop ---
    if trace.hit_max_steps:
        failure_reasons.append("hit_max_steps")

    # --- Check 4: expected substrings appear in the final answer ---
    answer_lower = trace.final_answer.lower()
    missing_substrings = [s for s in task.answer_contains if s.lower() not in answer_lower]
    if missing_substrings:
        failure_reasons.append(f"missing_expected_content:{missing_substrings}")

    # --- Check 4b: at least ONE of a set of acceptable phrasings appears ---
    if task.answer_contains_any:
        if not any(s.lower() in answer_lower for s in task.answer_contains_any):
            failure_reasons.append(
                f"missing_expected_content_any:{task.answer_contains_any}"
            )

    # --- Check 5: numeric answer matches within tolerance ---
    if task.numeric_check is not None:
        expected_value, tolerance = task.numeric_check
        found_numbers = _extract_numbers(trace.final_answer)
        if not any(abs(n - expected_value) <= tolerance for n in found_numbers):
            failure_reasons.append(
                f"numeric_mismatch:expected={expected_value},found={found_numbers}"
            )

    return TaskResult(
        task=task,
        trace=trace,
        success=(len(failure_reasons) == 0),
        failure_reasons=failure_reasons,
    )


async def run_benchmark(
    agent: Agent,
    tasks: list[Task],
    verbose: bool = False,
    pace_delay_seconds: float = 1.5,
) -> list[TaskResult]:
    """Run every task against the agent sequentially, scoring each one.
    Sequential (not parallel) on purpose for Phase 4: the agent's chat
    session and short-term memory are stateful, and we don't want one
    task's context bleeding into the next. We reset memory between tasks
    for the same reason — each task should be evaluated independently."""
    results: list[TaskResult] = []

    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] Running task '{task.id}'...", end=" ", flush=True)

        agent.reset_memory()  # each task starts with a clean conversation
        await agent.run(task.prompt, verbose=verbose)
        trace = agent.last_trace
        if trace is None:
            raise RuntimeError(f"Agent did not generate a trace for task '{task.id}'")

        result = score_task(task, trace)
        results.append(result)

        status = "PASS" if result.success else f"FAIL ({result.failure_reasons})"
        print(status)

        if i < len(tasks) and pace_delay_seconds > 0:
            await asyncio.sleep(pace_delay_seconds)

    return results
