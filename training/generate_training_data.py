"""
generate_training_data.py — runs the real agent over training_data_prompts.py,
captures full trajectories, and writes ONLY the good ones as fine-tuning data.

Why we filter:
Training on a trajectory that hit an error, looped past max_steps, or
never used a required tool teaches the fine-tuned model to REPRODUCE that
mistake. A smaller, clean dataset beats a larger, noisy one for supervised
tool-calling fine-tuning.

Output format: standard chat template messages format (JSONL), ready for
HuggingFace / Unsloth / TRL training.
"""

from __future__ import annotations

import asyncio
import json

from agent import Agent, RunTrace
from training_data_prompts import TRAINING_PROMPTS

SYSTEM_PROMPT = (
    "You are a helpful AI agent with access to tools: calculator, word_count, "
    "web_search, and get_weather. Use tools when you need real information or "
    "exact computation. Respond directly when you already know the answer."
)


def trace_to_training_example(trace: RunTrace) -> dict:
    """Convert one RunTrace into the messages-list format used for
    supervised fine-tuning. Tool calls and their results become
    (assistant tool_call) -> (tool result) message pairs, in chronological order,
    followed by the final assistant answer.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": trace.user_prompt},
    ]

    for tc in trace.tool_calls:
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": tc.tool_name, "arguments": tc.arguments}],
        })
        messages.append({
            "role": "tool",
            "name": tc.tool_name,
            "content": tc.result,
        })

    messages.append({"role": "assistant", "content": trace.final_answer})

    return {"messages": messages}


def is_good_trajectory(trace: RunTrace) -> tuple[bool, str]:
    """Filter criteria for what counts as a trajectory worth training on."""
    if trace.hit_max_steps:
        return False, "hit_max_steps (agent got stuck in loop)"
    if trace.had_tool_error:
        return False, "had_tool_error (tool failed during run)"
    if len(trace.final_answer.strip()) < 4:
        return False, "final_answer suspiciously short/empty"
    return True, ""


async def main():
    agent = Agent(model_name="gemini-3.6-flash", max_steps=6, long_term_memory=None)
    await agent.connect(verbose=True)

    kept = 0
    rejected = 0

    try:
        with open("training_data.jsonl", "w", encoding="utf-8") as f:
            for i, prompt in enumerate(TRAINING_PROMPTS, start=1):
                print(f"[{i}/{len(TRAINING_PROMPTS)}] {prompt[:50]}...", end=" ", flush=True)

                agent.reset_memory()
                await agent.run(prompt, verbose=False)
                trace = agent.last_trace
                if trace is None:
                    print("SKIPPED (no trace)")
                    continue

                keep, reason = is_good_trajectory(trace)
                if keep:
                    example = trace_to_training_example(trace)
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    f.flush()
                    kept += 1
                    print(f"KEPT ({len(trace.tool_calls)} tool call(s))")
                else:
                    rejected += 1
                    print(f"REJECTED — {reason}")

                if i < len(TRAINING_PROMPTS):
                    await asyncio.sleep(1.5)
    finally:
        await agent.close()

    print(f"\n{'=' * 60}")
    print(f"Done. {kept} good trajectories written to training_data.jsonl")
    print(f"{rejected} rejected (bad trajectories, NOT used for training)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
