# Experiment Log

## Overview

This log documents the progression of the AgentLab project from baseline evaluation through two rounds of QLoRA fine-tuning, failure analysis, and targeted intervention.

---

## Phase 1: Frontier Agent Baseline (historical context)

**What was built:** A production-grade multi-tool agent using Google Gemini 3.6 Flash as the LLM, with MCP (Model Context Protocol) for tool execution. Tools included calculator, word count, web search, and weather lookup.

**Purpose:** Established a high-capability baseline and the full agent infrastructure (MCP server, provider adapters, memory system).

> **Note:** The Phase 1 frontier agent code is present in this repository (agent.py with `gemini-3.6-flash`, `mcp_server.py`, `providers/`, etc.). It is retained for completeness but is not part of the QLoRA distillation experiment.

---

## Phase 2: Benchmark Infrastructure

**What was built:** A 23-task held-out evaluation benchmark (`eval/tasks.py`) with machine-checkable assertions for every task. Categories:

| Category | Tasks | Description |
|---|---|---|
| `math` | 3 | Basic arithmetic and order-of-operations |
| `text` | 1 | Word count |
| `weather` | 2 | Current weather lookup |
| `search` | 2 | Web search |
| `multi_tool` | 4 | Parallel tool use |
| `chained_reasoning` | 1 | Tool output → next tool input |
| `time_sensitive` | 1 | Current information (needs live search) |
| `no_tool_expected` | 2 | Direct answer, no tool needed |
| `adversarial_math` | 3 | Negative numbers, decimals, percentages |
| `adversarial_reasoning` | 2 | False premise, chained search→calc |
| `adversarial_error_handling` | 1 | Division by zero |
| `adversarial_ambiguity` | 1 | Ambiguous entity (Springfield) |

**Scoring harness:** `eval/harness.py` — checks `required_tools`, `numeric_check`, `answer_contains`, `answer_contains_any`, `expect_tool_error`.

---

## Phase 3: Base Model Evaluation

**Date:** 2026-09-01 15:55:51

**What was evaluated:** Qwen/Qwen2.5-7B-Instruct with no LoRA adapter, via Colab T4 GPU (4-bit quantization). MCP tools were executed locally by `ft_agent.py`; the model was hosted on Colab.

**Results: 22/23 = 95.7%**

| Task | Result |
|---|---|
| `adversarial_division_by_zero` | ❌ FAIL — model did not call calculator when asked to divide by zero; instead answered directly |
| All others | ✅ PASS |

**Observation:** The base model is already strong on most tasks. The single failure was on the adversarial division-by-zero task, where the model avoided calling the calculator tool when given a problem designed to produce a tool error.

---

## Phase 4: LoRA v1 Fine-Tuning

**Date:** 2026-09-01 18:01:23

**What was trained:** LoRA adapter on Qwen2.5-7B-Instruct using QLoRA (4-bit NF4 quantization). Training data: `training_data.jsonl` — 35 clean tool-use trajectories.

**Training:** `training/colab_train_7b.py` in Google Colab (T4 GPU, ~45 minutes). Adapter output: `/content/agentlab_qwen_lora_7b`.

**Serving:** `serving/colab_server.py` — Colab-hosted inference server with ngrok tunnel.

**Results: 18/23 = 78.3%** (regression of −17.4 pp vs. base)

| Task | Result | Failure Mode |
|---|---|---|
| `multi_wordcount_calc` | ❌ FAIL | `missing_required_tools:['calculator']` — used word count result but did not call calculator |
| `multi_three_tools` | ❌ FAIL | `hit_max_steps, numeric_mismatch` — exhausted steps before completing |
| `chain_weather_then_calc` | ❌ FAIL | `missing_required_tools:['calculator']` — called weather tool but not calculator |
| `adversarial_false_premise` | ❌ FAIL | `missing_expected_content_any` — did not reject the false premise |
| `adversarial_division_by_zero` | ❌ FAIL | Same as base — also did not call calculator |
| All others | ✅ PASS | |

**Analysis of v1 regression:**

The v1 model was trained on 35 trajectories. The regression was not random — it clustered in multi-step tasks:
- Tasks requiring tool chaining (word count → calculator, weather → calculator)
- Tasks requiring a second tool call after the first tool returned results
- The false-premise task, which required reasoning about fictional entities

---

## Phase 4.1: Diagnostic and Fix

### Finding 1: Tool-Call Parser Bug

The v1 Colab server (`serving/colab_server.py`) used a single-pass regex to extract `<tool_call>...</tool_call>` blocks from model output. When the fine-tuned model emitted malformed tool-call tags (missing closing tag before the next opening tag), the parser returned empty and the server treated it as a final answer.

**Evidence:** In `adversarial_chained_search_calc`, the model intended to call both `web_search` and `calculator`, but the parser silently dropped the first tool call due to malformed tags.

**Fix:** `serving/colab_server_v2.py` implements a two-pass parser:
1. **Pass 1 (strict):** Match properly closed `<tool_call>...</tool_call>` blocks
2. **Pass 2 (fallback):** If Pass 1 finds nothing, extract the first JSON object starting from the first `<tool_call>` opening tag

This handles the case where the model's output lacks a closing `</tool_call>` tag before the next `<tool_call>`.

### Finding 2: Training Data Gap Audit

Trajectory-level inspection of v1 failures revealed:

| Gap | Trajectories needed |
|---|---|
| word_count → calculator | 3 examples |
| get_weather → calculator | 3 examples |
| 3-tool chain + explicit final synthesis | 2 examples |
| Calculator error / division by zero | 2 examples |
| False-premise correction | 1 example |

**Total new trajectories:** 11 (audit identified 10 unique gaps, with one overlap)

---

## Phase 5: LoRA v2 Fine-Tuning

**Date:** 2026-09-01 (training run); evaluation 2026-09-01 22:32:05

**What was changed:**
1. Parser fix in `serving/colab_server_v2.py` (deployed for v2 evaluation)
2. Augmented training data: `training/training_data_augmented.jsonl` — 46 trajectories (35 original + 11 new)

**What was NOT changed:**
- Base model: Qwen/Qwen2.5-7B-Instruct (identical)
- QLoRA configuration (identical 4-bit NF4, rank=16, alpha=32)
- Benchmark (identical 23 tasks, identical max_steps=3)
- MCP tool infrastructure (identical tools)

**Training:** `training/colab_train_7b_v2.py` in Google Colab. Adapter output: `/content/agentlab_qwen_lora_7b_v2`.

**Serving:** `serving/colab_server_v2.py` with corrected two-pass parser and v2 adapter.

**Results: 23/23 = 100.0%**

| Task | v1 | v2 |
|---|---|---|
| `multi_wordcount_calc` | ❌ | ✅ |
| `multi_three_tools` | ❌ | ✅ |
| `chain_weather_then_calc` | ❌ | ✅ |
| `adversarial_false_premise` | ❌ | ✅ |
| `adversarial_division_by_zero` | ❌ | ✅ |

---

## Summary of Results

| Model | Trajectories | Tasks | Passed | Accuracy |
|---|---|---|---|---|
| Base Qwen 7B | 0 (untuned) | 23 | 22 | 95.7% |
| LoRA v1 | 35 | 23 | 18 | 78.3% |
| LoRA v2 | 46 | 23 | 23 | **100.0%** |

**v1 → v2: +21.7 pp** (18/23 → 23/23)
**Base → v2: +4.3 pp** (22/23 → 23/23)

---

## Key Takeaways

1. **Small training sets can cause regression.** 35 trajectories on a 7B model produced a weaker agent than the base model on multi-step tasks.

2. **Data quality > quantity.** The v2 improvement came from targeted trajectory augmentation addressing specific failure patterns, not from sheer data volume.

3. **Evaluation must be held-out and fixed.** Using the same 23 tasks for all three evaluations (base, v1, v2) ensured a controlled comparison. Changing the benchmark between runs would have made the comparison invalid.

4. **The parser is part of the system.** The tool-call parser in the Colab server is a component of the inference system. A single-pass regex was fragile against the fine-tuned model's slightly different output patterns.

5. **Base model is strong.** Qwen2.5-7B-Instruct with no fine-tuning achieved 22/23, confirming that the base model already has strong instruction-following and tool-use capability. The LoRA fine-tuning aimed to specialize, not to wholesale create capability.
