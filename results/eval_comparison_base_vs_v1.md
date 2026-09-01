# Agent Benchmark Comparison

## Executive Summary

| Metric | Base Qwen 7B | LoRA v1 | Delta |
|---|---|---|---|
| **Overall Accuracy** | **22/23 (95.7%)** | **18/23 (78.3%)** | -17.4 pp |
| **Avg Steps / Task** | 1.96 | 1.96 | — |
| **Avg Compute Latency** | 8.34s | 9.77s | — |

---

## Category-by-Category Breakdown

| Category | Tasks | Base Qwen 7B | LoRA v1 |
|---|---|---|---|
| `adversarial_ambiguity` | 1 | 1/1 (100%) | 1/1 (100%) |
| `adversarial_error_handling` | 1 | 0/1 (0%) | 0/1 (0%) |
| `adversarial_math` | 3 | 3/3 (100%) | 3/3 (100%) |
| `adversarial_reasoning` | 2 | 2/2 (100%) | 1/2 (50%) |
| `chained_reasoning` | 1 | 1/1 (100%) | 0/1 (0%) |
| `math` | 3 | 3/3 (100%) | 3/3 (100%) |
| `multi_tool` | 4 | 4/4 (100%) | 2/4 (50%) |
| `no_tool_expected` | 2 | 2/2 (100%) | 2/2 (100%) |
| `search` | 2 | 2/2 (100%) | 2/2 (100%) |
| `text` | 1 | 1/1 (100%) | 1/1 (100%) |
| `time_sensitive` | 1 | 1/1 (100%) | 1/1 (100%) |
| `weather` | 2 | 2/2 (100%) | 2/2 (100%) |

---

## Task-by-Task Detailed Matrix

| Task ID | Category | Base Qwen 7B | LoRA v1 |
|---|---|---|---|
| `calc_basic_1` | `math` | PASS | PASS |
| `calc_basic_2` | `math` | PASS | PASS |
| `calc_order_of_ops` | `math` | PASS | PASS |
| `wordcount_basic` | `text` | PASS | PASS |
| `weather_basic_1` | `weather` | PASS | PASS |
| `weather_basic_2` | `weather` | PASS | PASS |
| `search_basic_1` | `search` | PASS | PASS |
| `search_basic_2` | `search` | PASS | PASS |
| `multi_weather_calc` | `multi_tool` | PASS | PASS |
| `multi_wordcount_calc` | `multi_tool` | PASS | FAIL |
| `multi_search_weather` | `multi_tool` | PASS | PASS |
| `multi_three_tools` | `multi_tool` | PASS | FAIL |
| `chain_weather_then_calc` | `chained_reasoning` | PASS | FAIL |
| `time_sensitive_1` | `time_sensitive` | PASS | PASS |
| `no_tool_needed_1` | `no_tool_expected` | PASS | PASS |
| `no_tool_needed_2` | `no_tool_expected` | PASS | PASS |
| `adversarial_negative_multiply` | `adversarial_math` | PASS | PASS |
| `adversarial_decimal_division` | `adversarial_math` | PASS | PASS |
| `adversarial_percentage` | `adversarial_math` | PASS | PASS |
| `adversarial_false_premise` | `adversarial_reasoning` | PASS | FAIL |
| `adversarial_chained_search_calc` | `adversarial_reasoning` | PASS | PASS |
| `adversarial_division_by_zero` | `adversarial_error_handling` | FAIL | FAIL |
| `adversarial_ambiguous_city` | `adversarial_ambiguity` | PASS | PASS |

---

## Task-Level Discrepancies

The following tasks show performance differences between models:

| Task ID | Category | Base Qwen 7B | LoRA v1 |
|---|---|---|---|
| `multi_wordcount_calc` | `multi_tool` | PASS | FAIL |
| `multi_three_tools` | `multi_tool` | PASS | FAIL |
| `chain_weather_then_calc` | `chained_reasoning` | PASS | FAIL |
| `adversarial_false_premise` | `adversarial_reasoning` | PASS | FAIL |

### Base Qwen 7B — Failed Tasks


### LoRA v1 — Failed Tasks

- **multi_wordcount_calc** (`multi_tool`): missing_required_tools:['calculator']
- **multi_three_tools** (`multi_tool`): hit_max_steps, numeric_mismatch:expected=305989,found=[]
- **chain_weather_then_calc** (`chained_reasoning`): missing_required_tools:['calculator']
- **adversarial_false_premise** (`adversarial_reasoning`): missing_expected_content_any:['fictional', "doesn't exist", 'does not exist', 'not a real country', 'not real', 'marvel']

---

## Failure Mode Taxonomy

### Base Qwen 7B

- **missing_required_tools**: 1 task(s)
- **expected_tool_error_but_none_occurred**: 1 task(s)

### LoRA v1

- **missing_required_tools**: 3 task(s)
- **hit_max_steps**: 1 task(s)
- **numeric_mismatch**: 1 task(s)
- **missing_expected_content_any**: 1 task(s)
- **expected_tool_error_but_none_occurred**: 1 task(s)