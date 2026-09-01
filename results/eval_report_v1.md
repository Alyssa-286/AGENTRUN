# Fine-Tuned Qwen 7B Benchmark Report

**Overall: 18/23 passed (78.3%)**

| Task ID | Category | Result | Steps | Tool Calls | Latency (s) |
|---|---|---|---|---|---|
| calc_basic_1 | math | ✅ PASS | 2 | 1 | 5.53 |
| calc_basic_2 | math | ✅ PASS | 2 | 1 | 5.55 |
| calc_order_of_ops | math | ✅ PASS | 2 | 1 | 7.09 |
| wordcount_basic | text | ✅ PASS | 2 | 1 | 5.44 |
| weather_basic_1 | weather | ✅ PASS | 2 | 1 | 8.12 |
| weather_basic_2 | weather | ✅ PASS | 2 | 1 | 9.33 |
| search_basic_1 | search | ✅ PASS | 2 | 1 | 9.64 |
| search_basic_2 | search | ✅ PASS | 2 | 1 | 9.64 |
| multi_weather_calc | multi_tool | ✅ PASS | 3 | 2 | 16.14 |
| multi_wordcount_calc | multi_tool | ❌ FAIL | 2 | 1 | 9.64 |
| multi_search_weather | multi_tool | ✅ PASS | 2 | 2 | 14.38 |
| multi_three_tools | multi_tool | ❌ FAIL | 3 | 3 | 29.42 |
| chain_weather_then_calc | chained_reasoning | ❌ FAIL | 2 | 1 | 18.03 |
| time_sensitive_1 | time_sensitive | ✅ PASS | 2 | 1 | 10.36 |
| no_tool_needed_1 | no_tool_expected | ✅ PASS | 1 | 0 | 1.67 |
| no_tool_needed_2 | no_tool_expected | ✅ PASS | 1 | 0 | 3.30 |
| adversarial_negative_multiply | adversarial_math | ✅ PASS | 2 | 1 | 5.53 |
| adversarial_decimal_division | adversarial_math | ✅ PASS | 2 | 1 | 5.33 |
| adversarial_percentage | adversarial_math | ✅ PASS | 2 | 1 | 6.28 |
| adversarial_false_premise | adversarial_reasoning | ❌ FAIL | 2 | 1 | 17.61 |
| adversarial_chained_search_calc | adversarial_reasoning | ✅ PASS | 2 | 2 | 12.83 |
| adversarial_division_by_zero | adversarial_error_handling | ❌ FAIL | 1 | 0 | 3.50 |
| adversarial_ambiguous_city | adversarial_ambiguity | ✅ PASS | 2 | 1 | 10.25 |

## Category Breakdown

- **adversarial_ambiguity**: 1/1 passed
- **adversarial_error_handling**: 0/1 passed
- **adversarial_math**: 3/3 passed
- **adversarial_reasoning**: 1/2 passed
- **chained_reasoning**: 0/1 passed
- **math**: 3/3 passed
- **multi_tool**: 2/4 passed
- **no_tool_expected**: 2/2 passed
- **search**: 2/2 passed
- **text**: 1/1 passed
- **time_sensitive**: 1/1 passed
- **weather**: 2/2 passed

## Failed Tasks

- **multi_wordcount_calc**: missing_required_tools:['calculator']
- **multi_three_tools**: hit_max_steps, numeric_mismatch:expected=305989,found=[]
- **chain_weather_then_calc**: missing_required_tools:['calculator']
- **adversarial_false_premise**: missing_expected_content_any:['fictional', "doesn't exist", 'does not exist', 'not a real country', 'not real', 'marvel']
- **adversarial_division_by_zero**: missing_required_tools:['calculator'], expected_tool_error_but_none_occurred