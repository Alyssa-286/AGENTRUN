# Base Qwen 2.5 7B Benchmark Report

**Overall: 22/23 passed (95.7%)**

| Task ID | Category | Result | Steps | Tool Calls | Latency (s) |
|---|---|---|---|---|---|
| calc_basic_1 | math | ✅ PASS | 2 | 1 | 5.30 |
| calc_basic_2 | math | ✅ PASS | 2 | 1 | 4.88 |
| calc_order_of_ops | math | ✅ PASS | 2 | 1 | 6.05 |
| wordcount_basic | text | ✅ PASS | 2 | 1 | 5.03 |
| weather_basic_1 | weather | ✅ PASS | 2 | 1 | 7.91 |
| weather_basic_2 | weather | ✅ PASS | 2 | 1 | 7.89 |
| search_basic_1 | search | ✅ PASS | 2 | 1 | 8.33 |
| search_basic_2 | search | ✅ PASS | 2 | 1 | 7.62 |
| multi_weather_calc | multi_tool | ✅ PASS | 2 | 2 | 9.28 |
| multi_wordcount_calc | multi_tool | ✅ PASS | 3 | 3 | 14.20 |
| multi_search_weather | multi_tool | ✅ PASS | 2 | 2 | 11.39 |
| multi_three_tools | multi_tool | ✅ PASS | 2 | 3 | 22.06 |
| chain_weather_then_calc | chained_reasoning | ✅ PASS | 3 | 3 | 16.42 |
| time_sensitive_1 | time_sensitive | ✅ PASS | 2 | 1 | 9.06 |
| no_tool_needed_1 | no_tool_expected | ✅ PASS | 1 | 0 | 2.69 |
| no_tool_needed_2 | no_tool_expected | ✅ PASS | 1 | 0 | 3.31 |
| adversarial_negative_multiply | adversarial_math | ✅ PASS | 2 | 1 | 4.55 |
| adversarial_decimal_division | adversarial_math | ✅ PASS | 2 | 1 | 5.17 |
| adversarial_percentage | adversarial_math | ✅ PASS | 2 | 1 | 4.55 |
| adversarial_false_premise | adversarial_reasoning | ✅ PASS | 2 | 1 | 16.41 |
| adversarial_chained_search_calc | adversarial_reasoning | ✅ PASS | 2 | 2 | 11.48 |
| adversarial_division_by_zero | adversarial_error_handling | ❌ FAIL | 1 | 0 | 2.06 |
| adversarial_ambiguous_city | adversarial_ambiguity | ✅ PASS | 2 | 1 | 6.16 |

## Category Breakdown

- **adversarial_ambiguity**: 1/1 passed
- **adversarial_error_handling**: 0/1 passed
- **adversarial_math**: 3/3 passed
- **adversarial_reasoning**: 2/2 passed
- **chained_reasoning**: 1/1 passed
- **math**: 3/3 passed
- **multi_tool**: 4/4 passed
- **no_tool_expected**: 2/2 passed
- **search**: 2/2 passed
- **text**: 1/1 passed
- **time_sensitive**: 1/1 passed
- **weather**: 2/2 passed

## Failed Tasks

- **adversarial_division_by_zero**: missing_required_tools:['calculator'], expected_tool_error_but_none_occurred