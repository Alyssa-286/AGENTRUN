# Qwen 7B + LoRA v2 Benchmark Report

**Overall: 23/23 passed (100.0%)**

| Task ID | Category | Result | Steps | Tool Calls | Latency (s) |
|---|---|---|---|---|---|
| calc_basic_1 | math | ✅ PASS | 2 | 1 | 6.56 |
| calc_basic_2 | math | ✅ PASS | 2 | 1 | 5.58 |
| calc_order_of_ops | math | ✅ PASS | 2 | 1 | 5.36 |
| wordcount_basic | text | ✅ PASS | 2 | 1 | 5.34 |
| weather_basic_1 | weather | ✅ PASS | 2 | 1 | 7.56 |
| weather_basic_2 | weather | ✅ PASS | 2 | 1 | 7.72 |
| search_basic_1 | search | ✅ PASS | 2 | 1 | 11.44 |
| search_basic_2 | search | ✅ PASS | 2 | 1 | 9.75 |
| multi_weather_calc | multi_tool | ✅ PASS | 2 | 2 | 11.34 |
| multi_wordcount_calc | multi_tool | ✅ PASS | 3 | 3 | 13.14 |
| multi_search_weather | multi_tool | ✅ PASS | 2 | 2 | 14.16 |
| multi_three_tools | multi_tool | ✅ PASS | 2 | 3 | 22.80 |
| chain_weather_then_calc | chained_reasoning | ✅ PASS | 2 | 2 | 15.67 |
| time_sensitive_1 | time_sensitive | ✅ PASS | 2 | 1 | 9.53 |
| no_tool_needed_1 | no_tool_expected | ✅ PASS | 1 | 0 | 3.58 |
| no_tool_needed_2 | no_tool_expected | ✅ PASS | 1 | 0 | 3.39 |
| adversarial_negative_multiply | adversarial_math | ✅ PASS | 2 | 1 | 5.70 |
| adversarial_decimal_division | adversarial_math | ✅ PASS | 2 | 1 | 5.38 |
| adversarial_percentage | adversarial_math | ✅ PASS | 2 | 1 | 6.02 |
| adversarial_false_premise | adversarial_reasoning | ✅ PASS | 2 | 1 | 17.72 |
| adversarial_chained_search_calc | adversarial_reasoning | ✅ PASS | 2 | 2 | 20.44 |
| adversarial_division_by_zero | adversarial_error_handling | ✅ PASS | 2 | 1 | 7.70 |
| adversarial_ambiguous_city | adversarial_ambiguity | ✅ PASS | 2 | 1 | 7.09 |

## Category Breakdown

- **adversarial_ambiguity**: 1/1 passed
- **adversarial_error_handling**: 1/1 passed
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