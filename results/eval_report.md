# Agent Benchmark Report

**Overall: 22/23 passed (95.7%)**

| Task ID | Category | Result | Steps | Tool Calls | Compute Latency (s) | Retry Wait (s) |
|---|---|---|---|---|---|---|
| calc_basic_1 | math | ✅ PASS | 2 | 1 | 1.48 | 0.0 |
| calc_basic_2 | math | ✅ PASS | 2 | 1 | 1.75 | 0.0 |
| calc_order_of_ops | math | ✅ PASS | 2 | 1 | 1.47 | 0.0 |
| wordcount_basic | text | ✅ PASS | 2 | 1 | 1.58 | 0.0 |
| weather_basic_1 | weather | ✅ PASS | 2 | 1 | 3.19 | 0.0 |
| weather_basic_2 | weather | ✅ PASS | 2 | 1 | 2.95 | 0.0 |
| search_basic_1 | search | ✅ PASS | 2 | 1 | 2.55 | 0.0 |
| search_basic_2 | search | ✅ PASS | 2 | 1 | 2.34 | 0.0 |
| multi_weather_calc | multi_tool | ✅ PASS | 2 | 2 | 3.39 | 17.3 |
| multi_wordcount_calc | multi_tool | ✅ PASS | 3 | 2 | 2.36 | 0.0 |
| multi_search_weather | multi_tool | ✅ PASS | 2 | 2 | 4.16 | 0.0 |
| multi_three_tools | multi_tool | ✅ PASS | 2 | 3 | 4.55 | 0.0 |
| chain_weather_then_calc | chained_reasoning | ✅ PASS | 3 | 2 | 3.78 | 0.0 |
| time_sensitive_1 | time_sensitive | ✅ PASS | 4 | 3 | 6.67 | 0.0 |
| no_tool_needed_1 | no_tool_expected | ✅ PASS | 1 | 0 | 0.69 | 0.0 |
| no_tool_needed_2 | no_tool_expected | ✅ PASS | 1 | 0 | 1.06 | 26.0 |
| adversarial_negative_multiply | adversarial_math | ✅ PASS | 2 | 1 | 1.39 | 0.0 |
| adversarial_decimal_division | adversarial_math | ✅ PASS | 2 | 1 | 1.84 | 0.0 |
| adversarial_percentage | adversarial_math | ✅ PASS | 2 | 1 | 1.49 | 0.0 |
| adversarial_false_premise | adversarial_reasoning | ✅ PASS | 3 | 2 | 6.11 | 0.0 |
| adversarial_chained_search_calc | adversarial_reasoning | ✅ PASS | 3 | 2 | 3.70 | 0.0 |
| adversarial_division_by_zero | adversarial_error_handling | ✅ PASS | 2 | 1 | 1.98 | 0.0 |
| adversarial_ambiguous_city | adversarial_ambiguity | ❌ FAIL | 2 | 1 | 2.70 | 0.0 |