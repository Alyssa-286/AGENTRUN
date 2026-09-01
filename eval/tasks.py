"""
eval/tasks.py — the benchmark: a set of realistic prompts, each with a
MACHINE-CHECKABLE definition of what counts as success.

Phase 4.1 Addition:
  - 23 total tasks including 7 adversarial tests probing for known agent weaknesses:
    * Negative arithmetic and decimal precision
    * Percentage reasoning & formula translation
    * False-premise resistance (fictional entities like Wakanda)
    * Chained search -> calculation
    * Expected tool error handling (division by zero)
    * Ambiguous entity resolution (Springfield)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    prompt: str
    required_tools: set[str] = field(default_factory=set)
    answer_contains: list[str] = field(default_factory=list)  # ALL must appear (case-insensitive)
    answer_contains_any: list[str] = field(default_factory=list)  # AT LEAST ONE must appear
    numeric_check: tuple[float, float] | None = None  # (expected_value, tolerance)
    expect_tool_error: bool = False  # True = tool error is EXPECTED and required
    category: str = "general"  # groups tasks for reporting


TASKS: list[Task] = [
    # --- Single-tool tasks: baseline sanity checks ---
    Task(
        id="calc_basic_1",
        prompt="What is 128 * 47?",
        required_tools={"calculator"},
        numeric_check=(6016, 0.5),
        category="math",
    ),
    Task(
        id="calc_basic_2",
        prompt="What is 900 divided by 12?",
        required_tools={"calculator"},
        numeric_check=(75, 0.5),
        category="math",
    ),
    Task(
        id="calc_order_of_ops",
        prompt="Calculate (15 + 5) * 3 - 10",
        required_tools={"calculator"},
        numeric_check=(50, 0.5),
        category="math",
    ),
    Task(
        id="wordcount_basic",
        prompt="How many words are in this sentence: 'the quick brown fox jumps over the lazy dog'?",
        required_tools={"word_count"},
        numeric_check=(9, 0.5),
        category="text",
    ),
    Task(
        id="weather_basic_1",
        prompt="What's the current weather in Tokyo?",
        required_tools={"get_weather"},
        category="weather",
    ),
    Task(
        id="weather_basic_2",
        prompt="Is it raining in London right now?",
        required_tools={"get_weather"},
        category="weather",
    ),
    Task(
        id="search_basic_1",
        prompt="Search the web for the current CEO of OpenAI.",
        required_tools={"web_search"},
        category="search",
    ),
    Task(
        id="search_basic_2",
        prompt="What is the capital of Australia? Search the web to confirm.",
        required_tools={"web_search"},
        answer_contains=["canberra"],
        category="search",
    ),

    # --- Multi-tool tasks: force chaining / parallel tool use ---
    Task(
        id="multi_weather_calc",
        prompt="What's the weather in Delhi, and separately, what is 45 times 12?",
        required_tools={"get_weather", "calculator"},
        numeric_check=(540, 0.5),
        category="multi_tool",
    ),
    Task(
        id="multi_wordcount_calc",
        prompt="Count the words in 'artificial intelligence is transforming the world today' "
               "then multiply that count by 100.",
        required_tools={"word_count", "calculator"},
        numeric_check=(700, 0.5),
        category="multi_tool",
    ),
    Task(
        id="multi_search_weather",
        prompt="Search for who the current Prime Minister of the UK is, and also tell me "
               "the weather in Mumbai.",
        required_tools={"web_search", "get_weather"},
        category="multi_tool",
    ),
    Task(
        id="multi_three_tools",
        prompt="What's the weather in Bangalore, what is 67 * 4567, and search the web "
               "for the latest iPhone model.",
        required_tools={"get_weather", "calculator", "web_search"},
        numeric_check=(305989, 0.5),
        category="multi_tool",
    ),

    # --- Chained reasoning: output of one step must feed into the next ---
    Task(
        id="chain_weather_then_calc",
        prompt="Get the temperature in Mumbai in Celsius, then convert it to Fahrenheit "
               "using the formula F = C * 9/5 + 32. Use the calculator tool for the conversion.",
        required_tools={"get_weather", "calculator"},
        category="chained_reasoning",
    ),

    # --- Ambiguity / time-sensitivity ---
    Task(
        id="time_sensitive_1",
        prompt="Who won the most recent IPL season? Make sure your answer is current.",
        required_tools={"web_search"},
        category="time_sensitive",
    ),

    # --- Should NOT need a tool at all ---
    Task(
        id="no_tool_needed_1",
        prompt="What is the chemical symbol for gold?",
        required_tools=set(),
        answer_contains=["au"],
        category="no_tool_expected",
    ),
    Task(
        id="no_tool_needed_2",
        prompt="In one sentence, what does the word 'agentic' mean in AI?",
        required_tools=set(),
        category="no_tool_expected",
    ),

    # =======================================================================
    # ADVERSARIAL TASKS (Phase 4.1)
    # =======================================================================
    Task(
        id="adversarial_negative_multiply",
        prompt="What is -18 * 34?",
        required_tools={"calculator"},
        numeric_check=(-612, 0.5),
        category="adversarial_math",
    ),
    Task(
        id="adversarial_decimal_division",
        prompt="What is 17 divided by 8? Give the exact decimal result.",
        required_tools={"calculator"},
        numeric_check=(2.125, 0.01),
        category="adversarial_math",
    ),
    Task(
        id="adversarial_percentage",
        prompt="What is 15% of 340?",
        required_tools={"calculator"},
        numeric_check=(51, 0.5),
        category="adversarial_math",
    ),
    Task(
        id="adversarial_false_premise",
        prompt="Search the web and tell me the current GDP of the fictional country of Wakanda.",
        required_tools={"web_search"},
        answer_contains_any=[
            "fictional", "doesn't exist", "does not exist",
            "not a real country", "not real", "marvel",
        ],
        category="adversarial_reasoning",
    ),
    Task(
        id="adversarial_chained_search_calc",
        prompt="Search for what year the first iPhone was released, then use the "
               "calculator to work out how many years ago that was from the year 2026.",
        required_tools={"web_search", "calculator"},
        numeric_check=(19, 1),  # first iPhone: 2007. 2026 - 2007 = 19
        category="adversarial_reasoning",
    ),
    Task(
        id="adversarial_division_by_zero",
        prompt="What is 50 divided by 0?",
        required_tools={"calculator"},
        expect_tool_error=True,
        answer_contains_any=["undefined", "cannot divide", "not defined", "infinite", "error", "division by zero"],
        category="adversarial_error_handling",
    ),
    Task(
        id="adversarial_ambiguous_city",
        prompt="What's the weather in Springfield?",
        required_tools={"get_weather"},
        category="adversarial_ambiguity",
    ),
]
