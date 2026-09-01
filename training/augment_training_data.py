"""
augment_training_data.py — Adds 10 targeted training trajectories to
fix the audit's identified gaps:

  - word_count -> calculator (3 examples, awkward values)
  - get_weather -> calculator (3 examples, fractional temps)
  - 3-tool chain + explicit final synthesis (2 examples)
  - calculator error / zero division (2 examples)
  - false-premise correction (1 example)

Disjoint from the 23 eval tasks (every new prompt checked for lexical overlap).
Format matches training_data.jsonl exactly (system prompt, role=tool messages,
plain JSON tool_calls structure).

Inputs/outputs are relative to this file's location (the training/ directory).
"""
import json
import os
import sys
from pathlib import Path

# Paths are relative to this script's location (training/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = (
    "You are a helpful AI agent with access to tools: calculator, word_count, "
    "web_search, and get_weather. Use tools when you need real information or "
    "exact computation. Respond directly when you already know the answer."
)


def word_count_then_calc(prompt: str, sentence: str, multiplier: int, multiplier_rationale: str) -> dict:
    """word_count -> calculator: count words, then multiply by an awkward number."""
    word_count = len(sentence.split())
    product = word_count * multiplier
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "word_count", "arguments": {"text": sentence}},
            ]},
            {"role": "tool", "name": "word_count", "content": str(word_count)},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "calculator", "arguments": {"expression": f"{word_count} * {multiplier}"}},
            ]},
            {"role": "tool", "name": "calculator", "content": str(product)},
            {"role": "assistant", "content": (
                f"The sentence has {word_count} words. {multiplier_rationale} "
                f"{word_count} * {multiplier} = {product}."
            )},
        ]
    }


def weather_then_calc(prompt: str, city: str, temperature_c: float, formula: str) -> dict:
    """get_weather -> calculator: convert C to F using the calculator (awkward temps)."""
    expr = formula.replace("C", str(temperature_c))
    # Evaluate safely — these are pre-vetted training values.
    result = eval(expr)  # nosec — controlled training data
    formatted_temp = f"{temperature_c:.1f}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "get_weather", "arguments": {"city": city}},
            ]},
            {"role": "tool", "name": "get_weather", "content": (
                f'{{"city": "{city}", "temperature_c": {temperature_c}, '
                f'"condition": "partly cloudy", "source": "open-meteo"}}'
            )},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "calculator", "arguments": {"expression": formula.replace("C", str(temperature_c))}},
            ]},
            {"role": "tool", "name": "calculator", "content": f"{result:.1f}"},
            {"role": "assistant", "content": (
                f"The current temperature in {city} is {formatted_temp}°C. "
                f"Using the formula F = C * 9/5 + 32, that converts to {result:.1f}°F."
            )},
        ]
    }


def three_tool_chain(prompt: str, weather_city: str, weather_temp_c: float,
                      calc_expression: str, search_query: str, search_answer: str) -> dict:
    """Three-tool chain with explicit final synthesis."""
    calc_result = eval(calc_expression)  # nosec
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "get_weather", "arguments": {"city": weather_city}},
            ]},
            {"role": "tool", "name": "get_weather", "content": (
                f'{{"city": "{weather_city}", "temperature_c": {weather_temp_c}, '
                f'"condition": "clear", "source": "open-meteo"}}'
            )},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "calculator", "arguments": {"expression": calc_expression}},
            ]},
            {"role": "tool", "name": "calculator", "content": str(calc_result)},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "web_search", "arguments": {"query": search_query}},
            ]},
            {"role": "tool", "name": "web_search", "content": search_answer},
            {"role": "assistant", "content": (
                f"Summary:\n1. Weather in {weather_city}: {weather_temp_c}°C and clear.\n"
                f"2. Calculation: {calc_expression} = {calc_result}.\n"
                f"3. {search_query.capitalize()}: {search_answer}"
            )},
        ]
    }


def calc_error(prompt: str, expression: str, error_msg: str, explanation: str) -> dict:
    """Calculator error / division by zero pattern."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "calculator", "arguments": {"expression": expression}},
            ]},
            {"role": "tool", "name": "calculator", "content": error_msg, "is_error": True},
            {"role": "assistant", "content": explanation},
        ]
    }


def false_premise_correction(prompt: str, search_query: str, search_result: str,
                              final_answer: str) -> dict:
    """False premise: user asks about fictional entity; model searches and explains."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "web_search", "arguments": {"query": search_query}},
            ]},
            {"role": "tool", "name": "web_search", "content": search_result},
            {"role": "assistant", "content": final_answer},
        ]
    }


# ── 10 new trajectories ──────────────────────────────────────────────────────
NEW_EXAMPLES = [
    # ── word_count -> calculator (3) ── awkward multipliers, varied sentences
    word_count_then_calc(
        "Count the words in 'transformers revolutionized natural language processing in 2017' "
        "and then multiply that count by 37.",
        "transformers revolutionized natural language processing in 2017",
        37,
        "37 times the word count equals",
    ),
    word_count_then_calc(
        "Tell me how many words are in 'reinforcement learning agents explore environments to maximize cumulative reward' "
        "and then compute that count times 19.",
        "reinforcement learning agents explore environments to maximize cumulative reward",
        19,
        "19 multiplied by the word count is",
    ),
    word_count_then_calc(
        "Count words in 'vector databases store high-dimensional embeddings for similarity search' "
        "then multiply by 23.",
        "vector databases store high-dimensional embeddings for similarity search",
        23,
        "23 times the word count is",
    ),

    # ── get_weather -> calculator (3) ── fractional, awkward temperatures
    weather_then_calc(
        "What's the temperature in Reykjavik, and then convert it from Celsius to Fahrenheit "
        "using the calculator?",
        "Reykjavik", 4.7, "4.7 * 9/5 + 32",
    ),
    weather_then_calc(
        "Check the current weather in Buenos Aires and use the calculator to convert the "
        "Celsius temperature to Fahrenheit.",
        "Buenos Aires", 28.3, "28.3 * 9/5 + 32",
    ),
    weather_then_calc(
        "Look up the temperature in Helsinki right now, then use the calculator to convert it "
        "from Celsius to Fahrenheit.",
        "Helsinki", -3.4, "-3.4 * 9/5 + 32",
    ),

    # ── 3-tool chains with explicit finalization (2) ──
    three_tool_chain(
        "What's the weather in Lisbon, calculate 89 times 47, and search the web for what "
        "language they speak in Portugal.",
        "Lisbon", 19.8, "89 * 47",
        "what language do they speak in Portugal",
        "Portuguese is the official language of Portugal, spoken by the entire population.",
    ),
    three_tool_chain(
        "Find the current temperature in Marrakech, compute 1234 divided by 7, and search "
        "for what currency Morocco uses.",
        "Marrakech", 26.5, "1234 / 7",
        "currency used in Morocco",
        "The Moroccan Dirham (MAD) is the official currency of Morocco.",
    ),

    # ── calculator error / zero division (2) ──
    calc_error(
        "What is 25 divided by 0?",
        "25 / 0",
        "Error: division by zero is undefined.",
        "Division by zero is mathematically undefined. The calculator returned an error "
        "because no number can be the result of dividing 25 by 0.",
    ),
    calc_error(
        "Compute 0 divided by 0.",
        "0 / 0",
        "Error: 0/0 is indeterminate.",
        "Zero divided by zero is an indeterminate form, not a defined number. The "
        "calculator reported an error because this expression has no single value.",
    ),

    # ── false premise correction (1) ──
    false_premise_correction(
        "What is the population of the country of El Dorado?",
        "El Dorado country population",
        'No results found. "El Dorado" is a legendary city of gold from Spanish colonial '
        "mythology, not a real country. There is no modern nation by that name.",
        'I searched for El Dorado, but it is not a real country — it is a legendary city '
        "from Spanish colonial folklore, often described as a city of gold. The question is "
        "based on a false premise.",
    ),
]


def load_eval_prompts() -> set[str]:
    """Return set of lowercase eval task prompt substrings for overlap check."""
    sys.path.insert(0, ".")
    from eval.tasks import TASKS  # type: ignore[import]
    prompts = set()
    for t in TASKS:
        prompts.add(t.prompt.lower().strip())
    return prompts


def validate_disjoint(new_examples, eval_prompts):
    """Check no new example has high lexical overlap with any eval task."""
    rejected = []
    for i, ex in enumerate(new_examples):
        new_prompt = ex["messages"][1]["content"].lower()
        for ep in eval_prompts:
            # Check if any eval task prompt is a substring of the new prompt
            # or shares a high-overlap phrase.
            # Use a simple shared-noun check: split on whitespace, compare token sets.
            new_tokens = set(new_prompt.split())
            eval_tokens = set(ep.split())
            # Remove common stop words
            stop = {"a", "an", "the", "is", "in", "of", "to", "and", "or",
                    "for", "with", "on", "at", "by", "what", "how", "?"}
            new_meaningful = new_tokens - stop
            eval_meaningful = eval_tokens - stop
            if new_meaningful and eval_meaningful:
                overlap = len(new_meaningful & eval_meaningful) / len(eval_meaningful)
                # If >70% of eval task tokens appear in the new prompt, reject
                if overlap > 0.7 and len(eval_meaningful) >= 3:
                    rejected.append((i, new_prompt[:80], ep[:80], overlap))
    return rejected


def main():
    # Load existing training data
    existing = []
    with open(os.path.join(_SCRIPT_DIR, "training_data.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            existing.append(json.loads(line))

    print(f"Existing training trajectories: {len(existing)}")

    # Disjoint check
    eval_prompts = load_eval_prompts()
    rejected = validate_disjoint(NEW_EXAMPLES, eval_prompts)
    if rejected:
        print("\n--- REJECTED examples (overlap with eval) ---")
        for r in rejected:
            print(f"  #{r[0]}: new={r[1]!r} eval={r[2]!r} overlap={r[3]:.0%}")
        return

    # Append
    augmented = existing + NEW_EXAMPLES
    with open(os.path.join(_SCRIPT_DIR, "training_data_augmented.jsonl"), "w", encoding="utf-8") as f:
        for ex in augmented:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Also write a prompts file for the augmented set
    new_prompts = [ex["messages"][1]["content"] for ex in NEW_EXAMPLES]
    with open("training_data_prompts_augmented.py", "w", encoding="utf-8") as f:
        f.write('"""\ntraining_data_prompts_augmented.py — new prompts added to the training set.\n\n'
                'These are the 10 additional prompts in training_data_augmented.jsonl.\n"""\n\n')
        f.write("NEW_TRAINING_PROMPTS: list[str] = [\n")
        for p in new_prompts:
            f.write(f"    {p!r},\n")
        f.write("]\n")

    # Validation report
    print(f"\nNew trajectories generated:  {len(NEW_EXAMPLES)}")
    print(f"Total augmented dataset:     {len(augmented)}")
    print(f"Rejections (eval overlap):   {len(rejected)}")
    print()

    # Count chain types
    def assistant_tool_seqs(ex):
        return [[tc["name"] for tc in m.get("tool_calls", [])]
                for m in ex["messages"] if m["role"] == "assistant"]

    chain_counts = {
        "word_count -> calculator": 0,
        "get_weather -> calculator": 0,
        "web_search -> calculator": 0,
        "three_tool": 0,
        "tool_error (calculator)": 0,
        "false_premise_correction": 0,
        "single_tool": 0,
    }
    for ex in NEW_EXAMPLES:
        seqs = assistant_tool_seqs(ex)
        flat = [t for s in seqs for t in s]
        if len(flat) == 0:
            continue
        if any(m.get("is_error") for m in ex["messages"]):
            chain_counts["tool_error (calculator)"] += 1
        elif "El Dorado" in ex["messages"][1]["content"]:
            chain_counts["false_premise_correction"] += 1
        elif len(flat) >= 3:
            chain_counts["three_tool"] += 1
        elif ["word_count", "calculator"] == flat:
            chain_counts["word_count -> calculator"] += 1
        elif ["get_weather", "calculator"] == flat:
            chain_counts["get_weather -> calculator"] += 1
        elif ["web_search", "calculator"] == flat:
            chain_counts["web_search -> calculator"] += 1
        else:
            chain_counts["single_tool"] += 1

    print("Chain-type breakdown (new examples):")
    for k, v in chain_counts.items():
        print(f"  {k:35s} {v}")
    print()

    # Final disjoint confirmation
    print("Disjoint check: PASS (no new prompt has >70% token overlap with any eval task)")

    print(f"\nFiles written:")
    print(f"  training_data_augmented.jsonl     ({len(augmented)} trajectories)")
    print(f"  training_data_prompts_augmented.py")


if __name__ == "__main__":
    main()
