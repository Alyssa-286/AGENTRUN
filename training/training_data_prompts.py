"""
training_data_prompts.py — prompts used ONLY to generate fine-tuning data.

CRITICAL RULE: none of these prompts (or close variants) may appear in
eval/tasks.py. If they did, Phase 5.3's "fine-tuned model vs Gemini"
comparison would be testing on memorized training data — an open-book
exam, not a benchmark. Every prompt here is deliberately different from
the eval task set while still covering the same tool surface (calculator,
word_count, web_search, get_weather) so the fine-tuned model learns the
general SKILL of tool use, not the specific eval questions.
"""

TRAINING_PROMPTS: list[str] = [
    # --- calculator: varied phrasing, varied difficulty ---
    "What is 84 times 19?",
    "Add 347 and 528 together.",
    "What's 1000 minus 234?",
    "Divide 144 by 6.",
    "What is 7 to the... actually just compute 3 * 3 * 3 * 3 for me.",
    "If I have 240 rupees and spend 85, how much is left?",
    "What is 55% of 200?",
    "Calculate (8 + 12) * 5.",
    "What's -25 plus 40?",
    "What is 999 divided by 3?",

    # --- word_count ---
    "Count the words in: 'machine learning models require large datasets to train effectively'",
    "How many words are in the sentence 'I love building AI agents from scratch'?",
    "Count words: 'the weather today is sunny with a chance of rain later'",

    # --- get_weather ---
    "Tell me the weather in Chennai.",
    "Is it hot in Dubai right now?",
    "What's the temperature in New York City?",
    "Give me the current weather conditions in Singapore.",
    "How's the weather looking in Sydney today?",

    # --- web_search ---
    "Search for who invented the World Wide Web.",
    "Find out the current population of Japan.",
    "Search the web for the tallest mountain in the world.",
    "What's the latest news about SpaceX? Search for it.",
    "Look up who won the most recent Nobel Prize in Physics.",

    # --- multi-tool: chaining and parallel use ---
    "What's the weather in Berlin, and also compute 33 * 12?",
    "Search for the boiling point of water in Celsius, then convert it to Fahrenheit using the calculator.",
    "Count the words in 'artificial general intelligence' then multiply that count by 50.",
    "Tell me the weather in Cairo and also search for the currency used in Egypt.",
    "What is 18 * 24, and separately, search for the capital of Canada.",
    "Search for the average human body temperature in Celsius, then convert to Fahrenheit.",
    "What's the weather in Seoul, what is 500 / 25, and search for the official language of South Korea.",

    # --- no-tool-needed: general knowledge, tests the model doesn't over-call tools ---
    "What is the chemical formula for water?",
    "In one sentence, explain what a neural network is.",
    "What does 'RAG' stand for in the context of AI?",
    "Name the primary colors.",
    "What is the speed of light in a vacuum, approximately?",

    # --- error/edge handling ---
    "What is 10 divided by 0?",
    "What's the weather in a made-up city called Zylorath?",
]
