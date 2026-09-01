"""
training_data_prompts_augmented.py — new prompts added to the training set.

These are the 10 additional prompts in training_data_augmented.jsonl.
"""

NEW_TRAINING_PROMPTS: list[str] = [
    "Count the words in 'transformers revolutionized natural language processing in 2017' and then multiply that count by 37.",
    "Tell me how many words are in 'reinforcement learning agents explore environments to maximize cumulative reward' and then compute that count times 19.",
    "Count words in 'vector databases store high-dimensional embeddings for similarity search' then multiply by 23.",
    "What's the temperature in Reykjavik, and then convert it from Celsius to Fahrenheit using the calculator?",
    'Check the current weather in Buenos Aires and use the calculator to convert the Celsius temperature to Fahrenheit.',
    'Look up the temperature in Helsinki right now, then use the calculator to convert it from Celsius to Fahrenheit.',
    "What's the weather in Lisbon, calculate 89 times 47, and search the web for what language they speak in Portugal.",
    'Find the current temperature in Marrakech, compute 1234 divided by 7, and search for what currency Morocco uses.',
    'What is 25 divided by 0?',
    'Compute 0 divided by 0.',
    'What is the population of the country of El Dorado?',
]
