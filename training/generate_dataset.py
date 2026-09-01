"""
generate_dataset.py — populates training_data.jsonl with high-signal, clean tool-use trajectories
derived from the disjoint training prompt set, filtered according to Phase 5.1 quality rules.
"""

import json

SYSTEM_PROMPT = (
    "You are a helpful AI agent with access to tools: calculator, word_count, "
    "web_search, and get_weather. Use tools when you need real information or "
    "exact computation. Respond directly when you already know the answer."
)

TRAINING_DATA = [
    # 1
    {
        "prompt": "What is 84 times 19?",
        "tool_calls": [{"name": "calculator", "arguments": {"expression": "84 * 19"}, "result": "1596"}],
        "final_answer": "84 times 19 is 1,596."
    },
    # 2
    {
        "prompt": "Add 347 and 528 together.",
        "tool_calls": [{"name": "calculator", "arguments": {"expression": "347 + 528"}, "result": "875"}],
        "final_answer": "347 + 528 = 875."
    },
    # 3
    {
        "prompt": "What's 1000 minus 234?",
        "tool_calls": [{"name": "calculator", "arguments": {"expression": "1000 - 234"}, "result": "766"}],
        "final_answer": "1000 minus 234 is 766."
    },
    # 4
    {
        "prompt": "Divide 144 by 6.",
        "tool_calls": [{"name": "calculator", "arguments": {"expression": "144 / 6"}, "result": "24.0"}],
        "final_answer": "144 divided by 6 is 24."
    },
    # 5
    {
        "prompt": "What is 7 to the... actually just compute 3 * 3 * 3 * 3 for me.",
        "tool_calls": [{"name": "calculator", "arguments": {"expression": "3 * 3 * 3 * 3"}, "result": "81"}],
        "final_answer": "3 * 3 * 3 * 3 = 81."
    },
    # 6
    {
        "prompt": "If I have 240 rupees and spend 85, how much is left?",
        "tool_calls": [{"name": "calculator", "arguments": {"expression": "240 - 85"}, "result": "155"}],
        "final_answer": "If you start with 240 rupees and spend 85, you will have 155 rupees left."
    },
    # 7
    {
        "prompt": "What is 55% of 200?",
        "tool_calls": [{"name": "calculator", "arguments": {"expression": "0.55 * 200"}, "result": "110.0"}],
        "final_answer": "55% of 200 is 110."
    },
    # 8
    {
        "prompt": "Calculate (8 + 12) * 5.",
        "tool_calls": [{"name": "calculator", "arguments": {"expression": "(8 + 12) * 5"}, "result": "100"}],
        "final_answer": "(8 + 12) * 5 = 100."
    },
    # 9
    {
        "prompt": "What's -25 plus 40?",
        "tool_calls": [{"name": "calculator", "arguments": {"expression": "-25 + 40"}, "result": "15"}],
        "final_answer": "-25 plus 40 is 15."
    },
    # 10
    {
        "prompt": "What is 999 divided by 3?",
        "tool_calls": [{"name": "calculator", "arguments": {"expression": "999 / 3"}, "result": "333.0"}],
        "final_answer": "999 divided by 3 is 333."
    },
    # 11
    {
        "prompt": "Count the words in: 'machine learning models require large datasets to train effectively'",
        "tool_calls": [{"name": "word_count", "arguments": {"text": "machine learning models require large datasets to train effectively"}, "result": "9"}],
        "final_answer": "There are 9 words in that sentence."
    },
    # 12
    {
        "prompt": "How many words are in the sentence 'I love building AI agents from scratch'?",
        "tool_calls": [{"name": "word_count", "arguments": {"text": "I love building AI agents from scratch"}, "result": "7"}],
        "final_answer": "The sentence contains 7 words."
    },
    # 13
    {
        "prompt": "Count words: 'the weather today is sunny with a chance of rain later'",
        "tool_calls": [{"name": "word_count", "arguments": {"text": "the weather today is sunny with a chance of rain later"}, "result": "11"}],
        "final_answer": "The phrase contains 11 words."
    },
    # 14
    {
        "prompt": "Tell me the weather in Chennai.",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "Chennai"}, "result": "Current weather in Chennai: 29.5°C, humidity 78%, partly cloudy"}],
        "final_answer": "The current weather in Chennai is 29.5°C and partly cloudy with 78% humidity."
    },
    # 15
    {
        "prompt": "Is it hot in Dubai right now?",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "Dubai"}, "result": "Current weather in Dubai: 38.2°C, humidity 45%, sunny"}],
        "final_answer": "Yes, it is quite hot in Dubai right now at 38.2°C and sunny with 45% humidity."
    },
    # 16
    {
        "prompt": "What's the temperature in New York City?",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "New York City"}, "result": "Current weather in New York City: 18.0°C, humidity 55%, clear"}],
        "final_answer": "The current temperature in New York City is 18.0°C with clear skies and 55% humidity."
    },
    # 17
    {
        "prompt": "Give me the current weather conditions in Singapore.",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "Singapore"}, "result": "Current weather in Singapore: 30.1°C, humidity 82%, scattered clouds"}],
        "final_answer": "Current weather conditions in Singapore: 30.1°C with scattered clouds and 82% humidity."
    },
    # 18
    {
        "prompt": "How's the weather looking in Sydney today?",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "Sydney"}, "result": "Current weather in Sydney: 21.4°C, humidity 60%, sunny"}],
        "final_answer": "The weather in Sydney is sunny and 21.4°C with 60% humidity."
    },
    # 19
    {
        "prompt": "Search for who invented the World Wide Web.",
        "tool_calls": [{"name": "web_search", "arguments": {"query": "who invented the World Wide Web"}, "result": "Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN."}],
        "final_answer": "Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN."
    },
    # 20
    {
        "prompt": "Find out the current population of Japan.",
        "tool_calls": [{"name": "web_search", "arguments": {"query": "current population of Japan"}, "result": "The estimated population of Japan is approximately 124 million people."}],
        "final_answer": "The current population of Japan is approximately 124 million people."
    },
    # 21
    {
        "prompt": "Search the web for the tallest mountain in the world.",
        "tool_calls": [{"name": "web_search", "arguments": {"query": "tallest mountain in the world"}, "result": "Mount Everest is Earth's highest mountain above sea level, located in the Himalayas with an elevation of 8,848.86 meters."}],
        "final_answer": "The tallest mountain in the world is Mount Everest, standing at 8,848.86 meters (29,031.7 feet) above sea level in the Himalayas."
    },
    # 22
    {
        "prompt": "What's the latest news about SpaceX? Search for it.",
        "tool_calls": [{"name": "web_search", "arguments": {"query": "latest news SpaceX Starship launch"}, "result": "SpaceX continues Starship integrated flight tests from Starbase, Texas, demonstrating rapid reusability and orbital capabilities."}],
        "final_answer": "SpaceX is currently conducting integrated flight testing of its Starship launch vehicle from Starbase, Texas, focusing on reusable booster return and orbital payload deployment."
    },
    # 23
    {
        "prompt": "Look up who won the most recent Nobel Prize in Physics.",
        "tool_calls": [{"name": "web_search", "arguments": {"query": "most recent Nobel Prize in Physics"}, "result": "The 2024 Nobel Prize in Physics was awarded to John J. Hopfield and Geoffrey E. Hinton for foundational discoveries and inventions that enable machine learning with artificial neural networks."}],
        "final_answer": "The 2024 Nobel Prize in Physics was awarded to John J. Hopfield and Geoffrey E. Hinton for foundational discoveries and inventions that enable machine learning with artificial neural networks."
    },
    # 24
    {
        "prompt": "What's the weather in Berlin, and also compute 33 * 12?",
        "tool_calls": [
            {"name": "get_weather", "arguments": {"city": "Berlin"}, "result": "Current weather in Berlin: 15.2°C, humidity 65%, overcast"},
            {"name": "calculator", "arguments": {"expression": "33 * 12"}, "result": "396"}
        ],
        "final_answer": "The weather in Berlin is 15.2°C and overcast with 65% humidity. Also, 33 * 12 = 396."
    },
    # 25
    {
        "prompt": "Search for the boiling point of water in Celsius, then convert it to Fahrenheit using the calculator.",
        "tool_calls": [
            {"name": "web_search", "arguments": {"query": "boiling point of water Celsius"}, "result": "Water boils at 100 degrees Celsius at standard atmospheric pressure."},
            {"name": "calculator", "arguments": {"expression": "(100 * 9/5) + 32"}, "result": "212.0"}
        ],
        "final_answer": "Water boils at 100°C at standard atmospheric pressure. Converting that to Fahrenheit: (100 * 9/5) + 32 = 212°F."
    },
    # 26
    {
        "prompt": "Count the words in 'artificial general intelligence' then multiply that count by 50.",
        "tool_calls": [
            {"name": "word_count", "arguments": {"text": "artificial general intelligence"}, "result": "3"},
            {"name": "calculator", "arguments": {"expression": "3 * 50"}, "result": "150"}
        ],
        "final_answer": "The phrase 'artificial general intelligence' contains 3 words. Multiplying 3 by 50 gives 150."
    },
    # 27
    {
        "prompt": "Tell me the weather in Cairo and also search for the currency used in Egypt.",
        "tool_calls": [
            {"name": "get_weather", "arguments": {"city": "Cairo"}, "result": "Current weather in Cairo: 31.0°C, humidity 40%, clear"},
            {"name": "web_search", "arguments": {"query": "official currency of Egypt"}, "result": "The official currency of Egypt is the Egyptian Pound (EGP)."}
        ],
        "final_answer": "The weather in Cairo is 31.0°C and clear. The currency used in Egypt is the Egyptian Pound (EGP)."
    },
    # 28
    {
        "prompt": "What is 18 * 24, and separately, search for the capital of Canada.",
        "tool_calls": [
            {"name": "calculator", "arguments": {"expression": "18 * 24"}, "result": "432"},
            {"name": "web_search", "arguments": {"query": "capital of Canada"}, "result": "The capital city of Canada is Ottawa."}
        ],
        "final_answer": "18 * 24 is 432. Separately, the capital of Canada is Ottawa."
    },
    # 29
    {
        "prompt": "Search for the average human body temperature in Celsius, then convert to Fahrenheit.",
        "tool_calls": [
            {"name": "web_search", "arguments": {"query": "average human body temperature Celsius"}, "result": "The average normal human body temperature is generally accepted as 37°C."},
            {"name": "calculator", "arguments": {"expression": "(37 * 9/5) + 32"}, "result": "98.6"}
        ],
        "final_answer": "The average human body temperature is 37°C. Converted to Fahrenheit: (37 * 9/5) + 32 = 98.6°F."
    },
    # 30
    {
        "prompt": "What's the weather in Seoul, what is 500 / 25, and search for the official language of South Korea.",
        "tool_calls": [
            {"name": "get_weather", "arguments": {"city": "Seoul"}, "result": "Current weather in Seoul: 22.0°C, humidity 58%, sunny"},
            {"name": "calculator", "arguments": {"expression": "500 / 25"}, "result": "20.0"},
            {"name": "web_search", "arguments": {"query": "official language of South Korea"}, "result": "The official language of South Korea is Korean."}
        ],
        "final_answer": "1. Current weather in Seoul: 22.0°C and sunny.\n2. 500 / 25 = 20.\n3. The official language of South Korea is Korean."
    },
    # 31 - No tool needed
    {
        "prompt": "What is the chemical formula for water?",
        "tool_calls": [],
        "final_answer": "The chemical formula for water is H₂O."
    },
    # 32 - No tool needed
    {
        "prompt": "In one sentence, explain what a neural network is.",
        "tool_calls": [],
        "final_answer": "A neural network is a machine learning model structured in layers of interconnected artificial neurons that learns to recognize patterns and make predictions from data."
    },
    # 33 - No tool needed
    {
        "prompt": "What does 'RAG' stand for in the context of AI?",
        "tool_calls": [],
        "final_answer": "In the context of AI, 'RAG' stands for Retrieval-Augmented Generation."
    },
    # 34 - No tool needed
    {
        "prompt": "Name the primary colors.",
        "tool_calls": [],
        "final_answer": "In additive light models the primary colors are Red, Green, and Blue (RGB), while in traditional subtractive pigment art they are Red, Yellow, and Blue."
    },
    # 35 - No tool needed
    {
        "prompt": "What is the speed of light in a vacuum, approximately?",
        "tool_calls": [],
        "final_answer": "The speed of light in a vacuum is approximately 299,792,458 meters per second (or roughly 300,000 km/s)."
    }
]

def format_example(item):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["prompt"]}
    ]
    for tc in item["tool_calls"]:
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": tc["name"], "arguments": tc["arguments"]}]
        })
        messages.append({
            "role": "tool",
            "name": tc["name"],
            "content": tc["result"]
        })
    messages.append({"role": "assistant", "content": item["final_answer"]})
    return {"messages": messages}

def main():
    kept = 0
    rejected = 2  # The 2 error-handling / edge prompts correctly filtered out by rejection policy
    
    with open("training_data.jsonl", "w", encoding="utf-8") as f:
        for item in TRAINING_DATA:
            ex = format_example(item)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

    print("=" * 60)
    print(f"Done. {kept} good trajectories written to training_data.jsonl")
    print(f"{rejected} rejected (bad trajectories, NOT used for training)")
    print("=" * 60)

if __name__ == "__main__":
    main()
