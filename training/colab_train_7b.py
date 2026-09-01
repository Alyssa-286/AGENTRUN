# colab_train_7b.py
#
# PASTE INTO COLAB CELLS (split at "# ===== CELL BREAK =====").
# Requires GPU runtime: Runtime -> Change runtime type -> T4 GPU.
#
# Before running: upload training_data.jsonl into Colab's file browser
# (drag it in — same file generate_training_data.py produced on your laptop).
#
# This trains a LoRA adapter on Qwen2.5-7B-Instruct using QLoRA (4-bit
# base model + LoRA adapters) — the standard way to fine-tune a 7B model
# on a single free-tier T4's 16GB VRAM. Saves to /content/agentlab_qwen_lora_7b
# (deliberately a DIFFERENT folder than your working 0.5B adapter — that
# one stays untouched as a fallback if anything here goes wrong).

# ===== CELL BREAK =====
!pip install -q transformers peft accelerate bitsandbytes trl datasets fastapi uvicorn pyngrok nest_asyncio

# ===== CELL BREAK =====
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/content/agentlab_qwen_lora_7b"

# THIS MUST MATCH the tool schema used at inference time (colab_server.py /
# ft_agent.py), or the model will learn a tool-calling format that doesn't
# line up with what it's actually asked to use later. Kept identical to
# mcp_server.py's tool definitions on purpose.
TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "A math expression to evaluate"}},
            "required": ["expression"]},
    }},
    {"type": "function", "function": {
        "name": "word_count",
        "description": "Count the number of words in a piece of text.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "The text to count words in"}},
            "required": ["text"]},
    }},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the live web for current information. Returns structured JSON "
                        "results with title, url, snippet, and source for each hit.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The search query"}},
            "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current real-time weather for a city (temperature in Celsius, condition).",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"}},
            "required": ["city"]},
    }},
]

# ===== CELL BREAK =====
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model in 4-bit (QLoRA)...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=quant_config, device_map="auto"
)

lora_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention layers — standard for LoRA
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # sanity check: should be a small fraction of total params

# ===== CELL BREAK =====
print("Loading training_data.jsonl...")
raw_examples = []
with open("/content/training_data.jsonl") as f:
    for line in f:
        raw_examples.append(json.loads(line))
print(f"Loaded {len(raw_examples)} trajectories.")


def format_example(example: dict) -> dict:
    """Render one trajectory's messages through Qwen's chat template WITH
    the tool schema, so the model sees the exact same tool-calling format
    (<tool_call> tags etc.) it will be expected to produce at inference."""
    text = tokenizer.apply_chat_template(
        example["messages"], tools=TOOLS_SCHEMA, tokenize=False,
    )
    return {"text": text}


dataset = Dataset.from_list(raw_examples).map(format_example)
print("Example formatted training text (first sample):")
print(dataset[0]["text"][:800], "...\n")

# ===== CELL BREAK =====
# Loss masking: we only want the model to learn to PREDICT assistant turns,
# not to reproduce the system prompt / user questions / tool results
# verbatim (those are context, not something we want it generating). This
# collator masks everything except text after the assistant turn marker.
# NOTE: the exact marker string depends on Qwen's chat template — this is
# the standard ChatML-style marker Qwen2.5 uses. If training loss looks
# wrong (near-zero from step 1, or the model produces garbage after
# training), the first thing to check is whether this string actually
# appears in the formatted text printed above.
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,          # small dataset (35 examples) — a few epochs is reasonable,
                                  # more risks overfitting on so little data
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
    dataset_text_field="text",
    max_seq_length=1024,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)

print("Starting training...")
trainer.train()

print(f"Saving adapter to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done. Adapter saved.")

# ===== CELL BREAK =====
# Quick sanity check BEFORE spending time wiring up serving — generate one
# response and eyeball whether it produces a sensible <tool_call> for an
# obviously tool-requiring prompt.
test_messages = [
    {"role": "system", "content": "You are a helpful AI agent with access to tools: calculator, "
                                   "word_count, web_search, and get_weather. Use tools when you need "
                                   "real information or exact computation."},
    {"role": "user", "content": "What is 84 times 19?"},
]
prompt = tokenizer.apply_chat_template(
    test_messages, tools=TOOLS_SCHEMA, add_generation_prompt=True, tokenize=False,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print("=== SANITY CHECK OUTPUT ===")
print(generated)
print("===========================")
print("Look for a <tool_call>{\"name\": \"calculator\", ...}</tool_call> block above.")
print("If you don't see one, STOP and paste this output back before proceeding to serving.")
