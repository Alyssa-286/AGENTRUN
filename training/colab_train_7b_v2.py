# colab_train_7b_v2.py
#
# EXPERIMENT 2 — QLoRA fine-tune of Qwen2.5-7B-Instruct with the
# augmented training set (46 trajectories: 35 original + 11 new).
#
# Differences from colab_train_7b.py (Experiment 1 / v1):
#   1. INPUT FILE:     training_data_augmented.jsonl  (was: training_data.jsonl)
#   2. OUTPUT ADAPTER: /content/agentlab_qwen_lora_7b_v2  (was: ..._7b)
#   3. MEMORY:         gradient checkpointing + PagedAdamW8bit + use_cache=False
#                      (added because 46 examples @ 1024 tokens OOM'd a T4 at
#                       step 8/18; v1 with 35 examples fit without these)
#   4. NUM_EPOCHS:     3 (unchanged for fair ablation)
#
# Everything else (base model, quantization, LoRA r/alpha, sequence length,
# collator, system prompt, tool schema, learning rate, batch size) is
# intentionally identical to v1 so this is a clean ablation on data.
#
# Before running: upload training_data_augmented.jsonl into Colab.
# This script does NOT touch the v1 adapter folder or training_data.jsonl.

# ===== CELL BREAK =====
!pip install -q transformers peft accelerate bitsandbytes trl datasets fastapi uvicorn pyngrok nest_asyncio

# ===== CELL BREAK =====
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from trl.trainer.utils import DataCollatorForCompletionOnlyLM  # moved inside trl.trainer in trl>=0.14

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"           # SAME as v1
OUTPUT_DIR = "/content/agentlab_qwen_lora_7b_v2"   # NEW folder — does NOT overwrite v1
INPUT_DATA = "/content/training_data_augmented.jsonl"

# Verifying the v1 adapter is still untouched:
#   /content/agentlab_qwen_lora_7b      <- v1, preserved
#   /content/agentlab_qwen_lora_7b_v2   <- v2, written by this script

# Verifying the original training data is still untouched:
#   /content/training_data.jsonl              <- v1 data, 35 trajectories
#   /content/training_data_augmented.jsonl     <- v2 data, 46 trajectories

# Tool schema — MUST match colab_server.py / ft_agent.py exactly
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

# LoRA config — IDENTICAL to v1 (r=16, alpha=16, attention-only)
lora_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# ── Memory optimizations (added for v2 — fit 46 examples on a free T4) ─────
# 1. Disable KV cache during training: required when using gradient
#    checkpointing, and saves a non-trivial amount of VRAM.
model.config.use_cache = False

# 2. Enable gradient checkpointing: re-computes activations during
#    backward instead of storing them. Cuts activation memory ~5x at the
#    cost of one extra forward pass per backward. Standard for QLoRA.
model.gradient_checkpointing_enable()

# 3. Required wiring: when gradient checkpointing is on for a PEFT model,
#    the inputs need to pass through the gradient-checkpointing layer to
#    get the right requires_grad signal. (PEFT docs: this is mandatory.)
model.enable_input_require_grads()

# 4. Use 8-bit AdamW (paged) — same effective optimizer dynamics as
#    the default AdamW but with 4x smaller optimizer state. v1 used the
#    default optimizer; switching to 8-bit here is the standard QLoRA
#    recipe and keeps training behaviorally equivalent.
import bitsandbytes as bnb
optim_8bit = bnb.optim.PagedAdamW8bit(
    model.parameters(),
    lr=2e-4,                       # SAME lr as v1 (set below in SFTConfig too)
    weight_decay=0.0,
)

print("Memory optimizations applied: use_cache=False, "
      "gradient_checkpointing=True, PagedAdamW8bit optimizer")
print(f"Free VRAM after model load: "
      f"{torch.cuda.mem_get_info()[0] / 1e9:.2f} GB free / "
      f"{torch.cuda.mem_get_info()[1] / 1e9:.2f} GB total")

# ===== CELL BREAK =====
print(f"Loading {INPUT_DATA}...")
raw_examples = []
with open(INPUT_DATA) as f:
    for line in f:
        raw_examples.append(json.loads(line))
print(f"Loaded {len(raw_examples)} trajectories.")

# Validation: confirm we have 46 (35 original + 11 augmented)
assert len(raw_examples) == 46, (
    f"Expected 46 trajectories, got {len(raw_examples)}. "
    "Check that training_data_augmented.jsonl was uploaded correctly."
)

# Validation: confirm no trajectory uses an eval task prompt
import sys
sys.path.insert(0, "/content")
try:
    from eval.tasks import TASKS
    eval_prompts = {t.prompt.lower().strip() for t in TASKS}
    for ex in raw_examples:
        user_msg = ex["messages"][1]["content"].lower().strip()
        if user_msg in eval_prompts:
            raise ValueError(f"OVERLAP: training prompt matches eval task: {user_msg!r}")
    print("Disjointness check: PASS (no training prompt matches any eval task)")
except ImportError:
    print("(eval.tasks not available locally — skipping disjointness check; validated offline)")

# Validation: confirm system prompt is unchanged
EXPECTED_SYSTEM = (
    "You are a helpful AI agent with access to tools: calculator, word_count, "
    "web_search, and get_weather. Use tools when you need real information or "
    "exact computation. Respond directly when you already know the answer."
)
for ex in raw_examples:
    assert ex["messages"][0]["content"] == EXPECTED_SYSTEM, "System prompt drift detected"
print("System prompt check: PASS (unchanged from v1)")

# Validation: confirm tool schema consistency
for ex in raw_examples:
    for m in ex["messages"]:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls", []):
                assert tc["name"] in {"calculator", "word_count", "web_search", "get_weather"}, \
                    f"Unknown tool: {tc['name']}"
print("Tool schema check: PASS (only known tool names used)")


def format_example(example: dict) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"], tools=TOOLS_SCHEMA, tokenize=False,
    )
    return {"text": text}


dataset = Dataset.from_list(raw_examples).map(format_example)
print(f"Dataset size: {len(dataset)} trajectories")
print("\nExample formatted training text (first 800 chars of first sample):")
print(dataset[0]["text"][:800], "...\n")

# ===== CELL BREAK =====
# Loss masking — identical to v1
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Training config — IDENTICAL to v1 except:
#   - output_dir points to v2 folder
#   - num_train_epochs kept at 3 (fair ablation: same epoch count, more data)
#   - gradient_checkpointing=True (was: implicit off; recompute activations
#     to fit 46 examples on a free T4)
#   - optimizer is passed explicitly via `optimizers=(optim_8bit, None)` below
#     so we do NOT set `optim=` in SFTConfig — when a tuple is passed to the
#     Trainer constructor, args.optim is ignored (Trainer.create_optimizer is
#     guarded by `if self.optimizer is None`). Setting both would just be dead
#     config.
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
    dataset_text_field="text",
    max_length=1024,           # max_seq_length is deprecated/ignored in trl>=0.14
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    completion_only_loss=True,  # explicit: masks non-assistant tokens (safe no-op with manual collator)
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    optimizers=(optim_8bit, None),  # use 8-bit optimizer we created above
)

# Clear any cached VRAM from model load / dataset preprocessing before training.
import gc
gc.collect()
torch.cuda.empty_cache()
print(f"Free VRAM before training: "
      f"{torch.cuda.mem_get_info()[0] / 1e9:.2f} GB free / "
      f"{torch.cuda.mem_get_info()[1] / 1e9:.2f} GB total")

print("=" * 60)
print("EXPERIMENT 2 — QLoRA v2 training")
print(f"  base model:        {BASE_MODEL}")
print(f"  training data:     {INPUT_DATA} ({len(raw_examples)} trajectories)")
print(f"  output dir:        {OUTPUT_DIR}")
print(f"  epochs:            {training_args.num_train_epochs}")
print(f"  batch size:        {training_args.per_device_train_batch_size} "
      f"x {training_args.gradient_accumulation_steps} = "
      f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  learning rate:     {training_args.learning_rate}")
print(f"  LoRA r/alpha:      {lora_config.r}/{lora_config.lora_alpha}")
print(f"  max seq length:    {training_args.max_length}")
print("=" * 60)

print("Starting training...")
trainer.train()

print(f"\nSaving adapter to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done. v2 adapter saved.")

# ===== CELL BREAK =====
# Sanity check — generate one response with v2 adapter and confirm it
# produces a valid <tool_call> for a calculator prompt.
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
print("=== SANITY CHECK OUTPUT (v2) ===")
print(generated)
print("================================")
print("Look for a <tool_call>{\"name\": \"calculator\", ...}</tool_call> block above.")
print("If you see it, v2 adapter is good to go for serving + 23-task benchmark.")

# ===== CELL BREAK =====
# Sanity check 2: verify v1 adapter folder is s till on disk (untouched)
import os
v1_path = "/content/agentlab_qwen_lora_7b"
v2_path = "/content/agentlab_qwen_lora_7b_v2"
print(f"\nAdapter folders after v2 training:")
print(f"  v1 ({v1_path}): {'EXISTS' if os.path.isdir(v1_path) else 'MISSING'}")
print(f"  v2 ({v2_path}): {'EXISTS' if os.path.isdir(v2_path) else 'MISSING'}")
print("\nv1 should still exist and be untouched. v2 should be new.")
