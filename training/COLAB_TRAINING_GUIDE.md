# Phase 5: Google Colab Fine-Tuning Guide

This guide walks you through fine-tuning a small open-weight language model (**Qwen 2.5 0.5B/1.5B Instruct**) on your generated `training_data.jsonl` trajectories using **QLoRA / Unsloth** on a free Google Colab GPU.

---

## Step 1: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **New Notebook**.
3. Enable GPU:
   * Go to **Runtime** > **Change runtime type**.
   * Select **T4 GPU** (Free tier).
   * Click **Save**.

---

## Step 2: Upload `training_data.jsonl`
1. On the left sidebar of Google Colab, click the **Folder icon (📁)** (Files).
2. Click the **Upload icon (⬆️)**.
3. Select and upload `training_data.jsonl` from your `e:\agentrun` folder.

---

## Step 3: Run the Training Cell
Create a code cell in Colab, paste the script below, and run it (**Shift + Enter**):

```python
# 1. Install Unsloth & dependencies (takes ~60s)
!pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.29" "trl<0.9.0" peft accelerate bitsandbytes triton

# 2. Import FastLanguageModel
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 3. Load Qwen 2.5 0.5B Instruct (Ultra-fast, fits on free T4)
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-0.5B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# 4. Attach LoRA adapters for Tool-Use
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 5. Format Dataset using Chat Template
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

def format_prompts(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)

# 6. Train with SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=40,  # ~3 epochs for 37 examples
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()
print("\n✅ Training Complete!")

# 7. Test Inference on a Sample Prompt
FastLanguageModel.for_inference(model)
test_messages = [
    {"role": "system", "content": "You are a helpful AI agent with access to tools: calculator, word_count, web_search, and get_weather."},
    {"role": "user", "content": "What is 45 times 18?"}
]
inputs = tokenizer.apply_chat_template(test_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=128, use_cache=True)
print("\n--- Model Output Test ---")
print(tokenizer.decode(outputs[0][inputs.shape[1]:]))

# 8. Save LoRA Adapters
model.save_pretrained("agentlab_qwen_lora")
tokenizer.save_pretrained("agentlab_qwen_lora")
!zip -r agentlab_qwen_lora.zip agentlab_qwen_lora/
print("\n📦 Download 'agentlab_qwen_lora.zip' from the Colab Files panel on the left!")
```

---

## Step 4: Download the Result
1. In Colab's left sidebar (Files 📁), find `agentlab_qwen_lora.zip`.
2. Click the three dots `...` next to `agentlab_qwen_lora.zip` and click **Download**.
3. Once downloaded, come back to this chat and we'll connect it to our Phase 5 evaluation comparison!
