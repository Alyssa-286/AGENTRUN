# colab_server_base.py
#
# PASTE THIS INTO A COLAB CELL (split at the blank lines marked "CELL BREAK").
# Requires a GPU runtime: Runtime -> Change runtime type -> T4 GPU.
#
# IDENTICAL to colab_server.py EXCEPT: does NOT load the LoRA adapter.
# The base Qwen2.5-7B-Instruct model is loaded directly so we can run the
# same 23-task benchmark against the base model as a control experiment.
#
# All parsing, prompt construction, generation settings, and endpoints are
# EXACTLY the same (verified line-by-line) so that any performance difference
# is attributable to fine-tuning, not server-side drift.

# ===== CELL BREAK =====
!pip install -q transformers peft accelerate bitsandbytes fastapi uvicorn pyngrok nest_asyncio

# ===== CELL BREAK =====
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model in 4-bit (no LoRA adapter) on T4...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
model.eval()
print("Model ready. (Base model — no LoRA adapter applied.)")

# ===== CELL BREAK =====
import re
import json
from fastapi import FastAPI, Request
import uvicorn
import nest_asyncio
from pyngrok import ngrok

app = FastAPI()

# --- EXACT COPY: tool-call parser (identical to colab_server.py) ---
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL
)


def parse_model_output(raw_text: str) -> dict:
    """Qwen2.5's tool-calling chat template renders tool calls as
     icode{...json...} blocks (possibly several, for
    parallel calls). If we find any, this IS a tool-call turn — the
    surrounding text is template scaffolding, not a real answer. If we
    find none, the whole output is the final answer text."""
    matches = TOOL_CALL_PATTERN.findall(raw_text)
    if matches:
        tool_calls = []
        for m in matches:
            try:
                parsed = json.loads(m)
                tool_calls.append({
                    "name": parsed.get("name"),
                    "arguments": parsed.get("arguments", {}),
                })
            except json.JSONDecodeError:
                continue  # malformed tool call from the model — skip it, don't crash the server
        if tool_calls:
            return {"tool_calls": tool_calls, "text": None}

    return {"tool_calls": [], "text": raw_text.strip()}


# --- EXACT COPY: /chat endpoint (identical to colab_server.py) ---
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools", [])

    prompt = tokenizer.apply_chat_template(
        messages, tools=tools if tools else None,
        add_generation_prompt=True, tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate: use greedy (do_sample=False) for determinism — matches training
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True)

    # --- EXACT COPY: DIAGNOSTIC LOGGING ---
    print()
    print("=" * 60)
    print("[RAW MODEL OUTPUT]")
    print(raw_text[:1000])
    print("=" * 60)
    print("[PARSER RESULT]", parse_model_output(raw_text))
    print()

    return parse_model_output(raw_text)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ===== CELL BREAK =====
# Run this LAST. Paste your ngrok authtoken at https://dashboard.ngrok.com/get-started/your-authtoken

NGROK_AUTH_TOKEN = "PASTE_YOUR_NGROK_TOKEN_HERE"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

nest_asyncio.apply()
public_url = ngrok.connect(8000)
print(f"\n{'=' * 60}")
print(f"PUBLIC URL — paste this into ft_agent.py: {public_url}")
print(f"{'=' * 60}\n")

uvicorn.run(app, host="0.0.0.0", port=8000)
