# colab_server.py
#
# PASTE THIS INTO A COLAB CELL (or a few cells — split at the blank lines
# marked "CELL BREAK" if you prefer). Requires a GPU runtime: Runtime ->
# Change runtime type -> T4 GPU.
#
# What this does: loads Qwen2.5-7B-Instruct + your LoRA adapter, exposes
# a single POST /chat endpoint that accepts a messages+tools payload (same
# shape our local ft_agent.py will send) and returns EITHER structured
# tool_calls or final text — the parsing of Qwen's <tool_call> tag format
# happens HERE, server-side, so the local client stays simple and doesn't
# need to know anything Qwen-specific. This mirrors the same "protocol
# adapter" pattern we used for MCP: the ugly format-specific bits live in
# one place.

# ===== CELL BREAK =====
!pip install -q transformers peft accelerate bitsandbytes fastapi uvicorn pyngrok nest_asyncio

# ===== CELL BREAK =====
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "/content/agentlab_qwen_lora_7b"  # matches colab_train_7b.py's OUTPUT_DIR

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model in 4-bit (fits comfortably on a free T4)...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=quant_config, device_map="auto"
)

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("Model ready.")

# ===== CELL BREAK =====
import re
import json
from fastapi import FastAPI, Request
import uvicorn
import nest_asyncio
from pyngrok import ngrok

app = FastAPI()

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_model_output(raw_text: str) -> dict:
    """Qwen2.5's tool-calling chat template renders tool calls as
    <tool_call>{...json...}</tool_call> blocks (possibly several, for
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


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools", [])

    # apply_chat_template with tools= is what makes Qwen2.5 render the
    # tool schema into the prompt AND expect/produce <tool_call> tags in
    # its output — this must match how training data was templated.
    # If messages already contain a system role entry, tokenizer applies
    # it as part of the template. Pass it as-is (don't double-wrap it).
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools if tools else None,
        add_generation_prompt=True, tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate: use greedy (do_sample=False) for determinism.
    # The training sanity check used do_sample=False. Use the same setting
    # so the server reproduces what the model was trained to produce.
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,          # was: temperature=0.3, do_sample=True
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True)

    # --- DIAGNOSTIC LOGGING ---
    # Always log the raw model output so we can see exactly what Qwen produced
    # before the parser ran. This is the single most useful thing for debugging
    # "why didn't it call a tool" questions.
    print()
    print("=" * 60)
    print("[RAW MODEL OUTPUT]")
    print(raw_text[:1000])  # cap at 1000 chars to keep Colab output readable
    print("=" * 60)
    print("[PARSER RESULT]", parse_model_output(raw_text))
    print()

    return parse_model_output(raw_text)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ===== CELL BREAK =====
# Run this LAST. It prints a public URL — copy that into ft_agent.py's
# COLAB_SERVER_URL on your laptop. Get a free ngrok authtoken at
# https://dashboard.ngrok.com/get-started/your-authtoken and paste it below.

NGROK_AUTH_TOKEN = "PASTE_YOUR_NGROK_TOKEN_HERE"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

nest_asyncio.apply()
public_url = ngrok.connect(8000)
print(f"\n{'=' * 60}")
print(f"PUBLIC URL — paste this into ft_agent.py: {public_url}")
print(f"{'=' * 60}\n")

uvicorn.run(app, host="0.0.0.0", port=8000)
