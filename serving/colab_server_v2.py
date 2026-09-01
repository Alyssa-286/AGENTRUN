# colab_server_v2.py
#
# PASTE THIS INTO A COLAB CELL (split at the blank lines marked "CELL BREAK").
# Requires a GPU runtime: Runtime -> Change runtime type -> T4 GPU.
#
# This is the v2 server for Experiment 2. Differences from colab_server.py (v1):
#   1. Loads the v2 LoRA adapter: /content/agentlab_qwen_lora_7b_v2
#   2. CORRECTED robust parser that extracts the FIRST valid tool call even when
#      the model emits malformed/nested tags (the v1 parser failed when Qwen
#      generated <tool_call> without a closing tag before the next opening tag).
#
# The parser fix is critical: the adversarial_chained_search_calc failure showed
# the model intended web_search + calculator but emitted broken tags. The v1 parser
# required properly closed <tool_call>...</tool_call> blocks. The v2 parser
# falls back to extracting the first JSON object after any <tool_call> tag.
#
# All other settings (system prompt, tool schema, generation config) are unchanged.

# ===== CELL BREAK =====
!pip install -q transformers peft accelerate bitsandbytes fastapi uvicorn pyngrok nest_asyncio

# ===== CELL BREAK =====
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "/content/agentlab_qwen_lora_7b_v2"   # v2 adapter

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model in 4-bit...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=quant_config, device_map="auto"
)

print("Applying v2 LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print(f"v2 model ready. Adapter: {ADAPTER_PATH}")

# ===== CELL BREAK =====
import re
import json
from fastapi import FastAPI, Request
import uvicorn
import nest_asyncio
from pyngrok import ngrok

app = FastAPI()

# ── Corrected robust parser ────────────────────────────────────────────────────
# Strategy 1: standard properly-closed <tool_call>...</tool_call> blocks
TOOL_CALL_CLOSED = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL
)

# Strategy 2: when the model emits malformed tags (missing closing tag),
# extract the first JSON object starting from the first <tool_call> tag.
# E.g. malformed output:
#   <tool_call>\n{"name": "web_search", ...}\n<tool_call>\n<tool_call>\n...
# Strategy 1 finds zero matches. Strategy 2 scans for the first <tool_call>
# tag and greedily captures the first {...} block from that point.
TOOL_CALL_OPEN = re.compile(
    r"<tool_call>\s*(\{.*?\})",
    re.DOTALL
)


def parse_model_output(raw_text: str) -> dict:
    """Robust two-pass parser for Qwen tool-call output.

    Pass 1 (strict): matches properly closed <tool_call>...</tool_call> blocks.
    Pass 2 (fallback): if pass 1 finds nothing, finds the first JSON object
    starting at the first <tool_call> opening tag. This handles the malformed
    case where the model's generation lacks a closing tag before the next
    opening tag — the adversarial_chained_search_calc failure case.

    Returns {"tool_calls": [...], "text": None} if any tool call found,
    otherwise {"tool_calls": [], "text": raw_text} for a final answer.
    """
    # Pass 1: standard closed blocks
    matches = TOOL_CALL_CLOSED.findall(raw_text)
    if matches:
        tool_calls = []
        for m in matches:
            try:
                parsed = json.loads(m)
                name = parsed.get("name")
                if name:
                    tool_calls.append({"name": name, "arguments": parsed.get("arguments", {})})
            except json.JSONDecodeError:
                continue
        if tool_calls:
            return {"tool_calls": tool_calls, "text": None}

    # Pass 2: malformed / unclosed first tag — extract first JSON from first
    # <tool_call> tag. This handles cases like:
    #   <tool_call>\n{"name": "web_search", ...}\n<tool_call>\n<tool_call>\n...
    # where the first block has no closing </tool_call> before the next <tool_call>.
    first_open = TOOL_CALL_OPEN.search(raw_text)
    if first_open:
        json_str = first_open.group(1)
        try:
            parsed = json.loads(json_str)
            name = parsed.get("name")
            if name:
                return {
                    "tool_calls": [{"name": name, "arguments": parsed.get("arguments", {})}],
                    "text": None,
                }
        except json.JSONDecodeError:
            pass

    return {"tool_calls": [], "text": raw_text.strip()}


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

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True)

    # Diagnostic logging
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
    return {"status": "ok", "adapter": ADAPTER_PATH}


# ===== CELL BREAK =====
# Run this LAST. Get your ngrok authtoken at https://dashboard.ngrok.com/get-started/your-authtoken
# Paste it below.

NGROK_AUTH_TOKEN = "PASTE_YOUR_NGROK_TOKEN_HERE"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

nest_asyncio.apply()
public_url = ngrok.connect(8000)
print(f"\n{'=' * 60}")
print(f"v2 SERVER URL — paste this into the v2 evaluator: {public_url}")
print(f"Adapter loaded: {ADAPTER_PATH}")
print(f"{'=' * 60}\n")

uvicorn.run(app, host="0.0.0.0", port=8000)
