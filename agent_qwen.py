"""
agent_qwen.py — Agent loop powered by our fine-tuned Qwen 2.5 0.5B LoRA model.

Connects to the exact same MCP server (`mcp_server.py`) as the Gemini agent,
providing a drop-in replacement for comparative evaluation.

Mental model:
    1. THINK    — format conversation using Qwen chat template and generate tokens.
                  The fine-tuned model emits structured `<tool_call>` XML tags.
    2. ACT      — parse the tool name & arguments, call the MCP server over stdio.
    3. OBSERVE  — format the MCP tool observation as `<tool_response>` and append to context.
    4. REPEAT   — generate the final answer or next tool step until complete.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import AsyncExitStack

import torch
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent import RunTrace, ToolCallRecord

SYSTEM_PROMPT = (
    "You are a helpful AI agent with access to tools: calculator, word_count, "
    "web_search, and get_weather. Use tools when you need real information or "
    "exact computation. Respond directly when you already know the answer."
)


def _parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Extract tool calls from model output.
    Supports <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    as well as embedded JSON objects with 'name' and 'arguments'.
    """
    calls = []
    
    # Check for <tool_call> tags
    matches = re.findall(r"<tool_call>\s*({.*?})\s*</tool_call>", text, re.DOTALL)
    if not matches:
        # Fallback: check for raw tool call json if tags were slightly malformed
        raw_match = re.search(r'\{\s*"name"\s*:\s*"([a-zA-Z0-9_-]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}', text, re.DOTALL)
        if raw_match:
            try:
                name = raw_match.group(1)
                args = json.loads(raw_match.group(2))
                calls.append((name, args))
                return calls
            except Exception:
                pass
        return calls

    for m in matches:
        try:
            data = json.loads(m)
            name = data.get("name")
            args = data.get("arguments", {})
            if name:
                calls.append((name, args))
        except Exception:
            continue

    return calls


def _clean_final_text(text: str) -> str:
    """Remove tool tags or artifacts from final assistant message."""
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<tool_response>.*?</tool_response>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    return cleaned


class QwenAgent:
    def __init__(
        self,
        lora_dir: str = "agentlab_qwen_lora",
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_steps: int = 3,
        mcp_server_script: str = "mcp_server.py",
        device: str | None = None,
    ):
        self.lora_dir = lora_dir
        self.base_model_name = base_model_name
        self.max_steps = max_steps
        self.mcp_server_script = mcp_server_script

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None
        self.session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self.last_trace: RunTrace | None = None
        self.messages: list[dict] = []

    def _load_model(self, verbose: bool = True):
        if self.model is not None:
            return

        if verbose:
            print(f"[QwenAgent] Loading tokenizer from {self.lora_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.lora_dir)

        if verbose:
            print(f"[QwenAgent] Loading base model '{self.base_model_name}' on {self.device}...")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                local_files_only=True,
            )
        except Exception:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
            )

        if verbose:
            print(f"[QwenAgent] Attaching LoRA adapter from {self.lora_dir}...")
        self.model = PeftModel.from_pretrained(base_model, self.lora_dir)
        self.model.eval()

    async def connect(self, verbose: bool = True):
        """Load model and launch MCP server session."""
        self._load_model(verbose=verbose)

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.mcp_server_script],
            env=dict(os.environ),
        )

        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self.session.initialize()

        discovered = await self.session.list_tools()
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
            }
            for tool in discovered.tools
        ]
        if verbose:
            print(
                f"[MCP] Connected. Discovered {len(discovered.tools)} tool(s): "
                f"{[tool.name for tool in discovered.tools]}"
            )

        self.reset_memory()

    def reset_memory(self):
        """Reset conversation context for a clean benchmark run."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def close(self):
        await self._exit_stack.aclose()

    def _generate(self, max_new_tokens: int = 96) -> str:
        prompt_text = self.tokenizer.apply_chat_template(
            self.messages,
            tools=self.tools_schema,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()

    async def run(self, user_prompt: str, verbose: bool = True) -> str:
        if self.session is None or self.model is None:
            raise RuntimeError("QwenAgent not connected — call `await agent.connect()` first.")

        run_start_time = time.monotonic()
        tool_call_records: list[ToolCallRecord] = []

        if not self.messages or self.messages[0].get("role") != "system":
            self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.messages.append({"role": "user", "content": user_prompt})

        for step in range(1, self.max_steps + 1):
            if verbose:
                print(f"\n--- Step {step} (Qwen 0.5B LoRA) ---")

            generation = self._generate()
            tool_calls = _parse_tool_calls(generation)

            if not tool_calls:
                final_text = _clean_final_text(generation)
                if not final_text:
                    final_text = generation
                if verbose:
                    print(f"[THINK] Model gave final answer: {final_text}")

                self.messages.append({"role": "assistant", "content": final_text})
                self.last_trace = RunTrace(
                    user_prompt=user_prompt,
                    final_answer=final_text,
                    steps_taken=step,
                    tool_calls=tool_call_records,
                    latency_seconds=time.monotonic() - run_start_time,
                    retry_wait_seconds=0.0,  # Local model has no API rate limit waits
                    hit_max_steps=False,
                )
                return final_text

            # Model requested tool calls
            for tool_name, tool_args in tool_calls:
                if verbose:
                    print(f"[ACT] Model wants to call (via MCP): {tool_name}({tool_args})")

                try:
                    mcp_result = await self.session.call_tool(tool_name, tool_args)
                    observation = mcp_result.content[0].text
                except Exception as exc:
                    observation = f"ERROR calling MCP tool '{tool_name}': {exc}"

                if verbose:
                    print(f"[OBSERVE] Result: {observation}")

                tool_call_records.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        arguments=tool_args,
                        result=observation,
                        is_error=observation.startswith("ERROR"),
                    )
                )

                self.messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": tool_name, "arguments": tool_args}],
                })
                self.messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": observation,
                })

        timeout_msg = "Max steps reached without a final answer."
        self.last_trace = RunTrace(
            user_prompt=user_prompt,
            final_answer=timeout_msg,
            steps_taken=self.max_steps,
            tool_calls=tool_call_records,
            latency_seconds=time.monotonic() - run_start_time,
            retry_wait_seconds=0.0,
            hit_max_steps=True,
        )
        return timeout_msg


async def _main():
    agent = QwenAgent()
    await agent.connect(verbose=True)
    try:
        prompt = "What is 45 times 18?"
        print(f"\nRunning test prompt: '{prompt}'")
        res = await agent.run(prompt, verbose=True)
        print(f"\n[FINAL ANSWER] {res}")
    finally:
        await agent.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())

