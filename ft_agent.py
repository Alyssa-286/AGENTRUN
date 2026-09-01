"""
ft_agent.py - Local client for the remote fine-tuned Qwen 2.5 7B + LoRA inference server.

Provides an interface compatible with the existing AgentLab evaluation workflow.
Communicates with the Colab-hosted model via HTTP POST to /chat endpoint.

The mental model is the same as agent_qwen.py, but generation happens remotely
via HTTP rather than local GPU inference:
    1. THINK    - send conversation to remote model via HTTP
    2. ACT      - parse tool calls from response, execute via local MCP server
    3. OBSERVE  - send tool results back to remote model (repeat 1-3)
    4. UNTIL    - final answer or max steps reached

The Colab server expects:
    POST /chat  body: {
        "messages": [
            {"role": "system", "content": "..."},    # MUST match training system prompt
            {"role": "user", "content": "..."},     # user turns
            ...
        ],
        "tools": [                               # MCP tool schema — CRITICAL
            {"type": "function", "function": {"name": "...", "parameters": {...}}}
        ]
    }
    returns: {
        "text": "...",                           # final answer text (or None if tool calls)
        "tool_calls": [{"name": "...", "arguments": {...}}, ...]
    }

The three things that MUST match training to produce tool calls:
  1. System prompt (same string used during fine-tuning)
  2. Tools schema (same JSON structure passed to apply_chat_template)
  3. Generation mode (greedy / do_sample=False — the server enforces this)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent import ToolCallRecord, RunTrace


# Tool-call tag delimiters emitted by the fine-tuned Qwen model.
TOOL_CALL_OPEN = "<" + "tool_call" + ">"
TOOL_CALL_CLOSE = "</" + "tool_call" + ">"

# System prompt — MUST match the one used in training data generation exactly.
# The model was fine-tuned with this specific system prompt. If it changes,
# Qwen won't recognize the context and won't produce tool calls.
SYSTEM_PROMPT = (
    "You are a helpful AI agent with access to tools: calculator, word_count, "
    "web_search, and get_weather. Use tools when you need real information or "
    "exact computation. Respond directly when you already know the answer."
)


@dataclass
class ToolCallInfo:
    """Parsed tool call information from model response."""
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class ParsedResponse:
    """Parsed response from remote model, with tool calls."""
    text: str
    tool_calls: List[ToolCallInfo] = field(default_factory=list)


class QwenRemoteClient:
    """Local client for the remote fine-tuned Qwen model served on Colab.

    Implements the same agent-loop contract as Agent (agent.py) and
    QwenAgent (agent_qwen.py): a connect() / run() / reset_memory() / close()
    interface that produces a RunTrace the eval harness can score.

    The LLM "thinking" step is delegated to the Colab server over HTTP; the
    "acting" step (tool execution) is local via the same MCP server
    (mcp_server.py) that the Phase 1 frontier-agent code used, so all three
    benchmarked models (Base Qwen 7B, LoRA v1, LoRA v2) are tested against
    identical tools.
    """

    def __init__(
        self,
        colab_server_url: Optional[str] = None,
        mcp_server_script: str = "mcp_server.py",
        max_steps: int = 6,
        request_timeout: float = 120.0,
    ):
        # Get server URL from argument or environment; never hard-code it.
        self.colab_server_url = (
            colab_server_url
            or os.environ.get("COLAB_SERVER_URL")
            or "http://localhost:8000"
        )
        self.colab_server_url = self.colab_server_url.rstrip("/")
        self.mcp_server_script = mcp_server_script
        self.max_steps = max_steps
        self.request_timeout = request_timeout

        # HTTP client for Colab server (lazy-init in connect())
        self.http_client: Optional[httpx.AsyncClient] = None

        # MCP client for local tool execution
        self.session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()

        # Tool schema discovered from MCP server (converted to Qwen function format)
        self.tools_schema: List[Dict] = []

        # State tracking - compatible with eval harness (same fields as Agent)
        self.last_trace: Optional[RunTrace] = None
        self.tool_call_records: List[ToolCallRecord] = []

    async def connect(self, verbose: bool = True) -> None:
        """Initialize HTTP client and MCP server connection, discover tools."""
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.request_timeout, connect=10.0),
            headers={"Content-Type": "application/json"},
        )

        if verbose:
            print("[QwenRemote] Connecting to Colab server at " + str(self.colab_server_url))

        # Health check
        try:
            health_resp = await self.http_client.get(self.colab_server_url + "/health")
            if health_resp.status_code != 200:
                raise RuntimeError(
                    "Server health check failed with status " + str(health_resp.status_code)
                )
            if verbose:
                print("[QwenRemote] Server health check passed")
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Cannot reach Colab server at " + self.colab_server_url + "/health: " + str(exc)
            )

        # Start local MCP server for tool execution
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

        # Discover tools from MCP and convert to the Qwen function-calling format.
        # This is the schema the server passes to tokenizer.apply_chat_template(tools=...).
        # CRITICAL: this schema must match TOOLS_SCHEMA in colab_train_7b.py exactly.
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
                "[MCP] Connected. Discovered "
                + str(len(discovered.tools))
                + " tool(s): "
                + str([tool.name for tool in discovered.tools])
            )
            print(
                "[QwenRemote] Tool schema prepared for server ("
                + str(len(self.tools_schema))
                + " tools)"
            )

        self.reset_memory()

    def reset_memory(self) -> None:
        """Reset per-run state so each benchmark task is evaluated independently."""
        self.tool_call_records = []
        self.last_trace = None

    async def close(self) -> None:
        """Clean up resources."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        await self._exit_stack.aclose()

    def _parse_tool_calls_from_text(self, text: str) -> List[ToolCallInfo]:
        """Extract tool calls from model raw text output.

        Handles the Qwen tool-call XML-tag format that the fine-tuned model
        was trained to emit, e.g.:
            <tool_call>{"name": "calculator", "arguments": {"expression": "84 * 19"}}</tool_call>

        This is a fallback — the server's parse_model_output() also parses this,
        but we also check the text for debugging purposes.
        """
        calls = []

        # Strategy 1: Look for TOOL_CALL_OPEN/CLOSE tags
        pattern = re.escape(TOOL_CALL_OPEN) + r"\s*({.*?})\s*" + re.escape(TOOL_CALL_CLOSE)
        matches = re.findall(pattern, text, re.DOTALL)

        # Strategy 2: If no tagged calls, try raw JSON objects with name/args
        if not matches:
            fallback = r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{.*?\}\}'
            matches = re.findall(fallback, text, re.DOTALL)

        for m in matches:
            try:
                data = json.loads(m)
                name = data.get("name")
                args = data.get("arguments", {})
                if name:
                    calls.append(ToolCallInfo(tool_name=name, arguments=args))
            except Exception:
                continue

        return calls

    async def _query_remote_model(
        self,
        messages: List[Dict[str, str]],
        verbose: bool = True,
        step: int = 1,
    ) -> ParsedResponse:
        """Send messages + tools schema to remote Colab server and parse response."""
        if self.http_client is None:
            raise RuntimeError("HTTP client not initialized")

        # Build the request payload with BOTH messages and tools.
        # The server passes tools= to tokenizer.apply_chat_template(), which is
        # what makes Qwen render the tool schema in its context and emit <tool_call> tags.
        # Without tools=, Qwen behaves like a plain chat model — no tool calls will appear.
        request_payload = {
            "messages": messages,
            "tools": self.tools_schema,
        }

        if verbose:
            print(
                "[QwenRemote] Step "
                + str(step)
                + ": sending "
                + str(len(messages))
                + " message(s) + "
                + str(len(self.tools_schema))
                + " tool(s) to server"
            )

        try:
            response = await self.http_client.post(
                self.colab_server_url + "/chat",
                json=request_payload,
                timeout=httpx.Timeout(self.request_timeout, connect=10.0),
            )

            if response.status_code != 200:
                error_text = (
                    response.text[:500] if len(response.text) > 500 else response.text
                )
                raise RuntimeError(
                    "Colab server error (" + str(response.status_code) + "): " + error_text
                )

            server_response = response.json()

            # --- DIAGNOSTIC LOGGING ---
            # Show what the server's parser extracted. This is the most direct
            # view of "did the model produce a tool call or not" without re-parsing.
            #
            # null-handling: the server returns text=null whenever tool_calls
            # is non-empty (it treats a tool-call turn as a "no text" turn).
            # We coerce that to "" here so the regex fallback only runs when
            # there is actually text to scan — otherwise re.findall throws
            # "expected string or bytes-like object, got 'NoneType'".
            text = server_response.get("text") or ""
            raw_tool_calls = server_response.get("tool_calls", [])

            # Only run the <tool_call> regex fallback when text is a real string
            # with something to parse. Prefer the server's structured tool_calls
            # list whenever it is non-empty; the regex is purely a safety net
            # for the rare case where the server missed a tool call we can
            # see in the raw text.
            if text and not raw_tool_calls:
                parsed_from_text = self._parse_tool_calls_from_text(text)
            else:
                parsed_from_text = []

            if verbose:
                print(
                    "[QwenRemote] Server parsed: text="
                    + repr(text[:200] + "..." if len(text) > 200 else text)
                )
                if raw_tool_calls:
                    print("[QwenRemote] Server found tool_calls: " + str(raw_tool_calls))
                elif parsed_from_text:
                    print("[QwenRemote] Local re-parse found tool_calls: " + str(parsed_from_text))
                else:
                    print("[QwenRemote] No tool calls found in response")
            # --- END DIAGNOSTIC ---

            # Use the server's parsed result as the primary source.
            # Also include our own local re-parse as a fallback in case
            # the server's regex missed something.
            all_tool_calls: List[ToolCallInfo] = []

            for tc in raw_tool_calls:
                all_tool_calls.append(
                    ToolCallInfo(
                        tool_name=tc.get("name", ""),
                        arguments=tc.get("arguments", {}),
                    )
                )

            # Fallback: if server found nothing but our local parse found tool calls,
            # use those (handles cases where server regex missed but text has them)
            if not all_tool_calls and parsed_from_text:
                all_tool_calls = parsed_from_text

            return ParsedResponse(text=text, tool_calls=all_tool_calls)

        except httpx.TimeoutException:
            raise RuntimeError("Timeout waiting for Colab server response")
        except httpx.NetworkError as exc:
            raise RuntimeError("Network error connecting to Colab server: " + str(exc))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid JSON response from Colab server: " + str(exc))
        except Exception as exc:
            raise RuntimeError("Unexpected error querying Colab server: " + str(exc))

    async def run(self, user_prompt: str, verbose: bool = True) -> str:
        """Run agent on user prompt, communicating with remote model and local MCP.

        This implements the same agent loop pattern as agent.py and agent_qwen.py:
        - Step 1: Send system + user prompt + tools to model (via HTTP)
        - Step 2: Model responds with final answer OR tool calls
        - Step 3: If tool calls, execute via MCP and send results back
        - Step 4: Repeat until final answer or max steps
        """
        if self.session is None:
            raise RuntimeError(
                "QwenRemoteClient not connected - call `await connect()` first."
            )

        run_start_time = time.monotonic()

        # Build conversation starting with system prompt + user turn.
        # The system prompt MUST be first — it sets the agent's identity and
        # available tools. This matches how training data was constructed.
        conversation: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for step in range(1, self.max_steps + 1):
            if verbose:
                print("")
                print("--- Step " + str(step) + " (QwenRemote) ---")

            # Query model for response (sends messages + tools to Colab server)
            parsed_response = await self._query_remote_model(
                conversation, verbose=verbose, step=step
            )
            tool_calls = parsed_response.tool_calls

            # Check if we got a final answer (no tool calls)
            if not tool_calls:
                final_text = parsed_response.text.strip() if parsed_response.text else ""

                if verbose:
                    print("[THINK] Model gave final answer: " + repr(final_text[:100]))

                # Create trace for evaluation harness
                self.last_trace = RunTrace(
                    user_prompt=user_prompt,
                    final_answer=final_text,
                    steps_taken=step,
                    tool_calls=self.tool_call_records,
                    latency_seconds=time.monotonic() - run_start_time,
                    retry_wait_seconds=0.0,  # No retry waits for HTTP
                    hit_max_steps=False,
                )

                return final_text

            # Model requested tool calls — execute them via local MCP server
            for tool_call in tool_calls:
                if verbose:
                    print(
                        "[ACT] Model wants to call: "
                        + str(tool_call.tool_name)
                        + "("
                        + str(tool_call.arguments)
                        + ")"
                    )

                try:
                    mcp_result = await self.session.call_tool(
                        tool_call.tool_name, tool_call.arguments
                    )
                    observation = mcp_result.content[0].text
                except Exception as exc:
                    observation = (
                        "ERROR calling MCP tool '" + str(tool_call.tool_name) + "': " + str(exc)
                    )

                if verbose:
                    print("[OBSERVE] Result: " + observation)

                # Record tool call for evaluation
                self.tool_call_records.append(
                    ToolCallRecord(
                        tool_name=tool_call.tool_name,
                        arguments=tool_call.arguments,
                        result=observation,
                        is_error=observation.startswith("ERROR"),
                    )
                )

                # Append tool call + result to conversation for next model turn.
                # Qwen's chat template renders role="tool" messages as <tool_response>
                # blocks inside a user-role wrapper. We use role="tool" so the
                # server's tokenizer.apply_chat_template produces the format
                # the model was trained on.
                conversation.append({
                    "role": "assistant",
                    "content": "<tool_call>"
                    + json.dumps({"name": tool_call.tool_name, "arguments": tool_call.arguments})
                    + "</tool_call>",
                })
                conversation.append({
                    "role": "tool",
                    "name": tool_call.tool_name,
                    "content": observation,
                })

        # Max steps reached
        timeout_message = "Max steps reached without a final answer."
        self.last_trace = RunTrace(
            user_prompt=user_prompt,
            final_answer=timeout_message,
            steps_taken=self.max_steps,
            tool_calls=self.tool_call_records,
            latency_seconds=time.monotonic() - run_start_time,
            retry_wait_seconds=0.0,
            hit_max_steps=True,
        )

        return timeout_message


async def diagnostic_adversarial_chained_search_calc():
    """Focused diagnostic for the adversarial_chained_search_calc failure.

    Reproduces the exact multi-step chain the model failed on:
        Step 1: web_search  (model's first call was malformed → server returned empty tool_calls → treated as final answer)
        Step 2: calculator  (never reached)

    Now with the parser fix, the first <tool_call>...</tool_call> block is extracted
    and the model's malformed trailing tags do not poison the response.
    """
    server_url = os.environ.get("COLAB_SERVER_URL", "http://localhost:8000")
    TASK = (
        "Search for what year the first iPhone was released, then use the calculator "
        "to work out how many years ago that was from the year 2026."
    )

    print("=" * 60)
    print("DIAGNOSTIC: adversarial_chained_search_calc")
    print("Server: " + server_url)
    print("Expected: Step 1 -> web_search, Step 2 -> calculator, final answer 19")
    print("=" * 60)

    client = QwenRemoteClient(colab_server_url=server_url, max_steps=6)

    try:
        await client.connect(verbose=True)
        client.reset_memory()
        result = await client.run(TASK, verbose=True)

        print()
        print("=" * 60)
        print("[FINAL ANSWER] " + result)
        print(f"[STEPS TAKEN]  {client.last_trace.steps_taken}")
        print(f"[TOOL CALLS]   {len(client.last_trace.tool_calls)}")
        for i, tc in enumerate(client.last_trace.tool_calls, 1):
            print(f"  {i}. {tc.tool_name}({tc.arguments})")
        print("=" * 60)

        # Assertions
        assert len(client.last_trace.tool_calls) >= 2, (
            f"Expected at least 2 tool calls, got {len(client.last_trace.tool_calls)}"
        )
        assert client.last_trace.tool_calls[0].tool_name == "web_search", (
            f"First call should be web_search, got {client.last_trace.tool_calls[0].tool_name}"
        )
        assert client.last_trace.tool_calls[1].tool_name == "calculator", (
            f"Second call should be calculator, got {client.last_trace.tool_calls[1].tool_name}"
        )
        print("[PASS] Both tool calls extracted correctly from malformed output")
    finally:
        await client.close()


async def _demo():
    """Demo the QwenRemoteClient with one calculator task and one no-tool task."""
    server_url = os.environ.get("COLAB_SERVER_URL", "http://localhost:8000")

    print("=" * 60)
    print("QwenRemoteClient Diagnostic Demo")
    print("Server: " + server_url)
    print("=" * 60)

    client = QwenRemoteClient(colab_server_url=server_url, max_steps=6)

    try:
        await client.connect(verbose=True)

        # --- Test 1: Calculator task (MUST call a tool) ---
        print("")
        print("=" * 60)
        print("TEST 1: Calculator task")
        print("Expected: Model emits <tool_call> for calculator")
        print("=" * 60)
        client.reset_memory()
        result1 = await client.run("What is 45 times 18?", verbose=True)
        print("")
        print("[RESULT] " + result1)

        # --- Test 2: No-tool task (should NOT call a tool) ---
        print("")
        print("=" * 60)
        print("TEST 2: No-tool task")
        print("Expected: Model gives final answer directly, no tool call")
        print("=" * 60)
        client.reset_memory()
        result2 = await client.run(
            "What is the chemical symbol for gold?",
            verbose=True
        )
        print("")
        print("[RESULT] " + result2)

    except Exception as exc:
        print("")
        print("[ERROR] " + str(exc))
        import traceback
        traceback.print_exc()
    finally:
        await client.close()


if __name__ == "__main__":
    if os.environ.get("DEMO") == "1":
        asyncio.run(_demo())
    else:
        # Run the adversarial_chained_search_calc diagnostic by default
        asyncio.run(diagnostic_adversarial_chained_search_calc())
