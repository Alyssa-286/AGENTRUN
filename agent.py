"""
agent.py — the agent loop, now speaking MCP with Phase 4.2 measured retry tracking.

Mental model:
    1. THINK    — send conversation + tool definitions to the model.
                  The model either replies with a final answer, or with a
                  "function call" (structured intent, not executed code).
    2. ACT      — WE (the Python code) look up the matching tool and run it
                  over MCP stdio transport to mcp_server.py.
    3. OBSERVE  — we take the tool's return value and append it back into
                  the conversation as a "function response" message.
    4. repeat   — send the updated conversation back to the model.

PHASE 4.2 INSTRUMENTATION:
    - Measures compute latency vs. rate-limit wait time explicitly.
    - Adds `retry_wait_seconds` and `compute_latency_seconds` to `RunTrace`
      so that API quota pauses do not contaminate benchmarking measurements.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from memory import VectorMemory


load_dotenv()


@dataclass
class ToolCallRecord:
    """One tool call made during a run — what was asked for, and what came back."""
    tool_name: str
    arguments: dict
    result: str
    is_error: bool  # True if observation indicates an error


@dataclass
class RunTrace:
    """The full structured record of one agent.run() call.
    This is the PHASE 4.2 payload — everything the eval harness needs to
    score a run lives here, not just the final text answer.
    """
    user_prompt: str
    final_answer: str
    steps_taken: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    latency_seconds: float = 0.0
    retry_wait_seconds: float = 0.0  # time spent waiting on rate-limit backoff
    hit_max_steps: bool = False

    @property
    def compute_latency_seconds(self) -> float:
        """Latency with retry/backoff time subtracted out — true model/tool execution time."""
        return max(0.0, self.latency_seconds - self.retry_wait_seconds)

    @property
    def tool_names_called(self) -> set[str]:
        return {tc.tool_name for tc in self.tool_calls}

    @property
    def had_tool_error(self) -> bool:
        return any(tc.is_error for tc in self.tool_calls)


def _sanitize_schema_for_gemini(schema: dict) -> dict:
    """Convert MCP/Pydantic JSON Schema into the subset Gemini accepts."""
    allowed_keys = {"type", "description", "properties", "required", "items", "enum"}
    cleaned = {key: value for key, value in schema.items() if key in allowed_keys}
    if "properties" in cleaned:
        cleaned["properties"] = {
            prop_name: _sanitize_schema_for_gemini(prop_schema)
            for prop_name, prop_schema in cleaned["properties"].items()
        }
    if "items" in cleaned and isinstance(cleaned["items"], dict):
        cleaned["items"] = _sanitize_schema_for_gemini(cleaned["items"])
    return cleaned


def _extract_retry_delay(err: Exception, fallback: float = 5.0) -> float:
    """Extract server-recommended retry delay from error payload or message."""
    err_str = str(err)
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.5
    match_delay = re.search(r"'retryDelay':\s*'(\d+)s'", err_str)
    if match_delay:
        return float(match_delay.group(1)) + 1.5
    return fallback


class Agent:
    def __init__(
        self,
        model_name: str = "gemini-3.6-flash",
        max_steps: int = 6,
        long_term_memory: VectorMemory | None = None,
        mcp_server_script: str = "mcp_server.py",
    ):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY environment variable first.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_steps = max_steps
        self.long_term_memory = long_term_memory
        self.mcp_server_script = mcp_server_script

        self.config: types.GenerateContentConfig | None = None
        self.chat = None
        self.session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self.last_trace: RunTrace | None = None

    async def connect(self, verbose: bool = True):
        """Launch the MCP server, initialize the session, and discover tools."""
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
        if verbose:
            print(
                f"[MCP] Connected. Discovered {len(discovered.tools)} tool(s): "
                f"{[tool.name for tool in discovered.tools]}"
            )

        function_declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=_sanitize_schema_for_gemini(tool.inputSchema),
            )
            for tool in discovered.tools
        ]

        self.config = types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=function_declarations)]
        )
        self.chat = self.client.chats.create(model=self.model_name, config=self.config)

    def reset_memory(self):
        """Wipe short-term (conversation) memory and start fresh."""
        self.chat = self.client.chats.create(model=self.model_name, config=self.config)

    async def close(self):
        await self._exit_stack.aclose()

    def _send_with_measured_retry(
        self,
        message_to_send: object,
        verbose: bool = False,
        max_attempts: int = 6,
    ) -> tuple[object, float]:
        """Send message to Gemini, handling rate limits with measured backoff.

        Returns (response, retry_wait_seconds_spent).
        """
        base_delay = 4.0
        total_retry_wait = 0.0

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.chat.send_message(message_to_send)
                return response, total_retry_wait
            except (ClientError, ServerError, APIError) as err:
                code = getattr(err, "code", None)
                is_transient = code in (429, 500, 502, 503, 504) or "quota" in str(err).lower()
                if is_transient and attempt < max_attempts:
                    delay = _extract_retry_delay(err, fallback=base_delay)
                    if verbose or True:  # always show diagnostic retry notice
                        print(
                            f"\n[RATE LIMIT] {code or 'Quota'} received, waiting {delay:.1f}s "
                            f"(attempt {attempt}/{max_attempts})...",
                            flush=True,
                        )
                    time.sleep(delay)
                    total_retry_wait += delay
                    base_delay *= 2.0
                else:
                    raise
            except Exception:
                raise

        raise RuntimeError("Retry loop exhausted")

    async def run(self, user_prompt: str, verbose: bool = True) -> str:
        if self.session is None or self.chat is None:
            raise RuntimeError("Agent not connected — call `await agent.connect()` first.")

        run_start_time = time.monotonic()
        tool_call_records: list[ToolCallRecord] = []
        total_retry_wait = 0.0

        message_to_send: object = user_prompt
        if self.long_term_memory is not None:
            relevant_memories = self.long_term_memory.search(user_prompt, top_k=3)
            if relevant_memories:
                if verbose:
                    print(f"[MEMORY] Retrieved {len(relevant_memories)} relevant memory item(s):")
                    for memory_text in relevant_memories:
                        print(f"         - {memory_text}")
                memory_context = "\n".join(f"- {memory_text}" for memory_text in relevant_memories)
                message_to_send = (
                    f"[Relevant facts from long-term memory:]\n{memory_context}\n\n"
                    f"[Current user message:]\n{user_prompt}"
                )

        for step in range(1, self.max_steps + 1):
            if verbose:
                print(f"\n--- Step {step} ---")

            response, retry_wait = self._send_with_measured_retry(message_to_send, verbose=verbose)
            total_retry_wait += retry_wait

            function_calls = response.function_calls

            if not function_calls:
                final_text = response.text or ""
                if verbose:
                    print("[THINK] Model gave final answer, no tool needed.")

                if self.long_term_memory is not None:
                    self.long_term_memory.add(f"User asked: {user_prompt}\nAgent answered: {final_text}")

                self.last_trace = RunTrace(
                    user_prompt=user_prompt,
                    final_answer=final_text,
                    steps_taken=step,
                    tool_calls=tool_call_records,
                    latency_seconds=time.monotonic() - run_start_time,
                    retry_wait_seconds=total_retry_wait,
                    hit_max_steps=False,
                )
                return final_text

            function_responses = []
            for function_call in function_calls:
                tool_name = function_call.name
                tool_args = dict(function_call.args) if function_call.args else {}

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

                function_responses.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": observation},
                    )
                )

            if len(function_responses) == 1:
                message_to_send = function_responses[0]
            else:
                message_to_send = function_responses

        timeout_message = "Max steps reached without a final answer — agent may be stuck in a loop."
        self.last_trace = RunTrace(
            user_prompt=user_prompt,
            final_answer=timeout_message,
            steps_taken=self.max_steps,
            tool_calls=tool_call_records,
            latency_seconds=time.monotonic() - run_start_time,
            retry_wait_seconds=total_retry_wait,
            hit_max_steps=True,
        )
        return timeout_message