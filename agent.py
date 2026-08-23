"""
agent.py — the agent loop, now speaking MCP instead of importing tools directly.

The THINK / ACT / OBSERVE loop did not change. Only ACT changed:
instead of calling in-process Python tool functions, the agent sends a
JSON-RPC request over stdio to a separate MCP server process.
"""

from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from memory import VectorMemory


load_dotenv()


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

    async def close(self):
        await self._exit_stack.aclose()

    def reset_memory(self):
        """Wipe short-term (conversation) memory and start fresh."""

        self.chat = self.client.chats.create(model=self.model_name, config=self.config)

    async def run(self, user_prompt: str, verbose: bool = True) -> str:
        if self.session is None or self.chat is None:
            raise RuntimeError("Agent not connected — call `await agent.connect()` first.")

        chat = self.chat

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

            response = chat.send_message(message_to_send)

            function_calls = response.function_calls

            if not function_calls:
                final_text = response.text or ""
                if verbose:
                    print("[THINK] Model gave final answer, no tool needed.")

                if self.long_term_memory is not None:
                    self.long_term_memory.add(f"User asked: {user_prompt}\nAgent answered: {final_text}")

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

        return "Max steps reached without a final answer — agent may be stuck in a loop."