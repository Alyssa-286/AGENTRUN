"""
main.py — run this to see the agent loop in action, now over MCP.

This file no longer imports tools.py, providers/, or the provider classes
directly. The MCP server owns tool construction; the agent only discovers
and calls tools over the protocol.
"""

from __future__ import annotations

import asyncio

from agent import Agent
from memory import VectorMemory


async def main():
    long_term_memory = VectorMemory(filepath="memory_store.json")
    agent = Agent(
        model_name="gemini-3.6-flash",
        max_steps=6,
        long_term_memory=long_term_memory,
        mcp_server_script="mcp_server.py",
    )

    await agent.connect(verbose=True)

    print("=" * 60)
    print(f"Long-term memory loaded: {len(long_term_memory)} stored item(s)")
    print("Type 'quit' to exit, 'forget' to wipe short-term (this session) memory.")
    print("=" * 60)

    try:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("quit", "exit"):
                break
            if user_input.lower() == "forget":
                agent.reset_memory()
                print("(short-term/conversation memory cleared — long-term memory untouched)")
                continue
            answer = await agent.run(user_input, verbose=True)
            print(f"\n[FINAL ANSWER] {answer}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())