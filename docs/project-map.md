# AgentLab Project Map

This file is the reference for where new code should go.

## Canonical folders

- `providers/` for external-service adapters and provider interfaces
- `agent.py` for the MCP client runtime loop and memory orchestration
- `memory.py` for retrieval and persistence logic
- `tools.py` for tool factories and structured tool schemas
- `mcp_server.py` for the MCP server process and tool registration
- `main.py` for dependency wiring and CLI entrypoint

## Add future files here

- New external integrations: `providers/`
- New agent runtime behavior: `agent.py` or a future `runtime/` package
- New memory strategies: `memory.py` or a future `memory/` package
- New tool factories: `tools.py` or a future `tools/` package
- New project notes or diagrams: `docs/`
- MCP server changes: `mcp_server.py`

## Files that should not hold provider logic

- `agent.py`
- `tools.py`
- `memory.py`

`main.py` should stay thin: it wires the runtime together, but it should
not reimplement tool logic or provider logic.

Those files should stay provider-agnostic.
