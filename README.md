# Agent From Scratch — Phase 1

A minimal agentic loop (think → act → observe) built directly on the modern
Gemini API, with no LangChain in between, so every mechanic is visible and
owned by you.

## Setup

1. Get a free API key: https://aistudio.google.com/apikey
2. Create a local `.env` file with:

```
GEMINI_API_KEY=your_key_here
```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   python main.py
   ```

The runtime uses `GEMINI_API_KEY`. Keep `GOOGLE_API_KEY` unset so the SDK
does not prefer the legacy key path.

## API Links

- Gemini API key: https://aistudio.google.com/apikey
- Tavily API key: https://tavily.com
- Open-Meteo: no API key required

## Project Map

- `agent.py` — MCP client runtime loop, short-term memory reuse, long-term memory hooks
- `memory.py` — local vector memory and cosine similarity search
- `tools.py` — tool definitions and provider-backed tool factories
- `providers/` — provider interfaces and concrete HTTP adapters
- `mcp_server.py` — standalone MCP server that exposes tools over stdio
- `main.py` — wiring for providers, tools, and the agent runtime
- `phase3files/` — archived phase references; not used by the live runtime
- `README.md` — setup, architecture, and usage notes

## Try these prompts

- `What's the weather in Bangalore?`
- `What is 45 * 12?`
- `What's the weather in Delhi, and also what's 100 / 4?` (forces multiple tool calls)
- `Count the words in "the quick brown fox jumps over the lazy dog" then multiply that by 50`
  (forces the agent to CHAIN tools — output of one feeds into the next)

Watch the terminal — it prints every THINK / ACT / OBSERVE step so you can see
the loop happening in real time, not just the final answer. The runtime now
captures structured step traces so the same execution can later power
evaluation, failure analysis, and visualization.

## What's actually happening (for your own notes / interview prep)

- `tools.py` — wraps plain Python functions so the model can "see" and request them.
  The `description` field is critical: it's the only thing the model uses to decide
  _when_ to call a tool, so vague descriptions cause bad tool selection.
- `agent.py` — the loop. The model never executes code; it only ever outputs a
  structured "I want to call X with args Y." Our code executes it and feeds the
  result back in as new context. This repeats until the model stops requesting tools.
- Tool failures are caught and returned as text observations (not crashes) — the
  model needs to see failures to recover from them.
- `max_steps` exists because ungrounded agent loops can genuinely run forever.

## Known limitations of Phase 1 (intentional — later phases fix these)

- No memory across separate `agent.run()` calls beyond one chat session
- No long-term/vector memory yet
- Tools are toy examples (calculator, mock weather, word count)
- No evaluation/benchmarking yet
- Not exposed as MCP servers yet
