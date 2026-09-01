# Project Map

This document is the canonical reference for where code lives and what it does in the final (post-Phase-5) state of the repository.

## Top-level layout

```
AGENTRUN/
├── README.md                  # Main project README — start here
├── LICENSE                    # License file
├── requirements.txt           # Python dependencies
├── .env.example               # Template for API keys (placeholders only)
├── .gitignore                 # Git ignore patterns (model artifacts, .env, etc.)
│
├── main.py                    # Local CLI entrypoint (Phase 1 frontier agent)
├── agent.py                   # Phase 1 Gemini-based agent (THINK/ACT/OBSERVE loop)
├── ft_agent.py                # Local client for the Colab-hosted Qwen 7B server
├── mcp_server.py              # MCP server process (tool definitions)
├── tools.py                   # Tool factory module
├── memory.py                  # Long-term vector memory (Phase 1)
│
├── providers/                 # External service adapters (Phase 1)
│   ├── __init__.py
│   ├── openmeteo_weather.py   # Open-Meteo weather API
│   ├── search_provider.py     # Search interface (abstract)
│   ├── tavily_search.py       # Tavily implementation
│   └── weather_provider.py    # Weather interface (abstract)
│
├── eval/                      # 23-task held-out benchmark framework
│   ├── __init__.py
│   ├── tasks.py               # TASKS list — 23 task definitions
│   ├── harness.py             # run_benchmark() and scoring
│   └── report.py              # print_report() and results_to_markdown()
│
├── training/                  # Training data and Colab training scripts
│   ├── training_data.jsonl                # v1: 35 clean trajectories
│   ├── training_data_augmented.jsonl      # v2: 46 trajectories (35 + 11)
│   ├── training_data_prompts.py           # Source prompts for v1
│   ├── training_data_prompts_augmented.py # Source prompts for v2
│   ├── generate_dataset.py                # Local: build v1 data from prompts
│   ├── generate_training_data.py          # Local: agent-driven v1 data generation
│   ├── augment_training_data.py           # Local: add 11 v2-specific trajectories
│   ├── colab_train_7b.py                  # Colab: train v1 LoRA (QLoRA, 4-bit)
│   ├── colab_train_7b_v2.py               # Colab: train v2 LoRA (same config)
│   └── COLAB_TRAINING_GUIDE.md            # Human-readable Colab walkthrough
│
├── serving/                   # Colab inference servers
│   ├── colab_server.py                    # v1 server
│   ├── colab_server_base.py               # Base-model server
│   └── colab_server_v2.py                 # v2 server with corrected two-pass parser
│
├── evaluation/                # Local evaluation scripts
│   ├── run_eval.py                        # Phase 1 frontier-agent eval
│   ├── run_eval_base.py                   # Base Qwen 7B eval
│   ├── run_eval_finetuned.py              # v1 LoRA eval
│   ├── run_eval_v2.py                     # v2 LoRA eval
│   ├── compare_results.py                 # Compare any two result JSONs (model-agnostic)
│   └── run_comparison.py                  # Legacy Phase 1 frontier-vs-distilled comparison
│
├── results/                   # Tracked benchmark outputs
│   ├── eval_results_base.json             # Base: 22/23
│   ├── eval_results_v1.json               # v1: 18/23
│   ├── eval_results_v2.json               # v2: 23/23
│   ├── eval_results_finetuned.json        # Historical alias of v1 (preserved)
│   ├── eval_report_base.md
│   ├── eval_report_v1.md
│   ├── eval_report_v2.md
│   ├── eval_comparison_report.md          # Latest (Base vs v2)
│   ├── eval_comparison_base_vs_v1.md      # Base vs v1
│   └── experiment_summary.json            # Full machine-readable experiment metadata
│
└── docs/
    ├── experiment-log.md                  # Phase-by-phase narrative
    ├── reproducibility.md                 # End-to-end reproduction guide
    └── project-map.md                     # This file
```

---

## Code organization rules

### Where new code should go

| Kind of code | Location |
|---|---|
| External-service integration (e.g. new search provider) | `providers/` |
| New agent runtime behavior | `ft_agent.py` (Qwen path) or `agent.py` (frontier path) |
| Long-term memory | `memory.py` |
| New tool factory | `tools.py` |
| New MCP server tool | `mcp_server.py` |
| New benchmark task | `eval/tasks.py` (only if it doesn't overlap with existing tasks) |
| New training data | `training/training_data_*.jsonl` |
| New Colab training script | `training/colab_train_*.py` |
| New Colab serving script | `serving/colab_server_*.py` |
| New evaluation script | `evaluation/run_eval_*.py` |
| New comparison report | run `evaluation/compare_results.py` |
| Project documentation | `docs/` |

### Files that should NOT contain provider logic

- `agent.py` — provider-agnostic
- `ft_agent.py` — provider-agnostic
- `tools.py` — provider-agnostic
- `memory.py` — provider-agnostic

### Files that should NOT change in a benchmark

- `eval/tasks.py` — the held-out benchmark is fixed
- `eval/harness.py` — scoring is fixed

Changing the benchmark would invalidate comparison with the published Base / v1 / v2 results.

---

## Where the experiment data lives

### Three result files, three distinct evaluations

| File | Score | Notes |
|---|---|---|
| `eval_results_base.json` | 22/23 | Qwen 7B base, no LoRA, max_steps=3 |
| `eval_results_v1.json` | 18/23 | Qwen 7B + LoRA v1 (35 trajectories), max_steps=3 |
| `eval_results_v2.json` | 23/23 | Qwen 7B + LoRA v2 (46 trajectories), max_steps=3 |

All three used:
- The same 23 tasks in the same order
- The same MCP tool infrastructure (`mcp_server.py`)
- The same base model (`Qwen/Qwen2.5-7B-Instruct`)
- Greedy decoding (`do_sample=False`)
- The same scoring harness

The only differences were the LoRA adapter (or none) and the training data.

### Historical alias

`eval_results_finetuned.json` is a bit-for-bit identical copy of `eval_results_v1.json`. It is preserved because the original Phase 4 evaluation used that filename. Do not delete it — it serves as a historical audit trail.

### Comparison reports

- `eval_comparison_report.md` — current default: Base vs LoRA v2
- `eval_comparison_base_vs_v1.md` — Base vs LoRA v1

To regenerate:
```bash
python evaluation/compare_results.py results/eval_results_base.json results/eval_results_v2.json \
    --label-a "Base Qwen 7B" --label-b "LoRA v2" \
    --output results/eval_comparison_report.md

python evaluation/compare_results.py results/eval_results_base.json results/eval_results_v1.json \
    --label-a "Base Qwen 7B" --label-b "LoRA v1" \
    --output results/eval_comparison_base_vs_v1.md
```

---

## What the model files look like (if you do download them)

If you reproduce the training, you will get two adapter directories:

```
agentlab_qwen_lora/                  # v1
├── adapter_config.json
├── adapter_model.safetensors
├── chat_template.jinja
├── tokenizer.json
├── tokenizer_config.json
└── README.md

agentlab_qwen_lora_7b_v2/            # v2
├── adapter_config.json
├── adapter_model.safetensors
├── chat_template.jinja
├── tokenizer.json
├── tokenizer_config.json
└── README.md
```

These are **NOT** committed to the repository. They are excluded by `.gitignore`. The v1 and v2 directories are not the same — different training data, different adapter weights.

---

## What the Colab server looks like

Each `serving/colab_server_*.py` is a single self-contained file designed to be pasted into Colab cells. They:

1. Install dependencies (`transformers`, `peft`, `bitsandbytes`, `fastapi`, `uvicorn`, `pyngrok`, `nest_asyncio`)
2. Load the base model in 4-bit
3. Apply the LoRA adapter from the corresponding `/content/agentlab_qwen_lora_7b[_v2]/` directory
4. Define a `/chat` endpoint that accepts a messages+tools payload
5. Define a `/health` endpoint
6. Expose the server publicly via ngrok

The v2 server (`serving/colab_server_v2.py`) is the only one with the corrected two-pass parser.

---

## What the eval scripts look like

Each `evaluation/run_eval_*.py`:

1. Connects to the local MCP server (`mcp_server.py`) for tool execution
2. Connects to a Colab-hosted Qwen model via the env var (`COLAB_SERVER_URL`, `COLAB_SERVER_URL_V2`, or via `--url`)
3. Iterates over `eval.tasks.TASKS` (23 tasks)
4. For each task: builds a message, calls the remote model, executes any tool calls, repeats until final answer or `max_steps` (3)
5. Scores the run with `eval.harness`
6. Saves results to `results/eval_results_*.json` and `results/eval_report_*.md`
