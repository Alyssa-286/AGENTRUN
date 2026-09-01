# AGENTRUN

**A controlled QLoRA distillation study of tool-use capability on Qwen2.5-7B-Instruct, evaluated against a held-out 23-task multi-tool agent benchmark.**

---

## TL;DR

We took a strong instruction-tuned 7B model (Qwen2.5-7B-Instruct), fine-tuned it with QLoRA on small numbers of clean tool-use trajectories, and measured what happened — including the regression that the first round of fine-tuning produced. We diagnosed the failure, fixed two contributing factors (a brittle tool-call parser and a gap in training-data coverage), and the resulting adapter (v2) reaches 23/23 on the same held-out benchmark where the base model was 22/23 and an early v1 attempt was 18/23.

| Model | Training trajectories | Tasks | Passed | Accuracy |
|---|---|---|---|---|
| **Base Qwen 7B** (untuned) | 0 | 23 | 22 | 95.7% |
| **LoRA v1** | 35 | 23 | 18 | 78.3% |
| **LoRA v2** | 46 | 23 | **23** | **100.0%** |

v1 → v2: **+21.7 pp** (recovered from regression)
Base → v2: **+4.3 pp** (full pass on the held-out set)

All claims are tied to the specific 23-task benchmark in `eval/tasks.py`. See [Scope & limitations](#scope--limitations).

---

## Why this project

The AgentLab project started with a frontier-API agent (Google Gemini, MCP-based tool execution) as a high-capability baseline. The natural follow-up question is: can a small open model learn the same tool-use patterns from a small number of high-quality trajectories, and what does that process actually look like?

This repository is the record of that experiment. It is intentionally an *engineering and research log*, not a leaderboard entry:

- A 23-task held-out evaluation benchmark with machine-checkable scoring
- A small SFT training set (35 trajectories for v1, 46 for v2)
- A controlled comparison using the same base model, same benchmark, same tool infrastructure, same evaluation harness, and the same `max_steps=3` setting across all three runs
- Trace-level failure analysis of v1's regression, with the diagnosed causes and the targeted interventions that produced v2

---

## Repository structure

```
AGENTRUN/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .env.example                   # API-key template (placeholders only)
│
├── agent.py                       # Phase 1 frontier-agent (Gemini)
├── ft_agent.py                    # Local client for the Colab Qwen server
├── mcp_server.py                  # MCP server (calculator, word_count, web_search, get_weather)
├── tools.py                       # Tool factory
├── memory.py                      # Long-term vector memory (Phase 1)
├── main.py                        # Local CLI entrypoint
├── providers/                     # External-service adapters
│
├── eval/                          # Benchmark framework
│   ├── tasks.py                   # 23 task definitions
│   ├── harness.py                 # Scoring harness
│   └── report.py
│
├── training/                      # Training data and Colab training scripts
│   ├── training_data.jsonl                # v1: 35 clean trajectories
│   ├── training_data_augmented.jsonl      # v2: 46 trajectories (35 + 11)
│   ├── colab_train_7b.py                  # Train v1 LoRA
│   ├── colab_train_7b_v2.py               # Train v2 LoRA
│   ├── augment_training_data.py           # Adds the 11 v2-specific trajectories
│   └── COLAB_TRAINING_GUIDE.md
│
├── serving/                       # Colab inference servers
│   ├── colab_server.py                    # v1 server
│   ├── colab_server_base.py               # Base-model server
│   └── colab_server_v2.py                 # v2 server with corrected parser
│
├── evaluation/                    # Local evaluation scripts
│   ├── run_eval_base.py                   # Eval base model
│   ├── run_eval_finetuned.py              # Eval v1
│   ├── run_eval_v2.py                     # Eval v2
│   ├── compare_results.py                 # Compare any two result JSONs
│   └── run_eval.py                        # Phase 1 frontier-agent eval
│
├── results/                       # Tracked benchmark outputs
│   ├── eval_results_base.json
│   ├── eval_results_v1.json
│   ├── eval_results_v2.json
│   ├── eval_results_finetuned.json        # Historical alias of v1
│   ├── eval_report_base.md
│   ├── eval_report_v1.md
│   ├── eval_report_v2.md
│   ├── eval_comparison_report.md
│   ├── eval_comparison_base_vs_v1.md
│   └── experiment_summary.json
│
└── docs/
    ├── experiment-log.md          # Phase-by-phase narrative
    ├── reproducibility.md         # How to reproduce end-to-end
    └── project-map.md             # Code organization reference
```

For a deeper walk, see [`docs/project-map.md`](docs/project-map.md).

---

## Architecture

### Tool-execution protocol: MCP

The agent uses the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) over stdio. The local Python script (`ft_agent.py`) runs `mcp_server.py` as a subprocess and exchanges JSON-RPC requests to invoke tools. The tools are:

- `calculator` — exact arithmetic (Python expression evaluator)
- `word_count` — word count for a string
- `web_search` — current web information (Tavily or configurable provider)
- `get_weather` — current weather for a city (Open-Meteo)

This is the same tool surface for all three evaluated models (Base, v1, v2). The MCP layer is what made the controlled comparison possible.

### QLoRA training

Fine-tuning a 7B model on a free-tier Colab T4 (15 GB VRAM) is only feasible with quantization. We used:

- 4-bit NF4 quantization for the base model
- LoRA adapters with rank 16, alpha 32, applied to all attention and MLP projections
- `paged_adamw_8bit` optimizer
- Effective batch size 16 (per-device 1, grad-accum 16)
- 3 epochs, learning rate 2e-4

A full run takes about 45 minutes on a T4. The same hyperparameters were used for v1 and v2 — only the training data changed.

### Inference: Colab-hosted, ngrok-tunneled

The 4-bit model + LoRA cannot run on consumer laptops. The Colab server (`serving/colab_server_*.py`) hosts the model on a T4 GPU and exposes a single `POST /chat` endpoint that accepts a `messages` + `tools` payload. The local eval client (`ft_agent.py`) sends requests to that endpoint, parses the model's tool-call output, executes tools locally via MCP, and feeds the tool results back to the model. This loop continues until the model produces a final answer or `max_steps=3` is reached.

---

## Experimental design

### The 23-task held-out benchmark

Defined in `eval/tasks.py`. Each task has a machine-checkable success criterion:

| Category | Tasks | What it tests |
|---|---|---|
| `math` | 3 | Basic arithmetic, order of operations |
| `text` | 1 | Word count |
| `weather` | 2 | Current weather lookup |
| `search` | 2 | Web search |
| `multi_tool` | 4 | Parallel tool use in one turn |
| `chained_reasoning` | 1 | Tool output feeds next tool call |
| `time_sensitive` | 1 | Needs live information |
| `no_tool_expected` | 2 | Direct answer, no tool needed |
| `adversarial_math` | 3 | Negative numbers, decimals, percentages |
| `adversarial_reasoning` | 2 | False premise (Wakanda GDP), chained search→calc |
| `adversarial_error_handling` | 1 | Division by zero (tool must error) |
| `adversarial_ambiguity` | 1 | Ambiguous entity (Springfield) |

Categories like `multi_tool`, `chained_reasoning`, and `adversarial_*` were added deliberately to probe known weaknesses in tool-using agents.

### Controlled comparison

All three runs used:
- The same 23 tasks in the same order
- The same `max_steps=3`
- The same MCP tool infrastructure
- The same base model (`Qwen/Qwen2.5-7B-Instruct`)
- Greedy decoding (`do_sample=False`)
- The same scoring harness

The only differences were the LoRA adapter (or none) and the training data. **The benchmark itself was not modified between runs.**

### Training data

- **v1:** 35 trajectories in `training/training_data.jsonl`, generated from a hand-curated prompt set via `training/generate_training_data.py` (with a quality filter that rejects trajectories hitting errors or exhausting max steps).
- **v2:** 46 trajectories in `training/training_data_augmented.jsonl`. 35 from v1 plus 11 new trajectories added by `training/augment_training_data.py`, targeted at the specific failure patterns v1 exhibited:

| Gap | Trajectories added |
|---|---|
| `word_count` → `calculator` | 3 |
| `get_weather` → `calculator` | 3 |
| 3-tool chain + explicit final synthesis | 2 |
| Calculator error / division by zero | 2 |
| False-premise correction | 1 |

---

## Results

### Headline

| Model | Trajectories | Tasks | Passed | Accuracy |
|---|---|---|---|---|
| Base Qwen 7B | 0 | 23 | 22 | 95.7% |
| LoRA v1 | 35 | 23 | 18 | 78.3% |
| **LoRA v2** | **46** | **23** | **23** | **100.0%** |

### Per-task outcome

The full 23×3 matrix is in [`results/eval_comparison_report.md`](results/eval_comparison_report.md) (Base vs v2) and [`results/eval_comparison_base_vs_v1.md`](results/eval_comparison_base_vs_v1.md) (Base vs v1).

### Failure-mode analysis

| Failure | Base | v1 | v2 |
|---|---|---|---|
| `adversarial_division_by_zero` (must call calculator and let it error) | ❌ | ❌ | ✅ |
| `multi_wordcount_calc` (word count → multiply) | ✅ | ❌ | ✅ |
| `multi_three_tools` (3 parallel tools) | ✅ | ❌ | ✅ |
| `chain_weather_then_calc` (use weather output as calc input) | ✅ | ❌ | ✅ |
| `adversarial_false_premise` (Wakanda GDP) | ✅ | ❌ | ✅ |

### The v1 regression — what we learned

v1 underperformed the base model by 17.4 pp. Two contributing factors were identified by trace-level analysis:

1. **Tool-call parser bug.** The v1 Colab server used a single-pass regex to extract `<tool_call>...</tool_call>` blocks. When the fine-tuned model emitted malformed tags (missing closing tag before the next opening tag), the parser returned empty and the server treated it as a final answer. The fix is in `serving/colab_server_v2.py`: a two-pass parser that falls back to extracting the first JSON object after any `<tool_call>` tag.

2. **Training data gap.** Trace-level analysis of v1 failures clustered in multi-step tasks. v1 had limited exposure to:
   - Chained tool use (one tool's output feeding another tool's input)
   - Tool-error handling (what to do when a tool returns an error)
   - False-premise reasoning

v2 was the same QLoRA configuration, the same base model, and the same benchmark — with 11 targeted trajectories added and the parser fix deployed.

---

## Reproducibility

End-to-end reproduction steps are in [`docs/reproducibility.md`](docs/reproducibility.md). The short version:

```bash
git clone https://github.com/Alyssa-286/AGENTRUN.git
cd AGENTRUN
python -m venv .venv
source .venv/bin/activate          # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Train v1 on Colab (paste training/colab_train_7b.py into a T4 cell)
# Train v2 on Colab (paste training/colab_train_7b_v2.py into a T4 cell)

# Serve v2 from Colab (paste serving/colab_server_v2.py into a T4 cell)
# Set NGROK_AUTH_TOKEN, run all cells, copy the printed ngrok URL

# Run the benchmark locally:
export COLAB_SERVER_URL_V2=https://....ngrok.io
python evaluation/run_eval_v2.py

# Generate a comparison:
python evaluation/compare_results.py \
    results/eval_results_base.json results/eval_results_v2.json \
    --label-a "Base Qwen 7B" --label-b "LoRA v2" \
    --output results/eval_comparison_report.md
```

### Expected artifacts

| File | Score |
|---|---|
| `results/eval_results_base.json` | 22/23 |
| `results/eval_results_v1.json` | 18/23 |
| `results/eval_results_v2.json` | 23/23 |

These exact values are what the repository contains. They were generated by the scripts in `evaluation/` and have not been altered.

---

## Scope & limitations

This project is honest about its scope:

- **The 23-task benchmark is the whole benchmark.** Pass-rate claims apply to this specific held-out set, not to tool use in general.
- **Base → v2 +4.3 pp is a ceiling effect, not a model improvement.** The base model was already at 22/23. v2 added the one task the base model failed. Claiming v2 is "better" than the base would overstate the evidence.
- **v2 training data was crafted with reference to v1 failures.** This is iterative engineering, not a clean ablation. The +21.7 pp jump from v1 to v2 is partly data-targeted improvement and partly the parser fix.
- **All evaluations used `max_steps=3`.** Larger step budgets might give different results, especially for the multi-tool tasks.
- **Inference uses greedy decoding.** Sampling-based decoding was not evaluated.
- **No comparison to other open models.** The comparison is only between Base, v1, and v2 — all derived from `Qwen/Qwen2.5-7B-Instruct`. No claims about other 7B models are made.
- **No claim of generalization.** The fine-tuned v2 adapter was evaluated only on this benchmark. It has not been evaluated on other agent benchmarks, on production tool-use tasks, or on safety-related adversarial cases.

---

## Security

- **No real API keys are committed.** `.env` is in `.gitignore`. `.env.example` contains placeholders only.
- **No model weights are committed.** The LoRA adapters (`*.safetensors`, `tokenizer.json`, `checkpoint-*/`) are too large for GitHub and are ignored. They are reproducible from the training scripts in `training/`.
- **If you fork and start adding your own keys**, copy `.env.example` to `.env`, fill in your real keys, and verify `.env` is still in your local `.gitignore` before any push.
- If a real key is ever accidentally committed, **revoke the key immediately** at the provider's dashboard. Removing the file from a commit does not remove it from history.

---

## Future work

Things we would explore next, but did not have time to do here:

- **Ablation: v2 with v1's parser.** This would isolate the contribution of the parser fix from the contribution of the additional training data.
- **Larger training sets.** The v2 set is 46 trajectories. Whether further targeted data would yield further gains is open.
- **Cross-benchmark evaluation.** Running v2 on a different tool-use benchmark (e.g. ToolBench, BFCL) to test for overfitting to our 23-task set.
- **Adversarial robustness.** The benchmark has 7 adversarial tasks, but a much wider set of failure modes exists.
- **Curriculum and DPO.** The current approach is SFT on filtered trajectories. A preference-optimization step (DPO/KTO) might handle some failure modes more cleanly.

---

## Citation

If you use this codebase or benchmark, please cite the model and the MCP protocol:

```bibtex
@software{agentrun,
  title  = {AgentLab: a controlled QLoRA distillation study of tool-use on Qwen2.5-7B-Instruct},
  year   = {2026},
  url    = {https://github.com/Alyssa-286/AGENTRUN},
}

@model{qwen2.5-7b,
  title  = {Qwen2.5-7B-Instruct},
  url    = {https://huggingface.co/Qwen/Qwen2.5-7B-Instruct},
}

@software{mcp,
  title  = {Model Context Protocol},
  url    = {https://modelcontextprotocol.io/},
}
```

---

## License

This project is released under the MIT License. See `LICENSE`.

The base model `Qwen/Qwen2.5-7B-Instruct` is subject to the Qwen Research License; see the model card for details.
