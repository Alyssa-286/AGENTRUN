# AGENTRUN

**Failure-driven QLoRA specialization of a 7B tool-using language model, evaluated on a fixed 23-task multi-tool benchmark.**

## Overview

AGENTRUN is an experimental agent system for studying whether a small instruction-tuned open model can be specialized for structured tool use with a small number of high-quality trajectories.

The project combines:

- a multi-step agent loop with explicit tool execution and observations;
- MCP-based local tool execution;
- Qwen2.5-7B-Instruct as the open-model backbone;
- parameter-efficient QLoRA fine-tuning under constrained GPU memory;
- a machine-checkable 23-task held-out benchmark;
- trace-level failure analysis and targeted training-data augmentation.

The central experiment is intentionally iterative: establish a baseline, fine-tune, inspect failures, intervene, and evaluate again on the unchanged benchmark.

## Results

| Model | Training trajectories | Tasks | Passed | Accuracy |
|---|---:|---:|---:|---:|
| Base Qwen 7B | 0 | 23 | 22 | 95.7% |
| LoRA v1 | 35 | 23 | 18 | 78.3% |
| **LoRA v2** | **46** | **23** | **23** | **100.0%** |

- **v1 → v2:** +21.7 percentage points
- **Base → v2:** +4.3 percentage points

These numbers apply only to the specific held-out benchmark in `eval/tasks.py`. They are not claims of general tool-use reliability or state-of-the-art performance.

## Research Question

> Can a 7B instruction-tuned language model be specialized for reliable multi-tool agent behavior using a small trajectory dataset, and can failure-driven trajectory augmentation recover behaviors missed by an initial fine-tuning run?

A secondary objective is to distinguish model-behavior failures from inference/protocol failures by inspecting execution traces rather than relying only on aggregate accuracy.

## Architecture

```text
                     ┌─────────────────────────┐
                     │      User Request       │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │  Qwen2.5-7B-Instruct    │
                     │   + LoRA adapter        │
                     └────────────┬────────────┘
                                  │ tool call / answer
                                  ▼
                     ┌─────────────────────────┐
                     │   Tool-call parser      │
                     │  (v2 robust fallback)   │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │     MCP tool layer      │
                     │ calculator              │
                     │ word_count              │
                     │ web_search              │
                     │ get_weather             │
                     └────────────┬────────────┘
                                  │ observation
                                  └──────────────► model
```

The local client executes tools through MCP and feeds structured observations back to the remote Qwen model. Evaluation uses the same tool surface and execution harness across Base, v1, and v2.

## Tooling

The four benchmark tools are:

| Tool | Purpose |
|---|---|
| `calculator` | exact arithmetic |
| `word_count` | word-count computation |
| `web_search` | live web retrieval |
| `get_weather` | current weather retrieval |

## QLoRA Configuration

The actual v1/v2 training script is the source of truth for training configuration. The core settings are:

| Parameter | Setting |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Quantization | 4-bit NF4 |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Per-device batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Maximum sequence length | 1024 |
| Decoding for evaluation | greedy (`do_sample=False`) |
| Evaluation step budget | `max_steps=3` |

For v2, memory-saving measures were added so the 46-trajectory run could fit on a free-tier T4: gradient checkpointing, `use_cache=False`, input-gradient support for checkpointing, an 8-bit paged AdamW optimizer, and explicit CUDA cache cleanup.

## Experimental Progression

### Base

The untuned Qwen2.5-7B-Instruct model achieved **22/23 (95.7%)** on the fixed benchmark.

### LoRA v1

A first LoRA adapter was trained on **35 trajectories**. It achieved **18/23 (78.3%)**, a regression relative to the base model.

Trace analysis exposed several distinct failure modes, including prematurely stopping after one tool, manually computing values that the benchmark required the calculator to produce, sequential multi-tool execution that exhausted the fixed step budget, and explicit tool-error handling failures.

A separate trace also exposed a serving-layer parser weakness: malformed/nested `<tool_call>` tags could cause a valid intended tool request to be missed.

### Intervention

The parser was hardened and the training set was augmented with **11 targeted trajectories**, bringing the dataset from 35 to 46 examples. The added examples specifically covered:

- `word_count` → `calculator` chains;
- `get_weather` → `calculator` chains;
- 3-tool sequences with explicit final synthesis;
- calculator error / division-by-zero behavior;
- false-premise correction.

The 23 evaluation prompts remained unchanged and were kept disjoint from training prompts.

### LoRA v2

The resulting v2 adapter achieved **23/23 (100.0%)** on the same held-out benchmark.

## Benchmark

The 23 tasks cover:

| Category | Tasks |
|---|---:|
| Math | 3 |
| Text | 1 |
| Weather | 2 |
| Search | 2 |
| Multi-tool | 4 |
| Chained reasoning | 1 |
| Time-sensitive | 1 |
| No-tool expected | 2 |
| Adversarial math | 3 |
| Adversarial reasoning | 2 |
| Adversarial error handling | 1 |
| Adversarial ambiguity | 1 |

All three evaluated Qwen conditions use the same 23 tasks, same tool infrastructure, same scoring harness, same `max_steps=3`, and greedy decoding.

## Key Failure-Mode Recovery

| Task / behavior | Base | v1 | v2 |
|---|:---:|:---:|:---:|
| `multi_wordcount_calc` | ✓ | ✗ | ✓ |
| `multi_three_tools` | ✓ | ✗ | ✓ |
| `chain_weather_then_calc` | ✓ | ✗ | ✓ |
| `adversarial_false_premise` | ✓ | ✗ | ✓ |
| `adversarial_division_by_zero` | ✗ | ✗ | ✓ |
| `adversarial_chained_search_calc` | ✓ | trace/parser issue | ✓ |

The v1→v2 improvement should **not** be attributed solely to the 11 new trajectories because the serving-layer parser was also corrected between the effective v1 and v2 systems.

## Reproducibility

See [`docs/reproducibility.md`](docs/reproducibility.md) for the full procedure.

Typical workflow:

```bash
git clone https://github.com/Alyssa-286/AGENTRUN.git
cd AGENTRUN
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Training is performed on a GPU-backed Colab runtime using the scripts in `training/`. Inference for the 7B model is served from Colab and reached by the local evaluator through a temporary tunnel.

Run the final evaluator with a live v2 server URL:

```powershell
$env:COLAB_SERVER_URL_V2="https://<your-v2-server>"
python evaluation/run_eval_v2.py
```

The benchmark writes machine-readable JSON and a Markdown report under `results/`.

## Repository Layout

```text
AGENTRUN/
├── agent.py                     # original frontier-model agent
├── ft_agent.py                  # local remote-Qwen agent client
├── mcp_server.py                # MCP tool server
├── tools.py                     # tool definitions
├── memory.py                    # local vector memory
├── providers/                  # external-service adapters
├── eval/                       # 23-task benchmark + harness
├── training/                   # datasets and QLoRA training scripts
├── serving/                    # Base/v1/v2 Colab servers
├── evaluation/                # benchmark runners + comparisons
├── results/                   # verified benchmark artifacts
└── docs/                      # experiment log + reproducibility docs
```

## Scope and Limitations

This project is deliberately conservative about its conclusions:

- The benchmark contains 23 tasks; all pass-rate claims refer only to this benchmark.
- The Base model already scored 22/23, so the Base→v2 gain is a one-task ceiling-effect improvement.
- v2 was developed after observing v1 failures, so the v1→v2 result is an iterative intervention rather than a blind pre-registered ablation.
- The parser correction and training-data augmentation changed together; their individual contributions are therefore not separately identified.
- `max_steps=3` is fixed for comparability but can disadvantage models that emit multi-tool calls sequentially rather than in parallel.
- Only one base model family and one decoding mode were evaluated.
- No independent external agent benchmark was used to establish cross-benchmark generalization.

## Future Work

The most informative next experiments would be:

1. parser-only ablation;
2. training-data-only ablation;
3. evaluation on an independent tool-use benchmark;
4. multiple seeds and larger datasets;
5. broader adversarial and error-recovery evaluation.

## Security

No real API keys, passwords, ngrok tokens, or `.env` files should be committed. Large trained model weights are intentionally kept outside the Git repository; the training scripts provide the reproducible route to recreate the adapters.

## Citation

A formal manuscript for this work is being prepared from the experiment log and verified artifacts in this repository.
