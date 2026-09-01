# Reproducibility Guide

This document describes how to reproduce the AgentLab QLoRA distillation experiment from scratch.

---

## 1. Environment Assumptions

### Hardware

- **Training:** Google Colab free-tier T4 GPU (15GB VRAM, 4-bit NF4 quantization required)
- **Local evaluation:** Any machine with Python 3.10+ (no GPU required for the eval client)
- **Colab server:** T4 GPU required for inference; Qwen 7B base + 4-bit LoRA fits on T4 VRAM

### Software

- Python 3.10 or 3.11
- See `requirements.txt` at the project root
- For Colab training/serving, packages are installed inside the notebook cells (no local install needed for the Colab-side scripts)

---

## 2. Repository Layout (for reproduction)

```
AGENTRUN/
├── README.md                              # Project overview
├── .env.example                           # Template for API keys (DO NOT commit .env)
├── requirements.txt                       # Python dependencies
├── main.py                                # Local CLI entrypoint (Phase 1 frontier agent)
├── agent.py                               # Phase 1 Gemini-based agent
├── ft_agent.py                            # Local client for the Colab-hosted Qwen server
├── mcp_server.py                          # MCP server process (tool definitions)
├── tools.py                               # Tool factory module
├── memory.py                              # Long-term vector memory
├── providers/                             # External service adapters
│   ├── openmeteo_weather.py
│   ├── search_provider.py
│   ├── tavily_search.py
│   └── weather_provider.py
├── eval/                                  # Evaluation framework
│   ├── harness.py
│   ├── tasks.py                           # 23-task held-out benchmark
│   └── report.py
├── training/                              # Training data and Colab training scripts
│   ├── training_data.jsonl                # v1: 35 trajectories
│   ├── training_data_augmented.jsonl      # v2: 46 trajectories
│   ├── training_data_prompts.py           # Source prompts for v1
│   ├── training_data_prompts_augmented.py # Source prompts for v2
│   ├── generate_dataset.py                # Local data synthesis (optional)
│   ├── generate_training_data.py          # Local data generation (optional)
│   ├── augment_training_data.py           # Adds the 11 v2 augmentation trajectories
│   ├── colab_train_7b.py                  # Colab: train v1 LoRA
│   ├── colab_train_7b_v2.py               # Colab: train v2 LoRA
│   └── COLAB_TRAINING_GUIDE.md            # Human-readable Colab walkthrough
├── serving/                               # Colab inference servers
│   ├── colab_server.py                    # v1 server
│   ├── colab_server_base.py               # Base-model server (for the base eval)
│   └── colab_server_v2.py                 # v2 server with corrected parser
├── evaluation/                            # Local evaluation scripts
│   ├── run_eval.py                        # Frontier-agent eval
│   ├── run_eval_base.py                   # Base Qwen 7B eval
│   ├── run_eval_finetuned.py              # v1 LoRA eval
│   ├── run_eval_v2.py                     # v2 LoRA eval
│   ├── compare_results.py                 # Compare any two result JSONs
│   └── run_comparison.py                  # Legacy Phase 1 comparison
├── results/                               # Tracked benchmark outputs
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
└── docs/
    ├── experiment-log.md                  # Phase-by-phase narrative
    ├── reproducibility.md                 # This file
    └── project-map.md                     # Code organization reference
```

---

## 3. Reproduction Steps

### Step 1: Local environment

```bash
git clone https://github.com/Alyssa-286/AGENTRUN.git
cd AGENTRUN
python -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

For the Phase 1 frontier-agent path, copy `.env.example` to `.env` and fill in `GEMINI_API_KEY` and `TAVILY_API_KEY`. **Never commit `.env`.**

### Step 2: Reproduce the training data (optional)

The training data JSONLs are already committed (`training/training_data.jsonl`, `training/training_data_augmented.jsonl`). To regenerate from scratch:

```bash
cd training
python generate_dataset.py            # produces training_data.jsonl
python augment_training_data.py       # produces training_data_augmented.jsonl
```

Both scripts are deterministic given the same prompts.

### Step 3: Train v1 LoRA on Colab

1. Open Google Colab, set runtime to T4 GPU
2. Upload `training/training_data.jsonl` to Colab
3. Open `training/colab_train_7b.py`, paste cells, run
4. Output: `/content/agentlab_qwen_lora_7b` (LoRA adapter)
5. **Back this up locally** — the `/content/` filesystem is volatile

### Step 4: Train v2 LoRA on Colab

1. Open Google Colab, set runtime to T4 GPU
2. Upload `training/training_data_augmented.jsonl` to Colab
3. Open `training/colab_train_7b_v2.py`, paste cells, run
4. Output: `/content/agentlab_qwen_lora_7b_v2` (LoRA adapter)

### Step 5: Run the Colab inference server

1. With the trained adapter in `/content/agentlab_qwen_lora_7b_v2`, open `serving/colab_server_v2.py`
2. Replace `NGROK_AUTH_TOKEN = "PASTE_YOUR_NGROK_TOKEN_HERE"` with your free ngrok authtoken from <https://dashboard.ngrok.com/get-started/your-authtoken>
3. Run the cells in order
4. Copy the printed public URL (e.g., `https://....ngrok.io`)

### Step 6: Run the 23-task benchmark

```bash
# Set the ngrok URL on your local machine
export COLAB_SERVER_URL_V2=https://....ngrok.io   # Mac/Linux
set COLAB_SERVER_URL_V2=https://....ngrok.io      # Windows

# Run from the project root
python evaluation/run_eval_v2.py
```

Expected output:
- `results/eval_results_v2.json` — machine-readable results
- `results/eval_report_v2.md` — human-readable report
- 23/23 = 100.0% pass rate

### Step 7: Generate a comparison report

```bash
python evaluation/compare_results.py results/eval_results_base.json results/eval_results_v2.json \
    --label-a "Base Qwen 7B" --label-b "LoRA v2" \
    --output results/eval_comparison_report.md
```

The `compare_results.py` script is model-agnostic: it reads labels from the JSON metadata and accepts explicit overrides.

---

## 4. LoRA Training Configuration (v1 and v2)

Both v1 and v2 used identical hyperparameters, except the training data file:

| Hyperparameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Quantization | 4-bit NF4, fp16 compute |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Optimizer | paged_adamw_8bit |
| Learning rate | 2e-4 |
| Batch size | 1 (with grad accumulation 16) |
| Epochs | 3 |
| Max seq length | 2048 |

Source: `training/colab_train_7b.py` and `training/colab_train_7b_v2.py`.

---

## 5. Benchmark Specificity

**All claims of pass rate in this repository are tied to this specific 23-task held-out set in `eval/tasks.py`.** Generalization beyond this benchmark is not implied.

| Score | Conditions |
|---|---|
| 22/23 (Base) | max_steps=3, 4-bit NF4, greedy decoding |
| 18/23 (v1)   | max_steps=3, 4-bit NF4, greedy decoding, original parser |
| 23/23 (v2)   | max_steps=3, 4-bit NF4, greedy decoding, corrected parser |

Changing any of the following invalidates comparison: benchmark tasks, max_steps, generation mode, parser, base model.

---

## 6. Security Precautions

**Before pushing to GitHub:**

1. Verify `.gitignore` is in place (it is at the project root)
2. **Never** commit:
   - `.env` files with real API keys
   - `*.safetensors` (LoRA weights — too large and not version-controlled)
   - `checkpoint-*/` directories
   - `agentlab_qwen_lora*/` directories (local adapter copies)
   - `tokenizer.json` (re-downloadable from HuggingFace)
   - `memory_store.json` (contains user-specific data)
3. If a real API key is accidentally committed:
   - **Immediately rotate the key** (revoke it at the provider's dashboard)
   - Remove the key from the file
   - Use `git filter-branch` or BFG Repo-Cleaner to remove from history
4. The `.env.example` file contains placeholders only — safe to commit

---

## 7. Expected Artifact Files

After a successful end-to-end reproduction, you should have:

| File | Source |
|---|---|
| `agentlab_qwen_lora/` (local backup) | v1 training output |
| `agentlab_qwen_lora_7b_v2/` (local backup) | v2 training output |
| `results/eval_results_base.json` | `evaluation/run_eval_base.py` |
| `results/eval_results_v1.json` | `evaluation/run_eval_finetuned.py` |
| `results/eval_results_v2.json` | `evaluation/run_eval_v2.py` |
| `results/eval_report_*.md` | (same scripts) |
| `results/eval_comparison_report.md` | `evaluation/compare_results.py` |

---

## 8. Differences from the original Phase 1 frontier-agent code

This repository contains both the Phase 1 frontier-agent code and the Phase 4–5 QLoRA distillation code. The two are decoupled:

- **Phase 1** is the `agent.py` + `providers/` + `mcp_server.py` infrastructure, originally built around Gemini. It still functions; it is not the subject of the published experiment.
- **Phase 4–5** is the QLoRA experiment (Base / v1 / v2) on Qwen 7B. This is what the README and results sections describe.

If you want to reproduce **only** the QLoRA experiment, the minimum needed is:
- `ft_agent.py` (local client)
- `eval/` (benchmark)
- `mcp_server.py` + `tools.py` + `providers/` (MCP infrastructure for tool execution)
- `training/training_data_*.jsonl` (training data)
- `training/colab_train_*.py` (training scripts)
- `serving/colab_server_*.py` (Colab servers)
- `evaluation/run_eval_*.py` + `compare_results.py` (eval pipeline)
