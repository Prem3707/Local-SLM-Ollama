# 🤖 Project 2: Local SLM App with Ollama

> 🔒 **Extended by [Offline-Voice-Assistant](https://github.com/Prem3707/Offline-Voice-Assistant)** — a fully offline, local-model version of this work (faster-whisper + Ollama + Piper, no API keys).

## What Is It?
Runs 3 small language models **entirely on your own machine** — no internet, no API keys, no cost per query, no data exposure. Benchmarks them on identical tasks and generates a quality-vs-speed tradeoff report so you can make evidence-based decisions about which model to use.

---

## Industrial Applications
| Scenario | Why Local Matters |
|---|---|
| Healthcare records | Patient data never leaves the building |
| Legal document review | Client confidentiality preserved |
| Air-gapped environments | Military, banking, government |
| Cost reduction | Zero per-token API costs |
| Edge deployment | Robots, kiosks, vehicles with no connectivity |
| Latency requirements | Sub-100ms when API round-trips are too slow |

---

## Models Compared
| Model | Size | Strength |
|---|---|---|
| `llama3.2:3b` | ~2GB | Balanced general purpose |
| `phi3.5:3.8b` | ~2.3GB | Strong reasoning for size |
| `qwen2.5:3b` | ~1.9GB | Excellent instruction following |

---

## Key Technical Concepts

### SLM (Small Language Model)
Models with 1-7B parameters that run on consumer hardware. Llama 3.2 3B runs on a laptop with 8GB RAM. Quality gap vs GPT-4 is closing fast.

### Ollama
A runtime that downloads, manages, and serves LLMs locally via a simple REST API. Think of it as Docker for language models — `ollama pull llama3.2:3b` and it runs.

### Tokens/Second
The throughput metric for LLM inference. On CPU: ~5-15 tok/s. On GPU: ~30-100 tok/s. Directly determines response latency for long outputs.

### Quality Scoring
Since we don't have ground-truth outputs for every prompt, we score by checking whether expected keywords appear in the response — a lightweight proxy for correctness.

---

## File Structure
```
02-local-slm-ollama/
├── src/
│   ├── benchmark.py      # Pull models, run all tasks, collect metrics
│   ├── chat.py           # Interactive offline chat CLI
│   └── report.py         # Generate markdown + chart from benchmark results
├── results/              # Auto-created: JSON results, chart, report
└── requirements.txt
```

---

# ▶️ How to Run

## Prerequisites
- Python 3.11+
- [Ollama installed](https://ollama.com/download) (free, open source)
- 8GB+ RAM recommended (16GB for all 3 models simultaneously)

## Setup
```bash
cd 02-local-slm-ollama
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start Ollama (in a separate terminal)
ollama serve
```

## Run Benchmarks
```bash
python -m src.benchmark
# Automatically pulls missing models (~2GB each on first run)
# Outputs results/benchmark_results.json
```

## Generate Report
```bash
python -m src.report
# Outputs results/tradeoff_report.md + results/tradeoff_chart.png
```

## Interactive Chat
```bash
python -m src.chat                      # Default: llama3.2:3b
python -m src.chat phi3.5:3.8b          # Specify model
# Type 'switch qwen2.5:3b' mid-chat to switch models
# Type 'exit' to quit
```

---

# 📐 Strategy

## Design Decisions
1. **3 runs per prompt** — median latency is more stable than single-shot
2. **Temperature 0.1 for benchmark** — low randomness = reproducible quality scoring
3. **Keyword-based quality scoring** — avoids needing an LLM judge (which would require internet)
4. **Rich CLI** — professional terminal output, easy to screenshot for portfolio
