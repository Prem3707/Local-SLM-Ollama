"""
Generates a visual quality-vs-speed tradeoff report from benchmark results.
Outputs: results/tradeoff_report.md + a matplotlib chart.
"""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


RESULTS_PATH = Path("results/benchmark_results.json")
REPORT_PATH = Path("results/tradeoff_report.md")
CHART_PATH = Path("results/tradeoff_chart.png")

MODEL_COLORS = {
    "llama3.2:3b": "#4e9af1",
    "phi3.5:3.8b": "#f18f4e",
    "qwen2.5:3b": "#4ef1a0",
}


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def aggregate_by_model(results):
    """Average metrics per model across all tasks."""
    model_data = defaultdict(lambda: {"latencies": [], "tps": [], "quality": []})
    for r in results:
        model_data[r["model"]]["latencies"].append(r["latency_ms"])
        model_data[r["model"]]["tps"].append(r["tokens_per_second"])
        quality = r["keyword_hits"] / r["keyword_total"] if r["keyword_total"] > 0 else 0
        model_data[r["model"]]["quality"].append(quality)

    summary = {}
    for model, data in model_data.items():
        summary[model] = {
            "avg_latency_ms": np.mean(data["latencies"]),
            "avg_tps": np.mean(data["tps"]),
            "avg_quality": np.mean(data["quality"]),
        }
    return summary


def generate_chart(summary):
    models = list(summary.keys())
    latencies = [summary[m]["avg_latency_ms"] for m in models]
    quality = [summary[m]["avg_quality"] * 100 for m in models]
    colors = [MODEL_COLORS.get(m, "#aaa") for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Quality vs Speed Tradeoff — Local SLM Benchmark", fontsize=14, fontweight="bold")

    # Latency bar chart
    ax1.barh(models, latencies, color=colors)
    ax1.set_xlabel("Median Latency (ms) ← Lower is better")
    ax1.set_title("Inference Speed")
    for i, v in enumerate(latencies):
        ax1.text(v + 5, i, f"{v:.0f}ms", va="center")

    # Quality bar chart
    ax2.barh(models, quality, color=colors)
    ax2.set_xlabel("Keyword Hit Rate (%) → Higher is better")
    ax2.set_title("Output Quality")
    ax2.set_xlim(0, 100)
    for i, v in enumerate(quality):
        ax2.text(v + 1, i, f"{v:.0f}%", va="center")

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {CHART_PATH}")


def generate_markdown_report(summary, raw_results):
    lines = [
        "# 📊 Local SLM Benchmark Report\n",
        "## Summary: Quality vs Speed Tradeoff\n",
        "| Model | Avg Latency (ms) | Tokens/s | Quality Score |",
        "|---|---|---|---|",
    ]
    for model, data in summary.items():
        lines.append(
            f"| {model} | {data['avg_latency_ms']:.0f} | {data['avg_tps']:.1f} | {data['avg_quality']*100:.0f}% |"
        )

    lines += [
        "\n## Tradeoff Analysis\n",
        "See `tradeoff_chart.png` for visual comparison.\n",
        "\n## Per-Task Results\n",
        "| Model | Task | Latency (ms) | Quality |",
        "|---|---|---|---|",
    ]
    for r in raw_results:
        q = f"{r['keyword_hits']}/{r['keyword_total']}"
        lines.append(f"| {r['model']} | {r['prompt_id']} | {r['latency_ms']:.0f} | {q} |")

    lines += [
        "\n## Recommendations\n",
        "- **Privacy-first**: All models run 100% offline — no data sent externally.",
        "- **Latency-critical** (edge, real-time): Choose the fastest model.",
        "- **Quality-critical** (document analysis): Choose highest quality score.",
        "- **Balanced** (general chat): Use the best quality-per-millisecond ratio.",
    ]

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    raw = load_results()
    summary = aggregate_by_model(raw)
    generate_chart(summary)
    generate_markdown_report(summary, raw)
    print("✅ Report generation complete.")
