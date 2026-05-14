"""
Local SLM Benchmarker using Ollama.
Runs 3 models on the same prompts, measures latency + quality, saves report.

Models compared: llama3.2:3b, phi3.5:3.8b, qwen2.5:3b
"""

import time
import json
import statistics
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

import ollama
from rich.console import Console
from rich.table import Table


console = Console()

MODELS = [
    "llama3.2:3b",
    "phi3.5:3.8b",
    "qwen2.5:3b",
]

# Benchmark prompts — diverse to test different capabilities
BENCHMARK_PROMPTS = [
    {
        "id": "reasoning",
        "prompt": "If a train travels 120 miles in 2 hours, then stops for 30 minutes, then travels 90 miles in 1.5 hours — what is the average speed for the entire journey including the stop?",
        "expected_keywords": ["60", "mph", "miles per hour"],
    },
    {
        "id": "coding",
        "prompt": "Write a Python function that checks if a string is a palindrome. Include type hints and a docstring.",
        "expected_keywords": ["def", "str", "return", "reverse"],
    },
    {
        "id": "summarization",
        "prompt": "Summarize the concept of transformer architecture in AI in 3 bullet points for a software engineer.",
        "expected_keywords": ["attention", "encoder", "decoder"],
    },
    {
        "id": "instruction_follow",
        "prompt": "List exactly 5 capitals of European countries. Format: numbered list only, no extra text.",
        "expected_keywords": ["1.", "2.", "3.", "4.", "5."],
    },
]

RESULTS_DIR = Path("results")
RUNS_PER_PROMPT = 3  # Run each prompt N times for stable latency stats


@dataclass
class RunResult:
    model: str
    prompt_id: str
    latency_ms: float
    tokens_per_second: float
    output: str
    keyword_hits: int
    keyword_total: int


def check_model_available(model: str) -> bool:
    """Check if model is pulled locally."""
    try:
        models = ollama.list()
        return any(m.model.startswith(model.split(":")[0]) for m in models.models)
    except Exception:
        return False


def pull_model_if_needed(model: str):
    """Pull model if not present."""
    if not check_model_available(model):
        console.print(f"[yellow]Pulling {model}...[/yellow]")
        ollama.pull(model)
        console.print(f"[green]✓ {model} ready[/green]")
    else:
        console.print(f"[green]✓ {model} already present[/green]")


def run_inference(model: str, prompt: str) -> Dict[str, Any]:
    """Run a single inference and return latency + response."""
    start = time.perf_counter()
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": 0.1, "num_predict": 512},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    tokens = response.get("eval_count", 0)
    duration_s = response.get("eval_duration", 1) / 1e9  # nanoseconds → seconds
    tps = tokens / duration_s if duration_s > 0 else 0

    return {
        "latency_ms": elapsed_ms,
        "tokens_per_second": tps,
        "output": response["response"],
    }


def score_output(output: str, expected_keywords: List[str]) -> int:
    """Count how many expected keywords appear in output."""
    output_lower = output.lower()
    return sum(1 for kw in expected_keywords if kw.lower() in output_lower)


def run_benchmarks() -> List[RunResult]:
    results = []

    for model in MODELS:
        pull_model_if_needed(model)
        console.print(f"\n[bold cyan]Benchmarking {model}[/bold cyan]")

        for task in BENCHMARK_PROMPTS:
            latencies = []
            tps_list = []
            last_output = ""

            for run in range(RUNS_PER_PROMPT):
                console.print(f"  [{run+1}/{RUNS_PER_PROMPT}] {task['id']}...", end="")
                try:
                    res = run_inference(model, task["prompt"])
                    latencies.append(res["latency_ms"])
                    tps_list.append(res["tokens_per_second"])
                    last_output = res["output"]
                    console.print(f" {res['latency_ms']:.0f}ms")
                except Exception as e:
                    console.print(f" ERROR: {e}")

            if latencies:
                hits = score_output(last_output, task["expected_keywords"])
                results.append(RunResult(
                    model=model,
                    prompt_id=task["id"],
                    latency_ms=statistics.median(latencies),
                    tokens_per_second=statistics.mean(tps_list),
                    output=last_output,
                    keyword_hits=hits,
                    keyword_total=len(task["expected_keywords"]),
                ))

    return results


def save_results(results: List[RunResult]):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "benchmark_results.json"
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    console.print(f"\n[green]Results saved to {path}[/green]")


def print_summary_table(results: List[RunResult]):
    table = Table(title="Model Benchmark Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Task", style="magenta")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Tokens/s", justify="right")
    table.add_column("Quality", justify="right")

    for r in results:
        quality = f"{r.keyword_hits}/{r.keyword_total}"
        table.add_row(
            r.model,
            r.prompt_id,
            f"{r.latency_ms:.0f}",
            f"{r.tokens_per_second:.1f}",
            quality,
        )

    console.print(table)


if __name__ == "__main__":
    console.print("[bold]🚀 Local SLM Benchmark — Ollama[/bold]")
    results = run_benchmarks()
    print_summary_table(results)
    save_results(results)
