"""
Offline Chat CLI — run a private, fully local conversation.
No data leaves your machine. Supports model switching mid-chat.
"""

import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
import ollama


console = Console()
DEFAULT_MODEL = "llama3.2:3b"


def chat(model: str = DEFAULT_MODEL):
    """Interactive chat loop with chosen model."""
    console.print(f"\n[bold green]🔒 Offline Chat — {model}[/bold green]")
    console.print("[dim]Type 'exit' to quit, 'switch <model>' to change model[/dim]\n")

    history = []

    while True:
        user_input = Prompt.ask("[bold blue]You[/bold blue]")

        if user_input.lower() == "exit":
            break

        if user_input.lower().startswith("switch "):
            model = user_input.split(" ", 1)[1].strip()
            console.print(f"[yellow]Switched to {model}[/yellow]")
            continue

        history.append({"role": "user", "content": user_input})

        try:
            response = ollama.chat(
                model=model,
                messages=history,
                options={"temperature": 0.7, "num_predict": 1024},
            )
            assistant_msg = response["message"]["content"]
            history.append({"role": "assistant", "content": assistant_msg})

            console.print("\n[bold green]Assistant:[/bold green]")
            console.print(Markdown(assistant_msg))
            console.print()

        except ollama.ResponseError as e:
            console.print(f"[red]Error: {e}. Is Ollama running? Try: ollama serve[/red]")
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    chat(model)
