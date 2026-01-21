#!/usr/bin/env python3
"""Generate safety datasets (safe and unsafe prompts) for axis extraction."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

from src.config import get_config
from src.model_wrapper import ModelWrapper
from src.dataset_generator import DatasetGenerator


def main():
    """Generate safety datasets."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]Safety Dataset Generator[/bold blue]\n"
        "Generates safe and unsafe prompts for safety axis extraction",
        border_style="blue"
    ))
    
    # Load config
    config = get_config()
    config.ensure_directories()
    
    # Initialize model
    console.print("\n[yellow]Loading model...[/yellow]")
    model = ModelWrapper(config)
    model.load()
    
    # Create generator
    generator = DatasetGenerator(model, config)
    
    # Generate safe prompts
    console.print("\n[green]Generating safe prompts...[/green]")
    safe_prompts = generator.generate_safe_prompts(n=200)
    safe_path = config.safety_dir / "safe_prompts.jsonl"
    generator.save_jsonl(safe_prompts, safe_path)
    console.print(f"  Saved {len(safe_prompts)} safe prompts to {safe_path}")
    
    # Generate unsafe prompts
    console.print("\n[red]Generating unsafe prompts...[/red]")
    unsafe_prompts = generator.generate_unsafe_prompts(n=200)
    unsafe_path = config.safety_dir / "unsafe_prompts.jsonl"
    generator.save_jsonl(unsafe_prompts, unsafe_path)
    console.print(f"  Saved {len(unsafe_prompts)} unsafe prompts to {unsafe_path}")
    
    # Cleanup
    model.unload()
    
    console.print("\n[bold green]✓ Safety datasets generated successfully![/bold green]")


if __name__ == "__main__":
    main()
