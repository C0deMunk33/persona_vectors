#!/usr/bin/env python3
"""Generate persona datasets (extraction questions and archetypes)."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

from src.config import get_config
from src.model_wrapper import ModelWrapper
from src.dataset_generator import DatasetGenerator, ARCHETYPES


def main():
    """Generate persona datasets."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold magenta]Persona Dataset Generator[/bold magenta]\n"
        "Generates extraction questions and archetype system prompts",
        border_style="magenta"
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
    
    # Generate extraction questions
    console.print("\n[cyan]Generating extraction questions...[/cyan]")
    questions = generator.generate_extraction_questions(n=100)
    questions_path = config.personas_dir / "extraction_questions.jsonl"
    generator.save_jsonl(questions, questions_path)
    console.print(f"  Saved {len(questions)} questions to {questions_path}")
    
    # Generate archetype system prompts
    console.print("\n[magenta]Generating archetype prompts...[/magenta]")
    
    for archetype_name, archetype_info in ARCHETYPES.items():
        console.print(f"  [dim]Generating {archetype_name}...[/dim]")
        
        prompts = generator.generate_archetype_prompts(
            archetype=archetype_name,
            traits=archetype_info["traits"],
            n=5
        )
        
        # Add IDs
        for i, prompt in enumerate(prompts):
            prompt["id"] = f"{archetype_name}_prompt_{i+1}"
        
        archetype_path = config.archetypes_dir / f"{archetype_name}.jsonl"
        generator.save_jsonl(prompts, archetype_path)
        console.print(f"    Saved {len(prompts)} prompts to {archetype_path}")
    
    # Cleanup
    model.unload()
    
    console.print("\n[bold green]✓ Persona datasets generated successfully![/bold green]")


if __name__ == "__main__":
    main()
