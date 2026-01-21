#!/usr/bin/env python3
"""
Create a persona from a description in one step.

This script combines dataset generation and vector extraction into a single command.

Usage:
    python scripts/create_persona.py "A grumpy but secretly kind wizard" --name grumpy_wizard
    python scripts/create_persona.py "A cheerful pirate captain" --name cheerful_pirate --prompts 5
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import get_config
from src.model_wrapper import ModelWrapper
from src.activation_extractor import ActivationExtractor
from src.dataset_generator import DatasetGenerator


def create_persona(
    description: str,
    name: str,
    num_prompts: int = 5,
    num_questions: int | None = None,
    console: Console | None = None,
) -> bool:
    """
    Create a persona from a description.
    
    Args:
        description: Natural language description of the persona.
        name: Name for the persona (used for file naming).
        num_prompts: Number of system prompts to generate.
        num_questions: Number of extraction questions (None = use existing or generate 100).
        console: Rich console for output.
        
    Returns:
        True if successful, False otherwise.
    """
    if console is None:
        console = Console()
    
    console.print(Panel.fit(
        f"[bold magenta]Creating Persona: {name}[/bold magenta]\n"
        f"[dim]{description}[/dim]",
        border_style="magenta"
    ))
    
    # Load config
    config = get_config()
    config.ensure_directories()
    
    # Initialize model
    console.print("\n[yellow]Loading model...[/yellow]")
    model = ModelWrapper(config)
    model.load()
    console.print(f"[green]✓[/green] Model loaded: {config.model_name}")
    
    generator = DatasetGenerator(model, config)
    extractor = ActivationExtractor(model, config)
    
    # Step 1: Check/generate extraction questions
    questions_path = config.personas_dir / "extraction_questions.jsonl"
    
    if not questions_path.exists():
        console.print("\n[cyan]Step 1/4: Generating extraction questions...[/cyan]")
        target_questions = num_questions or 100
        questions = generator.generate_extraction_questions(n=target_questions)
        generator.save_jsonl(questions, questions_path)
        console.print(f"[green]✓[/green] Generated {len(questions)} extraction questions")
    else:
        console.print("\n[cyan]Step 1/4: Loading existing extraction questions...[/cyan]")
        questions = generator.load_jsonl(questions_path)
        console.print(f"[green]✓[/green] Loaded {len(questions)} extraction questions")
    
    question_texts = [q["question"] for q in questions]
    
    # Step 2: Generate persona system prompts
    console.print(f"\n[cyan]Step 2/4: Generating {num_prompts} system prompts for persona...[/cyan]")
    
    persona_prompts = generator.generate_persona_prompts(description, n=num_prompts)
    
    if not persona_prompts:
        console.print("[red]Error: Failed to generate persona prompts![/red]")
        model.unload()
        return False
    
    # Add IDs and save
    for i, p in enumerate(persona_prompts):
        p["id"] = f"{name}_prompt_{i+1}"
    
    prompts_path = config.archetypes_dir / f"{name}.jsonl"
    generator.save_jsonl(persona_prompts, prompts_path)
    console.print(f"[green]✓[/green] Saved {len(persona_prompts)} prompts to {prompts_path.name}")
    
    # Show generated prompts
    console.print("\n[dim]Generated prompts:[/dim]")
    for i, p in enumerate(persona_prompts, 1):
        console.print(f"  [dim]{i}. {p['system_prompt'][:80]}...[/dim]")
    
    system_prompts = [p["system_prompt"] for p in persona_prompts]
    
    # Step 3: Extract activations
    console.print(f"\n[cyan]Step 3/4: Extracting persona activations...[/cyan]")
    console.print(f"[dim]  {len(system_prompts)} prompts × {len(question_texts)} questions = {len(system_prompts) * len(question_texts)} generations[/dim]")
    
    activations = extractor.extract_for_persona(
        system_prompts=system_prompts,
        extraction_questions=question_texts,
        show_progress=True
    )
    
    # Combine layer vectors
    valid_layers = {k: v for k, v in activations.items() if v is not None}
    
    if not valid_layers:
        console.print("[red]Error: No valid activations extracted![/red]")
        model.unload()
        return False
    
    # Compute mean across layers
    stacked = torch.stack(list(valid_layers.values()))
    raw_vector = stacked.mean(dim=0)
    raw_vector = raw_vector / raw_vector.norm()
    
    # Step 4: Save persona vector
    console.print(f"\n[cyan]Step 4/4: Saving persona vector...[/cyan]")
    
    # Save persona
    output_path = config.persona_vectors_dir / f"{name}.pt"
    torch.save({
        "raw_vector": raw_vector,
        "safe_vector": raw_vector,  # No orthogonalization - use persona mixing for safety
        "per_layer_vectors": valid_layers,
        "layers": list(valid_layers.keys()),
        "source": "generated",
        "name": name,
        "description": description,
        "num_system_prompts": len(system_prompts),
        "num_questions": len(question_texts),
        "metadata": {
            "model": config.model_name,
            "prompts_file": str(prompts_path),
        }
    }, output_path)
    
    # Cleanup
    model.unload()
    
    # Success summary
    console.print(Panel.fit(
        f"[bold green]✓ Persona '{name}' created successfully![/bold green]\n\n"
        f"Vector saved to: [cyan]{output_path}[/cyan]\n"
        f"Prompts saved to: [cyan]{prompts_path}[/cyan]\n\n"
        f"[dim]Use it with:[/dim]\n"
        f"  persona-steer chat --persona {name}\n"
        f"  persona-steer generate \"Hello\" --persona {name}",
        border_style="green"
    ))
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create a persona from a natural language description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/create_persona.py "A wise old sage" --name wise_sage
  python scripts/create_persona.py "A sarcastic robot butler" --name robot_butler --prompts 7
  python scripts/create_persona.py "A cheerful pirate who loves puns" --name punny_pirate
        """
    )
    
    parser.add_argument(
        "description",
        help="Natural language description of the persona character"
    )
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Name for the persona (used for file naming, no spaces)"
    )
    parser.add_argument(
        "--prompts", "-p",
        type=int,
        default=5,
        help="Number of system prompts to generate (default: 5)"
    )
    parser.add_argument(
        "--questions", "-q",
        type=int,
        default=None,
        help="Number of extraction questions to generate if none exist (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Validate name (no spaces, filesystem-safe)
    if " " in args.name:
        print("Error: Persona name cannot contain spaces. Use underscores instead.")
        sys.exit(1)
    
    console = Console()
    
    success = create_persona(
        description=args.description,
        name=args.name,
        num_prompts=args.prompts,
        num_questions=args.questions,
        console=console,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
