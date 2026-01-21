#!/usr/bin/env python3
"""Extract persona vectors from archetypes or descriptions."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from src.config import get_config
from src.model_wrapper import ModelWrapper
from src.activation_extractor import ActivationExtractor
from src.dataset_generator import DatasetGenerator, ARCHETYPES
from src.vector_math import orthogonalize_persona


def extract_from_archetype(archetype: str, console: Console) -> None:
    """Extract persona vector from an existing archetype."""
    config = get_config()
    
    # Check archetype exists
    archetype_path = config.archetypes_dir / f"{archetype}.jsonl"
    if not archetype_path.exists():
        console.print(f"[red]Error: Archetype '{archetype}' not found![/red]")
        console.print(f"Available archetypes: {', '.join(ARCHETYPES.keys())}")
        console.print("Or run generate_persona_dataset.py to create archetype files.")
        return
    
    # Check extraction questions
    questions_path = config.personas_dir / "extraction_questions.jsonl"
    if not questions_path.exists():
        console.print("[red]Error: extraction_questions.jsonl not found![/red]")
        console.print("Run generate_persona_dataset.py first.")
        return
    
    # Check for baseline (required for contrastive extraction)
    baseline_path = config.vectors_dir / "assistant_baseline.pt"
    has_baseline = baseline_path.exists()
    if not has_baseline:
        console.print("[yellow]Warning: assistant_baseline.pt not found![/yellow]")
        console.print("[yellow]Run extract_assistant_axis.py first for better results.[/yellow]")
        console.print("[yellow]Continuing without baseline (steering may be weak)...[/yellow]\n")
    
    # Safety axis is optional now
    safety_path = config.vectors_dir / "safety_axis.pt"
    has_safety = safety_path.exists()
    if not has_safety:
        console.print("[dim]Note: No safety axis found, skipping orthogonalization.[/dim]")
    
    # Load model
    console.print("\n[yellow]Loading model...[/yellow]")
    model = ModelWrapper(config)
    model.load()
    
    extractor = ActivationExtractor(model, config)
    generator = DatasetGenerator(model, config)
    
    # Load data
    console.print("\n[cyan]Loading data...[/cyan]")
    archetype_prompts = generator.load_jsonl(archetype_path)
    questions = generator.load_jsonl(questions_path)
    
    # Load baseline if available
    baseline_vector = None
    baseline_per_layer = None
    if has_baseline:
        baseline_data = torch.load(baseline_path, weights_only=False)
        baseline_vector = baseline_data["vector"]
        baseline_per_layer = baseline_data.get("per_layer_vectors", {})
        console.print("  ✓ Loaded baseline for contrastive extraction")
    
    # Load safety axis if available
    safety_axis = None
    if has_safety:
        safety_data = torch.load(safety_path, weights_only=False)
        safety_axis = safety_data["vector"]
    
    system_prompts = [p["system_prompt"] for p in archetype_prompts]
    question_texts = [q["question"] for q in questions]
    
    console.print(f"  {len(system_prompts)} system prompts")
    console.print(f"  {len(question_texts)} extraction questions")
    
    # Extract persona activations
    console.print(f"\n[green]Extracting {archetype} persona vector...[/green]")
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
        return
    
    # Compute CONTRASTIVE vectors (persona - baseline) per layer
    if baseline_per_layer:
        console.print("\n[cyan]Computing contrastive direction (persona - baseline)...[/cyan]")
        contrastive_layers = {}
        for layer_idx, persona_act in valid_layers.items():
            if layer_idx in baseline_per_layer and baseline_per_layer[layer_idx] is not None:
                # Direction = where persona is relative to baseline
                direction = persona_act - baseline_per_layer[layer_idx]
                contrastive_layers[layer_idx] = direction / direction.norm()
            else:
                contrastive_layers[layer_idx] = persona_act
        valid_layers = contrastive_layers
    
    # Compute mean across layers
    stacked = torch.stack(list(valid_layers.values()))
    raw_vector = stacked.mean(dim=0)
    raw_vector = raw_vector / raw_vector.norm()
    
    # Orthogonalize against safety axis if available
    if safety_axis is not None:
        safe_vector = orthogonalize_persona(raw_vector, safety_axis)
    else:
        safe_vector = raw_vector.clone()
    
    # Save persona
    output_path = config.persona_vectors_dir / f"{archetype}.pt"
    torch.save({
        "raw_vector": raw_vector,
        "safe_vector": safe_vector,
        "per_layer_vectors": valid_layers,
        "layers": list(valid_layers.keys()),
        "source": "archetype",
        "archetype": archetype,
        "traits": ARCHETYPES.get(archetype, {}).get("traits", []),
        "description": ARCHETYPES.get(archetype, {}).get("description", ""),
        "num_system_prompts": len(system_prompts),
        "num_questions": len(question_texts),
        "contrastive": has_baseline,  # Flag if this was contrastive extraction
        "metadata": {
            "model": config.model_name,
            "archetype_file": str(archetype_path),
        }
    }, output_path)
    
    console.print(f"\n[bold green]✓ Saved {archetype} persona to {output_path}[/bold green]")
    console.print(f"  Raw vector shape: {raw_vector.shape}")
    console.print(f"  Safe vector shape: {safe_vector.shape}")
    console.print(f"  Contrastive: {has_baseline}")
    
    # Cleanup
    model.unload()


def extract_from_description(description: str, name: str, console: Console) -> None:
    """Extract persona vector from a natural language description."""
    config = get_config()
    
    # Check extraction questions
    questions_path = config.personas_dir / "extraction_questions.jsonl"
    if not questions_path.exists():
        console.print("[red]Error: extraction_questions.jsonl not found![/red]")
        console.print("Run generate_persona_dataset.py first.")
        return
    
    # Check for baseline (required for contrastive extraction)
    baseline_path = config.vectors_dir / "assistant_baseline.pt"
    has_baseline = baseline_path.exists()
    if not has_baseline:
        console.print("[yellow]Warning: assistant_baseline.pt not found![/yellow]")
        console.print("[yellow]Run extract_assistant_axis.py first for better results.[/yellow]")
        console.print("[yellow]Continuing without baseline (steering may be weak)...[/yellow]\n")
    
    # Safety axis is optional now
    safety_path = config.vectors_dir / "safety_axis.pt"
    has_safety = safety_path.exists()
    if not has_safety:
        console.print("[dim]Note: No safety axis found, skipping orthogonalization.[/dim]")
    
    # Load model
    console.print("\n[yellow]Loading model...[/yellow]")
    model = ModelWrapper(config)
    model.load()
    
    extractor = ActivationExtractor(model, config)
    generator = DatasetGenerator(model, config)
    
    # Generate system prompts from description
    console.print("\n[magenta]Generating system prompts from description...[/magenta]")
    persona_prompts = generator.generate_persona_prompts(description, n=5)
    
    if not persona_prompts:
        console.print("[red]Error: Failed to generate persona prompts![/red]")
        model.unload()
        return
    
    # Save generated prompts
    prompts_path = config.archetypes_dir / f"{name}.jsonl"
    for i, p in enumerate(persona_prompts):
        p["id"] = f"{name}_prompt_{i+1}"
    generator.save_jsonl(persona_prompts, prompts_path)
    console.print(f"  Saved {len(persona_prompts)} prompts to {prompts_path}")
    
    # Load questions
    questions = generator.load_jsonl(questions_path)
    
    # Load baseline if available
    baseline_vector = None
    baseline_per_layer = None
    if has_baseline:
        baseline_data = torch.load(baseline_path, weights_only=False)
        baseline_vector = baseline_data["vector"]
        baseline_per_layer = baseline_data.get("per_layer_vectors", {})
        console.print("  ✓ Loaded baseline for contrastive extraction")
    
    # Load safety axis if available
    safety_axis = None
    if has_safety:
        safety_data = torch.load(safety_path, weights_only=False)
        safety_axis = safety_data["vector"]
    
    system_prompts = [p["system_prompt"] for p in persona_prompts]
    question_texts = [q["question"] for q in questions]
    
    # Extract persona activations
    console.print(f"\n[green]Extracting {name} persona vector...[/green]")
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
        return
    
    # Compute CONTRASTIVE vectors (persona - baseline) per layer
    if baseline_per_layer:
        console.print("\n[cyan]Computing contrastive direction (persona - baseline)...[/cyan]")
        contrastive_layers = {}
        for layer_idx, persona_act in valid_layers.items():
            if layer_idx in baseline_per_layer and baseline_per_layer[layer_idx] is not None:
                # Direction = where persona is relative to baseline
                direction = persona_act - baseline_per_layer[layer_idx]
                contrastive_layers[layer_idx] = direction / direction.norm()
            else:
                contrastive_layers[layer_idx] = persona_act
        valid_layers = contrastive_layers
    
    # Compute mean across layers
    stacked = torch.stack(list(valid_layers.values()))
    raw_vector = stacked.mean(dim=0)
    raw_vector = raw_vector / raw_vector.norm()
    
    # Orthogonalize against safety axis if available
    if safety_axis is not None:
        safe_vector = orthogonalize_persona(raw_vector, safety_axis)
    else:
        safe_vector = raw_vector.clone()
    
    # Save persona
    output_path = config.persona_vectors_dir / f"{name}.pt"
    torch.save({
        "raw_vector": raw_vector,
        "safe_vector": safe_vector,
        "per_layer_vectors": valid_layers,
        "layers": list(valid_layers.keys()),
        "source": "generated",
        "name": name,
        "description": description,
        "num_system_prompts": len(system_prompts),
        "num_questions": len(question_texts),
        "contrastive": has_baseline,  # Flag if this was contrastive extraction
        "metadata": {
            "model": config.model_name,
            "prompts_file": str(prompts_path),
        }
    }, output_path)
    
    console.print(f"\n[bold green]✓ Saved {name} persona to {output_path}[/bold green]")
    console.print(f"  Raw vector shape: {raw_vector.shape}")
    console.print(f"  Safe vector shape: {safe_vector.shape}")
    console.print(f"  Contrastive: {has_baseline}")
    
    # Cleanup
    model.unload()


def extract_all_archetypes(console: Console) -> None:
    """Extract vectors for all base archetypes."""
    console.print("\n[bold]Extracting all base archetypes...[/bold]")
    
    for archetype in ARCHETYPES.keys():
        console.print(f"\n{'='*50}")
        console.print(f"[bold cyan]Archetype: {archetype}[/bold cyan]")
        extract_from_archetype(archetype, console)


def main():
    """Main entry point for persona extraction."""
    console = Console()
    
    parser = argparse.ArgumentParser(
        description="Extract persona vectors from archetypes or descriptions"
    )
    parser.add_argument(
        "--from-archetype", "-a",
        help="Extract from an existing archetype (sage, trickster, guardian, explorer, creator)"
    )
    parser.add_argument(
        "--from-description", "-d",
        help="Generate and extract from a natural language description"
    )
    parser.add_argument(
        "--name", "-n",
        help="Name for the persona (required with --from-description)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all base archetypes"
    )
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold magenta]Persona Vector Extractor[/bold magenta]\n"
        "Extract persona steering vectors from archetypes or descriptions",
        border_style="magenta"
    ))
    
    if args.all:
        extract_all_archetypes(console)
    elif args.from_archetype:
        extract_from_archetype(args.from_archetype, console)
    elif args.from_description:
        if not args.name:
            console.print("[red]Error: --name is required with --from-description[/red]")
            return
        extract_from_description(args.from_description, args.name, console)
    else:
        # Interactive mode
        console.print("\n[yellow]No arguments provided. Running interactively.[/yellow]")
        
        mode = Prompt.ask(
            "Extract from",
            choices=["archetype", "description", "all"],
            default="archetype"
        )
        
        if mode == "all":
            extract_all_archetypes(console)
        elif mode == "archetype":
            archetype = Prompt.ask(
                "Archetype name",
                choices=list(ARCHETYPES.keys()),
                default="sage"
            )
            extract_from_archetype(archetype, console)
        else:
            description = Prompt.ask("Character description")
            name = Prompt.ask("Persona name (no spaces)")
            extract_from_description(description, name, console)


if __name__ == "__main__":
    main()
