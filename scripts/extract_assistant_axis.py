#!/usr/bin/env python3
"""Extract the assistant baseline vector and compute assistant axis."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from rich.console import Console
from rich.panel import Panel

from src.config import get_config
from src.model_wrapper import ModelWrapper
from src.activation_extractor import ActivationExtractor
from src.dataset_generator import DatasetGenerator


def main():
    """Extract assistant baseline vector."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]Assistant Axis Extractor[/bold blue]\n"
        "Extracts baseline assistant behavior vector (no system prompt)",
        border_style="blue"
    ))
    
    # Load config
    config = get_config()
    config.ensure_directories()
    
    # Check for extraction questions
    questions_path = config.personas_dir / "extraction_questions.jsonl"
    if not questions_path.exists():
        console.print("[red]Error: extraction_questions.jsonl not found![/red]")
        console.print("Run generate_persona_dataset.py first.")
        return
    
    # Initialize model
    console.print("\n[yellow]Loading model...[/yellow]")
    model = ModelWrapper(config)
    model.load()
    
    # Create extractor and generator
    extractor = ActivationExtractor(model, config)
    generator = DatasetGenerator(model, config)
    
    # Load extraction questions
    console.print("\n[cyan]Loading extraction questions...[/cyan]")
    questions = generator.load_jsonl(questions_path)
    console.print(f"  Loaded {len(questions)} questions")
    
    # Build prompts with NO system prompt (default assistant behavior)
    prompts = [{"system": None, "user": q["question"]} for q in questions]
    
    # Extract activations
    console.print("\n[green]Extracting assistant baseline activations...[/green]")
    activations = extractor.extract_batch(prompts, show_progress=True)
    
    # Combine layer activations
    # For now, we'll save per-layer vectors
    valid_layers = {k: v for k, v in activations.items() if v is not None}
    
    if not valid_layers:
        console.print("[red]Error: No valid activations extracted![/red]")
        model.unload()
        return
    
    # Compute mean across layers for a single baseline vector
    stacked = torch.stack(list(valid_layers.values()))
    mean_vector = stacked.mean(dim=0)
    mean_vector = mean_vector / mean_vector.norm()  # Normalize
    
    # Save baseline
    output_path = config.vectors_dir / "assistant_baseline.pt"
    torch.save({
        "vector": mean_vector,
        "per_layer_vectors": valid_layers,
        "layers": list(valid_layers.keys()),
        "num_samples": len(prompts),
        "metadata": {
            "model": config.model_name,
            "questions_file": str(questions_path),
        }
    }, output_path)
    
    console.print(f"\n[bold green]✓ Saved assistant baseline to {output_path}[/bold green]")
    console.print(f"  Vector shape: {mean_vector.shape}")
    console.print(f"  Layers: {list(valid_layers.keys())}")
    
    # Cleanup
    model.unload()


def compute_assistant_axis():
    """
    Compute the full assistant axis (requires persona vectors).
    
    This should be called after persona extraction to compute:
    assistant_axis = assistant_baseline - mean(persona_vectors)
    """
    console = Console()
    config = get_config()
    
    # Load assistant baseline
    baseline_path = config.vectors_dir / "assistant_baseline.pt"
    if not baseline_path.exists():
        console.print("[red]Error: assistant_baseline.pt not found![/red]")
        console.print("Run extract_assistant_axis.py first.")
        return
    
    baseline_data = torch.load(baseline_path, weights_only=False)
    baseline_vector = baseline_data["vector"]
    
    # Load all persona vectors
    persona_dir = config.persona_vectors_dir
    persona_files = list(persona_dir.glob("*.pt"))
    
    if not persona_files:
        console.print("[yellow]No persona vectors found yet.[/yellow]")
        console.print("The assistant axis will be computed after persona extraction.")
        return
    
    console.print(f"\n[cyan]Loading {len(persona_files)} persona vectors...[/cyan]")
    
    persona_vectors = []
    for pf in persona_files:
        data = torch.load(pf, weights_only=False)
        # Use raw vector (before safety orthogonalization)
        if "raw_vector" in data:
            persona_vectors.append(data["raw_vector"])
        elif "vector" in data:
            persona_vectors.append(data["vector"])
    
    if not persona_vectors:
        console.print("[red]Error: No valid persona vectors loaded![/red]")
        return
    
    # Compute mean persona vector
    mean_persona = torch.stack(persona_vectors).mean(dim=0)
    
    # Compute assistant axis
    assistant_axis = baseline_vector - mean_persona
    assistant_axis = assistant_axis / assistant_axis.norm()
    
    # Save assistant axis
    output_path = config.vectors_dir / "assistant_axis.pt"
    torch.save({
        "vector": assistant_axis,
        "baseline_vector": baseline_vector,
        "mean_persona_vector": mean_persona,
        "num_personas": len(persona_vectors),
        "metadata": {
            "persona_files": [str(p) for p in persona_files],
        }
    }, output_path)
    
    console.print(f"\n[bold green]✓ Saved assistant axis to {output_path}[/bold green]")
    console.print(f"  Computed from {len(persona_vectors)} personas")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract assistant axis")
    parser.add_argument(
        "--compute-axis", 
        action="store_true",
        help="Compute full axis from baseline + personas (run after persona extraction)"
    )
    
    args = parser.parse_args()
    
    if args.compute_axis:
        compute_assistant_axis()
    else:
        main()
