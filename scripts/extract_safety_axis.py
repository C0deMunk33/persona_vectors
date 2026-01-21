#!/usr/bin/env python3
"""Extract the safety axis from contrastive safe/unsafe prompts."""

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
from src.vector_math import compute_safe_floor


def main():
    """Extract safety axis and compute floor threshold."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold red]Safety Axis Extractor[/bold red]\n"
        "Extracts safety direction from contrastive safe/unsafe prompts",
        border_style="red"
    ))
    
    # Load config
    config = get_config()
    config.ensure_directories()
    
    # Check for safety datasets
    safe_path = config.safety_dir / "safe_prompts.jsonl"
    unsafe_path = config.safety_dir / "unsafe_prompts.jsonl"
    
    if not safe_path.exists() or not unsafe_path.exists():
        console.print("[red]Error: Safety datasets not found![/red]")
        console.print("Run generate_safety_dataset.py first.")
        return
    
    # Initialize model
    console.print("\n[yellow]Loading model...[/yellow]")
    model = ModelWrapper(config)
    model.load()
    
    # Create extractor and generator
    extractor = ActivationExtractor(model, config)
    generator = DatasetGenerator(model, config)
    
    # Load datasets
    console.print("\n[cyan]Loading safety datasets...[/cyan]")
    safe_prompts_raw = generator.load_jsonl(safe_path)
    unsafe_prompts_raw = generator.load_jsonl(unsafe_path)
    
    console.print(f"  Loaded {len(safe_prompts_raw)} safe prompts")
    console.print(f"  Loaded {len(unsafe_prompts_raw)} unsafe prompts")
    
    # Convert to extraction format
    safe_prompts = [{"system": None, "user": p["prompt"]} for p in safe_prompts_raw]
    unsafe_prompts = [{"system": None, "user": p["prompt"]} for p in unsafe_prompts_raw]
    
    # Extract contrastive axis
    # Note: For safety, positive = safe behavior, negative = unsafe/refusal
    console.print("\n[green]Extracting contrastive safety axis...[/green]")
    contrastive = extractor.extract_contrastive(
        positive_prompts=safe_prompts,
        negative_prompts=unsafe_prompts,
        show_progress=True
    )
    
    # Combine layer vectors
    valid_layers = {k: v for k, v in contrastive.items() if v is not None}
    
    if not valid_layers:
        console.print("[red]Error: No valid contrastive activations extracted![/red]")
        model.unload()
        return
    
    # Compute mean across layers
    stacked = torch.stack(list(valid_layers.values()))
    safety_axis = stacked.mean(dim=0)
    safety_axis = safety_axis / safety_axis.norm()
    
    # Compute safety floor
    console.print("\n[yellow]Computing safety floor threshold...[/yellow]")
    
    # Collect individual safe activations for floor computation
    safe_activations = extractor.collect_safe_activations(
        safe_prompts[:100],  # Use subset for efficiency
        show_progress=True
    )
    
    # Get activations for a representative layer
    representative_layer = config.target_layers[len(config.target_layers) // 2]
    layer_activations = safe_activations.get(representative_layer, [])
    
    if layer_activations:
        floor_value = compute_safe_floor(
            layer_activations,
            safety_axis,
            percentile=config.safety_floor_percentile
        )
    else:
        floor_value = 0.0
        console.print("[yellow]Warning: Could not compute floor, using 0.0[/yellow]")
    
    # Save safety axis
    output_path = config.vectors_dir / "safety_axis.pt"
    torch.save({
        "vector": safety_axis,
        "per_layer_vectors": valid_layers,
        "floor": floor_value,
        "floor_percentile": config.safety_floor_percentile,
        "layers": list(valid_layers.keys()),
        "num_safe_samples": len(safe_prompts),
        "num_unsafe_samples": len(unsafe_prompts),
        "metadata": {
            "model": config.model_name,
            "safe_file": str(safe_path),
            "unsafe_file": str(unsafe_path),
        }
    }, output_path)
    
    console.print(f"\n[bold green]✓ Saved safety axis to {output_path}[/bold green]")
    console.print(f"  Vector shape: {safety_axis.shape}")
    console.print(f"  Safety floor: {floor_value:.4f}")
    console.print(f"  Layers: {list(valid_layers.keys())}")
    
    # Cleanup
    model.unload()


if __name__ == "__main__":
    main()
