"""Persona lifecycle management."""

from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from .config import Config
from .vector_math import blend_vectors, orthogonalize_persona


class PersonaManager:
    """
    Manages persona vectors: loading, unloading, listing, and blending.
    
    Handles the lifecycle of persona vectors including:
    - Loading from disk into memory
    - Unloading to free memory
    - Listing available and loaded personas
    - Creating blended personas
    - Creating new personas on-the-fly
    """
    
    def __init__(self, config: Config):
        """
        Initialize the persona manager.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self._loaded: dict[str, dict[str, Any]] = {}
        self._safety_axis: torch.Tensor | None = None
        self._safety_floor: float | None = None
        
        # Load safety axis if available
        self._load_safety_axis()
    
    def _load_safety_axis(self) -> None:
        """Load the safety axis for orthogonalization."""
        safety_path = self.config.vectors_dir / "safety_axis.pt"
        if safety_path.exists():
            data = torch.load(safety_path, weights_only=False)
            self._safety_axis = data["vector"]
            self._safety_floor = data.get("floor", 0.0)
    
    @property
    def safety_axis(self) -> torch.Tensor | None:
        """Get the safety axis vector."""
        return self._safety_axis
    
    @property
    def safety_floor(self) -> float | None:
        """Get the safety floor threshold."""
        return self._safety_floor
    
    def list_available(self) -> list[str]:
        """
        List all available persona .pt files.
        
        Returns:
            List of persona names (without .pt extension).
        """
        persona_dir = self.config.persona_vectors_dir
        if not persona_dir.exists():
            return []
        
        return [f.stem for f in persona_dir.glob("*.pt")]
    
    def list_loaded(self) -> list[str]:
        """
        List currently loaded persona names.
        
        Returns:
            List of loaded persona names.
        """
        return list(self._loaded.keys())
    
    def is_loaded(self, name: str) -> bool:
        """Check if a persona is loaded."""
        return name in self._loaded
    
    def load(self, name: str) -> bool:
        """
        Load a persona from disk into memory.
        
        Args:
            name: Name of the persona to load.
            
        Returns:
            True if successful, False otherwise.
        """
        if name in self._loaded:
            return True  # Already loaded
        
        persona_path = self.config.persona_vectors_dir / f"{name}.pt"
        if not persona_path.exists():
            return False
        
        try:
            data = torch.load(persona_path, weights_only=False)
            self._loaded[name] = data
            return True
        except Exception:
            return False
    
    def unload(self, name: str) -> bool:
        """
        Remove persona from active memory.
        
        Args:
            name: Name of the persona to unload.
            
        Returns:
            True if successful, False if not loaded.
        """
        if name not in self._loaded:
            return False
        
        del self._loaded[name]
        return True
    
    def unload_all(self) -> None:
        """Unload all personas from memory."""
        self._loaded.clear()
    
    def delete(self, name: str) -> bool:
        """
        Delete persona from disk and memory.
        
        Args:
            name: Name of the persona to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        # Unload if loaded
        self.unload(name)
        
        # Delete from disk
        persona_path = self.config.persona_vectors_dir / f"{name}.pt"
        if persona_path.exists():
            persona_path.unlink()
            return True
        return False
    
    def get_vector(self, name: str, safe: bool = True) -> torch.Tensor | None:
        """
        Get the vector for a loaded persona.
        
        Args:
            name: Name of the persona.
            safe: If True, return orthogonalized vector; otherwise raw.
            
        Returns:
            Persona vector or None if not loaded.
        """
        if name not in self._loaded:
            # Try to load it
            if not self.load(name):
                return None
        
        data = self._loaded[name]
        
        if safe and "safe_vector" in data:
            return data["safe_vector"]
        elif "raw_vector" in data:
            return data["raw_vector"]
        elif "vector" in data:
            return data["vector"]
        
        return None
    
    def get_per_layer_vectors(self, name: str) -> dict[int, torch.Tensor] | None:
        """
        Get the per-layer vectors for a loaded persona.
        
        These have layer-specific magnitudes which can give more stable steering.
        
        Args:
            name: Name of the persona.
            
        Returns:
            Dictionary mapping layer index to vector, or None if not available.
        """
        if name not in self._loaded:
            if not self.load(name):
                return None
        
        data = self._loaded[name]
        return data.get("per_layer_vectors")
    
    def get_info(self, name: str) -> dict[str, Any] | None:
        """
        Get metadata for a persona.
        
        Args:
            name: Name of the persona.
            
        Returns:
            Dictionary with persona info or None if not found.
        """
        if name not in self._loaded:
            if not self.load(name):
                return None
        
        data = self._loaded[name]
        
        return {
            "name": name,
            "source": data.get("source", "unknown"),
            "description": data.get("description", ""),
            "traits": data.get("traits", []),
            "layers": data.get("layers", []),
            "has_safe_vector": "safe_vector" in data,
            "has_raw_vector": "raw_vector" in data,
        }
    
    def get_composite(self, personas: dict[str, float]) -> torch.Tensor | None:
        """
        Create a blended persona vector.
        
        Args:
            personas: Dictionary mapping persona names to weights.
                     Example: {"sage": 0.5, "trickster": 0.3}
                     
        Returns:
            Blended and normalized vector, or None if any persona unavailable.
        """
        if not personas:
            return None
        
        vectors = []
        weights = []
        
        for name, weight in personas.items():
            vec = self.get_vector(name, safe=True)
            if vec is None:
                return None
            vectors.append(vec)
            weights.append(weight)
        
        return blend_vectors(vectors, weights)
    
    def create_from_data(
        self,
        name: str,
        raw_vector: torch.Tensor,
        description: str = "",
        traits: list[str] | None = None,
        layers: list[int] | None = None,
    ) -> bool:
        """
        Create a new persona from a raw vector.
        
        Automatically orthogonalizes against safety axis.
        
        Args:
            name: Name for the new persona.
            raw_vector: The raw activation vector.
            description: Optional description.
            traits: Optional list of traits.
            layers: Optional list of layers the vector was extracted from.
            
        Returns:
            True if successful.
        """
        # Normalize raw vector
        raw_vector = raw_vector / raw_vector.norm()
        
        # Orthogonalize if safety axis available
        if self._safety_axis is not None:
            safe_vector = orthogonalize_persona(raw_vector, self._safety_axis)
        else:
            safe_vector = raw_vector.clone()
        
        # Create data structure
        data = {
            "raw_vector": raw_vector,
            "safe_vector": safe_vector,
            "source": "created",
            "name": name,
            "description": description,
            "traits": traits or [],
            "layers": layers or [],
            "metadata": {}
        }
        
        # Save to disk
        output_path = self.config.persona_vectors_dir / f"{name}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_path)
        
        # Load into memory
        self._loaded[name] = data
        
        return True
    
    def print_available(self, console: Console | None = None) -> None:
        """Print a table of available personas."""
        if console is None:
            console = Console()
        
        available = self.list_available()
        loaded = set(self.list_loaded())
        
        if not available:
            console.print("[dim]No personas available.[/dim]")
            return
        
        table = Table(title="Available Personas")
        table.add_column("Name", style="cyan")
        table.add_column("Loaded", style="green")
        table.add_column("Source", style="yellow")
        table.add_column("Description", style="dim")
        
        for name in sorted(available):
            is_loaded = "✓" if name in loaded else ""
            
            # Load temporarily to get info
            info = self.get_info(name)
            source = info.get("source", "unknown") if info else "unknown"
            desc = info.get("description", "")[:50] if info else ""
            if desc and len(info.get("description", "")) > 50:
                desc += "..."
            
            # Unload if we just loaded it for info
            if name not in loaded:
                self.unload(name)
            
            table.add_row(name, is_loaded, source, desc)
        
        console.print(table)
    
    def print_loaded(self, console: Console | None = None) -> None:
        """Print a table of loaded personas."""
        if console is None:
            console = Console()
        
        loaded = self.list_loaded()
        
        if not loaded:
            console.print("[dim]No personas loaded.[/dim]")
            return
        
        table = Table(title="Loaded Personas")
        table.add_column("Name", style="cyan")
        table.add_column("Source", style="yellow")
        table.add_column("Traits", style="magenta")
        
        for name in sorted(loaded):
            info = self.get_info(name)
            source = info.get("source", "unknown") if info else "unknown"
            traits = ", ".join(info.get("traits", [])[:3]) if info else ""
            
            table.add_row(name, source, traits)
        
        console.print(table)
