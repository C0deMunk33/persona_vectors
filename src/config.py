"""Configuration loading and validation for the persona steering system."""

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Configuration manager for the persona steering system."""
    
    def __init__(self, config_path: str | Path | None = None):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config.yaml. If None, looks in default locations.
        """
        if config_path is None:
            # Look for config in standard locations
            candidates = [
                Path.cwd() / "config.yaml",
                Path(__file__).parent.parent / "config.yaml",
            ]
            for candidate in candidates:
                if candidate.exists():
                    config_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    "config.yaml not found. Please create one or specify path."
                )
        
        self.config_path = Path(config_path)
        self._data = self._load_config()
        self._validate()
    
    def _load_config(self) -> dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _validate(self) -> None:
        """Validate configuration has required sections."""
        required_sections = ["model", "extraction", "steering", "paths"]
        for section in required_sections:
            if section not in self._data:
                raise ValueError(f"Missing required config section: {section}")
    
    @property
    def model_name(self) -> str:
        """Get model name/path."""
        return self._data["model"]["name"]
    
    @property
    def device(self) -> str:
        """Get target device."""
        return self._data["model"].get("device", "cuda")
    
    @property
    def dtype(self) -> str:
        """Get model dtype."""
        return self._data["model"].get("dtype", "bfloat16")
    
    @property
    def target_layers(self) -> list[int]:
        """Get layers to extract activations from."""
        return self._data["extraction"]["target_layers"]
    
    @property
    def num_samples_per_prompt(self) -> int:
        """Get number of samples per prompt for extraction."""
        return self._data["extraction"].get("num_samples_per_prompt", 50)
    
    @property
    def max_new_tokens(self) -> int:
        """Get max tokens for generation."""
        return self._data["extraction"].get("max_new_tokens", 256)
    
    @property
    def temperature(self) -> float:
        """Get generation temperature."""
        return self._data["extraction"].get("temperature", 0.7)
    
    @property
    def batch_size(self) -> int:
        """Get batch size for extraction."""
        return self._data["extraction"].get("batch_size", 1)
    
    @property
    def num_questions(self) -> int:
        """Get number of extraction questions to use."""
        return self._data["extraction"].get("num_questions", 100)
    
    @property
    def default_strength(self) -> float:
        """Get default steering strength."""
        return self._data["steering"].get("default_strength", 0.3)
    
    @property
    def steering_scale(self) -> float:
        """Get steering scale factor (multiplied with raw vector magnitudes)."""
        return self._data["steering"].get("steering_scale", 1.0)
    
    @property
    def safety_floor_percentile(self) -> int:
        """Get safety floor percentile."""
        return self._data["steering"].get("safety_floor_percentile", 25)
    
    @property
    def capping_layers(self) -> list[int] | None:
        """Get layers for safety capping (None if disabled)."""
        return self._data["steering"].get("capping_layers")
    
    @property
    def datasets_dir(self) -> Path:
        """Get datasets directory path."""
        path = Path(self._data["paths"]["datasets"])
        if not path.is_absolute():
            path = self.config_path.parent / path
        return path
    
    @property
    def vectors_dir(self) -> Path:
        """Get vectors directory path."""
        path = Path(self._data["paths"]["vectors"])
        if not path.is_absolute():
            path = self.config_path.parent / path
        return path
    
    @property
    def safety_dir(self) -> Path:
        """Get safety datasets directory."""
        return self.datasets_dir / "safety"
    
    @property
    def personas_dir(self) -> Path:
        """Get personas datasets directory."""
        return self.datasets_dir / "personas"
    
    @property
    def archetypes_dir(self) -> Path:
        """Get archetypes directory."""
        return self.personas_dir / "archetypes"
    
    @property
    def persona_vectors_dir(self) -> Path:
        """Get persona vectors directory."""
        return self.vectors_dir / "personas"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a nested config value using dot notation."""
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a config value using dot notation and save to file."""
        keys = key.split(".")
        data = self._data
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value
        self._save()
    
    def _save(self) -> None:
        """Save configuration back to YAML file."""
        with open(self.config_path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False)
    
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.datasets_dir,
            self.safety_dir,
            self.personas_dir,
            self.archetypes_dir,
            self.vectors_dir,
            self.persona_vectors_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def get_config(config_path: str | Path | None = None) -> Config:
    """Get or create singleton config instance."""
    return Config(config_path)
