"""Steering engine for runtime generation with persona steering and safety capping."""

import torch

from .config import Config
from .model_wrapper import ModelWrapper
from .persona_manager import PersonaManager


class SteeringEngine:
    """
    Combines model, personas, and safety for runtime generation.
    
    Provides a high-level interface for generating text with:
    - Persona steering (adding persona vector to activations)
    - Safety capping (clamping activations above safety floor)
    - Composite/blended personas
    """
    
    def __init__(
        self, 
        model_wrapper: ModelWrapper, 
        persona_manager: PersonaManager,
        config: Config
    ):
        """
        Initialize the steering engine.
        
        Args:
            model_wrapper: The model wrapper for generation.
            persona_manager: Manager for persona vectors.
            config: Configuration object.
        """
        self.model = model_wrapper
        self.personas = persona_manager
        self.config = config
        
        self._default_strength = config.default_strength
        self._safety_enabled = False  # Disabled by default - use persona mixing instead
    
    @property
    def default_strength(self) -> float:
        """Get the default steering strength."""
        return self._default_strength
    
    @default_strength.setter
    def default_strength(self, value: float) -> None:
        """Set the default steering strength."""
        self._default_strength = max(0.0, min(2.0, value))  # Clamp to reasonable range
    
    @property
    def safety_enabled(self) -> bool:
        """Check if safety capping is enabled."""
        return self._safety_enabled
    
    @safety_enabled.setter
    def safety_enabled(self, value: bool) -> None:
        """Enable or disable safety capping."""
        self._safety_enabled = value
    
    def generate(
        self,
        user_prompt: str,
        persona: str | None = None,
        persona_strength: float | None = None,
        enable_safety_capping: bool | None = None,
        system_override: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Generate a response with optional persona steering and safety capping.
        
        Args:
            user_prompt: The user's message.
            persona: Name of persona to use (None for no steering).
            persona_strength: Steering strength (None uses default).
            enable_safety_capping: Override for safety capping (None uses instance setting).
            system_override: Optional system prompt override.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated response text.
        """
        # Resolve parameters
        strength = persona_strength if persona_strength is not None else self._default_strength
        use_safety = enable_safety_capping if enable_safety_capping is not None else self._safety_enabled
        
        # Get persona vector if specified
        persona_vector = None
        if persona:
            persona_vector = self.personas.get_vector(persona, safe=True)
            if persona_vector is None:
                # Persona not found - generate without steering
                pass
        
        # Get safety axis and floor
        safety_axis = self.personas.safety_axis
        safety_floor = self.personas.safety_floor
        
        # Determine which generation method to use
        if persona_vector is not None and use_safety and safety_axis is not None and safety_floor is not None:
            # Full steering with safety capping
            return self.model.generate_with_steering_and_capping(
                user_prompt=user_prompt,
                steering_vector=persona_vector,
                strength=strength,
                axis=safety_axis,
                floor=safety_floor,
                system_prompt=system_override,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                steering_layers=self.config.target_layers,
                capping_layers=self.config.capping_layers,
            )
        elif persona_vector is not None:
            # Steering only
            return self.model.generate_with_steering(
                user_prompt=user_prompt,
                steering_vector=persona_vector,
                strength=strength,
                system_prompt=system_override,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                layers=self.config.target_layers,
            )
        elif use_safety and safety_axis is not None and safety_floor is not None:
            # Safety capping only
            return self.model.generate_with_capping(
                user_prompt=user_prompt,
                axis=safety_axis,
                floor=safety_floor,
                system_prompt=system_override,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                layers=self.config.capping_layers,
            )
        else:
            # No steering or capping
            return self.model.generate(
                user_prompt=user_prompt,
                system_prompt=system_override,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
    
    def generate_with_composite(
        self,
        user_prompt: str,
        persona_blend: dict[str, float],
        enable_safety_capping: bool | None = None,
        system_override: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Generate with a blended persona using per-layer additive steering.
        
        Args:
            user_prompt: The user's message.
            persona_blend: Dictionary mapping persona names to weights.
                          Example: {"sage": 0.5, "trickster": 0.3}
            enable_safety_capping: Override for safety capping.
            system_override: Optional system prompt override.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated response text.
        """
        import logging
        log = logging.getLogger("persona_steering")
        log.info(f"generate_with_composite called with blend: {persona_blend}")
        
        total_weight = sum(persona_blend.values())
        
        # Try to get per-layer vectors for more stable steering
        # Each layer gets its own vector with appropriate magnitude
        blended_per_layer: dict[int, torch.Tensor] = {}
        
        # Get steering scale from config (default 1.0, but we recommend 0.1 for stability)
        steering_scale = self.config.steering_scale
        
        for name, weight in persona_blend.items():
            per_layer = self.personas.get_per_layer_vectors(name)
            if per_layer:
                for layer_idx, vec in per_layer.items():
                    if vec is not None:
                        # Apply both weight and global scale
                        scaled_vec = weight * steering_scale * vec
                        if layer_idx in blended_per_layer:
                            blended_per_layer[layer_idx] = blended_per_layer[layer_idx] + scaled_vec
                        else:
                            blended_per_layer[layer_idx] = scaled_vec
        
        if blended_per_layer:
            # Log per-layer magnitudes
            norms = {l: v.norm().item() for l, v in blended_per_layer.items()}
            log.info(f"Per-layer steering: weight={total_weight:.2f}, scale={steering_scale}, layer_norms={norms}")
            
            return self.model.generate_with_per_layer_steering(
                user_prompt=user_prompt,
                layer_vectors=blended_per_layer,
                strength=1.0,  # Weight already applied in blend
                system_prompt=system_override,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        
        # Fallback to single-vector steering if no per-layer vectors
        log.warning("No per-layer vectors available, falling back to composite")
        composite_vector = self.personas.get_composite(persona_blend)
        
        if composite_vector is None:
            log.warning("composite_vector is None! Falling back to normal generation")
            return self.generate(
                user_prompt=user_prompt,
                enable_safety_capping=enable_safety_capping,
                system_override=system_override,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        
        # Apply global steering scale
        steering_scale = self.config.steering_scale
        scaled_vector = steering_scale * composite_vector
        vec_norm = scaled_vector.norm().item()
        log.info(f"Single-vector steering: weight={total_weight:.2f}, scale={steering_scale}, magnitude={vec_norm:.2f}")
        
        return self.model.generate_with_steering(
            user_prompt=user_prompt,
            steering_vector=scaled_vector,
            strength=1.0,
            system_prompt=system_override,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            layers=self.config.target_layers,
        )
    
    def set_safety_floor(self, percentile: int) -> None:
        """
        Adjust safety floor threshold.
        
        Note: This requires re-computing the floor from stored activations,
        which is not currently supported at runtime. This method is a placeholder.
        
        Args:
            percentile: New percentile for floor computation.
        """
        # TODO: Implement runtime floor adjustment
        # This would require storing the safe activations or recomputing
        pass
    
    def set_default_strength(self, strength: float) -> None:
        """
        Adjust default persona strength.
        
        Args:
            strength: New default strength (0.0 to 2.0).
        """
        self.default_strength = strength
    
    def get_status(self) -> dict:
        """
        Get current engine status.
        
        Returns:
            Dictionary with status information.
        """
        return {
            "model_loaded": self.model.model is not None,
            "default_strength": self._default_strength,
            "safety_enabled": self._safety_enabled,
            "safety_axis_loaded": self.personas.safety_axis is not None,
            "safety_floor": self.personas.safety_floor,
            "loaded_personas": self.personas.list_loaded(),
            "available_personas": self.personas.list_available(),
        }
    
    def ensure_loaded(self) -> None:
        """Ensure the model is loaded."""
        if self.model.model is None:
            self.model.load()
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        self.model.unload()
        self.personas.unload_all()
