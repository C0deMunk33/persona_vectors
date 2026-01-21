"""Model wrapper with activation hooks for capturing and steering."""

from contextlib import contextmanager
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import Config


class ModelWrapper:
    """
    Wrapper around a HuggingFace model that provides:
    - Activation capture via forward hooks
    - Activation steering via forward hooks
    - Safety capping via forward hooks
    """
    
    def __init__(self, config: Config):
        """
        Initialize the model wrapper.
        
        Args:
            config: Configuration object with model settings.
        """
        self.config = config
        self.device = config.device
        self.model = None
        self.tokenizer = None
        self._activation_cache: dict[int, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        
    def load(self) -> None:
        """Load the model and tokenizer into memory."""
        if self.model is not None:
            return
            
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        # Check for quantization config
        quantization = self.config.get("model.quantization")
        quantization_config = None
        
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=self.device,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Ensure pad token is set and use left padding for batched generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Required for batched generation
    
    def unload(self) -> None:
        """Unload model from memory."""
        self._clear_hooks()
        self._activation_cache.clear()
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
    
    def _clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def _clear_cache(self) -> None:
        """Clear the activation cache."""
        self._activation_cache.clear()
    
    def _get_transformer_layers(self) -> torch.nn.ModuleList:
        """Get the transformer layers from the model."""
        # Qwen2.5 architecture: model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        # Fallback for other architectures
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise AttributeError("Could not find transformer layers in model architecture")
    
    def _make_capture_hook(self, layer_idx: int) -> Callable:
        """Create a hook that captures activations."""
        def hook(module: torch.nn.Module, input: tuple, output: Any) -> None:
            # Output is (hidden_states, ...) or just hidden_states
            hidden = output[0] if isinstance(output, tuple) else output
            # Store a clone to avoid memory issues
            self._activation_cache[layer_idx] = hidden.detach().clone()
        return hook
    
    def _make_steering_hook(
        self, 
        steering_vector: torch.Tensor, 
        strength: float
    ) -> Callable:
        """Create a hook that injects a steering vector."""
        import logging
        log = logging.getLogger("persona_steering")
        vec_norm = steering_vector.norm().item()
        log.info(f"Creating steering hook: vector norm={vec_norm:.4f}, strength={strength}, effective_magnitude={strength*vec_norm:.4f}")
        
        # Track if hook was called
        hook_call_count = [0]
        
        def hook(module: torch.nn.Module, input: tuple, output: Any) -> Any:
            hook_call_count[0] += 1
            
            hidden = output[0] if isinstance(output, tuple) else output
            
            # Debug first call - show scale comparison
            if hook_call_count[0] == 1:
                hidden_norm = hidden.norm(dim=-1).mean().item()
                effective_steer = strength * vec_norm
                relative_change = effective_steer / hidden_norm * 100
                log.info(f"Hidden norm: {hidden_norm:.2f}, Adding: {effective_steer:.4f} ({relative_change:.2f}% of hidden)")
            
            # Add steering vector scaled by strength
            # Vector should broadcast: (hidden_dim,) + (batch, seq, hidden_dim)
            steering = strength * steering_vector.to(hidden.device, hidden.dtype)
            steered = hidden + steering
            
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered
        return hook
    
    def _make_capping_hook(
        self, 
        axis: torch.Tensor, 
        floor: float
    ) -> Callable:
        """Create a hook that enforces minimum projection onto an axis (floor capping)."""
        import logging
        log = logging.getLogger("persona_steering")
        hook_call_count = [0]
        
        def hook(module: torch.nn.Module, input: tuple, output: Any) -> Any:
            hook_call_count[0] += 1
            hidden = output[0] if isinstance(output, tuple) else output
            axis_device = axis.to(hidden.device, hidden.dtype)
            
            # Project hidden states onto axis
            proj = (hidden * axis_device).sum(dim=-1, keepdim=True)
            
            # How much we're below floor (only positive values need correction)
            deficit = torch.clamp(floor - proj, min=0)
            
            # Debug first call
            if hook_call_count[0] == 1:
                mean_proj = proj.mean().item()
                mean_deficit = deficit.mean().item()
                pct_below = (proj < floor).float().mean().item() * 100
                log.info(f"Floor hook: mean_proj={mean_proj:.3f}, floor={floor:.3f}, {pct_below:.1f}% below floor, mean_correction={mean_deficit:.4f}")
            
            # Add correction to bring below-floor projections up to floor
            correction = deficit * axis_device
            hidden = hidden + correction
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook
    
    
    @contextmanager
    def _capture_context(self, layers: list[int]):
        """Context manager for capturing activations at specified layers."""
        self._clear_cache()
        transformer_layers = self._get_transformer_layers()
        
        try:
            for layer_idx in layers:
                hook = transformer_layers[layer_idx].register_forward_hook(
                    self._make_capture_hook(layer_idx)
                )
                self._hooks.append(hook)
            yield
        finally:
            self._clear_hooks()
    
    @contextmanager
    def _steering_context(
        self, 
        steering_vector: torch.Tensor, 
        strength: float, 
        layers: list[int]
    ):
        """Context manager for steering at specified layers."""
        transformer_layers = self._get_transformer_layers()
        
        try:
            for layer_idx in layers:
                hook = transformer_layers[layer_idx].register_forward_hook(
                    self._make_steering_hook(steering_vector, strength)
                )
                self._hooks.append(hook)
            yield
        finally:
            self._clear_hooks()
    
    @contextmanager
    def _capping_context(
        self, 
        axis: torch.Tensor, 
        floor: float, 
        layers: list[int]
    ):
        """Context manager for floor capping at specified layers."""
        transformer_layers = self._get_transformer_layers()
        
        try:
            for layer_idx in layers:
                hook = transformer_layers[layer_idx].register_forward_hook(
                    self._make_capping_hook(axis, floor)
                )
                self._hooks.append(hook)
            yield
        finally:
            self._clear_hooks()
    
    
    def _format_prompt(
        self, 
        user_prompt: str, 
        system_prompt: str | None = None
    ) -> str:
        """Format a prompt using the model's chat template."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        # Try to disable thinking mode for Qwen3 models
        try:
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False,  # Disable Qwen3 thinking mode
            )
        except TypeError:
            # Fallback for models that don't support enable_thinking
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
    
    def _count_prompt_tokens(self, formatted_prompt: str) -> int:
        """Count the number of tokens in a formatted prompt."""
        return len(self.tokenizer.encode(formatted_prompt, add_special_tokens=False))
    
    @torch.inference_mode()
    def generate(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response without any hooks.
        
        Args:
            user_prompt: The user's message.
            system_prompt: Optional system prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.
            
        Returns:
            Generated response text.
        """
        if self.model is None:
            self.load()
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        
        formatted = self._format_prompt(user_prompt, system_prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response_tokens = outputs[0][prompt_length:]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    @torch.inference_mode()
    def generate_with_activations(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        layers: list[int] | None = None,
    ) -> tuple[str, dict[int, torch.Tensor]]:
        """
        Generate a response and capture activations at specified layers.
        
        Args:
            user_prompt: The user's message.
            system_prompt: Optional system prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            layers: Layers to capture activations from.
            
        Returns:
            Tuple of (response_text, {layer_idx: mean_activation_tensor})
        """
        if self.model is None:
            self.load()
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        layers = layers or self.config.target_layers
        
        formatted = self._format_prompt(user_prompt, system_prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        
        # We need to generate token by token to capture activations for each
        # But for efficiency, we'll do a single forward pass and capture
        # Actually, for extraction we want the activations during generation
        # The simplest approach: generate with hooks that accumulate
        
        accumulated_activations: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
        
        def make_accumulating_hook(layer_idx: int) -> Callable:
            def hook(module: torch.nn.Module, input: tuple, output: Any) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                # Only capture the last token's activation (during generation)
                # Shape: (batch, seq, hidden) -> take last token
                last_token_activation = hidden[:, -1, :].detach().clone()
                accumulated_activations[layer_idx].append(last_token_activation)
            return hook
        
        transformer_layers = self._get_transformer_layers()
        
        try:
            for layer_idx in layers:
                hook = transformer_layers[layer_idx].register_forward_hook(
                    make_accumulating_hook(layer_idx)
                )
                self._hooks.append(hook)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            self._clear_hooks()
        
        response_tokens = outputs[0][prompt_length:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Average activations across all generated tokens (excluding prompt)
        # The first few captured activations are from the prompt forward pass
        # We want only the response token activations
        mean_activations = {}
        for layer_idx, acts in accumulated_activations.items():
            if acts:
                # Stack all captured activations and take mean
                stacked = torch.cat(acts, dim=0)  # (num_tokens, hidden)
                mean_activations[layer_idx] = stacked.mean(dim=0).cpu()
            else:
                mean_activations[layer_idx] = None
        
        self._clear_cache()
        return response_text, mean_activations
    
    @torch.inference_mode()
    def generate_batch_with_activations(
        self,
        prompts: list[dict],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        layers: list[int] | None = None,
    ) -> list[dict[int, torch.Tensor]]:
        """
        Generate responses for a batch of prompts and capture activations.
        
        Args:
            prompts: List of {"system": str|None, "user": str} dicts.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            layers: Layers to capture activations from.
            
        Returns:
            List of {layer_idx: mean_activation_tensor} for each prompt.
        """
        if self.model is None:
            self.load()
        
        if not prompts:
            return []
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        layers = layers or self.config.target_layers
        batch_size = len(prompts)
        
        # Format all prompts
        formatted = [
            self._format_prompt(p["user"], p.get("system"))
            for p in prompts
        ]
        
        # Tokenize with padding
        inputs = self.tokenizer(
            formatted, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.device)
        
        # Track prompt lengths for each item (before padding)
        prompt_lengths = [
            len(self.tokenizer.encode(f, add_special_tokens=False))
            for f in formatted
        ]
        
        # Accumulated activations: layer -> batch_idx -> list of activations
        accumulated: dict[int, dict[int, list[torch.Tensor]]] = {
            l: {i: [] for i in range(batch_size)} for l in layers
        }
        
        def make_batch_accumulating_hook(layer_idx: int):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                # hidden shape: (batch, seq, hidden)
                # Take last token for each batch item
                for batch_idx in range(hidden.shape[0]):
                    last_token = hidden[batch_idx, -1, :].detach().clone()
                    accumulated[layer_idx][batch_idx].append(last_token)
            return hook
        
        transformer_layers = self._get_transformer_layers()
        
        try:
            for layer_idx in layers:
                hook = transformer_layers[layer_idx].register_forward_hook(
                    make_batch_accumulating_hook(layer_idx)
                )
                self._hooks.append(hook)
            
            # Generate for the batch
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            self._clear_hooks()
        
        # Compute mean activations per prompt
        results = []
        for batch_idx in range(batch_size):
            mean_activations = {}
            for layer_idx in layers:
                acts = accumulated[layer_idx][batch_idx]
                if acts:
                    stacked = torch.stack(acts)  # (num_tokens, hidden)
                    mean_activations[layer_idx] = stacked.mean(dim=0).cpu()
                else:
                    mean_activations[layer_idx] = None
            results.append(mean_activations)
        
        self._clear_cache()
        return results
    
    @torch.inference_mode()
    def generate_with_steering(
        self,
        user_prompt: str,
        steering_vector: torch.Tensor,
        strength: float,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        layers: list[int] | None = None,
    ) -> str:
        """
        Generate a response with a steering vector applied.
        
        Args:
            user_prompt: The user's message.
            steering_vector: Vector to add to activations.
            strength: Scaling factor for the steering vector.
            system_prompt: Optional system prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            layers: Layers to apply steering to.
            
        Returns:
            Generated response text.
        """
        if self.model is None:
            self.load()
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        layers = layers or self.config.target_layers
        
        formatted = self._format_prompt(user_prompt, system_prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        
        with self._steering_context(steering_vector, strength, layers):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response_tokens = outputs[0][prompt_length:]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    @torch.inference_mode()
    def generate_with_per_layer_steering(
        self,
        user_prompt: str,
        layer_vectors: dict[int, torch.Tensor],
        strength: float,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Generate with different steering vectors per layer.
        
        This is more stable than using the same vector for all layers,
        as different layers encode different aspects of language.
        
        Args:
            user_prompt: The user's message.
            layer_vectors: Dictionary mapping layer index to steering vector.
            strength: Scaling factor applied to all vectors.
            system_prompt: Optional system prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated response text.
        """
        import logging
        log = logging.getLogger("persona_steering")
        
        if self.model is None:
            self.load()
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        
        formatted = self._format_prompt(user_prompt, system_prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        
        transformer_layers = self._get_transformer_layers()
        
        # Create per-layer hooks
        try:
            for layer_idx, vec in layer_vectors.items():
                vec_norm = vec.norm().item()
                effective = strength * vec_norm
                log.info(f"Layer {layer_idx}: vec_norm={vec_norm:.2f}, effective={effective:.2f}")
                
                hook = transformer_layers[layer_idx].register_forward_hook(
                    self._make_steering_hook(vec, strength)
                )
                self._hooks.append(hook)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            self._clear_hooks()
        
        response_tokens = outputs[0][prompt_length:]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    @torch.inference_mode()
    def generate_with_capping(
        self,
        user_prompt: str,
        axis: torch.Tensor,
        floor: float,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        layers: list[int] | None = None,
    ) -> str:
        """
        Generate a response with safety capping applied.
        
        Args:
            user_prompt: The user's message.
            axis: The axis to cap on (e.g., safety axis).
            floor: Minimum projection value to enforce.
            system_prompt: Optional system prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            layers: Layers to apply capping to.
            
        Returns:
            Generated response text.
        """
        if self.model is None:
            self.load()
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        layers = layers or self.config.capping_layers or self.config.target_layers
        
        formatted = self._format_prompt(user_prompt, system_prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        
        with self._capping_context(axis, floor, layers):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response_tokens = outputs[0][prompt_length:]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    @torch.inference_mode()
    def generate_with_steering_and_capping(
        self,
        user_prompt: str,
        steering_vector: torch.Tensor,
        strength: float,
        axis: torch.Tensor,
        floor: float,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        steering_layers: list[int] | None = None,
        capping_layers: list[int] | None = None,
    ) -> str:
        """
        Generate with both steering and safety capping.
        
        Args:
            user_prompt: The user's message.
            steering_vector: Vector to add to activations.
            strength: Scaling factor for the steering vector.
            axis: The axis to cap on (e.g., safety axis).
            floor: Minimum projection value to enforce.
            system_prompt: Optional system prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            steering_layers: Layers to apply steering to.
            capping_layers: Layers to apply capping to.
            
        Returns:
            Generated response text.
        """
        if self.model is None:
            self.load()
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        steering_layers = steering_layers or self.config.target_layers
        capping_layers = capping_layers or self.config.capping_layers or self.config.target_layers
        
        formatted = self._format_prompt(user_prompt, system_prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        
        transformer_layers = self._get_transformer_layers()
        
        try:
            # Register steering hooks
            for layer_idx in steering_layers:
                hook = transformer_layers[layer_idx].register_forward_hook(
                    self._make_steering_hook(steering_vector, strength)
                )
                self._hooks.append(hook)
            
            # Register capping hooks
            for layer_idx in capping_layers:
                hook = transformer_layers[layer_idx].register_forward_hook(
                    self._make_capping_hook(axis, floor)
                )
                self._hooks.append(hook)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            self._clear_hooks()
        
        response_tokens = outputs[0][prompt_length:]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    
    @property
    def hidden_size(self) -> int:
        """Get the model's hidden dimension size."""
        if self.model is None:
            self.load()
        return self.model.config.hidden_size
    
    @property
    def num_layers(self) -> int:
        """Get the number of transformer layers."""
        if self.model is None:
            self.load()
        return self.model.config.num_hidden_layers
