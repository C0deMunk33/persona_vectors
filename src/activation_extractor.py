"""Activation extraction for persona and axis vectors."""

import torch
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from .config import Config
from .model_wrapper import ModelWrapper
from .vector_math import normalize, compute_axis_from_contrastive


class ActivationExtractor:
    """
    Extracts activation vectors from model generations.
    
    Used to create persona vectors, safety axes, and assistant baselines
    by capturing and averaging activations during generation.
    """
    
    def __init__(self, model_wrapper: ModelWrapper, config: Config):
        """
        Initialize the extractor.
        
        Args:
            model_wrapper: The model wrapper for generation.
            config: Configuration object.
        """
        self.model = model_wrapper
        self.config = config
        self.target_layers = config.target_layers
        self.batch_size = config.batch_size
    
    def extract_single(
        self, 
        user_prompt: str,
        system_prompt: str | None = None,
        layers: list[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """
        Generate a response and return mean activation per layer.
        
        Args:
            user_prompt: The user prompt to generate from.
            system_prompt: Optional system prompt.
            layers: Layers to extract from (defaults to config).
            
        Returns:
            Dictionary mapping layer index to mean activation tensor.
        """
        layers = layers or self.target_layers
        
        _, activations = self.model.generate_with_activations(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            layers=layers,
        )
        
        return activations
    
    def extract_batch(
        self, 
        prompts: list[dict],
        layers: list[int] | None = None,
        show_progress: bool = True,
        batch_size: int | None = None,
    ) -> dict[int, torch.Tensor]:
        """
        Extract activations for multiple prompts and return averaged.
        
        Uses batched generation for efficiency when batch_size > 1.
        
        Args:
            prompts: List of {"system": str|None, "user": str} dicts.
            layers: Layers to extract from.
            show_progress: Whether to show progress bar.
            batch_size: Override batch size (defaults to config).
            
        Returns:
            Dictionary mapping layer index to averaged activation tensor.
        """
        layers = layers or self.target_layers
        batch_size = batch_size or self.batch_size
        
        # Accumulate activations per layer
        accumulated: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
        
        # Calculate number of batches
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                if batch_size > 1:
                    task = progress.add_task(
                        f"Extracting (batch={batch_size})...", 
                        total=num_batches
                    )
                    
                    for batch_idx in range(num_batches):
                        start = batch_idx * batch_size
                        end = min(start + batch_size, len(prompts))
                        batch_prompts = prompts[start:end]
                        
                        # Use batched extraction
                        batch_results = self.model.generate_batch_with_activations(
                            prompts=batch_prompts,
                            layers=layers,
                        )
                        
                        # Accumulate results
                        for result in batch_results:
                            for layer_idx, act in result.items():
                                if act is not None:
                                    accumulated[layer_idx].append(act)
                        
                        progress.advance(task)
                else:
                    # Fall back to sequential for batch_size=1
                    task = progress.add_task("Extracting...", total=len(prompts))
                    
                    for prompt in prompts:
                        activations = self.extract_single(
                            user_prompt=prompt["user"],
                            system_prompt=prompt.get("system"),
                            layers=layers,
                        )
                        
                        for layer_idx, act in activations.items():
                            if act is not None:
                                accumulated[layer_idx].append(act)
                        
                        progress.advance(task)
        else:
            if batch_size > 1:
                for batch_idx in range(num_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, len(prompts))
                    batch_prompts = prompts[start:end]
                    
                    batch_results = self.model.generate_batch_with_activations(
                        prompts=batch_prompts,
                        layers=layers,
                    )
                    
                    for result in batch_results:
                        for layer_idx, act in result.items():
                            if act is not None:
                                accumulated[layer_idx].append(act)
            else:
                for prompt in prompts:
                    activations = self.extract_single(
                        user_prompt=prompt["user"],
                        system_prompt=prompt.get("system"),
                        layers=layers,
                    )
                    
                    for layer_idx, act in activations.items():
                        if act is not None:
                            accumulated[layer_idx].append(act)
        
        # Average across all prompts - DON'T normalize here
        # (normalization should happen only on final contrastive direction)
        averaged: dict[int, torch.Tensor] = {}
        for layer_idx, acts in accumulated.items():
            if acts:
                stacked = torch.stack(acts)
                averaged[layer_idx] = stacked.mean(dim=0)  # No normalization!
            else:
                averaged[layer_idx] = None
        
        return averaged
    
    def _extract_batch_raw(
        self,
        prompts: list[dict],
        layers: list[int],
        label: str,
        show_progress: bool = True,
        batch_size: int | None = None,
    ) -> dict[int, list[torch.Tensor]]:
        """
        Extract activations for prompts without averaging (returns raw list).
        
        Args:
            prompts: List of {"system": str|None, "user": str} dicts.
            layers: Layers to extract from.
            label: Label for progress bar.
            show_progress: Whether to show progress bar.
            batch_size: Override batch size.
            
        Returns:
            Dictionary mapping layer index to list of activation tensors.
        """
        batch_size = batch_size or self.batch_size
        accumulated: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                if batch_size > 1:
                    task = progress.add_task(f"{label} (batch={batch_size})...", total=num_batches)
                    
                    for batch_idx in range(num_batches):
                        start = batch_idx * batch_size
                        end = min(start + batch_size, len(prompts))
                        batch_prompts = prompts[start:end]
                        
                        batch_results = self.model.generate_batch_with_activations(
                            prompts=batch_prompts,
                            layers=layers,
                        )
                        
                        for result in batch_results:
                            for layer_idx, act in result.items():
                                if act is not None:
                                    accumulated[layer_idx].append(act)
                        
                        progress.advance(task)
                else:
                    task = progress.add_task(f"{label}...", total=len(prompts))
                    
                    for prompt in prompts:
                        activations = self.extract_single(
                            user_prompt=prompt["user"],
                            system_prompt=prompt.get("system"),
                            layers=layers,
                        )
                        
                        for layer_idx, act in activations.items():
                            if act is not None:
                                accumulated[layer_idx].append(act)
                        
                        progress.advance(task)
        else:
            if batch_size > 1:
                for batch_idx in range(num_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, len(prompts))
                    batch_prompts = prompts[start:end]
                    
                    batch_results = self.model.generate_batch_with_activations(
                        prompts=batch_prompts,
                        layers=layers,
                    )
                    
                    for result in batch_results:
                        for layer_idx, act in result.items():
                            if act is not None:
                                accumulated[layer_idx].append(act)
            else:
                for prompt in prompts:
                    activations = self.extract_single(
                        user_prompt=prompt["user"],
                        system_prompt=prompt.get("system"),
                        layers=layers,
                    )
                    
                    for layer_idx, act in activations.items():
                        if act is not None:
                            accumulated[layer_idx].append(act)
        
        return accumulated
    
    def extract_contrastive(
        self, 
        positive_prompts: list[dict], 
        negative_prompts: list[dict],
        layers: list[int] | None = None,
        show_progress: bool = True,
    ) -> dict[int, torch.Tensor]:
        """
        Extract difference vector: mean(positive) - mean(negative).
        
        Args:
            positive_prompts: Prompts for positive direction.
            negative_prompts: Prompts for negative direction.
            layers: Layers to extract from.
            show_progress: Whether to show progress bar.
            
        Returns:
            Dictionary mapping layer index to normalized difference tensor.
        """
        layers = layers or self.target_layers
        
        # Extract positive activations with batching
        print("Extracting positive activations...")
        positive_accumulated = self._extract_batch_raw(
            positive_prompts, layers, "Positive samples", show_progress
        )
        
        # Extract negative activations with batching
        print("Extracting negative activations...")
        negative_accumulated = self._extract_batch_raw(
            negative_prompts, layers, "Negative samples", show_progress
        )
        
        # Compute contrastive difference
        contrastive: dict[int, torch.Tensor] = {}
        for layer_idx in layers:
            pos_acts = positive_accumulated[layer_idx]
            neg_acts = negative_accumulated[layer_idx]
            
            if pos_acts and neg_acts:
                axis = compute_axis_from_contrastive(pos_acts, neg_acts)
                contrastive[layer_idx] = axis
            else:
                contrastive[layer_idx] = None
        
        return contrastive
    
    def extract_for_persona(
        self,
        system_prompts: list[str],
        extraction_questions: list[str],
        layers: list[int] | None = None,
        show_progress: bool = True,
        max_questions: int | None = None,
    ) -> dict[int, torch.Tensor]:
        """
        Extract persona vector from system prompts and questions.
        
        Generates responses for each (system_prompt, question) combination
        and averages the activations.
        
        Args:
            system_prompts: List of system prompts defining the persona.
            extraction_questions: List of questions to ask.
            layers: Layers to extract from.
            show_progress: Whether to show progress bar.
            max_questions: Limit number of questions (defaults to config.num_questions).
            
        Returns:
            Dictionary mapping layer index to persona activation tensor.
        """
        layers = layers or self.target_layers
        max_questions = max_questions or self.config.num_questions
        
        # Limit questions if needed
        questions = extraction_questions[:max_questions]
        
        # Build all prompt combinations
        prompts = []
        for system in system_prompts:
            for question in questions:
                prompts.append({"system": system, "user": question})
        
        print(f"Extracting from {len(prompts)} prompt combinations ({len(system_prompts)} prompts × {len(questions)} questions)...")
        return self.extract_batch(prompts, layers=layers, show_progress=show_progress)
    
    def extract_for_persona_with_judge(
        self,
        system_prompts: list[str],
        extraction_questions: list[str],
        character_description: str,
        judge_func,
        layers: list[int] | None = None,
        show_progress: bool = True,
        max_questions: int | None = None,
        min_score: int = 4,
    ) -> dict[int, torch.Tensor]:
        """
        Extract persona vector with LLM judge filtering for quality roleplay.
        
        Only keeps activations from responses that the judge rates highly.
        
        Args:
            system_prompts: List of system prompts defining the persona.
            extraction_questions: List of questions to ask.
            character_description: Description for the judge to evaluate against.
            judge_func: Function(description, response) -> (score, reason)
            layers: Layers to extract from.
            show_progress: Whether to show progress bar.
            max_questions: Limit number of questions.
            min_score: Minimum judge score to keep (1-5, default 4).
            
        Returns:
            Dictionary mapping layer index to persona activation tensor.
        """
        layers = layers or self.target_layers
        max_questions = max_questions or self.config.num_questions
        questions = extraction_questions[:max_questions]
        
        # Accumulate only high-quality activations
        accumulated: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
        kept_count = 0
        total_count = 0
        
        print(f"Extracting with judge filtering (min_score={min_score})...")
        
        for system in system_prompts:
            for question in questions:
                total_count += 1
                
                # Generate response and get activations
                response, activations = self.model.generate_with_activations(
                    user_prompt=question,
                    system_prompt=system,
                    layers=layers,
                )
                
                # Judge the response quality
                score, reason = judge_func(character_description, response)
                
                if score >= min_score:
                    kept_count += 1
                    for layer_idx, act in activations.items():
                        if act is not None:
                            accumulated[layer_idx].append(act)
                
                if total_count % 20 == 0:
                    print(f"  Progress: {total_count} samples, {kept_count} kept ({100*kept_count/total_count:.0f}%)")
        
        print(f"Judge kept {kept_count}/{total_count} samples ({100*kept_count/total_count:.0f}%)")
        
        # Average the kept activations
        averaged: dict[int, torch.Tensor] = {}
        for layer_idx, acts in accumulated.items():
            if acts:
                stacked = torch.stack(acts)
                averaged[layer_idx] = stacked.mean(dim=0)
            else:
                averaged[layer_idx] = None
        
        return averaged
    
    def collect_safe_activations(
        self,
        safe_prompts: list[dict],
        layers: list[int] | None = None,
        show_progress: bool = True,
    ) -> dict[int, list[torch.Tensor]]:
        """
        Collect individual activations for safe prompts (for floor computation).
        
        Unlike extract_batch, this returns the individual activations
        rather than averaging them, which is needed for computing percentiles.
        
        Args:
            safe_prompts: List of safe prompts.
            layers: Layers to extract from.
            show_progress: Whether to show progress bar.
            
        Returns:
            Dictionary mapping layer index to list of activation tensors.
        """
        layers = layers or self.target_layers
        return self._extract_batch_raw(
            safe_prompts, layers, "Collecting activations", show_progress
        )
