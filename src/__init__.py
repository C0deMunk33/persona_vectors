"""Persona Steering System - Steer LLM behavior with extracted persona vectors."""

from .config import Config, get_config
from .model_wrapper import ModelWrapper
from .vector_math import (
    normalize,
    project_onto,
    remove_component,
    orthogonalize_persona,
    compute_safe_floor,
    blend_vectors,
    cosine_similarity,
)
from .activation_extractor import ActivationExtractor
from .dataset_generator import DatasetGenerator, ARCHETYPES
from .persona_manager import PersonaManager
from .steering_engine import SteeringEngine

__version__ = "0.1.0"

__all__ = [
    # Config
    "Config",
    "get_config",
    # Model
    "ModelWrapper",
    # Vector math
    "normalize",
    "project_onto",
    "remove_component",
    "orthogonalize_persona",
    "compute_safe_floor",
    "blend_vectors",
    "cosine_similarity",
    # Extraction
    "ActivationExtractor",
    "DatasetGenerator",
    "ARCHETYPES",
    # Runtime
    "PersonaManager",
    "SteeringEngine",
]
