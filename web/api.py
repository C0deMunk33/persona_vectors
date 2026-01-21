#!/usr/bin/env python3
"""FastAPI backend for the Persona Steering Web UI."""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import get_config
from src.model_wrapper import ModelWrapper
from src.persona_manager import PersonaManager
from src.steering_engine import SteeringEngine
from src.dataset_generator import DatasetGenerator
from src.activation_extractor import ActivationExtractor
import torch


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

# Custom filter to suppress 200 OK logs
class SuppressSuccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # Suppress successful HTTP responses (200, 304)
        if '" 200' in msg or '" 304' in msg:
            return False
        return True


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)

# Get our logger
log = logging.getLogger("persona_steering")
log.setLevel(logging.INFO)

# Apply filter to uvicorn access logger
uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.addFilter(SuppressSuccessFilter())


# Global state
class AppState:
    config = None
    model: ModelWrapper | None = None
    personas: PersonaManager | None = None
    engine: SteeringEngine | None = None
    generator: DatasetGenerator | None = None
    extractor: ActivationExtractor | None = None
    model_loading = False
    creating_persona: str | None = None
    creating_trait: str | None = None
    loaded_traits: dict[str, dict] = {}  # name -> trait data


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    # Startup
    log.info("═" * 50)
    log.info("Persona Steering API starting...")
    state.config = get_config()
    state.config.ensure_directories()
    state.personas = PersonaManager(state.config)
    
    # Log available personas
    available = state.personas.list_available()
    log.info("Model: %s", state.config.model_name)
    log.info("Available personas: %s", ", ".join(available) if available else "none")
    log.info("Server ready - model will load on first request")
    log.info("═" * 50)
    
    yield
    
    # Shutdown
    log.info("Shutting down...")
    if state.engine:
        state.engine.unload()
    log.info("Shutdown complete")


app = FastAPI(
    title="Persona Steering API",
    description="API for persona-steered LLM generation",
    version="0.1.0",
    lifespan=lifespan,
)


# Request/Response models
class CreatePersonaRequest(BaseModel):
    name: str
    description: str
    num_prompts: int = 5
    num_questions: int = 40
    batch_size: int = 8
    use_judge: bool = False  # Use LLM judge to filter extractions by roleplay quality


class CreateTraitRequest(BaseModel):
    word: str
    opposite: str | None = None  # Auto-generate if not provided
    num_questions: int = 30
    batch_size: int = 8


class ChatRequest(BaseModel):
    message: str
    personas: dict[str, float] = {}  # name -> weight (0-1)
    traits: dict[str, float] = {}    # name -> weight (0-1) for trait vectors
    safety_enabled: bool = True
    safety_strength: float = 1.0


class PersonaInfo(BaseModel):
    name: str
    source: str
    description: str
    traits: list[str]
    loaded: bool


class TraitInfo(BaseModel):
    name: str
    opposite: str
    loaded: bool


class StatusResponse(BaseModel):
    model_loaded: bool
    model_loading: bool
    model_name: str
    safety_axis_available: bool
    safety_floor: float | None
    creating_persona: str | None
    creating_trait: str | None
    steering_scale: float


# Helper functions
def ensure_model_loaded():
    """Ensure model is loaded, load if not."""
    if state.model is None:
        state.model_loading = True
        log.info("Loading model: %s", state.config.model_name)
        state.model = ModelWrapper(state.config)
        state.model.load()
        log.info("Model loaded successfully (hidden_size=%d, layers=%d)", 
                 state.model.hidden_size, state.model.num_layers)
        state.engine = SteeringEngine(state.model, state.personas, state.config)
        state.generator = DatasetGenerator(state.model, state.config)
        state.extractor = ActivationExtractor(state.model, state.config)
        state.model_loading = False
        log.info("Engine initialized and ready")
        
        # Auto-extract baseline if it doesn't exist
        _ensure_baseline_exists()


def _ensure_baseline_exists():
    """Extract assistant baseline if it doesn't exist for current model."""
    config = state.config
    baseline_path = config.vectors_dir / "assistant_baseline.pt"
    
    # Check if baseline exists and is for the current model
    if baseline_path.exists():
        try:
            data = torch.load(baseline_path, weights_only=False)
            saved_model = data.get("metadata", {}).get("model", "")
            if saved_model == config.model_name:
                log.info("✓ Baseline exists for %s", config.model_name)
                return
            else:
                log.info("Baseline is for different model (%s), re-extracting...", saved_model)
        except Exception as e:
            log.warning("Could not load existing baseline: %s", e)
    
    log.info("═" * 40)
    log.info("Extracting assistant baseline (one-time setup)...")
    log.info("This captures default model behavior for contrastive persona extraction")
    log.info("═" * 40)
    
    # Check for extraction questions
    questions_path = config.personas_dir / "extraction_questions.jsonl"
    if not questions_path.exists():
        log.info("Generating extraction questions first...")
        questions = state.generator.generate_extraction_questions(n=100)
        state.generator.save_jsonl(questions, questions_path)
        log.info("Generated %d extraction questions", len(questions))
    else:
        questions = state.generator.load_jsonl(questions_path)
    
    # Build prompts with NO system prompt (default assistant behavior)
    # Use fewer questions for faster baseline extraction
    num_baseline_questions = min(40, len(questions))
    prompts = [{"system": None, "user": q["question"]} for q in questions[:num_baseline_questions]]
    
    log.info("Extracting baseline from %d prompts...", len(prompts))
    
    # Extract activations
    activations = state.extractor.extract_batch(prompts, show_progress=True)
    
    # Combine layer activations - DON'T normalize
    # We need the natural magnitude for contrastive direction computation
    valid_layers = {k: v for k, v in activations.items() if v is not None}
    
    if not valid_layers:
        log.error("Failed to extract baseline - no valid activations!")
        return
    
    # Log activation magnitudes for debugging
    norms = [v.norm().item() for v in valid_layers.values()]
    log.info("  Baseline activation norms: min=%.2f, max=%.2f, mean=%.2f", min(norms), max(norms), sum(norms)/len(norms))
    
    # Compute mean across layers (for backward compatibility)
    stacked = torch.stack(list(valid_layers.values()))
    mean_vector = stacked.mean(dim=0)
    
    # Save baseline (used for contrastive persona extraction)
    torch.save({
        "vector": mean_vector,
        "per_layer_vectors": valid_layers,
        "layers": list(valid_layers.keys()),
        "num_samples": len(prompts),
        "metadata": {
            "model": config.model_name,
            "questions_file": str(questions_path),
        }
    }, baseline_path)
    
    log.info("═" * 40)
    log.info("✓ Baseline extracted and saved!")
    log.info("  Layers: %s", list(valid_layers.keys()))
    log.info("  Shape: %s", mean_vector.shape)
    log.info("═" * 40)


async def ensure_model_loaded_async():
    """Async wrapper for model loading."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, ensure_model_loaded)


# API Routes
@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current system status."""
    return StatusResponse(
        model_loaded=state.model is not None and state.model.model is not None,
        model_loading=state.model_loading,
        model_name=state.config.model_name if state.config else "unknown",
        safety_axis_available=state.personas.safety_axis is not None if state.personas else False,
        safety_floor=state.personas.safety_floor if state.personas else None,
        creating_persona=state.creating_persona,
        creating_trait=state.creating_trait,
        steering_scale=state.config.steering_scale if state.config else 0.1,
    )


@app.post("/api/load-model")
async def load_model():
    """Explicitly load the model."""
    if state.model_loading:
        return {"status": "already_loading"}
    if state.model is not None:
        return {"status": "already_loaded"}
    
    await ensure_model_loaded_async()
    return {"status": "loaded"}


class SteeringScaleRequest(BaseModel):
    scale: float


@app.post("/api/steering-scale")
async def set_steering_scale(request: SteeringScaleRequest):
    """Update the steering scale at runtime."""
    if state.config is None:
        raise HTTPException(500, "Config not initialized")
    
    # Clamp to reasonable range
    scale = max(0.01, min(2.0, request.scale))
    
    # Update config in memory and save to file
    state.config.set("steering.steering_scale", scale)
    log.info(f"Steering scale updated to {scale}")
    
    return {"status": "ok", "steering_scale": scale}


@app.get("/api/personas")
async def list_personas() -> list[PersonaInfo]:
    """List all available personas."""
    if state.personas is None:
        return []
    
    available = state.personas.list_available()
    loaded = set(state.personas.list_loaded())
    
    result = []
    for name in available:
        info = state.personas.get_info(name)
        if info:
            result.append(PersonaInfo(
                name=name,
                source=info.get("source", "unknown"),
                description=info.get("description", ""),
                traits=info.get("traits", []),
                loaded=name in loaded,
            ))
            # Unload if we just loaded it for info
            if name not in loaded:
                state.personas.unload(name)
    
    return result


@app.post("/api/personas")
async def create_persona(request: CreatePersonaRequest, background_tasks: BackgroundTasks):
    """Create a new persona from description."""
    if state.creating_persona:
        raise HTTPException(400, f"Already creating persona: {state.creating_persona}")
    
    # Validate name
    if " " in request.name:
        raise HTTPException(400, "Persona name cannot contain spaces")
    
    if request.name in state.personas.list_available():
        raise HTTPException(400, f"Persona '{request.name}' already exists")
    
    # Start creation in background
    state.creating_persona = request.name
    background_tasks.add_task(
        create_persona_task,
        request.name,
        request.description,
        request.num_prompts,
        request.num_questions,
        request.batch_size,
        request.use_judge
    )
    
    return {"status": "creating", "name": request.name}


async def create_persona_task(name: str, description: str, num_prompts: int, num_questions: int, batch_size: int, use_judge: bool):
    """Background task to create a persona."""
    try:
        await ensure_model_loaded_async()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            _create_persona_sync,
            name,
            description,
            num_prompts,
            num_questions,
            batch_size,
            use_judge
        )
    finally:
        state.creating_persona = None


def _create_persona_sync(name: str, description: str, num_prompts: int, num_questions: int = 40, batch_size: int = 8, use_judge: bool = False):
    """Synchronous persona creation using contrastive mean-difference extraction."""
    log.info("Creating persona '%s'...", name)
    log.info("Description: %s", description[:80] + "..." if len(description) > 80 else description)
    log.info("Settings: prompts=%d, questions=%d, batch=%d, use_judge=%s", num_prompts, num_questions, batch_size, use_judge)
    
    config = state.config
    generator = state.generator
    extractor = state.extractor
    
    # Check for extraction questions
    questions_path = config.personas_dir / "extraction_questions.jsonl"
    if not questions_path.exists():
        log.info("Generating extraction questions (this happens once)...")
        questions = generator.generate_extraction_questions(n=100)
        generator.save_jsonl(questions, questions_path)
        log.info("Generated %d extraction questions", len(questions))
    else:
        questions = generator.load_jsonl(questions_path)
        log.info("Loaded %d extraction questions", len(questions))
    
    # Limit to requested number of questions
    question_texts = [q["question"] for q in questions[:num_questions]]
    log.info("Using %d questions (batch_size=%d)", len(question_texts), batch_size)
    
    # Check for baseline (required for contrastive extraction)
    baseline_path = config.vectors_dir / "assistant_baseline.pt"
    has_baseline = baseline_path.exists()
    baseline_per_layer = None
    
    if has_baseline:
        baseline_data = torch.load(baseline_path, weights_only=False)
        baseline_per_layer = baseline_data.get("per_layer_vectors", {})
        log.info("✓ Loaded baseline for contrastive extraction")
    else:
        log.warning("⚠ No baseline found - steering may not work well")
    
    # Generate persona prompts
    log.info("Generating %d system prompts for persona...", num_prompts)
    persona_prompts = generator.generate_persona_prompts(description, n=num_prompts)
    if not persona_prompts:
        log.error("Failed to generate persona prompts!")
        raise Exception("Failed to generate persona prompts")
    
    # Save prompts
    for i, p in enumerate(persona_prompts):
        p["id"] = f"{name}_prompt_{i+1}"
    
    prompts_path = config.archetypes_dir / f"{name}.jsonl"
    generator.save_jsonl(persona_prompts, prompts_path)
    log.info("Saved %d prompts to %s", len(persona_prompts), prompts_path.name)
    
    system_prompts = [p["system_prompt"] for p in persona_prompts]
    
    # Extract activations
    total_combinations = len(system_prompts) * len(question_texts)
    
    # Use judged extraction for higher quality (slower but filters bad roleplay)
    # use_judge is now controlled by the UI toggle
    
    if use_judge:
        log.info("Extracting with judge filtering from %d combinations...", total_combinations)
        activations = extractor.extract_for_persona_with_judge(
            system_prompts=system_prompts,
            extraction_questions=question_texts,
            character_description=description,
            judge_func=generator.judge_roleplay_quality,
            show_progress=True,
            max_questions=num_questions,
            min_score=4,  # Only keep good roleplay (4-5)
        )
    else:
        log.info("Extracting activations from %d combinations (%d prompts × %d questions, batch=%d)...", 
                 total_combinations, len(system_prompts), len(question_texts), batch_size)
        
        # Temporarily override batch_size in extractor
        old_batch_size = extractor.batch_size
        extractor.batch_size = batch_size
        
        # Extract persona activations (averaged across all prompt/question combinations)
        activations = extractor.extract_for_persona(
            system_prompts=system_prompts,
            extraction_questions=question_texts,
            show_progress=True,
            max_questions=num_questions,
        )
        
        # Restore batch size
        extractor.batch_size = old_batch_size
    
    # Combine layer vectors
    valid_layers = {k: v for k, v in activations.items() if v is not None}
    if not valid_layers:
        log.error("No valid activations extracted!")
        raise Exception("No valid activations extracted")
    
    log.info("Extracted activations from layers: %s", list(valid_layers.keys()))
    
    # Compute CONTRASTIVE vectors per layer
    # Based on paper insights: we want to move AWAY from assistant behavior
    # The "assistant axis" points toward assistant-ness
    # So: baseline - persona = direction toward assistant
    #     persona - baseline = direction AWAY from assistant (what we want)
    mean_diff_magnitude = 1.0  # Default scaling factor
    if baseline_per_layer:
        log.info("Computing contrastive direction (persona - baseline = away from assistant)...")
        contrastive_layers = {}
        raw_magnitudes = []
        cosine_sims = []
        for layer_idx, persona_act in valid_layers.items():
            if layer_idx in baseline_per_layer and baseline_per_layer[layer_idx] is not None:
                baseline_act = baseline_per_layer[layer_idx]
                
                # Compute cosine similarity to see how different they are
                cos_sim = torch.nn.functional.cosine_similarity(
                    persona_act.unsqueeze(0), baseline_act.unsqueeze(0)
                ).item()
                cosine_sims.append(cos_sim)
                
                # Direction from baseline toward persona - KEEP THE MAGNITUDE
                direction = persona_act - baseline_act
                raw_mag = direction.norm().item()
                raw_magnitudes.append(raw_mag)
                
                log.info(f"  Layer {layer_idx}: cos_sim={cos_sim:.4f}, diff_norm={raw_mag:.3f}")
                
                # DON'T normalize - keep raw difference magnitude
                # The magnitude IS the signal strength
                contrastive_layers[layer_idx] = direction
            else:
                # Fallback for non-contrastive - use unit normalization
                norm = persona_act.norm()
                contrastive_layers[layer_idx] = persona_act / norm if norm > 0 else persona_act
        valid_layers = contrastive_layers
        mean_diff_magnitude = sum(raw_magnitudes) / len(raw_magnitudes) if raw_magnitudes else 1.0
        log.info("  Cosine similarities: min=%.4f, max=%.4f, mean=%.4f (closer to 1 = more similar)", 
                 min(cosine_sims), max(cosine_sims), sum(cosine_sims)/len(cosine_sims))
        log.info("  Raw diff magnitudes: min=%.3f, max=%.3f, mean=%.3f", 
                 min(raw_magnitudes), max(raw_magnitudes), mean_diff_magnitude)
    
    # Average across layers - keep the magnitude!
    stacked = torch.stack(list(valid_layers.values()))
    raw_vector = stacked.mean(dim=0)
    final_norm = raw_vector.norm().item()
    log.info("  Final vector norm: %.4f (preserving magnitude for steering)", final_norm)
    
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
        "contrastive": has_baseline,
        "extraction_method": "mean_difference_with_magnitude",
        "vector_magnitude": final_norm,  # Preserved magnitude for steering
        "mean_diff_magnitude": mean_diff_magnitude,  # Average per-layer diff magnitude
        "metadata": {
            "model": config.model_name,
            "prompts_file": str(prompts_path),
        }
    }, output_path)
    
    log.info("✓ Persona '%s' created! (contrastive=%s)", name, has_baseline)


@app.delete("/api/personas/{name}")
async def delete_persona(name: str):
    """Delete a persona."""
    if state.personas is None:
        raise HTTPException(500, "System not initialized")
    
    if name not in state.personas.list_available():
        raise HTTPException(404, f"Persona '{name}' not found")
    
    state.personas.delete(name)
    return {"status": "deleted", "name": name}


# ═══════════════════════════════════════════════════════════════════════════
# TRAIT API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

def _get_traits_dir() -> Path:
    """Get the directory for trait vectors."""
    traits_dir = state.config.vectors_dir / "traits"
    traits_dir.mkdir(parents=True, exist_ok=True)
    return traits_dir


def _list_available_traits() -> list[str]:
    """List all available trait .pt files."""
    traits_dir = _get_traits_dir()
    return [f.stem for f in traits_dir.glob("*.pt")]


def _load_trait(name: str) -> dict | None:
    """Load a trait from disk."""
    if name in state.loaded_traits:
        return state.loaded_traits[name]
    
    trait_path = _get_traits_dir() / f"{name}.pt"
    if not trait_path.exists():
        return None
    
    try:
        data = torch.load(trait_path, weights_only=False)
        state.loaded_traits[name] = data
        return data
    except Exception:
        return None


@app.get("/api/traits")
async def list_traits() -> list[TraitInfo]:
    """List all available traits."""
    available = _list_available_traits()
    loaded = set(state.loaded_traits.keys())
    
    result = []
    for name in available:
        data = _load_trait(name)
        if data:
            result.append(TraitInfo(
                name=name,
                opposite=data.get("opposite", ""),
                loaded=name in loaded,
            ))
    
    return result


@app.post("/api/traits")
async def create_trait(request: CreateTraitRequest, background_tasks: BackgroundTasks):
    """Create a new trait from a single word using contrastive extraction."""
    if state.creating_trait:
        raise HTTPException(400, f"Already creating trait: {state.creating_trait}")
    
    word = request.word.lower().strip()
    
    if not word:
        raise HTTPException(400, "Trait word cannot be empty")
    
    if word in _list_available_traits():
        raise HTTPException(400, f"Trait '{word}' already exists")
    
    state.creating_trait = word
    background_tasks.add_task(
        create_trait_task,
        word,
        request.opposite,
        request.num_questions,
        request.batch_size,
    )
    
    return {"status": "creating", "name": word}


async def create_trait_task(word: str, opposite: str | None, num_questions: int, batch_size: int):
    """Background task to create a trait."""
    try:
        await ensure_model_loaded_async()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            _create_trait_sync,
            word,
            opposite,
            num_questions,
            batch_size,
        )
    finally:
        state.creating_trait = None


def _get_opposite_trait(word: str) -> str:
    """Generate an opposite trait word using common antonyms or the LLM."""
    # Common antonym pairs
    antonyms = {
        "compliant": "defiant",
        "cooperative": "uncooperative", 
        "helpful": "unhelpful",
        "friendly": "hostile",
        "polite": "rude",
        "honest": "deceptive",
        "confident": "insecure",
        "calm": "anxious",
        "optimistic": "pessimistic",
        "patient": "impatient",
        "creative": "conventional",
        "curious": "incurious",
        "assertive": "passive",
        "formal": "casual",
        "serious": "playful",
        "verbose": "concise",
        "cautious": "reckless",
        "empathetic": "callous",
        "humble": "arrogant",
        "direct": "evasive",
    }
    
    # Check both directions
    if word in antonyms:
        return antonyms[word]
    for k, v in antonyms.items():
        if v == word:
            return k
    
    # Default: prefix with "un-" or "not "
    if word.startswith("un"):
        return word[2:]
    return f"un{word}"


def _create_trait_sync(word: str, opposite: str | None, num_questions: int = 30, batch_size: int = 8):
    """Synchronous trait creation using contrastive positive/negative prompts."""
    log.info("Creating trait vector for '%s'...", word)
    
    config = state.config
    extractor = state.extractor
    
    # Generate opposite if not provided
    if not opposite:
        opposite = _get_opposite_trait(word)
    
    log.info("Contrastive pair: '%s' ↔ '%s'", word, opposite)
    
    # Check for extraction questions
    questions_path = config.personas_dir / "extraction_questions.jsonl"
    if not questions_path.exists():
        log.info("Generating extraction questions...")
        questions = state.generator.generate_extraction_questions(n=100)
        state.generator.save_jsonl(questions, questions_path)
    else:
        questions = state.generator.load_jsonl(questions_path)
    
    question_texts = [q["question"] for q in questions[:num_questions]]
    log.info("Using %d extraction questions", len(question_texts))
    
    # Create contrastive system prompts
    positive_prompts = [
        f"You are a highly {word} assistant. Be {word} in all your responses.",
        f"Respond in a {word} manner. Your personality is defined by being {word}.",
        f"You embody the trait of being {word}. Let this trait guide your responses.",
    ]
    
    negative_prompts = [
        f"You are a highly {opposite} assistant. Be {opposite} in all your responses.",
        f"Respond in a {opposite} manner. Your personality is defined by being {opposite}.",
        f"You embody the trait of being {opposite}. Let this trait guide your responses.",
    ]
    
    # Temporarily set batch size
    old_batch_size = extractor.batch_size
    extractor.batch_size = batch_size
    
    # Extract POSITIVE activations (trait present)
    log.info("Extracting POSITIVE activations (trait='%s')...", word)
    positive_activations = extractor.extract_for_persona(
        system_prompts=positive_prompts,
        extraction_questions=question_texts,
        show_progress=True,
        max_questions=num_questions,
    )
    
    # Extract NEGATIVE activations (opposite trait)
    log.info("Extracting NEGATIVE activations (trait='%s')...", opposite)
    negative_activations = extractor.extract_for_persona(
        system_prompts=negative_prompts,
        extraction_questions=question_texts,
        show_progress=True,
        max_questions=num_questions,
    )
    
    # Restore batch size
    extractor.batch_size = old_batch_size
    
    # Compute contrastive direction: positive - negative
    valid_layers = {}
    raw_magnitudes = []
    cosine_sims = []
    
    for layer_idx in positive_activations.keys():
        pos_act = positive_activations.get(layer_idx)
        neg_act = negative_activations.get(layer_idx)
        
        if pos_act is None or neg_act is None:
            continue
        
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            pos_act.unsqueeze(0), neg_act.unsqueeze(0)
        ).item()
        cosine_sims.append(cos_sim)
        
        # Direction from negative to positive (captures the trait)
        direction = pos_act - neg_act
        raw_mag = direction.norm().item()
        raw_magnitudes.append(raw_mag)
        
        log.info(f"  Layer {layer_idx}: cos_sim={cos_sim:.4f}, diff_norm={raw_mag:.3f}")
        
        # Keep raw magnitude
        valid_layers[layer_idx] = direction
    
    if not valid_layers:
        log.error("No valid activations extracted!")
        raise Exception("No valid activations extracted")
    
    mean_diff_magnitude = sum(raw_magnitudes) / len(raw_magnitudes) if raw_magnitudes else 1.0
    log.info("  Cosine similarities: min=%.4f, max=%.4f, mean=%.4f", 
             min(cosine_sims), max(cosine_sims), sum(cosine_sims)/len(cosine_sims))
    log.info("  Raw diff magnitudes: min=%.3f, max=%.3f, mean=%.3f", 
             min(raw_magnitudes), max(raw_magnitudes), mean_diff_magnitude)
    
    # Average across layers
    stacked = torch.stack(list(valid_layers.values()))
    raw_vector = stacked.mean(dim=0)
    final_norm = raw_vector.norm().item()
    log.info("  Final trait vector norm: %.4f", final_norm)
    
    # Save trait
    traits_dir = _get_traits_dir()
    output_path = traits_dir / f"{word}.pt"
    torch.save({
        "raw_vector": raw_vector,
        "safe_vector": raw_vector,
        "per_layer_vectors": valid_layers,
        "layers": list(valid_layers.keys()),
        "source": "contrastive_extraction",
        "name": word,
        "opposite": opposite,
        "num_questions": len(question_texts),
        "extraction_method": "contrastive_positive_negative",
        "vector_magnitude": final_norm,
        "mean_diff_magnitude": mean_diff_magnitude,
        "metadata": {
            "model": config.model_name,
            "positive_prompts": positive_prompts,
            "negative_prompts": negative_prompts,
        }
    }, output_path)
    
    log.info("✓ Trait '%s' created! (opposite='%s')", word, opposite)


@app.delete("/api/traits/{name}")
async def delete_trait(name: str):
    """Delete a trait."""
    traits_dir = _get_traits_dir()
    trait_path = traits_dir / f"{name}.pt"
    
    if not trait_path.exists():
        raise HTTPException(404, f"Trait '{name}' not found")
    
    # Unload from memory
    if name in state.loaded_traits:
        del state.loaded_traits[name]
    
    # Delete from disk
    trait_path.unlink()
    return {"status": "deleted", "name": name}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Send a chat message and get a response."""
    await ensure_model_loaded_async()
    
    loop = asyncio.get_event_loop()
    
    # Get active personas and traits
    active_personas = {k: v for k, v in request.personas.items() if v != 0} if request.personas else {}
    active_traits = {k: v for k, v in request.traits.items() if v != 0} if request.traits else {}
    
    # Log the request
    parts = []
    if active_personas:
        parts.append(" + ".join(f"{k}:{v:+.0%}" for k, v in active_personas.items()))
    if active_traits:
        parts.append(" + ".join(f"T:{k}:{v:+.0%}" for k, v in active_traits.items()))
    persona_info = " | ".join(parts) if parts else "no vectors"

    log.info("Chat: [%s] %s", 
             persona_info,
             request.message[:50] + "..." if len(request.message) > 50 else request.message)
    
    # Combine persona and trait vectors
    has_vectors = active_personas or active_traits
    
    if has_vectors:
        # Load required personas
        for name in active_personas.keys():
            state.personas.load(name)
        
        # Load required traits
        for name in active_traits.keys():
            _load_trait(name)
        
        response = await loop.run_in_executor(
            None,
            lambda: _generate_with_vectors(
                request.message,
                active_personas,
                active_traits,
                request.safety_enabled,
            )
        )
    else:
        response = await loop.run_in_executor(
            None,
            lambda: state.engine.generate(
                user_prompt=request.message,
                enable_safety_capping=request.safety_enabled,
            )
        )
    
    log.info("Response: %d chars", len(response))
    return {"response": response}


def _generate_with_vectors(message: str, personas: dict, traits: dict, safety_enabled: bool) -> str:
    """Generate with combined persona and trait vectors."""
    from src.vector_math import blend_vectors
    
    # Collect all vectors with weights
    all_vectors = []
    all_weights = []
    per_layer_vectors: dict[int, list[tuple[torch.Tensor, float]]] = {}
    
    # Add persona vectors
    for name, weight in personas.items():
        # Try per-layer first
        per_layer = state.personas.get_per_layer_vectors(name)
        if per_layer:
            for layer_idx, vec in per_layer.items():
                if layer_idx not in per_layer_vectors:
                    per_layer_vectors[layer_idx] = []
                per_layer_vectors[layer_idx].append((vec, weight))
        else:
            vec = state.personas.get_vector(name, safe=True)
            if vec is not None:
                all_vectors.append(vec)
                all_weights.append(weight)
    
    # Add trait vectors
    for name, weight in traits.items():
        trait_data = state.loaded_traits.get(name)
        if trait_data:
            per_layer = trait_data.get("per_layer_vectors", {})
            if per_layer:
                for layer_idx, vec in per_layer.items():
                    if layer_idx not in per_layer_vectors:
                        per_layer_vectors[layer_idx] = []
                    per_layer_vectors[layer_idx].append((vec, weight))
            else:
                vec = trait_data.get("raw_vector") or trait_data.get("safe_vector")
                if vec is not None:
                    all_vectors.append(vec)
                    all_weights.append(weight)
    
    steering_scale = state.config.steering_scale
    
    # If we have per-layer vectors, use them
    if per_layer_vectors:
        blended_per_layer = {}
        for layer_idx, vec_weight_pairs in per_layer_vectors.items():
            layer_sum = None
            for vec, weight in vec_weight_pairs:
                scaled_vec = weight * steering_scale * vec
                if layer_sum is None:
                    layer_sum = scaled_vec
                else:
                    layer_sum = layer_sum + scaled_vec
            if layer_sum is not None:
                blended_per_layer[layer_idx] = layer_sum
        
        return state.model.generate_with_per_layer_steering(
            user_prompt=message,
            layer_vectors=blended_per_layer,
            strength=1.0,  # Scaling already applied
            max_new_tokens=state.config.max_new_tokens,
        )
    elif all_vectors:
        # Fallback to single blended vector
        composite_vector = blend_vectors(all_vectors, all_weights)
        scaled_vector = steering_scale * composite_vector
        
        return state.model.generate_with_steering(
            user_prompt=message,
            steering_vector=scaled_vector,
            strength=1.0,  # Scaling already applied
            max_new_tokens=state.config.max_new_tokens,
        )
    else:
        return state.engine.generate(
            user_prompt=message,
            enable_safety_capping=safety_enabled,
        )


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat."""
    await websocket.accept()
    log.info("WebSocket client connected")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            message = data.get("message", "")
            personas = data.get("personas", {})
            traits = data.get("traits", {})
            safety_enabled = data.get("safety_enabled", True)
            
            if not message:
                await websocket.send_json({"type": "error", "error": "No message provided"})
                continue
            
            # Ensure model loaded
            await ensure_model_loaded_async()
            
            # Get active personas and traits
            active_personas = {k: v for k, v in personas.items() if v != 0} if personas else {}
            active_traits = {k: v for k, v in traits.items() if v != 0} if traits else {}
            
            # Log the request
            parts = []
            if active_personas:
                parts.append(" + ".join(f"{k}:{v:+.0%}" for k, v in active_personas.items()))
            if active_traits:
                parts.append(" + ".join(f"T:{k}:{v:+.0%}" for k, v in active_traits.items()))
            vector_info = " | ".join(parts) if parts else "no vectors"
            
            log.info("WS Chat: [%s] %s", 
                     vector_info,
                     message[:50] + "..." if len(message) > 50 else message)
            
            # Send "thinking" status
            await websocket.send_json({"type": "status", "status": "generating"})
            
            # Generate response (not truly streaming, but simulates it)
            loop = asyncio.get_event_loop()
            
            try:
                has_vectors = active_personas or active_traits
                
                if has_vectors:
                    # Load personas and traits
                    for name in active_personas.keys():
                        state.personas.load(name)
                    for name in active_traits.keys():
                        _load_trait(name)
                    
                    response = await loop.run_in_executor(
                        None,
                        lambda: _generate_with_vectors(
                            message,
                            active_personas,
                            active_traits,
                            safety_enabled,
                        )
                    )
                else:
                    response = await loop.run_in_executor(
                        None,
                        lambda: state.engine.generate(
                            user_prompt=message,
                            enable_safety_capping=safety_enabled,
                        )
                    )
                
                log.info("WS Response: %d chars", len(response))
                
                # Send response in chunks to simulate streaming
                chunk_size = 20
                for i in range(0, len(response), chunk_size):
                    chunk = response[i:i + chunk_size]
                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk,
                        "done": i + chunk_size >= len(response)
                    })
                    await asyncio.sleep(0.02)  # Small delay for effect
                
                await websocket.send_json({"type": "done"})
                
            except Exception as e:
                log.error("WS Error: %s", str(e))
                await websocket.send_json({"type": "error", "error": str(e)})
    
    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")


# Serve static files
web_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/")
async def root():
    """Serve the main page."""
    return FileResponse(web_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn with custom log config
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s │ %(levelname)-8s │ %(message)s"
    log_config["formatters"]["default"]["datefmt"] = "%H:%M:%S"
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s │ %(levelname)-8s │ %(message)s"
    log_config["formatters"]["access"]["datefmt"] = "%H:%M:%S"
    
    # Handle Ctrl+C gracefully - exit immediately without waiting for background tasks
    def handle_sigint(sig, frame):
        log.info("Ctrl+C received - shutting down immediately")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_sigint)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_config=log_config,
    )
