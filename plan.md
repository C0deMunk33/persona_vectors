# Persona Steering System - Implementation Guide

## Overview

Build a system that extracts persona vectors from an LLM, maintains a safety baseline via the Assistant Axis, and allows runtime persona steering through a CLI.

**Target Hardware:** RTX 5080 (16GB VRAM)
**Recommended Model:** `qwen2.5:14b` or `gemma2:9b` via Ollama for inference, but we need raw model access for activation extraction, so we'll use the HuggingFace version with `transformers` for extraction and can use Ollama for the generation parts.

**Revised Model Choice:** `Qwen/Qwen2.5-7B-Instruct` via HuggingFace transformers (fits in 16GB with room for activation caching)

---

## Project Structure

```
persona-steering/
├── README.md
├── requirements.txt
├── config.yaml
├── datasets/
│   ├── safety/
│   │   ├── safe_prompts.jsonl
│   │   └── unsafe_prompts.jsonl
│   └── personas/
│       ├── extraction_questions.jsonl
│       └── archetypes/
│           └── {persona_name}.jsonl
├── vectors/
│   ├── assistant_axis.pt
│   ├── safety_axis.pt
│   └── personas/
│       └── {persona_name}.pt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model_wrapper.py
│   ├── activation_extractor.py
│   ├── vector_math.py
│   ├── persona_manager.py
│   ├── steering_engine.py
│   ├── dataset_generator.py
│   └── cli.py
└── scripts/
    ├── generate_safety_dataset.py
    ├── generate_persona_dataset.py
    ├── extract_assistant_axis.py
    ├── extract_safety_axis.py
    └── extract_persona.py
```

---

## TODO 1: Environment Setup

### File: `requirements.txt`

```
torch>=2.0
transformers>=4.40
accelerate
safetensors
pyyaml
click
rich
prompt_toolkit
jsonlines
numpy
```

### File: `config.yaml`

Define all configurable parameters:

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  device: "cuda"
  dtype: "bfloat16"
  
extraction:
  target_layers: [14, 15, 16, 17, 18, 19, 20, 21]  # middle layers for 28-layer model
  num_samples_per_prompt: 50
  max_new_tokens: 256
  
steering:
  default_strength: 0.3
  safety_floor_percentile: 25
  capping_layers: [18, 19, 20, 21, 22, 23, 24, 25]  # later layers for capping
  
paths:
  datasets: "./datasets"
  vectors: "./vectors"
```

---

## TODO 2: Model Wrapper with Activation Hooks

### File: `src/model_wrapper.py`

Create a wrapper that:
1. Loads the model with hooks to capture activations
2. Provides methods to get activations at specific layers
3. Supports activation injection for steering

**Requirements:**
- Load Qwen2.5-7B-Instruct in bfloat16
- Register forward hooks on residual stream (post-MLP) at configurable layers
- Store activations in a dictionary keyed by layer index
- Provide `generate_with_activations(prompt, system) -> (response, activations_dict)`
- Provide `generate_with_steering(prompt, system, steering_vector, strength, layers) -> response`
- Provide `generate_with_capping(prompt, system, axis, floor, layers) -> response`
- Clear activation cache between generations

**Key implementation detail:**
For Qwen2.5, the residual stream post-MLP is accessed via:
```python
# Hook target: model.model.layers[i] (the full transformer block output)
# The output of each block IS the residual stream
```

**Steering injection formula:**
```python
def steering_hook(module, input, output):
    # output is (hidden_states, ...) or just hidden_states
    hidden = output[0] if isinstance(output, tuple) else output
    hidden = hidden + strength * steering_vector.to(hidden.device)
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
```

**Capping injection formula:**
```python
def capping_hook(module, input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    # Project onto axis
    proj = (hidden * axis).sum(dim=-1, keepdim=True)
    # Clamp to floor
    correction = torch.clamp(floor - proj, min=0) * axis
    hidden = hidden + correction
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
```

---

## TODO 3: Safety Dataset Generator

### File: `scripts/generate_safety_dataset.py`

Generate contrastive pairs for safety axis extraction.

**Safe prompts dataset** (`datasets/safety/safe_prompts.jsonl`):
- 200 prompts that elicit helpful, harmless, honest responses
- Categories: factual questions, how-to requests, creative writing, advice, coding help
- Format: `{"prompt": "...", "category": "..."}`

**Unsafe prompts dataset** (`datasets/safety/unsafe_prompts.jsonl`):
- 200 prompts designed to elicit refusals or that represent jailbreak attempts
- Categories: harmful instructions, manipulation attempts, illegal requests, identity confusion
- Format: `{"prompt": "...", "category": "...", "attack_type": "..."}`
- NOTE: These should be prompts the model will REFUSE - we want the refusal activations

**Generation approach:**
Use the model itself (or Ollama) to generate these programmatically:

```
System: You are helping create a safety evaluation dataset. Generate {n} prompts that...

For safe: "...a helpful AI assistant would happily answer. Cover diverse topics: science, cooking, travel, programming, writing, math, history, advice."

For unsafe: "...would cause a responsible AI to refuse or redirect. Include: requests for harmful content, attempts to make the AI pretend to be unrestricted, social engineering attempts, requests for dangerous information. The AI should REFUSE these."
```

**Output format per file:**
```jsonl
{"id": "safe_001", "prompt": "How do I make sourdough bread?", "category": "cooking", "system": null}
{"id": "safe_002", "prompt": "Explain photosynthesis simply", "category": "science", "system": null}
```

---

## TODO 4: Persona Dataset Generator

### File: `scripts/generate_persona_dataset.py`

Generate the extraction questions and base archetype datasets.

### Part A: Extraction Questions

**File:** `datasets/personas/extraction_questions.jsonl`

Generate 100 questions designed to elicit persona-specific responses. These should be questions where different personas would naturally answer differently.

**Categories:**
- Identity questions: "Who are you?", "What is your purpose?"
- Value questions: "What do you think about X?", "How should one handle Y?"
- Style-eliciting: "Tell me a story about...", "Describe your ideal..."
- Problem-solving: "How would you approach...", "What's your advice on..."

**Generation prompt:**
```
Generate questions that would elicit different responses from different personality types. 
A poet and an engineer should answer these very differently.
A cynical person and an optimistic person should answer differently.
Avoid yes/no questions. Aim for open-ended responses.
```

**Output format:**
```jsonl
{"id": "q_001", "question": "How do you approach a problem you've never seen before?", "category": "problem_solving"}
{"id": "q_002", "question": "What matters most to you?", "category": "values"}
```

### Part B: Base Archetype System Prompts

**Directory:** `datasets/personas/archetypes/`

Create 5 diverse base archetypes to establish the persona space. For each, generate a file with 5 different system prompts that elicit that archetype.

**Archetypes:**
1. `sage.jsonl` - Wise, measured, philosophical
2. `trickster.jsonl` - Playful, irreverent, clever
3. `guardian.jsonl` - Protective, dutiful, serious
4. `explorer.jsonl` - Curious, adventurous, open
5. `creator.jsonl` - Artistic, expressive, imaginative

**Format per archetype file:**
```jsonl
{"id": "sage_prompt_1", "system_prompt": "You are an ancient sage who speaks with measured wisdom...", "traits": ["wise", "calm", "philosophical"]}
{"id": "sage_prompt_2", "system_prompt": "Embody a contemplative scholar who values deep understanding...", "traits": ["wise", "calm", "philosophical"]}
```

**Generation prompt for each archetype:**
```
Generate 5 different system prompts that would make an AI embody the {archetype} archetype.
Each prompt should be 2-4 sentences.
They should elicit the same core personality but with slightly different emphasis.
Traits to embody: {traits}
Do not mention being an AI. Write as if instructing the AI to BE this persona.
```

---

## TODO 5: Activation Extractor

### File: `src/activation_extractor.py`

Core extraction logic.

**Class: `ActivationExtractor`**

**Methods:**

```python
def __init__(self, model_wrapper, config):
    """Store reference to model wrapper and config."""
    
def extract_single(self, system_prompt: str | None, user_prompt: str) -> dict[int, torch.Tensor]:
    """
    Generate a response and return mean activation per layer.
    
    Returns: {layer_idx: tensor of shape (hidden_dim,)}
    Mean is taken across all response tokens.
    """
    
def extract_batch(self, prompts: list[dict]) -> dict[int, torch.Tensor]:
    """
    Extract activations for multiple prompts and return averaged.
    
    prompts: [{"system": str|None, "user": str}, ...]
    Returns: {layer_idx: tensor of shape (hidden_dim,)} averaged across all prompts
    """
    
def extract_contrastive(self, positive_prompts: list[dict], negative_prompts: list[dict]) -> dict[int, torch.Tensor]:
    """
    Extract difference vector: mean(positive) - mean(negative)
    
    Returns: {layer_idx: normalized difference tensor}
    """
```

**Implementation notes:**
- Run generation with `model_wrapper.generate_with_activations()`
- For each generation, get activations at target layers
- Average across sequence positions (response tokens only)
- Accumulate and average across all prompts
- Normalize final vectors to unit length

---

## TODO 6: Vector Math Utilities

### File: `src/vector_math.py`

**Functions:**

```python
def normalize(v: torch.Tensor) -> torch.Tensor:
    """L2 normalize a vector."""
    return v / v.norm()

def project_onto(v: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Project v onto axis (assumes axis is normalized)."""
    return (v @ axis) * axis

def remove_component(v: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Remove the component of v along axis."""
    return v - project_onto(v, axis)

def orthogonalize_persona(persona_vec: torch.Tensor, safety_axis: torch.Tensor) -> torch.Tensor:
    """
    Remove safety-relevant component from persona vector.
    Returns normalized orthogonalized vector.
    """
    orthogonal = remove_component(persona_vec, safety_axis)
    return normalize(orthogonal)

def compute_safe_floor(activations: list[torch.Tensor], axis: torch.Tensor, percentile: int = 25) -> float:
    """
    Compute the safety floor threshold.
    
    activations: list of activation tensors from normal/safe generations
    axis: the safety or assistant axis
    percentile: which percentile to use as floor
    
    Returns: scalar threshold value
    """
    projections = [act @ axis for act in activations]
    return torch.quantile(torch.stack(projections), percentile / 100.0).item()

def blend_vectors(vectors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """
    Weighted combination of vectors.
    Returns normalized result.
    """
    result = sum(w * v for w, v in zip(weights, vectors))
    return normalize(result)
```

---

## TODO 7: Axis Extraction Scripts

### File: `scripts/extract_assistant_axis.py`

**Purpose:** Extract the baseline "Assistant" vector and compute the Assistant Axis.

**Process:**
1. Load extraction questions from `datasets/personas/extraction_questions.jsonl`
2. Generate responses WITH NO SYSTEM PROMPT (default assistant behavior)
3. Extract mean activations across all responses
4. This becomes `assistant_baseline` vector
5. Later, after personas are extracted, compute:
   ```
   assistant_axis = assistant_baseline - mean(all_persona_vectors)
   ```
6. Save to `vectors/assistant_axis.pt`

**Initial run (before personas exist):**
- Just save `assistant_baseline.pt`
- The axis computation happens after persona extraction

**Output format:**
```python
torch.save({
    "vector": tensor,  # shape: (hidden_dim,)
    "layers": [14, 15, ...],  # which layers this was extracted from
    "num_samples": 100,
    "metadata": {...}
}, "vectors/assistant_baseline.pt")
```

### File: `scripts/extract_safety_axis.py`

**Purpose:** Extract a dedicated safety axis from contrastive safe/unsafe prompts.

**Process:**
1. Load `datasets/safety/safe_prompts.jsonl`
2. Load `datasets/safety/unsafe_prompts.jsonl`
3. Use `extractor.extract_contrastive(safe, unsafe)`
4. This gives us a direction where positive = safe behavior
5. Save to `vectors/safety_axis.pt`
6. Also compute and save the safety floor threshold

**Output:**
```python
torch.save({
    "vector": tensor,
    "floor": float,  # 25th percentile threshold
    "layers": [...],
    "metadata": {...}
}, "vectors/safety_axis.pt")
```

---

## TODO 8: Persona Extraction Script

### File: `scripts/extract_persona.py`

**Purpose:** Extract a persona vector from either archetype files or on-the-fly generation.

**CLI interface:**
```bash
# From existing archetype
python scripts/extract_persona.py --from-archetype sage

# From a description (generates dataset on the fly)
python scripts/extract_persona.py --from-description "A grumpy but secretly kind old wizard" --name grumpy_wizard
```

**Process for `--from-archetype`:**
1. Load system prompts from `datasets/personas/archetypes/{name}.jsonl`
2. Load extraction questions
3. For each (system_prompt, question) pair, generate response and extract activations
4. Average all activations
5. Orthogonalize against safety axis
6. Save to `vectors/personas/{name}.pt`

**Process for `--from-description`:**
1. Use LLM to generate 5 system prompts for the description:
   ```
   Given this character description: "{description}"
   Generate 5 different system prompts that would make an AI embody this character.
   Each should be 2-4 sentences. Do not mention being an AI.
   ```
2. Save generated prompts to `datasets/personas/archetypes/{name}.jsonl`
3. Proceed as above

**Output format:**
```python
torch.save({
    "raw_vector": tensor,  # before orthogonalization
    "safe_vector": tensor,  # after orthogonalization
    "layers": [...],
    "source": "archetype" | "generated",
    "description": str | None,
    "metadata": {...}
}, "vectors/personas/{name}.pt")
```

---

## TODO 9: Persona Manager

### File: `src/persona_manager.py`

**Class: `PersonaManager`**

Handles loading, saving, listing, and managing persona vectors.

```python
class PersonaManager:
    def __init__(self, vectors_dir: str, safety_axis_path: str):
        """Load safety axis, initialize empty active personas dict."""
        
    def list_available(self) -> list[str]:
        """List all persona .pt files in vectors/personas/"""
        
    def list_loaded(self) -> list[str]:
        """List currently loaded persona names."""
        
    def load(self, name: str) -> bool:
        """
        Load a persona from disk into memory.
        Returns True if successful.
        """
        
    def unload(self, name: str) -> bool:
        """Remove persona from active memory."""
        
    def delete(self, name: str) -> bool:
        """
        Delete persona from disk and memory.
        Requires confirmation.
        """
        
    def get_vector(self, name: str, safe: bool = True) -> torch.Tensor | None:
        """Get the (optionally safe) vector for a loaded persona."""
        
    def create_from_description(self, description: str, name: str, extractor, generator) -> bool:
        """
        Generate a new persona on the fly.
        Uses LLM to create prompts, extracts vectors, saves.
        """
        
    def get_composite(self, personas: dict[str, float]) -> torch.Tensor:
        """
        Create a blended persona vector.
        personas: {"sage": 0.5, "trickster": 0.3, ...}
        """
```

---

## TODO 10: Steering Engine

### File: `src/steering_engine.py`

Combines everything for runtime generation with steering.

**Class: `SteeringEngine`**

```python
class SteeringEngine:
    def __init__(self, model_wrapper, persona_manager, config):
        """Initialize with model, personas, safety settings."""
        
    def generate(
        self,
        user_prompt: str,
        persona: str | None = None,
        persona_strength: float = 0.3,
        enable_safety_capping: bool = True,
        system_override: str | None = None
    ) -> str:
        """
        Generate a response with optional persona steering and safety capping.
        
        1. If persona specified, get its safe vector
        2. Apply steering hooks at steering layers
        3. Apply capping hooks at capping layers
        4. Generate response
        5. Clean up hooks
        6. Return response
        """
        
    def generate_with_composite(
        self,
        user_prompt: str,
        persona_blend: dict[str, float],
        enable_safety_capping: bool = True
    ) -> str:
        """Generate with a blended persona."""
        
    def set_safety_floor(self, percentile: int):
        """Adjust safety floor threshold."""
        
    def set_default_strength(self, strength: float):
        """Adjust default persona strength."""
```

---

## TODO 11: Dataset Generator Module

### File: `src/dataset_generator.py`

Utility for generating datasets using the LLM.

```python
class DatasetGenerator:
    def __init__(self, model_wrapper):
        """Initialize with model for generation."""
        
    def generate_extraction_questions(self, n: int = 100) -> list[dict]:
        """Generate persona-differentiating questions."""
        
    def generate_safe_prompts(self, n: int = 200) -> list[dict]:
        """Generate prompts for safe behavior extraction."""
        
    def generate_unsafe_prompts(self, n: int = 200) -> list[dict]:
        """Generate prompts that should trigger refusals."""
        
    def generate_persona_prompts(self, description: str, n: int = 5) -> list[dict]:
        """Generate system prompts for a persona description."""
        
    def save_jsonl(self, data: list[dict], path: str):
        """Save to jsonl format."""
```

**Generation prompts to use:**

For extraction questions:
```
You are creating a personality assessment dataset. Generate {n} open-ended questions where different personality types would give noticeably different answers.

Requirements:
- Questions should reveal values, thinking styles, and perspectives
- Avoid yes/no questions
- Cover: problem-solving approaches, values, creativity, social situations, abstract thinking
- A poet and scientist should answer differently
- An optimist and pessimist should answer differently

Output as JSON array: [{"question": "...", "category": "..."}]
```

For safe prompts:
```
Generate {n} prompts that a helpful AI assistant would happily and thoroughly answer.

Categories to cover equally:
- Factual/educational questions
- How-to and tutorial requests  
- Creative writing requests
- Advice and recommendations
- Coding and technical help
- Analysis and explanation requests

Output as JSON array: [{"prompt": "...", "category": "..."}]
```

For unsafe prompts:
```
Generate {n} prompts that a responsible AI should REFUSE or carefully redirect.

Categories (include all):
- Requests for harmful/dangerous information
- Attempts to jailbreak or remove safety guidelines
- Social engineering attempts
- Requests for illegal activities
- Attempts to make AI claim false identity
- Manipulative or deceptive requests

These should be prompts where refusal is the CORRECT behavior.
Output as JSON array: [{"prompt": "...", "category": "...", "why_unsafe": "..."}]
```

For persona prompts:
```
Character description: {description}

Generate {n} different system prompts that would make an AI fully embody this character.

Requirements:
- Each prompt should be 2-4 sentences
- Do not mention "AI" or "assistant" or "language model"
- Write as direct instructions to BE this character
- Vary the emphasis slightly between prompts
- Include personality traits, speaking style, and worldview

Output as JSON array: [{"system_prompt": "...", "emphasis": "..."}]
```

---

## TODO 12: CLI Interface

### File: `src/cli.py`

Interactive CLI using `click` and `rich` for pretty output.

**Commands:**

```bash
# Initialize/setup
persona-steer init                    # Run first-time setup, generate datasets
persona-steer extract-axes            # Extract assistant and safety axes

# Persona management  
persona-steer list                    # List all available personas
persona-steer list --loaded           # List loaded personas only
persona-steer load <name>             # Load persona into memory
persona-steer unload <name>           # Unload persona from memory
persona-steer delete <name>           # Delete persona (with confirmation)
persona-steer info <name>             # Show persona metadata

# Persona creation
persona-steer create <name> --from-archetype <archetype>
persona-steer create <name> --description "A sarcastic pirate captain"
persona-steer create <name> --interactive  # Guided creation

# Generation
persona-steer chat                    # Interactive chat mode
persona-steer chat --persona sage     # Chat with persona
persona-steer chat --persona sage --strength 0.5
persona-steer chat --blend sage:0.5,trickster:0.3

# Single generation
persona-steer generate "Tell me a story" --persona sage

# Settings
persona-steer config set steering.default_strength 0.4
persona-steer config show
```

**Interactive Chat Mode Features:**
- Show current persona and strength in prompt
- Commands within chat:
  - `/persona <name>` - switch persona
  - `/strength <float>` - adjust strength
  - `/blend <name:weight,...>` - set blend
  - `/none` - disable persona
  - `/safety on|off` - toggle safety capping
  - `/list` - show loaded personas
  - `/load <name>` - load persona
  - `/create <name>` - create new persona interactively
  - `/quit` - exit

**Implementation structure:**

```python
import click
from rich.console import Console
from rich.prompt import Prompt
from prompt_toolkit import PromptSession

@click.group()
@click.pass_context
def cli(ctx):
    """Persona Steering System"""
    ctx.ensure_object(dict)
    # Initialize console, load config
    
@cli.command()
def init():
    """Initialize the system and generate base datasets."""
    # 1. Check/create directory structure
    # 2. Generate safety dataset
    # 3. Generate extraction questions
    # 4. Generate base archetype datasets
    # 5. Extract assistant baseline
    # 6. Extract safety axis
    # 7. Extract base persona vectors
    
@cli.command()
@click.argument('name')
@click.option('--description', '-d', help='Character description')
@click.option('--from-archetype', '-a', help='Base archetype to use')
@click.option('--interactive', '-i', is_flag=True, help='Guided creation')
def create(name, description, from_archetype, interactive):
    """Create a new persona."""
    
@cli.command()
@click.option('--persona', '-p', help='Persona to use')
@click.option('--strength', '-s', type=float, default=0.3)
@click.option('--blend', '-b', help='Blend spec like sage:0.5,trickster:0.3')
def chat(persona, strength, blend):
    """Start interactive chat session."""
    
# ... etc for other commands
```

---

## TODO 13: Main Entry Point

### File: `src/__init__.py` and `__main__.py`

Set up package and entry point.

```python
# __main__.py
from src.cli import cli

if __name__ == "__main__":
    cli()
```

### File: `setup.py` or `pyproject.toml`

```toml
[project]
name = "persona-steering"
version = "0.1.0"
dependencies = [...]

[project.scripts]
persona-steer = "src.cli:cli"
```

---

## Execution Order

1. **Setup environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize system**
   ```bash
   persona-steer init
   ```
   This runs:
   - Directory creation
   - Dataset generation (safety + extraction questions + archetypes)
   - Axis extraction (assistant baseline + safety axis)
   - Base persona extraction (5 archetypes)

3. **Test basic chat**
   ```bash
   persona-steer chat
   ```

4. **Load and test a persona**
   ```bash
   persona-steer load sage
   persona-steer chat --persona sage
   ```

5. **Create custom persona**
   ```bash
   persona-steer create grumpy_wizard --description "A grumpy but secretly kind old wizard who speaks in riddles"
   persona-steer chat --persona grumpy_wizard
   ```

---

## Key Implementation Notes

### Memory Management
- Load model in bfloat16 to fit in 16GB
- Only keep a few persona vectors loaded at once (each is ~14KB for 7B model)
- Clear activation cache between generations
- Use gradient checkpointing if needed during extraction

### Activation Hook Details
For Qwen2.5 architecture:
```python
# Target the output of transformer blocks
# model.model.layers[i] returns (hidden_states, present_key_value, ...)
# We want hidden_states which IS the residual stream

def make_capture_hook(storage, layer_idx):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        storage[layer_idx] = hidden.detach().clone()
        return output
    return hook

# Register:
for i in target_layers:
    handle = model.model.layers[i].register_forward_hook(make_capture_hook(storage, i))
```

### Safety Floor Calibration
Run ~100 normal generations (no persona steering) and compute projections onto safety axis:
```python
floor = torch.quantile(torch.tensor(projections), 0.25)
```

### Extraction Best Practices
- Use temperature=0.7 for diverse but coherent responses during extraction
- Exclude the prompt tokens, only average over response tokens
- Extract from multiple layers and store per-layer vectors (you may want to experiment with which layer works best for steering)

---

## Testing Checklist

- [ ] Model loads and generates without steering
- [ ] Activation hooks capture correct shapes
- [ ] Safety axis points in expected direction (safe prompts project positive)
- [ ] Persona vectors are meaningfully different from each other
- [ ] Orthogonalization removes safety component without destroying persona
- [ ] Steering produces noticeable style changes
- [ ] Safety capping prevents drift into refusal territory with unsafe prompts
- [ ] CLI commands all work
- [ ] On-the-fly persona creation produces usable vectors
- [ ] Blended personas work as expected