# Persona Steering System

Steer LLM behavior with extracted persona vectors. This system extracts activation patterns from an LLM when it embodies different personas, then uses those patterns to steer the model's behavior at runtime.

<img width="2718" height="1436" alt="image" src="https://github.com/user-attachments/assets/3ef262b5-cfd2-4203-8c88-78a0b3878d5e" />


## Features

- **Persona Vector Extraction**: Extract characteristic activation patterns from any persona defined by system prompts
- **Trait Vectors**: Create single-word trait vectors (e.g., "confident", "playful") using contrastive extraction
- **Runtime Steering**: Apply persona and trait vectors during generation to shift model behavior
- **Persona Blending**: Combine multiple personas and traits with weighted blending
- **Web UI**: Modern web interface for real-time persona steering and creation
- **Interactive CLI**: Full-featured command-line interface for all operations

## Requirements

- Python 3.10+
- CUDA-capable GPU (supports 4-bit quantization for reduced VRAM usage)
- Default model (Qwen3-14B with 4-bit quantization) requires ~8GB VRAM

## Installation

```bash
# Clone the repository
git clone https://github.com/user/persona-steering.git
cd persona-steering

# Install dependencies
pip install -e .
```

## Quick Start

### Option 1: Web UI (Recommended)

```bash
# Start the web server
python web/api.py

# Open http://localhost:8000 in your browser
```

The web UI provides:
- Real-time persona creation from descriptions
- Trait vector creation from single words
- Interactive chat with adjustable persona/trait weights
- Steering scale adjustment

### Option 2: Command Line

#### 1. Initialize the System

```bash
# Generate datasets and set up directories
persona-steer init
```

#### 2. Extract Vectors

```bash
# Extract assistant baseline (required for persona extraction)
python scripts/extract_assistant_axis.py

# Extract persona from a description
python scripts/extract_persona.py --from-description "a grumpy old wizard" --name grumpy_wizard

# Or extract all base archetypes
python scripts/extract_persona.py --all
```

#### 3. Start Chatting

```bash
# Basic chat (no persona)
persona-steer chat

# Chat with a persona
persona-steer chat --persona grumpy_wizard

# Chat with persona blend
persona-steer chat --blend sage:0.5,trickster:0.3
```

## CLI Commands

### Initialization
- `persona-steer init` - Initialize system and generate datasets
- `persona-steer extract-axes` - Extract assistant baseline

### Persona Management
- `persona-steer list` - List available personas
- `persona-steer list --loaded` - List loaded personas
- `persona-steer load <name>` - Load persona into memory
- `persona-steer unload <name>` - Unload persona
- `persona-steer delete <name>` - Delete persona
- `persona-steer info <name>` - Show persona details
- `persona-steer clear-personas` - Clear all persona vectors

### Persona Creation
- `persona-steer create <name> --from-archetype sage` - Create from archetype
- `persona-steer create <name> --description "A grumpy wizard"` - Create from description

### Generation
- `persona-steer generate "prompt" --persona sage` - Single generation
- `persona-steer chat` - Interactive chat session

### Configuration
- `persona-steer config show` - Show configuration
- `persona-steer config set <key> <value>` - Update configuration

## Interactive Chat Commands

Within the chat session:
- `/persona <name>` - Switch persona
- `/strength <0.0-2.0>` - Adjust steering strength
- `/blend sage:0.5,trickster:0.3` - Set persona blend
- `/none` - Disable persona steering
- `/safety on|off` - Toggle safety capping
- `/list` - Show available personas
- `/load <name>` - Load persona into memory
- `/status` - Show current settings
- `/quit` - Exit

## Base Archetypes

The system includes 17 base persona archetypes:

**Classic Archetypes:**
- **Sage** - Cryptically wise, speaks in riddles, ancient, mystical
- **Trickster** - Chaotic, pun-obsessed, mischievous, irreverent
- **Guardian** - Stern, protective, honor-bound, stoic
- **Explorer** - Breathlessly excited, tangent-prone, wonder-struck
- **Creator** - Dramatic, emotionally intense, metaphor-heavy
- **Prophet** - Apocalyptic, warning, seeing visions, urgent
- **Hermit** - Antisocial, grumpy, reluctant, paranoid

**Character Personas:**
- **Pirate** - Salty, nautical, treasure-obsessed, superstitious
- **Noir Detective** - Cynical, world-weary, narrating, suspicious
- **Valley Girl** - Like-totally, uptalking, bubbly, dramatic
- **Drill Sergeant** - Yelling, demanding, insulting, military
- **Surfer Dude** - Chill, wave-obsessed, philosophical, laid-back
- **Shakespearean** - Flowery, dramatic, thee-thy, poetic
- **Robot** - Logical, literal, emotionless, analyzing
- **Excited Puppy** - Hyperactive, loving, distracted, happy
- **Conspiracy Theorist** - Paranoid, connecting-dots, urgent, suspicious
- **Medieval Peasant** - Downtrodden, superstitious, simple, plague-fearing

## Trait Vectors

Trait vectors capture single behavioral dimensions and can be combined with personas:

```bash
# Via Web UI: Enter a word like "confident" and its opposite "insecure"

# Traits support negative weights to steer away from a trait
# Example: traits = {"confident": 0.5, "formal": -0.3}
```

## Creating Custom Personas

### From Description (Web UI or CLI)

```bash
persona-steer create grumpy_wizard --description "A grumpy but secretly kind old wizard who speaks in riddles and reluctantly helps those in need"
```

### Programmatically

```python
from src import ModelWrapper, ActivationExtractor, PersonaManager, get_config

config = get_config()
model = ModelWrapper(config)
extractor = ActivationExtractor(model, config)
manager = PersonaManager(config)

# Extract from custom system prompts
activations = extractor.extract_for_persona(
    system_prompts=["You are a grumpy wizard...", ...],
    extraction_questions=questions,
)

# Create persona from activations
manager.create_from_data(
    name="my_persona",
    raw_vector=combined_vector,
    description="My custom persona",
)
```

## How It Works

### 1. Activation Extraction
The system captures residual stream activations from transformer blocks during generation. These activations encode the model's "mental state" including personality characteristics.

### 2. Contrastive Analysis
By comparing activations from different conditions (baseline assistant vs persona), we extract direction vectors that represent specific behavioral dimensions. The contrastive approach (persona - baseline) produces vectors that steer *away* from default assistant behavior.

### 3. Runtime Steering
During generation, persona vectors are added to activations at target layers, shifting the model's behavior toward the desired persona. The steering scale controls the magnitude of this effect.

### 4. Vector Blending
Multiple persona and trait vectors can be combined with different weights, allowing fine-grained control over the model's personality.

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  name: "huihui-ai/Huihui-Qwen3-14B-abliterated-v2"
  device: "cuda"
  dtype: "float16"
  quantization: "4bit"  # Optional: "4bit", "8bit", or null
  
extraction:
  target_layers: [24, 26, 28, 30, 32, 34, 36, 38]
  max_new_tokens: 128
  temperature: 0.7
  batch_size: 10
  num_questions: 40
  
steering:
  default_strength: 1.0
  steering_scale: 0.1  # Global multiplier for steering effect

paths:
  datasets: ./datasets
  vectors: ./vectors
```

## Web API

The web server exposes a REST API:

- `GET /api/status` - System status
- `GET /api/personas` - List personas
- `POST /api/personas` - Create persona from description
- `DELETE /api/personas/{name}` - Delete persona
- `GET /api/traits` - List trait vectors
- `POST /api/traits` - Create trait vector
- `DELETE /api/traits/{name}` - Delete trait
- `POST /api/chat` - Send chat message
- `POST /api/steering-scale` - Adjust steering scale
- `WS /ws/chat` - WebSocket for streaming chat

## Project Structure

```
persona-steering/
├── config.yaml           # Configuration
├── pyproject.toml        # Package definition
├── requirements.txt      # Dependencies
├── datasets/             # Generated datasets
│   └── personas/        
│       ├── archetypes/   # System prompts for each persona
│       └── extraction_questions.jsonl
├── vectors/              # Extracted vectors
│   ├── assistant_baseline.pt  # Baseline for contrastive extraction
│   ├── personas/         # Persona vectors
│   └── traits/           # Trait vectors
├── src/                  # Source code
│   ├── config.py         # Configuration loading
│   ├── model_wrapper.py  # Model with activation hooks
│   ├── vector_math.py    # Vector operations
│   ├── activation_extractor.py
│   ├── dataset_generator.py
│   ├── persona_manager.py
│   ├── steering_engine.py
│   └── cli.py            # CLI interface
├── scripts/              # Extraction scripts
│   ├── generate_persona_dataset.py
│   ├── extract_assistant_axis.py
│   └── extract_persona.py
└── web/                  # Web interface
    ├── api.py            # FastAPI backend
    ├── index.html        # Main page
    ├── app.js            # Frontend JavaScript
    └── styles.css        # Styling
```

## Dependencies

Core:
- torch>=2.0
- transformers>=4.40
- accelerate
- bitsandbytes (for quantization)

CLI:
- click
- rich
- prompt_toolkit

Web:
- fastapi
- uvicorn
- websockets
- pydantic>=2.0

Optional:
- outlines>=0.1.0 (for structured JSON generation)

## License

MIT License
