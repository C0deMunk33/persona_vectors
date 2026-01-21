# Persona Steering System

Steer LLM behavior with extracted persona vectors. This system extracts activation patterns from an LLM when it embodies different personas, then uses those patterns to steer the model's behavior at runtime.

## Features

- **Persona Vector Extraction**: Extract characteristic activation patterns from any persona defined by system prompts
- **Safety Axis**: Maintain safe behavior through contrastive safety axis and floor capping
- **Runtime Steering**: Apply persona vectors during generation to shift model behavior
- **Persona Blending**: Combine multiple personas with weighted blending
- **Interactive CLI**: Full-featured command-line interface for all operations

## Requirements

- Python 3.10+
- CUDA-capable GPU with 16GB+ VRAM (tested on RTX 5080)
- ~14GB VRAM for Qwen2.5-7B-Instruct in bfloat16

## Installation

```bash
# Clone the repository
git clone https://github.com/user/persona-steering.git
cd persona-steering

# Install dependencies
pip install -e .
```

## Quick Start

### 1. Initialize the System

```bash
# Generate datasets and set up directories
persona-steer init
```

### 2. Extract Vectors

```bash
# Extract safety axis (required for safe steering)
python scripts/extract_safety_axis.py

# Extract assistant baseline
python scripts/extract_assistant_axis.py

# Extract all base persona archetypes
python scripts/extract_persona.py --all
```

### 3. Start Chatting

```bash
# Basic chat (no persona)
persona-steer chat

# Chat with a persona
persona-steer chat --persona sage

# Chat with persona blend
persona-steer chat --blend sage:0.5,trickster:0.3
```

## CLI Commands

### Initialization
- `persona-steer init` - Initialize system and generate datasets
- `persona-steer extract-axes` - Extract safety and assistant axes

### Persona Management
- `persona-steer list` - List available personas
- `persona-steer list --loaded` - List loaded personas
- `persona-steer load <name>` - Load persona into memory
- `persona-steer unload <name>` - Unload persona
- `persona-steer delete <name>` - Delete persona
- `persona-steer info <name>` - Show persona details

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
- `/status` - Show current settings
- `/quit` - Exit

## Base Archetypes

The system includes 5 base persona archetypes:

1. **Sage** - Wise, calm, philosophical, measured
2. **Trickster** - Playful, irreverent, clever, witty
3. **Guardian** - Protective, dutiful, serious, steadfast
4. **Explorer** - Curious, adventurous, open, enthusiastic
5. **Creator** - Artistic, expressive, imaginative, visionary

## Creating Custom Personas

### From Description
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
By comparing activations from different conditions (safe vs unsafe, different personas), we extract direction vectors that represent specific behavioral dimensions.

### 3. Safety Orthogonalization
Persona vectors are orthogonalized against the safety axis to ensure they don't inadvertently affect safety-related behavior.

### 4. Runtime Steering
During generation, persona vectors are added to activations at target layers, shifting the model's behavior toward the desired persona.

### 5. Safety Capping
Activations are clamped to maintain minimum projection onto the safety axis, preventing drift into unsafe territory.

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  device: "cuda"
  dtype: "bfloat16"
  
extraction:
  target_layers: [14, 15, 16, 17, 18, 19, 20, 21]
  max_new_tokens: 256
  temperature: 0.7
  
steering:
  default_strength: 0.3
  safety_floor_percentile: 25
  capping_layers: [18, 19, 20, 21, 22, 23, 24, 25]
```

## Project Structure

```
persona-steering/
├── config.yaml           # Configuration
├── pyproject.toml        # Package definition
├── requirements.txt      # Dependencies
├── datasets/             # Generated datasets
│   ├── safety/          # Safe/unsafe prompts
│   └── personas/        # Extraction questions, archetypes
├── vectors/              # Extracted vectors
│   ├── safety_axis.pt
│   ├── assistant_baseline.pt
│   └── personas/        # Persona vectors
├── src/                  # Source code
│   ├── config.py        # Configuration loading
│   ├── model_wrapper.py # Model with activation hooks
│   ├── vector_math.py   # Vector operations
│   ├── activation_extractor.py
│   ├── dataset_generator.py
│   ├── persona_manager.py
│   ├── steering_engine.py
│   └── cli.py           # CLI interface
└── scripts/              # Extraction scripts
    ├── generate_safety_dataset.py
    ├── generate_persona_dataset.py
    ├── extract_assistant_axis.py
    ├── extract_safety_axis.py
    └── extract_persona.py
```

## License

MIT License
