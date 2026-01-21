#!/usr/bin/env python3
"""Command-line interface for the persona steering system."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from .config import get_config, Config
from .model_wrapper import ModelWrapper
from .persona_manager import PersonaManager
from .steering_engine import SteeringEngine
from .activation_extractor import ActivationExtractor
from .dataset_generator import DatasetGenerator, ARCHETYPES


console = Console()


def get_history_path() -> Path:
    """Get path for command history file."""
    history_dir = Path.home() / ".persona_steering"
    history_dir.mkdir(exist_ok=True)
    return history_dir / "chat_history"


@click.group()
@click.option("--config", "-c", "config_path", default=None, help="Path to config.yaml")
@click.pass_context
def cli(ctx, config_path):
    """Persona Steering System - Steer LLM behavior with extracted persona vectors."""
    ctx.ensure_object(dict)
    try:
        ctx.obj["config"] = get_config(config_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        ctx.obj["config"] = None


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize the system and generate base datasets."""
    config: Config = ctx.obj.get("config")
    if config is None:
        console.print("[red]Cannot initialize without config.yaml[/red]")
        return
    
    console.print(Panel.fit(
        "[bold blue]Persona Steering System Initialization[/bold blue]\n"
        "This will generate datasets and extract base vectors.",
        border_style="blue"
    ))
    
    # Create directories
    console.print("\n[yellow]Creating directories...[/yellow]")
    config.ensure_directories()
    console.print("  ✓ Directories created")
    
    # Initialize model
    console.print("\n[yellow]Loading model (this may take a moment)...[/yellow]")
    model = ModelWrapper(config)
    model.load()
    console.print(f"  ✓ Model loaded: {config.model_name}")
    
    generator = DatasetGenerator(model, config)
    extractor = ActivationExtractor(model, config)
    
    # Step 1: Generate safety datasets
    console.print("\n[cyan]Step 1/5: Generating safety datasets...[/cyan]")
    
    safe_path = config.safety_dir / "safe_prompts.jsonl"
    if not safe_path.exists():
        safe_prompts = generator.generate_safe_prompts(n=200)
        generator.save_jsonl(safe_prompts, safe_path)
        console.print(f"  ✓ Generated {len(safe_prompts)} safe prompts")
    else:
        console.print("  ✓ Safe prompts already exist")
    
    unsafe_path = config.safety_dir / "unsafe_prompts.jsonl"
    if not unsafe_path.exists():
        unsafe_prompts = generator.generate_unsafe_prompts(n=200)
        generator.save_jsonl(unsafe_prompts, unsafe_path)
        console.print(f"  ✓ Generated {len(unsafe_prompts)} unsafe prompts")
    else:
        console.print("  ✓ Unsafe prompts already exist")
    
    # Step 2: Generate extraction questions
    console.print("\n[cyan]Step 2/5: Generating extraction questions...[/cyan]")
    
    questions_path = config.personas_dir / "extraction_questions.jsonl"
    if not questions_path.exists():
        questions = generator.generate_extraction_questions(n=100)
        generator.save_jsonl(questions, questions_path)
        console.print(f"  ✓ Generated {len(questions)} extraction questions")
    else:
        console.print("  ✓ Extraction questions already exist")
    
    # Step 3: Generate archetype prompts
    console.print("\n[cyan]Step 3/5: Generating archetype prompts...[/cyan]")
    
    for archetype_name, archetype_info in ARCHETYPES.items():
        archetype_path = config.archetypes_dir / f"{archetype_name}.jsonl"
        if not archetype_path.exists():
            prompts = generator.generate_archetype_prompts(
                archetype=archetype_name,
                traits=archetype_info["traits"],
                n=5
            )
            for i, p in enumerate(prompts):
                p["id"] = f"{archetype_name}_prompt_{i+1}"
            generator.save_jsonl(prompts, archetype_path)
            console.print(f"  ✓ Generated {archetype_name} prompts")
        else:
            console.print(f"  ✓ {archetype_name} prompts already exist")
    
    # Step 4: Extract safety axis
    console.print("\n[cyan]Step 4/5: Extracting safety axis...[/cyan]")
    
    safety_axis_path = config.vectors_dir / "safety_axis.pt"
    if not safety_axis_path.exists():
        # This would take a long time, so we'll just note it
        console.print("  [yellow]Run 'python scripts/extract_safety_axis.py' to extract[/yellow]")
    else:
        console.print("  ✓ Safety axis already exists")
    
    # Step 5: Extract assistant baseline
    console.print("\n[cyan]Step 5/5: Extracting assistant baseline...[/cyan]")
    
    baseline_path = config.vectors_dir / "assistant_baseline.pt"
    if not baseline_path.exists():
        console.print("  [yellow]Run 'python scripts/extract_assistant_axis.py' to extract[/yellow]")
    else:
        console.print("  ✓ Assistant baseline already exists")
    
    # Cleanup
    model.unload()
    
    console.print("\n[bold green]✓ Initialization complete![/bold green]")
    console.print("\nNext steps:")
    console.print("  1. Run extraction scripts to create vectors:")
    console.print("     python scripts/extract_safety_axis.py")
    console.print("     python scripts/extract_assistant_axis.py")
    console.print("     python scripts/extract_persona.py --all")
    console.print("  2. Start chatting: persona-steer chat")


@cli.command("extract-axes")
@click.pass_context
def extract_axes(ctx):
    """Extract assistant and safety axes (shortcut for running both scripts)."""
    config: Config = ctx.obj.get("config")
    if config is None:
        return
    
    console.print("[yellow]Running extraction scripts...[/yellow]")
    console.print("This will take a while. Consider running the scripts directly for progress output.")
    
    import subprocess
    
    scripts_dir = Path(__file__).parent.parent / "scripts"
    
    # Run safety axis extraction
    console.print("\n[cyan]Extracting safety axis...[/cyan]")
    subprocess.run([sys.executable, str(scripts_dir / "extract_safety_axis.py")])
    
    # Run assistant axis extraction
    console.print("\n[cyan]Extracting assistant baseline...[/cyan]")
    subprocess.run([sys.executable, str(scripts_dir / "extract_assistant_axis.py")])


@cli.command("list")
@click.option("--loaded", is_flag=True, help="Show only loaded personas")
@click.pass_context
def list_personas(ctx, loaded):
    """List available personas."""
    config: Config = ctx.obj.get("config")
    if config is None:
        return
    
    manager = PersonaManager(config)
    
    if loaded:
        manager.print_loaded(console)
    else:
        manager.print_available(console)


@cli.command()
@click.argument("name")
@click.pass_context
def load(ctx, name):
    """Load a persona into memory."""
    config: Config = ctx.obj.get("config")
    if config is None:
        return
    
    manager = PersonaManager(config)
    
    if manager.load(name):
        console.print(f"[green]✓ Loaded persona: {name}[/green]")
    else:
        console.print(f"[red]Error: Could not load persona '{name}'[/red]")
        console.print(f"Available: {', '.join(manager.list_available())}")


@cli.command()
@click.argument("name")
@click.pass_context
def unload(ctx, name):
    """Unload a persona from memory."""
    config: Config = ctx.obj.get("config")
    if config is None:
        return
    
    manager = PersonaManager(config)
    
    if manager.unload(name):
        console.print(f"[green]✓ Unloaded persona: {name}[/green]")
    else:
        console.print(f"[yellow]Persona '{name}' was not loaded[/yellow]")


@cli.command()
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete(ctx, name, yes):
    """Delete a persona from disk."""
    config: Config = ctx.obj.get("config")
    if config is None:
        return
    
    manager = PersonaManager(config)
    
    if name not in manager.list_available():
        console.print(f"[red]Persona '{name}' not found[/red]")
        return
    
    if not yes:
        if not Confirm.ask(f"Delete persona '{name}'?"):
            console.print("[yellow]Cancelled[/yellow]")
            return
    
    if manager.delete(name):
        console.print(f"[green]✓ Deleted persona: {name}[/green]")
    else:
        console.print(f"[red]Error deleting persona[/red]")


@cli.command("clear-personas")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.option("--keep-baseline", is_flag=True, help="Keep the assistant baseline vector")
@click.option("--archetypes", is_flag=True, help="Also clear archetype prompt files")
@click.pass_context
def clear_personas(ctx, yes, keep_baseline, archetypes):
    """Clear all persona vectors and optionally the baseline and archetypes.
    
    This removes all extracted persona vectors from vectors/personas/
    and the assistant baseline vector (vectors/assistant_baseline.pt).
    
    Use --archetypes to also remove archetype prompt files from
    datasets/personas/archetypes/.
    """
    config: Config = ctx.obj.get("config")
    if config is None:
        return
    
    personas_dir = config.vectors_dir / "personas"
    baseline_path = config.vectors_dir / "assistant_baseline.pt"
    archetypes_dir = config.archetypes_dir
    
    # Count what will be deleted
    persona_files = list(personas_dir.glob("*.pt")) if personas_dir.exists() else []
    baseline_exists = baseline_path.exists() and not keep_baseline
    archetype_files = list(archetypes_dir.glob("*.jsonl")) if archetypes and archetypes_dir.exists() else []
    
    if not persona_files and not baseline_exists and not archetype_files:
        console.print("[yellow]Nothing to clear - no persona vectors, baseline, or archetypes found.[/yellow]")
        return
    
    # Show what will be deleted
    console.print("[bold]The following will be deleted:[/bold]")
    if persona_files:
        console.print(f"  • {len(persona_files)} persona vector(s) in {personas_dir}/")
        for f in persona_files:
            console.print(f"    - {f.name}")
    if baseline_exists:
        console.print(f"  • Assistant baseline: {baseline_path}")
    if archetype_files:
        console.print(f"  • {len(archetype_files)} archetype file(s) in {archetypes_dir}/")
        for f in archetype_files:
            console.print(f"    - {f.name}")
    
    if not yes:
        if not Confirm.ask("\n[bold red]Delete these files?[/bold red]"):
            console.print("[yellow]Cancelled[/yellow]")
            return
    
    # Delete persona vectors
    deleted_count = 0
    for persona_file in persona_files:
        try:
            persona_file.unlink()
            deleted_count += 1
        except Exception as e:
            console.print(f"[red]Error deleting {persona_file.name}: {e}[/red]")
    
    if deleted_count > 0:
        console.print(f"[green]✓ Deleted {deleted_count} persona vector(s)[/green]")
    
    # Delete baseline
    if baseline_exists:
        try:
            baseline_path.unlink()
            console.print("[green]✓ Deleted assistant baseline[/green]")
        except Exception as e:
            console.print(f"[red]Error deleting baseline: {e}[/red]")
    
    # Delete archetype files
    if archetype_files:
        deleted_archetypes = 0
        for archetype_file in archetype_files:
            try:
                archetype_file.unlink()
                deleted_archetypes += 1
            except Exception as e:
                console.print(f"[red]Error deleting {archetype_file.name}: {e}[/red]")
        
        if deleted_archetypes > 0:
            console.print(f"[green]✓ Deleted {deleted_archetypes} archetype file(s)[/green]")
    
    console.print("\n[cyan]To regenerate vectors, run:[/cyan]")
    console.print("  persona-steer extract-axes")
    console.print("  python scripts/extract_persona.py --all")


@cli.command()
@click.argument("name")
@click.pass_context
def info(ctx, name):
    """Show persona metadata."""
    config: Config = ctx.obj.get("config")
    if config is None:
        return
    
    manager = PersonaManager(config)
    info = manager.get_info(name)
    
    if info is None:
        console.print(f"[red]Persona '{name}' not found[/red]")
        return
    
    console.print(Panel(
        f"[bold]{info['name']}[/bold]\n\n"
        f"Source: {info['source']}\n"
        f"Description: {info['description'] or 'N/A'}\n"
        f"Traits: {', '.join(info['traits']) if info['traits'] else 'N/A'}\n"
        f"Layers: {info['layers']}\n"
        f"Has safe vector: {info['has_safe_vector']}\n"
        f"Has raw vector: {info['has_raw_vector']}",
        title="Persona Info",
        border_style="cyan"
    ))


@cli.command()
@click.argument("name")
@click.option("--description", "-d", help="Character description")
@click.option("--from-archetype", "-a", "archetype", help="Base archetype to use")
@click.option("--interactive", "-i", is_flag=True, help="Guided creation")
@click.pass_context
def create(ctx, name, description, archetype, interactive):
    """Create a new persona."""
    config: Config = ctx.obj.get("config")
    if config is None:
        return
    
    if interactive:
        description = Prompt.ask("Character description")
    
    if archetype:
        # Extract from existing archetype
        import subprocess
        scripts_dir = Path(__file__).parent.parent / "scripts"
        subprocess.run([
            sys.executable, 
            str(scripts_dir / "extract_persona.py"),
            "--from-archetype", archetype
        ])
    elif description:
        # Generate from description
        import subprocess
        scripts_dir = Path(__file__).parent.parent / "scripts"
        subprocess.run([
            sys.executable,
            str(scripts_dir / "extract_persona.py"),
            "--from-description", description,
            "--name", name
        ])
    else:
        console.print("[red]Either --description, --from-archetype, or --interactive required[/red]")


@cli.command()
@click.argument("prompt")
@click.option("--persona", "-p", help="Persona to use")
@click.option("--strength", "-s", type=float, default=0.3, help="Steering strength")
@click.option("--no-safety", is_flag=True, help="Disable safety capping")
@click.pass_context
def generate(ctx, prompt, persona, strength, no_safety):
    """Generate a single response."""
    config: Config = ctx.obj.get("config")
    if config is None:
        return
    
    console.print("[yellow]Loading model...[/yellow]")
    
    model = ModelWrapper(config)
    manager = PersonaManager(config)
    engine = SteeringEngine(model, manager, config)
    
    engine.ensure_loaded()
    
    response = engine.generate(
        user_prompt=prompt,
        persona=persona,
        persona_strength=strength,
        enable_safety_capping=not no_safety,
    )
    
    console.print(Panel(Markdown(response), title="Response", border_style="green"))
    
    engine.unload()


@cli.command()
@click.option("--persona", "-p", help="Initial persona to use")
@click.option("--strength", "-s", type=float, default=0.3, help="Initial steering strength")
@click.option("--blend", "-b", help="Blend spec like sage:0.5,trickster:0.3")
@click.option("--no-safety", is_flag=True, help="Disable safety capping")
@click.pass_context
def chat(ctx, persona, strength, blend, no_safety):
    """Start interactive chat session."""
    config: Config = ctx.obj.get("config")
    if config is None:
        return
    
    console.print(Panel.fit(
        "[bold cyan]Persona Steering Chat[/bold cyan]\n"
        "Type your message or use commands:\n"
        "  /persona <name>  - switch persona\n"
        "  /strength <num>  - adjust strength\n"
        "  /blend <spec>    - set blend (sage:0.5,trickster:0.3)\n"
        "  /none            - disable persona\n"
        "  /safety on|off   - toggle safety capping\n"
        "  /list            - show personas\n"
        "  /load <name>     - load persona\n"
        "  /status          - show current settings\n"
        "  /help            - show this help\n"
        "  /quit            - exit",
        border_style="cyan"
    ))
    
    console.print("\n[yellow]Loading model...[/yellow]")
    
    model = ModelWrapper(config)
    manager = PersonaManager(config)
    engine = SteeringEngine(model, manager, config)
    
    engine.ensure_loaded()
    engine.default_strength = strength
    engine.safety_enabled = not no_safety
    
    # Parse initial blend if provided
    current_blend = None
    if blend:
        current_blend = parse_blend(blend)
    
    current_persona = persona
    
    # Create prompt session with history
    session = PromptSession(history=FileHistory(str(get_history_path())))
    
    console.print("[green]Ready! Type your message.[/green]\n")
    
    while True:
        # Build prompt string
        if current_blend:
            blend_str = "+".join(f"{k}:{v}" for k, v in current_blend.items())
            prompt_str = f"[blend:{blend_str}] > "
        elif current_persona:
            prompt_str = f"[{current_persona}:{engine.default_strength}] > "
        else:
            prompt_str = "[no persona] > "
        
        try:
            user_input = session.prompt(prompt_str)
        except (EOFError, KeyboardInterrupt):
            break
        
        user_input = user_input.strip()
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            cmd_result = handle_chat_command(
                user_input, 
                engine, 
                manager, 
                current_persona, 
                current_blend
            )
            
            if cmd_result == "quit":
                break
            elif cmd_result is not None:
                if isinstance(cmd_result, dict):
                    current_persona = cmd_result.get("persona", current_persona)
                    current_blend = cmd_result.get("blend", current_blend)
            continue
        
        # Generate response
        try:
            if current_blend:
                response = engine.generate_with_composite(
                    user_prompt=user_input,
                    persona_blend=current_blend,
                )
            else:
                response = engine.generate(
                    user_prompt=user_input,
                    persona=current_persona,
                )
            
            console.print()
            console.print(Panel(Markdown(response), border_style="green"))
            console.print()
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    console.print("\n[yellow]Cleaning up...[/yellow]")
    engine.unload()
    console.print("[green]Goodbye![/green]")


def parse_blend(blend_str: str) -> dict[str, float]:
    """Parse a blend specification string."""
    result = {}
    for part in blend_str.split(","):
        if ":" in part:
            name, weight = part.split(":", 1)
            result[name.strip()] = float(weight.strip())
    return result


def handle_chat_command(
    cmd: str, 
    engine: SteeringEngine, 
    manager: PersonaManager,
    current_persona: str | None,
    current_blend: dict | None,
) -> str | dict | None:
    """Handle a chat command. Returns 'quit' to exit, dict to update state, or None."""
    parts = cmd[1:].split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None
    
    if command == "quit" or command == "exit" or command == "q":
        return "quit"
    
    elif command == "help" or command == "?":
        console.print(Panel(
            "/persona <name>  - switch persona\n"
            "/strength <num>  - adjust strength (0.0-2.0)\n"
            "/blend <spec>    - set blend (sage:0.5,trickster:0.3)\n"
            "/none            - disable persona steering\n"
            "/safety on|off   - toggle safety capping\n"
            "/list            - show available personas\n"
            "/load <name>     - load persona into memory\n"
            "/status          - show current settings\n"
            "/quit            - exit chat",
            title="Commands",
            border_style="cyan"
        ))
    
    elif command == "persona" or command == "p":
        if not arg:
            console.print("[yellow]Usage: /persona <name>[/yellow]")
        elif arg in manager.list_available():
            manager.load(arg)
            console.print(f"[green]Switched to persona: {arg}[/green]")
            return {"persona": arg, "blend": None}
        else:
            console.print(f"[red]Persona '{arg}' not found[/red]")
            console.print(f"Available: {', '.join(manager.list_available())}")
    
    elif command == "strength" or command == "s":
        if not arg:
            console.print(f"[cyan]Current strength: {engine.default_strength}[/cyan]")
        else:
            try:
                engine.default_strength = float(arg)
                console.print(f"[green]Strength set to: {engine.default_strength}[/green]")
            except ValueError:
                console.print("[red]Invalid strength value[/red]")
    
    elif command == "blend" or command == "b":
        if not arg:
            if current_blend:
                console.print(f"[cyan]Current blend: {current_blend}[/cyan]")
            else:
                console.print("[cyan]No blend active[/cyan]")
        else:
            blend = parse_blend(arg)
            if blend:
                # Load all personas in blend
                for name in blend.keys():
                    manager.load(name)
                console.print(f"[green]Blend set: {blend}[/green]")
                return {"persona": None, "blend": blend}
            else:
                console.print("[red]Invalid blend specification[/red]")
    
    elif command == "none":
        console.print("[green]Persona steering disabled[/green]")
        return {"persona": None, "blend": None}
    
    elif command == "safety":
        if not arg:
            status = "on" if engine.safety_enabled else "off"
            console.print(f"[cyan]Safety capping: {status}[/cyan]")
        elif arg.lower() in ("on", "true", "1", "yes"):
            engine.safety_enabled = True
            console.print("[green]Safety capping enabled[/green]")
        elif arg.lower() in ("off", "false", "0", "no"):
            engine.safety_enabled = False
            console.print("[yellow]Safety capping disabled[/yellow]")
        else:
            console.print("[red]Usage: /safety on|off[/red]")
    
    elif command == "list" or command == "ls":
        manager.print_available(console)
    
    elif command == "load":
        if not arg:
            console.print("[yellow]Usage: /load <name>[/yellow]")
        elif manager.load(arg):
            console.print(f"[green]Loaded: {arg}[/green]")
        else:
            console.print(f"[red]Could not load '{arg}'[/red]")
    
    elif command == "status":
        status = engine.get_status()
        console.print(Panel(
            f"Model loaded: {status['model_loaded']}\n"
            f"Default strength: {status['default_strength']}\n"
            f"Safety enabled: {status['safety_enabled']}\n"
            f"Safety floor: {status['safety_floor']}\n"
            f"Current persona: {current_persona or 'None'}\n"
            f"Current blend: {current_blend or 'None'}\n"
            f"Loaded personas: {', '.join(status['loaded_personas']) or 'None'}",
            title="Status",
            border_style="cyan"
        ))
    
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("Type /help for available commands")
    
    return None


@cli.group()
def config():
    """Manage configuration."""
    pass


@config.command("show")
@click.pass_context
def config_show(ctx):
    """Show current configuration."""
    cfg: Config = ctx.obj.get("config")
    if cfg is None:
        return
    
    console.print(Panel(
        f"Model: {cfg.model_name}\n"
        f"Device: {cfg.device}\n"
        f"Dtype: {cfg.dtype}\n"
        f"Target layers: {cfg.target_layers}\n"
        f"Capping layers: {cfg.capping_layers}\n"
        f"Default strength: {cfg.default_strength}\n"
        f"Safety floor percentile: {cfg.safety_floor_percentile}\n"
        f"Max new tokens: {cfg.max_new_tokens}\n"
        f"Temperature: {cfg.temperature}",
        title="Configuration",
        border_style="cyan"
    ))


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx, key, value):
    """Set a configuration value."""
    cfg: Config = ctx.obj.get("config")
    if cfg is None:
        return
    
    # Try to parse value as appropriate type
    try:
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif "." in value:
            value = float(value)
        else:
            value = int(value)
    except ValueError:
        pass  # Keep as string
    
    cfg.set(key, value)
    console.print(f"[green]Set {key} = {value}[/green]")


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
