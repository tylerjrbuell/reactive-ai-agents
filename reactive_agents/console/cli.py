#!/usr/bin/env python3
"""
Reactive Agents CLI

A modern, reactive AI agent framework for intelligent task execution.
"""

import asyncio
import sys
import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

# Import the main framework components
from reactive_agents.app.builders.agent import ReactiveAgentBuilder
from reactive_agents.config.settings import get_settings
from reactive_agents.plugins.plugin_manager import get_plugin_generator
from reactive_agents.core.types.reasoning_types import ReasoningStrategies
from reactive_agents.config.natural_language_config import create_agent_from_nl
from reactive_agents.providers.llm.factory import ModelProviderFactory

# Initialize rich console
console = Console()

# ASCII Art Banner
BANNER = """
[bold bright_blue]
╔════════════════════════════════════════════════════════╗
║                                                        ║
║    [bold white]██████╗ ███████╗███████╗██╗     ███████╗██╗  ██╗[/bold white]    ║
║    [bold white]██╔══██╗██╔════╝██╔════╝██║     ██╔════╝╚██╗██╔╝[/bold white]    ║
║    [bold white]██████╔╝█████╗  █████╗  ██║     █████╗   ╚███╔╝[/bold white]     ║
║    [bold white]██╔══██╗██╔══╝  ██╔══╝  ██║     ██╔══╝   ██╔██╗[/bold white]     ║
║    [bold white]██║  ██║███████╗██║     ███████╗███████╗██╔╝ ██╗[/bold white]    ║
║    [bold white]╚═╝  ╚═╝╚══════╝╚═╝     ╚══════╝╚══════╝╚═╝  ╚═╝[/bold white]    ║
║                                                        ║
║     [bold bright_blue]Reflex[/bold bright_blue] - [italic bold yellow]Reactive Agents Taking Action[/italic bold yellow]             ║
╚════════════════════════════════════════════════════════╝
[/bold bright_blue]
"""

VERSION = "0.1.0a6"


def print_banner():
    """Print the ASCII art banner."""
    console.print(BANNER, markup=True)
    console.print(f"[dim]Version: {VERSION}[/dim]\n")


def print_info():
    """Print framework information."""
    info_table = Table(
        title="[bold cyan]Framework Information[/bold cyan]", show_header=False
    )
    info_table.add_column("Property", style="cyan", no_wrap=True)
    info_table.add_column("Value", style="white")

    info_table.add_row("Framework", "Reflex")
    info_table.add_row("Version", VERSION)
    info_table.add_row(
        "Description",
        "A modern, reactive AI agent framework for intelligent task execution",
    )
    info_table.add_row("Tagline", "Reactive Agents Taking Action")
    info_table.add_row("Author", "Tyler Buell")
    info_table.add_row("License", "MIT")
    info_table.add_row("Repository", "https://github.com/tylerbuell/reactive-ai-agent")

    console.print(info_table)
    console.print()


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version information")
@click.option("--info", is_flag=True, help="Show framework information")
@click.pass_context
def main(ctx, version: bool, info: bool):
    """
    🚀 Reflex - Reactive Agents Taking Action

    A modern, reactive AI agent framework for intelligent task execution.
    Built for developers who want powerful, flexible AI agents.
    """
    if ctx.invoked_subcommand is not None:
        return

    if version:
        console.print(f"[bold cyan]Reflex v{VERSION}[/bold cyan]")
        return

    if info:
        print_banner()
        print_info()
        return

    # Default behavior: show help
    print_banner()
    console.print("[bold yellow]Welcome to Reflex![/bold yellow]")
    console.print("Use [cyan]--help[/cyan] to see available commands.\n")

    # Show quick start example
    console.print("[bold]Quick Start:[/bold]")
    console.print("  [cyan]reflex make agent[/cyan] - Create and run an agent")
    console.print(
        "  [cyan]reflex make agent --task 'Your task here'[/cyan] - Run with specific task"
    )
    console.print(
        "  [cyan]reflex make strategy my_strategy[/cyan] - Create a strategy plugin"
    )
    console.print("  [cyan]reflex make tool my_tool[/cyan] - Create a tool plugin")
    console.print(
        "  [cyan]reflex make provider my_provider[/cyan] - Create a provider plugin"
    )
    console.print("  [cyan]reflex info[/cyan] - Show framework information")
    console.print()


@main.group()
def make():
    """
    🛠️ Create and generate framework components

    Build agents, tools, workflows, and other reactive components.
    """
    pass


@main.group()
def agents():
    """
    🤖 Agent management commands

    Run, test, monitor, and manage agents and configurations.
    """
    pass


@main.group()
def project():
    """
    ⚙️ Project configuration commands

    Initialize and manage project settings.
    """
    pass


@main.group()
def plugins():
    """
    🔌 Plugin management commands

    Load, unload, and manage framework plugins.
    """
    pass


@make.command()
@click.option("--task", "-t", help="Task to execute")
@click.option(
    "--model",
    "-m",
    default="ollama:cogito:14b",
    help="Model to use (default: ollama:cogito:14b)",
)
@click.option(
    "--tools", "-T", multiple=True, help="Tools to include (can specify multiple)"
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Enable interactive mode with confirmations",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--strategy",
    "-s",
    default="reactive",
    type=click.Choice(
        ["reflect_decide_act", "plan_execute_reflect", "reactive", "adaptive"]
    ),
    help="Reasoning strategy to use",
)
def agent(
    task: Optional[str],
    model: str,
    tools: tuple,
    interactive: bool,
    verbose: bool,
    strategy: str,
):
    """
    🚀 Create and run a reactive agent

    Execute tasks using intelligent AI agents with dynamic reasoning strategies.
    """
    print_banner()

    # Get task from user if not provided
    if not task:
        task = Prompt.ask("[bold cyan]What would you like the agent to do?[/bold cyan]")
        if not task.strip():
            console.print("[red]No task provided. Exiting.[/red]")
            return

    # Set default tools if none provided
    if not tools:
        tools = ("brave-search", "time")
        console.print(f"[dim]Using default tools: {', '.join(tools)}[/dim]")

    # Show execution plan
    console.print("[bold]Execution Plan:[/bold]")
    plan_table = Table(show_header=False)
    plan_table.add_column("Setting", style="cyan", no_wrap=True)
    plan_table.add_column("Value", style="white")

    plan_table.add_row("Task", task[:100] + "..." if len(task) > 100 else task)
    plan_table.add_row("Model", model)
    plan_table.add_row("Tools", ", ".join(tools))
    plan_table.add_row("Strategy", strategy)
    plan_table.add_row("Interactive", "Yes" if interactive else "No")

    console.print(plan_table)
    console.print()

    # Confirm execution
    if not Confirm.ask("[bold yellow]Start agent execution?[/bold yellow]"):
        console.print("[dim]Execution cancelled.[/dim]")
        return

    # Run the agent
    asyncio.run(_run_agent(task, model, list(tools), interactive, verbose, strategy))


@make.command()
@click.option("--name", "-n", required=True, help="Agent name")
@click.option("--description", "-d", help="Natural language description of the agent")
@click.option("--model", "-m", default="ollama:cogito:14b", help="Model to use")
@click.option(
    "--tools", "-t", multiple=True, default=["brave-search", "time"], help="MCP tools"
)
@click.option(
    "--strategy",
    "-s",
    default="reactive",
    type=click.Choice(
        ["reflect_decide_act", "plan_execute_reflect", "reactive", "adaptive"]
    ),
    help="Reasoning strategy",
)
@click.option("--vector-memory", is_flag=True, help="Enable vector memory")
@click.option("--output", "-o", help="Output configuration file path")
def config(
    name: str,
    description: Optional[str],
    model: str,
    tools: tuple,
    strategy: str,
    vector_memory: bool,
    output: Optional[str],
):
    """
    📋 Create a reusable agent configuration

    Generate agent configuration files that can be reused with 'reflex agent run'.
    """
    print_banner()

    try:
        if description:
            # Use natural language configuration
            console.print("[dim]Using natural language configuration...[/dim]")
            model_provider = ModelProviderFactory.get_model_provider(
                model, context=None
            )
            config = asyncio.run(
                create_agent_from_nl(description, model_provider=model_provider)
            )

            # Override with CLI arguments
            config.agent_name = name
            config.provider_model_name = model
        else:
            # Use builder pattern
            console.print("[dim]Using builder configuration...[/dim]")
            builder = (
                ReactiveAgentBuilder()
                .with_name(name)
                .with_model(model)
                .with_reasoning_strategy(ReasoningStrategies(strategy))
                .with_mcp_tools(list(tools))
            )

            if vector_memory:
                builder.with_vector_memory()

            # Create config from builder
            config = builder._config

        # Save configuration
        config_path = output or f"{name.lower().replace(' ', '_')}_config.json"

        config_data = {
            "agent_name": getattr(config, "agent_name", name),
            "provider_model_name": getattr(config, "provider_model_name", model),
            "reasoning_strategy": getattr(config, "reasoning_strategy", strategy),
            "mcp_server_filter": list(tools),
            "vector_memory_enabled": vector_memory,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "cli_generated": True,
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        console.print(f"[bold green]✅ Agent configuration saved![/bold green]")
        console.print(f"[dim]Path: {config_path}[/dim]")
        console.print("\n[bold]Configuration Details:[/bold]")

        details_table = Table(show_header=False)
        details_table.add_column("Setting", style="cyan", no_wrap=True)
        details_table.add_column("Value", style="white")

        details_table.add_row("Name", name)
        details_table.add_row("Model", model)
        details_table.add_row("Strategy", strategy)
        details_table.add_row("Tools", ", ".join(tools))
        details_table.add_row("Vector Memory", "Yes" if vector_memory else "No")

        console.print(details_table)

        console.print("\n[bold]📝 Next steps:[/bold]")
        console.print(
            f"  [cyan]reflex agent run --config {config_path} --task 'Your task'[/cyan]"
        )
        console.print(f"  [cyan]reflex agent test --config {config_path}[/cyan]")

    except Exception as e:
        console.print(f"[bold red]❌ Error creating config:[/bold red] {str(e)}")
        sys.exit(1)


@make.command()
@click.option("--description", "-d", help="Description of the strategy")
@click.option("--author", "-a", default="", help="Author name")
@click.argument("name")
def strategy(name: str, description: Optional[str], author: str):
    """
    🧠 Create a new reasoning strategy plugin

    Generate a new reasoning strategy plugin for custom agent behavior.
    """
    print_banner()

    try:
        generator = get_plugin_generator()
        plugin_path = generator.generate_strategy_plugin(
            name, description or f"Custom {name} strategy", author
        )

        console.print(f"[bold green]✅ Strategy plugin '{name}' created![/bold green]")
        console.print(f"[dim]Path: {plugin_path}[/dim]")
        console.print("\n[bold]📝 Next steps:[/bold]")
        console.print(f"  1. Add {name.upper()} to ReasoningStrategies enum")
        console.print(f"  2. Implement strategy logic in execute_iteration method")
        console.print(
            f"  3. Test with: [cyan]reflex make agent --strategy {name}[/cyan]"
        )

    except Exception as e:
        console.print(
            f"[bold red]❌ Error creating strategy plugin:[/bold red] {str(e)}"
        )
        sys.exit(1)


@make.command()
@click.option("--description", "-d", help="Description of the tool")
@click.option("--author", "-a", default="", help="Author name")
@click.argument("name")
def tool(name: str, description: Optional[str], author: str):
    """
    🔧 Create a new tool plugin

    Generate a new tool plugin for use with reactive agents.
    """
    print_banner()

    try:
        generator = get_plugin_generator()
        plugin_path = generator.generate_tool_plugin(
            name, description or f"Custom {name} tool", author
        )

        console.print(f"[bold green]✅ Tool plugin '{name}' created![/bold green]")
        console.print(f"[dim]Path: {plugin_path}[/dim]")
        console.print("\n[bold]📝 Next steps:[/bold]")
        console.print(f"  1. Implement tool logic in execute method")
        console.print(f"  2. Test with: [cyan]reflex make agent --tools {name}[/cyan]")

    except Exception as e:
        console.print(f"[bold red]❌ Error creating tool plugin:[/bold red] {str(e)}")
        sys.exit(1)


@make.command()
@click.option("--description", "-d", help="Description of the provider")
@click.option("--author", "-a", default="", help="Author name")
@click.argument("name")
def provider(name: str, description: Optional[str], author: str):
    """
    🌐 Create a new provider plugin

    Generate a new model provider plugin for custom LLM integrations.
    """
    print_banner()

    try:
        generator = get_plugin_generator()
        plugin_path = generator.generate_provider_plugin(
            name, description or f"Custom {name} provider", author
        )

        console.print(f"[bold green]✅ Provider plugin '{name}' created![/bold green]")
        console.print(f"[dim]Path: {plugin_path}[/dim]")
        console.print("\n[bold]📝 Next steps:[/bold]")
        console.print(f"  1. Implement provider logic in generate_response method")
        console.print(f"  2. Test with: [cyan]reflex make agent --model {name}[/cyan]")

    except Exception as e:
        console.print(
            f"[bold red]❌ Error creating provider plugin:[/bold red] {str(e)}"
        )
        sys.exit(1)


@make.command()
@click.option("--name", "-n", help="Workflow name")
@click.option("--steps", "-s", multiple=True, help="Workflow steps")
def workflow(name: Optional[str], steps: tuple):
    """
    🔄 Create a new workflow

    Generate a new workflow for orchestrating agent tasks.
    """
    print_banner()

    # Get workflow details from user if not provided
    if not name:
        name = Prompt.ask("[bold cyan]What should the workflow be called?[/bold cyan]")
        if not name.strip():
            console.print("[red]No workflow name provided. Exiting.[/red]")
            return

    console.print(f"[bold green]✅ Workflow '{name}' creation planned![/bold green]")
    if steps:
        console.print(f"[dim]Steps: {', '.join(steps)}[/dim]")
    console.print("[yellow]⚠️  Workflow generation not yet implemented[/yellow]")


async def _run_agent(
    task: str, model: str, tools: list, interactive: bool, verbose: bool, strategy: str
):
    """Run the agent with the specified configuration."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_id = progress.add_task("Initializing agent...", total=None)

            # Build the agent
            progress.update(task_id, description="Building agent...")
            # Convert strategy string to enum
            strategy_enum = ReasoningStrategies(strategy)

            agent = (
                await ReactiveAgentBuilder()
                .with_model(model)
                .with_mcp_tools(tools)
                .with_reasoning_strategy(strategy_enum)
                .build()
            )

            # Set up confirmation callback if interactive
            if interactive:

                async def confirmation_callback(
                    action_description: str, details: dict
                ) -> bool:
                    console.print(
                        f"\n[bold cyan]Tool:[/bold cyan] {details.get('tool', 'unknown')}"
                    )
                    console.print(f"[dim]Action:[/dim] {action_description}")
                    return Confirm.ask(
                        "[bold yellow]Proceed?[/bold yellow]", default=True
                    )

                agent.context.confirmation_callback = confirmation_callback

            progress.update(task_id, description="Executing task...")

            # Execute the task
            result = await agent.run(initial_task=task)

            progress.update(task_id, description="Finalizing...")

        # Display results
        console.print("\n[bold green]✅ Task completed![/bold green]\n")

        # Show final answer
        if result.get("final_answer"):
            console.print("[bold]Final Answer:[/bold]")
            console.print(
                Panel(result["final_answer"], title="[bold cyan]Result[/bold cyan]")
            )

        # Show metrics if available
        if result.get("metrics"):
            metrics = result["metrics"]
            console.print("\n[bold]Execution Metrics:[/bold]")
            metrics_table = Table(show_header=False)
            metrics_table.add_column("Metric", style="cyan", no_wrap=True)
            metrics_table.add_column("Value", style="white")

            if "total_time" in metrics:
                metrics_table.add_row("Total Time", f"{metrics['total_time']:.2f}s")
            if "iterations" in metrics:
                metrics_table.add_row("Iterations", str(metrics["iterations"]))
            if "tool_calls" in metrics:
                metrics_table.add_row("Tool Calls", str(metrics["tool_calls"]))
            if "status" in metrics:
                metrics_table.add_row("Status", metrics["status"])

            console.print(metrics_table)

        # Clean up
        await agent.close()

    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@main.command()
def info():
    """
    ℹ️ Show framework information

    Display information about the reactive agents framework.
    """
    print_banner()
    print_info()


@main.command()
@click.option("--category", "-c", help="Test category to run")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--coverage", is_flag=True, help="Run with coverage reporting")
def test(category: Optional[str], verbose: bool, coverage: bool):
    """
    🧪 Run tests

    Execute the test suite for the reactive agents framework.
    """
    print_banner()

    # Build test command
    cmd = ["python", "-m", "pytest"]

    if category:
        if category == "all":
            pass  # Run all tests
        elif category == "unit":
            cmd.extend(["reactive_agents/tests/unit/"])
        elif category == "integration":
            cmd.extend(["reactive_agents/tests/integration/"])
        else:
            cmd.extend([f"reactive_agents/tests/unit/{category}/"])

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=reactive_agents", "--cov-report=term-missing"])

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]\n")

    # Run tests
    import subprocess

    try:
        result = subprocess.run(cmd, check=True)
        console.print("[bold green]✅ Tests passed![/bold green]")
    except subprocess.CalledProcessError:
        console.print("[bold red]❌ Tests failed![/bold red]")
        sys.exit(1)


@agents.command()
@click.option("--config", "-c", help="Agent configuration file")
@click.option("--task", "-t", required=True, help="Task for the agent to perform")
@click.option(
    "--interactive", "-i", is_flag=True, help="Interactive mode with confirmations"
)
def run(config: Optional[str], task: str, interactive: bool):
    """
    🚀 Run an agent with a task

    Execute tasks using a saved agent configuration or quick setup.
    """
    print_banner()

    try:
        if config:
            # Load configuration file
            with open(config, "r") as f:
                config_data = json.load(f)

            console.print(f"[dim]Loading configuration from: {config}[/dim]")

            # Create agent from config
            strategy_enum = ReasoningStrategies(
                config_data.get("reasoning_strategy", "reflect_decide_act")
            )

            builder = (
                ReactiveAgentBuilder()
                .with_name(config_data.get("agent_name", "Configured Agent"))
                .with_model(config_data.get("provider_model_name", "ollama:cogito:14b"))
                .with_reasoning_strategy(strategy_enum)
                .with_mcp_tools(
                    config_data.get("mcp_server_filter", ["brave-search", "time"])
                )
            )

            if config_data.get("vector_memory_enabled"):
                builder.with_vector_memory()

        else:
            # Quick setup without config
            console.print("[dim]Using quick agent setup...[/dim]")
            builder = (
                ReactiveAgentBuilder()
                .with_name("Quick Agent")
                .with_model("ollama:cogito:14b")
                .with_reasoning_strategy(ReasoningStrategies.REFLECT_DECIDE_ACT)
                .with_mcp_tools(["brave-search", "time"])
            )

        # Run the agent
        asyncio.run(_run_config_agent(builder, task, interactive))

    except FileNotFoundError:
        console.print(f"[bold red]❌ Configuration file not found:[/bold red] {config}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]❌ Error running agent:[/bold red] {str(e)}")
        sys.exit(1)


@agents.command()
@click.option("--detail", "-d", is_flag=True, help="Show detailed information")
def list(detail: bool):
    """
    📋 List configured agents

    Show all saved agent configurations.
    """
    print_banner()

    try:
        configs_dir = Path("configs")
        if not configs_dir.exists():
            console.print("[yellow]⚠️  No configs directory found[/yellow]")
            console.print("\n[dim]Create an agent config with:[/dim]")
            console.print("  [cyan]reflex make config --name 'Agent Name'[/cyan]")
            return

        config_files = list(configs_dir.glob("*_config.json")) + list(
            Path(".").glob("*_config.json")
        )

        if not config_files:
            console.print("[yellow]⚠️  No agent configurations found[/yellow]")
            console.print("\n[dim]Create an agent config with:[/dim]")
            console.print("  [cyan]reflex make config --name 'Agent Name'[/cyan]")
            return

        console.print("[bold]📋 Agent Configurations:[/bold]\n")

        for config_file in config_files:
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)

                name = config.get("agent_name", "Unknown")
                model = config.get("provider_model_name", "Unknown")
                strategy = config.get("reasoning_strategy", "Unknown")

                console.print(f"[bold green]🤖 {name}[/bold green]")
                if detail:
                    console.print(f"   [dim]Model:[/dim] {model}")
                    console.print(f"   [dim]Strategy:[/dim] {strategy}")
                    console.print(f"   [dim]Config:[/dim] {config_file}")
                    console.print()

            except Exception as e:
                console.print(f"[yellow]⚠️  Could not read {config_file}: {e}[/yellow]")

    except Exception as e:
        console.print(f"[bold red]❌ Error listing agents:[/bold red] {str(e)}")


@agents.command()
@click.option("--config", "-c", required=True, help="Agent configuration file")
@click.option("--task", "-t", default="What is the current time?", help="Test task")
def test_config(config: str, task: str):
    """
    🧪 Test agent configuration

    Run a simple test task to verify agent configuration works.
    """
    print_banner()

    try:
        with open(config, "r") as f:
            config_data = json.load(f)

        console.print(f"[dim]Testing configuration: {config}[/dim]")

        # Test agent creation
        strategy_enum = ReasoningStrategies(
            config_data.get("reasoning_strategy", "reflect_decide_act")
        )

        builder = (
            ReactiveAgentBuilder()
            .with_name(config_data.get("agent_name", "Test Agent"))
            .with_model(config_data.get("provider_model_name", "ollama:cogito:14b"))
            .with_reasoning_strategy(strategy_enum)
            .with_mcp_tools(
                config_data.get("mcp_server_filter", ["brave-search", "time"])
            )
        )

        # Run simple test
        asyncio.run(_run_config_agent(builder, task, False))

        console.print("[bold green]✅ Agent configuration test passed![/bold green]")

    except Exception as e:
        console.print(f"[bold red]❌ Agent test failed:[/bold red] {str(e)}")
        sys.exit(1)


@agents.command()
@click.option("--agent", "-a", help="Agent name to monitor")
@click.option("--events", is_flag=True, help="Show real-time events")
@click.option("--stats", is_flag=True, help="Show performance statistics")
def monitor(agent: Optional[str], events: bool, stats: bool):
    """
    📊 Monitor agent execution

    Real-time monitoring of agent performance and events.
    """
    print_banner()

    console.print("[yellow]⚠️  Monitoring functionality coming soon![/yellow]")
    console.print("\n[dim]This will show:[/dim]")
    console.print("  • Real-time event stream")
    console.print("  • Performance metrics")
    console.print("  • Memory usage")
    console.print("  • Tool execution stats")


@project.command()
@click.option("--name", "-n", required=True, help="Project name")
@click.option("--description", "-d", help="Project description")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing config")
def init(name: str, description: Optional[str], force: bool):
    """
    🚀 Initialize project configuration

    Set up a new Reactive Agents project with default configuration.
    """
    print_banner()

    try:
        config_path = "reactive_config.json"

        if os.path.exists(config_path) and not force:
            console.print(
                f"[bold red]❌ Configuration already exists:[/bold red] {config_path}"
            )
            console.print("[dim]Use --force to overwrite[/dim]")
            return

        config = {
            "project_name": name,
            "description": description or f"Reactive Agents project: {name}",
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "agents": {},
            "default_model": "ollama:cogito:14b",
            "default_tools": ["brave-search", "time"],
            "vector_memory": {
                "enabled": True,
                "persist_directory": "vector_memory",
                "embedding_model": "all-MiniLM-L6-v2",
            },
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Create directories
        os.makedirs("agents", exist_ok=True)
        os.makedirs("configs", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        console.print(f"[bold green]✅ Project initialized![/bold green]")
        console.print(f"[dim]Configuration: {config_path}[/dim]")
        console.print("[dim]Created directories: agents/, configs/, logs/[/dim]")

        console.print("\n[bold]📝 Next steps:[/bold]")
        console.print(
            "  [cyan]reflex make config --name 'My Agent' --description 'Agent description'[/cyan]"
        )

    except Exception as e:
        console.print(f"[bold red]❌ Failed to initialize project:[/bold red] {str(e)}")
        sys.exit(1)


@project.command()
@click.option("--backup", "-b", is_flag=True, help="Create backup before migration")
@click.option("--force", "-f", is_flag=True, help="Force migration even if risky")
def migrate_memory(backup: bool, force: bool):
    """
    🔄 Migrate vector memory to new format

    Update existing vector memory databases to the latest format.
    """
    print_banner()

    console.print("[yellow]⚠️  Memory migration functionality coming soon![/yellow]")
    console.print("\n[dim]This will:[/dim]")
    console.print("  • Detect existing vector memory databases")
    console.print("  • Create backups (if requested)")
    console.print("  • Migrate to new format")
    console.print("  • Verify migration integrity")
    console.print("  • Update configuration files")

    if backup:
        console.print("\n[dim]Backup will be created at: ./vector_memory_backup/[/dim]")

    if force:
        console.print(
            "\n[yellow]⚠️  Force mode enabled - will overwrite existing data[/yellow]"
        )


async def _run_config_agent(
    builder: ReactiveAgentBuilder, task: str, interactive: bool
):
    """Run agent from configuration with progress tracking."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_id = progress.add_task("Building agent...", total=None)

            # Build the agent
            agent = await builder.build()

            # Set up confirmation callback if interactive
            if interactive:

                async def confirmation_callback(
                    action_description: str, details: dict
                ) -> bool:
                    console.print(
                        f"\n[bold cyan]Tool:[/bold cyan] {details.get('tool', 'unknown')}"
                    )
                    console.print(f"[dim]Action:[/dim] {action_description}")
                    return Confirm.ask(
                        "[bold yellow]Proceed?[/bold yellow]", default=True
                    )

                agent.context.confirmation_callback = confirmation_callback

            progress.update(task_id, description="Executing task...")

            # Execute the task
            result = await agent.run(initial_task=task)

            progress.update(task_id, description="Finalizing...")

        # Display results
        console.print("\n[bold green]✅ Task completed![/bold green]\n")

        # Show final answer
        if result.get("final_answer"):
            console.print("[bold]Final Answer:[/bold]")
            console.print(
                Panel(result["final_answer"], title="[bold cyan]Result[/bold cyan]")
            )

        # Show metrics if available
        if result.get("metrics"):
            metrics = result["metrics"]
            console.print("\n[bold]Execution Metrics:[/bold]")
            metrics_table = Table(show_header=False)
            metrics_table.add_column("Metric", style="cyan", no_wrap=True)
            metrics_table.add_column("Value", style="white")

            if "total_time" in metrics:
                metrics_table.add_row("Total Time", f"{metrics['total_time']:.2f}s")
            if "iterations" in metrics:
                metrics_table.add_row("Iterations", str(metrics["iterations"]))
            if "tool_calls" in metrics:
                metrics_table.add_row("Tool Calls", str(metrics["tool_calls"]))
            if "status" in metrics:
                metrics_table.add_row("Status", metrics["status"])

            console.print(metrics_table)

        # Clean up
        await agent.close()

    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")
        raise


@main.command()
def version():
    """
    📋 Show version information

    Display the current version of the reactive agents framework.
    """
    console.print(f"[bold cyan]Reflex v{VERSION}[/bold cyan]")


@plugins.command()
@click.option("--all", "show_all", is_flag=True, help="Show all available plugins")
@click.option(
    "--type", "plugin_type", help="Filter by plugin type (strategy, tool, provider)"
)
def status(show_all, plugin_type):
    """Show plugin system status and loaded plugins"""
    try:
        from reactive_agents.plugins.plugin_manager import (
            get_plugin_manager,
            PluginType,
        )

        plugin_manager = get_plugin_manager()

        # Show loaded plugins
        loaded_plugins = plugin_manager.list_loaded_plugins()

        if loaded_plugins:
            console.print("\n[bold green]Loaded Plugins:[/bold green]")
            for plugin in loaded_plugins:
                console.print(
                    f"  • {plugin['name']} v{plugin['version']} ({plugin['type']})"
                )
                console.print(f"    {plugin['description']}")
        else:
            console.print("[yellow]No plugins currently loaded[/yellow]")

        # Show available plugins if requested
        if show_all:
            console.print("\n[bold blue]Available Plugins:[/bold blue]")
            discovered = plugin_manager.discover_plugins()

            for plugin_meta in discovered:
                if plugin_type and plugin_meta.plugin_type.value != plugin_type:
                    continue

                console.print(f"  • {plugin_meta.name} v{plugin_meta.version}")
                console.print(f"    Type: {plugin_meta.plugin_type.value}")
                console.print(f"    Author: {plugin_meta.author}")
                console.print(f"    {plugin_meta.description}")

        # Show integration status
        console.print("\n[bold cyan]Plugin Integration Status:[/bold cyan]")

        # Check strategy integration
        try:
            from reactive_agents.core.reasoning.strategies.strategy_manager import (
                StrategyManager,
            )
            from reactive_agents.core.context.agent_context import AgentContext

            # Create mock context for testing
            mock_context = AgentContext(
                agent_name="test", provider_model_name="test", session_id="test"
            )
            strategy_manager = StrategyManager(mock_context)
            available_strategies = strategy_manager.get_available_strategies()
            plugin_strategies = strategy_manager.get_plugin_strategies()

            console.print(
                f"  • Strategy Manager: [green]✓[/green] ({len(available_strategies)} strategies)"
            )
            if plugin_strategies:
                console.print(
                    f"  • Plugin Strategies: [green]✓[/green] ({len(plugin_strategies)} plugin strategies)"
                )
            else:
                console.print(f"  • Plugin Strategies: [yellow]None loaded[/yellow]")

        except Exception as e:
            console.print(f"  • Strategy Manager: [red]✗[/red] Error: {e}")

        # Check tool integration
        try:
            from reactive_agents.core.tools.tool_manager import ToolManager

            # Note: Tool manager integration will be checked when agent is created
            console.print(
                f"  • Tool Manager: [green]✓[/green] Plugin integration available"
            )
        except Exception as e:
            console.print(f"  • Tool Manager: [red]✗[/red] Error: {e}")

    except ImportError:
        console.print("[red]Plugin system not available[/red]")
    except Exception as e:
        console.print(f"[red]Error checking plugin status: {e}[/red]")


@plugins.command()
@click.argument("plugin_name")
@click.option("--force", is_flag=True, help="Force reload if already loaded")
def load(plugin_name, force):
    """Load a specific plugin"""
    try:
        from reactive_agents.plugins.plugin_manager import get_plugin_manager
        import asyncio

        plugin_manager = get_plugin_manager()

        async def load_plugin():
            try:
                if force and plugin_name in plugin_manager.loaded_plugins:
                    await plugin_manager.unload_plugin(plugin_name)

                plugin = await plugin_manager.load_plugin(plugin_name)
                console.print(
                    f"[green]✓[/green] Successfully loaded plugin: {plugin.name} v{plugin.version}"
                )
                console.print(f"  Type: {plugin.plugin_type.value}")
                console.print(f"  Description: {plugin.description}")

            except Exception as e:
                console.print(f"[red]✗[/red] Failed to load plugin {plugin_name}: {e}")

        asyncio.run(load_plugin())

    except ImportError:
        console.print("[red]Plugin system not available[/red]")
    except Exception as e:
        console.print(f"[red]Error loading plugin: {e}[/red]")


@plugins.command()
@click.argument("plugin_name")
def unload(plugin_name):
    """Unload a specific plugin"""
    try:
        from reactive_agents.plugins.plugin_manager import get_plugin_manager
        import asyncio

        plugin_manager = get_plugin_manager()

        async def unload_plugin():
            try:
                await plugin_manager.unload_plugin(plugin_name)
                console.print(
                    f"[green]✓[/green] Successfully unloaded plugin: {plugin_name}"
                )

            except Exception as e:
                console.print(
                    f"[red]✗[/red] Failed to unload plugin {plugin_name}: {e}"
                )

        asyncio.run(unload_plugin())

    except ImportError:
        console.print("[red]Plugin system not available[/red]")
    except Exception as e:
        console.print(f"[red]Error unloading plugin: {e}[/red]")


if __name__ == "__main__":
    main()
