"""
Plugin Generation Commands for Reactive Agents CLI

Laravel-style make commands for generating plugin scaffolding.
"""

import argparse

from reactive_agents.console.commands.base import Command
from reactive_agents.plugins.plugin_manager import get_plugin_generator


class MakeStrategyCommand(Command):
    """Generate a reasoning strategy plugin."""

    def __init__(self):
        super().__init__()
        self.name = "make:strategy"
        self.description = "Create a new reasoning strategy plugin"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command arguments."""
        parser.add_argument("name", help="Name of the strategy plugin")
        parser.add_argument("--description", help="Description of the strategy")
        parser.add_argument("--author", default="", help="Author name")

    async def execute_with_args(self, args: argparse.Namespace) -> int:
        """Execute the command."""
        try:
            generator = get_plugin_generator()
            plugin_path = generator.generate_strategy_plugin(
                args.name,
                args.description or f"Custom {args.name} strategy",
                args.author,
            )

            print(f"✅ Strategy plugin '{args.name}' created at {plugin_path}")
            print(f"📝 Next steps:")
            print(f"   1. Add {args.name.upper()} to ReasoningStrategies enum")
            print(f"   2. Implement strategy logic in execute_iteration method")
            print(f"   3. Test with: reactive agent:run --strategy {args.name}")

            return 0
        except Exception as e:
            print(f"Error generating strategy plugin: {e}")
            return 1


class MakeToolCommand(Command):
    """Generate a tool plugin."""

    def __init__(self):
        super().__init__()
        self.name = "make:tool"
        self.description = "Create a new tool plugin"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command arguments."""
        parser.add_argument("name", help="Name of the tool plugin")
        parser.add_argument("--description", help="Description of the tool")
        parser.add_argument("--author", default="", help="Author name")

    async def execute_with_args(self, args: argparse.Namespace) -> int:
        """Execute the command."""
        try:
            generator = get_plugin_generator()
            plugin_path = generator.generate_tool_plugin(
                args.name,
                args.description or f"Custom {args.name} tool",
                args.author,
            )

            print(f"✅ Tool plugin '{args.name}' created at {plugin_path}")
            print(f"📝 Next steps:")
            print(f"   1. Implement tool logic in execute method")
            print(f"   2. Test with: reactive agent:run --tools {args.name}")

            return 0
        except Exception as e:
            print(f"Error generating tool plugin: {e}")
            return 1


class MakeProviderCommand(Command):
    """Generate a provider plugin."""

    def __init__(self):
        super().__init__()
        self.name = "make:provider"
        self.description = "Create a new provider plugin"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command arguments."""
        parser.add_argument("name", help="Name of the provider plugin")
        parser.add_argument("--description", help="Description of the provider")
        parser.add_argument("--author", default="", help="Author name")

    async def execute_with_args(self, args: argparse.Namespace) -> int:
        """Execute the command."""
        try:
            generator = get_plugin_generator()
            plugin_path = generator.generate_provider_plugin(
                args.name,
                args.description or f"Custom {args.name} provider",
                args.author,
            )

            print(f"✅ Provider plugin '{args.name}' created at {plugin_path}")
            print(f"📝 Next steps:")
            print(f"   1. Implement provider logic in generate_response method")
            print(f"   2. Test with: reactive agent:run --provider {args.name}")

            return 0
        except Exception as e:
            print(f"Error generating provider plugin: {e}")
            return 1


class MakeAgentCommand(Command):
    """Generate an agent plugin."""

    def __init__(self):
        super().__init__()
        self.name = "make:agent"
        self.description = "Create a new agent plugin"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command arguments."""
        parser.add_argument("name", help="Name of the agent plugin")
        parser.add_argument("--description", help="Description of the agent")
        parser.add_argument("--author", default="", help="Author name")

    async def execute_with_args(self, args: argparse.Namespace) -> int:
        """Execute the command."""
        try:
            generator = get_plugin_generator()
            plugin_path = generator.generate_agent_plugin(
                args.name,
                args.description or f"Custom {args.name} agent",
                args.author,
            )

            print(f"✅ Agent plugin '{args.name}' created at {plugin_path}")
            print(f"📝 Next steps:")
            print(f"   1. Implement agent logic in process_task method")
            print(f"   2. Test with: reactive agent:run --agent {args.name}")

            return 0
        except Exception as e:
            print(f"Error generating agent plugin: {e}")
            return 1


# Export commands for CLI registration
PLUGIN_COMMANDS = [
    MakeStrategyCommand(),
    MakeToolCommand(),
    MakeProviderCommand(),
    MakeAgentCommand(),
]
