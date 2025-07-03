"""
Base Command Classes

Enhanced command system with argparse support for CLI commands.
"""

import argparse
from abc import ABC, abstractmethod
from typing import List


class Command(ABC):
    """Base class for all CLI commands with argparse support."""

    def __init__(self):
        self.name = ""
        self.description = ""
        self.parser: argparse.ArgumentParser = argparse.ArgumentParser(
            prog=self.name, description=self.description
        )
        self.add_arguments(self.parser)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments to the parser. Override in subclasses."""
        pass

    @abstractmethod
    async def execute_with_args(self, args: argparse.Namespace) -> int:
        """Execute the command with parsed arguments. Must be implemented by subclasses."""
        return 0

    async def execute(self, args: List[str]) -> int:
        """
        Execute the command with given arguments.

        This method handles argument parsing and delegates to execute_with_args.

        Args:
            args: Command line arguments as strings

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Parse arguments
            parsed_args = self.parser.parse_args(args)

            # Execute with parsed arguments
            return await self.execute_with_args(parsed_args)

        except SystemExit as e:
            # argparse calls sys.exit on error - convert to return code
            code = e.code
            if isinstance(code, int):
                return code
            return 1
        except Exception as e:
            print(f"Error executing command {self.name}: {e}")
            return 1

    def get_help(self) -> str:
        """Get help text for the command."""
        return self.parser.format_help()


# Backward compatibility alias
BaseCommand = Command
