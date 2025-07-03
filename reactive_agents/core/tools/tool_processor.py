"""
Tool processing module for reactive-ai-agent framework.
Handles tool processing, validation, and management.
"""

from typing import List, Any, Optional, Dict
from reactive_agents.core.tools.base import Tool
from reactive_agents.core.tools.abstractions import ToolResult
from reactive_agents.utils.logging import Logger


class ToolProcessor:
    """
    Handles tool processing, validation, and management for agents.
    """

    def __init__(self, agent):
        """Initialize the ToolProcessor with an agent reference."""
        self.agent = agent
        self.context = agent.context
        self.agent_logger = Logger("ToolProcessor", "processor", self.context.log_level)

    def process_custom_tools(self, tools: List[Any]) -> List[Tool]:
        """
        Process custom tools to ensure they comply with the ToolProtocol interface.

        Args:
            tools: List of tools, which could be functions decorated with @tool

        Returns:
            List[Tool]: List of tools that all comply with the ToolProtocol interface
        """
        processed_tools = []

        for tool in tools:
            # Skip None values
            if tool is None:
                continue

            # If it's already a proper Tool class instance, use it as is
            if isinstance(tool, Tool):
                processed_tools.append(tool)
            # If it has a tool_definition attribute (likely a decorated function)
            elif hasattr(tool, "tool_definition"):
                # Create a wrapper class that implements the Tool interface
                class DecoratedFunctionWrapper(Tool):
                    # Use the function's attributes
                    name = tool.__name__
                    tool_definition = tool.tool_definition

                    def __init__(self, func):
                        self.func = func

                    async def use(self, params):
                        # Call the original function
                        result = await self.func(**params)
                        return ToolResult(result)

                # Create a wrapper instance and add it
                processed_tools.append(DecoratedFunctionWrapper(tool))
            # Otherwise it's not a compatible tool
            else:
                raise ValueError(
                    f"Custom tool {tool} is not compatible with ToolProtocol"
                )

        return processed_tools

    def validate_tool(self, tool: Any) -> bool:
        """
        Validate if a tool is properly configured and compatible.

        Args:
            tool: The tool to validate

        Returns:
            bool: True if the tool is valid, False otherwise
        """
        if tool is None:
            return False

        # Check if it's a Tool instance
        if isinstance(tool, Tool):
            return True

        # Check if it's a decorated function
        if hasattr(tool, "tool_definition"):
            return True

        return False

    def get_tool_by_name(self, name: str) -> Optional[Tool]:
        """
        Get a tool by its name.

        Args:
            name: The name of the tool to find

        Returns:
            Optional[Tool]: The tool if found, None otherwise
        """
        if not self.context.tool_manager:
            return None

        for tool in self.context.tool_manager.tools:
            if getattr(tool, "name", None) == name:
                return tool

        return None

    def get_available_tools(self) -> List[Tool]:
        """
        Get all available tools.

        Returns:
            List[Tool]: List of all available tools
        """
        if not self.context.tool_manager:
            return []

        return self.context.tool_manager.tools

    def get_tool_signatures(self) -> List[Dict[str, Any]]:
        """
        Get signatures for all available tools.

        Returns:
            List[Dict[str, Any]]: List of tool signatures
        """
        if not self.context.tool_manager:
            return []

        return self.context.tool_manager.tool_signatures
