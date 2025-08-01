"""
Comprehensive mocking for agent_mcp to prevent Docker usage in tests.

This module provides functions to completely isolate tests from any Docker
or container operations by replacing key components with mock implementations.
"""

import os
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any, Optional, List


def create_mock_mcp_response(tool_name: str, result: Any = None) -> Dict[str, Any]:
    """
    Create a standardized mock response for MCP tool calls.

    Args:
        tool_name: The name of the tool being mocked
        result: Optional specific result to return

    Returns:
        A dict with the standard MCP response format
    """
    if result is None:
        result = f"Mocked {tool_name} result"

    return {
        "tool": tool_name,
        "result": result,
        "status": "success",
        "metadata": {"execution_time": 0.001, "mocked": True, "ci_environment": True},
    }


class MockMCPClient:
    """
    A comprehensive mock implementation of MCPClient that doesn't use Docker.

    This class can be used as a drop-in replacement for the real MCPClient
    during testing to avoid any Docker dependencies.
    """

    def __init__(self, server_filter: Optional[List[str]] = None):
        self.server_filter = server_filter or []
        # Mock properties
        self._stdio_client = False
        self._suppress_exit_errors = True
        self.initialized = True
        self._closed = False

        # Add mock methods for standard tools
        for tool_name in ["brave-search", "sqlite", "time", "shell", "file"]:
            setattr(
                self,
                tool_name.replace("-", "_"),
                AsyncMock(return_value=create_mock_mcp_response(tool_name)),
            )

    async def initialize(self):
        """Mock initialization that returns self without doing anything"""
        return self

    async def close(self):
        """Mock close method that does nothing"""
        self._closed = True

    async def call_tool(self, tool_name: str, params: dict) -> Dict[str, Any]:
        """Generic tool runner that returns mock responses for any tool"""
        return create_mock_mcp_response(tool_name)

    async def get_tools(self):
        """Return empty list of tools"""
        return []

    async def run_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Generic tool runner that returns mock responses for any tool

        Args:
            tool_name: Name of the tool to run
            **kwargs: Arguments to pass to the tool

        Returns:
            Mock response dictionary
        """
        # Convert dashes to underscores in tool name
        method_name = tool_name.replace("-", "_")

        # If we have a specific implementation, use it
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            if callable(method):
                return await method(**kwargs)

        # Otherwise return a generic mock response
        return create_mock_mcp_response(tool_name, f"Generic mock for {tool_name}")


def apply_mcp_mocks():
    """
    Apply all necessary mocks to prevent real MCP client usage

    This function patches various modules and functions to prevent
    any Docker or container operations during tests.
    """
    mocks = []

    # Only apply mocks if we're in CI or explicitly requested
    if (
        os.environ.get("DISABLE_MCP_CLIENT_SYSTEM_EXIT") == "1"
        or os.environ.get("MOCK_MCP_CLIENT") == "1"
        or os.environ.get("CI") == "true"
    ):

        # Replace the entire MCPClient class with our mock
        mcp_mock = patch(
            "reactive_agents.providers.external.client.MCPClient", MockMCPClient
        )
        mocks.append(mcp_mock)

        # Patch subprocess in other modules that might use it
        mocks.append(patch("subprocess.Popen", MagicMock()))
        mocks.append(patch("subprocess.check_output", MagicMock()))
        mocks.append(patch("subprocess.run", MagicMock()))

        # Patch shutil.which to pretend Docker is installed
        mocks.append(patch("shutil.which", return_value="/usr/bin/docker"))

        # Patch mcp module
        mocks.append(patch("mcp.client.stdio.stdio_client", AsyncMock()))
        mocks.append(patch("mcp.ClientSession", AsyncMock()))

        # Patch os.environ to ensure Docker-related vars are set
        env_vars = {"MOCK_MCP_CLIENT": "1", "NO_DOCKER": "1", "CI": "true"}
        mocks.append(patch.dict("os.environ", env_vars, clear=False))

        # Start all mocks
        for m in mocks:
            m.start()

    return mocks
