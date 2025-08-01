"""
MCP client fixtures for integration tests.

This module provides fixtures that properly mock MCPClient for integration testing,
addressing the issue with AsyncMock in await expressions.
"""

import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock


class SimpleMockMCPClient:
    """A very simple mock MCPClient that doesn't try to use Docker"""

    def __init__(self, *args, **kwargs):
        self.initialized = True
        self._closed = False
        self._stdio_client = False
        self._suppress_exit_errors = True
        # Add mock tool methods
        for tool_name in ["brave-search", "sqlite", "time", "shell", "file"]:
            setattr(
                self,
                tool_name.replace("-", "_"),
                AsyncMock(return_value={"result": f"Mocked {tool_name} result"}),
            )

    async def initialize(self):
        """Mock initialization that returns self"""
        return self

    async def close(self):
        """Mock close method"""
        self._closed = True
        print("SimpleMockMCPClient: close() called")

    async def get_tools(self):
        """Return empty list of tools"""
        return []

    async def call_tool(self, tool_name, params):
        """Mock tool calling"""
        return {"result": f"Mocked {tool_name} result"}


@pytest.fixture
def mock_mcp_initialize():
    """
    Fixture that sets up a proper mock for MCPClient.initialize.

    This avoids the "object AsyncMock can't be used in await expression" error
    by making the mock an awaitable function rather than an AsyncMock object.

    In CI environment with DISABLE_MCP_CLIENT_SYSTEM_EXIT=1, it uses a completely
    isolated implementation that won't attempt any system calls.
    """
    # Create a mock client that will be returned
    mock_client = MagicMock()
    mock_client.close = AsyncMock()

    # Create additional mocks for CI environment
    if (
        os.environ.get("DISABLE_MCP_CLIENT_SYSTEM_EXIT") == "1"
        or os.environ.get("MOCK_MCP_CLIENT") == "1"
    ):
        mock_client._stdio_client = False
        mock_client._suppress_exit_errors = True
        # Add all known tool methods as AsyncMocks to prevent any AttributeError
        for tool_name in ["brave-search", "sqlite", "time", "shell", "file"]:
            setattr(
                mock_client,
                tool_name.replace("-", "_"),
                AsyncMock(return_value={"result": f"Mocked {tool_name} result"}),
            )

    # Create a proper awaitable mock for initialize
    async def mock_initialize(*args, **kwargs):
        return mock_client

    # Patch MCPClient and its initialize method
    with patch("reactive_agents.providers.external.client.MCPClient") as mock_mcp_class:
        # Make the mock class return a MagicMock with a proper initialize method
        mock_instance = MagicMock()
        mock_instance.initialize = mock_initialize
        mock_mcp_class.return_value = mock_instance

        yield mock_client


@pytest.fixture
def mock_agent_run():
    """
    Fixture that sets up a proper mock for ReactAgent.run
    """
    with patch("reactive_agents.app.agents.reactive_agent.ReactiveAgent.run") as mock_run:
        mock_run.return_value = {"status": "complete", "result": "Test successful"}
        yield mock_run


@pytest.fixture
def model_validation_bypass():
    """
    Fixture that bypasses model validation in OllamaModelProvider
    """
    with patch(
        "reactive_agents.providers.llm.ollama.OllamaModelProvider.validate_model"
    ) as mock_validate:
        # Make validate_model a no-op
        mock_validate.return_value = None
        yield mock_validate
