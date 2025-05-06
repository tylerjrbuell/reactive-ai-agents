"""
MCP client fixtures for integration tests.

This module provides fixtures that properly mock MCPClient for integration testing,
addressing the issue with AsyncMock in await expressions.
"""

import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture
def mock_mcp_initialize():
    """
    Fixture that sets up a proper mock for MCPClient.initialize.

    This avoids the "object AsyncMock can't be used in await expression" error
    by making the mock an awaitable function rather than an AsyncMock object.

    In CI environment with DISABLE_MCP_CLIENT_SYSTEM_EXIT=1, it uses a completely
    isolated implementation that won't attempt any system calls.
    """
    # Check if we're in CI with special environment flag
    disable_system_exit = os.environ.get("DISABLE_MCP_CLIENT_SYSTEM_EXIT") == "1"

    # Create a mock client that will be returned
    mock_client = MagicMock()
    mock_client.close = AsyncMock()

    # Create additional mocks for CI environment
    if disable_system_exit:
        # Create comprehensive mocks for properties that might be accessed
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
    with patch("agent_mcp.client.MCPClient") as mock_mcp_class:
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
    with patch("agents.react_agent.ReactAgent.run") as mock_run:
        mock_run.return_value = {"status": "complete", "result": "Test successful"}
        yield mock_run


@pytest.fixture
def model_validation_bypass():
    """
    Fixture that bypasses model validation in OllamaModelProvider
    """
    with patch(
        "model_providers.ollama.OllamaModelProvider.validate_model"
    ) as mock_validate:
        # Make validate_model a no-op
        mock_validate.return_value = None
        yield mock_validate
