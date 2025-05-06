"""
MCP client fixtures for integration tests.

This module provides fixtures that properly mock MCPClient for integration testing,
addressing the issue with AsyncMock in await expressions.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture
def mock_mcp_initialize():
    """
    Fixture that sets up a proper mock for MCPClient.initialize.

    This avoids the "object AsyncMock can't be used in await expression" error
    by making the mock an awaitable function rather than an AsyncMock object.
    """
    # Create a mock client that will be returned
    mock_client = MagicMock()
    mock_client.close = AsyncMock()

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
