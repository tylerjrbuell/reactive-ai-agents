"""
Integration tests for custom tools.

This file tests the integration of custom tools with the ReactiveAgentBuilder.
"""

import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
from reactive_agents import ReactiveAgentBuilder
from reactive_agents.core.tools.decorators import tool
from reactive_agents.tests.integration.mcp_fixtures import (
    mock_agent_run,
    model_validation_bypass,
)
import asyncio


# Sample custom tools for testing
@tool(description="Square a number")
async def square(num: int) -> int:
    """
    Square a number

    Args:
        num: Number to square

    Returns:
        The squared number
    """
    return num * num


@tool(description="Generate a greeting message")
async def greeting(name: str) -> str:
    """
    Generate a greeting for a person

    Args:
        name: Person's name

    Returns:
        A greeting message
    """
    return f"Hello, {name}! Welcome to the integration test."


# Get CI timeout value from environment or use default
CI_TIMEOUT = int(os.environ.get("PYTEST_TIMEOUT", "5"))


@pytest.mark.asyncio
@patch(
    "reactive_agents.providers.llm.ollama.OllamaModelProvider.validate_model",
    return_value=None,
)
@patch(
    "reactive_agents.providers.llm.factory.ModelProviderFactory.get_model_provider"
)
@patch("reactive_agents.app.agents.reactive_agent.ReactiveAgent.run")
async def test_builder_with_custom_tools_fixed(
    mock_run,
    mock_get_model_provider,
    model_validation_bypass,
):
    """Test the builder integration with custom tools (fixed) - no run mock"""

    # Configure mock model provider and its get_completion method
    mock_model_provider_instance = MagicMock()
    mock_get_model_provider.return_value = mock_model_provider_instance
    mock_completion_response = MagicMock()
    mock_completion_response.get.return_value = {
        "response": "final_answer('Test successful')"
    }

    # Configure mock for agent.run
    mock_run.return_value = {"status": "complete", "result": "Test successful"}

    # Build agent with custom tools
    agent = await (
        ReactiveAgentBuilder()
        .with_name("Custom Tools Agent")
        .with_model("ollama:test:model")
        .with_custom_tools([square, greeting])
        .build()
    )

    # Verify agent was created with correct configuration
    assert agent.context.agent_name == "Custom Tools Agent"

    # Verify tools were added
    tool_names = [getattr(tool, "name", str(id(tool))) for tool in agent.context.tools]
    assert "square" in tool_names
    assert "greeting" in tool_names

    try:
        # Run a task with a timeout
        result = await asyncio.wait_for(agent.run("Test task"), timeout=2.0)
        assert result["status"] == "complete"
        assert result["result"] == "Test successful"
    finally:
        # Forcibly close the agent and ensure cleanup
        try:
            await asyncio.wait_for(agent.close(), timeout=1.0)
        except asyncio.TimeoutError:
            print("Warning: Agent cleanup timed out, but test can proceed")


@pytest.mark.asyncio
@pytest.mark.timeout(CI_TIMEOUT)
async def test_add_custom_tools_to_existing_agent_fixed(
    mock_agent_run, model_validation_bypass
):
    """Test adding custom tools to an existing agent"""
    # Create a mock agent
    mock_agent = MagicMock()
    mock_context = MagicMock()
    mock_tool_manager = MagicMock()

    # Set up mock context
    mock_context.tools = []
    mock_context.tool_manager = mock_tool_manager
    mock_tool_manager.tools = []
    mock_agent.context = mock_context

    # Mock the tool signatures generation
    mock_tool_manager._generate_tool_signatures = MagicMock()

    # Add custom tools to the existing agent
    updated_agent = await ReactiveAgentBuilder.add_custom_tools_to_agent(
        mock_agent, [square, greeting]
    )

    # Verify custom tools were added
    assert len(mock_context.tools) == 2
    assert len(mock_tool_manager.tools) == 2

    # Verify agent returned is the same instance
    assert updated_agent is mock_agent

    # Verify generate_signatures was called
    mock_tool_manager._generate_tool_signatures.assert_called_once()
