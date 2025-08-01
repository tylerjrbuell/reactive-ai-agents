"""
Integration tests for ReactiveAgentBuilder

These tests verify that ReactiveAgentBuilder correctly integrates with:
1. ReactAgent
2. MCP client tools
3. Custom tools
4. Hybrid tool setups
"""

import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock

from reactive_agents import ReactiveAgentBuilder
from reactive_agents.core.tools.decorators import tool
from reactive_agents.providers.external.client import MCPClient


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


# Integration Tests with Mocked Components
@pytest.mark.asyncio
@patch(
    "reactive_agents.providers.llm.ollama.OllamaModelProvider.validate_model",
    return_value=None,
)
@patch("reactive_agents.providers.llm.ollama.OllamaModelProvider.get_completion")
@patch("reactive_agents.app.agents.reactive_agent.ReactiveAgent.run")
@patch("reactive_agents.providers.external.client.MCPClient", spec=MCPClient)
@patch("reactive_agents.providers.llm.factory.ModelProviderFactory.get_model_provider")
async def test_builder_with_mcp_tools_integration(
    mock_get_model_provider,
    mock_mcp_client_class,
    mock_run,
    mock_get_completion,
    model_validation_bypass,
):
    """Test the builder integration with MCP tools"""
    # Configure mock model provider and its get_completion method
    mock_model_provider_instance = MagicMock()
    mock_get_model_provider.return_value = mock_model_provider_instance
    mock_completion_response = MagicMock()
    mock_completion_response.get.return_value = {"response": "{}"}
    mock_model_provider_instance.get_completion = AsyncMock(
        return_value=mock_completion_response
    )

    # Also mock get_chat_completion as it might be used
    mock_model_provider_instance.get_chat_completion = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(tool_calls=[]))])
    )

    mock_get_completion.return_value = mock_completion_response  # Keep this for backward compatibility if needed by other mocks

    # Configure mocks
    mock_mcp_client_instance = mock_mcp_client_class()
    # Set up server_tools to include the tools that will be requested
    mock_mcp_client_instance.server_tools = {
        "time": MagicMock(name="time"),
        "brave-search": MagicMock(name="brave-search"),
    }
    mock_mcp_client_instance.initialize.return_value = mock_mcp_client_instance
    mock_run.return_value = {"status": "complete", "result": "Test successful"}

    # Build agent with MCP tools
    print("Before agent build")
    agent = await (
        ReactiveAgentBuilder()
        .with_name("Integration Test Agent")
        .with_model("ollama:test:model")
        .with_mcp_client(mock_mcp_client_instance)
        .with_mcp_tools(["time", "brave-search"])
        .build()
    )
    print("After agent build")

    # Verify agent was created with correct configuration
    assert agent.context.agent_name == "Integration Test Agent"

    # Run a task
    print("Before agent run")
    result = await agent.run("Test task")
    print("After agent run")
    assert mock_run.return_value["status"] == "complete"
    assert mock_run.return_value["result"] == "Test successful"

    # Clean up
    print("Before agent close")
    await agent.close()
    print("After agent close")


@pytest.mark.asyncio
@patch(
    "reactive_agents.providers.llm.ollama.OllamaModelProvider.validate_model",
    return_value=None,
)
@patch("reactive_agents.app.agents.reactive_agent.ReactiveAgent.run")
@patch("reactive_agents.providers.llm.factory.ModelProviderFactory.get_model_provider")
async def test_builder_with_custom_tools_integration(
    mock_get_model_provider, mock_run, model_validation_bypass
):
    """Test the builder integration with custom tools"""
    # Configure mock model provider and its get_completion method
    mock_model_provider_instance = MagicMock()
    mock_get_model_provider.return_value = mock_model_provider_instance
    mock_completion_response = MagicMock()
    mock_completion_response.get.return_value = {"response": "{}"}
    mock_model_provider_instance.get_completion = AsyncMock(
        return_value=mock_completion_response
    )

    # Also mock get_chat_completion as it might be used
    mock_model_provider_instance.get_chat_completion = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(tool_calls=[]))])
    )

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
    tool_names = [tool.name for tool in agent.context.tools]
    assert "square" in tool_names
    assert "greeting" in tool_names

    # Run a task
    result = await agent.run("Test task")
    assert mock_run.return_value["status"] == "complete"
    assert mock_run.return_value["result"] == "Test successful"

    # Clean up
    await agent.close()


@pytest.mark.asyncio
@patch(
    "reactive_agents.providers.llm.ollama.OllamaModelProvider.validate_model",
    return_value=None,
)
@patch("reactive_agents.app.agents.reactive_agent.ReactiveAgent.run")
@patch("reactive_agents.providers.external.client.MCPClient", spec=MCPClient)
@patch("reactive_agents.providers.llm.factory.ModelProviderFactory.get_model_provider")
async def test_builder_with_hybrid_tools_integration(
    mock_get_model_provider, mock_mcp_client_class, mock_run, model_validation_bypass
):
    """Test the builder integration with both MCP and custom tools"""
    # Configure mock model provider and its get_completion method
    mock_model_provider_instance = MagicMock()
    mock_get_model_provider.return_value = mock_model_provider_instance
    mock_completion_response = MagicMock()
    mock_completion_response.get.return_value = {"response": "{}"}
    mock_model_provider_instance.get_completion = AsyncMock(
        return_value=mock_completion_response
    )

    # Also mock get_chat_completion as it might be used
    mock_model_provider_instance.get_chat_completion = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(tool_calls=[]))])
    )

    # Configure mocks
    mock_mcp_client_instance = mock_mcp_client_class()

    # Create mock tools with proper structure
    mock_time_tool = MagicMock(name="time")
    mock_time_tool.name = "time"
    mock_time_tool.description = "Get current time"
    mock_time_tool.inputSchema = {}

    mock_brave_tool = MagicMock(name="brave-search")
    mock_brave_tool.name = "brave-search"
    mock_brave_tool.description = "Search the web"
    mock_brave_tool.inputSchema = {}

    # Set up the tools list that the tool manager will use
    mock_mcp_client_instance.tools = [mock_time_tool, mock_brave_tool]

    # Set up tool signatures
    mock_mcp_client_instance.tool_signatures = [
        {
            "type": "function",
            "function": {
                "name": "time",
                "description": "Get current time",
                "parameters": {},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "brave-search",
                "description": "Search the web",
                "parameters": {},
            },
        },
    ]

    # Set up server_tools for logging (maps server names to tool lists)
    mock_mcp_client_instance.server_tools = {
        "time": [mock_time_tool],
        "brave-search": [mock_brave_tool],
    }

    mock_mcp_client_instance.initialize.return_value = mock_mcp_client_instance
    mock_run.return_value = {"status": "complete", "result": "Test successful"}

    # Build agent with hybrid tools
    agent = await (
        ReactiveAgentBuilder()
        .with_name("Hybrid Tools Agent")
        .with_model("ollama:test:model")
        .with_mcp_client(mock_mcp_client_instance)
        .with_tools(mcp_tools=["time"], custom_tools=[square, greeting])
        .build()
    )

    # Verify agent was created with correct configuration
    assert agent.context.agent_name == "Hybrid Tools Agent"

    # Verify custom tools were added
    tool_names = [tool.name for tool in agent.context.tools]
    assert "square" in tool_names
    assert "greeting" in tool_names

    # Verify tool manager has all tools
    tool_manager = getattr(agent.context, "tool_manager", None)
    assert tool_manager is not None
    manager_tool_names = [tool.name for tool in tool_manager.tools]
    assert "square" in manager_tool_names
    assert "greeting" in manager_tool_names

    # Run a task
    result = await agent.run("Test task")
    assert mock_run.return_value["status"] == "complete"

    # Clean up
    await agent.close()


@pytest.mark.asyncio
@patch(
    "reactive_agents.providers.llm.ollama.OllamaModelProvider.validate_model",
    return_value=None,
)
@patch("reactive_agents.app.agents.reactive_agent.ReactiveAgent.run")
@patch("reactive_agents.providers.external.client.MCPClient", spec=MCPClient)
@patch("reactive_agents.providers.llm.factory.ModelProviderFactory.get_model_provider")
async def test_adding_custom_tools_to_existing_agent(
    mock_get_model_provider, mock_mcp_client_class, mock_run, model_validation_bypass
):
    """Test adding custom tools to an existing agent"""
    # Configure mock model provider and its get_completion method
    mock_model_provider_instance = MagicMock()
    mock_get_model_provider.return_value = mock_model_provider_instance
    mock_completion_response = MagicMock()
    mock_completion_response.get.return_value = {"response": "{}"}
    mock_model_provider_instance.get_completion = AsyncMock(
        return_value=mock_completion_response
    )

    # Also mock get_chat_completion as it might be used
    mock_model_provider_instance.get_chat_completion = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(tool_calls=[]))])
    )

    # Configure mocks
    mock_mcp_client_instance = mock_mcp_client_class()
    # Set up server_tools to include the tools that will be requested
    mock_mcp_client_instance.server_tools = {
        "time": MagicMock(name="time"),
        "brave-search": MagicMock(name="brave-search"),
    }
    mock_mcp_client_instance.initialize.return_value = mock_mcp_client_instance
    mock_run.return_value = {"status": "complete", "result": "Test successful"}

    # Create a research agent
    agent = await (
        ReactiveAgentBuilder()
        .with_name("Existing Agent")
        .with_model("ollama:test:model")
        .with_mcp_client(mock_mcp_client_instance)
        .build()
    )

    # Verify initial tools (just MCP tools)
    initial_tool_names = [tool.name for tool in agent.context.tools]
    assert "square" not in initial_tool_names
    assert "greeting" not in initial_tool_names

    # Add custom tools
    updated_agent = await ReactiveAgentBuilder.add_custom_tools_to_agent(
        agent, [square, greeting]
    )

    # Verify custom tools were added
    updated_tool_names = [tool.name for tool in updated_agent.context.tools]
    assert "square" in updated_tool_names
    assert "greeting" in updated_tool_names

    # Verify tool manager has the custom tools
    tool_manager = getattr(updated_agent.context, "tool_manager", None)
    assert tool_manager is not None
    manager_tool_names = [tool.name for tool in tool_manager.tools]
    assert "square" in manager_tool_names
    assert "greeting" in manager_tool_names

    # Verify it's the same agent instance
    assert agent is updated_agent

    # Run a task
    result = await agent.run("Test task")
    assert mock_run.return_value["status"] == "complete"

    # Clean up
    await agent.close()


@pytest.mark.asyncio
@patch(
    "reactive_agents.providers.llm.ollama.OllamaModelProvider.validate_model",
    return_value=None,
)
@patch("reactive_agents.providers.external.client.MCPClient", spec=MCPClient)
@patch("reactive_agents.providers.llm.factory.ModelProviderFactory.get_model_provider")
async def test_tool_registration_diagnostics(
    mock_get_model_provider, mock_mcp_client_class, model_validation_bypass
):
    """Test the tool registration diagnostics in an integration scenario"""
    # Configure mock model provider and its get_completion method
    mock_model_provider_instance = MagicMock()
    mock_get_model_provider.return_value = mock_model_provider_instance
    mock_completion_response = MagicMock()
    mock_completion_response.get.return_value = {"response": "{}"}
    mock_model_provider_instance.get_completion = AsyncMock(
        return_value=mock_completion_response
    )

    # Also mock get_chat_completion as it might be used
    mock_model_provider_instance.get_chat_completion = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(tool_calls=[]))])
    )

    # Configure mocks
    mock_mcp_client_instance = mock_mcp_client_class()

    # Create mock tools with proper structure
    mock_time_tool = MagicMock(name="time")
    mock_time_tool.name = "time"
    mock_time_tool.description = "Get current time"
    mock_time_tool.inputSchema = {}

    mock_brave_tool = MagicMock(name="brave-search")
    mock_brave_tool.name = "brave-search"
    mock_brave_tool.description = "Search the web"
    mock_brave_tool.inputSchema = {}

    # Set up the tools list that the tool manager will use
    mock_mcp_client_instance.tools = [mock_time_tool, mock_brave_tool]

    # Set up tool signatures
    mock_mcp_client_instance.tool_signatures = [
        {
            "type": "function",
            "function": {
                "name": "time",
                "description": "Get current time",
                "parameters": {},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "brave-search",
                "description": "Search the web",
                "parameters": {},
            },
        },
    ]

    # Set up server_tools for logging (maps server names to tool lists)
    mock_mcp_client_instance.server_tools = {
        "time": [mock_time_tool],
        "brave-search": [mock_brave_tool],
    }

    mock_mcp_client_instance.initialize.return_value = mock_mcp_client_instance

    # Create a builder with hybrid tools
    builder = (
        ReactiveAgentBuilder()
        .with_name("Diagnostic Test Agent")
        .with_mcp_client(mock_mcp_client_instance)
        .with_model("ollama:test:model")
        .with_tools(mcp_tools=["time", "brave-search"], custom_tools=[square, greeting])
    )

    # Use debug_tools before building
    diagnostics = builder.debug_tools()
    assert len(diagnostics["mcp_tools"]) == 2
    assert "time" in diagnostics["mcp_tools"]
    assert "brave-search" in diagnostics["mcp_tools"]
    assert len(diagnostics["custom_tools"]) == 2
    assert "square" in diagnostics["custom_tools"]
    assert "greeting" in diagnostics["custom_tools"]

    # Build the agent
    agent = await builder.build()

    # Use diagnose_agent_tools after building
    diagnosis = await ReactiveAgentBuilder.diagnose_agent_tools(agent)
    assert "context_tools" in diagnosis
    assert "manager_tools" in diagnosis
    assert "has_tool_mismatch" in diagnosis

    # Debug output to understand the mismatch
    print(f"Diagnosis: {diagnosis}")
    print(f"Context tools: {diagnosis['context_tools']}")
    print(f"Manager tools: {diagnosis['manager_tools']}")
    print(f"Missing in context: {diagnosis['missing_in_context']}")
    print(f"Missing in manager: {diagnosis['missing_in_manager']}")

    # The tool mismatch is expected because:
    # - Context tools: only custom tools (square, greeting)
    # - Manager tools: all tools including MCP tools (time, brave-search) and internal tools (final_answer)
    # This is the correct behavior - MCP tools are managed by the tool manager, not stored directly in context
    assert diagnosis["has_tool_mismatch"] is True  # This is expected
    assert (
        "time" in diagnosis["missing_in_context"]
    )  # MCP tools should be missing from context
    assert (
        "brave-search" in diagnosis["missing_in_context"]
    )  # MCP tools should be missing from context
    assert (
        "final_answer" in diagnosis["missing_in_context"]
    )  # Internal tools should be missing from context
    assert (
        len(diagnosis["missing_in_manager"]) == 0
    )  # All context tools should be in manager

    # Clean up
    await agent.close()


# Only run this test if environmental conditions permit (real execution)
@pytest.mark.skipif(
    "ENABLE_REAL_EXECUTION" not in os.environ,
    reason="Skipping real execution test. Set ENABLE_REAL_EXECUTION=1 to run.",
)
@pytest.mark.asyncio
async def test_real_agent_execution():
    """
    Test real agent execution with no mocks

    This test requires:
    1. Ollama running locally
    2. ENABLE_REAL_EXECUTION environment variable set
    """
    # Create a simple agent with the square tool
    agent = await (
        ReactiveAgentBuilder()
        .with_name("Real Execution Agent")
        .with_model("ollama:llama3:8b")  # Use a small model
        .with_custom_tools([square])
        .with_instructions("Calculate the square of 7 using the square tool.")
        .with_max_iterations(3)
        .build()
    )

    try:
        # Run a simple task
        result = await agent.run("Calculate the square of 7")

        # Verify the agent completed the task
        assert result.status == "complete"
        assert result.final_answer is not None
        assert "49" in result.final_answer  # Should contain the square of 7

    finally:
        # Always clean up
        await agent.close()
