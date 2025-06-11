"""
Tests for ReactAgentBuilder class and related components.

This module tests:
1. ReactAgentBuilder initialization and configuration
2. Builder methods like with_name, with_model, etc.
3. Tool registration (MCP tools, custom tools, and hybrid)
4. Pydantic models integration
5. Builder Factory methods
6. Diagnostic utilities
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List, Dict, Any

from reactive_agents.agents.builders import (
    ReactAgentBuilder,
    quick_create_agent,
    LogLevel,
    ConfirmationConfig,
    ToolConfig,
)
from reactive_agents.tools.decorators import tool
from reactive_agents.tools.base import Tool


# Custom tools for testing
@tool(description="Test tool for squaring numbers")
async def square_tool(number: int) -> int:
    """Square a number"""
    return number * number


@tool(description="Test tool for getting weather")
async def weather_tool(location: str) -> str:
    """Get weather for a location"""
    return f"Weather in {location}: Sunny, 72°F"


# Fixtures
@pytest.fixture
def basic_builder():
    """Create a basic ReactAgentBuilder instance"""
    return ReactAgentBuilder()


@pytest.fixture
async def mock_mcp_client():
    """Create a mock MCP client"""
    mock_client = MagicMock()
    mock_client.initialize = AsyncMock(return_value=mock_client)
    mock_client.close = AsyncMock()
    return mock_client


# Basic Builder Tests
def test_builder_initialization(basic_builder):
    """Test ReactAgentBuilder initialization with default values"""
    assert basic_builder._config["agent_name"] == "ReactAgent"
    assert basic_builder._config["role"] == "Task Executor"
    assert basic_builder._config["provider_model_name"] == "ollama:qwen2:7b"
    assert basic_builder._mcp_client is None
    assert basic_builder._mcp_server_filter is None
    assert len(basic_builder._custom_tools) == 0
    assert isinstance(basic_builder._registered_tools, set)
    assert len(basic_builder._registered_tools) == 0


def test_builder_with_methods(basic_builder):
    """Test the basic with_* methods of ReactAgentBuilder"""
    # Test with_name
    builder = basic_builder.with_name("TestAgent")
    assert builder._config["agent_name"] == "TestAgent"
    assert builder is basic_builder  # Verify fluent interface

    # Test with_role
    builder = builder.with_role("Researcher")
    assert builder._config["role"] == "Researcher"

    # Test with_model
    builder = builder.with_model("ollama:llama3:8b")
    assert builder._config["provider_model_name"] == "ollama:llama3:8b"

    # Test with_instructions
    builder = builder.with_instructions("Test instructions")
    assert builder._config["instructions"] == "Test instructions"

    # Test with_max_iterations
    builder = builder.with_max_iterations(5)
    assert builder._config["max_iterations"] == 5

    # Test with_reflection
    builder = builder.with_reflection(True)
    assert builder._config["reflect_enabled"] is True

    # Test with_log_level with string
    builder = builder.with_log_level("debug")
    assert builder._config["log_level"] == "debug"

    # Test with_log_level with enum
    builder = builder.with_log_level(LogLevel.WARNING)
    assert builder._config["log_level"] == "warning"


# Tool Registration Tests
def test_with_mcp_tools(basic_builder):
    """Test the with_mcp_tools method"""
    builder = basic_builder.with_mcp_tools(["time", "brave-search"])
    assert builder._mcp_server_filter == ["time", "brave-search"]
    assert "mcp:time" in builder._registered_tools
    assert "mcp:brave-search" in builder._registered_tools
    assert len(builder._registered_tools) == 2


def test_with_custom_tools(basic_builder):
    """Test the with_custom_tools method"""
    builder = basic_builder.with_custom_tools([square_tool, weather_tool])
    assert len(builder._custom_tools) == 2
    assert "custom:square_tool" in builder._registered_tools
    assert "custom:weather_tool" in builder._registered_tools

    # Check tool wrapping logic
    for tool in builder._custom_tools:
        assert hasattr(tool, "name")
        assert hasattr(tool, "tool_definition")


def test_with_tools_method(basic_builder):
    """Test the combined with_tools method"""
    builder = basic_builder.with_tools(
        mcp_tools=["time", "brave-search"], custom_tools=[square_tool]
    )

    assert builder._mcp_server_filter == ["time", "brave-search"]
    assert len(builder._custom_tools) == 1
    assert "mcp:time" in builder._registered_tools
    assert "mcp:brave-search" in builder._registered_tools
    assert "custom:square_tool" in builder._registered_tools

    # Check hybrid_tools_config
    assert "hybrid_tools_config" in builder._config
    assert builder._config["hybrid_tools_config"]["mcp_tools"] == [
        "time",
        "brave-search",
    ]
    assert builder._config["hybrid_tools_config"]["custom_tools_count"] == 1


def test_with_confirmation(basic_builder):
    """Test the with_confirmation method"""
    # Create a mock callback
    mock_callback = AsyncMock()

    # Test with dict config
    config_dict = {"enabled": True, "strategy": "always", "excluded_tools": ["tool1"]}
    builder = basic_builder.with_confirmation(mock_callback, config_dict)
    assert builder._config["confirmation_callback"] is mock_callback
    assert builder._config["confirmation_config"] == config_dict

    # Test with Pydantic model config
    config_model = ConfirmationConfig(
        enabled=True, strategy="selective", allowed_silent_tools=["tool2"]
    )
    builder = basic_builder.with_confirmation(mock_callback, config_model)
    assert builder._config["confirmation_callback"] is mock_callback
    assert builder._config["confirmation_config"] == config_model.dict()


def test_with_advanced_config(basic_builder):
    """Test the with_advanced_config method"""
    builder = basic_builder.with_advanced_config(
        custom_param1="value1", custom_param2=123
    )
    assert builder._config["custom_param1"] == "value1"
    assert builder._config["custom_param2"] == 123


# Diagnostic Methods Tests
def test_debug_tools(basic_builder):
    """Test the debug_tools method"""
    # Create a builder with MCP and custom tools
    builder = basic_builder.with_mcp_tools(["time", "brave-search"]).with_custom_tools(
        [square_tool, weather_tool]
    )

    # Get diagnostics
    diagnostics = builder.debug_tools()

    # Verify diagnostics information
    assert len(diagnostics["mcp_tools"]) == 2
    assert "time" in diagnostics["mcp_tools"]
    assert "brave-search" in diagnostics["mcp_tools"]

    assert len(diagnostics["custom_tools"]) == 2
    assert "square_tool" in diagnostics["custom_tools"]
    assert "weather_tool" in diagnostics["custom_tools"]

    assert len(diagnostics["custom_tool_details"]) == 2
    for tool_detail in diagnostics["custom_tool_details"]:
        assert "name" in tool_detail
        assert "has_name_attr" in tool_detail
        assert "has_tool_definition" in tool_detail
        assert "type" in tool_detail

    assert diagnostics["mcp_client_initialized"] is False
    assert diagnostics["server_filter"] == ["time", "brave-search"]
    assert diagnostics["total_tools"] == 4


@pytest.mark.asyncio
async def test_diagnose_agent_tools():
    """Test the diagnose_agent_tools static method"""
    # Create a mock agent
    mock_agent = MagicMock()
    mock_context = MagicMock()
    mock_tool_manager = MagicMock()

    # Set up context tools
    context_tool1 = MagicMock()
    context_tool1.name = "tool1"
    context_tool2 = MagicMock()
    context_tool2.name = "tool2"
    mock_context.tools = [context_tool1, context_tool2]

    # Set up tool manager tools
    manager_tool1 = MagicMock()
    manager_tool1.name = "tool1"
    manager_tool3 = MagicMock()
    manager_tool3.name = "tool3"
    mock_tool_manager.tools = [manager_tool1, manager_tool3]

    # Connect everything
    mock_context.tool_manager = mock_tool_manager
    mock_agent.context = mock_context

    # Call diagnose_agent_tools
    diagnosis = await ReactAgentBuilder.diagnose_agent_tools(mock_agent)

    # Verify diagnosis information
    assert "context_tools" in diagnosis
    assert "manager_tools" in diagnosis
    assert "has_tool_mismatch" in diagnosis
    assert "missing_in_context" in diagnosis
    assert "missing_in_manager" in diagnosis

    assert set(diagnosis["context_tools"]) == {"tool1", "tool2"}
    assert set(diagnosis["manager_tools"]) == {"tool1", "tool3"}
    assert diagnosis["has_tool_mismatch"] is True
    assert set(diagnosis["missing_in_context"]) == {"tool3"}
    assert set(diagnosis["missing_in_manager"]) == {"tool2"}


# Tool Unification Logic Tests
@pytest.mark.asyncio
async def test_unify_tool_registration():
    """Test the _unify_tool_registration method"""
    builder = ReactAgentBuilder()

    # Create a mock agent
    mock_agent = MagicMock()
    mock_context = MagicMock()
    mock_tool_manager = MagicMock()

    # Set up context tools list
    context_tool1 = MagicMock()
    context_tool1.name = "tool1"
    context_tool2 = MagicMock()
    context_tool2.name = "tool2"
    context_tools = [context_tool1, context_tool2]
    mock_context.tools = context_tools

    # Set up tool manager tools
    manager_tool1 = MagicMock()
    manager_tool1.name = "tool1"
    manager_tool3 = MagicMock()
    manager_tool3.name = "tool3"
    mock_tool_manager.tools = [manager_tool1, manager_tool3]

    # Connect everything
    mock_context.tool_manager = mock_tool_manager
    mock_agent.context = mock_context

    # Create a mock for generate_signatures
    mock_generate_signatures = MagicMock()
    mock_tool_manager._generate_tool_signatures = mock_generate_signatures

    # Call the method
    builder._unify_tool_registration(mock_agent)

    # Verify tools were unified
    all_tools_names = {tool.name for tool in mock_context.tools}
    assert all_tools_names == {"tool1", "tool2", "tool3"}

    all_manager_tools_names = {tool.name for tool in mock_tool_manager.tools}
    assert all_manager_tools_names == {"tool1", "tool2", "tool3"}

    # Verify generate_signatures was called
    mock_generate_signatures.assert_called_once()


# Factory Methods Tests
@pytest.mark.asyncio
@patch("reactive_agents.agents.builders.ReactAgentBuilder.build")
@patch.object(ReactAgentBuilder, "__new__")
async def test_research_agent_factory(mock_new, mock_build):
    """Test the research_agent factory method"""
    # Create a mock agent
    mock_agent = MagicMock()
    mock_build.return_value = mock_agent

    # Create a mock builder that will be returned by __new__
    mock_builder = MagicMock()
    mock_builder._config = {}
    mock_builder._mcp_server_filter = None
    mock_builder.with_name.return_value = mock_builder
    mock_builder.with_role.return_value = mock_builder
    mock_builder.with_model.return_value = mock_builder
    mock_builder.with_instructions.return_value = mock_builder
    mock_builder.with_mcp_tools.return_value = mock_builder
    mock_builder.with_reflection.return_value = mock_builder
    mock_builder.with_max_iterations.return_value = mock_builder
    # Set build as an AsyncMock
    mock_builder.build = AsyncMock(return_value=mock_agent)
    mock_new.return_value = mock_builder

    # Call the factory method
    agent = await ReactAgentBuilder.research_agent(model="ollama:test:model")

    # Verify the agent was built correctly
    assert agent is mock_agent

    # Verify the builder was configured correctly
    mock_builder.with_name.assert_called_once_with("Research Agent")
    mock_builder.with_role.assert_called_once_with("Research Assistant")
    mock_builder.with_model.assert_called_once_with("ollama:test:model")
    mock_builder.with_mcp_tools.assert_called_once_with(["brave-search", "time"])
    mock_builder.with_reflection.assert_called_once_with(True)
    mock_builder.with_max_iterations.assert_called_once_with(15)
    mock_builder.build.assert_called_once()


@pytest.mark.asyncio
@patch("reactive_agents.agents.builders.ReactAgentBuilder.build")
@patch.object(ReactAgentBuilder, "__new__")
async def test_database_agent_factory(mock_new, mock_build):
    """Test the database_agent factory method"""
    # Create a mock agent
    mock_agent = MagicMock()
    mock_build.return_value = mock_agent

    # Create a mock builder that will be returned by __new__
    mock_builder = MagicMock()
    mock_builder._config = {}
    mock_builder._mcp_server_filter = None
    mock_builder.with_name.return_value = mock_builder
    mock_builder.with_role.return_value = mock_builder
    mock_builder.with_model.return_value = mock_builder
    mock_builder.with_instructions.return_value = mock_builder
    mock_builder.with_mcp_tools.return_value = mock_builder
    mock_builder.with_reflection.return_value = mock_builder
    # Set build as an AsyncMock
    mock_builder.build = AsyncMock(return_value=mock_agent)
    mock_new.return_value = mock_builder

    # Call the factory method
    agent = await ReactAgentBuilder.database_agent()

    # Verify the agent was built correctly
    assert agent is mock_agent

    # Verify the builder was configured correctly
    mock_builder.with_name.assert_called_once_with("Database Agent")
    mock_builder.with_role.assert_called_once_with("Database Assistant")
    mock_builder.with_mcp_tools.assert_called_once_with(["sqlite"])
    mock_builder.with_reflection.assert_called_once_with(True)
    mock_builder.build.assert_called_once()


@pytest.mark.asyncio
@patch("reactive_agents.agents.builders.ReactAgentBuilder.build")
@patch.object(ReactAgentBuilder, "__new__")
async def test_crypto_research_agent_factory(mock_new, mock_build):
    """Test the crypto_research_agent factory method"""
    # Create a mock agent
    mock_agent = MagicMock()
    mock_build.return_value = mock_agent

    # Create a mock builder that will be returned by __new__
    mock_builder = MagicMock()
    mock_builder._config = {}
    mock_builder._mcp_server_filter = None
    mock_builder.with_name.return_value = mock_builder
    mock_builder.with_role.return_value = mock_builder
    mock_builder.with_model.return_value = mock_builder
    mock_builder.with_instructions.return_value = mock_builder
    mock_builder.with_mcp_tools.return_value = mock_builder
    mock_builder.with_reflection.return_value = mock_builder
    mock_builder.with_max_iterations.return_value = mock_builder
    mock_builder.with_confirmation.return_value = mock_builder
    # Set build as an AsyncMock
    mock_builder.build = AsyncMock(return_value=mock_agent)
    mock_new.return_value = mock_builder

    # Create a mock callback
    mock_callback = AsyncMock()

    # Call the factory method with custom cryptocurrencies
    agent = await ReactAgentBuilder.crypto_research_agent(
        model="ollama:test:model",
        confirmation_callback=mock_callback,
        cryptocurrencies=["Bitcoin", "Ethereum", "Solana"],
    )

    # Verify the agent was built correctly
    assert agent is mock_agent

    # Verify the builder was configured correctly
    mock_builder.with_name.assert_called_once_with("Crypto Research Agent")
    mock_builder.with_role.assert_called_once_with("Financial Data Analyst")
    mock_builder.with_model.assert_called_once_with("ollama:test:model")
    mock_builder.with_mcp_tools.assert_called_once_with(
        ["brave-search", "sqlite", "time"]
    )
    mock_builder.with_reflection.assert_called_once_with(True)
    mock_builder.with_max_iterations.assert_called_once_with(15)
    mock_builder.with_confirmation.assert_called_once_with(mock_callback)

    # Check that instructions were set and contained the cryptocurrencies
    instructions_call = mock_builder.with_instructions.call_args[0][0]
    assert "Bitcoin" in instructions_call
    assert "Ethereum" in instructions_call
    assert "Solana" in instructions_call

    mock_builder.build.assert_called_once()


@pytest.mark.asyncio
async def test_add_custom_tools_to_agent():
    """Test the add_custom_tools_to_agent method"""
    # Create mock agent
    mock_agent = MagicMock()
    mock_context = MagicMock()
    mock_tool_manager = MagicMock()

    # Mock context tools
    mock_context.tools = []
    mock_tool_manager.tools = []
    mock_context.tool_manager = mock_tool_manager
    mock_agent.context = mock_context

    # Create a mock for generate_signatures
    mock_generate_signatures = MagicMock()
    mock_tool_manager._generate_tool_signatures = mock_generate_signatures

    # Call the method
    updated_agent = await ReactAgentBuilder.add_custom_tools_to_agent(
        mock_agent, [square_tool, weather_tool]
    )

    # Verify tools were added
    assert len(mock_context.tools) == 2
    assert len(mock_tool_manager.tools) == 2

    # Verify agent was returned
    assert updated_agent is mock_agent

    # Verify generate_signatures was called
    mock_generate_signatures.assert_called_once()


# Quick Create Agent Tests
@pytest.mark.asyncio
@patch(
    "reactive_agents.model_providers.ollama.OllamaModelProvider", new_callable=MagicMock
)
@patch("reactive_agents.agents.builders.ReactAgentBuilder")
async def test_quick_create_agent(
    mock_builder_class, mock_ollama_provider_class, model_validation_bypass
):
    """Test the quick_create_agent function"""
    # Configure the mocked OllamaModelProvider instance
    mock_ollama_provider_instance = MagicMock()
    mock_ollama_provider_instance.validate_model = MagicMock(return_value=None)
    mock_ollama_provider_instance.get_chat_completion = AsyncMock(
        return_value={"response": "mocked chat completion"}
    )
    mock_ollama_provider_instance.get_completion = AsyncMock(
        return_value={"response": "mocked completion"}
    )
    mock_ollama_provider_instance.name = "mock-ollama"
    mock_ollama_provider_instance.model = "mock:latest"

    # Make the mocked OllamaModelProvider class return our mocked instance
    mock_ollama_provider_class.return_value = mock_ollama_provider_instance

    # Create mocks for ReactAgentBuilder and ReactAgent
    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value={"result": "test_result"})
    mock_agent.close = AsyncMock()

    mock_builder_instance = MagicMock()
    mock_builder_instance.with_model.return_value = mock_builder_instance
    mock_builder_instance.with_mcp_tools.return_value = mock_builder_instance
    mock_builder_instance.build = AsyncMock(return_value=mock_agent)

    # Make the class return our mocked instance
    mock_builder_class.return_value = mock_builder_instance

    # Call the function
    result = await quick_create_agent(
        task="Test task",
        model="ollama:test:model",
        tools=["time", "brave-search"],
        interactive=True,
    )

    # Verify builder was configured correctly
    mock_builder_class.assert_called_once()
    mock_builder_instance.with_model.assert_called_once_with("ollama:test:model")
    mock_builder_instance.with_mcp_tools.assert_called_once_with(
        ["time", "brave-search"]
    )
    mock_builder_instance.build.assert_called_once()

    # Verify run and close were called
    mock_agent.run.assert_called_once_with(initial_task="Test task")
    mock_agent.close.assert_called_once()

    # Verify result
    assert result == {"result": "test_result"}


# Pydantic Models Tests
def test_log_level_enum():
    """Test the LogLevel enum"""
    assert LogLevel.DEBUG.value == "debug"
    assert LogLevel.INFO.value == "info"
    assert LogLevel.WARNING.value == "warning"
    assert LogLevel.ERROR.value == "error"


def test_tool_config_model():
    """Test the ToolConfig Pydantic model"""
    # Create a basic config
    config = ToolConfig(name="test_tool")
    assert config.name == "test_tool"
    assert config.is_custom is False
    assert config.description is None
    assert config.source == "unknown"

    # Create a full config
    full_config = ToolConfig(
        name="full_tool", is_custom=True, description="A test tool", source="custom"
    )
    assert full_config.name == "full_tool"
    assert full_config.is_custom is True
    assert full_config.description == "A test tool"
    assert full_config.source == "custom"


def test_confirmation_config_model():
    """Test the ConfirmationConfig Pydantic model"""
    # Create a basic config
    config = ConfirmationConfig()
    assert config.enabled is True
    assert config.strategy == "always"
    assert len(config.excluded_tools) == 0
    assert config.included_tools is None
    assert len(config.allowed_silent_tools) == 0
    assert config.timeout is None

    # Create a full config
    full_config = ConfirmationConfig(
        enabled=False,
        strategy="selective",
        excluded_tools=["tool1", "tool2"],
        included_tools=["tool3"],
        allowed_silent_tools=["tool4"],
        timeout=30.0,
    )
    assert full_config.enabled is False
    assert full_config.strategy == "selective"
    assert full_config.excluded_tools == ["tool1", "tool2"]
    assert full_config.included_tools == ["tool3"]
    assert full_config.allowed_silent_tools == ["tool4"]
    assert full_config.timeout == 30.0

    # Test dict conversion
    config_dict = full_config.dict()
    assert isinstance(config_dict, dict)
    assert config_dict["enabled"] is False
    assert config_dict["strategy"] == "selective"
    assert config_dict["excluded_tools"] == ["tool1", "tool2"]
