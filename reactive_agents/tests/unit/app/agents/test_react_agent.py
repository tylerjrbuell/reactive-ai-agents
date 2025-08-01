import pytest
from unittest.mock import patch, MagicMock
from reactive_agents.app.agents.reactive_agent import ReactiveAgent
from reactive_agents.app.agents.base import Agent
from reactive_agents.core.types.agent_types import ReactiveAgentConfig
from reactive_agents.tests.mocks import MockModelProvider, MockMCPClient
from reactive_agents.providers.llm.base import BaseModelProvider


@pytest.fixture
def mock_mcp_client():
    return MockMCPClient()


@pytest.mark.asyncio
@patch("reactive_agents.providers.llm.factory.ModelProviderFactory.get_model_provider")
async def test_react_agent_initialization(mock_factory, mock_mcp_client):
    """Test reactive agent initialization"""
    mock_provider = MockModelProvider(model="mock:latest")
    mock_factory.return_value = mock_provider

    config = ReactiveAgentConfig(
        agent_name="TestAgent",
        provider_model_name="mock:latest",
        mcp_client=mock_mcp_client,
        instructions="Test instructions",
        min_completion_score=1.0,
        max_iterations=3,
        reflect_enabled=True,
    )

    agent = ReactiveAgent(config=config)

    assert agent.context.agent_name == "TestAgent"
    assert agent.context.instructions == "Test instructions"
    assert agent.context.min_completion_score == 1.0
    assert agent.context.max_iterations == 3
    assert agent.context.reflect_enabled is True


@pytest.mark.asyncio
@patch("reactive_agents.providers.llm.factory.ModelProviderFactory.get_model_provider")
async def test_agent_reflection_initialization(mock_factory):
    """Test that agent reflections are properly initialized"""
    mock_provider = MockModelProvider(model="mock:latest")
    mock_factory.return_value = mock_provider

    config = ReactiveAgentConfig(
        agent_name="ReflectAgent",
        provider_model_name="mock:latest",
        instructions="Test reflection",
        reflect_enabled=True,
    )

    agent = ReactiveAgent(config=config)

    # Test that reflection is enabled
    assert agent.context.reflect_enabled is True

    # Test that reflection methods exist and work
    assert hasattr(agent.context, "get_reflections")
    assert isinstance(agent.context.get_reflections(), list)
    assert len(agent.context.get_reflections()) == 0

    # Test that memory manager exists for reflection
    assert hasattr(agent.context, "memory_manager")
    assert agent.context.memory_manager is not None


@pytest.mark.asyncio
@patch("reactive_agents.providers.llm.factory.ModelProviderFactory.get_model_provider")
async def test_agent_workflow_context(mock_factory):
    """Test agent with workflow context"""
    mock_provider = MockModelProvider(model="mock:latest")
    mock_factory.return_value = mock_provider

    workflow_context_data = {
        "test_agent": {"status": "pending", "reflections": [], "current_progress": ""}
    }

    config = ReactiveAgentConfig(
        agent_name="test_agent",
        provider_model_name="mock:latest",
        instructions="Test workflow",
        workflow_context_shared=workflow_context_data,
    )

    agent = ReactiveAgent(config=config)

    assert agent.context.workflow_context_shared is not None
    assert agent.context.workflow_context_shared == workflow_context_data


@pytest.mark.asyncio
async def test_agent_task_execution(mock_mcp_client):
    """
    Test basic agent task execution functionality.

    This test verifies that the agent can run a task and return a result.
    For detailed model provider testing, see test_react_agent_isolated.py
    """
    config = ReactiveAgentConfig(
        agent_name="TestAgent",
        provider_model_name="mock:latest",
        mcp_client=mock_mcp_client,
        instructions="Test task execution",
        min_completion_score=0.5,
        max_iterations=3,
    )

    # Create agent with config
    agent = ReactiveAgent(config=config)

    try:
        result = await agent.run("Test task")
        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "final_answer")
    finally:
        # Ensure proper cleanup
        if hasattr(agent, "close"):
            await agent.close()
        if hasattr(agent.context, "close"):
            await agent.context.close()
