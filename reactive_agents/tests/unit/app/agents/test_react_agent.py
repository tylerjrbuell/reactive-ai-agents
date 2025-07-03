import pytest
from unittest.mock import patch
from reactive_agents.app.agents.react_agent import ReactAgent, ReactAgentConfig
from reactive_agents.tests.mocks import MockModelProvider, MockMCPClient


@pytest.fixture
async def mock_mcp_client():
    """Fixture to create a mock MCP client for testing"""
    client = MockMCPClient(server_filter=["local"])
    await client.initialize()
    yield client
    await client.close()


@pytest.mark.asyncio
@patch(
    "reactive_agents.model_providers.factory.ModelProviderFactory.get_model_provider"
)
async def test_react_agent_initialization(mock_factory, mock_mcp_client):
    """Test basic ReactAgent initialization"""
    mock_provider = MockModelProvider(name="mock", model="mock:latest")
    mock_factory.return_value = mock_provider

    config = ReactAgentConfig(
        agent_name="TestAgent",
        provider_model_name="mock:latest",
        mcp_client=mock_mcp_client,
        instructions="Test instructions",
        min_completion_score=1.0,
        max_iterations=3,
        reflect_enabled=True,
    )

    agent = ReactAgent(config=config)

    assert agent.context.agent_name == "TestAgent"
    assert agent.context.instructions == "Test instructions"
    assert agent.context.min_completion_score == 1.0
    assert agent.context.max_iterations == 3
    assert agent.context.reflect_enabled is True


@pytest.mark.asyncio
@patch(
    "reactive_agents.model_providers.factory.ModelProviderFactory.get_model_provider"
)
async def test_agent_reflection_initialization(mock_factory):
    """Test that agent reflections are properly initialized"""
    mock_provider = MockModelProvider(name="mock", model="mock:latest")
    mock_factory.return_value = mock_provider

    config = ReactAgentConfig(
        agent_name="ReflectAgent",
        provider_model_name="mock:latest",
        instructions="Test reflection",
        reflect_enabled=True,
    )

    agent = ReactAgent(config=config)

    assert hasattr(agent.context, "state_observer")
    assert agent.context.state_observer is not None
    assert isinstance(agent.context.get_reflections(), list)
    assert len(agent.context.get_reflections()) == 0


@pytest.mark.asyncio
@patch(
    "reactive_agents.model_providers.factory.ModelProviderFactory.get_model_provider"
)
async def test_agent_workflow_context(mock_factory):
    """Test agent with workflow context"""
    mock_provider = MockModelProvider(name="mock", model="mock:latest")
    mock_factory.return_value = mock_provider

    workflow_context_data = {
        "test_agent": {"status": "pending", "reflections": [], "current_progress": ""}
    }

    config = ReactAgentConfig(
        agent_name="test_agent",
        provider_model_name="mock:latest",
        instructions="Test workflow",
        workflow_context_shared=workflow_context_data,
    )

    agent = ReactAgent(config=config)

    assert agent.context.workflow_context_shared is not None
    assert agent.context.workflow_context_shared == workflow_context_data


@pytest.mark.asyncio
@patch(
    "reactive_agents.model_providers.factory.ModelProviderFactory.get_model_provider"
)
async def test_agent_task_execution(mock_factory, mock_mcp_client):
    """Test agent task execution with mocked dependencies"""
    mock_provider = MockModelProvider(name="mock", model="mock:latest")
    mock_factory.return_value = mock_provider

    config = ReactAgentConfig(
        agent_name="TestAgent",
        provider_model_name="mock:latest",
        mcp_client=mock_mcp_client,
        instructions="Test task execution",
        min_completion_score=1.0,
        max_iterations=1,
    )

    agent = ReactAgent(config=config)

    result = await agent.run("Test task")
    assert result is not None
    mock_provider.get_chat_completion.assert_called_once()
