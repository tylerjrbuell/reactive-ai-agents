import pytest
from unittest.mock import patch
from agents.react_agent import ReactAgent
from tests.mocks import MockModelProvider, MockMCPClient


@pytest.fixture
async def mock_mcp_client():
    """Fixture to create a mock MCP client for testing"""
    client = MockMCPClient(server_filter=["local"])
    await client.initialize()
    yield client
    await client.close()


@pytest.mark.asyncio
@patch("model_providers.factory.ModelProviderFactory.get_model_provider")
async def test_react_agent_initialization(mock_factory, mock_mcp_client):
    """Test basic ReactAgent initialization"""
    mock_provider = MockModelProvider(name="mock", model="mock:latest")
    mock_factory.return_value = mock_provider

    agent = ReactAgent(
        name="TestAgent",
        provider_model="mock:latest",
        mcp_client=mock_mcp_client,
        instructions="Test instructions",
        min_completion_score=1.0,
        max_iterations=3,
        reflect=True,
    )

    assert agent.name == "TestAgent"
    assert agent.model_provider.name == "mock"
    assert agent.model_provider.model == "mock:latest"
    assert agent.instructions == "Test instructions"
    assert agent.min_completion_score == 1.0
    assert agent.max_iterations == 3
    assert agent.reflect is True


@pytest.mark.asyncio
@patch("model_providers.factory.ModelProviderFactory.get_model_provider")
async def test_agent_reflection_initialization(mock_factory):
    """Test that agent reflections are properly initialized"""
    mock_provider = MockModelProvider(name="mock", model="mock:latest")
    mock_factory.return_value = mock_provider

    agent = ReactAgent(
        name="ReflectAgent",
        provider_model="mock:latest",
        instructions="Test reflection",
        reflect=True,
    )

    assert hasattr(agent, "reflections")
    assert isinstance(agent.reflections, list)
    assert len(agent.reflections) == 0


@pytest.mark.asyncio
@patch("model_providers.factory.ModelProviderFactory.get_model_provider")
async def test_agent_workflow_context(mock_factory):
    """Test agent with workflow context"""
    mock_provider = MockModelProvider(name="mock", model="mock:latest")
    mock_factory.return_value = mock_provider

    workflow_context = {
        "test_agent": {"status": "pending", "reflections": [], "current_progress": ""}
    }

    agent = ReactAgent(
        name="test_agent",
        provider_model="mock:latest",
        instructions="Test workflow",
        workflow_context=workflow_context,
    )

    assert agent.workflow_context == workflow_context
    assert agent.workflow_context["test_agent"]["status"] == "pending"  # type: ignore


@pytest.mark.asyncio
@patch("model_providers.factory.ModelProviderFactory.get_model_provider")
async def test_agent_task_execution(mock_factory, mock_mcp_client):
    """Test agent task execution with mocked dependencies"""
    mock_provider = MockModelProvider(name="mock", model="mock:latest")
    mock_factory.return_value = mock_provider

    agent = ReactAgent(
        name="TestAgent",
        provider_model="mock:latest",
        mcp_client=mock_mcp_client,
        instructions="Test task execution",
        min_completion_score=1.0,
        max_iterations=1,
    )

    result = await agent.run("Test task")
    assert result is not None
    mock_provider.get_chat_completion.assert_called_once()
