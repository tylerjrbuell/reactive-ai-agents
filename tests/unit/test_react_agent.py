import pytest
from agents.react_agent import ReactAgent
from model_providers.factory import ModelProviderFactory
from agent_mcp.client import MCPClient


@pytest.fixture
async def mock_mcp_client():
    """Fixture to create a mock MCP client for testing"""
    client = MCPClient(server_filter=["local"])
    await client.initialize()
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_react_agent_initialization(mock_mcp_client):
    """Test basic ReactAgent initialization"""
    agent = ReactAgent(
        name="TestAgent",
        provider_model="ollama:cogito:14b",
        mcp_client=mock_mcp_client,
        instructions="Test instructions",
        min_completion_score=1.0,
        max_iterations=3,
        reflect=True,
    )

    assert agent.name == "TestAgent"
    assert agent.model_provider.name == "ollama"
    assert agent.model_provider.model == "cogito:14b"
    assert agent.instructions == "Test instructions"
    assert agent.min_completion_score == 1.0
    assert agent.max_iterations == 3
    assert agent.reflect is True


@pytest.mark.asyncio
async def test_agent_reflection_initialization():
    """Test that agent reflections are properly initialized"""
    agent = ReactAgent(
        name="ReflectAgent",
        provider_model="ollama:cogito:14b",
        instructions="Test reflection",
        reflect=True,
    )

    assert hasattr(agent, "reflections")
    assert isinstance(agent.reflections, list)
    assert len(agent.reflections) == 0


@pytest.mark.asyncio
async def test_agent_workflow_context():
    """Test agent with workflow context"""
    workflow_context = {
        "test_agent": {"status": "pending", "reflections": [], "current_progress": ""}
    }

    agent = ReactAgent(
        name="test_agent",
        provider_model="ollama:cogito:14b",
        instructions="Test workflow",
        workflow_context=workflow_context,
    )

    assert agent.workflow_context == workflow_context
    assert agent.workflow_context["test_agent"]["status"] == "pending"  # type: ignore
