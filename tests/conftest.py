import pytest
from agent_mcp.client import MCPClient


@pytest.fixture
async def mock_mcp_client():
    """Fixture to create a mock MCP client for testing"""
    client = MCPClient(server_filter=["local"])
    await client.initialize()
    yield client
    await client.close()
