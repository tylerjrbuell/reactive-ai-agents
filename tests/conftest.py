"""
Shared test fixtures and configuration
"""

import pytest
from tests.mocks import MockMCPClient


@pytest.fixture
async def mock_mcp_client():
    """Fixture to create a mock MCP client for testing"""
    client = MockMCPClient(server_filter=["local"])
    await client.initialize()
    yield client
    await client.close()
