"""
Integration test-specific pytest configuration.

This conftest.py file applies additional mocking specifically for integration tests
that might attempt to use Docker.
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock
from tests.integration.mcp_fixtures import SimpleMockMCPClient


@pytest.fixture(scope="session", autouse=True)
def integration_mock_environment():
    """
    Global fixture specifically for integration tests to prevent Docker usage.
    """
    # Detect CI environment
    in_ci = (
        os.environ.get("DISABLE_MCP_CLIENT_SYSTEM_EXIT") == "1"
        or os.environ.get("MOCK_MCP_CLIENT") == "1"
        or os.environ.get("CI") == "true"
        or os.environ.get("NO_DOCKER") == "1"
    )

    if in_ci:
        patches = []

        # More aggressive patching for integration tests
        mock_subprocess = MagicMock()
        mock_popen = MagicMock()
        mock_popen.communicate = MagicMock(return_value=(b"", b""))
        mock_popen.returncode = 0
        mock_popen.poll = MagicMock(return_value=0)
        mock_popen.wait = MagicMock(return_value=0)
        mock_subprocess.Popen.return_value = mock_popen

        # Apply critical patches
        patches.extend(
            [
                patch("agent_mcp.client.MCPClient", SimpleMockMCPClient),
                patch("subprocess.Popen", mock_subprocess.Popen),
                patch(
                    "os.path.exists", lambda path: True
                ),  # Make all file checks succeed
                patch.dict(
                    "os.environ",
                    {"NO_DOCKER": "1", "CI": "true", "MOCK_MCP_CLIENT": "1"},
                ),
            ]
        )

        # Start all patches
        for p in patches:
            p.start()

        # Modify sys.modules to prevent Docker-related imports
        if "docker" in sys.modules:
            sys.modules["docker"] = MagicMock()

        print(
            "Integration test mocks applied - Docker operations disabled in test cases"
        )

        yield
    else:
        yield
