"""
Integration test-specific pytest configuration.

This conftest.py file applies additional mocking specifically for integration tests
that might attempt to use Docker.
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock
from reactive_agents.tests.integration.mcp_fixtures import SimpleMockMCPClient

# Import OllamaModelProvider for type hinting, not direct use here
from reactive_agents.providers.llm.ollama import OllamaModelProvider


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
        mock_popen.communicate = AsyncMock(return_value=(b"", b""))
        mock_popen.returncode = 0
        mock_popen.poll = MagicMock(return_value=0)
        mock_popen.wait = AsyncMock(return_value=0)
        mock_subprocess.Popen.return_value = mock_popen

        # --- Patch and Configure OllamaModelProvider Separately ---
        ollama_patcher = patch(
            "reactive_agents.providers.llm.ollama.OllamaModelProvider",
            new_callable=MagicMock,  # Use MagicMock as new_callable
        )
        # Start the ollama patcher immediately to get the mocked class
        MockOllamaProviderClass = ollama_patcher.start()

        # Configure the returned mock instance
        MockOllamaProviderClass.return_value.validate_model = MagicMock(
            return_value=None
        )
        MockOllamaProviderClass.return_value.get_chat_completion = AsyncMock(
            return_value={"response": "mocked chat completion"}
        )
        MockOllamaProviderClass.return_value.get_completion = AsyncMock(
            return_value={"response": "mocked completion"}
        )
        MockOllamaProviderClass.return_value.name = "mock-ollama"
        MockOllamaProviderClass.return_value.model = "mock:latest"

        # Add the ollama patcher to the list of patches to be stopped
        patches.append(ollama_patcher)
        # --- End OllamaModelProvider Patching ---

        # Apply critical patches (excluding Ollama which is handled above)
        patches.extend(
            [
                patch(
                    "reactive_agents.providers.external.client.MCPClient",
                    SimpleMockMCPClient,
                ),
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

        # Start all other patches
        for p in patches:
            # Skip the ollama_patcher as it's already started
            if p == ollama_patcher:
                continue
            p.start()

        # Modify sys.modules to prevent Docker-related imports
        if "docker" in sys.modules:
            sys.modules["docker"] = MagicMock()

        print(
            "Integration test mocks applied - Docker operations disabled in test cases"
        )

        yield  # Yield control to the test function

        # Stop all patches during teardown
        for p in patches:
            p.stop()

    else:
        yield
