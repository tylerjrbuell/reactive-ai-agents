"""
Global pytest configuration and fixtures.

This file configures pytest behavior for all tests in the project.
"""

import pytest
import os
from unittest.mock import patch

# Configure asyncio for all tests
pytest_plugins = ["pytest_asyncio"]


# Register custom pytest marks to avoid warnings
def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "providers: mark test as provider test")
    config.addinivalue_line("markers", "slow: mark test as slow running test")


@pytest.fixture(scope="session", autouse=True)
def mock_docker_environment():
    """
    Global fixture to ensure Docker operations are mocked for all tests.
    """
    # Only mock if we're in a test environment that requires it
    if (
        os.environ.get("DISABLE_MCP_CLIENT_SYSTEM_EXIT") == "1"
        or os.environ.get("MOCK_MCP_CLIENT") == "1"
        or os.environ.get("CI") == "true"
    ):

        # Set environment variables for test mode
        env_patch = patch.dict(
            "os.environ",
            {"CI": "true", "MOCK_MCP_CLIENT": "true", "NO_DOCKER": "1"},
            clear=False,
        )

        # Apply patches
        env_patch.start()

        yield
    else:
        yield


# Add the model_validation_bypass fixture here
@pytest.fixture
def model_validation_bypass():
    """
    Fixture that bypasses model validation in OllamaModelProvider
    """
    with patch(
        "reactive_agents.providers.llm.ollama.OllamaModelProvider.validate_model"
    ) as mock_validate:
        # Make validate_model a no-op
        mock_validate.return_value = None
        yield mock_validate
