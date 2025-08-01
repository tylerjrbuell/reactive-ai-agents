"""
CI-specific mock utilities for testing without Docker.

This module is loaded as early as possible in CI environments to completely
prevent Docker operations at a lower level
"""

import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock


# Function to create a mock version of a module if it doesn't exist
def mock_import(name):
    """Create a mock module to prevent imports from failing"""
    if name not in sys.modules:
        sys.modules[name] = MagicMock()
    return sys.modules[name]


# Early patching applied as soon as this module is imported
def apply_ci_mocks():
    """
    Apply aggressive mocking for CI environments

    This function applies patches at the lowest possible level to completely
    prevent Docker operations.
    """
    patches = []

    # Create a fake subprocess module that prevents command execution
    mock_subprocess = MagicMock()
    mock_subprocess.Popen = MagicMock(return_value=MagicMock())
    mock_subprocess.run = MagicMock(return_value=MagicMock(returncode=0))
    mock_subprocess.check_output = MagicMock(return_value=b"")
    mock_subprocess.PIPE = MagicMock()
    mock_subprocess.CalledProcessError = Exception

    # Create a fake shutil module
    mock_shutil = MagicMock()
    mock_shutil.which = MagicMock(return_value="/usr/bin/docker")

    # Apply patches
    patches.extend(
        [
            patch("subprocess.Popen", mock_subprocess.Popen),
            patch("subprocess.run", mock_subprocess.run),
            patch("subprocess.check_output", mock_subprocess.check_output),
            patch("subprocess.PIPE", mock_subprocess.PIPE),
            patch("shutil.which", mock_shutil.which),
        ]
    )

    # Overwrite the entire MCPClient implementation
    from reactive_agents.tests.integration.mcp_fixtures import SimpleMockMCPClient

    patch_mcp = patch(
        "reactive_agents.providers.external.client.MCPClient", SimpleMockMCPClient
    )
    patches.append(patch_mcp)

    # Apply all patches
    for p in patches:
        p.start()

    return patches


# Apply mocks when in CI environment
if (
    os.environ.get("DISABLE_MCP_CLIENT_SYSTEM_EXIT") == "1"
    or os.environ.get("MOCK_MCP_CLIENT") == "1"
    or os.environ.get("CI") == "true"
    or os.environ.get("NO_DOCKER") == "1"
):

    # Set environment variables that might influence Docker behavior
    os.environ["DISABLE_MCP_CLIENT_SYSTEM_EXIT"] = "1"
    os.environ["MOCK_MCP_CLIENT"] = "1"
    os.environ["NO_DOCKER"] = "1"
    os.environ["CI"] = "true"

    # Apply the aggressive mocking
    applied_patches = apply_ci_mocks()
    print("CI mocking applied - Docker operations disabled")
