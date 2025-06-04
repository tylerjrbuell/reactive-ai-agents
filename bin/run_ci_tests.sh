#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Set environment variables to disable potentially problematic components, mirroring the CI workflow
export DISABLE_MCP_CLIENT_SYSTEM_EXIT=1
export PYTEST_TIMEOUT=60
export NO_DOCKER=1
export MOCK_ALL_TOOLS=1
export MOCK_MCP_CLIENT=1
export SKIP_DOCKER_CHECKS=1

echo "Environment variables set for CI simulation."

# Create an explicit mock script to run before pytest, mirroring the CI workflow
echo "import reactive_agents.tests.ci_mock; print('CI mocks loaded successfully')" > ci_pytest_loader.py
echo "Created ci_pytest_loader.py"

# Run tests with CI-specific mock loader and pytest flags
echo "Running tests..."
# Use && to ensure pytest only runs if the loader script succeeds
poetry run python ci_pytest_loader.py && poetry run pytest -v --cov=./ --cov-report=xml --timeout=60 --no-header --tb=native -p no:docker

echo "CI tests finished."

# You may want to clean up the loader file afterwards, but keeping it can be useful for debugging
# rm ci_pytest_loader.py 