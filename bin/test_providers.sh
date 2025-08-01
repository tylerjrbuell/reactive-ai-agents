#!/bin/bash

# Reactive Agents Provider Testing Script
# Convenient wrapper for the provider testing system

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Path to the actual test runner
TEST_RUNNER="$PROJECT_ROOT/reactive_agents/tests/integration/run_provider_tests.py"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 Reactive Agents Provider Testing System${NC}"
echo -e "${GREEN}Running from: $TEST_RUNNER${NC}"
echo ""

# Check if the test runner exists
if [ ! -f "$TEST_RUNNER" ]; then
    echo "❌ Error: Test runner not found at $TEST_RUNNER"
    exit 1
fi

# Change to project root directory
cd "$PROJECT_ROOT"

# Run the test runner with all provided arguments
python "$TEST_RUNNER" "$@"