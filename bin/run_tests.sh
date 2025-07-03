#!/bin/bash

# Test runner script for reactive-agents framework
# Usage: ./bin/run_tests.sh [category] [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CATEGORY="all"
VERBOSE=""
COVERAGE=""
PARALLEL=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --category|-c)
            CATEGORY="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        --coverage)
            COVERAGE="--cov=reactive_agents --cov-report=html --cov-report=term-missing"
            shift
            ;;
        --parallel|-p)
            PARALLEL="-n auto"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -c, --category CATEGORY    Test category to run (default: all)"
            echo "  -v, --verbose              Verbose output"
            echo "  --coverage                 Run with coverage reporting"
            echo "  -p, --parallel             Run tests in parallel"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Categories:"
            echo "  all                        Run all tests"
            echo "  unit                       Run unit tests only"
            echo "  integration                Run integration tests only"
            echo "  core                       Run core framework tests"
            echo "  app                        Run application layer tests"
            echo "  providers                  Run provider tests"
            echo "  console                    Run console/CLI tests"
            echo "  plugins                    Run plugin tests"
            echo "  engine                     Run engine tests"
            echo "  tools                      Run tools tests"
            echo "  reasoning                  Run reasoning tests"
            echo "  events                     Run events tests"
            echo "  memory                     Run memory tests"
            echo "  workflows                  Run workflow tests"
            echo "  types                      Run type tests"
            echo "  agents                     Run agent tests"
            echo "  builders                   Run builder tests"
            echo "  communication              Run communication tests"
            echo "  llm                        Run LLM provider tests"
            echo "  storage                    Run storage provider tests"
            echo "  external                   Run external provider tests"
            echo "  commands                   Run command tests"
            echo "  output                     Run output tests"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    print_error "Must be run from the project root directory"
    exit 1
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_error "pytest is not installed. Please install it first."
    exit 1
fi

# Build the pytest command
PYTEST_CMD="pytest $VERBOSE $COVERAGE $PARALLEL"

case $CATEGORY in
    "all")
        print_status "Running all tests..."
        $PYTEST_CMD
        ;;
    "unit")
        print_status "Running unit tests..."
        $PYTEST_CMD reactive_agents/tests/unit/
        ;;
    "integration")
        print_status "Running integration tests..."
        $PYTEST_CMD reactive_agents/tests/integration/
        ;;
    "core")
        print_status "Running core framework tests..."
        $PYTEST_CMD reactive_agents/tests/unit/core/ -m "core"
        ;;
    "app")
        print_status "Running application layer tests..."
        $PYTEST_CMD reactive_agents/tests/unit/app/ -m "app"
        ;;
    "providers")
        print_status "Running provider tests..."
        $PYTEST_CMD reactive_agents/tests/unit/providers/ -m "providers"
        ;;
    "console")
        print_status "Running console/CLI tests..."
        $PYTEST_CMD reactive_agents/tests/unit/console/ -m "console"
        ;;
    "plugins")
        print_status "Running plugin tests..."
        $PYTEST_CMD reactive_agents/tests/unit/plugins/ -m "plugins"
        ;;
    "engine")
        print_status "Running engine tests..."
        $PYTEST_CMD reactive_agents/tests/unit/core/engine/ -m "engine"
        ;;
    "tools")
        print_status "Running tools tests..."
        $PYTEST_CMD reactive_agents/tests/unit/core/tools/ -m "tools"
        ;;
    "reasoning")
        print_status "Running reasoning tests..."
        $PYTEST_CMD reactive_agents/tests/unit/core/reasoning/ -m "reasoning"
        ;;
    "events")
        print_status "Running events tests..."
        $PYTEST_CMD reactive_agents/tests/unit/core/events/ -m "events"
        ;;
    "memory")
        print_status "Running memory tests..."
        $PYTEST_CMD reactive_agents/tests/unit/core/memory/ -m "memory"
        ;;
    "workflows")
        print_status "Running workflow tests..."
        $PYTEST_CMD reactive_agents/tests/unit/core/workflows/ -m "workflows"
        ;;
    "types")
        print_status "Running type tests..."
        $PYTEST_CMD reactive_agents/tests/unit/core/types/ -m "types"
        ;;
    "agents")
        print_status "Running agent tests..."
        $PYTEST_CMD reactive_agents/tests/unit/app/agents/ -m "agents"
        ;;
    "builders")
        print_status "Running builder tests..."
        $PYTEST_CMD reactive_agents/tests/unit/app/builders/ -m "builders"
        ;;
    "communication")
        print_status "Running communication tests..."
        $PYTEST_CMD reactive_agents/tests/unit/app/communication/ -m "communication"
        ;;
    "llm")
        print_status "Running LLM provider tests..."
        $PYTEST_CMD reactive_agents/tests/unit/providers/llm/ -m "llm"
        ;;
    "storage")
        print_status "Running storage provider tests..."
        $PYTEST_CMD reactive_agents/tests/unit/providers/storage/ -m "storage"
        ;;
    "external")
        print_status "Running external provider tests..."
        $PYTEST_CMD reactive_agents/tests/unit/providers/external/ -m "external"
        ;;
    "commands")
        print_status "Running command tests..."
        $PYTEST_CMD reactive_agents/tests/unit/console/commands/ -m "commands"
        ;;
    "output")
        print_status "Running output tests..."
        $PYTEST_CMD reactive_agents/tests/unit/console/output/ -m "output"
        ;;
    *)
        print_error "Unknown category: $CATEGORY"
        echo "Use --help to see available categories"
        exit 1
        ;;
esac

if [[ $? -eq 0 ]]; then
    print_success "Tests completed successfully!"
else
    print_error "Tests failed!"
    exit 1
fi 