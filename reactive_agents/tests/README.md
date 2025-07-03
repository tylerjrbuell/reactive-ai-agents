# Test Suite Documentation

This directory contains the comprehensive test suite for the reactive-agents framework. The tests are organized to mirror the framework's architecture and make it easy to add tests as development progresses.

## Test Structure

```
tests/
├── unit/                          # Unit tests
│   ├── core/                      # Core framework tests
│   │   ├── engine/                # Execution engine tests
│   │   ├── tools/                 # Tool management tests
│   │   ├── reasoning/             # Reasoning and reflection tests
│   │   ├── events/                # Event system tests
│   │   ├── memory/                # Memory management tests
│   │   ├── workflows/             # Workflow management tests
│   │   └── types/                 # Type definition tests
│   ├── app/                       # Application layer tests
│   │   ├── agents/                # Agent implementation tests
│   │   ├── builders/              # Agent builder tests
│   │   ├── communication/         # Communication protocol tests
│   │   └── workflows/             # App workflow tests
│   ├── providers/                 # Provider tests
│   │   ├── llm/                   # LLM provider tests
│   │   ├── storage/               # Storage provider tests
│   │   └── external/              # External integration tests
│   ├── console/                   # Console/CLI tests
│   │   ├── commands/              # Command tests
│   │   └── output/                # Output formatting tests
│   └── plugins/                   # Plugin system tests
├── integration/                   # Integration tests
│   ├── conftest.py               # Integration test configuration
│   ├── mcp_fixtures.py           # MCP test fixtures
│   ├── mock_mcp.py               # MCP mocking utilities
│   ├── test_builder_integration.py
│   ├── test_custom_tools.py
│   └── test_mcp_tools.py
├── conftest.py                   # Global test configuration
├── mocks.py                      # Global mock utilities
├── ci_mock.py                    # CI-specific mocks
└── README.md                     # This file
```

## Running Tests

### Quick Start

```bash
# Run all tests
./bin/run_tests.sh

# Run unit tests only
./bin/run_tests.sh --category unit

# Run integration tests only
./bin/run_tests.sh --category integration

# Run with coverage
./bin/run_tests.sh --coverage

# Run tests in parallel
./bin/run_tests.sh --parallel
```

### Test Categories

#### Core Framework Tests

```bash
# All core tests
./bin/run_tests.sh --category core

# Specific core components
./bin/run_tests.sh --category engine
./bin/run_tests.sh --category tools
./bin/run_tests.sh --category reasoning
./bin/run_tests.sh --category events
./bin/run_tests.sh --category memory
./bin/run_tests.sh --category workflows
./bin/run_tests.sh --category types
```

#### Application Layer Tests

```bash
# All app tests
./bin/run_tests.sh --category app

# Specific app components
./bin/run_tests.sh --category agents
./bin/run_tests.sh --category builders
./bin/run_tests.sh --category communication
```

#### Provider Tests

```bash
# All provider tests
./bin/run_tests.sh --category providers

# Specific provider types
./bin/run_tests.sh --category llm
./bin/run_tests.sh --category storage
./bin/run_tests.sh --category external
```

#### Console/CLI Tests

```bash
# All console tests
./bin/run_tests.sh --category console

# Specific console components
./bin/run_tests.sh --category commands
./bin/run_tests.sh --category output
```

### Direct pytest Commands

```bash
# Run specific test file
pytest reactive_agents/tests/unit/core/engine/test_execution_engine.py

# Run tests with specific marker
pytest -m "engine"

# Run tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=reactive_agents --cov-report=html

# Run tests in parallel
pytest -n auto
```

## Adding New Tests

### Template Structure

Each test file follows this template structure:

```python
"""
Tests for [Component Name].

Tests the [component] functionality including [key features].
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from reactive_agents.[module.path] import [ComponentClass]


class Test[ComponentClass]:
    """Test cases for [ComponentClass]."""

    @pytest.fixture
    def [component_instance](self):
        """Create a [component] instance."""
        return [ComponentClass]()

    def test_initialization(self, [component_instance]):
        """Test [component] initialization."""
        # TODO: Implement test for initialization
        pass

    def test_[feature_name](self, [component_instance]):
        """Test [feature description]."""
        # TODO: Implement test for [feature]
        pass
```

### Test Guidelines

1. **Naming Convention**: Test files should be named `test_[component_name].py`
2. **Class Naming**: Test classes should be named `Test[ComponentClass]`
3. **Method Naming**: Test methods should be named `test_[feature_description]`
4. **Documentation**: Each test should have a clear docstring explaining what it tests
5. **Fixtures**: Use pytest fixtures for common setup
6. **Mocking**: Use `unittest.mock` for external dependencies
7. **Markers**: Use appropriate pytest markers for categorization

### Example Test Implementation

```python
def test_tool_registration(self, tool_manager, mock_tool):
    """Test tool registration."""
    # Arrange
    tool_name = "test_tool"

    # Act
    tool_manager.register_tool(mock_tool)

    # Assert
    assert tool_name in tool_manager.tools
    assert tool_manager.tools[tool_name] == mock_tool
```

## Test Configuration

### pytest.ini

The `pytest.ini` file contains:

- Test discovery paths
- Default options (verbose, colors, etc.)
- Custom markers for test categorization
- Test file and function naming patterns

### conftest.py

The `conftest.py` file contains:

- Global fixtures shared across tests
- Test configuration setup
- Common mock objects and utilities

### Integration Test Configuration

The `integration/conftest.py` file contains:

- Integration-specific fixtures
- External service mocking
- Test data setup

## Test Coverage

### Coverage Goals

- **Unit Tests**: 80%+ coverage for core components
- **Integration Tests**: Critical path coverage
- **Edge Cases**: Error handling and boundary conditions

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=reactive_agents --cov-report=html

# View coverage in terminal
pytest --cov=reactive_agents --cov-report=term-missing
```

## Continuous Integration

### CI Pipeline

The test suite is integrated into the CI pipeline:

- Runs on every pull request
- Runs unit tests and integration tests
- Generates coverage reports
- Fails build on test failures

### Local Development

```bash
# Run tests before committing
./bin/run_tests.sh --category unit

# Run full test suite
./bin/run_tests.sh --coverage
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the module path is correct
2. **Mock Issues**: Check that mocks are properly configured
3. **Async Tests**: Use `pytest-asyncio` for async test functions
4. **Fixture Errors**: Verify fixture dependencies are correct

### Debug Mode

```bash
# Run tests with debug output
pytest -v -s --tb=long

# Run specific test with debug
pytest -v -s test_file.py::TestClass::test_method
```

## Contributing

When adding new tests:

1. Follow the template structure
2. Use appropriate markers
3. Add comprehensive docstrings
4. Test both success and failure cases
5. Mock external dependencies
6. Update this README if adding new test categories

## Test Dependencies

- `pytest`: Test framework
- `pytest-asyncio`: Async test support
- `pytest-cov`: Coverage reporting
- `pytest-xdist`: Parallel test execution
- `unittest.mock`: Mocking utilities
