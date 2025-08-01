# Integration Tests

This directory contains integration tests for the Reactive Agents framework, including comprehensive provider testing systems.

## Provider Testing System

The provider testing system ensures consistent behavior and performance across all supported LLM providers.

### Quick Start

```bash
# Run from project root

# Quick consistency test (mocked)
./bin/test_providers.sh --consistency

# Test specific providers
./bin/test_providers.sh --consistency --providers ollama openai

# Performance benchmarking
./bin/test_providers.sh --performance

# All tests with reports
./bin/test_providers.sh --all --save
```

### Test Files

- **`test_provider_consistency.py`** - Tests all providers with identical configurations
- **`test_provider_parameters.py`** - Validates provider-specific parameters and edge cases
- **`performance_benchmarking.py`** - Comprehensive performance measurement system
- **`provider_test_config.py`** - Centralized configuration for all providers and scenarios
- **`run_provider_tests.py`** - Main test runner with CLI interface
- **`test_provider_system.py`** - Quick demo/validation script

### Integration with pytest

You can also run the tests using pytest:

```bash
# Run consistency tests
pytest reactive_agents/tests/integration/test_provider_consistency.py -v

# Run parameter validation
pytest reactive_agents/tests/integration/test_provider_parameters.py -v

# Run all provider tests
pytest reactive_agents/tests/integration/ -m "providers" -v
```

### Integration with bin/run_tests.sh

The provider tests are integrated with the main test runner:

```bash
# Provider-specific test categories
./bin/run_tests.sh --category provider-consistency
./bin/run_tests.sh --category provider-parameters  
./bin/run_tests.sh --category provider-all
```

## Documentation

For comprehensive documentation on the provider testing system, see:
- `reactive_agents/docs/PROVIDER_TESTING.md` - Complete guide and reference

## Supported Providers

- **Ollama** - Local models (cogito:14b, llama3.1:8b, qwen2.5:7b)
- **OpenAI** - gpt-4o-mini, gpt-3.5-turbo
- **Anthropic** - claude-3-haiku, claude-3-sonnet
- **Groq** - llama-3.1-8b-instant, mixtral-8x7b
- **Google** - gemini-1.5-flash, gemini-1.5-pro

## Environment Setup

For real API testing, set environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GROQ_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

By default, tests run in mock mode for CI/testing without API costs.