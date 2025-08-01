# Multi-Provider Integration Testing System

This document describes the comprehensive testing system for validating consistency, performance, and reliability across all LLM providers supported by the Reactive Agents framework.

## Overview

The multi-provider testing system addresses a critical need for production-ready AI agent frameworks: ensuring consistent behavior and performance across different LLM providers. This system includes:

- **Provider Consistency Testing**: Validates that all providers handle the same tasks consistently
- **Performance Benchmarking**: Measures and compares execution time, quality, and resource usage
- **Parameter Validation**: Tests provider-specific parameters and configurations
- **Configuration Management**: Centralized configuration for all providers and test scenarios

## Key Components

### 1. Provider Consistency Tester (`test_provider_consistency.py`)

Tests all providers with identical agent configurations to identify inconsistencies:

```python
from reactive_agents.tests.integration.test_provider_consistency import ProviderConsistencyTester

# Run comprehensive consistency tests
tester = ProviderConsistencyTester(enable_real_execution=False)
results = await tester.run_comprehensive_test()

# Generate detailed report
report = tester.generate_report(results)
print(report)
```

**Features:**
- Standardized test scenarios across all providers
- Mock mode for CI/testing without API calls
- Real execution mode for actual provider testing
- Comprehensive result analysis and reporting
- Quality scoring based on validation criteria

### 2. Performance Benchmarking (`performance_benchmarking.py`)

Measures and compares performance characteristics:

```python
from reactive_agents.tests.integration.performance_benchmarking import PerformanceBenchmarker

benchmarker = PerformanceBenchmarker()
metrics = await benchmarker.benchmark_provider_model(
    "openai", "gpt-4o-mini", "simple_task", agent_builder_func
)
```

**Metrics Collected:**
- Execution time (total, first response, average response)
- Throughput (tokens/sec, tool calls/sec, iterations/sec)
- Quality scores (accuracy, coherence, efficiency)
- Resource usage (memory, CPU, network requests)
- Cost metrics (estimated cost, cost per success)
- Token usage (input, output, total)

### 3. Provider Configuration Management (`provider_test_config.py`)

Centralized configuration system for all providers and test scenarios:

```python
from reactive_agents.tests.integration.provider_test_config import get_test_config

config = get_test_config()

# Get providers that support a specific capability
compatible_providers = config.get_providers_for_scenario("tool_calling_task")

# Get recommended models within cost constraints
models = config.get_recommended_models("openai", max_cost=0.001)

# Validate environment setup
env_status = config.validate_environment(["openai", "anthropic"])
```

**Configuration Features:**
- Detailed provider capabilities and constraints
- Model-specific parameters and recommendations
- Test scenario definitions with validation criteria
- Cost estimation and budget management
- Environment validation

### 4. Parameter Validation (`test_provider_parameters.py`)

Tests provider-specific parameters and edge cases:

- Default parameter validation
- Parameter range and type checking
- Provider-specific optimizations
- Context window limit testing
- Error handling consistency
- Performance parameter impact analysis

## Supported Providers

The system currently supports comprehensive testing for:

| Provider | Models | Key Features |
|----------|--------|--------------|
| **Ollama** | cogito:14b, llama3.1:8b, qwen2.5:7b | Local models, free usage, customizable parameters |
| **OpenAI** | gpt-4o-mini, gpt-3.5-turbo | High reliability, JSON mode, function calling |
| **Anthropic** | claude-3-haiku, claude-3-sonnet | Large context, high quality, tool calling |
| **Groq** | llama-3.1-8b-instant, mixtral-8x7b | Ultra-fast inference, cost-effective |
| **Google** | gemini-1.5-flash, gemini-1.5-pro | Massive context windows, multimodal support |

## Test Scenarios

Standardized test scenarios ensure consistent evaluation:

### 1. Basic Tool Call (`simple_task`)
- **Task**: Use test_tool_simple to process text
- **Strategy**: Reactive
- **Validation**: Response contains expected processed text
- **Purpose**: Test basic tool calling capability

### 2. Multi-Step Math (`math_task`)
- **Task**: Calculate 25 + 17, then multiply by 3
- **Strategy**: Reflect-Decide-Act
- **Validation**: Contains intermediate (42) and final (126) results
- **Purpose**: Test multi-step reasoning with tools

### 3. JSON Processing (`json_processing`)
- **Task**: Process JSON data with test_tool_json
- **Strategy**: Plan-Execute-Reflect
- **Validation**: Successful JSON processing confirmed
- **Purpose**: Test structured data handling

### 4. Complex Workflow (`multi_tool_task`)
- **Task**: Sequential use of multiple tools
- **Strategy**: Plan-Execute-Reflect
- **Validation**: All tool outputs present in sequence
- **Purpose**: Test complex multi-tool workflows

### 5. Adaptive Reasoning (`adaptive_reasoning`)
- **Task**: Mathematical problem with strategy adaptation
- **Strategy**: Adaptive
- **Validation**: Correct mathematical results
- **Purpose**: Test dynamic strategy selection

## Usage Examples

### Quick Consistency Check

```bash
# Run basic consistency tests (mocked)
python run_provider_tests.py --consistency

# Test specific providers
python run_provider_tests.py --consistency --providers ollama openai

# Run with real API calls (requires API keys)
python run_provider_tests.py --consistency --real
```

### Performance Benchmarking

```bash
# Benchmark all providers
python run_provider_tests.py --performance

# Benchmark specific provider with multiple iterations
python run_provider_tests.py --performance --providers groq --benchmark-iterations 5

# Save detailed performance reports
python run_provider_tests.py --performance --save
```

### Parameter Validation

```bash
# Validate parameters for all providers
python run_provider_tests.py --parameters

# Test specific provider parameters
python run_provider_tests.py --parameters --providers ollama
```

### Comprehensive Testing

```bash
# Run all tests and save results
python run_provider_tests.py --all --save --output-dir results_$(date +%Y%m%d)
```

## Integration with CI/CD

### Pytest Integration

The system integrates with pytest for automated testing:

```bash
# Run consistency tests
pytest reactive_agents/tests/integration/test_provider_consistency.py -v

# Run parameter validation tests
pytest reactive_agents/tests/integration/test_provider_parameters.py -v

# Run with specific markers
pytest -m "providers and integration" --tb=short
```

### CI Configuration

Example GitHub Actions workflow:

```yaml
name: Provider Consistency Tests
on: [push, pull_request]

jobs:
  provider-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      
      - name: Run provider consistency tests
        run: |
          poetry run python run_provider_tests.py --consistency --save
        env:
          # Only test with mocks in CI by default
          ENABLE_REAL_EXECUTION: "false"
      
      - name: Upload test results
        uses: actions/upload-artifact@v2
        with:
          name: provider-test-results
          path: test_results/
```

## Configuration and Environment Setup

### Environment Variables

For real API testing, set the following environment variables:

```bash
# Required for real provider testing
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  
export GROQ_API_KEY="your-groq-key"
export GOOGLE_API_KEY="your-google-key"

# Optional: Enable real execution in tests
export ENABLE_REAL_EXECUTION=1

# Optional: Custom MCP configuration
export MCP_CONFIG_PATH="/path/to/mcp_config.json"
```

### Provider Configuration

Each provider can be configured with specific parameters:

```python
# Example: Custom Ollama configuration
ollama_config = {
    "temperature": 0.1,
    "num_ctx": 8000,        # Larger context window
    "num_gpu": 256,         # Use GPU acceleration
    "num_thread": 8,        # CPU threads
    "repeat_penalty": 1.1   # Reduce repetition
}

# Example: OpenAI optimization
openai_config = {
    "temperature": 0.0,     # Deterministic output
    "max_tokens": 1000,     # Limit response length
    "top_p": 0.9,          # Nucleus sampling
    "frequency_penalty": 0.1 # Reduce repetition
}
```

## Output and Reporting

### Test Reports

The system generates comprehensive reports in multiple formats:

1. **Console Output**: Real-time test progress and summaries
2. **JSON Reports**: Detailed machine-readable results
3. **Markdown Reports**: Human-readable analysis and recommendations  
4. **CSV Data**: Raw metrics for further analysis
5. **Performance Charts**: Visual comparisons (when matplotlib available)

### Sample Report Output

```
# Multi-Provider Integration Test Report

## Executive Summary
- **Total Tests**: 25
- **Successful**: 23 (92.0%)
- **Failed**: 2 (8.0%)

## Provider Performance Summary
### OpenAI
- Success Rate: 100.0%
- Average Execution Time: 1.20s
- Average Quality Score: 0.92

### Ollama  
- Success Rate: 85.7%
- Average Execution Time: 2.80s
- Average Quality Score: 0.78

## Recommendations
- OpenAI shows highest reliability for production use
- Ollama performance varies significantly with model size
- Consider implementing retry logic for failed requests
- Monitor token usage for cost optimization
```

## Best Practices

### 1. Regular Testing
- Run consistency tests before major releases
- Include provider tests in CI/CD pipelines
- Monitor performance trends over time

### 2. Cost Management
- Use mock mode for frequent testing
- Set budget limits for real API testing
- Monitor token usage and costs

### 3. Configuration Management
- Keep provider configurations in version control
- Document parameter changes and their impact
- Use environment-specific configurations

### 4. Performance Monitoring
- Establish baseline performance metrics
- Set up alerts for performance degradation
- Track cost efficiency over time

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   Error: Provider validation failed for openai
   Solution: Ensure OPENAI_API_KEY environment variable is set
   ```

2. **Timeout Issues**
   ```
   Error: Test timed out after 120 seconds
   Solution: Increase timeout in test configuration or check provider status
   ```

3. **Import Errors**
   ```
   Error: No module named 'reactive_agents'
   Solution: Ensure you're running from project root or install package
   ```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Run with debug logging
PYTHONPATH=. python run_provider_tests.py --consistency --providers ollama -v
```

## Contributing

To extend the testing system:

1. **Add New Providers**: Update `provider_test_config.py` with provider details
2. **Add Test Scenarios**: Define new scenarios in the configuration
3. **Add Metrics**: Extend `PerformanceMetrics` class for new measurements
4. **Add Validations**: Create new parameter validation tests

### Example: Adding a New Provider

```python
# In provider_test_config.py
"new_provider": ProviderConfig(
    name="new_provider",
    models={
        "model-v1": ModelConfig(
            name="model-v1",
            context_window=4000,
            capabilities={ProviderCapability.TOOL_CALLING},
            recommended_params={"temperature": 0.1}
        )
    },
    required_env_vars=["NEW_PROVIDER_API_KEY"],
    # ... other configuration
)
```

## Future Enhancements

Planned improvements to the testing system:

- **Real-time Monitoring**: Live performance dashboards
- **Automated Optimization**: AI-driven parameter tuning
- **Cost Optimization**: Intelligent provider selection based on cost/performance
- **Advanced Analytics**: Machine learning-based performance prediction
- **Multi-modal Testing**: Support for vision and audio capabilities
- **Stress Testing**: High-load and concurrent request testing

## Conclusion

This comprehensive testing system ensures the Reactive Agents framework maintains consistent, reliable, and performant behavior across all supported LLM providers. By systematically testing provider consistency, performance, and parameters, we can identify issues early and optimize the framework for production use.

The system provides both developers and users with confidence that their agents will behave predictably regardless of which LLM provider they choose, while also providing insights into the trade-offs between different providers in terms of cost, speed, and quality.