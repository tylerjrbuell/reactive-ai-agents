# Openai Provider Plugin

Custom openai provider

## Installation

1. Copy this plugin to your `plugins/` directory
2. The plugin will be automatically discovered and loaded

## Configuration

Configure the provider in your agent configuration:

```python
# Configure the provider
agent_config.kwargs["model_provider"] = "openai"
```

## Development

1. Implement your provider logic in the `generate_response` method
2. Handle authentication and API calls
3. Test your provider with different prompts

## API

### Provider Methods

- `generate_response(prompt, **kwargs)`: Generate response
- `is_available()`: Check availability
- `get_supported_models()`: List supported models
