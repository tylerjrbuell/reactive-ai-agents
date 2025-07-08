# LLM Providers

This module contains implementations of various LLM providers for the reactive-agents framework.

## Available Providers

### OpenAI Provider (`openai`)

- **Models**: GPT-4, GPT-3.5-turbo, GPT-4o, and other OpenAI models
- **Features**: Chat completions, text completions, tool/function calling, streaming support
- **Requirements**: `openai` package, `OPENAI_API_KEY` environment variable

### Anthropic Provider (`anthropic`)

- **Models**: Claude 3 models (Sonnet, Opus, Haiku), legacy Claude 1/2 models
- **Features**: Chat completions, text completions, tool/function calling (Claude 3 only), streaming support
- **Requirements**: `anthropic` package, `ANTHROPIC_API_KEY` environment variable

### Google Provider (`google`)

- **Models**: Gemini Pro, Gemini 1.5 Pro, and other Google Generative AI models
- **Features**: Chat completions, text completions, tool/function calling, streaming support, configurable safety settings, robust error handling
- **Requirements**: `google-generativeai` package, `GOOGLE_API_KEY` environment variable

### Groq Provider (`groq`)

- **Models**: Groq-hosted models (Llama, Mixtral, etc.)
- **Features**: Chat completions, text completions, tool/function calling
- **Requirements**: `groq` package, `GROQ_API_KEY` environment variable

### Ollama Provider (`ollama`)

- **Models**: Locally hosted models via Ollama
- **Features**: Chat completions, text completions, tool/function calling
- **Requirements**: `ollama` package, local Ollama installation

## Usage

### Using Provider Factory (Recommended)

```python
from reactive_agents.providers.llm.factory import ModelProviderFactory

# OpenAI
provider = ModelProviderFactory.get_model_provider("openai:gpt-4")

# Anthropic
provider = ModelProviderFactory.get_model_provider("anthropic:claude-3-sonnet-20240229")

# Google
provider = ModelProviderFactory.get_model_provider("google:gemini-pro")

# Groq
provider = ModelProviderFactory.get_model_provider("groq:mixtral-8x7b-32768")

# Ollama
provider = ModelProviderFactory.get_model_provider("ollama:llama2")
```

### Using Providers Directly

```python
from reactive_agents.providers.llm.openai import OpenAIModelProvider
from reactive_agents.providers.llm.anthropic import AnthropicModelProvider
from reactive_agents.providers.llm.google import GoogleModelProvider

# OpenAI
openai_provider = OpenAIModelProvider(
    model="gpt-4",
    options={"temperature": 0.7, "max_tokens": 1000}
)

# Anthropic
anthropic_provider = AnthropicModelProvider(
    model="claude-3-sonnet-20240229",
    options={"temperature": 0.7, "max_tokens": 1000}
)

# Google
google_provider = GoogleModelProvider(
    model="gemini-pro",
    options={"temperature": 0.7, "max_output_tokens": 1000}
)

# Configure safety settings (optional)
google_provider.configure_safety_settings()  # Use permissive settings
# Or customize safety settings
from google.generativeai.types import HarmCategory, HarmBlockThreshold
custom_safety = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}
google_provider.configure_safety_settings(custom_safety)
```

### Chat Completion

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

response = await provider.get_chat_completion(messages=messages)
print(response.message.content)
```

### Text Completion

```python
response = await provider.get_completion(
    prompt="The best programming language is",
    system="You are a helpful assistant."
)
print(response.message.content)
```

### JSON Output Mode

All providers support JSON output mode for structured responses:

```python
# Request JSON format response
response = await provider.get_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "List the top 3 programming languages with their popularity scores."}
    ],
    format="json"
)
print(response.message.content)
# Output: {"languages": [{"name": "Python", "score": 95}, {"name": "JavaScript", "score": 87}, {"name": "Java", "score": 82}]}
```

```python
# JSON format with text completion
response = await provider.get_completion(
    prompt="Generate a JSON object with user profile data",
    format="json"
)
print(response.message.content)
# Output: {"name": "John Doe", "age": 30, "email": "john@example.com"}
```

**Provider-Specific JSON Support:**

- **OpenAI**: Native JSON mode with `{"type": "json_object"}`
- **Anthropic**: JSON instruction added to prompt (both Claude 3 and legacy models)
- **Google**: JSON instruction added to prompt with enhanced formatting
- **Groq**: Native JSON mode with `{"type": "json_object"}`
- **Ollama**: Native JSON mode with `format="json"`

### Tool/Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"}
                },
                "required": ["location"]
            }
        }
    }
]

response = await provider.get_chat_completion(
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

if response.message.tool_calls:
    for tool_call in response.message.tool_calls:
        print(f"Tool: {tool_call['function']['name']}")
        print(f"Arguments: {tool_call['function']['arguments']}")
```

## Environment Variables

Set the following environment variables before using the providers:

```bash
# OpenAI
export OPENAI_API_KEY="your_openai_api_key"

# Anthropic
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Google
export GOOGLE_API_KEY="your_google_api_key"

# Groq
export GROQ_API_KEY="your_groq_api_key"

# Ollama (optional, defaults to http://localhost:11434)
export OLLAMA_HOST="http://localhost:11434"
```

## Model Support

### OpenAI Models

- `gpt-4` - GPT-4 (latest)
- `gpt-4o` - GPT-4 Omni
- `gpt-3.5-turbo` - GPT-3.5 Turbo
- `gpt-4-turbo` - GPT-4 Turbo
- And other OpenAI models

### Anthropic Models

- `claude-3-opus-20240229` - Claude 3 Opus
- `claude-3-sonnet-20240229` - Claude 3 Sonnet
- `claude-3-haiku-20240307` - Claude 3 Haiku
- `claude-3-5-sonnet-20240620` - Claude 3.5 Sonnet
- `claude-3-5-sonnet-20241022` - Claude 3.5 Sonnet (latest)
- `claude-3-5-haiku-20241022` - Claude 3.5 Haiku
- `claude-2` - Claude 2 (legacy, limited features)

### Google Models

- `gemini-pro` - Gemini Pro
- `gemini-1.5-pro` - Gemini 1.5 Pro
- `gemini-1.5-pro-latest` - Gemini 1.5 Pro (latest)
- `gemini-pro-vision` - Gemini Pro Vision (multimodal)
- And other Google Generative AI models

## Installation

Install the required dependencies:

```bash
# For OpenAI support
pip install openai

# For Anthropic support
pip install anthropic

# For Google support
pip install google-generativeai

# For Groq support
pip install groq

# For Ollama support
pip install ollama
```

Or install all at once:

```bash
pip install openai anthropic google-generativeai groq ollama
```

## Error Handling

All providers implement comprehensive error handling:

```python
try:
    response = await provider.get_chat_completion(messages=messages)
    print(response.message.content)
except Exception as e:
    print(f"Error: {e}")
    # Error details are automatically logged and stored in context
```

### Google Provider Safety Handling

The Google provider includes special handling for safety-filtered responses:

```python
response = await google_provider.get_chat_completion(messages=messages)

# Check if response was blocked by safety filters
if response.done_reason == "content_filter":
    print("Response was blocked by safety filters")
    # Configure more permissive settings
    google_provider.configure_safety_settings()  # Disable safety filtering
    response = await google_provider.get_chat_completion(messages=messages)
```

## Streaming Support

Both OpenAI and Anthropic providers support streaming responses:

```python
response = await provider.get_chat_completion(
    messages=messages,
    stream=True
)

# Handle streaming response
for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

## Testing

Run the test suite to verify provider functionality:

```bash
# Run all provider tests
pytest reactive_agents/tests/unit/providers/llm/

# Run specific provider tests
pytest reactive_agents/tests/unit/providers/llm/test_openai_anthropic.py
```

## Demo

See the demo script for usage examples:

```bash
python reactive_agents/examples/openai_anthropic_demo.py
```

## Architecture

All providers inherit from `BaseModelProvider` and implement:

- `get_chat_completion()` - Chat-based completions
- `get_completion()` - Text-based completions
- `validate_model()` - Model validation
- Error handling and logging integration
- Automatic registration with the factory

## Notes

- **OpenAI**: Uses the official OpenAI Python SDK
- **Anthropic**: Uses the official Anthropic Python SDK for Claude 3, falls back to HTTP requests for legacy models
- **Tool Support**: OpenAI and Anthropic (Claude 3) support tool/function calling
- **Streaming**: All providers support streaming where available
- **Error Handling**: Comprehensive error handling with context integration
- **Auto-registration**: Providers are automatically registered with the factory using metaclass
