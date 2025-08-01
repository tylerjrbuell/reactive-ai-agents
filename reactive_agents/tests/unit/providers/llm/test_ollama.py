"""
Tests for Ollama Model Provider.

Tests the Ollama model provider functionality including initialization,
validation, chat completion, and tool handling.
"""

import os
import pytest
from unittest.mock import Mock, patch, AsyncMock

from reactive_agents.providers.llm.ollama import OllamaModelProvider
from reactive_agents.core.types.provider_types import (
    CompletionResponse,
    CompletionMessage,
)


class TestOllamaModelProvider:
    """Test cases for Ollama Model Provider."""

    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock Ollama client."""
        with patch("ollama.Client") as mock_client_class, patch(
            "ollama.AsyncClient"
        ) as mock_async_client_class:

            # Mock sync client
            mock_client = Mock()
            mock_models = Mock()
            mock_models.models = [
                Mock(model="llama2:latest"),
                Mock(model="codellama:7b"),
                Mock(model="mistral:latest"),
            ]
            mock_client.list.return_value = mock_models
            mock_client_class.return_value = mock_client

            # Mock async client
            mock_async_client = Mock()
            mock_async_client_class.return_value = mock_async_client

            yield {
                "sync_client": mock_client,
                "async_client": mock_async_client,
                "sync_client_class": mock_client_class,
                "async_client_class": mock_async_client_class,
            }

    def test_initialization_default(self, mock_ollama_client):
        """Test OllamaModelProvider initialization with defaults."""
        provider = OllamaModelProvider(model="llama2")

        assert provider.model == "llama2"
        assert provider.id == "ollama"
        assert provider.options == {}
        assert provider.context is None
        assert provider.host == "http://localhost:11434"
        assert hasattr(provider, "client")

    def test_initialization_with_options(self, mock_ollama_client):
        """Test OllamaModelProvider initialization with custom options."""
        options = {"temperature": 0.7, "max_tokens": 1000}
        provider = OllamaModelProvider(model="codellama:7b", options=options)

        assert provider.model == "codellama:7b"
        assert provider.options == options

    def test_initialization_with_custom_host(self, mock_ollama_client):
        """Test OllamaModelProvider initialization with custom host."""
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://custom-host:11434"}):
            # Need to patch the class variable host before creating instance
            with patch.object(OllamaModelProvider, "host", "http://custom-host:11434"):
                provider = OllamaModelProvider(model="llama2")
                assert provider.host == "http://custom-host:11434"

    def test_validate_model_success(self, mock_ollama_client):
        """Test successful model validation."""
        provider = OllamaModelProvider(model="llama2")

        # Should not raise an exception since llama2:latest is in the mock models
        result = provider.validate_model()
        assert result is None  # validate_model doesn't return anything on success

    def test_validate_model_with_tag(self, mock_ollama_client):
        """Test model validation with explicit tag."""
        provider = OllamaModelProvider(model="llama2:latest")

        # Should not raise an exception since llama2:latest is in the mock models
        result = provider.validate_model()
        assert result is None

    def test_validate_model_failure(self, mock_ollama_client):
        """Test model validation failure."""
        # Mock the client to return models without our target model
        mock_ollama_client["sync_client"].list.return_value.models = [
            Mock(model="llama2:latest"),
            Mock(model="codellama:7b"),
        ]

        # This should raise an exception during initialization
        with pytest.raises(
            Exception, match="Model nonexistent is either not supported"
        ):
            OllamaModelProvider(model="nonexistent")

    @pytest.mark.asyncio
    async def test_get_chat_completion_basic(self, mock_ollama_client):
        """Test basic chat completion."""
        provider = OllamaModelProvider(model="llama2")

        # Mock the async client show method (for model capabilities)
        mock_model_info = Mock()
        mock_model_info.capabilities = []  # No tool support
        mock_ollama_client["async_client"].show = AsyncMock(
            return_value=mock_model_info
        )

        # Mock the async client chat response
        mock_response = Mock()
        mock_response.message = Mock()
        mock_response.message.content = "Hello! How can I help you?"
        mock_response.message.thinking = None
        mock_response.message.role = "assistant"
        mock_response.message.tool_calls = None
        mock_response.model = "llama2"
        mock_response.done = True
        mock_response.done_reason = "stop"
        mock_response.prompt_eval_count = 10
        mock_response.eval_count = 15
        mock_response.eval_duration = 1000
        mock_response.load_duration = 500
        mock_response.total_duration = 1500
        mock_response.created_at = "2024-01-01T00:00:00Z"

        mock_ollama_client["async_client"].chat = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        result = await provider.get_chat_completion(messages=messages)

        assert isinstance(result, CompletionResponse)
        assert result.message.content == "Hello! How can I help you?"
        assert result.message.role == "assistant"
        assert result.model == "llama2"
        assert result.done is True
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 15

    @pytest.mark.asyncio
    async def test_get_chat_completion_with_tools_supported(self, mock_ollama_client):
        """Test chat completion with tools when model supports them."""
        provider = OllamaModelProvider(model="llama2")

        # Mock model info to show tool support
        mock_model_info = Mock()
        mock_model_info.capabilities = ["tools"]
        mock_ollama_client["async_client"].show = AsyncMock(
            return_value=mock_model_info
        )

        # Mock chat response with tool calls
        mock_response = Mock()
        mock_response.message = Mock()
        mock_response.message.content = ""
        mock_response.message.thinking = None
        mock_response.message.role = "assistant"

        # Mock tool call
        mock_tool_call = Mock()
        mock_tool_call.model_dump.return_value = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}',
            },
        }
        mock_response.message.tool_calls = [mock_tool_call]
        mock_response.model = "llama2"
        mock_response.done = True
        mock_response.done_reason = "tool_calls"
        mock_response.prompt_eval_count = 20
        mock_response.eval_count = 5
        mock_response.eval_duration = 800
        mock_response.load_duration = 300
        mock_response.total_duration = 1100
        mock_response.created_at = "2024-01-01T00:00:00Z"

        mock_ollama_client["async_client"].chat = AsyncMock(return_value=mock_response)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]
        result = await provider.get_chat_completion(messages=messages, tools=tools)

        assert isinstance(result, CompletionResponse)
        assert result.message.tool_calls is not None
        assert len(result.message.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_get_chat_completion_with_tools_unsupported(self, mock_ollama_client):
        """Test chat completion with tools when model doesn't support them."""
        provider = OllamaModelProvider(model="llama2")

        # Mock model info to show no tool support
        mock_model_info = Mock()
        mock_model_info.capabilities = []
        mock_ollama_client["async_client"].show = AsyncMock(
            return_value=mock_model_info
        )

        # Mock the get_tool_calls method
        mock_tool_call = Mock()
        mock_tool_call.model_dump.return_value = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}',
            },
        }

        with patch.object(provider, "get_tool_calls", return_value=[mock_tool_call]):
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                    },
                }
            ]

            messages = [
                {"role": "user", "content": "What's the weather in San Francisco?"}
            ]
            result = await provider.get_chat_completion(messages=messages, tools=tools)

            assert isinstance(result, CompletionResponse)
            assert result.message.tool_calls is not None
            assert len(result.message.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_get_completion(self, mock_ollama_client):
        """Test text completion."""
        provider = OllamaModelProvider(model="llama2")

        # Mock the generate response
        mock_response = Mock()
        mock_response.response = "The capital of France is Paris."
        mock_response.thinking = None
        mock_response.model = "llama2"
        mock_response.done = True
        mock_response.done_reason = "stop"
        mock_response.prompt_eval_count = 8
        mock_response.eval_count = 12
        mock_response.eval_duration = 900
        mock_response.load_duration = 400
        mock_response.total_duration = 1300
        mock_response.created_at = "2024-01-01T00:00:00Z"

        mock_ollama_client["async_client"].generate = AsyncMock(
            return_value=mock_response
        )

        result = await provider.get_completion(prompt="What is the capital of France?")

        assert isinstance(result, CompletionResponse)
        assert result.message.content == "The capital of France is Paris."
        assert result.model == "llama2"
        assert result.done is True

    @pytest.mark.asyncio
    async def test_get_tool_calls(self, mock_ollama_client):
        """Test manual tool call generation."""
        provider = OllamaModelProvider(model="llama2")

        # Create a mock context with reasoning engine
        mock_context = Mock()
        mock_reasoning_engine = Mock()
        mock_prompt = Mock()
        mock_completion_result = Mock()
        mock_completion_result.result_json = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "San Francisco"},
                    },
                }
            ]
        }

        mock_prompt.get_completion = AsyncMock(return_value=mock_completion_result)
        mock_reasoning_engine.get_prompt.return_value = mock_prompt
        mock_context.reasoning_engine = mock_reasoning_engine
        provider.context = mock_context

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                },
            }
        ]

        result = await provider.get_tool_calls(
            task="Get weather for San Francisco", tools=tools, max_calls=1
        )

        assert len(result) == 1
        assert result[0].function.name == "get_weather"

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_ollama_client):
        """Test error handling in various methods."""
        provider = OllamaModelProvider(model="llama2")

        # Mock an exception in the async client
        mock_ollama_client["async_client"].chat = AsyncMock(
            side_effect=Exception("Connection error")
        )

        with patch.object(provider, "_handle_error") as mock_handle_error:
            mock_handle_error.side_effect = Exception("Handled error")

            with pytest.raises(Exception):
                await provider.get_chat_completion(
                    messages=[{"role": "user", "content": "test"}]
                )

            mock_handle_error.assert_called_once()

    def test_thinking_extraction(self, mock_ollama_client):
        """Test thinking content extraction and storage."""
        provider = OllamaModelProvider(model="llama2")

        # Create a mock context for thinking storage
        mock_context = Mock()
        mock_context.session = Mock()
        mock_context.session.thinking_log = []
        mock_context.agent_logger = Mock()
        provider.context = mock_context

        # Create a message with thinking content
        message = CompletionMessage(
            content="<think>Let me think about this...</think>The answer is 42.",
            role="assistant",
        )

        # Extract thinking
        cleaned_message = provider.extract_and_store_thinking(message, "test_context")

        # Verify thinking was extracted and content was cleaned
        assert cleaned_message.content == "The answer is 42."
        assert len(mock_context.session.thinking_log) == 1
        assert (
            mock_context.session.thinking_log[0]["thinking"]
            == "Let me think about this..."
        )
        assert mock_context.session.thinking_log[0]["call_context"] == "test_context"

    def test_default_options(self, mock_ollama_client):
        """Test that default options are properly set."""
        from reactive_agents.providers.llm.ollama import DEFAULT_OPTIONS

        provider = OllamaModelProvider(model="llama2")

        # Verify DEFAULT_OPTIONS is accessible
        assert DEFAULT_OPTIONS is not None
        assert isinstance(DEFAULT_OPTIONS, dict)

    @pytest.mark.asyncio
    async def test_stream_response(self, mock_ollama_client):
        """Test streaming response handling."""
        provider = OllamaModelProvider(model="llama2")

        # Mock the async client's show method (for model capabilities)
        mock_model_info = Mock()
        mock_model_info.capabilities = []
        mock_ollama_client["async_client"].show = AsyncMock(
            return_value=mock_model_info
        )

        # Mock a proper stream response structure
        mock_response = Mock()
        mock_response.message = Mock()
        mock_response.message.content = "Hello! How can I help you?"
        mock_response.message.thinking = None
        mock_response.message.role = "assistant"
        mock_response.message.tool_calls = None
        mock_response.model = "llama2"
        mock_response.done = True
        mock_response.done_reason = "stop"
        mock_response.prompt_eval_count = 10
        mock_response.eval_count = 15
        mock_response.eval_duration = 1000
        mock_response.load_duration = 500
        mock_response.total_duration = 1500
        mock_response.created_at = "2024-01-01T00:00:00Z"

        mock_ollama_client["async_client"].chat = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        result = await provider.get_chat_completion(messages=messages, stream=True)

        # Should return a CompletionResponse even for streaming
        assert isinstance(result, CompletionResponse)
        assert result.message.content == "Hello! How can I help you?"

    def test_model_with_colon_handling(self, mock_ollama_client):
        """Test that models with colons in names are handled correctly."""
        # Ensure the mock includes the model with tag
        mock_ollama_client["sync_client"].list.return_value.models = [
            Mock(model="llama2:7b-chat"),
            Mock(model="codellama:7b-instruct"),
        ]

        provider = OllamaModelProvider(model="llama2:7b-chat")

        # Should not raise exception since the exact model is in the list
        provider.validate_model()
        assert provider.model == "llama2:7b-chat"
