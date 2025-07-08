"""
Test suite for Google model provider.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from reactive_agents.providers.llm.google import GoogleModelProvider
from reactive_agents.providers.llm.base import CompletionResponse, CompletionMessage


class TestGoogleModelProvider:
    """Test Google model provider."""

    @pytest.fixture
    def mock_genai(self):
        """Mock Google Generative AI module."""
        with patch("reactive_agents.providers.llm.google.genai") as mock:
            # Mock model instance
            mock_model = Mock()
            mock.GenerativeModel.return_value = mock_model

            # Mock response
            mock_response = Mock()
            mock_candidate = Mock()
            mock_content = Mock()
            mock_part = Mock()

            # Set up the response chain
            mock_part.text = "Test response"
            mock_content.parts = [mock_part]
            mock_candidate.content = mock_content
            mock_candidate.finish_reason = 1  # STOP
            mock_response.candidates = [mock_candidate]
            mock_response.usage_metadata = Mock()
            mock_response.usage_metadata.prompt_token_count = 100
            mock_response.usage_metadata.candidates_token_count = 50

            mock_model.generate_content.return_value = mock_response

            # Mock available models
            mock_model_info = Mock()
            mock_model_info.name = "models/gemini-2.5-flash"
            mock_model_info.supported_generation_methods = ["generateContent"]
            mock.list_models.return_value = [mock_model_info]

            yield mock

    @pytest.fixture
    def mock_env(self):
        """Mock environment variables."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            yield

    def test_init_success(self, mock_env, mock_genai):
        """Test successful initialization."""
        provider = GoogleModelProvider(model="gemini-2.5-flash")
        assert provider.model == "gemini-2.5-flash"
        assert provider.id == "google"

    def test_init_no_api_key(self, mock_genai):
        """Test initialization without API key fails."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(
                ValueError, match="GOOGLE_API_KEY environment variable is required"
            ):
                GoogleModelProvider()

    def test_validate_model_success(self, mock_env, mock_genai):
        """Test model validation success."""
        provider = GoogleModelProvider(model="gemini-2.5-flash")
        result = provider.validate_model()
        assert result["valid"] is True
        assert result["model"] == "gemini-2.5-flash"

    def test_validate_model_invalid(self, mock_env, mock_genai):
        """Test model validation with invalid model."""
        # Mock no models available
        mock_genai.list_models.return_value = []

        provider = GoogleModelProvider(model="invalid-model")
        result = provider.validate_model()
        assert result["valid"] is False
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    async def test_get_chat_completion_success(self, mock_env, mock_genai):
        """Test successful chat completion."""
        provider = GoogleModelProvider(model="gemini-2.5-flash")

        # Mock the retry function to avoid async issues
        with patch.object(provider, "_retry_with_backoff") as mock_retry:
            mock_retry.return_value = (
                mock_genai.GenerativeModel().generate_content.return_value
            )

            messages = [{"role": "user", "content": "Hello"}]
            result = await provider.get_chat_completion(messages)

            assert isinstance(result, CompletionResponse)
            assert result.message.content == "Test response"
            assert result.message.role == "assistant"
            assert result.model == "gemini-2.5-flash"
            assert result.done is True
            assert result.done_reason == "stop"

    @pytest.mark.asyncio
    async def test_get_chat_completion_json_format(self, mock_env, mock_genai):
        """Test chat completion with JSON format."""
        provider = GoogleModelProvider(model="gemini-2.5-flash")

        # Mock JSON response with markdown
        mock_response = Mock()
        mock_candidate = Mock()
        mock_content = Mock()
        mock_part = Mock()

        mock_part.text = '```json\n{"key": "value"}\n```'
        mock_content.parts = [mock_part]
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = 1
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50

        with patch.object(provider, "_retry_with_backoff") as mock_retry:
            mock_retry.return_value = mock_response

            messages = [{"role": "user", "content": "Return JSON"}]
            result = await provider.get_chat_completion(messages, format="json")

            assert isinstance(result, CompletionResponse)
            # JSON should be cleaned (markdown removed)
            assert result.message.content == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_get_chat_completion_safety_filter(self, mock_env, mock_genai):
        """Test chat completion blocked by safety filters."""
        provider = GoogleModelProvider(model="gemini-2.5-flash")

        # Mock safety-blocked response
        mock_response = Mock()
        mock_candidate = Mock()
        mock_candidate.finish_reason = 3  # SAFETY
        mock_response.candidates = [mock_candidate]

        with patch.object(provider, "_retry_with_backoff") as mock_retry:
            mock_retry.return_value = mock_response

            messages = [{"role": "user", "content": "Blocked content"}]
            result = await provider.get_chat_completion(messages)

            assert isinstance(result, CompletionResponse)
            assert result.message.content == "[Response blocked by safety filters]"
            assert result.done_reason == "content_filter"

    @pytest.mark.asyncio
    async def test_get_chat_completion_no_candidates(self, mock_env, mock_genai):
        """Test chat completion with no candidates."""
        provider = GoogleModelProvider(model="gemini-2.5-flash")

        # Mock response with no candidates
        mock_response = Mock()
        mock_response.candidates = []

        with patch.object(provider, "_retry_with_backoff") as mock_retry:
            mock_retry.return_value = mock_response

            messages = [{"role": "user", "content": "Hello"}]
            result = await provider.get_chat_completion(messages)

            assert isinstance(result, CompletionResponse)
            assert result.message.content == "[No response generated]"
            assert result.done_reason == "error"

    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self, mock_env, mock_genai):
        """Test retry mechanism success on first try."""
        provider = GoogleModelProvider(model="gemini-2.5-flash")

        mock_func = Mock(return_value="success")
        result = await provider._retry_with_backoff(mock_func, "arg1", kwarg="value")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg="value")

    @pytest.mark.asyncio
    async def test_retry_with_backoff_rate_limit(self, mock_env, mock_genai):
        """Test retry mechanism with rate limiting."""
        from google.api_core.exceptions import ResourceExhausted

        provider = GoogleModelProvider(model="gemini-2.5-flash")

        mock_func = Mock(
            side_effect=[ResourceExhausted("Rate limit exceeded"), "success"]
        )

        with patch("asyncio.sleep") as mock_sleep:
            result = await provider._retry_with_backoff(mock_func, max_retries=2)

        assert result == "success"
        assert mock_func.call_count == 2
        mock_sleep.assert_called_once()

    def test_clean_json_response(self, mock_env, mock_genai):
        """Test JSON response cleaning."""
        provider = GoogleModelProvider(model="gemini-2.5-flash")

        # Test with markdown code blocks
        json_with_markdown = '```json\n{"key": "value"}\n```'
        cleaned = provider._clean_json_response(json_with_markdown)
        assert cleaned == '{"key": "value"}'

        # Test with just backticks
        json_with_backticks = '```\n{"key": "value"}\n```'
        cleaned = provider._clean_json_response(json_with_backticks)
        assert cleaned == '{"key": "value"}'

        # Test with no markdown
        plain_json = '{"key": "value"}'
        cleaned = provider._clean_json_response(plain_json)
        assert cleaned == '{"key": "value"}'

    def test_prepare_messages(self, mock_env, mock_genai):
        """Test message preparation for Google format."""
        provider = GoogleModelProvider(model="gemini-2.5-flash")

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        prepared = provider._prepare_messages(messages)

        # System message should be prepended to first user message
        assert len(prepared) == 3  # system merged with first user
        assert prepared[0]["role"] == "user"
        assert "You are a helpful assistant" in prepared[0]["parts"][0]
        assert "Hello" in prepared[0]["parts"][0]

        # Assistant role should be mapped to model
        assert prepared[1]["role"] == "model"
        assert prepared[1]["parts"][0] == "Hi there!"

        # User role stays as user
        assert prepared[2]["role"] == "user"
        assert prepared[2]["parts"][0] == "How are you?"

    @pytest.mark.asyncio
    async def test_get_completion(self, mock_env, mock_genai):
        """Test text completion method."""
        provider = GoogleModelProvider(model="gemini-2.5-flash")

        with patch.object(provider, "get_chat_completion") as mock_chat:
            mock_chat.return_value = CompletionResponse(
                message=CompletionMessage(content="Test response", role="assistant"),
                model="gemini-2.5-flash",
                done=True,
                done_reason="stop",
                created_at="123456789",
            )

            result = await provider.get_completion("Hello", system="Be helpful")

            assert isinstance(result, CompletionResponse)
            assert result.message.content == "Test response"

            # Check that chat completion was called with correct messages
            mock_chat.assert_called_once()
            call_args = mock_chat.call_args[1]
            messages = call_args["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "Be helpful"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "Hello"


# Integration test (requires actual API key)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_provider_integration():
    """Integration test with real Google API (requires GOOGLE_API_KEY)."""
    import os

    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set - skipping integration test")

    provider = GoogleModelProvider(model="gemini-2.5-flash")

    # Test simple completion
    messages = [{"role": "user", "content": "Say 'Hello World' and nothing else."}]
    result = await provider.get_chat_completion(messages)

    assert isinstance(result, CompletionResponse)
    assert "Hello World" in result.message.content
    assert result.done is True

    # Test JSON format
    json_messages = [{"role": "user", "content": 'Return JSON: {"greeting": "Hello"}'}]
    json_result = await provider.get_chat_completion(json_messages, format="json")

    assert isinstance(json_result, CompletionResponse)
    # Should contain valid JSON
    assert "greeting" in json_result.message.content
