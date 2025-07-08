"""
Tests for OpenAI and Anthropic model providers.

Tests the basic functionality of the new OpenAI and Anthropic providers including
instantiation, configuration, and basic API structure.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from reactive_agents.providers.llm.factory import ModelProviderFactory
from reactive_agents.providers.llm.openai import OpenAIModelProvider
from reactive_agents.providers.llm.anthropic import AnthropicModelProvider
from reactive_agents.providers.llm.google import GoogleModelProvider


class TestOpenAIProvider:
    """Test cases for OpenAI provider."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with patch("reactive_agents.providers.llm.openai.OpenAI") as mock_client:
            # Mock the models.list() method
            mock_models = Mock()
            mock_models.data = [
                Mock(id="gpt-4"),
                Mock(id="gpt-3.5-turbo"),
                Mock(id="gpt-4o"),
            ]
            mock_client.return_value.models.list.return_value = mock_models
            yield mock_client

    def test_openai_provider_initialization(self, mock_openai_client):
        """Test OpenAI provider initialization."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIModelProvider(model="gpt-4")
            assert provider.model == "gpt-4"
            assert provider.id == "openai"
            assert provider.name == "openai"

    def test_openai_provider_missing_api_key(self, mock_openai_client):
        """Test OpenAI provider fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="OPENAI_API_KEY environment variable is required"
            ):
                OpenAIModelProvider(model="gpt-4")

    def test_openai_provider_model_validation(self, mock_openai_client):
        """Test OpenAI provider model validation."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIModelProvider(model="gpt-4")
            result = provider.validate_model()
            assert result["valid"] is True
            assert result["model"] == "gpt-4"

    def test_openai_provider_via_factory(self, mock_openai_client):
        """Test creating OpenAI provider via factory."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = ModelProviderFactory.get_model_provider("openai:gpt-4")
            assert isinstance(provider, OpenAIModelProvider)
            assert provider.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_openai_json_output_mode(self, mock_openai_client):
        """Test OpenAI provider JSON output mode."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIModelProvider(model="gpt-4")

            # Mock the completion response
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message = Mock()
            mock_completion.choices[0].message.content = '{"test": "json response"}'
            mock_completion.choices[0].message.role = "assistant"
            mock_completion.choices[0].message.tool_calls = None
            mock_completion.choices[0].finish_reason = "stop"
            mock_completion.model = "gpt-4"
            mock_completion.created = 1234567890
            mock_completion.usage = Mock()
            mock_completion.usage.prompt_tokens = 10
            mock_completion.usage.completion_tokens = 20

            provider.client.chat.completions.create.return_value = mock_completion

            response = await provider.get_chat_completion(
                messages=[{"role": "user", "content": "Test"}], format="json"
            )

            # Verify JSON format was requested
            provider.client.chat.completions.create.assert_called_once()
            args, kwargs = provider.client.chat.completions.create.call_args
            assert "response_format" in kwargs
            assert kwargs["response_format"] == {"type": "json_object"}


class TestAnthropicProvider:
    """Test cases for Anthropic provider."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client."""
        with patch("reactive_agents.providers.llm.anthropic.Anthropic") as mock_client:
            yield mock_client

    def test_anthropic_provider_initialization(self, mock_anthropic_client):
        """Test Anthropic provider initialization."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicModelProvider(model="claude-3-sonnet-20240229")
            assert provider.model == "claude-3-sonnet-20240229"
            assert provider.id == "anthropic"
            assert provider.name == "anthropic"

    def test_anthropic_provider_missing_api_key(self, mock_anthropic_client):
        """Test Anthropic provider fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="ANTHROPIC_API_KEY environment variable is required"
            ):
                AnthropicModelProvider(model="claude-3-sonnet-20240229")

    def test_anthropic_provider_claude_3_detection(self, mock_anthropic_client):
        """Test Anthropic provider detects Claude 3 models correctly."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicModelProvider(model="claude-3-sonnet-20240229")
            assert provider.is_claude_3 is True

    def test_anthropic_provider_legacy_detection(self, mock_anthropic_client):
        """Test Anthropic provider detects legacy models correctly."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicModelProvider(model="claude-2")
            assert provider.is_claude_3 is False

    def test_anthropic_provider_via_factory(self, mock_anthropic_client):
        """Test creating Anthropic provider via factory."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = ModelProviderFactory.get_model_provider(
                "anthropic:claude-3-sonnet-20240229"
            )
            assert isinstance(provider, AnthropicModelProvider)
            assert provider.model == "claude-3-sonnet-20240229"

    @pytest.mark.asyncio
    async def test_anthropic_json_output_mode(self, mock_anthropic_client):
        """Test Anthropic provider JSON output mode."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicModelProvider(model="claude-3-sonnet-20240229")

            # Mock the completion response
            mock_completion = Mock()
            mock_completion.content = [Mock()]
            mock_completion.content[0].text = '{"test": "json response"}'
            mock_completion.content[0].type = "text"
            mock_completion.model = "claude-3-sonnet-20240229"
            mock_completion.stop_reason = "end_turn"
            mock_completion.usage = Mock()
            mock_completion.usage.input_tokens = 10
            mock_completion.usage.output_tokens = 20

            provider.client.messages.create.return_value = mock_completion

            response = await provider.get_chat_completion(
                messages=[{"role": "user", "content": "Test"}], format="json"
            )

            # Verify JSON instruction was added to the message
            provider.client.messages.create.assert_called_once()
            args, kwargs = provider.client.messages.create.call_args
            messages = kwargs["messages"]
            assert len(messages) == 1
            assert "Please respond in valid JSON format" in messages[0]["content"]


class TestProviderFactory:
    """Test the model provider factory with new providers."""

    @pytest.fixture
    def mock_clients(self):
        """Mock OpenAI, Anthropic, and Google clients."""
        with patch("reactive_agents.providers.llm.openai.OpenAI") as mock_openai:
            with patch(
                "reactive_agents.providers.llm.anthropic.Anthropic"
            ) as mock_anthropic:
                with patch("reactive_agents.providers.llm.google.genai") as mock_google:
                    # Mock OpenAI models
                    mock_models = Mock()
                    mock_models.data = [Mock(id="gpt-4"), Mock(id="gpt-3.5-turbo")]
                    mock_openai.return_value.models.list.return_value = mock_models

                    # Mock Google models
                    mock_google_model = Mock()
                    mock_google_model.name = "models/gemini-pro"
                    mock_google_model.supported_generation_methods = ["generateContent"]
                    mock_google.list_models.return_value = [mock_google_model]
                    mock_google.GenerativeModel.return_value = Mock()

                    yield mock_openai, mock_anthropic, mock_google

    def test_factory_supports_new_providers(self, mock_clients):
        """Test that factory recognizes new providers."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "ANTHROPIC_API_KEY": "test-key",
                "GOOGLE_API_KEY": "test-key",
            },
        ):
            # Test OpenAI
            openai_provider = ModelProviderFactory.get_model_provider("openai:gpt-4")
            assert isinstance(openai_provider, OpenAIModelProvider)

            # Test Anthropic
            anthropic_provider = ModelProviderFactory.get_model_provider(
                "anthropic:claude-3-sonnet-20240229"
            )
            assert isinstance(anthropic_provider, AnthropicModelProvider)

            # Test Google
            google_provider = ModelProviderFactory.get_model_provider(
                "google:gemini-pro"
            )
            assert isinstance(google_provider, GoogleModelProvider)

    def test_factory_provider_registration(self, mock_clients):
        """Test that providers are properly registered."""
        # Check that providers are in the factory's provider list
        from reactive_agents.providers.llm.base import BaseModelProvider

        providers = BaseModelProvider._providers
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert providers["openai"] == OpenAIModelProvider
        assert providers["anthropic"] == AnthropicModelProvider
        assert providers["google"] == GoogleModelProvider


class TestGoogleProvider:
    """Test cases for Google provider."""

    @pytest.fixture
    def mock_google_genai(self):
        """Mock Google Generative AI."""
        with patch("reactive_agents.providers.llm.google.genai") as mock_genai:
            # Mock model listing
            mock_model = Mock()
            mock_model.name = "models/gemini-pro"
            mock_model.supported_generation_methods = ["generateContent"]
            mock_genai.list_models.return_value = [mock_model]

            # Mock GenerativeModel
            mock_genai.GenerativeModel.return_value = Mock()

            yield mock_genai

    def test_google_provider_initialization(self, mock_google_genai):
        """Test Google provider initialization."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            provider = GoogleModelProvider(model="gemini-pro")
            assert provider.model == "gemini-pro"
            assert provider.id == "google"
            assert provider.name == "google"

    def test_google_provider_missing_api_key(self, mock_google_genai):
        """Test Google provider fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="GOOGLE_API_KEY environment variable is required"
            ):
                GoogleModelProvider(model="gemini-pro")

    def test_google_provider_model_validation(self, mock_google_genai):
        """Test Google provider model validation."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            provider = GoogleModelProvider(model="gemini-pro")
            result = provider.validate_model()
            assert result["valid"] is True
            assert result["model"] == "gemini-pro"

    def test_google_provider_via_factory(self, mock_google_genai):
        """Test creating Google provider via factory."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            provider = ModelProviderFactory.get_model_provider("google:gemini-pro")
            assert isinstance(provider, GoogleModelProvider)
            assert provider.model == "gemini-pro"

    def test_google_provider_safety_settings(self, mock_google_genai):
        """Test Google provider safety settings configuration."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            provider = GoogleModelProvider(model="gemini-pro")

            # Test default safety settings
            assert provider.default_safety_settings is not None

            # Test configuring permissive settings
            provider.configure_safety_settings()

            # Import Google types for testing
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            # Check that safety settings were updated to be more permissive
            assert (
                provider.default_safety_settings[HarmCategory.HARM_CATEGORY_HARASSMENT]
                == HarmBlockThreshold.BLOCK_NONE
            )

    @pytest.mark.asyncio
    async def test_google_provider_safety_blocked_response(self, mock_google_genai):
        """Test Google provider handling of safety-blocked responses."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            provider = GoogleModelProvider(model="gemini-pro")

            # Mock a safety-blocked response
            mock_response = Mock()
            mock_response.candidates = [Mock()]
            mock_response.candidates[0].finish_reason = 3  # SAFETY
            mock_response.candidates[0].content = None
            mock_response.usage_metadata = None

            # Mock the generative model
            provider.generative_model.generate_content.return_value = mock_response

            response = await provider.get_chat_completion(
                messages=[{"role": "user", "content": "Test message"}]
            )

            # Check that the response indicates content filtering
            assert response.done_reason == "content_filter"
            assert "blocked by safety filters" in response.message.content
