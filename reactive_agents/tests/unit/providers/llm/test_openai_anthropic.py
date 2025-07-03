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


class TestProviderFactory:
    """Test the model provider factory with new providers."""

    @pytest.fixture
    def mock_clients(self):
        """Mock both OpenAI and Anthropic clients."""
        with patch("reactive_agents.providers.llm.openai.OpenAI") as mock_openai:
            with patch(
                "reactive_agents.providers.llm.anthropic.Anthropic"
            ) as mock_anthropic:
                # Mock OpenAI models
                mock_models = Mock()
                mock_models.data = [Mock(id="gpt-4"), Mock(id="gpt-3.5-turbo")]
                mock_openai.return_value.models.list.return_value = mock_models

                yield mock_openai, mock_anthropic

    def test_factory_supports_new_providers(self, mock_clients):
        """Test that factory recognizes new providers."""
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"}
        ):
            # Test OpenAI
            openai_provider = ModelProviderFactory.get_model_provider("openai:gpt-4")
            assert isinstance(openai_provider, OpenAIModelProvider)

            # Test Anthropic
            anthropic_provider = ModelProviderFactory.get_model_provider(
                "anthropic:claude-3-sonnet-20240229"
            )
            assert isinstance(anthropic_provider, AnthropicModelProvider)

    def test_factory_provider_registration(self, mock_clients):
        """Test that providers are properly registered."""
        # Check that providers are in the factory's provider list
        from reactive_agents.providers.llm.base import BaseModelProvider

        providers = BaseModelProvider._providers
        assert "openai" in providers
        assert "anthropic" in providers
        assert providers["openai"] == OpenAIModelProvider
        assert providers["anthropic"] == AnthropicModelProvider
