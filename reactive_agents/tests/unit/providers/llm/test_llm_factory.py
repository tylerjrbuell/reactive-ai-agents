"""
Tests for LLM Factory.

Tests the LLM provider factory functionality including provider creation,
configuration, and factory patterns.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from reactive_agents.providers.llm.factory import ModelProviderFactory


class TestLLMFactory:
    """Test cases for LLM Factory."""

    @pytest.fixture
    def llm_factory(self):
        """Create an LLM factory instance."""
        return ModelProviderFactory()

    def test_initialization(self, llm_factory):
        """Test LLM factory initialization."""
        assert llm_factory is not None
        assert hasattr(llm_factory, "get_model_provider")
        assert hasattr(llm_factory, "register_provider")

    def test_create_groq_provider(self, llm_factory):
        """Test creating Groq provider."""
        # Skip if GROQ_API_KEY not available
        if not os.environ.get("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not available")

        with patch("groq.Groq") as mock_groq:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.model_dump.return_value = {
                "data": [{"id": "llama3-groq-70b-8192-tool-use-preview"}]
            }
            mock_client.models.list.return_value = mock_models
            mock_groq.return_value = mock_client

            # Patch validate_model to avoid validation errors
            with patch(
                "reactive_agents.providers.llm.groq.GroqModelProvider.validate_model"
            ):
                provider = llm_factory.get_model_provider(
                    "groq:llama3-groq-70b-8192-tool-use-preview"
                )
                assert provider is not None
                assert provider.id == "groq"
                assert provider.model == "llama3-groq-70b-8192-tool-use-preview"

    def test_create_ollama_provider(self, llm_factory):
        """Test creating Ollama provider."""
        with patch("ollama.Client") as mock_ollama, patch(
            "ollama.AsyncClient"
        ) as mock_async_ollama:
            # Mock the ollama client to avoid actual connection
            mock_client = Mock()
            mock_models = Mock()
            mock_models.models = [Mock(model="llama2:latest")]
            mock_client.list.return_value = mock_models
            mock_ollama.return_value = mock_client

            provider = llm_factory.get_model_provider("ollama:llama2")
            assert provider is not None
            assert provider.id == "ollama"
            assert provider.model == "llama2"

    def test_create_openai_provider(self, llm_factory):
        """Test creating OpenAI provider."""
        # Skip if OPENAI_API_KEY not available
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available")

        with patch("openai.OpenAI") as mock_openai:
            # Mock the OpenAI client
            mock_client = Mock()
            mock_models = Mock()
            mock_models.data = [Mock(id="gpt-4")]
            mock_client.models.list.return_value = mock_models
            mock_openai.return_value = mock_client

            # Patch validate_model to avoid validation errors
            with patch(
                "reactive_agents.providers.llm.openai.OpenAIModelProvider.validate_model"
            ):
                provider = llm_factory.get_model_provider("openai:gpt-4")
                assert provider is not None
                assert provider.id == "openai"
                assert provider.model == "gpt-4"

    def test_create_anthropic_provider(self, llm_factory):
        """Test creating Anthropic provider."""
        # Skip if ANTHROPIC_API_KEY not available
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not available")

        with patch("anthropic.Anthropic") as mock_anthropic:
            provider = llm_factory.get_model_provider(
                "anthropic:claude-3-sonnet-20240229"
            )
            assert provider is not None
            assert provider.id == "anthropic"
            assert provider.model == "claude-3-sonnet-20240229"

    def test_create_google_provider(self, llm_factory):
        """Test creating Google provider."""
        # Skip if GOOGLE_API_KEY not available
        if not os.environ.get("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not available")

        with patch("google.generativeai.configure") as mock_configure, patch(
            "google.generativeai.GenerativeModel"
        ) as mock_model, patch("google.generativeai.list_models") as mock_list:

            # Mock available models
            mock_list.return_value = [
                Mock(
                    name="models/gemini-pro",
                    supported_generation_methods=["generateContent"],
                )
            ]

            # Patch validate_model to avoid validation errors
            with patch(
                "reactive_agents.providers.llm.google.GoogleModelProvider.validate_model"
            ):
                provider = llm_factory.get_model_provider("google:gemini-pro")
                assert provider is not None
                assert provider.id == "google"
                assert provider.model == "gemini-pro"

    def test_create_provider_with_config(self, llm_factory):
        """Test creating provider with configuration."""
        with patch("ollama.Client") as mock_ollama, patch(
            "ollama.AsyncClient"
        ) as mock_async_ollama:
            mock_client = Mock()
            mock_models = Mock()
            mock_models.models = [Mock(model="llama2:latest")]
            mock_client.list.return_value = mock_models
            mock_ollama.return_value = mock_client

            options = {"temperature": 0.5, "max_tokens": 1000}
            provider = llm_factory.get_model_provider("ollama:llama2", options=options)
            assert provider is not None
            assert provider.options == options

    def test_get_available_providers(self, llm_factory):
        """Test getting available providers."""
        # Access the internal _providers dict after syncing
        from reactive_agents.providers.llm.base import BaseModelProvider

        llm_factory._providers.update(BaseModelProvider._providers)

        providers = list(llm_factory._providers.keys())
        assert isinstance(providers, list)
        assert len(providers) > 0

        # Check that all expected providers are available
        expected_providers = ["ollama", "openai", "anthropic", "groq", "google"]
        for expected in expected_providers:
            assert expected in providers

    def test_provider_interface_consistency(self, llm_factory):
        """Test that all providers implement the same interface consistently."""
        provider_types = ["ollama", "openai", "anthropic", "groq", "google"]

        for provider_type in provider_types:
            try:
                with patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "test_key",
                        "ANTHROPIC_API_KEY": "test_key",
                        "GROQ_API_KEY": "test_key",
                        "GOOGLE_API_KEY": "test_key",
                    },
                ):
                    # Mock various clients to avoid actual API calls
                    with patch("ollama.Client"), patch("ollama.AsyncClient"), patch(
                        "openai.OpenAI"
                    ), patch("anthropic.Anthropic"), patch("groq.Groq"), patch(
                        "google.generativeai.configure"
                    ), patch(
                        "google.generativeai.GenerativeModel"
                    ), patch(
                        "google.generativeai.list_models"
                    ) as mock_list_models:

                        # Mock model validation for all providers
                        if provider_type == "ollama":
                            mock_client = Mock()
                            mock_models = Mock()
                            mock_models.models = [Mock(model="test_model:latest")]
                            mock_client.list.return_value = mock_models
                            with patch(
                                "ollama.Client", return_value=mock_client
                            ), patch(
                                "reactive_agents.providers.llm.ollama.OllamaModelProvider.validate_model"
                            ):
                                provider = llm_factory.get_model_provider(
                                    f"{provider_type}:test_model"
                                )
                        elif provider_type == "openai":
                            mock_client = Mock()
                            mock_models = Mock()
                            mock_models.data = [Mock(id="test_model")]
                            mock_client.models.list.return_value = mock_models
                            with patch(
                                "openai.OpenAI", return_value=mock_client
                            ), patch(
                                "reactive_agents.providers.llm.openai.OpenAIModelProvider.validate_model"
                            ):
                                provider = llm_factory.get_model_provider(
                                    f"{provider_type}:test_model"
                                )
                        elif provider_type == "groq":
                            mock_client = Mock()
                            mock_models = Mock()
                            mock_models.model_dump.return_value = {
                                "data": [{"id": "test_model"}]
                            }
                            mock_client.models.list.return_value = mock_models
                            with patch("groq.Groq", return_value=mock_client), patch(
                                "reactive_agents.providers.llm.groq.GroqModelProvider.validate_model"
                            ):
                                provider = llm_factory.get_model_provider(
                                    f"{provider_type}:test_model"
                                )
                        elif provider_type == "google":
                            mock_list_models.return_value = [
                                Mock(
                                    name="models/test_model",
                                    supported_generation_methods=["generateContent"],
                                )
                            ]
                            with patch(
                                "reactive_agents.providers.llm.google.GoogleModelProvider.validate_model"
                            ):
                                provider = llm_factory.get_model_provider(
                                    f"{provider_type}:test_model"
                                )
                        else:  # anthropic
                            # Mock the Anthropic client properly to avoid initialization issues
                            mock_anthropic_client = Mock()
                            with patch(
                                "reactive_agents.providers.llm.anthropic.Anthropic",
                                return_value=mock_anthropic_client,
                            ), patch(
                                "reactive_agents.providers.llm.anthropic.AnthropicModelProvider.validate_model"
                            ):
                                provider = llm_factory.get_model_provider(
                                    f"{provider_type}:claude-3-sonnet-20240229"
                                )

                        # Test that all providers have consistent interface
                        assert hasattr(
                            provider, "validate_model"
                        ), f"{provider_type} missing validate_model"
                        assert hasattr(
                            provider, "get_chat_completion"
                        ), f"{provider_type} missing get_chat_completion"
                        assert hasattr(
                            provider, "get_completion"
                        ), f"{provider_type} missing get_completion"
                        assert hasattr(provider, "id"), f"{provider_type} missing id"
                        assert hasattr(
                            provider, "model"
                        ), f"{provider_type} missing model"
                        assert hasattr(
                            provider, "options"
                        ), f"{provider_type} missing options"
                        assert hasattr(
                            provider, "context"
                        ), f"{provider_type} missing context"

                        # Skip validate_model test since we're testing interface consistency, not functionality
                        # The validate_model method is tested in individual provider tests

            except Exception as e:
                pytest.fail(
                    f"Provider {provider_type} failed interface consistency test: {e}"
                )

    def test_invalid_provider_format(self, llm_factory):
        """Test that invalid provider format raises appropriate error."""
        with pytest.raises(ValueError, match="use the format provider:model"):
            llm_factory.get_model_provider("invalid_format")

    def test_unknown_provider(self, llm_factory):
        """Test that unknown provider raises appropriate error."""
        with pytest.raises(ValueError, match="Unknown model provider"):
            llm_factory.get_model_provider("unknown:model")
