from .base import BaseModelProvider, model_providers
import importlib, os
from pathlib import Path

non_provider_files = ["__init__.py", "base.py", "factory.py"]


class ModelProviderFactory:

    supported_providers = ["ollama", "groq"]

    @staticmethod
    def get_model_provider(provider_model: str) -> BaseModelProvider:
        """
        Get the model provider class instance based on the provider:model string ID provided
        Example provider_model: 'ollama:llama3.1'

        Args:
            provider_model (str): The provider:model string ID

        Raises:
            ValueError: Invalid provider:model ID

        Returns:
            BaseModelProvider: The model provider class instance
        """
        if not provider_model or ":" not in provider_model:
            raise ValueError(
                f"Invalid model provider: {provider_model}, use the format provider:model"
            )

        provider = provider_model.split(":")[0].lower()
        model = provider_model.split(":", 1)[1].lower()
        if provider not in ModelProviderFactory.supported_providers:
            raise ValueError(
                f"Invalid model provider: {provider}, supported providers: {ModelProviderFactory.supported_providers}"
            )
        try:
            valid_providers = [
                f.replace(".py", "")
                for f in os.listdir(Path(__file__).parent)
                if os.path.isfile(os.path.join(Path(__file__).parent, f))
                and f not in non_provider_files
            ]
            if provider not in valid_providers:
                raise ValueError(f"Invalid model provider id: {provider}")
            importlib.import_module(f"model_providers.{provider}")
            if not model_providers.get(provider):
                raise ValueError(f"Invalid model provider id: {provider}")
            return model_providers[provider](model)
        except ImportError:
            raise ValueError(f"Invalid model provider id: {provider} not found")
