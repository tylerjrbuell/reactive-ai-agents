from .base import BaseModelProvider
from typing import Dict, Type, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class ModelProviderFactory:
    """Factory for creating model providers."""

    _providers: Dict[str, Type[BaseModelProvider]] = {}

    @classmethod
    def register_provider(cls, provider_class: Type[BaseModelProvider]) -> None:
        """Register a model provider class."""
        provider_name = provider_class.__name__.replace("ModelProvider", "").lower()
        cls._providers[provider_name] = provider_class

    @classmethod
    def get_model_provider(
        cls,
        model_name: str,
        options: Optional[Dict] = None,
        context: Optional["AgentContext"] = None,
    ) -> BaseModelProvider:
        """
        Get a model provider instance.

        Args:
            model_name: The name of the model provider to get (format: "provider:model")
            options: Optional configuration options for the provider
            context: The agent context for error tracking and logging

        Returns:
            A model provider instance
        """
        if not model_name or ":" not in model_name:
            raise ValueError(
                f"Invalid model provider: {model_name}, use the format provider:model"
            )

        # Extract provider name and model from the provider:model string
        provider_name = model_name.split(":")[0].lower()
        model = model_name.split(":", 1)[1].lower()

        # Sync providers from BaseModelProvider
        cls._providers.update(BaseModelProvider._providers)

        if provider_name not in cls._providers:
            raise ValueError(
                f"Unknown model provider: {provider_name}, supported providers: {list(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_name]
        return provider_class(model=model, options=options, context=context)
