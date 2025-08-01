from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
from typing import List, Optional, Type, Dict, Any, Union, TYPE_CHECKING
import traceback
import time

# Import provider types from centralized location
from reactive_agents.core.types.provider_types import (
    CompletionMessage,
    CompletionResponse,
    ModelInfo,
    ProviderStatus,
    ProviderHealth,
)

from reactive_agents.core.types.status_types import TaskStatus
from reactive_agents.core.types.event_types import AgentStateEvent

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


# Dictionary to store registered model providers
model_providers: Dict[str, Type["BaseModelProvider"]] = {}


class AutoRegisterModelMeta(ABCMeta):
    """Metaclass for auto-registering model providers."""

    def __init__(cls, name: str, bases: tuple, attrs: dict):
        super().__init__(name, bases, attrs)
        if name != "BaseModelProvider":
            BaseModelProvider.register_provider(cls)


class BaseModelProvider(ABC, metaclass=AutoRegisterModelMeta):
    """Base class for model providers."""

    _providers: Dict[str, Type["BaseModelProvider"]] = {}

    @classmethod
    def register_provider(cls, provider_class: Type["BaseModelProvider"]) -> None:
        """Register a model provider class."""
        provider_name = provider_class.__name__.replace("ModelProvider", "").lower()
        cls._providers[provider_name] = provider_class

    def __init__(
        self,
        model: str,
        options: Optional[Dict[str, Any]] = None,
        context: Optional["AgentContext"] = None,
    ):
        """
        Initialize the model provider.

        Args:
            model: The model to use
            options: Optional configuration options
            context: The agent context for error tracking and logging
        """
        self.model = model
        self.options = options or {}
        self.context = context
        self.name = self.__class__.__name__.replace("ModelProvider", "").lower()

    def _handle_error(self, error: Exception, operation: str) -> None:
        """
        Centralized error handling for model provider operations.

        Args:
            error: The exception that occurred
            operation: The operation that failed (e.g., 'completion', 'chat_completion', 'validation')
        """
        if not self.context:
            raise error  # If no context, just raise the error

        error_data = {
            "error": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": time.time(),
            "error_type": "critical",
            "component": "model_provider",
            "operation": operation,
            "provider": self.name,
            "model": self.model,
        }

        # Add to errors list
        if hasattr(self.context, "session") and self.context.session:
            self.context.session.errors.append(error_data)
            self.context.session.error = str(error)
            self.context.session.task_status = TaskStatus.ERROR

        # Log the error
        if hasattr(self.context, "agent_logger") and self.context.agent_logger:
            self.context.agent_logger.error(
                f"Model provider error during {operation}: {error}\n{traceback.format_exc()}"
            )

        # Emit error event
        if hasattr(self.context, "emit_event"):
            self.context.emit_event(
                AgentStateEvent.ERROR_OCCURRED,
                {
                    "error": f"Model provider {operation} error",
                    "details": str(error),
                    "provider": self.name,
                    "model": self.model,
                    "is_critical": True,
                },
            )

        raise error  # Re-raise the error after handling

    def _adapt_context_for_provider(self, messages: List[dict], tools: Optional[List[dict]] = None) -> List[dict]:
        """
        Provider-specific context adaptation hook.
        
        This method allows providers to adapt agent context to fit their specific
        SDK requirements and best practices without corrupting the agent's core context.
        
        Design Principles: 
        - PRESERVE: Never replace agent role, instructions, or task context
        - ADAPT: Modify format/structure to fit provider requirements
        - MINIMAL: Keep adaptations focused and lightweight
        - ADDITIVE: Append provider-specific hints when necessary
        
        Args:
            messages: Original message chain from agent context
            tools: Available tools for the interaction
            
        Returns:
            Messages adapted for this provider's requirements
            
        Default implementation returns messages unchanged.
        Providers should override to add specific adaptations.
        
        Examples of valid adaptations:
        - Message role conversions (tool -> user for OpenAI)
        - Tool format conversions (OpenAI -> Anthropic format)
        - Adding provider-specific system context hints
        - Message ordering adjustments for API requirements
        
        Examples of INVALID adaptations:
        - Replacing agent role ("coding assistant" -> "helpful assistant")
        - Overriding agent instructions or task context
        - Changing the agent's personality or behavior guidelines
        """
        return messages

    @abstractmethod
    async def validate_model(self, **kwargs) -> dict:
        pass

    @abstractmethod
    async def get_chat_completion(self, **kwargs) -> CompletionResponse:
        """
        Abstract method to get a chat completion from the model.

        Args:
            **kwargs: Arbitrary keyword arguments. Should include:
                - messages: List of message dictionaries
                - options: Dict of model-specific parameters (temperature, max_tokens, etc.)
                - format: Response format ("json" or "")
                - stream: Whether to stream the response
                - tools: List of tool/function definitions
                - tool_choice: Tool choice preference
        """
        pass

    @abstractmethod
    async def get_completion(self, **kwargs) -> CompletionResponse:
        """
        Abstract method to get a text completion from the model.

        Args:
            **kwargs: Arbitrary keyword arguments. Should include:
                - prompt: The input prompt string
                - system: Optional system message
                - options: Dict of model-specific parameters
                - format: Response format ("json" or "")
        """
        pass

    def extract_and_store_thinking(
        self,
        message: CompletionMessage,
        call_context: str = "unknown",
    ) -> CompletionMessage:
        """
        Extracts <think> content from the message, cleans the message content, and stores the thinking in the context if present.

        Args:
            message: The message object (dict) from the model response
            call_context: The context of the call (e.g., "think_chain", "summary_generation")

        Returns:
            The updated message dict with cleaned content and thinking removed from content.
        """
        content = message.content
        if not content:
            return message

        think_start = content.find("<think>")
        think_end = content.find("</think>")
        thinking_content = None
        if think_start != -1 and think_end != -1 and think_end > think_start:
            thinking_content = content[think_start + 7 : think_end].strip()
            cleaned_content = (content[:think_start] + content[think_end + 8 :]).strip()
            message.content = cleaned_content
        else:
            message.content = content.strip()

        if thinking_content and self.context and hasattr(self.context, "session"):
            thinking_entry = {
                "timestamp": time.time(),
                "call_context": call_context,
                "thinking": thinking_content,
            }
            self.context.session.thinking_log.append(thinking_entry)
            if hasattr(self.context, "agent_logger") and self.context.agent_logger:
                self.context.agent_logger.debug(
                    f"Stored thinking for {call_context}: {thinking_content[:100]}..."
                )
        return message
