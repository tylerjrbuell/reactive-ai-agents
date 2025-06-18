from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
from typing import List, Optional, Type, Dict, Any, Union, TYPE_CHECKING
from pydantic import BaseModel
import traceback
import time

from reactive_agents.common.types.status_types import TaskStatus
from reactive_agents.common.types.event_types import AgentStateEvent

if TYPE_CHECKING:
    from reactive_agents.context.agent_context import AgentContext


class ChatCompletionMessage(BaseModel):
    role: str
    content: str
    thinking: Optional[str]
    tool_calls: Optional[list] = None
    images: Optional[list] = None


class ChatCompletionResponse(BaseModel):
    message: ChatCompletionMessage
    model: str
    done: bool
    done_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    prompt_eval_duration: Optional[float]
    load_duration: Optional[float]
    total_duration: Optional[float]
    created_at: Optional[int]


class CompletionResponse(BaseModel):
    content: str
    model: str
    done: bool
    done_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    prompt_eval_duration: Optional[int]
    load_duration: Optional[int]
    total_duration: Optional[int]
    created_at: str


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

    @abstractmethod
    async def validate_model(self, **kwargs) -> dict:
        pass

    @abstractmethod
    async def get_chat_completion(self, **kwargs) -> ChatCompletionResponse:
        """
        Abstract method to get a chat completion from the model.

        Args:
            **kwargs: Arbitrary keyword arguments. Should include 'messages' (list of dicts) and may include 'options' (dict) for model-specific parameters like temperature or num_ctx.
        """
        pass

    @abstractmethod
    async def get_completion(self, **kwargs) -> dict:
        """
        Abstract method to get a text completion from the model.

        Args:
            **kwargs: Arbitrary keyword arguments. Should include 'prompt' (str) and may include 'options' (dict) for model-specific parameters.
        """
        pass
