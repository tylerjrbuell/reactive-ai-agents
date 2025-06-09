from abc import ABC, ABCMeta, abstractmethod
from typing import List, Optional, Type
from pydantic import BaseModel


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
    prompt_eval_duration: Optional[int]
    load_duration: Optional[int]
    total_duration: Optional[int]
    created_at: Optional[str]


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


class AutoRegisterModelMeta(ABCMeta):
    def __init__(cls, name: str, bases: tuple, attrs: dict):
        super().__init__(name, bases, attrs)
        # Exclude abstract classes
        if not any(base.__name__ == "ABC" for base in bases):
            model_providers[attrs.get("id")] = cls  # type: ignore
            # print(f"Registered model provider: {attrs.get('id', name)}")


class BaseModelProvider(ABC, metaclass=AutoRegisterModelMeta):

    def __init__(self, model: str, name: str = ""):
        self.name = name
        self.model = model

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


model_providers: dict[str, Type[BaseModelProvider]] = {}
