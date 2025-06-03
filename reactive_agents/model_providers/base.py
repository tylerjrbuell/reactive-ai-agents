from abc import ABC, ABCMeta, abstractmethod
from typing import Type


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
    async def get_chat_completion(self, **kwargs) -> dict:
        pass

    @abstractmethod
    async def get_completion(self, **kwargs) -> dict:
        pass


model_providers: dict[str, Type[BaseModelProvider]] = {}
