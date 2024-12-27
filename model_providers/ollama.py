from typing import Literal, Sequence
from .base import BaseModelProvider
import ollama


class OllamaModelProvider(BaseModelProvider):
    id = "ollama"

    def __init__(self, model="llama3.2", name="ollama"):
        self.name = name
        self.model = model
        self.client = ollama.AsyncClient()
        super().__init__(model=model, name=self.name)

    async def get_chat_completion(self, **kwargs) -> ollama.ChatResponse:
        try:
            result = await self.client.chat(
                model=self.model,
                messages=kwargs["messages"],
                stream=kwargs["stream"] if kwargs.get("stream") else False,
                tools=kwargs["tools"] if kwargs.get("tools") else [],
                format=kwargs["format"] if kwargs.get("format") else None,
                options=(
                    kwargs["options"]
                    if kwargs.get("options")
                    else {"temperature": 0, "num_gpu": 256, "num_ctx": 15000}
                ),
            )
            return result
        except Exception as e:
            raise Exception(f"Ollama Chat Completion Error: {e}")

    async def get_completion(self, **kwargs):
        return ollama.generate(model=self.model, **kwargs)
