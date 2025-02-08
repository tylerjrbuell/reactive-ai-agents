from typing import Literal, Sequence
from .base import BaseModelProvider
import ollama
import asyncio


class OllamaModelProvider(BaseModelProvider):
    id = "ollama"

    def __init__(self, model="llama3.2", name="ollama"):
        self.name = name
        self.model = model
        self.client = ollama.AsyncClient()
        self.validate_model()
        super().__init__(model=model, name=self.name)

    def validate_model(self, **kwargs):
        models = ollama.Client().list().models
        if self.model not in [m.model for m in models]:
            raise Exception(
                f"Model {self.model} is either not supported or has not been downloaded from Ollama. Run `ollama pull {self.model}` to download the model."
            )

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
