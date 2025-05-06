from .base import BaseModelProvider
import ollama
import os




class OllamaModelProvider(BaseModelProvider):
    id = "ollama"
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    def __init__(self, model="llama3.2", name="ollama"):
        self.name = name
        self.model = model
        self.client = ollama.AsyncClient(host=self.host)
        self.validate_model()
        super().__init__(model=model, name=self.name)

    def validate_model(self, **kwargs):
        models = ollama.Client(host=self.host).list().models
        model = self.model
        if ":" not in self.model:
            model = f"{self.model}:latest"
        available_models = [m.model for m in models]
        if model not in available_models:
            raise Exception(
                f"Model {self.model} is either not supported or has not been downloaded from Ollama. Run `ollama pull {self.model}` to download the model."
            )

    async def get_chat_completion(self, **kwargs) -> ollama.ChatResponse:
        try:
            if not kwargs.get("model"):
                kwargs["model"] = self.model
            result = await self.client.chat(
                model=kwargs["model"],
                messages=kwargs["messages"],
                stream=kwargs["stream"] if kwargs.get("stream") else False,
                tools=kwargs["tools"] if kwargs.get("tools") else [],
                format=kwargs["format"] if kwargs.get("format") else None,
                options=(
                    kwargs["options"]
                    if kwargs.get("options")
                    else {"temperature": 0, "num_gpu": 256, "num_ctx": 10000}
                ),
            )
            return result
        except Exception as e:
            raise Exception(f"Ollama Chat Completion Error: {e}")

    async def get_completion(self, **kwargs):
        if not kwargs.get("model"):
            kwargs["model"] = self.model
        if not kwargs.get("options"):
            kwargs["options"] = {"temperature": 0, "num_gpu": 256, "num_ctx": 10000}
        return ollama.generate(**kwargs)
