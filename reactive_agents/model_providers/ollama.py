from .base import BaseModelProvider, ChatCompletionMessage, ChatCompletionResponse
import ollama
import os
import time
from typing import Optional, Dict, Any

DEFAULT_OPTIONS = {"temperature": 0.2, "num_gpu": 256, "num_ctx": 4000, "think": False}


class OllamaModelProvider(BaseModelProvider):
    id = "ollama"
    host = DEFAULT_OPTIONS.get("host") or os.getenv(
        "OLLAMA_HOST", "http://localhost:11434"
    )

    def __init__(
        self, model: str, options: Optional[Dict[str, Any]] = None, context=None
    ):
        """
        Initialize the Ollama model provider.

        Args:
            model: The model to use
            options: Optional configuration options
            context: The agent context for error tracking and logging
        """
        super().__init__(model=model, options=options, context=context)
        self.client = ollama.AsyncClient(host=self.host)
        self.validate_model()

    def validate_model(self, **kwargs):
        try:
            models = ollama.Client(host=self.host).list().models
            model = self.model
            if ":" not in self.model:
                model = f"{self.model}:latest"
            available_models = [m.model for m in models]
            if model not in available_models:
                raise Exception(
                    f"Model {self.model} is either not supported or has not been downloaded from Ollama. Run `ollama pull {self.model}` to download the model."
                )
        except Exception as e:
            self._handle_error(e, "validation")

    async def get_chat_completion(self, **kwargs) -> ChatCompletionResponse:
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
                    kwargs["options"] if kwargs.get("options") else DEFAULT_OPTIONS
                ),
                think=kwargs.get("options", {}).get("think", False),
            )

            return ChatCompletionResponse(
                message=ChatCompletionMessage(
                    content=result.message.content or "",
                    thinking=result.message.thinking,
                    role=result.message.role or "assistant",
                    tool_calls=(
                        list(result.message.tool_calls)
                        if result.message.tool_calls
                        else None
                    ),
                ),
                model=result.model or self.model,
                done=result.done or False,
                done_reason=result.done_reason,
                prompt_tokens=result.prompt_eval_count,
                completion_tokens=result.eval_count,
                prompt_eval_duration=result.eval_duration,
                load_duration=result.load_duration,
                total_duration=result.total_duration,
                created_at=int(time.time() * 1000),  # Convert to milliseconds timestamp
            )
        except Exception as e:
            self._handle_error(e, "chat_completion")
            # This line will never be reached due to _handle_error raising the exception
            # But we need it for type checking
            raise

    async def get_completion(self, **kwargs):
        try:
            if not kwargs.get("model"):
                kwargs["model"] = self.model
            if not kwargs.get("options"):
                kwargs["options"] = DEFAULT_OPTIONS
            else:
                kwargs["options"].update(DEFAULT_OPTIONS)
            return ollama.generate(**kwargs)
        except Exception as e:
            self._handle_error(e, "completion")
            # This line will never be reached due to _handle_error raising the exception
            # But we need it for type checking
            raise
