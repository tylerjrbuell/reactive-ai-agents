import json
import os
from groq import BadRequestError, Groq, InternalServerError, Stream
from groq.types.chat import ChatCompletion, ChatCompletionChunk
from .base import BaseModelProvider, CompletionMessage, CompletionResponse


class GroqModelProvider(BaseModelProvider):
    id = "groq"

    def __init__(
        self, model="llama3-groq-70b-8192-tool-use-preview", options=None, context=None
    ):
        # Call parent __init__ first for consistency
        super().__init__(model=model, options=options, context=context)

        # Initialize Groq client
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")

        self.client = Groq(api_key=api_key)

        # Validate model after initialization
        self.validate_model()

    def _clean_message(self, msg: dict):
        allowed = {"role", "content", "name", "tool_call_id"}
        return {k: v for k, v in msg.items() if k in allowed}

    def validate_model(self, **kwargs) -> dict:
        """Validate that the model is supported by Groq."""
        try:
            supported_models = self.client.models.list().model_dump().get("data", [])
            available_models = [m.get("id") for m in supported_models]

            if self.model not in available_models:
                raise ValueError(
                    f"Model '{self.model}' is not supported by Groq. "
                    f"Available models: {', '.join(available_models[:10])}..."
                )

            return {"valid": True, "model": self.model}
        except Exception as e:
            self._handle_error(e, "validation")
            return {"valid": False, "error": str(e)}

    async def get_chat_completion(
        self,
        messages: list,
        stream: bool = False,
        tools: list | None = None,
        tool_use: bool = True,
        options: dict | None = None,
        format: str = "",
    ) -> CompletionResponse | Stream[ChatCompletionChunk] | None:
        try:
            messages = [self._clean_message(m) for m in messages]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools and tool_use else None,
                response_format=(
                    {"type": "json_object"} if format == "json" else {"type": "text"}
                ),
                tool_choice="required" if tools and tool_use else "none",
                stream=stream,
                **(options or {}),
            )
            if type(completion) is ChatCompletion:
                result = completion.choices[0]
                message = CompletionMessage(
                    content=result.message.content if result.message.content else "",
                    role=result.message.role if result.message.role else "assistant",
                    thinking="False",
                    tool_calls=(
                        [call.model_dump() for call in result.message.tool_calls]
                        if result.message.tool_calls
                        else None
                    ),
                )
                return CompletionResponse(
                    message=self.extract_and_store_thinking(
                        message, call_context="chat_completion"
                    ),
                    model=completion.model or self.model,
                    done=True,
                    done_reason=result.finish_reason or None,
                    prompt_tokens=(
                        int(completion.usage.prompt_tokens or 0)
                        if completion.usage
                        else 0
                    ),
                    completion_tokens=(
                        int(completion.usage.completion_tokens)
                        if completion.usage
                        else 0
                    ),
                    prompt_eval_duration=(
                        int(completion.usage.prompt_time or 0)
                        if completion.usage
                        else 0
                    ),
                    load_duration=(
                        int(completion.usage.completion_time or 0)
                        if completion.usage
                        else 0
                    ),
                    total_duration=(
                        int(completion.usage.total_time or 0) if completion.usage else 0
                    ),
                    created_at=str(completion.created) if completion.created else None,
                )
            elif type(completion) is Stream[ChatCompletionChunk]:
                return completion
        except InternalServerError as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"Groq Internal Server Error: {e.message}")
        except BadRequestError as e:
            # Handle tool_use_failed gracefully but still use proper error handling
            error_data = getattr(e, "response", None)
            if error_data and hasattr(error_data, "json"):
                error_json = error_data.json()
                if (
                    isinstance(error_json, dict)
                    and error_json.get("error", {}).get("code") == "tool_use_failed"
                ):
                    # Log the specific tool failure but still raise an exception for consistency
                    tool_error = error_json["error"].get(
                        "failed_generation", "Tool use failed"
                    )
                    self._handle_error(e, "chat_completion")
                    raise Exception(f"Groq Tool Use Failed: {tool_error}")

            self._handle_error(e, "chat_completion")
            raise Exception(f"Groq Bad Request Error: {e.message}")
        except Exception as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"Groq Chat Completion Error: {e}")

    async def get_completion(
        self, **kwargs
    ) -> CompletionResponse | Stream[ChatCompletionChunk] | None:
        try:
            # Build messages for consistency with other providers
            messages = []
            if kwargs.get("system"):
                messages.append({"role": "system", "content": kwargs["system"]})
            messages.append({"role": "user", "content": kwargs.get("prompt", "")})

            # Use chat completion for text completion (like other providers)
            return await self.get_chat_completion(
                messages=messages,
                tools=kwargs.get("tools"),
                format=kwargs.get("format", ""),
                options=kwargs.get("options"),
            )

        except Exception as e:
            self._handle_error(e, "completion")
            raise Exception(f"Groq Completion Error: {e}")
