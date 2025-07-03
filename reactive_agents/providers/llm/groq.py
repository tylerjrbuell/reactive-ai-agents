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
        self.name = __class__.id
        self.model = model
        self.options = options
        self.context = context
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.validate_model()
        super().__init__(model=model, options=options, context=context)

    def _clean_message(self, msg: dict):
        allowed = {"role", "content", "name", "tool_call_id"}
        return {k: v for k, v in msg.items() if k in allowed}

    def validate_model(self, **kwargs):
        supported_models = self.client.models.list().model_dump().get("data", [])
        if self.model not in [m.get("id") for m in supported_models]:
            raise Exception(f"Model {self.model} is not currently supported by Groq.")

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
                        completion.usage.prompt_tokens if completion.usage else None
                    ),
                    completion_tokens=(
                        completion.usage.completion_tokens if completion.usage else None
                    ),
                    prompt_eval_duration=(
                        completion.usage.prompt_time if completion.usage else None
                    ),
                    load_duration=(
                        completion.usage.completion_time if completion.usage else None
                    ),
                    total_duration=(
                        completion.usage.total_time if completion.usage else None
                    ),
                    created_at=str(completion.created) if completion.created else None,
                )
            elif type(completion) is Stream[ChatCompletionChunk]:
                return completion
        except InternalServerError as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"Groq Chat Completion Error: {e.message}")
        except BadRequestError as e:
            self._handle_error(e, "chat_completion")
            # Check for tool_use_failed code and handle gracefully
            error_data = getattr(e, "response", None)
            if error_data and hasattr(error_data, "json"):
                error_json = error_data.json()
                if (
                    isinstance(error_json, dict)
                    and error_json.get("error", {}).get("code") == "tool_use_failed"
                ):
                    # Log and return a special response or None
                    # You can customize this as needed
                    print(
                        f"Tool use failed: {error_json['error'].get('failed_generation')}"
                    )
                    return None
            # For other BadRequestErrors, you may want to re-raise or handle differently
            raise Exception(f"Groq Chat Completion Error: {e.message}")
        except Exception as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"Groq Chat Completion Error: {e}")

    async def get_completion(
        self, **kwargs
    ) -> CompletionResponse | Stream[ChatCompletionChunk] | None:
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        try:
            completion = client.chat.completions.create(
                model=self.model,
                tools=kwargs["tools"] if kwargs.get("tools") else None,
                messages=[
                    {"role": "system", "content": kwargs["system"]},
                    {"role": "user", "content": kwargs["prompt"]},
                ],
                response_format=(
                    {"type": "json_object"}
                    if kwargs.get("format") == "json"
                    else {"type": "text"}
                ),
            )
            if type(completion) is ChatCompletion:
                result = completion.choices[0]
                message = CompletionMessage(
                    content=result.message.content if result.message.content else "",
                    role="assistant",
                    thinking=None,
                    tool_calls=None,
                )
                return CompletionResponse(
                    message=self.extract_and_store_thinking(
                        message, call_context="completion"
                    ),
                    model=completion.model or self.model,
                    done=True,
                    done_reason=result.finish_reason or None,
                    prompt_tokens=(
                        completion.usage.prompt_tokens if completion.usage else None
                    ),
                    completion_tokens=(
                        completion.usage.completion_tokens if completion.usage else None
                    ),
                    prompt_eval_duration=(
                        completion.usage.prompt_time if completion.usage else None
                    ),
                    load_duration=(
                        completion.usage.completion_time if completion.usage else None
                    ),
                    total_duration=(
                        completion.usage.total_time if completion.usage else None
                    ),
                    created_at=str(completion.created) if completion.created else None,
                )
            elif type(completion) is Stream[ChatCompletionChunk]:
                return completion

        except InternalServerError as e:
            self._handle_error(e, "completion")
            # For InternalServerError, you may want to re-raise or handle differently
            raise Exception(f"Groq Internal Server Error: {e.message}")
        except BadRequestError as e:
            self._handle_error(e, "completion")
            # Check for tool_use_failed code and handle gracefully
            error_data = getattr(e, "response", None)
            if error_data and hasattr(error_data, "json"):
                error_json = error_data.json()
                if (
                    isinstance(error_json, dict)
                    and error_json.get("error", {}).get("code") == "tool_use_failed"
                ):
                    # Log and return a special response or None
                    # You can customize this as needed
                    print(
                        f"Tool use failed: {error_json['error'].get('failed_generation')}"
                    )
                    return None
            # For other BadRequestErrors, you may want to re-raise or handle differently
            raise Exception(f"Groq Completion Error: {e.message}")

        except Exception as e:
            self._handle_error(e, "completion")
            raise Exception(f"Groq Completion Error: {e}")
