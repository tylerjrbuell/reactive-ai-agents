import json
import os
import time
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai import OpenAIError, RateLimitError, APITimeoutError

from .base import BaseModelProvider, CompletionMessage, CompletionResponse


class OpenAIModelProvider(BaseModelProvider):
    """OpenAI model provider using the official OpenAI Python SDK."""

    id = "openai"

    def __init__(
        self,
        model: str = "gpt-4",
        options: Optional[Dict[str, Any]] = None,
        context=None,
    ):
        """
        Initialize the OpenAI model provider.

        Args:
            model: The model to use (e.g., "gpt-4", "gpt-3.5-turbo", "gpt-4o")
            options: Optional configuration options
            context: The agent context for error tracking and logging
        """
        super().__init__(model=model, options=options, context=context)

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = OpenAI(api_key=api_key)

        # Default options
        self.default_options = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        # Validate model on initialization
        self.validate_model()

    def _clean_message(self, msg: dict) -> dict:
        """Clean message to only include fields supported by OpenAI API."""
        allowed = {"role", "content", "name", "tool_call_id", "tool_calls"}
        cleaned = {k: v for k, v in msg.items() if k in allowed}

        # Ensure required fields are present
        if "role" not in cleaned:
            cleaned["role"] = "user"
        if "content" not in cleaned:
            cleaned["content"] = ""

        return cleaned

    def validate_model(self, **kwargs) -> dict:
        """Validate that the model is supported by OpenAI."""
        try:
            # Get available models
            models = self.client.models.list()
            available_models = [model.id for model in models.data]

            if self.model not in available_models:
                raise ValueError(
                    f"Model '{self.model}' is not available. "
                    f"Available models: {', '.join(available_models[:10])}..."
                )

            return {"valid": True, "model": self.model}
        except Exception as e:
            self._handle_error(e, "validation")
            return {"valid": False, "error": str(e)}

    async def get_chat_completion(
        self,
        messages: List[dict],
        stream: bool = False,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        options: Optional[Dict[str, Any]] = None,
        format: str = "",
        **kwargs,
    ) -> Union[CompletionResponse, Any]:
        """
        Get a chat completion from OpenAI.

        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response
            tools: List of tool definitions
            tool_choice: Tool choice preference
            options: Model-specific options
            format: Response format ("json" or "")
            **kwargs: Additional arguments
        """
        try:
            # Clean messages
            cleaned_messages = [self._clean_message(msg) for msg in messages]

            # Merge options
            merged_options = {**self.default_options, **(options or {})}

            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": cleaned_messages,
                "stream": stream,
                **merged_options,
            }

            # Add optional parameters
            if tools:
                api_params["tools"] = tools
                if tool_choice is None:
                    api_params["tool_choice"] = "auto"
                else:
                    api_params["tool_choice"] = tool_choice

            if format == "json":
                api_params["response_format"] = {"type": "json_object"}

            # Create completion
            completion = self.client.chat.completions.create(**api_params)

            if stream:
                return completion  # Return stream object directly

            # Process non-streaming response
            result = completion.choices[0]

            # Extract tool calls if present
            tool_calls = None
            if result.message.tool_calls:
                tool_calls = [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in result.message.tool_calls
                ]

            message = CompletionMessage(
                content=result.message.content or "",
                role=result.message.role,
                tool_calls=tool_calls,
            )

            return CompletionResponse(
                message=self.extract_and_store_thinking(
                    message, call_context="chat_completion"
                ),
                model=completion.model,
                done=True,
                done_reason=result.finish_reason,
                prompt_tokens=(
                    int(completion.usage.prompt_tokens or 0) if completion.usage else 0
                ),
                completion_tokens=(
                    int(completion.usage.completion_tokens or 0)
                    if completion.usage
                    else 0
                ),
                total_duration=None,  # OpenAI doesn't provide timing info
                created_at=str(completion.created),
            )

        except RateLimitError as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"OpenAI Rate Limit Error: {str(e)}")
        except APITimeoutError as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"OpenAI API Timeout Error: {str(e)}")
        except OpenAIError as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"OpenAI API Error: {str(e)}")
        except Exception as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"OpenAI Chat Completion Error: {str(e)}")

    async def get_completion(
        self,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        format: str = "",
        **kwargs,
    ) -> CompletionResponse:
        """
        Get a text completion from OpenAI using the chat completions API.

        Args:
            prompt: The prompt text
            system: Optional system message
            options: Model-specific options
            format: Response format ("json" or "")
            **kwargs: Additional arguments
        """
        try:
            # Build messages
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # Use chat completion for text completion
            return await self.get_chat_completion(
                messages=messages, options=options, format=format, **kwargs
            )

        except Exception as e:
            self._handle_error(e, "completion")
            raise Exception(f"OpenAI Completion Error: {str(e)}")
