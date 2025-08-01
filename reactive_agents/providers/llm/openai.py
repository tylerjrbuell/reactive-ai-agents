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

        # Handle tool messages - OpenAI has strict requirements for tool message ordering
        # Convert tool messages to user messages with clear labeling
        if cleaned.get("role") == "tool":
            cleaned["role"] = "user"
            # Preserve the tool result information in the content
            original_content = cleaned.get("content", "")
            tool_call_id = cleaned.get("tool_call_id", "unknown")
            cleaned["content"] = f"[Tool Result for {tool_call_id}]: {original_content}"
            
            # Remove tool-specific fields since we're converting to user message
            cleaned.pop("tool_call_id", None)

        # Ensure required fields are present
        if "role" not in cleaned:
            cleaned["role"] = "user"
        if "content" not in cleaned:
            cleaned["content"] = ""

        return cleaned

    def _validate_message_sequence(self, messages: List[dict]) -> List[dict]:
        """Validate and fix message sequence for OpenAI's tool calling requirements."""
        if not messages:
            return messages
            
        validated_messages = []
        i = 0
        
        while i < len(messages):
            msg = messages[i]
            
            # If this is a tool message, ensure it follows an assistant message with tool_calls
            if msg.get("role") == "tool":
                # Look back to find the most recent assistant message with tool_calls
                assistant_with_tools_idx = None
                for j in range(len(validated_messages) - 1, -1, -1):
                    if (validated_messages[j].get("role") == "assistant" and 
                        validated_messages[j].get("tool_calls")):
                        assistant_with_tools_idx = j
                        break
                
                # If we found an assistant message with tool_calls, but it's not the immediate predecessor
                if assistant_with_tools_idx is not None and assistant_with_tools_idx != len(validated_messages) - 1:
                    # Remove any intermediate messages that aren't tool messages
                    # Keep only tool messages that might be part of the same tool call sequence
                    filtered_messages = validated_messages[:assistant_with_tools_idx + 1]
                    for k in range(assistant_with_tools_idx + 1, len(validated_messages)):
                        if validated_messages[k].get("role") == "tool":
                            filtered_messages.append(validated_messages[k])
                    validated_messages = filtered_messages
                
                validated_messages.append(msg)
            else:
                validated_messages.append(msg)
            
            i += 1
        
        return validated_messages

    def _process_tool_calls(self, tool_calls):
        """Process tool calls to ensure arguments are properly formatted as dictionaries."""
        if not tool_calls:
            return None
            
        processed_calls = []
        for call in tool_calls:
            processed_call = {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            
            # Ensure function arguments are dictionaries, not JSON strings
            args = processed_call["function"]["arguments"]
            if isinstance(args, str):
                try:
                    processed_call["function"]["arguments"] = json.loads(args) if args else {}
                except (json.JSONDecodeError, TypeError):
                    processed_call["function"]["arguments"] = {}
            elif not isinstance(args, dict):
                processed_call["function"]["arguments"] = {}
                    
            processed_calls.append(processed_call)
            
        return processed_calls

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
            
            # Validate message sequence for OpenAI's tool calling requirements
            cleaned_messages = self._validate_message_sequence(cleaned_messages)

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
            tool_calls = self._process_tool_calls(result.message.tool_calls)

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
