import json
import os
import time
from typing import List, Dict, Any, Optional, Union
import requests
from anthropic import Anthropic
from anthropic.types import Message, MessageParam, TextBlock, ToolUseBlock
from anthropic import APIError, RateLimitError, APITimeoutError

from .base import BaseModelProvider, CompletionMessage, CompletionResponse


class AnthropicModelProvider(BaseModelProvider):
    """Anthropic model provider using the official Anthropic Python SDK."""

    id = "anthropic"

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        options: Optional[Dict[str, Any]] = None,
        context=None,
    ):
        """
        Initialize the Anthropic model provider.

        Args:
            model: The model to use (e.g., "claude-3-sonnet-20240229", "claude-3-opus-20240229")
            options: Optional configuration options
            context: The agent context for error tracking and logging
        """
        super().__init__(model=model, options=options, context=context)

        # Initialize Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        # Initialize Anthropic client with supported parameters
        client_params = {"api_key": api_key}

        # Add supported client parameters from options if they exist
        if options:
            supported_client_params = {
                "base_url",
                "timeout",
                "max_retries",
                "default_headers",
                "default_query",
            }
            for param in supported_client_params:
                if param in options:
                    client_params[param] = options[param]

        self.client = Anthropic(**client_params)  # type: ignore

        # Default options
        self.default_options = {
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        # Check if model is Claude 3 (supported by SDK) or legacy (requires manual HTTP)
        self.is_claude_3 = self._is_claude_3_model(self.model)

        # Validate model on initialization
        self.validate_model()

    def _is_claude_3_model(self, model: str) -> bool:
        """Check if the model is Claude 3+ (supported by modern messages API)."""
        # Legacy models that require the old completion API
        legacy_model_prefixes = ["claude-1", "claude-2"]
        
        # Check if it's a legacy model
        for prefix in legacy_model_prefixes:
            if model.startswith(prefix):
                return False
        
        # All Claude 3+ models (including aliases) use the modern messages API
        # This includes:
        # - claude-3-5-sonnet-latest (alias for claude-3-5-sonnet-20241022)
        # - claude-3-7-sonnet-latest (alias for claude-3-7-sonnet-20250219)  
        # - claude-sonnet-4-0 (alias for claude-sonnet-4-20250514)
        # - claude-opus-4-0 (alias for claude-opus-4-20250514)
        modern_model_prefixes = ["claude-3", "claude-4", "claude-sonnet-4", "claude-opus-4"]
        
        for prefix in modern_model_prefixes:
            if model.startswith(prefix):
                return True
                
        # Default to modern API for unknown models (safer assumption)
        return True

    def _clean_message(self, msg: dict) -> dict:
        """Clean message to only include fields supported by Anthropic API."""
        # Handle tool results - Anthropic doesn't support "tool" role
        if msg.get("role") == "tool":
            # Convert tool results to user message with tool result content
            tool_result = msg.get("content", "")
            return {
                "role": "user",
                "content": f"Tool result: {tool_result}".rstrip()
            }
        
        allowed = {"role", "content"}
        cleaned = {k: v for k, v in msg.items() if k in allowed}

        # Ensure required fields are present and roles are valid
        if "role" not in cleaned or cleaned["role"] not in ["user", "assistant"]:
            cleaned["role"] = "user"
        if "content" not in cleaned:
            cleaned["content"] = ""
        
        # Strip trailing whitespace from content to avoid Anthropic API errors
        if isinstance(cleaned["content"], str):
            cleaned["content"] = cleaned["content"].rstrip()

        return cleaned

    def _convert_tools_to_anthropic_format(self, tools: List[dict]) -> List[dict]:
        """Convert tools from OpenAI format to Anthropic format."""
        converted_tools = []
        
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                # Convert from OpenAI format to Anthropic format
                func_def = tool["function"]
                converted_tool = {
                    "type": "custom",
                    "name": func_def["name"],
                    "description": func_def.get("description", ""),
                    "input_schema": func_def.get("parameters", {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
                converted_tools.append(converted_tool)
            else:
                # Tool is already in the correct format or unknown format
                converted_tools.append(tool)
        
        return converted_tools

    def _extract_system_message(
        self, messages: List[dict]
    ) -> tuple[Optional[str], List[dict]]:
        """Extract system message from messages list."""
        system_message = None
        filtered_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        return system_message, filtered_messages

    def validate_model(self, **kwargs) -> dict:
        """Validate that the model is supported by Anthropic."""
        try:
            if not self.is_claude_3:
                # For legacy models, we'll accept them but note they need manual HTTP
                return {
                    "valid": True,
                    "model": self.model,
                    "warning": "Legacy model - requires manual HTTP implementation",
                }

            # For Claude 3, we can't easily list models without making a request
            # So we'll just validate the model name format
            if "claude-3" in self.model.lower():
                return {"valid": True, "model": self.model}
            else:
                raise ValueError(
                    f"Model '{self.model}' is not a supported Claude 3 model"
                )

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
        Get a chat completion from Anthropic.

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
            if not self.is_claude_3:
                return await self._get_legacy_completion(
                    messages=messages,
                    stream=stream,
                    tools=tools,
                    options=options,
                    format=format,
                    **kwargs,
                )

            # Extract system message and clean messages
            system_message, cleaned_messages = self._extract_system_message(messages)
            cleaned_messages = [self._clean_message(msg) for msg in cleaned_messages]

            # Merge options
            merged_options = {**self.default_options, **(options or {})}

            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": cleaned_messages,
                "stream": stream,
                **merged_options,
            }

            # Add system message if present
            if system_message:
                api_params["system"] = system_message

            # Add tools if present (convert to Anthropic format)
            if tools:
                api_params["tools"] = self._convert_tools_to_anthropic_format(tools)
                if tool_choice and tool_choice != "auto":
                    api_params["tool_choice"] = tool_choice

            # Handle JSON format for Claude 3 models
            if format == "json":
                # Add JSON instruction to the last user message
                if cleaned_messages and cleaned_messages[-1].get("role") == "user":
                    cleaned_messages[-1][
                        "content"
                    ] += "\n\nPlease respond in valid JSON format."
                elif system_message:
                    system_message += "\n\nAlways respond in valid JSON format."
                else:
                    # Add system message if not present
                    api_params["system"] = "Always respond in valid JSON format."

            # Create completion
            completion = self.client.messages.create(**api_params)

            if stream:
                return completion  # Return stream object directly

            # Process non-streaming response
            content = ""
            tool_calls = None

            # Extract content and tool calls
            if completion.content:
                for block in completion.content:
                    if isinstance(block, TextBlock):
                        content += block.text
                    elif isinstance(block, ToolUseBlock):
                        if tool_calls is None:
                            tool_calls = []
                        tool_calls.append(
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": block.input,
                                },
                            }
                        )

            message = CompletionMessage(
                content=content,
                role="assistant",
                tool_calls=tool_calls,
            )

            return CompletionResponse(
                message=self.extract_and_store_thinking(
                    message, call_context="chat_completion"
                ),
                model=completion.model,
                done=True,
                done_reason=completion.stop_reason,
                prompt_tokens=completion.usage.input_tokens,
                completion_tokens=completion.usage.output_tokens,
                total_duration=None,  # Anthropic doesn't provide timing info
                created_at=str(time.time()),
            )

        except RateLimitError as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"Anthropic Rate Limit Error: {str(e)}")
        except APITimeoutError as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"Anthropic API Timeout Error: {str(e)}")
        except APIError as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"Anthropic API Error: {str(e)}")
        except Exception as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"Anthropic Chat Completion Error: {str(e)}")

    async def _get_legacy_completion(
        self,
        messages: List[dict],
        stream: bool = False,
        tools: Optional[List[dict]] = None,
        options: Optional[Dict[str, Any]] = None,
        format: str = "",
        **kwargs,
    ) -> CompletionResponse:
        """
        Get completion for legacy Claude models using manual HTTP requests.

        This is a fallback for Claude 1/2 models that aren't supported by the SDK.
        """
        try:
            # Convert messages to legacy prompt format
            prompt = self._convert_messages_to_legacy_prompt(messages)

            # Prepare request data
            api_key = os.getenv("ANTHROPIC_API_KEY")
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "anthropic-version": "2023-06-01",
            }

            merged_options = {**self.default_options, **(options or {})}

            data = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens_to_sample": merged_options.get("max_tokens", 1000),
                "temperature": merged_options.get("temperature", 0.7),
                "stream": stream,
            }

            # Handle JSON format for legacy models
            if format == "json":
                data["prompt"] += "\n\nPlease respond in valid JSON format."

            # Make request to legacy API
            response = requests.post(
                "https://api.anthropic.com/v1/complete",
                headers=headers,
                json=data,
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()

            message = CompletionMessage(
                content=result.get("completion", ""),
                role="assistant",
                tool_calls=None,  # Legacy API doesn't support tool calls
            )

            return CompletionResponse(
                message=self.extract_and_store_thinking(
                    message, call_context="chat_completion"
                ),
                model=self.model,
                done=True,
                done_reason=result.get("stop_reason"),
                prompt_tokens=0,  # Legacy API doesn't provide token counts
                completion_tokens=0,
                total_duration=None,
                created_at=str(time.time()),
            )

        except Exception as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"Anthropic Legacy Completion Error: {str(e)}")

    def _convert_messages_to_legacy_prompt(self, messages: List[dict]) -> str:
        """Convert messages to legacy Claude prompt format."""
        prompt = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt += f"{content}\n\n"
            elif role == "user":
                prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        # Ensure it ends with Assistant:
        if not prompt.strip().endswith("Assistant:"):
            prompt += "Assistant:"

        return prompt

    async def get_completion(
        self,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        format: str = "",
        **kwargs,
    ) -> CompletionResponse:
        """
        Get a text completion from Anthropic using the chat completions API.

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
            raise Exception(f"Anthropic Completion Error: {str(e)}")
