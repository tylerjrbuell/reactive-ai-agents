import json
import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Union
import google.generativeai as genai  # type: ignore
from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
from google.api_core import exceptions as google_exceptions

from .base import BaseModelProvider, CompletionMessage, CompletionResponse


class GoogleModelProvider(BaseModelProvider):
    """Google model provider using the Google Generative AI Python SDK."""

    id = "google"

    def __init__(
        self,
        model: str = "gemini-pro",
        options: Optional[Dict[str, Any]] = None,
        context=None,
    ):
        """
        Initialize the Google model provider.

        Args:
            model: The model to use (e.g., "gemini-pro", "gemini-pro-vision", "gemini-1.5-pro")
            options: Optional configuration options
            context: The agent context for error tracking and logging
        """
        super().__init__(model=model, options=options, context=context)

        # Initialize Google Generative AI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        genai.configure(api_key=api_key)  # type: ignore

        # Default options
        self.default_options = {
            "temperature": 0.7,
            "max_output_tokens": 1000,
            "top_p": 0.95,
            "top_k": 40,
        }

        # Safety settings (optional, can be overridden)
        # Using BLOCK_ONLY_HIGH to be less restrictive by default
        self.default_safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        # Initialize the model
        self.generative_model = genai.GenerativeModel(model_name=self.model)  # type: ignore

        # Validate model on initialization
        self.validate_model()

    def configure_safety_settings(
        self, safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None
    ):
        """
        Configure safety settings for the Google model.

        Args:
            safety_settings: Dictionary mapping harm categories to block thresholds
                           If None, uses more permissive defaults
        """
        if safety_settings is None:
            # More permissive settings
            self.default_safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        else:
            self.default_safety_settings = safety_settings

    def _clean_message(self, msg: dict) -> dict:
        """Clean message to only include fields supported by Google API."""
        # Google uses 'parts' instead of 'content' and 'role' is 'user' or 'model'
        cleaned = {}

        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Map roles: assistant -> model, user -> user, system -> user (with context)
        if role == "assistant":
            cleaned["role"] = "model"
        else:
            cleaned["role"] = "user"

        # Google expects 'parts' which can be text or other content
        if isinstance(content, str):
            cleaned["parts"] = [content]
        else:
            cleaned["parts"] = [str(content)]

        return cleaned

    def _clean_json_response(self, content: str) -> str:
        """Clean JSON response by removing markdown code blocks and fixing common JSON issues."""
        if not content:
            return content

        # Remove BOM and extra whitespace
        content = content.strip().lstrip("\ufeff").strip()

        # Remove markdown code blocks (```json ... ``` or ``` ... ```)
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        elif content.startswith("```"):
            content = content[3:]  # Remove ```

        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```

        # Remove any leading/trailing whitespace and newlines
        content = content.strip()

        # Remove any leading/trailing quotes that might wrap the JSON
        if (
            content.startswith('"')
            and content.endswith('"')
            and content.count('"') == 2
        ):
            content = content[1:-1]

        # Fix common JSON formatting issues that Google models might generate
        content = self._fix_json_formatting(content)

        return content.strip()

    def _clean_schema_for_google(self, schema: dict) -> dict:
        """
        Clean JSON schema to remove fields that Google's FunctionDeclaration doesn't support.

        Based on Google's FunctionDeclaration documentation, it supports OpenAPI 3.0 schema
        but with some limitations. This removes potentially problematic fields.
        """
        if not isinstance(schema, dict):
            return schema

        # Fields that are known to cause issues with Google's SDK
        unsupported_fields = {
            "title",  # Sometimes causes "Unknown field for Schema: title" error
            "$schema",  # JSON Schema meta field not supported
            "$id",  # JSON Schema meta field not supported
            "examples",  # Use 'example' instead
            "definitions",  # Use 'defs' instead
            "additionalItems",  # Not supported
            "patternProperties",  # Not supported
            "dependencies",  # Not supported
            "const",  # May not be supported in all versions
            "anyOf",  # JSON Schema composition not supported
            "oneOf",  # JSON Schema composition not supported
            "allOf",  # JSON Schema composition not supported
            "not",  # JSON Schema negation not supported
            "default",  # Default values not supported in function parameters
        }

        # Create a cleaned copy of the schema
        cleaned = {}

        for key, value in schema.items():
            if key in unsupported_fields:
                # Skip unsupported fields, but handle some special cases
                if (
                    key == "examples"
                    and "example" not in schema
                    and isinstance(value, list)
                    and value
                ):
                    # Convert 'examples' array to single 'example' if no 'example' exists
                    cleaned["example"] = value[0]
                elif (
                    key == "definitions"
                    and "defs" not in schema
                    and isinstance(value, dict)
                ):
                    # Convert 'definitions' to 'defs' if no 'defs' exists
                    cleaned["defs"] = self._clean_schema_for_google(value)
                elif key == "const":
                    # Convert 'const' to enum with single value
                    cleaned["enum"] = [value]
                continue

            # Recursively clean nested objects
            if isinstance(value, dict):
                cleaned[key] = self._clean_schema_for_google(value)
            elif isinstance(value, list):
                # Clean each item if it's a dict
                cleaned_list = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_list.append(self._clean_schema_for_google(item))
                    else:
                        cleaned_list.append(item)
                cleaned[key] = cleaned_list
            else:
                cleaned[key] = value

        return cleaned

    def _fix_json_formatting(self, content: str) -> str:
        """Fix common JSON formatting issues from Google models."""
        if not content:
            return content

        # First, try to identify if this looks like JSON at all
        if not (content.strip().startswith("{") or content.strip().startswith("[")):
            return content

        import re

        try:
            # Fix unescaped quotes in string values
            # Look for patterns like "value with "quotes" inside" and escape them
            # This is a simple heuristic - we'll look for quote patterns that break JSON

            # Fix trailing commas that might break JSON parsing
            content = re.sub(r",\s*}", "}", content)
            content = re.sub(r",\s*]", "]", content)

            # Fix any control characters that might break JSON
            content = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", content)

            # Fix common unescaped characters in strings
            content = re.sub(
                r"\\n", "\\n", content
            )  # Ensure newlines are properly escaped
            content = re.sub(r"\\t", "\\t", content)  # Ensure tabs are properly escaped

            # Try to fix unterminated strings by looking for odd quote counts
            # This is a basic attempt - for complex cases, we might need more sophisticated parsing
            lines = content.split("\n")
            fixed_lines = []

            for line in lines:
                # Count quotes in the line
                quote_count = line.count('"')
                # If odd number of quotes, there might be an unterminated string
                if quote_count % 2 == 1:
                    # Try to find the last quote and see if it needs escaping
                    # This is a simple heuristic
                    if line.strip().endswith('"'):
                        # The line ends with a quote, might be OK
                        fixed_lines.append(line)
                    else:
                        # Try to add a closing quote at the end
                        # This is very basic and might not work for all cases
                        fixed_lines.append(line + '"')
                else:
                    fixed_lines.append(line)

            content = "\n".join(fixed_lines)

        except Exception:
            # If any of the regex or fixing fails, return original content
            # Better to have malformed JSON than to crash
            pass

        return content

    def _looks_like_json(self, content: str) -> bool:
        """Check if content appears to be JSON wrapped in markdown or formatted."""
        if not content:
            return False

        content = content.strip().lstrip("\ufeff").strip()
        return (
            content.startswith("```json")
            or (content.startswith("```") and content.endswith("```"))
            or (content.startswith("{") and content.endswith("}"))
            or (content.startswith("[") and content.endswith("]"))
            # Handle JSON wrapped in quotes
            or (content.startswith('"{') and content.endswith('}"'))
            or (content.startswith('"[') and content.endswith(']"'))
            # Handle JSON with possible extra text around it
            or ("{" in content and "}" in content)
            or ("[" in content and "]" in content)
        )

    async def _retry_with_backoff(self, func, *args, max_retries=3, **kwargs) -> Any:
        """Retry a function with exponential backoff on rate limit errors."""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except google_exceptions.ResourceExhausted as e:
                if attempt == max_retries - 1:
                    raise e

                # Extract retry delay from error if available
                retry_delay = 30  # Default 30 seconds
                error_str = str(e)
                if "retry_delay" in error_str and "seconds:" in error_str:
                    try:
                        # Try to parse retry delay from error message
                        delay_part = (
                            error_str.split("retry_delay")[1]
                            .split("seconds:")[1]
                            .split("}")[0]
                            .strip()
                        )
                        retry_delay = int(delay_part)
                    except (IndexError, ValueError):
                        pass  # Use default

                # Use exponential backoff with jitter
                backoff_delay = min(retry_delay * (2**attempt), 120)  # Max 2 minutes

                print(
                    f"Rate limit hit, retrying in {backoff_delay}s (attempt {attempt + 1}/{max_retries})"
                )

                await asyncio.sleep(backoff_delay)
            except Exception as e:
                # Non-rate-limit errors should not be retried
                raise e

    def _prepare_messages(self, messages: List[dict]) -> List[dict]:
        """Prepare messages for Google's chat format, handling system messages."""
        prepared_messages = []
        system_message = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_message = content
            else:
                cleaned = self._clean_message(msg)
                prepared_messages.append(cleaned)

        # If there's a system message, prepend it to the first user message
        if system_message and prepared_messages:
            first_msg = prepared_messages[0]
            if first_msg["role"] == "user":
                first_msg["parts"][0] = f"{system_message}\n\n{first_msg['parts'][0]}"

        return prepared_messages

    def validate_model(self, **kwargs) -> dict:
        """Validate that the model is supported by Google."""
        try:
            # List available models
            available_models = []
            for model in genai.list_models():  # type: ignore
                if "generateContent" in model.supported_generation_methods:
                    available_models.append(model.name.split("/")[-1])

            if self.model not in available_models:
                raise ValueError(
                    f"Model '{self.model}' is not available. "
                    f"Available models: {', '.join(available_models)}..."
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
        Get a chat completion from Google.

        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response (not fully supported yet)
            tools: List of tool definitions (function calling)
            tool_choice: Tool choice preference
            options: Model-specific options
            format: Response format ("json" or "")
            **kwargs: Additional arguments
        """
        try:
            # Prepare messages
            prepared_messages = self._prepare_messages(messages)

            # Merge options
            merged_options = {**self.default_options, **(options or {})}

            # Create generation config
            generation_config = genai.types.GenerationConfig(  # type: ignore
                temperature=merged_options.get("temperature", 0.7),
                max_output_tokens=merged_options.get("max_output_tokens", 1000),
                top_p=merged_options.get("top_p", 0.95),
                top_k=merged_options.get("top_k", 40),
            )

            # Handle JSON format
            if format == "json":
                # For Google models, we need to add JSON instruction to the prompt
                # as Google doesn't have a native JSON mode like OpenAI
                if prepared_messages:
                    last_msg = prepared_messages[-1]
                    last_msg["parts"][
                        -1
                    ] += "\n\nPlease respond in valid JSON format only. Do not include any text before or after the JSON object."

            # Handle tools/function calling
            available_functions = None
            if tools:
                # Convert tools to Google's function format
                available_functions = []
                for tool in tools:
                    if tool.get("type") == "function":
                        func_def = tool.get("function", {})

                        # Clean the parameters schema to remove unsupported fields
                        parameters = func_def.get("parameters", {})
                        cleaned_parameters = (
                            self._clean_schema_for_google(parameters)
                            if isinstance(parameters, dict)
                            else {}
                        )

                        google_func = genai.types.FunctionDeclaration(  # type: ignore
                            name=func_def.get("name", ""),
                            description=func_def.get("description", ""),
                            parameters=cleaned_parameters,
                        )
                        available_functions.append(google_func)

                if available_functions:
                    available_functions = genai.types.Tool(  # type: ignore
                        function_declarations=available_functions
                    )

            # Create chat session or single generation with retry
            if len(prepared_messages) > 1:
                # Use chat for multi-turn conversation
                chat = self.generative_model.start_chat(
                    history=prepared_messages[:-1] if len(prepared_messages) > 1 else []  # type: ignore
                )

                response = await self._retry_with_backoff(
                    chat.send_message,
                    prepared_messages[-1]["parts"][0],
                    generation_config=generation_config,
                    safety_settings=self.default_safety_settings,
                    tools=[available_functions] if available_functions else None,
                    stream=stream,
                )
            else:
                # Single message generation
                content = prepared_messages[0]["parts"][0] if prepared_messages else ""
                response = await self._retry_with_backoff(
                    self.generative_model.generate_content,
                    content,
                    generation_config=generation_config,
                    safety_settings=self.default_safety_settings,
                    tools=[available_functions] if available_functions else None,
                    stream=stream,
                )

            if stream:
                return response  # Return stream object directly

            # Process non-streaming response
            content = ""
            tool_calls = None
            done_reason = "stop"

            # Check if response has candidates and extract content safely
            if (
                response
                and hasattr(response, "candidates")
                and response.candidates
                and len(response.candidates) > 0
            ):
                candidate = response.candidates[0]

                # Check finish reason
                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    if candidate.finish_reason == 1:  # STOP
                        done_reason = "stop"
                    elif candidate.finish_reason == 2:  # MAX_TOKENS
                        done_reason = "length"
                    elif candidate.finish_reason == 3:  # SAFETY
                        done_reason = "content_filter"
                        content = "[Response blocked by safety filters]"
                        return CompletionResponse(
                            message=CompletionMessage(
                                content=content, role="assistant"
                            ),
                            model=self.model,
                            done=True,
                            done_reason=done_reason,
                            created_at=str(time.time()),
                        )
                    elif candidate.finish_reason == 4:  # RECITATION
                        done_reason = "content_filter"
                        content = "[Response blocked due to recitation]"
                        return CompletionResponse(
                            message=CompletionMessage(
                                content=content, role="assistant"
                            ),
                            model=self.model,
                            done=True,
                            done_reason=done_reason,
                            created_at=str(time.time()),
                        )
                    elif candidate.finish_reason == 5:  # OTHER
                        done_reason = "stop"

                # Extract content from parts if available and not blocked
                if (
                    hasattr(candidate, "content")
                    and candidate.content
                    and hasattr(candidate.content, "parts")
                ):
                    for part in candidate.content.parts:
                        # Extract text content
                        if hasattr(part, "text") and part.text:
                            content += part.text

                        # Extract function calls
                        if hasattr(part, "function_call") and part.function_call:
                            if tool_calls is None:
                                tool_calls = []

                            # Convert Google function call to our format
                            func_call = part.function_call
                            tool_calls.append(
                                {
                                    "id": f"call_{int(time.time())}_{len(tool_calls)}",
                                    "type": "function",
                                    "function": {
                                        "name": func_call.name,
                                        "arguments": json.dumps(dict(func_call.args)),
                                    },
                                }
                            )
            else:
                # No candidates in response - this will result in "[No response generated]"
                pass

            # Always clean JSON response if format was requested or if it looks like JSON
            if content and (format == "json" or self._looks_like_json(content)):
                content = self._clean_json_response(content)

            # If no content was extracted, set error response
            if not content:
                content = "[No response generated]"
                done_reason = "error"

            message = CompletionMessage(
                content=content,
                role="assistant",
                tool_calls=tool_calls,
            )

            return CompletionResponse(
                message=self.extract_and_store_thinking(
                    message, call_context="chat_completion"
                ),
                model=self.model,
                done=True,
                done_reason=done_reason,
                prompt_tokens=(
                    response.usage_metadata.prompt_token_count
                    if response
                    and hasattr(response, "usage_metadata")
                    and response.usage_metadata
                    else None
                ),
                completion_tokens=(
                    response.usage_metadata.candidates_token_count
                    if response
                    and hasattr(response, "usage_metadata")
                    and response.usage_metadata
                    else None
                ),
                total_duration=None,  # Google doesn't provide timing info
                created_at=str(time.time()),
            )

        except google_exceptions.ResourceExhausted as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"Google API Quota Exceeded: {str(e)}")
        except google_exceptions.InvalidArgument as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"Google API Invalid Argument: {str(e)}")
        except google_exceptions.PermissionDenied as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"Google API Permission Denied: {str(e)}")
        except Exception as e:
            self._handle_error(e, "chat_completion")
            raise Exception(f"Google Chat Completion Error: {str(e)}")

    async def get_completion(
        self,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        format: str = "",
        **kwargs,
    ) -> CompletionResponse:
        """
        Get a text completion from Google using the generative model.

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
            raise Exception(f"Google Completion Error: {str(e)}")
