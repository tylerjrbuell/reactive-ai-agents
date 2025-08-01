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

    def _process_tool_calls(self, tool_calls):
        """Process tool calls to ensure arguments are properly formatted as dictionaries."""
        if not tool_calls:
            return None
            
        processed_calls = []
        for call in tool_calls:
            call_dict = call.model_dump() if hasattr(call, 'model_dump') else dict(call)
            
            # Ensure function arguments are dictionaries, not JSON strings
            if 'function' in call_dict and 'arguments' in call_dict['function']:
                args = call_dict['function']['arguments']
                if isinstance(args, str):
                    try:
                        import json
                        call_dict['function']['arguments'] = json.loads(args) if args else {}
                    except (json.JSONDecodeError, TypeError):
                        call_dict['function']['arguments'] = {}
                elif not isinstance(args, dict):
                    call_dict['function']['arguments'] = {}
                    
            processed_calls.append(call_dict)
            
        return processed_calls

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

    def _adapt_context_for_provider(self, messages: list, tools: list | None = None) -> list:
        """
        Adapt context for Groq's specific requirements and best practices.
        
        Since tool context is now centralized in base strategy system messages,
        this method focuses on Groq-specific optimizations only.
        
        Groq Adaptations:
        - Preserve all agent context (role, instructions, task, tools)
        - Follow Groq's recommendation for clear completion signals
        """
        if not tools:
            return messages
            
        adapted_messages = messages.copy()
        
        # Add minimal completion guidance to existing system message (preserves agent context)
        # This helps Groq models understand when to finish tasks properly
        completion_hint = "\n\nIMPORTANT: After using the required tools, you MUST call 'final_answer' to complete your response."
        
        system_message_found = False
        for i in range(len(adapted_messages)):
            if adapted_messages[i].get("role") == "system":
                original_content = adapted_messages[i].get("content", "")
                adapted_messages[i] = {
                    **adapted_messages[i], 
                    "content": original_content + completion_hint
                }
                system_message_found = True
                break
        
        # Only add minimal system message if none exists (rare in framework)
        if not system_message_found:
            minimal_system_msg = {
                "role": "system",
                "content": f"Use tools to complete tasks, then MUST call 'final_answer' to finish.{completion_hint}"
            }
            adapted_messages.insert(0, minimal_system_msg)
                
        return adapted_messages

    async def get_chat_completion(
        self,
        messages: list,
        stream: bool = False,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        tool_use: bool = True,
        tool_use_required: bool = False,
        options: dict | None = None,
        format: str = "",
        **kwargs
    ) -> CompletionResponse | Stream[ChatCompletionChunk] | None:
        try:
            messages = [self._clean_message(m) for m in messages]
            
            # Adapt context for Groq's requirements (preserves agent context)
            if tools and tool_use:
                messages = self._adapt_context_for_provider(messages, tools)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools and tool_use else None,
                response_format=(
                    {"type": "json_object"} if format == "json" else {"type": "text"}
                ),
                tool_choice=(
                    # Groq models struggle with "required" - use "auto" instead
                    "auto" if tools and tool_use 
                    else "none"
                ),
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
                        self._process_tool_calls(result.message.tool_calls)
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
                    # For tool_use_failed, try to provide a fallback response without tools
                    tool_error = error_json["error"].get("failed_generation", "")
                    
                    # If failed_generation is empty, the model couldn't generate tool calls properly
                    if not tool_error.strip():
                        # Retry without tools to get a basic response
                        try:
                            fallback_completion = self.client.chat.completions.create(
                                model=self.model,
                                messages=messages,  # messages are already cleaned at this point
                                tools=None,  # Disable tools for fallback
                                tool_choice="none",
                                stream=stream,
                                **(options or {}),
                            )
                            
                            if type(fallback_completion) is ChatCompletion:
                                result = fallback_completion.choices[0]
                                message = CompletionMessage(
                                    content=result.message.content if result.message.content else 
                                        "[Model failed to generate tool calls, provided fallback response]",
                                    role=result.message.role if result.message.role else "assistant",
                                    thinking="False",
                                    tool_calls=None,
                                )
                                return CompletionResponse(
                                    message=self.extract_and_store_thinking(
                                        message, call_context="chat_completion"
                                    ),
                                    model=fallback_completion.model or self.model,
                                    done=True,
                                    done_reason=result.finish_reason or None,
                                    prompt_tokens=(
                                        int(fallback_completion.usage.prompt_tokens or 0)
                                        if fallback_completion.usage
                                        else 0
                                    ),
                                    completion_tokens=(
                                        int(fallback_completion.usage.completion_tokens)
                                        if fallback_completion.usage
                                        else 0
                                    ),
                                    prompt_eval_duration=(
                                        int(fallback_completion.usage.prompt_time or 0)
                                        if fallback_completion.usage
                                        else 0
                                    ),
                                    load_duration=(
                                        int(fallback_completion.usage.completion_time or 0)
                                        if fallback_completion.usage
                                        else 0
                                    ),
                                    total_duration=(
                                        int(fallback_completion.usage.total_time or 0) 
                                        if fallback_completion.usage else 0
                                    ),
                                    created_at=str(fallback_completion.created) 
                                        if fallback_completion.created else None,
                                )
                        except Exception as fallback_error:
                            # If fallback also fails, log both errors
                            self._handle_error(e, "chat_completion")
                            self._handle_error(fallback_error, "chat_completion_fallback")
                            raise Exception(f"Groq Tool Use Failed and Fallback Failed: {str(e)} | Fallback: {str(fallback_error)}")
                    
                    # If we have failed_generation content, report it
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
