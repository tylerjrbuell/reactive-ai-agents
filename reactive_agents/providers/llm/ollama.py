import json

from reactive_agents.core.types.tool_types import ToolCall

from .base import (
    BaseModelProvider,
    CompletionMessage,
    CompletionResponse,
)
import ollama
import os
import time
from typing import List, Literal, Optional, Dict, Any

DEFAULT_OPTIONS = {"temperature": 0.2, "num_gpu": 256, "num_ctx": 4000, "think": False}


class OllamaModelProvider(BaseModelProvider):
    id = "ollama"
    host = DEFAULT_OPTIONS.get("host") or os.getenv(
        "OLLAMA_HOST", "http://localhost:11434"
    )

    def __init__(
        self,
        model: str,
        options: Optional[Dict[str, Any]] = None,
        context=None,
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

    async def get_chat_completion(self, **kwargs) -> CompletionResponse:
        try:
            if not kwargs.get("model"):
                kwargs["model"] = self.model

            model_info = await self.client.show(kwargs["model"])
            capabilities = model_info.capabilities or []
            tool_support = "tools" in capabilities

            if kwargs.get("tools") and not tool_support:
                print("Manually generating tool calls...")
                # Extract the last user message as the task
                task = "Complete the user's request"
                if kwargs.get("messages"):
                    for msg in reversed(kwargs["messages"]):
                        if msg.get("role") == "user":
                            task = msg.get("content", task)
                            break

                tool_calls = await self.get_tool_calls(task=task, **kwargs)
                return CompletionResponse(
                    message=CompletionMessage(
                        content="",
                        role="assistant",
                        tool_calls=[tool_call.model_dump() for tool_call in tool_calls],
                    ),
                    model=self.model,
                    done=True,
                    done_reason=None,
                    prompt_tokens=0,
                    completion_tokens=0,
                    prompt_eval_duration=0,
                    load_duration=0,
                    total_duration=0,
                    created_at=str(time.time()),
                )
            result = await self.client.chat(
                model=kwargs["model"],
                messages=kwargs["messages"],
                stream=kwargs["stream"] if kwargs.get("stream") else False,
                tools=kwargs["tools"] if kwargs.get("tools") and tool_support else [],
                format=kwargs["format"] if kwargs.get("format") else None,
                options=(
                    kwargs["options"] if kwargs.get("options") else DEFAULT_OPTIONS
                ),
                think=kwargs.get("options", {}).get("think", False),
            )

            message = CompletionMessage(
                content=result.message.content or "",
                thinking=result.message.thinking,
                role=result.message.role or "assistant",
                tool_calls=(
                    list(result.message.tool_calls)
                    if result.message.tool_calls
                    else None
                ),
            )

            return CompletionResponse(
                message=self.extract_and_store_thinking(
                    message, call_context="chat_completion"
                ),
                model=result.model or self.model,
                done=result.done or False,
                done_reason=result.done_reason,
                prompt_tokens=result.prompt_eval_count,
                completion_tokens=result.eval_count,
                prompt_eval_duration=result.eval_duration,
                load_duration=result.load_duration,
                total_duration=result.total_duration,
                created_at=result.created_at,
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
            completion = ollama.generate(**kwargs)
            message = CompletionMessage(
                content=completion.response,
                thinking=completion.thinking,
            )
            return CompletionResponse(
                message=self.extract_and_store_thinking(
                    message, call_context="completion"
                ),
                model=completion.model or self.model,
                done=completion.done or False,
                done_reason=completion.done_reason,
                prompt_tokens=completion.prompt_eval_count,
                completion_tokens=completion.eval_count,
                prompt_eval_duration=completion.eval_duration,
                load_duration=completion.load_duration,
                total_duration=completion.total_duration,
                created_at=completion.created_at,
            )
        except Exception as e:
            self._handle_error(e, "completion")
            # This line will never be reached due to _handle_error raising the exception
            # But we need it for type checking
            raise

    async def get_tool_calls(
        self, task: str, max_calls: int = 1, **kwargs
    ) -> List[ToolCall]:
        try:
            context = kwargs.get("context", "No additional context provided")
            prompt = f"""
            Generate a max of {max_calls} tool call(s) to aid in the following task:
            
            Task:{task}
            Context: {context}
            """

            print(f"Generating tool calls for task: {task}")

            # Use centralized prompt system if context is available
            if hasattr(self, "context") and self.context:
                try:
                    from reactive_agents.core.reasoning.prompts.base import (
                        OllamaManualToolPrompt,
                    )

                    system_prompt = OllamaManualToolPrompt(self.context).generate(
                        task=task,
                        tool_signatures=kwargs.get("tools", []),
                        max_calls=max_calls,
                    )
                except Exception as e:
                    print(f"Failed to use centralized prompt system: {e}")
                    # Fallback to basic prompt
                    system_prompt = f"""Role: Tool Selection and configuration Expert
                    Objective: Create a max of {max_calls} tool call(s) to aid in the following task

                    Guidelines:
                    - The tool call must adhere to the specific task
                    - Use the tool signatures to effectively create tool calls that align with the task
                    - Use the context provided in conjunction with the tool signatures to create tool calls that align with the task
                    - Only use valid parameters and valid parameter data types and avoid using tool signatures that are not available
                    - Check all data types are correct based on the tool signatures provided in available tools to avoid issues when the tool is used
                    - Pay close attention to the signatures and parameters provided
                    - Do not try to consolidate multiple tool calls into one call
                    - Do not try to use tools that are not available

                    Available Tool signatures: {kwargs["tools"]}

                    Output Format: JSON with the following structure:
                    {{
                        "tool_calls": [
                            {{
                                "function": {{
                                    "name": "<tool_name>",
                                    "arguments": <tool_parameters>
                                }}
                            }}
                        ]
                    }}"""

            result = await self.client.generate(
                model=kwargs["model"],
                system=system_prompt,
                prompt=prompt,
                format="json",
                options=kwargs["options"],
            )

            # Handle JSON parsing errors
            try:
                response = json.loads(result.response)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {result.response}")
                print(f"JSON error: {e}")
                # Try to extract JSON from the response if it's wrapped in markdown
                import re

                json_match = re.search(
                    r"```json\s*(.*?)\s*```", result.response, re.DOTALL
                )
                if json_match:
                    try:
                        response = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        print("Failed to extract JSON from markdown")
                        return []
                else:
                    return []

            valid_tool_calls = []
            for tool_call in response.get("tool_calls", []):
                try:
                    valid_tool_calls.append(ToolCall.model_validate(tool_call))
                except Exception as e:
                    print(f"Skipping invalid tool call: {tool_call}. Error: {e}")

            print(f"Generated {len(valid_tool_calls)} valid tool calls")

            # If no valid tool calls were generated, try a simpler approach
            if not valid_tool_calls:
                print("No valid tool calls generated, trying fallback approach...")
                # Try to extract tool calls from the raw response
                import re

                # Look for patterns like "use tool_name" or "call tool_name"
                tool_pattern = r"(?:use|call)\s+(\w+)(?:\s+with\s+(.+))?"
                matches = re.findall(tool_pattern, result.response, re.IGNORECASE)

                for tool_name, args_str in matches:
                    if tool_name in [
                        tool.get("function", {}).get("name") for tool in kwargs["tools"]
                    ]:
                        try:
                            # Try to parse arguments if provided
                            arguments = {}
                            if args_str:
                                # Simple argument parsing - this is a basic fallback
                                args_parts = args_str.split(",")
                                for part in args_parts:
                                    if "=" in part:
                                        key, value = part.split("=", 1)
                                        arguments[key.strip()] = value.strip().strip(
                                            "\"'"
                                        )

                            from reactive_agents.core.types.tool_types import (
                                ToolCallFunction,
                            )

                            tool_call = ToolCall(
                                function=ToolCallFunction(
                                    name=tool_name, arguments=arguments
                                )
                            )
                            valid_tool_calls.append(tool_call)
                            print(f"Fallback: Created tool call for {tool_name}")
                        except Exception as e:
                            print(
                                f"Failed to create fallback tool call for {tool_name}: {e}"
                            )

            return valid_tool_calls
        except Exception as e:
            self._handle_error(e, "tool_call")
            # This line will never be reached due to _handle_error raising the exception
            # But we need it for type checking
            raise
