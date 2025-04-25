from __future__ import annotations
import json
import traceback
import asyncio
from typing import List, Dict, Any, Optional, Union, Sequence, Callable
from loggers.base import Logger
from model_providers.base import BaseModelProvider
from prompts.agent_prompts import (
    TASK_PLANNING_SYSTEM_PROMPT,
    TOOL_ACTION_SUMMARY_PROMPT,
    AGENT_ACTION_PLAN_PROMPT,
)
from model_providers.factory import ModelProviderFactory
from pydantic import BaseModel

from agent_mcp.client import MCPClient
from tools.abstractions import ToolProtocol, MCPToolWrapper, ToolResult
import time


class Agent:
    def __init__(
        self,
        name: str,
        provider_model: str,
        mcp_client: Optional[MCPClient] = None,
        instructions: str = "",
        tools: Sequence[ToolProtocol] = (),  # Use empty tuple as default
        tool_use: bool = True,
        min_completion_score: float = 1.0,
        max_iterations: Optional[int] = None,
        log_level: str = "info",
        workflow_context: Optional[Dict[str, Any]] = None,
        confirmation_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        enable_caching: bool = True,
        cache_ttl: int = 3600,  # 1 hour default TTL
    ):
        try:
            ## Agent Attributes
            self.name: str = name
            self.model_provider: BaseModelProvider = (
                ModelProviderFactory.get_model_provider(provider_model)
            )
            self.initial_task: str = ""
            self.final_answer: Optional[str] = ""
            self.instructions: str = instructions
            self.mcp_client = mcp_client
            self.tool_use: bool = tool_use
            self.confirmation_callback = confirmation_callback
            self.reasoning_log: List[str] = []  # For thought surfacing

            # Handle tools based on whether we have an MCP client or manual tools
            self.tools = ()  # Use empty tuple for initialization
            self.memory: list = []
            self.task_progress: str = ""
            self.messages: list = [{"role": "system", "content": self.instructions}]
            self.plan_prompt: str = ""
            self.min_completion_score = min_completion_score
            self.max_iterations = max_iterations
            self.iterations: int = 0

            # Initialize loggers
            self.agent_logger = Logger(
                name=name,
                type="agent",
                level=log_level,
            )
            self.tool_logger = Logger(
                name=f"{self.name} Tool",
                type="tool",
                level=log_level,
            )
            self.result_logger = Logger(
                name=f"{self.name} Result",
                type="agent_response",
                level=log_level,
            )

            self.tool_signatures: List[Dict[str, Any]] = []

            if self.mcp_client:
                # Import at runtime to avoid circular dependency
                from tools.abstractions import MCPToolWrapper

                self.tools = [
                    MCPToolWrapper(t, self.mcp_client) for t in self.mcp_client.tools
                ]
                self.tool_signatures = self.mcp_client.tool_signatures
            elif tools:
                self.tools = tools
                self.tool_signatures = [
                    tool.tool_definition
                    for tool in tools
                    if hasattr(tool, "tool_definition")
                ]

            self.tool_history: List[Dict[str, Any]] = []

            # Only initialize workflow context if provided
            if workflow_context is not None:
                self.workflow_context = workflow_context
                # Initialize agent's own context if workflow tracking is enabled
                if self.name:
                    if self.name not in self.workflow_context:
                        self.workflow_context[self.name] = {
                            "status": "initialized",
                            "current_progress": "",
                            "iterations": 0,
                            "dependencies_met": True,
                        }

            self.agent_logger.info(f"{self.name} Initialized")
            self.agent_logger.info(
                f"Provider:Model: {self.model_provider.name}:{self.model_provider.model}"
            )
            if self.mcp_client:
                self.agent_logger.info(
                    f"Connected MCP Servers: {', '.join(list(self.mcp_client.server_tools.keys()))}"
                )
            self.agent_logger.info(
                f"Available Tools: {', '.join([tool.name for tool in self.tools])}"
            )

            # Tool caching setup
            self.enable_caching = enable_caching
            self.cache_ttl = cache_ttl
            self.tool_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
        except Exception as e:
            # Get the full stack trace
            stack_trace = "".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
            self.agent_logger.error(
                f"Initialization Error: {str(e)}\nStack trace:\n{stack_trace}"
            )
            raise  # Re-raise the exception after logging

    async def _think(self, **kwargs) -> dict | None:
        try:
            self.agent_logger.info("Thinking...")
            return await self.model_provider.get_completion(**kwargs)
        except Exception as e:
            self.agent_logger.error(f"Completion Error: {e}")
            return

    async def _plan(self, **kwargs) -> str | None:
        class format(BaseModel):
            next_step: str

        try:
            self.agent_logger.info("Planning...")
            self.plan_prompt = """
            <context>
                <main-task> {task} </main-task>
                <available-tools> {tools} </available-tools>
                <previous-steps-performed> {steps} </previous-steps-performed>
            </context>
            
            """.format(
                task=self.initial_task,
                tools=[
                    f"Name: {tool['function']['name']}\nParameters: {tool['function']['parameters']}"
                    for tool in self.tool_signatures
                ],
                steps="\n".join(
                    [
                        f"Step {index}: {step}"
                        for index, step in enumerate(self.memory, start=1)
                    ]
                ),
            )
            result = await self.model_provider.get_completion(
                system=AGENT_ACTION_PLAN_PROMPT,
                prompt=self.plan_prompt,
                format=(
                    format.model_json_schema()
                    if self.model_provider.name == "ollama"
                    else "json"
                ),
            )
            return result.get("response", None)
        except Exception as e:
            self.agent_logger.error(f"Completion Error: {e}")
            return

    async def _think_chain(
        self,
        tool_use: bool = True,
        remember_messages: bool = True,
        **kwargs,
    ) -> dict | None:
        try:
            if self.workflow_context:
                # Include workflow context in messages if available
                context_message = {
                    "role": "system",
                    "content": f"Previous workflow steps:\n{json.dumps(self.workflow_context, indent=2)}",
                }
                if context_message not in self.messages:
                    self.messages.insert(1, context_message)

            kwargs.setdefault("messages", self.messages)
            self.agent_logger.info(f"Thinking...")

            result = await self.model_provider.get_chat_completion(
                tools=self.tool_signatures if tool_use else [],
                **kwargs,
            )
            if not result:
                return None
            message_content = result["message"].get("content")
            tool_calls = result["message"].get("tool_calls")
            if message_content and remember_messages:

                self.messages.append({"role": "assistant", "content": message_content})
            elif tool_calls:
                await self._process_tool_calls(tool_calls=tool_calls)
                return await self._think_chain(tool_use=False, **kwargs)

            return result
        except Exception as e:
            self.agent_logger.error(f"Chat Completion Error: {e}")
            return

    async def _process_tool_calls(self, tool_calls) -> None:
        processed_tool_calls = []
        for tool_call in tool_calls:
            if str(tool_call) in processed_tool_calls:
                self.tool_logger.info(
                    f"Tool Call Already Processed: {tool_call['function']['name']}"
                )
                continue

            tool_result = await self._use_tool(tool_call=tool_call)
            if tool_result is not None:
                self.messages.append(
                    {
                        **(
                            {"tool_call_id": tool_call["id"]}
                            if tool_call.get("id")
                            else {}
                        ),
                        "role": "tool",
                        "name": tool_call["function"]["name"],
                        "content": f"{tool_result}",
                    }
                )
                processed_tool_calls.append(str(tool_call))

    async def _use_tool(self, tool_call) -> Union[str, List[str], None]:
        tool_name = tool_call["function"]["name"]
        params = (
            tool_call["function"]["arguments"]
            if type(tool_call["function"]["arguments"]) is dict
            else json.loads(tool_call["function"]["arguments"])
        )

        try:
            # Find the tool in our unified tool sequence
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                available_tools = [t.name for t in self.tools]
                error_message = f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
                self.tool_logger.error(error_message)
                return error_message

            # Add reasoning to the log if present in params
            if "reasoning" in params:
                self.reasoning_log.append(params["reasoning"])

            # Check if caching is enabled and result is in cache
            cache_key = None
            if self.enable_caching:
                # Generate a stable cache key from the tool name and params
                cache_key = f"{tool_name}:{json.dumps(params, sort_keys=True)}"
                if cache_key in self.tool_cache:
                    cache_entry = self.tool_cache[cache_key]
                    # Check if cache entry is still valid based on TTL
                    if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                        self.tool_logger.info(f"Cache hit for tool: {tool_name}")
                        self.cache_hits += 1

                        # Record the tool usage but note it came from cache
                        self.tool_history.append(
                            {
                                "name": tool_name,
                                "params": params,
                                "result": cache_entry["result"][:500]
                                + ("..." if len(cache_entry["result"]) > 500 else ""),
                                "cached": True,
                            }
                        )

                        # Handle final answer case
                        if tool_name == "final_answer":
                            self.final_answer = cache_entry["result"]

                        return cache_entry["result"]
                # Record cache miss if we get here
                self.cache_misses += 1

            # Request confirmation for potentially sensitive operations
            sensitive_actions = [
                "delete",
                "remove",
                "send",
                "email",
                "subscribe",
                "unsubscribe",
                "cancel",
                "payment",
            ]
            requires_confirmation = any(
                action in tool_name.lower() for action in sensitive_actions
            )

            if requires_confirmation:
                # Create a user-friendly description of the action
                action_description = f"Use tool '{tool_name}' with parameters: {json.dumps(params, indent=2)}"
                # Request confirmation from the user
                confirmed = await self.ask_user_confirmation(
                    action_description, {"tool": tool_name, "params": params}
                )
                if not confirmed:
                    return f"Action cancelled by user: {tool_name}"

            # Use the tool and get standardized result
            tool_start_time = time.time()
            self.tool_logger.info(f"Using tool: {tool_name} with {params}")
            result = await tool.use(params)
            tool_execution_time = time.time() - tool_start_time

            if not isinstance(result, ToolResult):
                result = ToolResult.wrap(result)

            # Store result in cache if caching is enabled
            if self.enable_caching and cache_key and "nocache" not in params:
                # Don't cache errors or temporary results
                if (
                    not str(result.to_list()).lower().startswith("error")
                    and tool_execution_time > 0.1
                ):
                    self.tool_cache[cache_key] = {
                        "result": result.to_list(),
                        "timestamp": time.time(),
                    }
                    # Limit cache size to prevent memory issues
                    if len(self.tool_cache) > 1000:
                        # Remove oldest entries
                        sorted_cache = sorted(
                            self.tool_cache.items(), key=lambda x: x[1]["timestamp"]
                        )
                        for old_key, _ in sorted_cache[: len(sorted_cache) // 2]:
                            del self.tool_cache[old_key]

            # Record the tool use in history
            tool_history_entry = {
                "name": tool_name,
                "params": params,
                "result": str(result.to_list())[:500]
                + ("..." if len(str(result.to_list())) > 500 else ""),
                "execution_time": tool_execution_time,
            }
            self.tool_history.append(tool_history_entry)

            # Update metrics if ReactAgent instance
            try:
                # Only call _update_metrics if collect_metrics is True
                if getattr(self, "collect_metrics", False):
                    self._update_metrics("tool", tool_history_entry)
            except Exception as metrics_error:
                self.tool_logger.debug(
                    f"Metrics update error (non-critical): {metrics_error}"
                )

            # Update workflow context if available
            if self.workflow_context and self.name in self.workflow_context:
                self.workflow_context[self.name]["last_tool_used"] = tool_name

            self.tool_logger.debug(f"Tool Result: {result.to_list()}")

            # Generate tool summary
            tool_action_summary = (
                await self.model_provider.get_completion(
                    system=TOOL_ACTION_SUMMARY_PROMPT,
                    prompt=f"""
                <context>
                    <tool_call>Tool Name: '{tool_name}' with parameters '{params}'</tool_call>
                    <tool_call_result>{result.to_list()}</tool_call_result>
                </context>
                """,
                )
            ).get("response")

            self.tool_logger.debug(f"Tool Action Summary: {tool_action_summary}")
            self.memory.append(tool_action_summary)
            self.task_progress = "\n".join(
                [
                    f"Step {index}: {step}"
                    for index, step in enumerate(self.memory, start=1)
                ]
            )

            return result.to_list()
        except Exception as e:
            error_message = f"Error using tool {tool_name}: {str(e)}"
            self.tool_logger.error(error_message)
            return error_message

    async def _run_task(self, task, tool_use: bool = True) -> dict | None:
        self.agent_logger.debug(f"MAIN TASK: {task}")
        self.messages.append(
            {
                "role": "user",
                "content": f"""
                
                MAIN TASK: {task}
                """.strip(),
            }
        )
        result = await self._think_chain(tool_use=tool_use)
        return result

    async def _run_task_iteration(self, task):
        running_task = task
        self.iterations = 0

        try:
            while True:
                if (
                    self.max_iterations is not None
                    and self.iterations >= self.max_iterations
                ):
                    self.agent_logger.info(
                        f"Reached maximum iterations ({self.max_iterations})"
                    )
                    break

                self.iterations += 1
                self.agent_logger.info(f"Running Iteration: {self.iterations}")

                result = await self._run_task(task=running_task)
                print(result)
                if not result:
                    self.agent_logger.info(f"{self.name} Failed\n")
                    self.agent_logger.info(f"Task: {task}\n")
                    break

                if self.final_answer:
                    return self.final_answer

                # Add continuation check
                if self.iterations > 1 and result.get("message", {}).get(
                    "content"
                ) == result.get("previous_content"):
                    self.agent_logger.info(
                        "No progress made in this iteration, stopping"
                    )
                    break

                result["previous_content"] = result.get("message", {}).get("content")

            return result["message"]["content"] if result else None

        except Exception as e:
            self.agent_logger.error(f"Iteration error: {str(e)}")
            return None
        finally:
            if (
                self.max_iterations is not None
                and self.iterations >= self.max_iterations
            ):
                self.agent_logger.info(
                    f"Reached maximum iterations ({self.max_iterations})"
                )

    async def _safe_close_mcp_client(self):
        """Safely close the MCP client without causing cancellation issues."""
        if not hasattr(self, "mcp_client") or not self.mcp_client:
            return

        self.agent_logger.info("Safely closing MCP client connection")
        try:
            # Create a new task in the same event loop to close the client
            # This ensures the cancel scope is managed properly
            loop = asyncio.get_running_loop()
            close_task = loop.create_task(self.mcp_client.close())

            # Wait for the close task to complete with a timeout
            try:
                await asyncio.wait_for(close_task, timeout=5.0)
                self.agent_logger.info("MCP client closed successfully")
            except asyncio.TimeoutError:
                self.agent_logger.warning(
                    "MCP client close timed out, client may not be fully closed"
                )
            except asyncio.CancelledError:
                # If this task gets cancelled, detach the close task so it can finish
                close_task.add_done_callback(
                    lambda _: self.agent_logger.info(
                        "Detached MCP client close completed"
                    )
                )
                # Don't wait for it since we're being cancelled
                raise
        except Exception as e:
            self.agent_logger.warning(f"Error during safe MCP client close: {str(e)}")
        finally:
            # Clear the reference regardless of success
            self.mcp_client = None

    async def run(self, initial_task):
        try:
            self.initial_task = initial_task
            self.agent_logger.info(f"Starting task: {initial_task}")

            result = await self._run_task_iteration(task=initial_task)

            if result:
                self.result_logger.info(result)
                return result
            else:
                self.agent_logger.warning("Task completed without result")
                return None

        except Exception as e:
            self.agent_logger.error(f"Agent Error: {e}")
            return None
        finally:
            # Use our safe method to close the MCP client
            if hasattr(self, "mcp_client") and self.mcp_client:
                await self._safe_close_mcp_client()

    def set_model_provider(self, provider_model: str):
        self.model_provider = ModelProviderFactory.get_model_provider(provider_model)
        self.agent_logger.info(f"Model Provider Set to: {self.model_provider.name}")

    async def ask_user_confirmation(
        self, action_description: str, action_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Ask for user confirmation before executing potentially impactful actions.

        Args:
            action_description: A description of the action the agent wants to perform
            action_details: Additional details about the action (e.g. tool name, parameters)

        Returns:
            bool: True if confirmed, False otherwise
        """
        self.agent_logger.info(f"Requesting confirmation for: {action_description}")

        if self.confirmation_callback:
            try:
                # Try to handle as awaitable first
                if asyncio.iscoroutinefunction(self.confirmation_callback):
                    return bool(
                        await self.confirmation_callback(
                            action_description, action_details or {}
                        )
                    )
                # Handle as regular function
                return bool(
                    self.confirmation_callback(action_description, action_details or {})
                )
            except Exception as e:
                self.agent_logger.error(f"Error in confirmation callback: {e}")
                return False

        # Default implementation always allows actions when no callback provided
        self.agent_logger.info(
            "No confirmation callback provided, proceeding with action"
        )
        return True

    def _update_metrics(self, metric_type: str, data: Dict[str, Any]) -> None:
        """
        Base implementation of metrics tracking.
        Subclasses like ReactAgent can override this for actual metrics tracking.

        Args:
            metric_type: The type of metric (tool, model, etc.)
            data: The metric data to track
        """
        # Base implementation does nothing but log
        self.agent_logger.debug(
            f"Metrics update requested (not tracked in base Agent): {metric_type} - {data}"
        )
        return
