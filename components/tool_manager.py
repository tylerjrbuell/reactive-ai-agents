from __future__ import annotations
import json
import time
import asyncio
import inspect
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Sequence,
    Callable,
    TYPE_CHECKING,
    Union,
    Tuple,
)

from tools.base import Tool
from pydantic import BaseModel, Field

from loggers.base import Logger
from tools.abstractions import ToolProtocol, ToolResult, MCPToolWrapper
from agent_mcp.client import MCPClient
from model_providers.base import BaseModelProvider  # Needed for summary generation
from prompts.agent_prompts import (
    TOOL_ACTION_SUMMARY_PROMPT,
    TOOL_SUMMARY_CONTEXT_PROMPT,
)
from common.types import (
    ConfirmationCallbackProtocol,
    ConfirmationConfig,
    ConfirmationResult,
)

if TYPE_CHECKING:
    from context.agent_context import AgentContext  # Keep import here


class FinalAnswerTool(Tool):
    """Wrapper class to make the final_answer function ToolProtocol compatible."""

    name = "final_answer"
    description = (
        "Provides the final answer to the user's query and concludes the task."
    )
    tool_definition = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final textual answer to the user's query as a complete response to the original task.",
                    }
                },
                "required": ["answer"],
            },
        },
    }

    def __init__(self, context: "AgentContext"):
        self.context = context

    async def use(self, params: Dict[str, Any]) -> ToolResult:
        """Executes the final answer tool with the provided parameters."""
        answer = params.get("answer")
        if answer is None:
            return ToolResult("Error: Missing required parameter 'answer'.")

        # Set the final answer in the context
        self.context.session.final_answer = answer
        return ToolResult(f"Final answer has been recorded: {answer}")


class ToolManager(BaseModel):
    """Manages tool discovery, execution, caching, and history."""

    context: "AgentContext" = Field(exclude=True)  # Reference back to the main context

    # Configuration (mirrored or derived from context)
    enable_caching: bool = True
    cache_ttl: int = 3600
    confirmation_callback: Optional[ConfirmationCallbackProtocol] = None

    # Confirmation configuration
    confirmation_config: Dict[str, Any] = Field(default_factory=dict)

    # State
    tools: List[Tool] = Field(default_factory=list)
    tool_signatures: List[Dict[str, Any]] = Field(default_factory=list)
    tool_history: List[Dict[str, Any]] = Field(default_factory=list)
    tool_cache: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize tools based on context (MCP or local)
        self._initialize_tools()
        # Set config from context
        self.enable_caching = self.context.enable_caching
        self.cache_ttl = self.context.cache_ttl
        # Avoid direct assignment to confirmation_callback
        object.__setattr__(
            self, "confirmation_callback", self.context.confirmation_callback
        )
        # Initialize confirmation config from context or defaults
        self._initialize_confirmation_config()

    @property
    def agent_logger(self) -> Logger:
        assert self.context.agent_logger is not None
        return self.context.agent_logger

    @property
    def tool_logger(self) -> Logger:
        assert self.context.tool_logger is not None
        return self.context.tool_logger

    @property
    def model_provider(self) -> BaseModelProvider:
        assert self.context.model_provider is not None
        return self.context.model_provider

    def _initialize_tools(self):
        """Populates tools and signatures from MCP client or local list."""
        if self.context.mcp_client:
            # Ensure MCP client tools are loaded (might need await if not already done)
            # Assuming mcp_client.tools and mcp_client.tool_signatures are populated
            mcp_tools = getattr(self.context.mcp_client, "tools", [])
            self.tools = [MCPToolWrapper(t, self.context.mcp_client) for t in mcp_tools]
            self.tool_signatures = getattr(
                self.context.mcp_client, "tool_signatures", []
            )
            self.agent_logger.info(f"Initialized {len(self.tools)} tools via MCP.")
            servers = self.context.mcp_client.server_tools
            self.agent_logger.info(
                f"MCP Servers: {[server for server in servers.keys()]}"
            )
            self.agent_logger.info(
                f"MCP Tools: {[t.name for tool in servers.values() for t in tool]}"
            )
        elif self.context.tools:
            # Use locally provided tools
            self.tools = self.context.tools
            self.tool_signatures = [
                tool.tool_definition
                for tool in self.tools
                if hasattr(tool, "tool_definition")
            ]
            self.agent_logger.info(f"Initialized {len(self.tools)} local tools.")
        else:
            self.agent_logger.info("No MCP client or local tools provided.")

        # --- Inject final_answer tool if missing ---
        has_final_answer = any(tool.name == "final_answer" for tool in self.tools)
        if not has_final_answer:
            self.tool_logger.info("Injecting internal 'final_answer' tool.")
            internal_final_answer_tool = FinalAnswerTool(context=self.context)
            self.tools.append(internal_final_answer_tool)
        # --- End injection ---

        # Generate signatures AFTER all tools (including injected ones) are loaded
        self._generate_tool_signatures()

        self.tool_logger.info(
            f"ToolManager initialized with {len(self.tools)} total tools: {', '.join([tool.name for tool in self.tools])}"
        )

    def get_tool(self, tool_name: str) -> Optional[ToolProtocol]:
        """Finds a tool by name."""
        for tool in self.tools:
            if hasattr(tool, "name") and tool.name == tool_name:
                return tool
        return None

    async def use_tool(self, tool_call: Dict[str, Any]) -> Union[str, List[str], None]:
        """Executes a tool, handling caching, confirmation, history, and errors."""
        tool_name = tool_call.get("function", {}).get("name")
        if not tool_name:
            self.tool_logger.error("Tool call missing function name.")
            return "Error: Tool call missing function name."

        try:
            # Argument parsing robustnes
            arguments_raw = tool_call.get("function", {}).get("arguments")
            if isinstance(arguments_raw, str):
                try:
                    params = json.loads(arguments_raw)
                except json.JSONDecodeError as json_err:
                    err_msg = f"Invalid JSON arguments for tool {tool_name}: {json_err}. Arguments: '{arguments_raw}'"
                    self.tool_logger.error(err_msg)
                    return f"Error: {err_msg}"
            elif isinstance(arguments_raw, dict):
                params = arguments_raw
            else:
                err_msg = f"Invalid argument type for tool {tool_name}: {type(arguments_raw)}. Expected dict or JSON string."
                self.tool_logger.error(err_msg)
                return f"Error: {err_msg}"

        except Exception as e:
            err_msg = f"Error processing arguments for tool {tool_name}: {e}"
            self.tool_logger.error(err_msg)
            return f"Error: {err_msg}"

        tool = self.get_tool(tool_name)
        if not tool:
            available_tools = [t.name for t in self.tools]
            error_message = f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
            self.tool_logger.error(error_message)
            # Update history for the failed attempt
            self._add_to_history(
                tool_name, params, error_message, error=True, execution_time=0.0
            )
            return error_message

        # Add reasoning to the main context log
        if "reasoning" in params:
            self.context.session.reasoning_log.append(params["reasoning"])
            # Optionally remove reasoning from params sent to tool if not part of schema
            # tool_schema = tool.tool_definition.get("function", {}).get("parameters", {}).get("properties", {})
            # if "reasoning" not in tool_schema:
            #    params.pop("reasoning", None)

        # --- Caching Logic ---
        cache_key = None
        if self.enable_caching:
            try:
                cache_key = f"{tool_name}:{json.dumps(params, sort_keys=True)}"
                if cache_key in self.tool_cache:
                    cache_entry = self.tool_cache[cache_key]
                    if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                        self.tool_logger.info(f"Cache hit for tool: {tool_name}")
                        self.cache_hits += 1
                        cached_result = cache_entry["result"]
                        summary = await self._generate_and_log_summary(
                            tool_name, params, cached_result
                        )
                        self._add_to_history(
                            tool_name, params, cached_result, summary, cached=True
                        )
                        # Handle final answer from cache
                        if tool_name == "final_answer":
                            self.context.session.final_answer = str(
                                cached_result
                            )  # Ensure string
                        return cached_result
                self.cache_misses += 1
            except TypeError as e:
                self.tool_logger.warning(
                    f"Could not generate cache key for {tool_name}: {e}. Skipping cache."
                )
                cache_key = None  # Ensure cache_key is None if serialization fails
            except Exception as e:
                self.tool_logger.error(
                    f"Unexpected error during cache check for {tool_name}: {e}"
                )
                cache_key = None  # Safely skip caching on unexpected errors

        # --- Confirmation Logic ---
        # Get tool description for confirmation check
        description_lower = ""
        if hasattr(tool, "tool_definition") and tool.tool_definition:
            func_def = tool.tool_definition.get("function", {})
            description = func_def.get("description", "")
            if description:
                description_lower = description.lower()

        # Check if tool requires confirmation using the new method
        requires_confirmation = self._tool_requires_confirmation(
            tool_name, description_lower
        )

        if requires_confirmation and self.confirmation_callback:
            action_description = f"Use tool '{tool_name}' with parameters: {json.dumps(params, indent=2)}"
            try:
                # The confirmation callback now can return tuple (confirmed, feedback)
                confirmation_result = await self._request_confirmation(
                    action_description, {"tool": tool_name, "params": params}
                )

                # Handle different return types from the callback
                if isinstance(confirmation_result, tuple):
                    confirmed, feedback = confirmation_result
                else:
                    confirmed = bool(confirmation_result)
                    feedback = None

                if not confirmed:
                    cancel_msg = f"Action cancelled by user: {tool_name}"
                    self.tool_logger.info(cancel_msg)
                    self._add_to_history(tool_name, params, cancel_msg, cancelled=True)

                    # If user provided feedback, inject it into the agent's context
                    if feedback:
                        self._inject_user_feedback(tool_name, params, feedback)
                        cancel_msg += f" - Feedback: {feedback}"

                    return cancel_msg

                # If confirmed but user provided feedback, still inject it
                if feedback:
                    self._inject_user_feedback(tool_name, params, feedback)

            except Exception as e:
                self.tool_logger.error(
                    f"Error during confirmation callback for {tool_name}: {e}"
                )
                # Decide how to proceed: default to cancel or allow? Let's default to cancel for safety.
                error_msg = (
                    f"Error during confirmation, cancelling action {tool_name}: {e}"
                )
                self._add_to_history(
                    tool_name, params, error_msg, error=True, execution_time=0.0
                )
                return error_msg

        # --- Tool Execution ---
        try:
            tool_start_time = time.time()
            self.tool_logger.info(f"Using tool: {tool_name} with {params}")
            # Ensure params match tool schema if possible (future enhancement)
            result_obj = await tool.use(params)  # Expects ToolResult or wraps it
            tool_execution_time = time.time() - tool_start_time

            if not isinstance(result_obj, ToolResult):
                result_obj = ToolResult.wrap(result_obj)

            result_list = result_obj.to_list()
            result_str = str(result_list)  # For logging and history

            # Log result
            self.tool_logger.info(
                f"Tool {tool_name} completed in {tool_execution_time:.2f}s"
            )
            if len(result_str) > 500:
                self.tool_logger.debug(
                    f"Result Preview: {result_str[:500]}... (truncated)"
                )
            else:
                self.tool_logger.debug(f"Result: {result_str}")

            # Generate and log summary
            summary = await self._generate_and_log_summary(
                tool_name, params, result_list
            )

            # Add to history BEFORE caching
            self._add_to_history(
                tool_name,
                params,
                result_list,
                summary,
                execution_time=tool_execution_time,
            )

            # Add to cache if enabled and successful
            if self.enable_caching and cache_key and "nocache" not in params:
                # Don't cache errors, very short results, or if tool failed
                if (
                    not result_str.lower().startswith("error")
                    and tool_execution_time > 0.1
                ):
                    self.tool_cache[cache_key] = {
                        "result": result_list,  # Store the structured result
                        "timestamp": time.time(),
                    }
                    self._limit_cache_size()  # Prune cache if needed

            # Log success before returning
            if tool_name != "final_answer":  # Don't track final_answer as successful?
                # Append to session successful_tools
                self.context.session.successful_tools.append(tool_name)
            return result_list  # Return the structured result

        except Exception as e:
            # Capture detailed traceback
            import traceback

            tb_str = traceback.format_exc()
            error_message = f"Error using tool {tool_name}: {str(e)}"
            self.tool_logger.error(error_message)
            self.tool_logger.debug(
                f"Traceback:\n{tb_str}"
            )  # Log full traceback for debugging
            self._add_to_history(
                tool_name, params, error_message, error=True, execution_time=0.0
            )
            return error_message  # Return error message to the agent loop

    async def _request_confirmation(
        self, description: str, details: Dict[str, Any]
    ) -> Union[bool, Tuple[bool, Optional[str]]]:
        """Handles calling the confirmation callback (async or sync)."""
        if not self.confirmation_callback:
            return True  # Default to allow if no callback

        result = None
        if asyncio.iscoroutinefunction(self.confirmation_callback):
            result = await self.confirmation_callback(description, details)
        else:
            # For synchronous callbacks
            result = self.confirmation_callback(description, details)

        # Check if result is awaitable and await it if needed
        if inspect.isawaitable(result):
            result = await result

        # Ensure we return the expected type
        if isinstance(result, tuple) and len(result) == 2:
            confirmed, feedback = result
            return (bool(confirmed), feedback)
        else:
            return bool(result)

    def _inject_user_feedback(
        self, tool_name: str, params: Dict[str, Any], feedback: str
    ) -> None:
        """Inject user feedback into the agent's context as a message."""
        self.tool_logger.info(
            f"Injecting user feedback for tool '{tool_name}': {feedback}"
        )

        # Create a user message with the feedback
        feedback_message = {
            "role": "user",
            "content": f"Feedback for your '{tool_name}' tool call: {feedback}\nPlease adjust your approach based on this feedback.",
        }

        # Add to the session messages
        self.context.session.messages.append(feedback_message)

        # Also add to reasoning log for clarity
        self.context.session.reasoning_log.append(
            f"User feedback for tool '{tool_name}': {feedback}"
        )

    def _initialize_confirmation_config(self):
        """Initialize the confirmation configuration from context or defaults."""
        if (
            hasattr(self.context, "confirmation_config")
            and self.context.confirmation_config
        ):
            self.confirmation_config = self.context.confirmation_config
            self.tool_logger.info("Using confirmation configuration from context.")
        else:
            # Default configuration
            self.confirmation_config = {
                "always_confirm": [],  # List of tool names that always require confirmation
                "never_confirm": [
                    "final_answer"
                ],  # Tools that never require confirmation
                "patterns": {  # Patterns to match in tool names or descriptions
                    "write": "confirm",
                    "delete": "confirm",
                    "remove": "confirm",
                    "update": "confirm",
                    "create": "confirm",
                    "send": "confirm",
                    "email": "confirm",
                    "subscribe": "confirm",
                    "unsubscribe": "confirm",
                    "payment": "confirm",
                    "post": "confirm",
                    "put": "confirm",
                },
                "default_action": "proceed",  # Default action if no pattern matches: "proceed" or "confirm"
            }
            self.tool_logger.info("Using default confirmation configuration.")

    def _tool_requires_confirmation(
        self, tool_name: str, description: str = ""
    ) -> bool:
        """Determine if a tool requires confirmation based on configuration."""
        config = self.confirmation_config

        # Check if tool is in the always_confirm list
        if tool_name in config.get("always_confirm", []):
            return True

        # Check if tool is in the never_confirm list
        if tool_name in config.get("never_confirm", []):
            return False

        # Check patterns
        patterns = config.get("patterns", {})
        for pattern, action in patterns.items():
            if pattern in tool_name.lower() or (
                description and pattern in description.lower()
            ):
                return action.lower() == "confirm"

        # Use default action
        return config.get("default_action", "proceed").lower() == "confirm"

    def _add_to_history(
        self,
        tool_name,
        params,
        result,
        summary=None,
        execution_time=None,
        cached=False,
        cancelled=False,
        error=False,
    ):
        """Adds an entry to the tool history."""
        # Truncate result for history storage if necessary
        result_repr = str(result)
        if len(result_repr) > 1000:  # Limit history result size
            result_repr = result_repr[:1000] + "..."

        entry = {
            "name": tool_name,
            "params": params,  # Consider masking sensitive params here
            "result": result_repr,
            "summary": summary,
            "timestamp": time.time(),
            "cached": cached,
            "cancelled": cancelled,
            "error": error,
            **(
                {"execution_time": execution_time} if execution_time is not None else {}
            ),
        }
        if not error and not cancelled:
            self.context.session.successful_tools.append(tool_name)

        self.tool_history.append(entry)

        # Optionally update metrics (delegated)
        if self.context.collect_metrics_enabled and self.context.metrics_manager:
            try:
                # Pass relevant info for metrics update
                self.context.metrics_manager.update_tool_metrics(entry)
            except Exception as metrics_error:
                self.tool_logger.debug(
                    f"Metrics update error (non-critical): {metrics_error}"
                )

    def _limit_cache_size(self, max_entries=1000):
        """Limits the size of the tool cache."""
        if len(self.tool_cache) > max_entries:
            self.tool_logger.info(
                f"Cache size ({len(self.tool_cache)}) exceeds limit ({max_entries}). Pruning..."
            )
            # Simple strategy: remove oldest half
            sorted_cache = sorted(
                self.tool_cache.items(), key=lambda item: item[1]["timestamp"]
            )
            num_to_remove = len(sorted_cache) - (max_entries // 2)  # Keep half
            keys_to_remove = [k for k, v in sorted_cache[:num_to_remove]]
            for key in keys_to_remove:
                del self.tool_cache[key]
            self.tool_logger.info(f"Removed {len(keys_to_remove)} entries from cache.")

    async def _generate_and_log_summary(
        self, tool_name: str, params: dict, result: Any
    ) -> str:
        """Generates a summary of the tool action and adds it to context memory/progress."""
        try:
            result_str = str(result)
            # Limit result string size for the prompt
            result_for_prompt = (
                result_str[:2000] + "..." if len(result_str) > 2000 else result_str
            )

            # Use centralized context prompt
            summary_context_prompt = TOOL_SUMMARY_CONTEXT_PROMPT.format(
                tool_name=tool_name,
                params=str(params),  # Ensure params are stringified safely
                result_str=result_for_prompt,
            )

            summary_result = await self.model_provider.get_completion(
                system=TOOL_ACTION_SUMMARY_PROMPT,  # Keep existing system prompt
                prompt=summary_context_prompt,
            )
            tool_action_summary = summary_result.get(
                "response", f"Executed tool {tool_name}."
            )
            self.tool_logger.debug(f"Tool Action Summary: {tool_action_summary}")

            # Add summary to context's reasoning log and update task progress
            self.context.session.reasoning_log.append(tool_action_summary)
            self.context.session.task_progress += f"\n- {tool_action_summary}"
            return tool_action_summary

        except Exception as e:
            self.tool_logger.error(
                f"Failed to generate tool action summary for {tool_name}: {e}"
            )
            fallback_summary = f"Successfully executed tool '{tool_name}'."
            self.context.session.reasoning_log.append(fallback_summary)
            self.context.session.task_progress += f"\n- {fallback_summary}"
            return fallback_summary

    def _generate_tool_signatures(self):
        """Generates tool signatures from the schemas of available tools."""
        self.tool_signatures = []
        for tool in self.tools:
            try:

                # Handle MCPToolWrapper specifically if needed
                if hasattr(tool, "tool_definition") and isinstance(
                    tool.tool_definition, dict
                ):
                    self.tool_signatures.append(tool.tool_definition)
                else:
                    self.tool_logger.warning(
                        f"Tool '{tool.name}' is missing a valid schema/tool_definition attribute for signature generation."
                    )
            except Exception as e:
                self.tool_logger.error(
                    f"Error generating signature for tool '{getattr(tool, 'name', 'unknown')}': {e}"
                )
        self.tool_logger.debug(
            f"Generated {len(self.tool_signatures)} tool signatures."
        )

    def get_available_tools(self) -> List[Tool]:
        """Returns a list of all available tools."""
        return self.tools

    def get_available_tool_names(self) -> set[str]:
        """Returns a set of all available tool names."""
        return set([tool.name for tool in self.tools])

    def get_last_tool_action(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent entry from the tool history, if any."""
        return self.tool_history[-1] if self.tool_history else None
