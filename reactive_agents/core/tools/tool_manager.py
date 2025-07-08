from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING
import time
import json
import asyncio
import inspect
from pydantic import BaseModel, Field
from reactive_agents.core.types.confirmation_types import ConfirmationCallbackProtocol
from reactive_agents.core.types.event_types import AgentStateEvent
from reactive_agents.core.tools.data_extractor import DataExtractor, SearchDataManager
from reactive_agents.core.reasoning.prompts.agent_prompts import (
    TOOL_ACTION_SUMMARY_PROMPT,
    TOOL_SUMMARY_CONTEXT_PROMPT,
)
from reactive_agents.core.tools.base import Tool
from reactive_agents.core.tools.abstractions import (
    MCPToolWrapper,
    ToolProtocol,
    ToolResult,
)
from reactive_agents.utils.logging import Logger
from reactive_agents.providers.llm.base import BaseModelProvider

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


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
        return ToolResult(answer)


class ToolGuard:
    """Middleware for enforcing tool usage policies (rate limits, cooldowns, confirmation, etc)."""

    def __init__(self):
        self.usage_log = {}  # {tool_name: [timestamps]}
        self.rate_limits = {}
        self.confirmation_required = set()
        self.admin_required = set()
        self.cooldowns = {}

    def add_default_guards(self):
        """Add default guard policies for common sensitive tools."""
        # Rate limits: (max_calls, per_seconds)
        self.rate_limits.update(
            {
                # Email operations
                "write_email": (1, 60),  # 1 email per 60 seconds
                "send_email": (1, 60),  # 1 email per 60 seconds
                "compose_email": (1, 60),  # 1 email per 60 seconds
                # File operations
                "delete_file": (5, 300),  # 5 deletions per 5 minutes
                "move_file": (10, 300),  # 10 moves per 5 minutes
                "copy_file": (10, 300),  # 10 copies per 5 minutes
                # Database operations
                "delete_record": (10, 300),  # 10 deletions per 5 minutes
                "update_record": (20, 300),  # 20 updates per 5 minutes
                "drop_table": (1, 3600),  # 1 table drop per hour
                "truncate_table": (1, 3600),  # 1 truncate per hour
                # API operations
                "api_call": (30, 60),  # 30 API calls per minute
                "http_request": (30, 60),  # 30 HTTP requests per minute
                "webhook": (10, 300),  # 10 webhooks per 5 minutes
                # System operations
                "execute_command": (5, 300),  # 5 commands per 5 minutes
                "run_script": (3, 600),  # 3 scripts per 10 minutes
                "install_package": (1, 3600),  # 1 package install per hour
                # Financial operations
                "make_payment": (1, 3600),  # 1 payment per hour
                "transfer_money": (1, 3600),  # 1 transfer per hour
                "create_invoice": (5, 300),  # 5 invoices per 5 minutes
                # User management
                "create_user": (3, 600),  # 3 users per 10 minutes
                "delete_user": (1, 3600),  # 1 user deletion per hour
                "change_password": (5, 300),  # 5 password changes per 5 minutes
                # Email management (Gmail specific)
                "trash_emails": (10, 300),  # 10 trash operations per 5 minutes
                "archive_emails": (20, 300),  # 20 archive operations per 5 minutes
                "star_emails": (30, 300),  # 30 star operations per 5 minutes
            }
        )

        # Tools requiring explicit confirmation
        self.confirmation_required.update(
            {
                # Email operations
                "write_email",
                "send_email",
                "compose_email",
                # Destructive file operations
                "delete_file",
                "delete_directory",
                "format_drive",
                # Database operations
                "delete_record",
                "drop_table",
                "truncate_table",
                "delete_database",
                # System operations
                "execute_command",
                "run_script",
                "install_package",
                "uninstall_package",
                "restart_service",
                "stop_service",
                "kill_process",
                # Financial operations
                "make_payment",
                "transfer_money",
                "create_invoice",
                "refund_payment",
                # User management
                "delete_user",
                "change_password",
                "reset_password",
                "grant_admin",
                # Network operations
                "open_port",
                "close_port",
                "block_ip",
                "unblock_ip",
                # Email management (bulk operations)
                "trash_emails",
                "archive_emails",
                "delete_emails",
            }
        )

        # Tools requiring admin privileges (additional logging)
        self.admin_required.update(
            {
                "drop_table",
                "delete_database",
                "format_drive",
                "kill_process",
                "grant_admin",
                "open_port",
                "close_port",
                "block_ip",
                "unblock_ip",
            }
        )

        # Tools with cooldown periods (minimum time between uses)
        self.cooldowns.update(
            {
                "restart_service": 300,  # 5 minutes between restarts
                "install_package": 1800,  # 30 minutes between installs
                "make_payment": 3600,  # 1 hour between payments
                "delete_user": 3600,  # 1 hour between user deletions
            }
        )

    def add_rate_limit(self, tool_name: str, max_calls: int, per_seconds: int):
        """Add a rate limit for a specific tool."""
        self.rate_limits[tool_name] = (max_calls, per_seconds)

    def add_confirmation_required(self, tool_name: str):
        """Add a tool to the confirmation required list."""
        self.confirmation_required.add(tool_name)

    def add_admin_required(self, tool_name: str):
        """Add a tool to the admin required list."""
        self.admin_required.add(tool_name)

    def add_cooldown(self, tool_name: str, cooldown_seconds: int):
        """Add a cooldown period for a specific tool."""
        self.cooldowns[tool_name] = cooldown_seconds

    def remove_rate_limit(self, tool_name: str):
        """Remove rate limit for a specific tool."""
        self.rate_limits.pop(tool_name, None)

    def remove_confirmation_required(self, tool_name: str):
        """Remove a tool from the confirmation required list."""
        self.confirmation_required.discard(tool_name)

    def remove_admin_required(self, tool_name: str):
        """Remove a tool from the admin required list."""
        self.admin_required.discard(tool_name)

    def remove_cooldown(self, tool_name: str):
        """Remove cooldown for a specific tool."""
        self.cooldowns.pop(tool_name, None)

    def clear_all_guards(self):
        """Remove all guard policies."""
        self.rate_limits.clear()
        self.confirmation_required.clear()
        self.admin_required.clear()
        self.cooldowns.clear()

    def can_use(self, tool_name: str) -> bool:
        import time

        now = time.time()

        # Check rate limits
        if tool_name in self.rate_limits:
            max_calls, per_seconds = self.rate_limits[tool_name]
            timestamps = self.usage_log.get(tool_name, [])
            # Only keep timestamps within the window
            timestamps = [t for t in timestamps if now - t < per_seconds]
            if len(timestamps) >= max_calls:
                return False
            self.usage_log[tool_name] = timestamps

        # Check cooldowns
        if tool_name in self.cooldowns:
            cooldown_seconds = self.cooldowns[tool_name]
            timestamps = self.usage_log.get(tool_name, [])
            if timestamps and (now - timestamps[-1]) < cooldown_seconds:
                return False

        return True

    def record_use(self, tool_name: str):
        import time

        self.usage_log.setdefault(tool_name, []).append(time.time())

    def needs_confirmation(self, tool_name: str) -> bool:
        return tool_name in self.confirmation_required

    def needs_admin(self, tool_name: str) -> bool:
        return tool_name in self.admin_required

    def get_rate_limit_info(self, tool_name: str) -> dict:
        """Get information about rate limits for a tool."""
        if tool_name in self.rate_limits:
            max_calls, per_seconds = self.rate_limits[tool_name]
            timestamps = self.usage_log.get(tool_name, [])
            import time

            now = time.time()
            recent_calls = len([t for t in timestamps if now - t < per_seconds])
            return {
                "max_calls": max_calls,
                "per_seconds": per_seconds,
                "recent_calls": recent_calls,
                "remaining_calls": max_calls - recent_calls,
                "window_remaining": (
                    per_seconds - (now - timestamps[-1]) if timestamps else 0
                ),
            }
        return {}

    def get_cooldown_info(self, tool_name: str) -> dict:
        """Get information about cooldown for a tool."""
        if tool_name in self.cooldowns:
            cooldown_seconds = self.cooldowns[tool_name]
            timestamps = self.usage_log.get(tool_name, [])
            import time

            now = time.time()
            if timestamps:
                time_since_last = now - timestamps[-1]
                cooldown_remaining = max(0, cooldown_seconds - time_since_last)
                return {
                    "cooldown_seconds": cooldown_seconds,
                    "time_since_last": time_since_last,
                    "cooldown_remaining": cooldown_remaining,
                    "can_use": cooldown_remaining <= 0,
                }
        return {}


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

    # Data extraction components
    data_extractor: DataExtractor = Field(default_factory=DataExtractor)
    search_data_manager: SearchDataManager = Field(default_factory=SearchDataManager)
    guard: ToolGuard = Field(default_factory=ToolGuard, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Set config from context
        self.enable_caching = self.context.enable_caching
        self.cache_ttl = self.context.cache_ttl
        # Avoid direct assignment to confirmation_callback
        object.__setattr__(
            self, "confirmation_callback", self.context.confirmation_callback
        )
        # Initialize confirmation config from context or defaults
        self._initialize_confirmation_config()
        # Add default guards
        self.guard.add_default_guards()

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

    async def _initialize_tools(self, attempts: int = 0):
        """Populates tools and signatures from MCP client and/or local list."""
        generate = False

        # Initialize tools list to accumulate both MCP and custom tools
        self.tools = []
        self.tool_signatures = []

        # Load MCP tools if available
        if self.context.mcp_client:
            # Ensure MCP client tools are loaded (might need await if not already done)
            # Assuming mcp_client.tools and mcp_client.tool_signatures are populated
            mcp_tools = getattr(self.context.mcp_client, "tools", [])
            mcp_wrapped_tools = [
                MCPToolWrapper(t, self.context.mcp_client) for t in mcp_tools
            ]
            self.tools.extend(mcp_wrapped_tools)

            mcp_signatures = getattr(self.context.mcp_client, "tool_signatures", [])
            self.tool_signatures.extend(mcp_signatures)

            self.agent_logger.info(
                f"Initialized {len(mcp_wrapped_tools)} tools via MCP."
            )
            servers = self.context.mcp_client.server_tools
            self.agent_logger.info(
                f"MCP Servers: {[server for server in servers.keys()]}"
            )
            self.agent_logger.info(
                f"MCP Tools: {[t.name for tool in servers.values() for t in tool]}"
            )
            generate = True

        # Load custom tools if available
        if self.context.tools:
            # Add locally provided tools
            self.tools.extend(self.context.tools)

            custom_signatures = [
                tool.tool_definition
                for tool in self.context.tools
                if hasattr(tool, "tool_definition")
            ]
            self.tool_signatures.extend(custom_signatures)

            self.agent_logger.info(
                f"Initialized {len(self.context.tools)} custom tools."
            )
            generate = True

        # Check if we have any tools at all
        if not self.tools:
            self.agent_logger.warning("No MCP client or custom tools provided")

        # --- Inject final_answer tool if missing ---
        has_final_answer = any(tool.name == "final_answer" for tool in self.tools)
        if not has_final_answer:
            self.tool_logger.info("Injecting internal 'final_answer' tool.")
            internal_final_answer_tool = FinalAnswerTool(context=self.context)
            self.tools.append(internal_final_answer_tool)
            generate = True

        # Generate signatures AFTER all tools (including injected ones) are loaded
        if generate:
            self._generate_tool_signatures()
            self.tool_logger.info(
                f"Initialized with {len(self.tools)} total tool(s): \n{', '.join([tool.name for tool in self.tools])}"
            )
            generated = True

    def get_tool(self, tool_name: str) -> Optional[ToolProtocol]:
        """Finds a tool by name."""
        for tool in self.tools:
            if hasattr(tool, "name") and tool.name == tool_name:
                return tool
        return None

    async def use_tool(self, tool_call: Dict[str, Any]) -> Union[str, List[str], None]:
        """Executes a tool, handling caching, confirmation, history, errors, and guard middleware."""
        tool_name = tool_call.get("function", {}).get("name")
        if not tool_name:
            tool_name = tool_call.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            return "Error: Tool name is missing or invalid."
        # === ToolGuard middleware ===
        if not self.guard.can_use(tool_name):
            return f"Error: Tool '{tool_name}' usage is rate-limited. Please wait before using it again."
        if self.guard.needs_confirmation(tool_name):
            # Use existing confirmation logic if available
            if self.confirmation_callback:
                confirmed = self.confirmation_callback(tool_name, tool_call)
                if inspect.iscoroutine(confirmed):
                    confirmed = await confirmed
                if isinstance(confirmed, tuple):
                    confirmed = confirmed[0]
                if not confirmed:
                    return f"Error: Tool '{tool_name}' usage was cancelled by user confirmation."
        # === End ToolGuard ===
        result = await self._actually_call_tool(tool_call)
        self.guard.record_use(tool_name)
        return result

    async def _actually_call_tool(
        self, tool_call: Dict[str, Any]
    ) -> Union[str, List[str], None]:
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
            # Emit TOOL_FAILED event
            self.context.emit_event(
                AgentStateEvent.TOOL_FAILED,
                {"tool_name": tool_name, "parameters": params, "error": error_message},
            )
            self._add_to_history(
                tool_name, params, error_message, error=True, execution_time=0.0
            )
            return error_message

        # Emit TOOL_CALLED event before execution
        self.context.emit_event(
            AgentStateEvent.TOOL_CALLED, {"tool_name": tool_name, "parameters": params}
        )

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
                        # Emit TOOL_COMPLETED event
                        self.context.emit_event(
                            AgentStateEvent.TOOL_COMPLETED,
                            {
                                "tool_name": tool_name,
                                "parameters": params,
                                "result": cached_result,
                                "execution_time": time.time() - time.time(),
                            },
                        )
                        # Emit FINAL_ANSWER_SET if tool is final_answer
                        if tool_name == "final_answer":
                            self.context.emit_event(
                                AgentStateEvent.FINAL_ANSWER_SET,
                                {
                                    "tool_name": tool_name,
                                    "answer": cached_result,
                                    "parameters": params,
                                },
                            )
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

                    # Emit TOOL_FAILED event
                    self.context.emit_event(
                        AgentStateEvent.TOOL_FAILED,
                        {
                            "tool_name": tool_name,
                            "parameters": params,
                            "error": cancel_msg,
                        },
                    )
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
                # Emit TOOL_FAILED event
                self.context.emit_event(
                    AgentStateEvent.TOOL_FAILED,
                    {"tool_name": tool_name, "parameters": params, "error": error_msg},
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
                # Add to session successful_tools (now a set)
                self.context.session.successful_tools.add(tool_name)
            # Emit TOOL_COMPLETED event
            self.context.emit_event(
                AgentStateEvent.TOOL_COMPLETED,
                {
                    "tool_name": tool_name,
                    "parameters": params,
                    "result": result_list,
                    "execution_time": tool_execution_time,
                },
            )
            # Emit FINAL_ANSWER_SET if tool is final_answer
            if tool_name == "final_answer":
                self.context.emit_event(
                    AgentStateEvent.FINAL_ANSWER_SET,
                    {
                        "tool_name": tool_name,
                        "answer": result_list,
                        "parameters": params,
                    },
                )
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
            # Emit TOOL_FAILED event
            self.context.emit_event(
                AgentStateEvent.TOOL_FAILED,
                {
                    "tool_name": tool_name,
                    "parameters": params,
                    "error": error_message,
                    "traceback": tb_str,
                },
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
            self.context.session.successful_tools.add(tool_name)

        self.tool_history.append(entry)

        # Optionally update metrics (delegated)
        if self.context.collect_metrics_enabled and self.context.metrics_manager:
            try:
                self.context.metrics_manager.update_tool_metrics(entry)
                # Emit METRICS_UPDATED event after metrics update
                self.context.emit_event(
                    AgentStateEvent.METRICS_UPDATED,
                    {"metrics": self.context.metrics_manager.get_metrics()},
                )
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

            # Use the enhanced TOOL_SUMMARY_CONTEXT_PROMPT from agent_prompts.py
            summary_context_prompt = TOOL_SUMMARY_CONTEXT_PROMPT.format(
                tool_name=tool_name,
                params=str(params),  # Ensure params are stringified safely
                result_str=result_for_prompt,
            )

            summary_result = await self.model_provider.get_completion(
                system=TOOL_ACTION_SUMMARY_PROMPT,  # Keep existing system prompt
                prompt=summary_context_prompt,  # Use the enhanced prompt from agent_prompts.py
                options=self.context.model_provider_options,
            )
            tool_action_summary = (
                summary_result.message.content.strip() or f"Executed tool {tool_name}."
            )
            # Ensure tool summary is clearly marked
            if not tool_action_summary.startswith("[TOOL SUMMARY]"):
                tool_action_summary = f"[TOOL SUMMARY] {tool_action_summary}"
            # Append as atomic assistant message to context
            # self.context.session.messages.append(
            #     {
            #         "role": "assistant",
            #         "content": tool_action_summary,
            #     }
            # )

            # Enhanced logging for debugging
            self.tool_logger.debug(f"Tool Action Summary: {tool_action_summary}")

            # Store structured data for search tools
            if "search" in tool_name and result:
                self.search_data_manager.store_search_data(
                    tool_name, params, result, self.data_extractor
                )

            # Validate tool result usage
            validation = self.validate_tool_result_usage(tool_name, params, result)
            if not validation["valid"]:
                self.tool_logger.warning(
                    f"Tool result validation failed for {tool_name}: {validation['warnings']}"
                )
                if validation["suggestions"]:
                    self.tool_logger.info(f"Suggestions: {validation['suggestions']}")

            # Add summary to context's reasoning log and update task progress
            self.context.session.reasoning_log.append(tool_action_summary)
            self.context.session.task_progress.append(tool_action_summary)
            return tool_action_summary

        except Exception as e:
            self.tool_logger.error(
                f"Failed to generate tool action summary for {tool_name}: {e}"
            )
            fallback_summary = f"Successfully executed tool '{tool_name}'."
            self.context.session.reasoning_log.append(fallback_summary)
            self.context.session.task_progress.append(fallback_summary)
            return fallback_summary

    def validate_tool_result_usage(
        self, tool_name: str, params: dict, result: Any
    ) -> Dict[str, Any]:
        """Validate that tool results are being used correctly and not ignored."""
        validation = {
            "valid": True,
            "warnings": [],
            "suggestions": [],
            "extracted_data": {},
        }

        try:
            # Get extracted data from search results using the new manager
            extracted_data = self.search_data_manager.get_extracted_data(tool_name)
            if extracted_data:
                validation["extracted_data"] = extracted_data

                # Check for various data types
                for data_type, values in extracted_data.items():
                    if values:
                        # Ensure values is a proper list and handle any unhashable types
                        try:
                            if isinstance(values, list):
                                # Filter out any unhashable types and take first 3 items
                                safe_values = []
                                for value in values:
                                    try:
                                        # Test if the value is hashable by trying to use it as a dict key
                                        _ = {value: None}
                                        safe_values.append(str(value))
                                        if len(safe_values) >= 3:
                                            break
                                    except (TypeError, ValueError):
                                        # Skip unhashable values
                                        continue

                                if safe_values:
                                    validation["suggestions"].append(
                                        f"Found {data_type}: {safe_values}..."
                                    )
                            else:
                                # Handle non-list values
                                validation["suggestions"].append(
                                    f"Found {data_type}: {str(values)}"
                                )
                        except Exception as e:
                            # If we can't process the values, just log it and continue
                            self.tool_logger.debug(
                                f"Could not process {data_type} values: {e}"
                            )

            # Tool-specific validation
            if tool_name in ["brave_web_search", "brave_local_search"]:
                result_str = str(result)

                # Check if search returned meaningful results
                if any(
                    phrase in result_str.lower()
                    for phrase in ["not available", "no results", "not found", "error"]
                ):
                    validation["warnings"].append(
                        "Search returned no meaningful results"
                    )

                # Check for placeholder indicators
                placeholder_indicators = [
                    "placeholder",
                    "example",
                    "sample",
                    "test data",
                    "dummy",
                ]
                if any(
                    indicator in result_str.lower()
                    for indicator in placeholder_indicators
                ):
                    validation["warnings"].append(
                        "Search results contain placeholder indicators"
                    )

            elif tool_name == "write_file":
                content = params.get("content", "")

                # Check for placeholder content
                placeholder_indicators = [
                    "placeholder",
                    "example",
                    "sample",
                    "test",
                    "dummy",
                    "TODO",
                    "FIXME",
                ]
                if any(
                    indicator in content.lower() for indicator in placeholder_indicators
                ):
                    validation["warnings"].append(
                        "File content appears to contain placeholder data"
                    )

                # Check if content matches extracted data from searches using the new manager
                all_search_data = self.search_data_manager.get_all_search_data()
                for search_tool, search_info in all_search_data.items():
                    extracted_data = search_info.get("extracted_data", {})

                    # Use the data extractor's validation method
                    data_validation = self.data_extractor.validate_data_usage(
                        content, self.data_extractor.extract_all("")
                    )
                    if not data_validation["valid"]:
                        validation["warnings"].extend(data_validation["warnings"])
                        validation["suggestions"].extend(data_validation["suggestions"])

            elif tool_name in ["read_file", "read_multiple_files"]:
                # Validate file reading operations
                result_str = str(result)
                if "error" in result_str.lower() or "not found" in result_str.lower():
                    validation["warnings"].append("File read operation may have failed")

            elif tool_name in ["create_directory", "move_file", "edit_file"]:
                # Validate file operations
                result_str = str(result)
                if "error" in result_str.lower() or "failed" in result_str.lower():
                    validation["warnings"].append("File operation may have failed")
                elif "success" in result_str.lower():
                    validation["suggestions"].append(
                        "File operation completed successfully"
                    )

            # General validation for any tool
            result_str = str(result)

            # Check for error indicators
            error_indicators = [
                "error",
                "failed",
                "exception",
                "timeout",
                "not found",
                "unauthorized",
            ]
            if any(indicator in result_str.lower() for indicator in error_indicators):
                validation["warnings"].append(
                    "Tool execution may have encountered an error"
                )

            # Check for success indicators
            success_indicators = ["success", "completed", "done", "ok", "successful"]
            if any(indicator in result_str.lower() for indicator in success_indicators):
                validation["suggestions"].append(
                    "Tool execution completed successfully"
                )

            # Check for empty or minimal results
            if len(result_str.strip()) < 10 and tool_name not in ["final_answer"]:
                validation["warnings"].append("Tool returned minimal or empty result")

            if validation["warnings"]:
                validation["valid"] = False

        except Exception as e:
            self.tool_logger.debug(f"Error in result validation: {e}")

        return validation

    def _generate_tool_signatures(self):
        """Generates tool signatures from the schemas of available tools."""
        self.tool_signatures = []

        for tool in self.tools:
            try:

                # Handle MCPToolWrapper specifically if needed
                if (
                    hasattr(tool, "tool_definition")
                    and isinstance(tool.tool_definition, dict)
                    and tool.tool_definition not in self.tool_signatures
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

    def _load_plugin_tools(self):
        """Load tools from plugin system and add them to the existing tools list"""
        try:
            from reactive_agents.plugins.plugin_manager import (
                get_plugin_manager,
                PluginType,
                ToolPlugin,
            )

            plugin_manager = get_plugin_manager()
            tool_plugins = plugin_manager.get_plugins_by_type(PluginType.TOOL)

            plugin_tool_count = 0
            for plugin_name, plugin in tool_plugins.items():
                try:
                    if isinstance(plugin, ToolPlugin):
                        tools = plugin.get_tools()
                        for tool_name, tool_callable in tools.items():
                            # Assume plugin tools are callable functions or Tool instances
                            if hasattr(tool_callable, "name") and hasattr(
                                tool_callable, "use"
                            ):
                                self.tools.append(tool_callable)  # type: ignore
                                plugin_tool_count += 1

                                if self.tool_logger:
                                    self.tool_logger.info(
                                        f"🔌 Loaded plugin tool: {tool_name} from {plugin_name}"
                                    )

                except Exception as e:
                    if self.tool_logger:
                        self.tool_logger.warning(
                            f"❌ Failed to load tools from plugin {plugin_name}: {e}"
                        )

            if plugin_tool_count > 0 and self.tool_logger:
                self.tool_logger.info(f"Loaded {plugin_tool_count} plugin tools")

        except ImportError:
            # Plugin system not available
            if self.tool_logger:
                self.tool_logger.debug("Plugin system not available for tool loading")
        except Exception as e:
            if self.tool_logger:
                self.tool_logger.warning(f"Error loading plugin tools: {e}")
