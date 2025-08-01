from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING
import time
import json
import asyncio
import inspect
from pydantic import BaseModel, Field
from reactive_agents.core.types.confirmation_types import ConfirmationCallbackProtocol
from reactive_agents.core.types.event_types import AgentStateEvent
from reactive_agents.core.tools.base import Tool
from reactive_agents.core.tools.abstractions import (
    MCPToolWrapper,
    ToolProtocol,
    ToolResult,
)
from reactive_agents.utils.logging import Logger
from reactive_agents.providers.llm.base import BaseModelProvider

# Import our new SOLID components
from reactive_agents.core.tools.default import FinalAnswerTool
from reactive_agents.core.tools.tool_guard import ToolGuard
from reactive_agents.core.tools.tool_cache import ToolCache
from reactive_agents.core.tools.tool_confirmation import ToolConfirmation
from reactive_agents.core.tools.tool_validator import ToolValidator
from reactive_agents.core.tools.tool_executor import ToolExecutor

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class ToolManager(BaseModel):
    """Orchestrates tool discovery, execution, caching, and validation using SOLID components."""

    context: Optional["AgentContext"] = Field(
        default=None, exclude=True
    )  # Reference back to the main context

    # State
    tools: List[Tool] = Field(default_factory=list)
    tool_signatures: List[Dict[str, Any]] = Field(default_factory=list)
    tool_history: List[Dict[str, Any]] = Field(default_factory=list)

    # SOLID Components (dependency injection)
    guard: ToolGuard = Field(default_factory=ToolGuard, exclude=True)
    cache: ToolCache = Field(default_factory=ToolCache, exclude=True)
    confirmation: Optional[ToolConfirmation] = Field(default=None, exclude=True)
    validator: Optional[ToolValidator] = Field(default=None, exclude=True)
    executor: Optional[ToolExecutor] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        # Initialize cache first since it's needed in __init__
        context = data.get("context")
        cache = ToolCache(
            enabled=getattr(context, "enable_caching", True) if context else True,
            ttl=getattr(context, "cache_ttl", 3600) if context else 3600,
        )
        data["cache"] = cache

        # Initialize other components
        if "context" in data:
            context = data["context"]
            data.update(
                {
                    "confirmation": ToolConfirmation(
                        context=context,
                        confirmation_callback=getattr(
                            context, "confirmation_callback", None
                        ),
                        confirmation_config=getattr(
                            context, "confirmation_config", None
                        ),
                    ),
                    "validator": ToolValidator(context),
                    "executor": ToolExecutor(context),
                }
            )

        super().__init__(**data)

        # Initialize guard with default policies
        self.guard.add_default_guards()

    @property
    def agent_logger(self) -> Optional[Logger]:
        return self.context.agent_logger if self.context else None

    @property
    def tool_logger(self) -> Optional[Logger]:
        return self.context.tool_logger if self.context else None

    @property
    def model_provider(self) -> Optional[BaseModelProvider]:
        return self.context.model_provider if self.context else None

    # Delegate cache properties for backward compatibility
    @property
    def cache_hits(self) -> int:
        return self.cache.hits

    @property
    def cache_misses(self) -> int:
        return self.cache.misses

    @property
    def enable_caching(self) -> bool:
        return self.cache.enabled

    @property
    def cache_ttl(self) -> int:
        return self.cache.ttl

    async def initialize(self):
        """Initialize the tool manager."""
        await self._initialize_tools()

    async def _initialize_tools(self):
        """Populates tools and signatures from MCP client and/or local list."""
        # Initialize tools list to accumulate both MCP and custom tools
        self.tools = []
        self.tool_signatures = []
        # Load MCP tools if available
        if self.context and self.context.mcp_client:
            # Ensure MCP client tools are loaded (might need await if not already done)
            # Assuming mcp_client.tools and mcp_client.tool_signatures are populated
            mcp_tools = getattr(self.context.mcp_client, "tools", [])
            mcp_wrapped_tools = [
                MCPToolWrapper(t, self.context.mcp_client) for t in mcp_tools
            ]
            self.tools.extend(mcp_wrapped_tools)
            ## Set
            mcp_signatures = getattr(self.context.mcp_client, "tool_signatures", [])
            self.tool_signatures.extend(mcp_signatures)
            if self.agent_logger:
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

        # Load custom tools if available
        if self.context and self.context.tools:
            # Add locally provided tools
            self.tools.extend(self.context.tools)
            custom_signatures = [
                tool.tool_definition
                for tool in self.context.tools
                if hasattr(tool, "tool_definition")
            ]
            self.tool_signatures.extend(custom_signatures)

            if self.agent_logger:
                self.agent_logger.info(
                    f"Initialized {len(self.context.tools)} custom tools."
                )

        # Check if we have any tools at all
        if not self.tools and self.agent_logger:
            self.agent_logger.warning("No MCP client or custom tools provided")

        # --- Inject final_answer tool if missing ---
        has_final_answer = any(tool.name == "final_answer" for tool in self.tools)
        if not has_final_answer and self.context:
            if self.tool_logger:
                self.tool_logger.info("Injecting internal 'final_answer' tool.")
            internal_final_answer_tool = FinalAnswerTool(context=self.context)
            self.tools.append(internal_final_answer_tool)

        self.register_tools()

    def get_tool(self, tool_name: str) -> Optional[ToolProtocol]:
        """Finds a tool by name."""
        for tool in self.tools:
            if hasattr(tool, "name") and tool.name == tool_name:
                return tool
        return None

    async def use_tool(self, tool_call: Dict[str, Any]) -> Union[str, List[str], None]:
        """Executes a tool using SOLID components for validation, caching, confirmation, and execution."""
        try:
            # Parse tool call
            if self.executor:
                tool_name, params = self.executor.parse_tool_arguments(tool_call)
            else:
                # Fallback parsing
                tool_name = tool_call.get("function", {}).get("name")
                if not tool_name:
                    raise ValueError("Tool call missing function name")
                arguments_raw = tool_call.get("function", {}).get("arguments", {})
                params = arguments_raw if isinstance(arguments_raw, dict) else {}
        except ValueError as e:
            return f"Error: {e}"

        # Check guard policies
        if not self.guard.can_use(tool_name):
            return f"Error: Tool '{tool_name}' usage is rate-limited. Please wait before using it again."

        # Check guard confirmation requirements
        if self.guard.needs_confirmation(tool_name) and self.confirmation:
            confirmed, feedback = await self.confirmation.request_confirmation(
                tool_name, params, ""
            )
            if not confirmed:
                cancel_msg = (
                    f"Tool '{tool_name}' usage was cancelled by user confirmation."
                )
                if feedback:
                    self.confirmation.inject_user_feedback(tool_name, params, feedback)
                    cancel_msg += f" - Feedback: {feedback}"
                self._add_to_history(tool_name, params, cancel_msg, cancelled=True)
                return cancel_msg

        # Execute the tool
        result = await self._actually_call_tool(tool_call)

        # Record usage in guard
        self.guard.record_use(tool_name)

        return result

    async def _actually_call_tool(
        self, tool_call: Dict[str, Any]
    ) -> Union[str, List[str], None]:
        """Executes a tool using the new component-based architecture."""
        try:
            # Parse tool call using executor
            if self.executor:
                tool_name, params = self.executor.parse_tool_arguments(tool_call)
            else:
                # Fallback parsing
                tool_name = tool_call.get("function", {}).get("name")
                if not tool_name:
                    raise ValueError("Tool call missing function name")
                arguments_raw = tool_call.get("function", {}).get("arguments", {})
                params = arguments_raw if isinstance(arguments_raw, dict) else {}
        except ValueError as e:
            error_msg = f"Error: {e}"
            if self.tool_logger:
                self.tool_logger.error(error_msg)
            return error_msg

        tool = self.get_tool(tool_name)
        if not tool:
            available_tools = [t.name for t in self.tools]
            error_message = f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
            if self.tool_logger:
                self.tool_logger.error(error_message)
            # Emit TOOL_FAILED event
            if self.context:
                self.context.emit_event(
                    AgentStateEvent.TOOL_FAILED,
                    {
                        "tool_name": tool_name,
                        "parameters": params,
                        "error": error_message,
                    },
                )
            self._add_to_history(
                tool_name, params, error_message, error=True, execution_time=0.0
            )
            return error_message

        # Emit TOOL_CALLED event before execution
        if self.context:
            self.context.emit_event(
                AgentStateEvent.TOOL_CALLED,
                {"tool_name": tool_name, "parameters": params},
            )

        # Add reasoning to the main context log
        if self.executor:
            self.executor.add_reasoning_to_context(params)

        # Check cache first
        cache_key = self.cache.generate_cache_key(tool_name, params)
        if cache_key:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                if self.tool_logger:
                    self.tool_logger.info(f"Cache hit for tool: {tool_name}")
                summary = await self._generate_and_log_summary(
                    tool_name, params, cached_result
                )
                self._add_to_history(
                    tool_name, params, cached_result, summary, cached=True
                )
                # Handle final answer from cache
                if tool_name == "final_answer" and self.context:
                    self.context.session.final_answer = str(cached_result)
                    if self.tool_logger:
                        self.tool_logger.info(
                            f"🔧 ToolManager: final_answer from cache, set session.final_answer = {self.context.session.final_answer[:100] if self.context.session.final_answer else 'None'}..."
                        )
                # Emit events for cached result
                self._emit_tool_completion_events(tool_name, params, cached_result, 0.0)
                return cached_result

        # Check if tool requires confirmation using new confirmation component
        description = self._get_tool_description(tool)
        requires_confirmation = (
            self.confirmation
            and self.confirmation.tool_requires_confirmation(tool_name, description)
        )

        if requires_confirmation and self.confirmation:
            confirmed, feedback = await self.confirmation.request_confirmation(
                tool_name, params, description
            )
            if not confirmed:
                cancel_msg = f"Action cancelled by user: {tool_name}"
                if feedback and self.confirmation:
                    self.confirmation.inject_user_feedback(tool_name, params, feedback)
                    cancel_msg += f" - Feedback: {feedback}"

                if self.tool_logger:
                    self.tool_logger.info(cancel_msg)
                self._add_to_history(tool_name, params, cancel_msg, cancelled=True)
                if self.context:
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
            if feedback and self.confirmation:
                self.confirmation.inject_user_feedback(tool_name, params, feedback)

        # Execute the tool using the executor component
        if self.executor:
            result_list = await self.executor.execute_tool(tool, tool_name, params)
        else:
            return "Error: Tool executor not initialized"

        if isinstance(result_list, str) and result_list.startswith("Error"):
            # Tool execution failed
            self._add_to_history(
                tool_name, params, result_list, error=True, execution_time=0.0
            )
            return result_list

        # Generate summary and validate result
        summary = await self._generate_and_log_summary(tool_name, params, result_list)

        # Validate tool result
        if self.validator:
            validation = self.validator.validate_tool_result_usage(
                tool_name, params, result_list
            )
        else:
            validation = {"valid": True, "warnings": [], "suggestions": []}
        if not validation["valid"]:
            if self.tool_logger:
                self.tool_logger.warning(
                    f"Tool result validation failed for {tool_name}: {validation['warnings']}"
                )
                if validation["suggestions"]:
                    self.tool_logger.info(f"Suggestions: {validation['suggestions']}")

        # Store search data if applicable
        if self.validator:
            self.validator.store_search_data(tool_name, params, result_list)

        # Add to history
        self._add_to_history(
            tool_name, params, result_list, summary, execution_time=0.1
        )

        # Cache successful results
        if cache_key:
            self.cache.put(cache_key, result_list, execution_time=0.1)

        return result_list

    def _get_tool_description(self, tool: ToolProtocol) -> str:
        """Get the description of a tool for confirmation purposes."""
        if hasattr(tool, "tool_definition") and tool.tool_definition:
            func_def = tool.tool_definition.get("function", {})
            return func_def.get("description", "")
        return ""

    def _emit_tool_completion_events(
        self, tool_name: str, params: Dict[str, Any], result: Any, execution_time: float
    ) -> None:
        """Emit tool completion events."""
        if self.context:
            self.context.emit_event(
                AgentStateEvent.TOOL_COMPLETED,
                {
                    "tool_name": tool_name,
                    "parameters": params,
                    "result": result,
                    "execution_time": execution_time,
                },
            )
            # Emit FINAL_ANSWER_SET if tool is final_answer
            if tool_name == "final_answer":
                self.context.emit_event(
                    AgentStateEvent.FINAL_ANSWER_SET,
                    {
                        "tool_name": tool_name,
                        "answer": result,
                        "parameters": params,
                    },
                )

    async def _generate_and_log_summary(
        self, tool_name: str, params: Dict[str, Any], result: Any
    ) -> str:
        """Generate summary using the executor component."""
        if self.executor:
            return await self.executor._generate_tool_summary(tool_name, params, result)
        else:
            return f"Executed tool '{tool_name}'"

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
        if not error and not cancelled and self.context:
            self.context.session.successful_tools.add(tool_name)

        self.tool_history.append(entry)

        # Optionally update metrics (delegated)
        if (
            self.context
            and self.context.collect_metrics_enabled
            and self.context.metrics_manager
        ):
            try:
                self.context.metrics_manager.update_tool_metrics(entry)
                # Emit METRICS_UPDATED event after metrics update
                self.context.emit_event(
                    AgentStateEvent.METRICS_UPDATED,
                    {"metrics": self.context.metrics_manager.get_metrics()},
                )
            except Exception as metrics_error:
                if self.tool_logger:
                    self.tool_logger.debug(
                        f"Metrics update error (non-critical): {metrics_error}"
                    )

    def _generate_tool_signatures(self):
        """Generates tool signatures from the schemas of available tools."""
        self.tool_signatures = []
        print("Generating tool signatures...")
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
                    if self.tool_logger:
                        self.tool_logger.warning(
                            f"Tool '{tool.name}' is missing a valid schema/tool_definition attribute for signature generation."
                        )
            except Exception as e:
                if self.tool_logger:
                    self.tool_logger.error(
                        f"Error generating signature for tool '{getattr(tool, 'name', 'unknown')}': {e}"
                    )
        if self.tool_logger:
            self.tool_logger.info(
                f"Initialized with {len(self.tools)} total tool(s): {', '.join([tool.name for tool in self.tools])}"
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

    def register_tools(self):
        """
        Register a list of tools, deduplicating by name, and update signatures.
        """
        seen = set(getattr(t, "name", str(id(t))) for t in self.tools)
        if self.context and self.context.tools:
            for tool in self.context.tools:
                name = getattr(tool, "name", str(id(tool)))
                if name not in seen:
                    self.tools.append(tool)
                    seen.add(name)
        if hasattr(self, "_generate_tool_signatures"):
            self._generate_tool_signatures()
