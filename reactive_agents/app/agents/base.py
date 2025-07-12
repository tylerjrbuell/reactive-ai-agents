from __future__ import annotations
import traceback
import asyncio
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Protocol, Callable
from abc import ABC, abstractmethod
import json
import re

# Import the AgentContext
from reactive_agents.core.context.agent_context import AgentContext

# Keep for type hinting if needed
from reactive_agents.core.types.agent_types import (
    AgentThinkChainResult,
    AgentThinkResult,
)
from reactive_agents.core.types.tool_types import ProcessedToolCall
from reactive_agents.utils.logging import Logger
from reactive_agents.providers.llm.base import BaseModelProvider
import time  # Keep time for metric tracking

# Import shared types from the new location
from reactive_agents.core.types.status_types import TaskStatus
from reactive_agents.core.types.session_types import AgentSession

if TYPE_CHECKING:
    from reactive_agents.core.engine.execution_engine import ExecutionEngine


class AgentLifecycleProtocol(Protocol):
    """Protocol for agent lifecycle management."""

    async def initialize(self) -> "Agent":
        """Initialize the agent."""
        ...

    async def close(self) -> None:
        """Close the agent and clean up resources."""
        ...

    async def run(self, initial_task: str, **kwargs) -> Dict[str, Any]:
        """Run the agent with a task."""
        ...


class AgentControlProtocol(Protocol):
    """Protocol for agent control operations."""

    async def pause(self) -> None:
        """Pause the agent execution."""
        ...

    async def resume(self) -> None:
        """Resume the agent execution."""
        ...

    async def stop(self) -> None:
        """Stop the agent execution."""
        ...

    async def terminate(self) -> None:
        """Terminate the agent execution."""
        ...


class Agent(ABC, AgentLifecycleProtocol, AgentControlProtocol):
    """
    Enhanced base class for AI agents with comprehensive lifecycle and control management.

    This base class provides:
    - Unified context management
    - Lifecycle protocols (initialize, run, close)
    - Control protocols (pause, resume, stop, terminate)
    - Event system integration
    - Tool management
    - Metrics and logging
    - Extensible architecture for future enhancements
    """

    context: AgentContext
    execution_engine: Optional["ExecutionEngine"] = None
    _closed: bool = False
    _initialized: bool = False
    config: Optional[Any] = None  # Optional config attribute for subclasses

    def __init__(self, context: AgentContext):
        """
        Initializes the Agent with a pre-configured AgentContext.

        Args:
            context: The AgentContext instance holding configuration, state, and managers.
        """
        self.context = context
        self._closed = False
        self._initialized = False

        # Set agent reference in context for engine to use
        self.context._agent = self  # type: ignore

        # Ensure logger is initialized before using it
        if not self.context.agent_logger:
            self.context._initialize_loggers()
        assert self.context.agent_logger is not None

        self.agent_logger.info(
            f"Base Agent '{self.context.agent_name}' initialized with context."
        )

    # --- Convenience properties to access context components ---
    @property
    def agent_logger(self) -> Logger:
        """Get the agent logger."""
        assert self.context.agent_logger is not None
        return self.context.agent_logger

    @property
    def tool_logger(self) -> Logger:
        """Get the tool logger."""
        assert self.context.tool_logger is not None
        return self.context.tool_logger

    @property
    def result_logger(self) -> Logger:
        """Get the result logger."""
        assert self.context.result_logger is not None
        return self.context.result_logger

    @property
    def model_provider(self) -> BaseModelProvider:
        """Get the model provider."""
        assert self.context.model_provider is not None
        return self.context.model_provider

    @property
    def is_initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self._initialized

    @property
    def is_closed(self) -> bool:
        """Check if the agent is closed."""
        return self._closed

    # --- End Convenience properties ---

    # --- Abstract methods that subclasses must implement ---
    @abstractmethod
    async def _execute_task(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a task. This is the core method that subclasses must implement.

        Args:
            task: The task to execute
            **kwargs: Additional execution parameters

        Returns:
            Execution results
        """
        pass

    # --- Core thinking methods ---
    async def _think(
        self, response_format: Optional[str] = None, **kwargs
    ) -> AgentThinkResult | None:
        """Directly calls the model provider for a simple completion."""
        start_time = time.time()
        try:
            self.agent_logger.info("Thinking (direct completion)...")

            # Add response format to kwargs if specified
            if response_format:
                kwargs["response_format"] = response_format

            result = await self.model_provider.get_completion(**kwargs)
            # Metric Tracking
            execution_time = time.time() - start_time
            if self.context.metrics_manager:
                self.context.metrics_manager.update_model_metrics(
                    {
                        "time": result.total_duration or execution_time,
                        "prompt_tokens": result.prompt_tokens or 0,
                        "completion_tokens": result.completion_tokens or 0,
                    }
                )

            if not result or not result.message:
                self.agent_logger.warning(
                    "Direct completion did not return a valid message structure."
                )
                return None
            result_json = {}
            message = result.message
            message_content: str = message.content or ""

            # Parse JSON from result.content if present and assign to result_json if available
            if message_content:
                result_json = extract_json_from_string(message_content)

            return AgentThinkResult(
                content=message_content,
                result_json=result_json,
                result=result.model_dump(),
            )
        except Exception as e:
            self.agent_logger.error(f"Direct Completion Error: {e}")
            return None

    def _should_use_tools(self, tool_calls: List[Dict[str, Any]]) -> bool:
        """
        Decide whether to allow tool use in the next think_chain call.

        This method implements intelligent tool usage policies:
        - Prevents tool use if final answer is set
        - Prevents tool use if plan is complete
        - Implements adaptive tool usage policies
        """
        # Final answer check: if final answer is set, do not allow tool use
        if self.context.session.final_answer is not None:
            self.agent_logger.debug("Final answer already set, not allowing tool use.")
            return False

        # Step-based plan completion check: if plan is complete, do not allow tool use
        if self.context.session.is_plan_complete():
            self.agent_logger.debug("Plan is complete, not allowing tool use.")
            return False

        # Existing tool_use_policy logic
        policy = getattr(self.context, "tool_use_policy", "adaptive")
        session = self.context.session

        if policy == "always":
            return True
        if policy == "never":
            return False
        if policy == "required_only":
            if session.min_required_tools:
                tools_completed, _ = self.context.has_completed_required_tools()
                return not tools_completed
            return False

        # Adaptive/heuristic
        # 1. If required tools left, use tools
        if session.min_required_tools:
            tools_completed, _ = self.context.has_completed_required_tools()
            if not tools_completed:
                return True
        # 2. If just used a tool, consider pausing for reflection
        if tool_calls and tool_calls[-1].get("name") in session.successful_tools:
            return False
        # 3. If too many tool calls in a row, force reflection
        max_calls = getattr(self.context, "tool_use_max_consecutive_calls", 3)
        if len(tool_calls) > max_calls:
            return False
        # 4. Otherwise, allow tool use
        return True

    async def _think_chain(
        self,
        remember_messages: bool = True,
        use_tools: bool = True,
        **kwargs,
    ) -> AgentThinkChainResult | None:
        """
        Performs a chat completion with tool support and message management.

        This is the core thinking method that handles:
        - Tool signature management
        - Message history
        - Tool usage policies
        - Metrics tracking
        """
        start_time = time.time()
        try:
            # Use tool signatures from the ToolManager via context
            tool_signatures = self.context.get_tool_signatures()
            use_tools = (
                self.context.tool_use_enabled and bool(tool_signatures) and use_tools
            )

            # Default to context messages
            kwargs.setdefault("messages", self.context.session.messages)
            self.agent_logger.debug(
                f"Thinking (chat completion, tool_use={use_tools})..."
            )

            if self.context.model_provider_options:
                self.agent_logger.debug(
                    f"Using model provider options: {self.context.model_provider_options}"
                )
                kwargs["options"] = self.context.model_provider_options

            result = await self.model_provider.get_chat_completion(
                tools=tool_signatures if use_tools else [],
                tool_use_required=use_tools,
                **kwargs,
            )
            self.agent_logger.debug(f"Chat completion result: {result}")

            # Metric Tracking
            execution_time = time.time() - start_time
            if self.context.metrics_manager:
                self.context.metrics_manager.update_model_metrics(
                    {
                        "time": result.total_duration or execution_time,
                        "prompt_tokens": result.prompt_tokens or 0,
                        "completion_tokens": result.completion_tokens or 0,
                    }
                )

            if not result or not result.message:
                self.agent_logger.warning(
                    "Chat completion did not return a valid message structure."
                )
                return None

            message = result.message
            message_content: str = message.content or ""
            tool_calls: List[Dict[str, Any]] = message.tool_calls or []

            result_json = {}
            if message_content:
                result_json = extract_json_from_string(message_content)
            if message_content.strip() and remember_messages:
                # Add assistant response to context's message history
                self.context.session.messages.append(
                    {"role": "assistant", "content": message_content}
                )
                self.agent_logger.debug(
                    f"Added assistant message to context: {message_content[:100]}..."
                )

            # Process tool calls if any
            if tool_calls:
                processed_calls = await self._process_tool_calls(tool_calls)
                return AgentThinkChainResult(
                    content=message_content,
                    result_json=result_json,
                    tool_calls=processed_calls or [],
                    result=result.model_dump(),
                )

            return AgentThinkChainResult(
                content=message_content,
                tool_calls=[],
                result_json=result_json,
                result=result.model_dump(),
            )

        except Exception as e:
            self.agent_logger.error(f"Think chain error: {e}")
            return None

    async def _process_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> list[ProcessedToolCall] | None:
        """
        Process tool calls and execute them.

        This method handles:
        - Tool execution
        - Result processing
        - Error handling
        - Message history updates
        """
        if not tool_calls:
            return []

        processed_calls: list[ProcessedToolCall] = []
        for tool_call in tool_calls:
            try:
                # Handle both dict and ToolCall object structures
                if hasattr(tool_call, "function") and not isinstance(tool_call, dict):
                    # ToolCall object with function attribute
                    function_obj = getattr(tool_call, "function")
                    tool_name = getattr(function_obj, "name", "unknown")
                    tool_args = getattr(function_obj, "arguments", {})
                elif isinstance(tool_call, dict):
                    # Dict structure - check for nested function
                    if "function" in tool_call:
                        tool_name = tool_call["function"].get("name", "unknown")
                        tool_args = tool_call["function"].get("arguments", {})
                    else:
                        # Legacy dict structure
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("arguments", {})
                else:
                    tool_name = "unknown"
                    tool_args = {}

                self.agent_logger.info(f"Executing tool: {tool_name}")

                # Extract tool_call_id safely BEFORE reassigning tool_call
                tool_call_id = None
                if hasattr(tool_call, "id") and not isinstance(tool_call, dict):
                    tool_call_id = getattr(tool_call, "id", None)
                elif isinstance(tool_call, dict):
                    tool_call_id = tool_call.get("id")

                # Execute the tool using tool manager
                if self.context.tool_manager:
                    tool_call_dict = {
                        "function": {"name": tool_name, "arguments": tool_args}
                    }
                    result = await self.context.tool_manager.use_tool(tool_call_dict)
                else:
                    result = f"Tool {tool_name} not available (no tool manager)"

                # Add tool result to message history

                self.context.session.messages.append(
                    {
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                    }
                )

                processed_calls.append(
                    ProcessedToolCall(
                        name=tool_name,
                        arguments=tool_args,
                        result=str(result),
                        success=True,
                    )
                )

            except Exception as e:
                # Extract tool name safely for error reporting
                error_tool_name = "unknown"
                error_tool_args = {}

                if hasattr(tool_call, "function") and not isinstance(tool_call, dict):
                    function_obj = getattr(tool_call, "function")
                    error_tool_name = getattr(function_obj, "name", "unknown")
                    error_tool_args = getattr(function_obj, "arguments", {})
                elif isinstance(tool_call, dict):
                    if "function" in tool_call:
                        error_tool_name = tool_call["function"].get("name", "unknown")
                        error_tool_args = tool_call["function"].get("arguments", {})
                    else:
                        error_tool_name = tool_call.get("name", "unknown")
                        error_tool_args = tool_call.get("arguments", {})

                self.agent_logger.error(
                    f"Tool execution error for {error_tool_name}: {e}"
                )
                processed_calls.append(
                    ProcessedToolCall(
                        name=error_tool_name,
                        arguments=error_tool_args,
                        result=str(e),
                        success=False,
                    )
                )

        return processed_calls

    # --- Lifecycle methods ---
    async def initialize(self) -> "Agent":
        """
        Initialize the agent and its components.

        This method should be called before using the agent.
        Subclasses can override this to add custom initialization logic.
        """
        if self._initialized:
            self.agent_logger.warning("Agent already initialized")
            return self

        try:
            self.agent_logger.info(f"Initializing {self.context.agent_name}...")

            # Initialize MCP if configured
            if (
                self.config
                and hasattr(self.config, "mcp_config")
                and (
                    self.config.mcp_config
                    or getattr(self.config, "mcp_server_filter", None)
                )
            ):
                from reactive_agents.config.mcp_config import MCPConfig
                from reactive_agents.providers.external.client import MCPClient

                self.context.mcp_config = (
                    MCPConfig.model_validate(self.config.mcp_config, strict=False)
                    if self.config.mcp_config
                    else None
                )
                self.context.mcp_client = await MCPClient(
                    server_config=self.context.mcp_config,
                    server_filter=getattr(self.config, "mcp_server_filter", None),
                ).initialize()

            # Initialize tool manager
            if self.context.tool_manager:
                await self.context.tool_manager.initialize()

            self._initialized = True
            self.agent_logger.info(
                f"{self.context.agent_name} initialized successfully"
            )
            return self

        except Exception as e:
            self.agent_logger.error(
                f"Error initializing {self.context.agent_name}: {e}"
            )
            raise e

    async def close(self) -> None:
        """
        Close the agent and clean up resources.

        This method should be called when done with the agent.
        """
        if self._closed:
            return

        try:
            self.agent_logger.info(f"Closing {self.context.agent_name}...")

            # Close context
            await self.context.close()

            # Close MCP client if exists
            if self.context.mcp_client is not None:
                self.agent_logger.info(
                    f"Closing MCPClient for {self.context.agent_name}..."
                )
                await self.context.mcp_client.close()
                self.agent_logger.info(
                    f"{self.context.agent_name} MCPClient closed successfully."
                )

            self._closed = True
            self.agent_logger.info(f"{self.context.agent_name} closed successfully.")

        except Exception as e:
            self.agent_logger.error(f"Error closing {self.context.agent_name}: {e}")
            raise e

    # --- Control methods (default implementations) ---
    async def pause(self) -> None:
        """Pause the agent execution. Override in subclasses for specific behavior."""
        self.agent_logger.info(f"{self.context.agent_name} pause requested")
        if hasattr(self, "execution_engine") and self.execution_engine:
            await self.execution_engine.pause()

    async def resume(self) -> None:
        """Resume the agent execution. Override in subclasses for specific behavior."""
        self.agent_logger.info(f"{self.context.agent_name} resume requested")
        if hasattr(self, "execution_engine") and self.execution_engine:
            await self.execution_engine.resume()

    async def stop(self) -> None:
        """Stop the agent execution. Override in subclasses for specific behavior."""
        self.agent_logger.info(f"{self.context.agent_name} stop requested")
        if hasattr(self, "execution_engine") and self.execution_engine:
            await self.execution_engine.stop()

    async def terminate(self) -> None:
        """Terminate the agent execution. Override in subclasses for specific behavior."""
        self.agent_logger.info(f"{self.context.agent_name} terminate requested")
        if hasattr(self, "execution_engine") and self.execution_engine:
            await self.execution_engine.terminate()

    # --- Context manager support ---
    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    # --- Core run method ---
    async def run(self, initial_task: str, **kwargs) -> Dict[str, Any]:
        """
        Run the agent with a task.

        This is the main entry point for agent execution.
        Subclasses should implement _execute_task for specific behavior.

        Args:
            initial_task: The task to execute
            **kwargs: Additional parameters for execution

        Returns:
            Execution results with comprehensive metrics
        """
        if not self._initialized:
            await self.initialize()

        if self._closed:
            raise RuntimeError("Cannot run a closed agent")

        # Initialize session tracking
        if self.context.session:
            self.context.session.initial_task = initial_task
            self.context.session.current_task = initial_task
            self.context.session.start_time = time.time()

        try:
            self.agent_logger.info(
                f"🚀 {self.context.agent_name} starting task: {initial_task[:100]}..."
            )
            result = await self._execute_task(initial_task, **kwargs)
            self.agent_logger.info(f"✅ {self.context.agent_name} completed task")

            # Finalize metrics and prepare comprehensive result
            return self._prepare_final_result_with_metrics(
                result, initial_task, success=True
            )

        except Exception as e:
            self.agent_logger.error(
                f"❌ {self.context.agent_name} execution failed: {e}"
            )

            # Prepare error result with metrics
            error_result = {
                "status": "error",
                "error": str(e),
                "final_answer": None,
                "completion_score": 0.0,
                "iterations": 0,
            }
            return self._prepare_final_result_with_metrics(
                error_result, initial_task, success=False
            )

    def _prepare_final_result_with_metrics(
        self, base_result: Dict[str, Any], initial_task: str, success: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare the final result with comprehensive metrics.

        Args:
            base_result: The base result from _execute_task
            initial_task: The initial task that was executed
            success: Whether the execution was successful

        Returns:
            Enhanced result with metrics
        """
        # Finalize metrics if available
        if self.context.metrics_manager:
            self.context.metrics_manager.finalize_run_metrics()
            metrics_data = self.context.metrics_manager.get_metrics()
        else:
            metrics_data = {}

        # Calculate execution time
        start_time = (
            getattr(self.context.session, "start_time", None)
            if self.context.session
            else None
        )
        end_time = time.time()
        execution_time = end_time - start_time if start_time else 0.0

        # Get session data
        session_data = {}
        if self.context.session:
            session_data = {
                "session_id": getattr(self.context.session, "session_id", "unknown"),
                "initial_task": self.context.session.initial_task,
                "current_task": self.context.session.current_task,
                "iterations": self.context.session.iterations,
                "final_answer": self.context.session.final_answer,
                "completion_score": self.context.session.completion_score,
                "task_status": (
                    self.context.session.task_status.value
                    if self.context.session.task_status
                    else "unknown"
                ),
                "successful_tools": (
                    list(self.context.session.successful_tools)
                    if self.context.session.successful_tools
                    else []
                ),
                "start_time": self.context.session.start_time,
                "end_time": end_time,
            }

        # Prepare comprehensive result
        enhanced_result = {
            # Core execution results
            "status": base_result.get("status", "complete" if success else "error"),
            "final_answer": base_result.get(
                "final_answer", session_data.get("final_answer")
            ),
            "completion_score": base_result.get(
                "completion_score",
                session_data.get("completion_score", 1.0 if success else 0.0),
            ),
            "iterations": base_result.get(
                "iterations", session_data.get("iterations", 0)
            ),
            "result": base_result.get("result"),
            "error": base_result.get("error"),
            # Execution metadata
            "execution_time": execution_time,
            "agent_name": self.context.agent_name,
            "model_provider": self.context.provider_model_name,
            # Comprehensive metrics
            "metrics": {
                "execution_time": execution_time,
                "total_time": metrics_data.get("total_time", execution_time),
                "start_time": metrics_data.get("start_time", start_time),
                "end_time": metrics_data.get("end_time", end_time),
                "status": metrics_data.get(
                    "status", "complete" if success else "error"
                ),
                "iterations": metrics_data.get(
                    "iterations", session_data.get("iterations", 0)
                ),
                "tool_calls": metrics_data.get("tool_calls", 0),
                "tool_errors": metrics_data.get("tool_errors", 0),
                "model_calls": metrics_data.get("model_calls", 0),
                "tokens": metrics_data.get(
                    "tokens",
                    {
                        "prompt": 0,
                        "completion": 0,
                        "total": 0,
                    },
                ),
                "cache": metrics_data.get(
                    "cache",
                    {
                        "hits": 0,
                        "misses": 0,
                        "ratio": 0.0,
                    },
                ),
                "latency": metrics_data.get(
                    "latency",
                    {
                        "tool_time": 0,
                        "model_time": 0,
                    },
                ),
                "tools": metrics_data.get("tools", {}),
            },
            # Task information
            "task": {
                "initial": session_data.get("initial_task", initial_task),
                "current": session_data.get("current_task", initial_task),
                "successful_tools": session_data.get("successful_tools", []),
            },
            # Session context
            "session": {
                "id": session_data.get("session_id", "unknown"),
                "start_time": session_data.get("start_time", start_time),
                "end_time": session_data.get("end_time", end_time),
                "tool_use_enabled": self.context.tool_use_enabled,
                "collect_metrics_enabled": self.context.collect_metrics_enabled,
                "max_iterations": self.context.max_iterations,
            },
        }

        # Include any additional fields from base_result that we haven't covered
        for key, value in base_result.items():
            if key not in enhanced_result:
                enhanced_result[key] = value

        return enhanced_result


def extract_json_from_string(s: str):
    """
    Try to extract and parse the first valid JSON object or array from a string.
    Returns the parsed object (dict/list) or {} if not found/invalid.
    """
    s = s.strip()
    # Try parsing the whole string first
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to find the first { ... } or [ ... ] block (greedy)
    import re

    match = re.search(r"({.*})|(\[.*\])", s, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except Exception:
            return {}
    return {}
