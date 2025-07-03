from __future__ import annotations
import asyncio
import time
import traceback
from typing import Dict, Any, Optional, TYPE_CHECKING, List
from reactive_agents.core.types.status_types import TaskStatus
from reactive_agents.core.types.event_types import AgentStateEvent
from reactive_agents.core.engine.execution_engine import AgentExecutionEngine

if TYPE_CHECKING:
    from reactive_agents.app.agents.base import Agent
    from reactive_agents.core.reasoning.strategies.strategy_manager import (
        StrategyManager,
    )
    from reactive_agents.core.reasoning.task_classifier import TaskClassifier


class ReactiveExecutionEngine(AgentExecutionEngine):
    """
    Reactive execution engine that implements truly reactive behavior by:
    1. Classifying tasks at runtime
    2. Selecting optimal reasoning strategies dynamically
    3. Planning every iteration instead of static upfront planning
    4. Switching strategies based on performance
    """

    def __init__(self, agent: "Agent"):
        """Initialize the reactive execution engine."""
        super().__init__(agent)
        self.strategy_manager: "StrategyManager"
        self.task_classifier: "TaskClassifier"
        self._initialize_components()

    def _initialize_components(self):
        """Initialize strategy manager and task classifier."""
        # Import here to avoid circular imports
        from reactive_agents.core.reasoning.strategies.strategy_manager import (
            StrategyManager,
        )
        from reactive_agents.core.reasoning.task_classifier import TaskClassifier

        # Initialize strategy manager
        self.strategy_manager = StrategyManager(self.context)

        # Initialize task classifier
        self.task_classifier = TaskClassifier(self.context)

        if self.agent.agent_logger:
            self.agent.agent_logger.info("✅ Reactive execution engine initialized")

        # Control state
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Initially not paused
        self._terminate_requested = False
        self._stop_requested = False

    async def execute(
        self,
        initial_task: str,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Execute the agent's task using reactive reasoning strategies.

        Args:
            initial_task: The task description to execute
            cancellation_event: Optional event to signal cancellation

        Returns:
            Dictionary containing execution results and status
        """
        if self.agent.agent_logger:
            self.agent.agent_logger.info(
                f"🚀 Starting reactive execution for task: {initial_task[:100]}..."
            )

        # Initialize session
        self._initialize_session(initial_task)

        try:
            # Phase 1: Classify the task
            task_classification = await self.task_classifier.classify_task(
                initial_task, self.context.session.messages
            )

            if self.agent.agent_logger:
                self.agent.agent_logger.info(
                    f"📋 Task classified as: {task_classification.task_type.value} "
                    f"(confidence: {task_classification.confidence:.2f}, "
                    f"complexity: {task_classification.complexity_score:.2f})"
                )

            # Phase 2: Select initial reasoning strategy
            # Only use dynamic strategy selection if enabled, otherwise use configured strategy
            if self.context.enable_dynamic_strategy_switching:
                self.strategy_manager.select_initial_strategy(task_classification)
            else:
                # Use explicitly configured strategy and log the choice
                configured_strategy_str = getattr(
                    self.context, "reasoning_strategy", None
                )
                if configured_strategy_str:
                    # Convert string to enum
                    from reactive_agents.core.types.reasoning_types import (
                        ReasoningStrategies,
                    )

                    try:
                        configured_strategy = ReasoningStrategies(
                            configured_strategy_str
                        )
                        self.strategy_manager.current_strategy = configured_strategy
                        self.strategy_manager.reasoning_context.current_strategy = (
                            configured_strategy
                        )
                        if self.agent.agent_logger:
                            self.agent.agent_logger.info(
                                f"🎯 Using configured strategy: {configured_strategy.value} "
                                f"(dynamic switching disabled)"
                            )
                    except ValueError:
                        if self.agent.agent_logger:
                            self.agent.agent_logger.warning(
                                f"Invalid configured strategy: {configured_strategy_str}, using task classification"
                            )
                        # Fallback to task classification if invalid strategy
                        self.strategy_manager.select_initial_strategy(
                            task_classification
                        )
                else:
                    # Fallback to default strategy if no configuration found
                    self.strategy_manager.select_initial_strategy(task_classification)

            # Phase 3: Execute main reasoning loop
            result = await self._execute_reasoning_loop(
                initial_task, cancellation_event
            )

            # Phase 4: Finalize and return results
            return await self._prepare_final_result(result)

        except Exception as e:
            if self.agent.agent_logger:
                self.agent.agent_logger.error(f"Execution failed: {e}")
            return await self._handle_execution_error(e)

    def _initialize_session(self, initial_task: str):
        """Initialize the session for a new task."""
        if not self.context.session:
            from reactive_agents.core.types.session_types import AgentSession

            self.context.session = AgentSession(
                initial_task=initial_task,
                current_task=initial_task,
                start_time=time.time(),
                task_status=TaskStatus.INITIALIZED,
                reasoning_log=[],
                task_progress=[],
                task_nudges=[],
                successful_tools=set(),
                metrics={},
                completion_score=0.0,
                tool_usage_score=0.0,
                progress_score=0.0,
                answer_quality_score=0.0,
                llm_evaluation_score=0.0,
                instruction_adherence_score=0.0,
            )

        # Reset session state
        self.context.session.iterations = 0
        self.context.session.final_answer = None
        self.context.session.current_task = initial_task
        self.context.session.initial_task = initial_task
        self.context.session.task_status = TaskStatus.RUNNING

        # Add the user's task to the session messages
        self.context.session.messages.append({"role": "user", "content": initial_task})

        # Reset strategy manager
        self.strategy_manager.reset()

        # Emit session start event
        self.context.emit_event(
            AgentStateEvent.SESSION_STARTED,
            {
                "initial_task": initial_task,
                "session_id": getattr(self.context.session, "session_id", "unknown"),
            },
        )

    async def _execute_reasoning_loop(
        self, task: str, cancellation_event: Optional[asyncio.Event] = None
    ) -> Dict[str, Any]:
        """Execute the main reasoning loop with dynamic strategy switching."""

        iteration_results = []
        max_iterations = self.context.max_iterations or 20

        while await self._should_continue():
            # Check for cancellation
            if cancellation_event and cancellation_event.is_set():
                if self.agent.agent_logger:
                    self.agent.agent_logger.info(
                        "🛑 Execution cancelled by external event"
                    )
                break

            # Check control signals
            if self._handle_control_signals():
                break

            self.context.session.iterations += 1

            if self.agent.agent_logger:
                self.agent.agent_logger.info(
                    f"🔄 Iteration {self.context.session.iterations}/{max_iterations}"
                )

            # Emit iteration started event
            self.context.emit_event(
                AgentStateEvent.ITERATION_STARTED,
                {
                    "iteration": self.context.session.iterations,
                    "max_iterations": max_iterations,
                },
            )

            try:
                # Execute one iteration using strategy manager
                iteration_result = await self.strategy_manager.execute_iteration(task)
                iteration_results.append(iteration_result)

                # Check if we should stop
                if not iteration_result.get("should_continue", True):
                    if self.agent.agent_logger:
                        self.agent.agent_logger.info("🏁 Strategy indicates completion")
                    break

                # Check for final answer
                if self.context.session.final_answer:
                    if self.agent.agent_logger:
                        self.agent.agent_logger.info("✅ Final answer provided")
                    break

                # Emit iteration completed event
                self.context.emit_event(
                    AgentStateEvent.ITERATION_COMPLETED,
                    {
                        "iteration": self.context.session.iterations,
                        "result": iteration_result,
                    },
                )

                # Context management
                await self.context.manage_context()

            except Exception as e:
                if self.agent.agent_logger:
                    self.agent.agent_logger.error(
                        f"Iteration {self.context.session.iterations} failed: {e}"
                    )

                iteration_results.append(
                    {"error": str(e), "iteration": self.context.session.iterations}
                )

                # Continue with error handling in strategy manager
                continue

        return {
            "iterations": iteration_results,
            "total_iterations": self.context.session.iterations,
            "final_answer": self.context.session.final_answer,
            "reasoning_strategy": (
                self.strategy_manager.current_strategy.value
                if self.strategy_manager.current_strategy
                else "unknown"
            ),
        }

    async def _should_continue(self) -> bool:
        """Determine if the reasoning loop should continue."""

        # Check terminal statuses
        if self.context.session.task_status in [
            TaskStatus.COMPLETE,
            TaskStatus.ERROR,
            TaskStatus.CANCELLED,
            TaskStatus.RESCOPED_COMPLETE,
            TaskStatus.MAX_ITERATIONS,
            TaskStatus.MISSING_TOOLS,
        ]:
            return False

        # Check max iterations
        max_iterations = self.context.max_iterations or 20
        if self.context.session.iterations >= max_iterations:
            self.context.session.task_status = TaskStatus.MAX_ITERATIONS
            return False

        # Check if final answer is set
        if self.context.session.final_answer:
            self.context.session.task_status = TaskStatus.COMPLETE
            return False

        return True

    def _handle_control_signals(self) -> bool:
        """Handle pause, terminate, and stop signals."""
        if self._terminate_requested:
            if self.agent.agent_logger:
                self.agent.agent_logger.info("🛑 Termination requested")
            self.context.session.task_status = TaskStatus.CANCELLED
            return True

        if self._stop_requested:
            if self.agent.agent_logger:
                self.agent.agent_logger.info("⏹️ Stop requested")
            self.context.session.task_status = TaskStatus.CANCELLED
            return True

        return False

    async def _prepare_final_result(
        self, execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare the final execution result with comprehensive metadata."""

        # Calculate final completion score
        completion_score = 1.0 if self.context.session.final_answer else 0.5

        # Update session status
        if self.context.session.final_answer:
            self.context.session.task_status = TaskStatus.COMPLETE
        elif self.context.session.task_status == TaskStatus.RUNNING:
            self.context.session.task_status = TaskStatus.ERROR

        # Save memory before preparing final result
        await self._save_memory()

        # Extract tool usage information
        tool_usage_stats = self._extract_tool_usage_stats(execution_result)

        # Extract performance metrics
        performance_metrics = self._extract_performance_metrics()

        # Extract reasoning context
        reasoning_context = self._extract_reasoning_context()

        result = {
            "status": self.context.session.task_status.value,
            "final_answer": self.context.session.final_answer,
            "completion_score": completion_score,
            "iterations": self.context.session.iterations,
            "reasoning_strategy": execution_result.get("reasoning_strategy", "unknown"),
            "execution_details": execution_result,
            "session_id": getattr(self.context.session, "session_id", "unknown"),
            "task_classification": self.strategy_manager.get_reasoning_context().task_classification,
            "tool_usage": tool_usage_stats,
            "performance_metrics": performance_metrics,
            "reasoning_context": reasoning_context,
            "session_metadata": self._extract_session_metadata(),
        }

        # Emit session ended event
        self.context.emit_event(
            AgentStateEvent.SESSION_ENDED,
            {
                "final_result": result,
                "session_id": result["session_id"],
            },
        )

        if self.agent.agent_logger:
            self.agent.agent_logger.info(
                f"🎯 Execution completed: {result['status']} "
                f"(iterations: {result['iterations']}, "
                f"strategy: {result['reasoning_strategy']})"
            )

        return result

    async def _save_memory(self):
        """Save memory using the same logic as the parent AgentExecutionEngine."""
        if self.context.memory_manager:
            if hasattr(self.context, "session") and self.context.session:
                # Use vector memory if available, otherwise fall back to traditional
                if hasattr(self.context.memory_manager, "store_session_memory"):
                    # Vector memory manager - use async method for vector storage
                    memory_manager = self.context.memory_manager
                    if hasattr(memory_manager, "is_ready"):
                        # This is a VectorMemoryManager
                        if memory_manager.is_ready():  # type: ignore
                            if self.context.agent_logger:
                                self.context.agent_logger.debug(
                                    "Storing session memory with vector memory manager"
                                )
                            await memory_manager.store_session_memory(self.context.session)  # type: ignore
                        else:
                            if self.context.agent_logger:
                                self.context.agent_logger.warning(
                                    "Vector memory not ready, skipping session storage"
                                )

                    # Always save traditional memory (reflections, session history, tool preferences)
                    self.context.memory_manager.update_session_history(
                        self.context.session
                    )
                    self.context.memory_manager.save_memory()
                else:
                    # Traditional memory manager only
                    self.context.memory_manager.update_session_history(
                        self.context.session
                    )
                    self.context.memory_manager.save_memory()

    async def _handle_execution_error(self, error: Exception) -> Dict[str, Any]:
        """Handle execution errors and return error result."""

        self.context.session.task_status = TaskStatus.ERROR

        error_result = {
            "status": TaskStatus.ERROR.value,
            "error": str(error),
            "final_answer": None,
            "completion_score": 0.0,
            "iterations": self.context.session.iterations,
            "reasoning_strategy": "unknown",
            "execution_details": {
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
            "session_id": getattr(self.context.session, "session_id", "unknown"),
            "tool_usage": self._extract_tool_usage_stats({"iterations": []}),
            "performance_metrics": self._extract_performance_metrics(),
            "reasoning_context": self._extract_reasoning_context(),
            "session_metadata": self._extract_session_metadata(),
        }

        # Emit error event
        self.context.emit_event(
            AgentStateEvent.ERROR_OCCURRED,
            {
                "error": str(error),
                "session_id": error_result["session_id"],
            },
        )

        return error_result

    def _extract_tool_usage_stats(
        self, execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract comprehensive tool usage statistics."""
        tool_history = []
        tool_counts = {}
        successful_tools = set()
        failed_tools = set()

        # Extract tool usage from tool manager history
        if hasattr(self.context, "tool_manager") and self.context.tool_manager:
            tool_manager = self.context.tool_manager

            # Process tool history from tool manager
            for entry in tool_manager.tool_history:
                tool_name = entry.get("name", "unknown")
                tool_result = entry.get("result", "")
                params = entry.get("params", {})
                execution_time = entry.get("execution_time")
                is_error = entry.get("error", False)
                is_cancelled = entry.get("cancelled", False)
                is_cached = entry.get("cached", False)
                timestamp = entry.get("timestamp")

                # Track tool usage
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

                # Determine success/failure
                is_success = not (is_error or is_cancelled)

                if is_success:
                    successful_tools.add(tool_name)
                else:
                    failed_tools.add(tool_name)

                # Add to history
                tool_history.append(
                    {
                        "tool_name": tool_name,
                        "params": params,
                        "result": tool_result,
                        "success": is_success,
                        "execution_time": execution_time,
                        "cached": is_cached,
                        "error": is_error,
                        "cancelled": is_cancelled,
                        "timestamp": timestamp,
                    }
                )

        # Extract from successful_tools set in session
        if hasattr(self.context.session, "successful_tools"):
            successful_tools.update(self.context.session.successful_tools)

        # Calculate success rate
        total_tools = len(tool_counts)
        success_rate = (
            len(successful_tools) / max(total_tools, 1) if total_tools > 0 else 0.0
        )

        return {
            "total_tool_calls": len(tool_history),
            "unique_tools_used": list(set(tool_counts.keys())),
            "tool_call_counts": tool_counts,
            "successful_tools": list(successful_tools),
            "failed_tools": list(failed_tools),
            "success_rate": success_rate,
            "tool_history": tool_history,
            "available_tools": [
                tool["function"]["name"] for tool in self.context.get_tool_signatures()
            ],
            "cache_stats": (
                {
                    "hits": (
                        getattr(self.context.tool_manager, "cache_hits", 0)
                        if hasattr(self.context, "tool_manager")
                        else 0
                    ),
                    "misses": (
                        getattr(self.context.tool_manager, "cache_misses", 0)
                        if hasattr(self.context, "tool_manager")
                        else 0
                    ),
                }
                if hasattr(self.context, "tool_manager")
                else {}
            ),
        }

    def _extract_performance_metrics(self) -> Dict[str, Any]:
        """Extract performance and timing metrics."""
        start_time = getattr(self.context.session, "start_time", None)
        end_time = time.time()

        metrics = {
            "execution_time_seconds": end_time - start_time if start_time else None,
            "iterations_per_second": (
                (self.context.session.iterations / (end_time - start_time))
                if start_time and (end_time - start_time) > 0
                else None
            ),
            "average_iteration_time": (
                (end_time - start_time) / max(self.context.session.iterations, 1)
                if start_time
                else None
            ),
            "max_iterations": self.context.max_iterations or 20,
            "iterations_used": self.context.session.iterations,
            "iteration_efficiency": self.context.session.iterations
            / max(self.context.max_iterations or 20, 1),
            "total_tool_execution_time": self._calculate_total_tool_time(),
            "average_tool_execution_time": self._calculate_average_tool_time(),
            "tool_execution_percentage": self._calculate_tool_execution_percentage(),
        }

        # Add memory usage if available
        if hasattr(self.context.session, "metrics"):
            metrics.update(self.context.session.metrics)

        return metrics

    def _calculate_total_tool_time(self) -> float:
        """Calculate total time spent executing tools."""
        if not hasattr(self.context, "tool_manager") or not self.context.tool_manager:
            return 0.0

        total_time = 0.0
        for entry in self.context.tool_manager.tool_history:
            execution_time = entry.get("execution_time", 0.0)
            if execution_time:
                total_time += execution_time
        return total_time

    def _calculate_average_tool_time(self) -> float:
        """Calculate average time per tool execution."""
        if not hasattr(self.context, "tool_manager") or not self.context.tool_manager:
            return 0.0

        tool_history = self.context.tool_manager.tool_history
        if not tool_history:
            return 0.0

        total_time = self._calculate_total_tool_time()
        return total_time / len(tool_history)

    def _calculate_tool_execution_percentage(self) -> float:
        """Calculate percentage of total execution time spent on tool execution."""
        start_time = getattr(self.context.session, "start_time", None)
        if not start_time:
            return 0.0

        total_execution_time = time.time() - start_time
        if total_execution_time <= 0:
            return 0.0

        tool_time = self._calculate_total_tool_time()
        return (tool_time / total_execution_time) * 100.0

    def _extract_mcp_servers(self) -> List[str]:
        """Extract MCP server information."""
        mcp_servers = []

        # Get from context if available
        context_mcp_servers = getattr(self.context, "mcp_servers", None)
        if context_mcp_servers:
            mcp_servers.extend(context_mcp_servers)

        # Get from MCP client if available
        mcp_client = getattr(self.context, "mcp_client", None)
        if mcp_client:
            try:
                # Extract server names from MCP client's server_tools dict
                server_tools = getattr(mcp_client, "server_tools", {})
                mcp_servers.extend(list(server_tools.keys()))
            except Exception:
                pass

        return list(set(mcp_servers))  # Remove duplicates

    def _extract_mcp_tools(self) -> List[str]:
        """Extract MCP tool information."""
        mcp_tools = []

        # Get from context if available
        context_mcp_tools = getattr(self.context, "mcp_tools", None)
        if context_mcp_tools:
            mcp_tools.extend(context_mcp_tools)

        # Get from MCP client if available
        mcp_client = getattr(self.context, "mcp_client", None)
        if mcp_client:
            try:
                # Extract tool names from MCP client
                tools = getattr(mcp_client, "tools", [])
                for tool in tools:
                    tool_name = getattr(tool, "name", None)
                    if tool_name:
                        mcp_tools.append(tool_name)
            except Exception:
                pass

        return list(set(mcp_tools))  # Remove duplicates

    def _extract_custom_tools(self) -> List[str]:
        """Extract custom tool information."""
        custom_tools = []

        # Get from context if available
        context_custom_tools = getattr(self.context, "custom_tools", None)
        if context_custom_tools:
            for tool in context_custom_tools:
                if hasattr(tool, "name"):
                    custom_tools.append(tool.name)
                elif isinstance(tool, dict) and "name" in tool:
                    custom_tools.append(tool["name"])

        # Get from tool manager if available
        if hasattr(self.context, "tool_manager") and self.context.tool_manager:
            try:
                for tool in self.context.tool_manager.tools:
                    if hasattr(tool, "name") and tool.name != "final_answer":
                        # Check if it's not an MCP tool
                        if not any(
                            mcp_tool in tool.name
                            for mcp_tool in self._extract_mcp_tools()
                        ):
                            custom_tools.append(tool.name)
            except Exception:
                pass

        return list(set(custom_tools))  # Remove duplicates

    def _extract_reasoning_context(self) -> Dict[str, Any]:
        """Extract reasoning context and strategy information."""
        reasoning_context = self.strategy_manager.get_reasoning_context()

        return {
            "current_strategy": (
                reasoning_context.current_strategy.value
                if reasoning_context.current_strategy
                else "unknown"
            ),
            "strategy_switches": getattr(reasoning_context, "strategy_switches", []),
            "iteration_count": reasoning_context.iteration_count,
            "error_count": reasoning_context.error_count,
            "stagnation_count": reasoning_context.stagnation_count,
            "tool_usage_history": reasoning_context.tool_usage_history,
            "task_classification": reasoning_context.task_classification,
            "last_action_result": reasoning_context.last_action_result,
            "success_indicators": reasoning_context.success_indicators,
        }

    def _extract_session_metadata(self) -> Dict[str, Any]:
        """Extract session metadata and configuration."""
        return {
            "agent_name": self.context.agent_name,
            "model_provider": self.context.provider_model_name,
            "role": self.context.role,
            "instructions": self.context.instructions,
            "max_iterations": self.context.max_iterations,
            "reflect_enabled": self.context.reflect_enabled,
            "tool_use_enabled": self.context.tool_use_enabled,
            "use_memory_enabled": self.context.use_memory_enabled,
            "collect_metrics_enabled": self.context.collect_metrics_enabled,
            "session_start_time": getattr(self.context.session, "start_time", None),
            "session_end_time": time.time(),
            "messages_count": (
                len(self.context.session.messages) if self.context.session else 0
            ),
            "context_tokens_estimate": getattr(
                self.context.session, "context_tokens", 0
            ),
            "mcp_servers": self._extract_mcp_servers(),
            "mcp_tools": self._extract_mcp_tools(),
            "custom_tools": self._extract_custom_tools(),
            "initial_task": getattr(self.context.session, "initial_task", ""),
            "current_task": getattr(self.context.session, "current_task", ""),
            "task_status": (
                self.context.session.task_status.value
                if self.context.session.task_status
                else None
            ),
            "model_provider_options": self.context.model_provider_options,
            "check_tool_feasibility": self.context.check_tool_feasibility,
            "enable_caching": self.context.enable_caching,
            "min_completion_score": self.context.min_completion_score,
        }

    # Control methods
    async def pause(self):
        """Pause the execution."""
        self._paused = True
        self._pause_event.clear()

    async def resume(self):
        """Resume the execution."""
        self._paused = False
        self._pause_event.set()

    async def terminate(self):
        """Terminate the execution."""
        self._terminate_requested = True

    async def stop(self):
        """Stop the execution."""
        self._stop_requested = True
