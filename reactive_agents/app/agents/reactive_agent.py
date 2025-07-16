import asyncio
import json
from typing import Dict, Any, Optional, List

from reactive_agents.app.agents.base import Agent
from reactive_agents.core.types.agent_types import ReactAgentConfig
from reactive_agents.core.context.agent_context import AgentContext
from reactive_agents.core.engine.execution_engine import ExecutionEngine
from reactive_agents.core.events.event_manager import EventManager
from reactive_agents.config.validators.config_validator import ConfigValidator
from reactive_agents.core.tools.tool_processor import ToolProcessor
from reactive_agents.core.types.execution_types import ExecutionResult
from reactive_agents.core.types.status_types import TaskStatus


class ReactiveAgent(Agent):
    """
    Clean, reactive agent using the unified execution engine.

    Provides a simple, efficient agent implementation that:
    - Uses the ExecutionEngine for task execution
    - Supports strategy configuration (static or adaptive)
    - Handles events and lifecycle management
    - Focuses on core functionality without complexity
    """

    def __init__(
        self,
        config: ReactAgentConfig,
        context: Optional[AgentContext] = None,
        event_manager: Optional[EventManager] = None,
        tool_processor: Optional[ToolProcessor] = None,
    ):
        self.config = config
        print("config", config)
        # Create context if not provided (for builder pattern compatibility)
        if context is None:
            # Convert ReactAgentConfig to AgentContext by extracting fields
            context_data = config.model_dump()
            context = AgentContext(**context_data)

        super().__init__(context)

        # Store unused parameters for potential future use
        self._event_manager = event_manager
        self._tool_processor = tool_processor

        # Initialize execution engine
        self.execution_engine = ExecutionEngine(self)

    async def run(
        self,
        initial_task: str,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> ExecutionResult:
        """
        Run the agent with the given task.

        This is the main entry point for agent execution.
        """
        result: Optional[ExecutionResult] = None

        if self.agent_logger:
            self.agent_logger.info(
                f"🚀 Starting reactive agent with task: {initial_task[:100]}..."
            )

        try:
            # Use the clean execution engine
            if self.execution_engine:
                result = await self.execution_engine.execute(
                    initial_task, cancellation_event
                )
            else:
                raise RuntimeError("Execution engine not initialized")

            if self.agent_logger:
                self.agent_logger.info(f"✅ Task completed: {result.status.value}")

            return result

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"❌ Agent execution failed: {e}")

            return ExecutionResult(
                status=TaskStatus.ERROR,
                final_answer=None,
                session=self.context.session,
                strategy_used=self.context.reasoning_strategy,
                execution_details=result.model_dump() if result else {},
                task_metrics={},
            )

    async def run_with_strategy(
        self,
        initial_task: str,
        strategy: str,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> ExecutionResult:
        """
        Run the agent with a specific strategy.

        Temporarily overrides the configured strategy settings.
        """
        # Store original settings
        original_switching = getattr(
            self.context, "enable_dynamic_strategy_switching", True
        )
        original_strategy = getattr(
            self.context, "reasoning_strategy", "reflect_decide_act"
        )

        # Set to use specific strategy with dynamic switching disabled
        setattr(self.context, "enable_dynamic_strategy_switching", False)
        setattr(self.context, "reasoning_strategy", strategy)

        try:
            if self.agent_logger:
                self.agent_logger.info(f"🎯 Running with forced strategy: {strategy}")

            result = await self.run(initial_task, cancellation_event=cancellation_event)
            return result

        finally:
            # Restore original settings
            setattr(
                self.context, "enable_dynamic_strategy_switching", original_switching
            )
            setattr(self.context, "reasoning_strategy", original_strategy)

    # === Agent Interface Implementation ===
    async def _execute_task(self, task: str) -> ExecutionResult:
        """Execute a task (required by base Agent class)."""
        return await self.run(task)

    def _register_event_callbacks(self):
        """
        Subscribe all registered event handler callbacks in _event_callbacks to the event manager.
        """
        if not hasattr(self, "_event_callbacks"):
            return
        event_manager = getattr(self, "_event_manager", None)
        if event_manager is None:
            return
        for event_type, callbacks in self._event_callbacks.items():
            subscribe_method = getattr(event_manager, f"on_{event_type}", None)
            if callable(subscribe_method):
                for cb in callbacks:
                    subscribe_method(cb)

    async def initialize(self) -> None:
        """Initialize the agent."""
        if self.agent_logger:
            self.agent_logger.info("🔧 Initializing reactive agent...")

        # Basic initialization
        await super().initialize()

        # Register event callbacks with the event manager
        self._register_event_callbacks()

        if self.agent_logger:
            self.agent_logger.info("✅ Reactive agent initialized")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.agent_logger:
            self.agent_logger.info("🧹 Cleaning up reactive agent...")

        # Basic cleanup - no parent cleanup method exists
        pass

        if self.agent_logger:
            self.agent_logger.info("✅ Reactive agent cleaned up")

    # === Control Methods ===
    async def pause(self):
        """Pause the agent execution."""
        if self.execution_engine:
            await self.execution_engine.pause()

    async def resume(self):
        """Resume the agent execution."""
        if self.execution_engine:
            await self.execution_engine.resume()

    async def stop(self):
        """Stop the agent execution."""
        if self.execution_engine:
            await self.execution_engine.stop()

    async def terminate(self):
        """Terminate the agent execution."""
        if self.execution_engine:
            await self.execution_engine.terminate()

    # === Status Methods ===
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        ts = (
            getattr(self.context.session, "task_status", "unknown")
            if self.context.session
            else "unknown"
        )
        return {
            "agent_name": self.context.agent_name,
            "reasoning_strategy": getattr(
                self.context, "reasoning_strategy", "reflect_decide_act"
            ),
            "dynamic_strategy_switching": getattr(
                self.context, "enable_dynamic_strategy_switching", True
            ),
            "session_active": self.context.session is not None,
            "iterations": (
                getattr(self.context.session, "iterations", 0)
                if self.context.session
                else 0
            ),
            "task_status": (
                ts.value if not isinstance(ts, str) and hasattr(ts, "value") else ts
            ),
        }

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies."""
        if (
            self.execution_engine
            and hasattr(self.execution_engine, "strategy_manager")
            and self.execution_engine.strategy_manager
        ):
            return list(
                self.execution_engine.strategy_manager.get_available_strategies().keys()
            )
        return []

    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """Get information about a specific strategy."""
        if (
            self.execution_engine
            and hasattr(self.execution_engine, "strategy_manager")
            and self.execution_engine.strategy_manager
        ):
            return self.execution_engine.strategy_manager.get_strategy_info(
                strategy_name
            )
        return {"error": "No strategy manager available"}

    # === Event Handler Registration Methods ===
    def on_session_started(self, callback):
        """Register a callback for session started events."""
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "session_started" not in self._event_callbacks:
            self._event_callbacks["session_started"] = []
        self._event_callbacks["session_started"].append(callback)

    def on_session_ended(self, callback):
        """Register a callback for session ended events."""
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "session_ended" not in self._event_callbacks:
            self._event_callbacks["session_ended"] = []
        self._event_callbacks["session_ended"].append(callback)

    def on_iteration_started(self, callback):
        """Register a callback for iteration started events."""
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "iteration_started" not in self._event_callbacks:
            self._event_callbacks["iteration_started"] = []
        self._event_callbacks["iteration_started"].append(callback)

    def on_iteration_completed(self, callback):
        """Register a callback for iteration completed events."""
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "iteration_completed" not in self._event_callbacks:
            self._event_callbacks["iteration_completed"] = []
        self._event_callbacks["iteration_completed"].append(callback)

    def on_tool_called(self, callback):
        """Register a callback for tool called events."""
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "tool_called" not in self._event_callbacks:
            self._event_callbacks["tool_called"] = []
        self._event_callbacks["tool_called"].append(callback)

    def on_tool_completed(self, callback):
        """Register a callback for tool completed events."""
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "tool_completed" not in self._event_callbacks:
            self._event_callbacks["tool_completed"] = []
        self._event_callbacks["tool_completed"].append(callback)

    def on_tool_failed(self, callback):
        """Register a callback for tool failed events."""
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "tool_failed" not in self._event_callbacks:
            self._event_callbacks["tool_failed"] = []
        self._event_callbacks["tool_failed"].append(callback)

    def on_task_status_changed(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "task_status_changed" not in self._event_callbacks:
            self._event_callbacks["task_status_changed"] = []
        self._event_callbacks["task_status_changed"].append(callback)

    def on_reflection_generated(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "reflection_generated" not in self._event_callbacks:
            self._event_callbacks["reflection_generated"] = []
        self._event_callbacks["reflection_generated"].append(callback)

    def on_final_answer_set(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "final_answer_set" not in self._event_callbacks:
            self._event_callbacks["final_answer_set"] = []
        self._event_callbacks["final_answer_set"].append(callback)

    def on_metrics_updated(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "metrics_updated" not in self._event_callbacks:
            self._event_callbacks["metrics_updated"] = []
        self._event_callbacks["metrics_updated"].append(callback)

    def on_error_occurred(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "error_occurred" not in self._event_callbacks:
            self._event_callbacks["error_occurred"] = []
        self._event_callbacks["error_occurred"].append(callback)

    def on_pause_requested(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "pause_requested" not in self._event_callbacks:
            self._event_callbacks["pause_requested"] = []
        self._event_callbacks["pause_requested"].append(callback)

    def on_paused(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "paused" not in self._event_callbacks:
            self._event_callbacks["paused"] = []
        self._event_callbacks["paused"].append(callback)

    def on_resume_requested(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "resume_requested" not in self._event_callbacks:
            self._event_callbacks["resume_requested"] = []
        self._event_callbacks["resume_requested"].append(callback)

    def on_resumed(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "resumed" not in self._event_callbacks:
            self._event_callbacks["resumed"] = []
        self._event_callbacks["resumed"].append(callback)

    def on_stop_requested(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "stop_requested" not in self._event_callbacks:
            self._event_callbacks["stop_requested"] = []
        self._event_callbacks["stop_requested"].append(callback)

    def on_stopped(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "stopped" not in self._event_callbacks:
            self._event_callbacks["stopped"] = []
        self._event_callbacks["stopped"].append(callback)

    def on_terminate_requested(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "terminate_requested" not in self._event_callbacks:
            self._event_callbacks["terminate_requested"] = []
        self._event_callbacks["terminate_requested"].append(callback)

    def on_terminated(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "terminated" not in self._event_callbacks:
            self._event_callbacks["terminated"] = []
        self._event_callbacks["terminated"].append(callback)

    def on_cancelled(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "cancelled" not in self._event_callbacks:
            self._event_callbacks["cancelled"] = []
        self._event_callbacks["cancelled"].append(callback)

    def on_context_changed(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "context_changed" not in self._event_callbacks:
            self._event_callbacks["context_changed"] = []
        self._event_callbacks["context_changed"].append(callback)

    def on_operation_completed(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "operation_completed" not in self._event_callbacks:
            self._event_callbacks["operation_completed"] = []
        self._event_callbacks["operation_completed"].append(callback)

    def on_tokens_used(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "tokens_used" not in self._event_callbacks:
            self._event_callbacks["tokens_used"] = []
        self._event_callbacks["tokens_used"].append(callback)

    def on_snapshot_taken(self, callback):
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}
        if "snapshot_taken" not in self._event_callbacks:
            self._event_callbacks["snapshot_taken"] = []
        self._event_callbacks["snapshot_taken"].append(callback)
