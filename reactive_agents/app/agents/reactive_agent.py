import asyncio
import json
from typing import Dict, Any, Optional, List

from reactive_agents.app.agents.base import Agent
from reactive_agents.core.types.agent_types import ReactiveAgentConfig
from reactive_agents.core.context.agent_context import AgentContext
from reactive_agents.core.engine.execution_engine import ExecutionEngine
from reactive_agents.core.events.event_bus import EventBus
from reactive_agents.config.validators.config_validator import ConfigValidator

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
        config: ReactiveAgentConfig,
        context: Optional[AgentContext] = None,
        event_bus: Optional[EventBus] = None,

    ):
        self.config = config
        # Create context if not provided (for builder pattern compatibility)
        if context is None:
            # Convert ReactAgentConfig to AgentContext by extracting fields
            context_data = config.model_dump()
            context = AgentContext(**context_data)

            # Ensure event_bus is initialized if not present
            if context.event_bus is None:
                context.event_bus = EventBus(config.agent_name)

        super().__init__(context)

        # Initialize event bus if not provided
        if event_bus is None:
            self._event_bus = context.event_bus
        else:
            self._event_bus = event_bus



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
                f"🚀 Starting reactive agent with task: {initial_task}..."
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

    async def initialize(self) -> None:
        """Initialize the agent."""
        if self.agent_logger:
            self.agent_logger.info("🔧 Initializing reactive agent...")

        # Basic initialization
        await super().initialize()

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

    # === Control State Queries ===
    def is_paused(self) -> bool:
        """Check if the agent is currently paused."""
        if self.execution_engine:
            return self.execution_engine.is_paused()
        return False

    def is_terminating(self) -> bool:
        """Check if termination has been requested."""
        if self.execution_engine:
            return self.execution_engine.is_terminating()
        return False

    def is_stopping(self) -> bool:
        """Check if stop has been requested."""
        if self.execution_engine:
            return self.execution_engine.is_stopping()
        return False

    def get_control_state(self) -> Dict[str, Any]:
        """Get current control state for monitoring."""
        if self.execution_engine:
            return self.execution_engine.get_control_state()
        return {
            "paused": False,
            "terminate_requested": False,
            "stop_requested": False,
            "pause_event_set": True,
        }

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

    @property
    def events(self):
        """Get the events interface for subscribing to agent events."""
        if hasattr(self, "_event_bus") and self._event_bus:
            return self._event_bus
        else:
            # Fallback to creating a new event bus
            agent_name = getattr(self.config, "agent_name", "UnknownAgent")
            return EventBus(agent_name)

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
    # These methods now delegate to the EventBus for a cleaner API
    def on_session_started(self, callback):
        """Register a callback for session started events."""
        return self.events.on_session_started().subscribe(callback)

    def on_session_ended(self, callback):
        """Register a callback for session ended events."""
        return self.events.on_session_ended().subscribe(callback)

    def on_iteration_started(self, callback):
        """Register a callback for iteration started events."""
        return self.events.on_iteration_started().subscribe(callback)

    def on_iteration_completed(self, callback):
        """Register a callback for iteration completed events."""
        return self.events.on_iteration_completed().subscribe(callback)

    def on_tool_called(self, callback):
        """Register a callback for tool called events."""
        return self.events.on_tool_called().subscribe(callback)

    def on_tool_completed(self, callback):
        """Register a callback for tool completed events."""
        return self.events.on_tool_completed().subscribe(callback)

    def on_tool_failed(self, callback):
        """Register a callback for tool failed events."""
        return self.events.on_tool_failed().subscribe(callback)

    def on_task_status_changed(self, callback):
        """Register a callback for task status changed events."""
        return self.events.on_task_status_changed().subscribe(callback)

    def on_reflection_generated(self, callback):
        """Register a callback for reflection generated events."""
        return self.events.on_reflection_generated().subscribe(callback)

    def on_final_answer_set(self, callback):
        """Register a callback for final answer set events."""
        return self.events.on_final_answer_set().subscribe(callback)

    def on_metrics_updated(self, callback):
        """Register a callback for metrics updated events."""
        return self.events.on_metrics_updated().subscribe(callback)

    def on_error_occurred(self, callback):
        """Register a callback for error occurred events."""
        return self.events.on_error_occurred().subscribe(callback)

    def on_pause_requested(self, callback):
        """Register a callback for pause requested events."""
        return self.events.on_pause_requested().subscribe(callback)

    def on_paused(self, callback):
        """Register a callback for paused events."""
        return self.events.on_paused().subscribe(callback)

    def on_resume_requested(self, callback):
        """Register a callback for resume requested events."""
        return self.events.on_resume_requested().subscribe(callback)

    def on_resumed(self, callback):
        """Register a callback for resumed events."""
        return self.events.on_resumed().subscribe(callback)

    def on_stop_requested(self, callback):
        """Register a callback for stop requested events."""
        return self.events.on_stop_requested().subscribe(callback)

    def on_stopped(self, callback):
        """Register a callback for stopped events."""
        return self.events.on_stopped().subscribe(callback)

    def on_terminate_requested(self, callback):
        """Register a callback for terminate requested events."""
        return self.events.on_terminate_requested().subscribe(callback)

    def on_terminated(self, callback):
        """Register a callback for terminated events."""
        return self.events.on_terminated().subscribe(callback)

    def on_cancelled(self, callback):
        """Register a callback for cancelled events."""
        return self.events.on_cancelled().subscribe(callback)

    # Note: The following events may not be implemented in the current EventBus
    # They are kept for backward compatibility but may need to be implemented
    def on_context_changed(self, callback):
        """Register a callback for context changed events."""
        # This event type may need to be added to the EventBus
        if hasattr(self.context, "event_bus") and self.context.event_bus:
            from reactive_agents.core.types.event_types import AgentStateEvent

            # For now, we'll use a generic event type or skip
            pass
        return None

    def on_operation_completed(self, callback):
        """Register a callback for operation completed events."""
        # This event type may need to be added to the EventBus
        if hasattr(self.context, "event_bus") and self.context.event_bus:
            from reactive_agents.core.types.event_types import AgentStateEvent

            # For now, we'll use a generic event type or skip
            pass
        return None

    def on_tokens_used(self, callback):
        """Register a callback for tokens used events."""
        # This event type may need to be added to the EventBus
        if hasattr(self.context, "event_bus") and self.context.event_bus:
            from reactive_agents.core.types.event_types import AgentStateEvent

            # For now, we'll use a generic event type or skip
            pass
        return None

    def on_snapshot_taken(self, callback):
        """Register a callback for snapshot taken events."""
        # This event type may need to be added to the EventBus
        if hasattr(self.context, "event_bus") and self.context.event_bus:
            from reactive_agents.core.types.event_types import AgentStateEvent

            # For now, we'll use a generic event type or skip
            pass
        return None
