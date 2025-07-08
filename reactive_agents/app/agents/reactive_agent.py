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
    ) -> Dict[str, Any]:
        """
        Run the agent with the given task.

        This is the main entry point for agent execution.
        """
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
                self.agent_logger.info(
                    f"✅ Task completed: {result.get('status', 'unknown')}"
                )

            return result

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"❌ Agent execution failed: {e}")

            return {
                "status": "error",
                "error": str(e),
                "final_answer": None,
                "completion_score": 0.0,
                "iterations": 0,
                "strategy": "unknown",
            }

    async def run_with_strategy(
        self,
        initial_task: str,
        strategy: str,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent with a specific strategy.

        Temporarily overrides the configured strategy mode.
        """
        # Store original settings
        original_mode = getattr(self.context, "strategy_mode", "adaptive")
        original_strategy = getattr(
            self.context, "static_strategy", "reflect_decide_act"
        )

        # Set to static mode with specified strategy
        setattr(self.context, "strategy_mode", "static")
        setattr(self.context, "static_strategy", strategy)

        try:
            if self.agent_logger:
                self.agent_logger.info(f"🎯 Running with forced strategy: {strategy}")

            result = await self.run(initial_task, cancellation_event=cancellation_event)
            return result

        finally:
            # Restore original settings
            setattr(self.context, "strategy_mode", original_mode)
            setattr(self.context, "static_strategy", original_strategy)

    # === Agent Interface Implementation ===
    async def _execute_task(self, task: str) -> Dict[str, Any]:
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
            "strategy_mode": getattr(self.context, "strategy_mode", "adaptive"),
            "static_strategy": getattr(
                self.context, "static_strategy", "reflect_decide_act"
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
            return self.execution_engine.strategy_manager.get_available_strategies()
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
