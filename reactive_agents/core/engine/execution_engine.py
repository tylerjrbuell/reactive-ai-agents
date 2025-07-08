from __future__ import annotations
import asyncio
import time
import traceback
from typing import Dict, Any, Optional, TYPE_CHECKING
from reactive_agents.core.types.status_types import TaskStatus
from reactive_agents.core.types.event_types import AgentStateEvent
from reactive_agents.core.types.reasoning_types import (
    ReasoningContext,
    ReasoningStrategies,
)

if TYPE_CHECKING:
    from reactive_agents.app.agents.base import Agent


class ExecutionEngine:
    """
    Clean, unified execution engine for reactive agents.

    Provides a single execution path that:
    - Selects strategies based on configuration
    - Executes reasoning loops with proper control flow
    - Manages sessions and lifecycle events
    - Handles errors and control signals
    """

    def __init__(self, agent: "Agent"):
        """Initialize the execution engine."""
        self.agent = agent
        self.context = agent.context
        self.agent_logger = agent.agent_logger

        # Initialize strategy management
        self._initialize_strategy_system()

        # Control state
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._terminate_requested = False
        self._stop_requested = False

        if self.agent_logger:
            self.agent_logger.info("✅ Execution engine initialized")

    def _initialize_strategy_system(self):
        """Initialize strategy management components."""
        from reactive_agents.core.reasoning.strategy_manager import StrategyManager
        from reactive_agents.core.reasoning.task_classifier import TaskClassifier
        from reactive_agents.core.reasoning.infrastructure import (
            get_reasoning_infrastructure,
        )

        # Get shared infrastructure
        self.infrastructure = get_reasoning_infrastructure(self.context)

        # Initialize strategy manager
        self.strategy_manager = StrategyManager(self.infrastructure, self.context)
        self.task_classifier = TaskClassifier(self.context)

    async def execute(
        self,
        initial_task: str,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Execute a task from start to finish.

        Args:
            initial_task: The task to execute
            cancellation_event: Optional cancellation event

        Returns:
            Comprehensive execution results
        """
        if self.agent_logger:
            self.agent_logger.info(f"🚀 Starting task: {initial_task[:100]}...")

        # Setup session
        self._setup_session(initial_task)

        try:
            # Select strategy based on configuration
            await self._select_strategy(initial_task)

            # Execute main loop
            result = await self._execute_loop(initial_task, cancellation_event)

            # Prepare final result
            return self._prepare_result(result)

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Execution failed: {e}")
            return self._handle_error(e)

    def _setup_session(self, initial_task: str):
        """Setup session for task execution."""
        # Initialize session if needed
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

        # Reset for new task
        self.context.session.iterations = 0
        self.context.session.final_answer = None
        self.context.session.current_task = initial_task
        self.context.session.initial_task = initial_task
        self.context.session.task_status = TaskStatus.RUNNING

        # Reset strategy system
        self.strategy_manager.reset()

        # Emit start event
        self.context.emit_event(
            AgentStateEvent.SESSION_STARTED,
            {
                "initial_task": initial_task,
                "session_id": getattr(self.context.session, "session_id", "unknown"),
            },
        )

    async def _select_strategy(self, task: str):
        """Select execution strategy based on configuration."""
        strategy_mode = getattr(self.context, "strategy_mode", "adaptive")

        if strategy_mode == "static":
            # Use configured strategy
            static_strategy = getattr(
                self.context, "static_strategy", "reflect_decide_act"
            )
            self.strategy_manager.set_strategy(static_strategy)

            if self.agent_logger:
                self.agent_logger.info(f"🎯 Using static strategy: {static_strategy}")

        elif strategy_mode in ["adaptive", "dynamic"]:
            # Classify and select optimal strategy
            if self.task_classifier:
                classification = await self.task_classifier.classify_task(task)
                strategy = self.strategy_manager.select_optimal_strategy(classification)

                if self.agent_logger:
                    self.agent_logger.info(
                        f"📋 Task: {classification.task_type.value} "
                        f"(confidence: {classification.confidence:.2f})"
                    )
                    self.agent_logger.info(f"🎯 Strategy: {strategy}")
            else:
                # Default fallback
                self.strategy_manager.set_strategy("reflect_decide_act")
                if self.agent_logger:
                    self.agent_logger.warning("Using default strategy (no classifier)")

    async def _execute_loop(
        self, task: str, cancellation_event: Optional[asyncio.Event] = None
    ) -> Dict[str, Any]:
        """Execute the main reasoning loop."""
        iteration_results = []
        max_iterations = self.context.max_iterations or 20

        # Create reasoning context
        reasoning_context = ReasoningContext(
            current_strategy=ReasoningStrategies.REFLECT_DECIDE_ACT
        )

        while self._should_continue():
            # Check for cancellation
            if cancellation_event and cancellation_event.is_set():
                if self.agent_logger:
                    self.agent_logger.info("🛑 Cancelled by external event")
                break

            # Check control signals
            if self._check_control_signals():
                break

            self.context.session.iterations += 1

            if self.agent_logger:
                self.agent_logger.info(
                    f"🔄 Iteration {self.context.session.iterations}/{max_iterations} "
                    f"using {self.strategy_manager.get_current_strategy_name()}"
                )

            # Emit iteration start
            self.context.emit_event(
                AgentStateEvent.ITERATION_STARTED,
                {
                    "iteration": self.context.session.iterations,
                    "max_iterations": max_iterations,
                    "strategy": self.strategy_manager.get_current_strategy_name(),
                },
            )

            try:
                # Execute one iteration using current strategy
                strategy_result = await self.strategy_manager.execute_iteration(
                    task, reasoning_context
                )

                # Convert to dict for compatibility
                iteration_result = strategy_result.to_dict()
                iteration_results.append(iteration_result)

                # Update context
                reasoning_context.iteration_count = self.context.session.iterations

                # Check if should continue
                if not iteration_result.get("should_continue", True):
                    if self.agent_logger:
                        self.agent_logger.info("🏁 Strategy completed")
                    break

                # Check for final answer
                if self.context.session.final_answer:
                    if self.agent_logger:
                        self.agent_logger.info("✅ Final answer provided")
                    break

                # Emit iteration complete
                self.context.emit_event(
                    AgentStateEvent.ITERATION_COMPLETED,
                    {
                        "iteration": self.context.session.iterations,
                        "result": iteration_result,
                    },
                )

                # Manage context
                # await self.context.manage_context()

            except Exception as e:
                if self.agent_logger:
                    self.agent_logger.error(
                        f"Iteration {self.context.session.iterations} failed: {e}"
                    )

                iteration_results.append(
                    {"error": str(e), "iteration": self.context.session.iterations}
                )
                continue

        return {
            "iterations": iteration_results,
            "total_iterations": self.context.session.iterations,
            "final_answer": self.context.session.final_answer,
            "strategy": self.strategy_manager.get_current_strategy_name(),
        }

    def _should_continue(self) -> bool:
        """Check if execution should continue."""
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

        # Check final answer
        if self.context.session.final_answer:
            self.context.session.task_status = TaskStatus.COMPLETE
            return False

        return True

    def _check_control_signals(self) -> bool:
        """Handle control signals (pause, stop, terminate)."""
        if self._terminate_requested:
            if self.agent_logger:
                self.agent_logger.info("🛑 Termination requested")
            self.context.session.task_status = TaskStatus.CANCELLED
            return True

        if self._stop_requested:
            if self.agent_logger:
                self.agent_logger.info("⏹️ Stop requested")
            self.context.session.task_status = TaskStatus.CANCELLED
            return True

        return False

    def _prepare_result(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare final execution result."""
        # Calculate completion score
        completion_score = 1.0 if self.context.session.final_answer else 0.5

        # Update session status
        if self.context.session.final_answer:
            self.context.session.task_status = TaskStatus.COMPLETE
        elif self.context.session.task_status == TaskStatus.RUNNING:
            self.context.session.task_status = TaskStatus.ERROR

        result = {
            "status": self.context.session.task_status.value,
            "final_answer": self.context.session.final_answer,
            "completion_score": completion_score,
            "iterations": self.context.session.iterations,
            "strategy": execution_result.get("strategy", "unknown"),
            "execution_details": execution_result,
            "session_id": getattr(self.context.session, "session_id", "unknown"),
        }

        # Emit session end
        self.context.emit_event(
            AgentStateEvent.SESSION_ENDED,
            {
                "final_result": result,
                "session_id": result["session_id"],
            },
        )

        if self.agent_logger:
            self.agent_logger.info(
                f"🎯 Completed: {result['status']} "
                f"(iterations: {result['iterations']}, strategy: {result['strategy']})"
            )

        return result

    def _handle_error(self, error: Exception) -> Dict[str, Any]:
        """Handle execution errors."""
        self.context.session.task_status = TaskStatus.ERROR

        error_result = {
            "status": TaskStatus.ERROR.value,
            "error": str(error),
            "final_answer": None,
            "completion_score": 0.0,
            "iterations": self.context.session.iterations,
            "strategy": "unknown",
            "execution_details": {
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
            "session_id": getattr(self.context.session, "session_id", "unknown"),
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

    # Control methods
    async def pause(self):
        """Pause execution."""
        self._paused = True
        self._pause_event.clear()

    async def resume(self):
        """Resume execution."""
        self._paused = False
        self._pause_event.set()

    async def terminate(self):
        """Terminate execution."""
        self._terminate_requested = True

    async def stop(self):
        """Stop execution."""
        self._stop_requested = True
