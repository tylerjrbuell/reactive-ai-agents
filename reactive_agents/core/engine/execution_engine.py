from __future__ import annotations
import asyncio
import time
import traceback
from typing import Dict, Any, Optional, TYPE_CHECKING
from reactive_agents.core.types.status_types import TaskStatus
from reactive_agents.core.types.event_types import AgentStateEvent
from reactive_agents.core.types.execution_types import ExecutionResult
from reactive_agents.core.types.reasoning_types import (
    ReasoningContext,
    ReasoningStrategies,
    StrategyAction,
    FinishTaskPayload,
    EvaluationPayload,
    ErrorPayload,
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

        # Get shared engine from context
        self.engine = self.context.reasoning_engine

        # Initialize managers
        self.strategy_manager = StrategyManager(self.context)
        self.task_classifier = TaskClassifier(self.context)
        self.context_manager = self.engine.get_context_manager()

    async def execute(
        self,
        initial_task: str,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> ExecutionResult:
        """
        Execute a task from start to finish.

        Args:
            initial_task: The task to execute
            cancellation_event: Optional cancellation event

        Returns:
            A structured, self-contained ExecutionResult object.
        """

        # Setup session
        self._setup_session(initial_task)

        try:
            # Select strategy based on configuration
            await self._select_strategy(initial_task)

            # Initialize the active strategy for this task
            reasoning_context = ReasoningContext(
                current_strategy=self.strategy_manager.get_current_strategy_enum()
            )
            await self.strategy_manager.initialize_active_strategy(
                initial_task, reasoning_context
            )

            # Execute main loop
            result = await self._execute_loop(
                initial_task, cancellation_event, reasoning_context
            )

            # Prepare final result
            return await self._prepare_result(result)

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Execution failed: {e}")
            return await self._handle_error(e)

    def _setup_session(self, initial_task: str):
        """Setup session for task execution."""
        # Initialize session if needed
        if not self.context.session:
            from reactive_agents.core.types.session_types import AgentSession

            self.context.session = AgentSession(
                initial_task=initial_task,
                current_task=initial_task,
                task_status=TaskStatus.INITIALIZED,
            )

        # Reset for new task
        self.context.session.iterations = 0
        self.context.session.final_answer = None
        self.context.session.current_task = initial_task
        self.context.session.initial_task = initial_task
        self.context.session.task_status = TaskStatus.RUNNING
        self.context.session.start_time = time.time()

        # Reset strategy system
        self.strategy_manager.reset()

        # Reset context manager state
        self.context_manager.set_active_strategy(None)

        # Emit start event
        self.context.emit_event(
            AgentStateEvent.SESSION_STARTED,
            {
                "initial_task": initial_task,
                "session_id": self.context.session.session_id,
            },
        )

    async def _select_strategy(self, task: str):
        """Select execution strategy based on configuration."""
        # Check if dynamic strategy switching is enabled
        dynamic_switching = getattr(
            self.context, "enable_dynamic_strategy_switching", True
        )
        configured_strategy = getattr(
            self.context,
            "reasoning_strategy",
            "reactive",  # Changed default to reactive
        )

        # Convert strategy name to string if it's an enum
        if hasattr(configured_strategy, "value") and not isinstance(
            configured_strategy, str
        ):
            strategy_name = configured_strategy.value
        else:
            strategy_name = str(configured_strategy)

        # If dynamic switching is disabled, use configured strategy
        if not dynamic_switching:
            self.strategy_manager.set_strategy(strategy_name)
            self.context_manager.set_active_strategy(strategy_name)
            if self.agent_logger:
                self.agent_logger.info(
                    f"🎯 Using configured strategy: {strategy_name} (dynamic switching disabled)"
                )
            return

        # For reactive or unset strategy, use task classification
        if strategy_name in ["adaptive"]:
            if self.task_classifier:
                classification = await self.task_classifier.classify_task(task)
                # Create reasoning context for initialization
                reasoning_context = ReasoningContext(
                    current_strategy=self.strategy_manager.get_current_strategy_enum()
                )
                strategy = await self.strategy_manager.select_and_initialize_strategy(
                    classification, task, reasoning_context
                )

                self.context_manager.set_active_strategy(strategy)

                if self.agent_logger:
                    self.agent_logger.info(
                        f"📋 Task: {classification.task_type.value} "
                        f"(confidence: {classification.confidence:.2f})"
                    )
                    self.agent_logger.info(f"🎯 Selected strategy: {strategy}")
            else:
                # Fallback to reactive if no classifier
                self.strategy_manager.set_strategy("reactive")
                self.context_manager.set_active_strategy("reactive")
                if self.agent_logger:
                    self.agent_logger.warning(
                        "Using reactive strategy (no classifier available)"
                    )
        else:
            # Use configured strategy but allow dynamic switching
            self.strategy_manager.set_strategy(strategy_name)
            self.context_manager.set_active_strategy(strategy_name)
            if self.agent_logger:
                self.agent_logger.info(
                    f"🎯 Using configured strategy: {strategy_name} (dynamic switching enabled)"
                )

    async def _execute_loop(
        self,
        task: str,
        cancellation_event: Optional[asyncio.Event] = None,
        reasoning_context: Optional[ReasoningContext] = None,
    ) -> Dict[str, Any]:
        """Execute the main reasoning loop."""
        iteration_results = []
        max_iterations = self.context.max_iterations or 20

        # Use provided reasoning context or create one
        if reasoning_context is None:
            reasoning_context = ReasoningContext(
                current_strategy=self.strategy_manager.get_current_strategy_enum()
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

            # Handle pause state
            if self._paused:
                if self.agent_logger:
                    self.agent_logger.info("⏸️ Execution paused, waiting for resume...")
                await self._pause_event.wait()
                if self.agent_logger:
                    self.agent_logger.info("▶️ Execution resumed")

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

                # Append the structured result payload to the log
                iteration_results.append(strategy_result.payload.model_dump())

                # --- REFACTORED LOGIC ---
                # This pattern is type-safe and easy for linters to analyze.
                match strategy_result.payload:
                    case FinishTaskPayload() as payload:
                        self.context.session.final_answer = payload.final_answer
                        self.context.session.task_status = TaskStatus.COMPLETE
                        if self.agent_logger:
                            self.agent_logger.info(
                                "✅ Task completed with final answer."
                            )
                        break  # Exit the loop

                    case EvaluationPayload() as payload:
                        if self.agent_logger:
                            self.agent_logger.info(
                                f"🧠 Task evaluation: is_complete={payload.is_complete}, reason: {payload.reasoning}"
                            )
                        if payload.is_complete:
                            # The strategy thinks the task is done, but didn't provide
                            # the final answer yet. Nudge it to do so.
                            self.context_manager.add_nudge(
                                "Task evaluation indicates completion. Please provide the final answer now."
                            )
                        # Continue the loop
                        continue

                    case ErrorPayload() as payload:
                        if self.agent_logger:
                            self.agent_logger.error(
                                f"Strategy reported an error: {payload.error_message}"
                            )
                        self.context.session.add_error(
                            source="Strategy",
                            details={"message": payload.error_message},
                            is_critical=payload.is_critical(),
                        )
                        continue

                    case _:
                        # Default case for CONTINUE_THINKING, CALL_TOOLS, or other actions
                        # that just let the loop proceed. The strategy itself handles the
                        # tool execution and state updates.
                        pass

                # Update context
                reasoning_context.iteration_count = self.context.session.iterations

                # Check if the strategy signals to stop
                if not strategy_result.should_continue:
                    if self.context.session.final_answer:
                        if self.agent_logger:
                            self.agent_logger.info(
                                "🏁 Strategy completed with final answer"
                            )
                        break
                    else:
                        if self.agent_logger:
                            self.agent_logger.warning(
                                "Strategy requested stop but no final answer was provided."
                            )
                        self.context_manager.add_nudge(
                            "Give a complete final answer to the task using the final_answer tool"
                        )

                # Check for final answer (redundant check, but safe)
                if self.context.session.final_answer:
                    self.context.session.task_status = TaskStatus.COMPLETE
                    if self.agent_logger:
                        self.agent_logger.info("✅ Final answer provided")
                    break

                # Emit iteration complete
                self.context.emit_event(
                    AgentStateEvent.ITERATION_COMPLETED,
                    {
                        "iteration": self.context.session.iterations,
                        "result": strategy_result.payload.model_dump(),
                    },
                )

                # Summarize and prune context after each iteration
                self.context_manager.summarize_and_prune()

            except Exception as e:
                if self.agent_logger:
                    self.agent_logger.error(
                        f"Iteration {self.context.session.iterations} failed: {e}"
                    )
                self.context.session.add_error(
                    source="ExecutionEngineLoop",
                    details={"error": str(e), "traceback": traceback.format_exc()},
                    is_critical=True,
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
        # Check for session failure or completion
        if self.context.session.has_failed or self.context.session.final_answer:
            return False

        # Check terminal statuses
        if self.context.session.task_status in [
            TaskStatus.COMPLETE,
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

    async def _prepare_result(
        self, execution_details: Dict[str, Any]
    ) -> ExecutionResult:
        """Prepare final execution result."""
        session = self.context.session
        session.end_time = time.time()

        # Ensure final status is set correctly
        if session.final_answer and not session.has_failed:
            session.task_status = TaskStatus.COMPLETE
        elif session.task_status == TaskStatus.RUNNING:
            session.task_status = TaskStatus.ERROR

        # Get final metrics
        task_metrics = {}
        if self.context.metrics_manager:
            self.context.metrics_manager.finalize_run_metrics()
            task_metrics = self.context.metrics_manager.get_metrics()

        result = ExecutionResult(
            session=session,
            status=session.task_status,
            final_answer=session.final_answer,
            strategy_used=execution_details.get("strategy", "unknown"),
            execution_details=execution_details,
            task_metrics=task_metrics,
        )
        await result.generate_summary(self.engine)

        # Emit session end
        self.context.emit_event(
            AgentStateEvent.SESSION_ENDED,
            {
                "final_result": result.model_dump(),
                "session_id": result.session.session_id,
            },
        )

        if self.agent_logger:
            self.agent_logger.info(f"🎯 {result.to_pretty_string()}")

        return result

    async def _handle_error(self, error: Exception) -> ExecutionResult:
        """Handle execution errors."""
        self.context.session.add_error(
            source="ExecutionEngine",
            details={"error": str(error), "traceback": traceback.format_exc()},
            is_critical=True,
        )
        return await self._prepare_result({})

    # Control methods
    async def pause(self):
        """Pause execution."""
        # Emit pause requested event
        self.context.emit_event(
            AgentStateEvent.PAUSE_REQUESTED,
            {
                "session_id": self.context.session.session_id,
                "agent_name": self.context.agent_name,
                "task": self.context.session.current_task,
                "task_status": self.context.session.task_status.value,
                "iterations": self.context.session.iterations,
            },
        )

        self._paused = True
        self._pause_event.clear()

        # Emit paused event
        self.context.emit_event(
            AgentStateEvent.PAUSED,
            {
                "session_id": self.context.session.session_id,
                "agent_name": self.context.agent_name,
                "task": self.context.session.current_task,
                "task_status": self.context.session.task_status.value,
                "iterations": self.context.session.iterations,
            },
        )

    async def resume(self):
        """Resume execution."""
        # Emit resume requested event
        self.context.emit_event(
            AgentStateEvent.RESUME_REQUESTED,
            {
                "session_id": self.context.session.session_id,
                "agent_name": self.context.agent_name,
                "task": self.context.session.current_task,
                "task_status": self.context.session.task_status.value,
                "iterations": self.context.session.iterations,
            },
        )

        self._paused = False
        self._pause_event.set()

        # Emit resumed event
        self.context.emit_event(
            AgentStateEvent.RESUMED,
            {
                "session_id": self.context.session.session_id,
                "agent_name": self.context.agent_name,
                "task": self.context.session.current_task,
                "task_status": self.context.session.task_status.value,
                "iterations": self.context.session.iterations,
            },
        )

    async def terminate(self):
        """Terminate execution."""
        # Emit terminate requested event
        self.context.emit_event(
            AgentStateEvent.TERMINATE_REQUESTED,
            {
                "session_id": self.context.session.session_id,
                "agent_name": self.context.agent_name,
                "task": self.context.session.current_task,
                "task_status": self.context.session.task_status.value,
                "iterations": self.context.session.iterations,
            },
        )

        self._terminate_requested = True

        # Emit terminated event
        self.context.emit_event(
            AgentStateEvent.TERMINATED,
            {
                "session_id": self.context.session.session_id,
                "agent_name": self.context.agent_name,
                "task": self.context.session.current_task,
                "task_status": self.context.session.task_status.value,
                "iterations": self.context.session.iterations,
            },
        )

    async def stop(self):
        """Stop execution."""
        # Emit stop requested event
        self.context.emit_event(
            AgentStateEvent.STOP_REQUESTED,
            {
                "session_id": self.context.session.session_id,
                "agent_name": self.context.agent_name,
                "task": self.context.session.current_task,
                "task_status": self.context.session.task_status.value,
                "iterations": self.context.session.iterations,
            },
        )

        self._stop_requested = True

        # Emit stopped event
        self.context.emit_event(
            AgentStateEvent.STOPPED,
            {
                "session_id": self.context.session.session_id,
                "agent_name": self.context.agent_name,
                "task": self.context.session.current_task,
                "task_status": self.context.session.task_status.value,
                "iterations": self.context.session.iterations,
            },
        )

    # === Control State Queries ===
    def is_paused(self) -> bool:
        """Check if execution is currently paused."""
        return self._paused

    def is_terminating(self) -> bool:
        """Check if termination has been requested."""
        return self._terminate_requested

    def is_stopping(self) -> bool:
        """Check if stop has been requested."""
        return self._stop_requested

    def get_control_state(self) -> Dict[str, Any]:
        """Get current control state for monitoring."""
        return {
            "paused": self._paused,
            "terminate_requested": self._terminate_requested,
            "stop_requested": self._stop_requested,
            "pause_event_set": self._pause_event.is_set(),
        }
