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
        from reactive_agents.core.reasoning.engine import (
            get_reasoning_engine,
        )

        # Get shared engine
        self.engine = get_reasoning_engine(self.context)

        # Initialize managers
        self.strategy_manager = StrategyManager(self.context)
        self.task_classifier = TaskClassifier(self.context)
        self.context_manager = self.engine.get_context_manager()

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

        # Reset context manager state
        self.context_manager.set_active_strategy(None)

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
        if strategy_name in ["reactive", "adaptive"]:
            if self.task_classifier:
                classification = await self.task_classifier.classify_task(task)
                strategy = self.strategy_manager.select_optimal_strategy(classification)

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

                # Handle task completion
                if iteration_result.get("action_taken") == "task_completed":
                    final_answer = iteration_result.get("final_answer")
                    completion_status = iteration_result.get("status", "unknown")
                    evaluation = iteration_result.get("evaluation", {})

                    if self.agent_logger:
                        self.agent_logger.debug(
                            f"Task completion status: {completion_status}"
                        )
                        self.agent_logger.debug(
                            f"Final answer present: {bool(final_answer)}"
                        )
                        self.agent_logger.debug(f"Evaluation result: {evaluation}")

                    # Check if task is actually complete based on evaluation
                    is_complete = evaluation.get("is_complete", False)
                    if is_complete and final_answer:
                        self.context.session.final_answer = final_answer
                        self.context.session.task_status = TaskStatus.COMPLETE
                        if self.agent_logger:
                            self.agent_logger.info(
                                "✅ Task completed with final answer"
                            )
                        break
                    elif not is_complete:
                        # Task not complete according to evaluation
                        if self.agent_logger:
                            self.agent_logger.warning(
                                f"Task evaluation indicates incomplete: {evaluation.get('reasoning', 'No reason provided')}"
                            )
                        # Update iteration result with evaluation
                        iteration_result["evaluation"] = evaluation
                        # Don't break - let strategy continue
                        continue
                    else:
                        # Complete but no final answer
                        if self.agent_logger:
                            self.agent_logger.warning(
                                "Task evaluation indicates complete but missing final answer"
                            )
                        # Update iteration result with evaluation
                        iteration_result["evaluation"] = evaluation
                        # Don't break - let strategy try to generate final answer
                        continue

                # Update context
                reasoning_context.iteration_count = self.context.session.iterations

                # Check if should continue
                if not iteration_result.get("should_continue", True):
                    # Only break if we have a final answer
                    if self.context.session.final_answer:
                        if self.agent_logger:
                            self.agent_logger.info(
                                "🏁 Strategy completed with final answer"
                            )
                        break
                    else:
                        if self.agent_logger:
                            self.agent_logger.warning(
                                "Strategy wants to complete but no final answer - continuing"
                            )

                # Check for final answer
                if (
                    self.context.session.final_answer
                    and self.context.session.task_status == TaskStatus.COMPLETE
                ):
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

                # Summarize and prune context after each iteration
                self.context_manager.summarize_and_prune()

            except Exception as e:
                if self.agent_logger:
                    self.agent_logger.error(
                        f"Iteration {self.context.session.iterations} failed: {e}"
                    )

                # Set error status on exception
                self.context.session.task_status = TaskStatus.ERROR
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

        # Get task metrics from context
        task_metrics = {}
        if self.context.metrics_manager:
            # Finalize metrics before getting them
            self.context.metrics_manager.finalize_run_metrics()
            task_metrics = self.context.metrics_manager.get_metrics()

        # Get session data for additional context
        session_data = {}
        if self.context.session:
            session_data = {
                "session_id": getattr(self.context.session, "session_id", "unknown"),
                "iterations": self.context.session.iterations,
                "successful_tools": (
                    list(self.context.session.successful_tools)
                    if self.context.session.successful_tools
                    else []
                ),
                "task_status": (
                    str(self.context.session.task_status)
                    if self.context.session.task_status
                    else "unknown"
                ),
                "final_answer": self.context.session.final_answer,
                "completion_score": self.context.session.completion_score,
            }

        result = {
            "status": self.context.session.task_status.value,
            "final_answer": self.context.session.final_answer,
            "completion_score": completion_score,
            "iterations": self.context.session.iterations,
            "strategy": execution_result.get("strategy", "unknown"),
            "execution_details": execution_result,
            "session_id": getattr(self.context.session, "session_id", "unknown"),
            "task_metrics": task_metrics,
            "session_data": session_data,
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
