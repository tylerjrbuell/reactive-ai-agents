from __future__ import annotations
import asyncio
import time
import traceback
import uuid
from typing import Dict, Any, Optional, TYPE_CHECKING
from reactive_agents.core.types.status_types import TaskStatus
from reactive_agents.core.types.event_types import AgentStateEvent
from reactive_agents.core.types.execution_types import ExecutionResult
from reactive_agents.core.types.reasoning_types import (
    ReasoningContext,
    FinishTaskPayload,
    EvaluationPayload,
    ErrorPayload,
)
from reactive_agents.core.reasoning.state_machine import (
    StrategyStateMachine,
    StrategyState,
    StateTransitionTrigger,
)
from reactive_agents.core.reasoning.recovery import (
    ErrorRecoveryOrchestrator,
    ErrorContext,
)
from reactive_agents.core.reasoning.performance_monitor import (
    StrategyPerformanceMonitor,
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

        # Initialize core systems
        self._initialize_core_systems()

        # Control state
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._terminate_requested = False
        self._stop_requested = False

        if self.agent_logger:
            self.agent_logger.info(
                "🚀 ExecutionEngine | Initialized and ready"
            )

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

    def _initialize_core_systems(self):
        """Initialize core architectural components."""
        # State machine for formal state transitions
        self.state_machine = StrategyStateMachine(StrategyState.INITIALIZING)

        # Error recovery orchestrator for intelligent error handling
        self.error_recovery = ErrorRecoveryOrchestrator()

        # Performance monitor for strategy optimization
        self.performance_monitor = StrategyPerformanceMonitor()

        # Current execution tracking
        self.current_execution_id: str = str(uuid.uuid4())

        if self.agent_logger:
            self.agent_logger.info(
                "🏗️  Core systems initialized | State machine, error recovery, performance monitoring ready"
            )

    async def execute(
        self,
        initial_task: str,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> ExecutionResult:
        """
        Execute a task from start to finish with modern architecture.

        Args:
            initial_task: The task to execute
            cancellation_event: Optional cancellation event

        Returns:
            A structured, self-contained ExecutionResult object.
        """

        # Initialize state machine
        await self.state_machine.transition_to(
            StrategyState.INITIALIZING,
            StateTransitionTrigger.START_EXECUTION,
            {"task": initial_task, "execution_id": self.current_execution_id},
        )

        try:
            # Setup session
            self._setup_session(initial_task)

            # Transition to planning state
            await self.state_machine.transition_to(
                StrategyState.PLANNING, StateTransitionTrigger.START_EXECUTION
            )

            # Select strategy based on configuration and performance data
            await self._select_strategy(initial_task)

            # Start performance tracking
            self.performance_monitor.start_execution_tracking(
                self.current_execution_id,
                self.strategy_manager.get_current_strategy_name(),
                initial_task,
                {"session_id": self.context.session.session_id},
            )

            # Initialize the active strategy for this task
            reasoning_context = ReasoningContext(
                current_strategy=self.strategy_manager.get_current_strategy_enum()
            )
            await self.strategy_manager.initialize_active_strategy(
                initial_task, reasoning_context
            )

            # Transition to execution state
            await self.state_machine.transition_to(
                StrategyState.EXECUTING, StateTransitionTrigger.PLANNING_COMPLETE
            )

            # Execute main reasoning loop
            result = await self._execute_loop(
                initial_task, cancellation_event, reasoning_context
            )

            # Complete performance tracking
            completion_score = 1.0 if result.get("final_answer") else 0.0
            self.performance_monitor.complete_execution_tracking(
                self.current_execution_id,
                success=bool(result.get("final_answer")),
                completion_score=completion_score,
                final_metadata={
                    "iterations": result.get("total_iterations", 0),
                    "strategy": result.get("strategy", "unknown"),
                },
            )

            # Transition to completion state
            await self.state_machine.transition_to(
                StrategyState.COMPLETING, StateTransitionTrigger.TASK_COMPLETE
            )

            # Prepare final result
            return await self._prepare_result(result)

        except Exception as e:
            # Handle errors with recovery
            await self._handle_error(e, initial_task)
            return await self._prepare_result({})

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
        """Select execution strategy using performance data and task classification."""
        # Check if dynamic strategy switching is enabled
        dynamic_switching = getattr(
            self.context, "enable_dynamic_strategy_switching", True
        )
        configured_strategy = getattr(
            self.context,
            "reasoning_strategy",
            "reactive",
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

        # Strategy selection with performance consideration
        if strategy_name == "adaptive" and dynamic_switching:
            # Check if we should switch based on performance
            current_strategy = self.strategy_manager.get_current_strategy_name()
            if current_strategy:
                recommended_switch = self.performance_monitor.should_switch_strategy(
                    current_strategy
                )
                if recommended_switch:
                    if self.agent_logger:
                        self.agent_logger.info(
                            f"🔄 Performance monitor recommends switching from {current_strategy} to {recommended_switch}"
                        )
                    strategy_name = recommended_switch

            # Use task classification with performance data
            if self.task_classifier:
                classification = await self.task_classifier.classify_task(task)

                # Get performance rankings to inform selection
                strategy_rankings = self.performance_monitor.get_strategy_rankings()

                reasoning_context = ReasoningContext(
                    current_strategy=self.strategy_manager.get_current_strategy_enum()
                )

                strategy = await self.strategy_manager.select_and_initialize_strategy(
                    classification, task, reasoning_context
                )

                # Consider performance data in final selection
                if strategy_rankings and len(strategy_rankings) > 1:
                    # If selected strategy is performing poorly, consider alternatives
                    selected_performance = next(
                        (
                            score
                            for name, score in strategy_rankings
                            if name == strategy
                        ),
                        0.5,
                    )

                    if selected_performance < 0.4 and strategy_rankings[0][1] > 0.7:
                        # Switch to best performing strategy if current is poor
                        best_strategy = strategy_rankings[0][0]
                        if self.agent_logger:
                            self.agent_logger.info(
                                f"🎯 Overriding selection: {strategy} -> {best_strategy} (performance-based)"
                            )
                        strategy = best_strategy

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
                        "⚠️  ExecutionEngine | Using reactive strategy (no classifier available)"
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
        """Execute the main reasoning loop with state management and error recovery."""
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
                    self.agent_logger.info("🛑 ExecutionEngine | Task cancelled by external signal")
                break

            # Check control signals
            if self._check_control_signals():
                break

            # Handle pause state
            if self._paused:
                await self.state_machine.transition_to(
                    StrategyState.PAUSED, StateTransitionTrigger.PAUSE_REQUESTED
                )
                if self.agent_logger:
                    self.agent_logger.info("⏸️  ExecutionEngine | Paused, waiting for resume signal")
                await self._pause_event.wait()
                await self.state_machine.transition_to(
                    StrategyState.EXECUTING, StateTransitionTrigger.RESUME_REQUESTED
                )
                if self.agent_logger:
                    self.agent_logger.info("▶️  ExecutionEngine | Resumed execution")

            self.context.session.iterations += 1
            # Update performance monitoring
            self.performance_monitor.update_execution_progress(
                self.current_execution_id,
                iterations=self.context.session.iterations,
            )

            if self.agent_logger:
                self.agent_logger.info(
                    f"🔄 ExecutionEngine | Iteration {self.context.session.iterations}/{max_iterations} | "
                    f"Strategy: {self.strategy_manager.get_current_strategy_name()}"
                )

            # Emit iteration start
            self.context.emit_event(
                AgentStateEvent.ITERATION_STARTED,
                {
                    "iteration": self.context.session.iterations,
                    "max_iterations": max_iterations,
                    "strategy": self.strategy_manager.get_current_strategy_name(),
                    "state": self.state_machine.current_state.value,
                },
            )

            try:
                # Execute one iteration using current strategy
                strategy_result = await self.strategy_manager.execute_iteration(
                    task, reasoning_context
                )

                # Append the structured result payload to the log
                iteration_results.append(strategy_result.payload.model_dump())

                # Result processing with state transitions
                match strategy_result.payload:
                    case FinishTaskPayload() as payload:
                        await self.state_machine.transition_to(
                            StrategyState.COMPLETING,
                            StateTransitionTrigger.TASK_COMPLETE,
                        )
                        self.context.session.final_answer = payload.final_answer
                        self.context.session.task_status = TaskStatus.COMPLETE
                        if self.agent_logger:
                            self.agent_logger.info(
                                "✅ ExecutionEngine | Task completed successfully"
                            )
                        break

                    case EvaluationPayload() as payload:
                        await self.state_machine.transition_to(
                            StrategyState.EVALUATING,
                            StateTransitionTrigger.EXECUTION_STEP_COMPLETE,
                        )
                        if self.agent_logger:
                            self.agent_logger.info(
                                f"🧠 ExecutionEngine | Task evaluation: complete={payload.is_complete} | "
                                f"Reasoning: {payload.reasoning[:100]}{'...' if len(payload.reasoning) > 100 else ''}"
                            )
                        if payload.is_complete:
                            self.context_manager.add_nudge(
                                "Task evaluation indicates completion. Please provide the final answer now."
                            )
                        await self.state_machine.transition_to(
                            StrategyState.EXECUTING,
                            StateTransitionTrigger.EVALUATION_COMPLETE,
                        )
                        continue

                    case ErrorPayload() as payload:
                        await self._handle_strategy_error(payload, task)
                        continue

                    case _:
                        # Default case for CONTINUE_THINKING, CALL_TOOLS, etc.
                        await self.state_machine.transition_to(
                            StrategyState.REFLECTING,
                            StateTransitionTrigger.EXECUTION_STEP_COMPLETE,
                        )
                        await self.state_machine.transition_to(
                            StrategyState.EXECUTING,
                            StateTransitionTrigger.REFLECTION_COMPLETE,
                        )

                # Update context
                reasoning_context.iteration_count = self.context.session.iterations

                # Check if the strategy signals to stop
                if not strategy_result.should_continue:
                    if self.context.session.final_answer:
                        if self.agent_logger:
                            self.agent_logger.info(
                                "🏁 ExecutionEngine | Strategy signaled completion with final answer"
                            )
                        break
                    else:
                        if self.agent_logger:
                            self.agent_logger.warning(
                                "⚠️  ExecutionEngine | Strategy requested stop but no final answer provided"
                            )
                        self.context_manager.add_nudge(
                            "Give a complete final answer to the task using the final_answer tool"
                        )

                # Check for final answer
                if self.context.session.final_answer:
                    self.context.session.task_status = TaskStatus.COMPLETE
                    if self.agent_logger:
                        self.agent_logger.info("✅ ExecutionEngine | Final answer provided")
                    break

                # Emit iteration complete
                self.context.emit_event(
                    AgentStateEvent.ITERATION_COMPLETED,
                    {
                        "iteration": self.context.session.iterations,
                        "result": strategy_result.payload.model_dump(),
                        "state": self.state_machine.current_state.value,
                    },
                )

                # Summarize and prune context after each iteration
                self.context_manager.summarize_and_prune()

            except Exception as e:
                # Error handling with recovery
                await self._handle_iteration_error(e, task, iteration_results)

        return {
            "iterations": iteration_results,
            "total_iterations": self.context.session.iterations,
            "final_answer": self.context.session.final_answer,
            "strategy": self.strategy_manager.get_current_strategy_name(),
            "state_history": self.state_machine.get_state_history(),
        }

    async def _handle_strategy_error(self, error_payload: ErrorPayload, task: str):
        """Handle errors reported by strategies."""
        await self.state_machine.transition_to(
            StrategyState.ERROR_RECOVERY, StateTransitionTrigger.ERROR_OCCURRED
        )

        if self.agent_logger:
            self.agent_logger.error(
                f"❌ ExecutionEngine | Strategy error: {error_payload.error_message}"
            )

        # Create error context
        error_context = ErrorContext(
            task=task,
            component="Strategy",
            operation="execute_iteration",
            iteration=self.context.session.iterations,
            strategy=self.strategy_manager.get_current_strategy_name(),
            error_message=error_payload.error_message,
            error_type="StrategyError",
        )

        # Use error recovery orchestrator
        recovery_result = await self.error_recovery.handle_error(
            Exception(error_payload.error_message), error_context
        )

        # Update performance monitoring
        self.performance_monitor.update_execution_progress(
            self.current_execution_id,
            error_count=len(self.context.session.errors),
            error_message=error_payload.error_message,
        )

        # Record error in session
        self.context.session.add_error(
            source="Strategy",
            details={"message": error_payload.error_message},
            is_critical=error_payload.is_critical(),
        )

        # Apply recovery action if recommended
        if (
            recovery_result.should_switch_strategy
            and recovery_result.recommended_strategy
        ):
            if self.agent_logger:
                self.agent_logger.info(
                    f"🔄 ExecutionEngine | Switching strategy due to error: {recovery_result.recommended_strategy}"
                )
            self.strategy_manager.set_strategy(recovery_result.recommended_strategy)

        await self.state_machine.transition_to(
            StrategyState.EXECUTING, StateTransitionTrigger.RECOVERY_COMPLETE
        )

    async def _handle_iteration_error(
        self, error: Exception, task: str, iteration_results: list
    ):
        """Handle errors that occur during iteration execution."""
        await self.state_machine.transition_to(
            StrategyState.ERROR_RECOVERY, StateTransitionTrigger.ERROR_OCCURRED
        )

        if self.agent_logger:
            self.agent_logger.error(
                f"❌ ExecutionEngine | Iteration {self.context.session.iterations} failed: {error}"
            )

        # Create error context
        error_context = ErrorContext(
            task=task,
            component="ExecutionEngineLoop",
            operation="execute_iteration",
            iteration=self.context.session.iterations,
            strategy=self.strategy_manager.get_current_strategy_name(),
            error_message=str(error),
            error_type=type(error).__name__,
            stack_trace=traceback.format_exc(),
        )

        # Use error recovery orchestrator
        await self.error_recovery.handle_error(error, error_context)

        # Update performance monitoring
        self.performance_monitor.update_execution_progress(
            self.current_execution_id,
            error_count=len(self.context.session.errors),
            error_message=str(error),
        )

        # Record error in session
        self.context.session.add_error(
            source="ExecutionEngineLoop",
            details={"error": str(error), "traceback": traceback.format_exc()},
            is_critical=True,
        )

        # Learn from recovery attempt
        if self.context.session.errors:
            # This would be updated when we know if recovery succeeded
            pass

        await self.state_machine.transition_to(
            StrategyState.EXECUTING, StateTransitionTrigger.RECOVERY_COMPLETE
        )

    async def _handle_error(self, error: Exception, task: str):
        """Handle execution errors with state machine and recovery."""
        await self.state_machine.transition_to(
            StrategyState.ERROR_RECOVERY, StateTransitionTrigger.ERROR_OCCURRED
        )

        if self.agent_logger:
            self.agent_logger.error(f"❌ ExecutionEngine | Execution failed: {error}")

        # Create error context
        error_context = ErrorContext(
            task=task,
            component="ExecutionEngine",
            operation="execute",
            iteration=self.context.session.iterations if self.context.session else 0,
            strategy=self.strategy_manager.get_current_strategy_name(),
            error_message=str(error),
            error_type=type(error).__name__,
            stack_trace=traceback.format_exc(),
        )

        # Use error recovery orchestrator
        await self.error_recovery.handle_error(error, error_context)

        # Complete performance tracking with failure
        self.performance_monitor.complete_execution_tracking(
            self.current_execution_id,
            success=False,
            completion_score=0.0,
            final_metadata={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

        # Record error in session
        self.context.session.add_error(
            source="ExecutionEngine",
            details={"error": str(error), "traceback": traceback.format_exc()},
            is_critical=True,
        )

        # Transition to failed state
        await self.state_machine.transition_to(
            StrategyState.FAILED, StateTransitionTrigger.TERMINATION_REQUESTED
        )


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
                self.agent_logger.info("🛑 ExecutionEngine | Termination requested")
            self.context.session.task_status = TaskStatus.CANCELLED
            return True

        if self._stop_requested:
            if self.agent_logger:
                self.agent_logger.info("⏹️ ExecutionEngine | Stop requested")
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

        # Calculate and update session scores
        self._update_session_scores(session, execution_details)

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

    def _update_session_scores(self, session, execution_details: Dict[str, Any]) -> None:
        """Update session scoring fields based on execution results."""
        
        # Calculate completion score
        if session.final_answer and session.task_status == TaskStatus.COMPLETE:
            session.completion_score = 1.0
        elif session.final_answer:
            session.completion_score = 0.8  # Has answer but maybe not complete
        else:
            session.completion_score = 0.0

        # Calculate tool usage score
        total_iterations = execution_details.get("total_iterations", session.iterations)
        if total_iterations > 0:
            # Higher score for successful tool usage, lower for errors
            error_count = len(session.errors)
            tool_efficiency = max(0.0, 1.0 - (error_count / max(1, total_iterations * 2)))
            session.tool_usage_score = tool_efficiency
        else:
            session.tool_usage_score = 0.0

        # Calculate progress score based on completion and iterations
        max_iterations = self.context.max_iterations or 20
        iteration_efficiency = 1.0 - (session.iterations / max_iterations)
        if session.task_status == TaskStatus.COMPLETE:
            session.progress_score = min(1.0, 0.5 + iteration_efficiency * 0.5)
        elif session.iterations > 0:
            session.progress_score = min(0.7, iteration_efficiency * 0.7)
        else:
            session.progress_score = 0.0

        # Calculate answer quality score
        if session.final_answer:
            answer_length = len(session.final_answer)
            if answer_length > 50:  # Substantial answer
                session.answer_quality_score = 0.9
            elif answer_length > 10:  # Basic answer
                session.answer_quality_score = 0.7
            else:  # Minimal answer
                session.answer_quality_score = 0.5
        else:
            session.answer_quality_score = 0.0

        # Calculate LLM evaluation score (based on success and errors)
        if session.task_status == TaskStatus.COMPLETE and not session.errors:
            session.llm_evaluation_score = 1.0
        elif session.task_status == TaskStatus.COMPLETE:
            session.llm_evaluation_score = 0.8
        elif session.final_answer and not session.has_failed:
            session.llm_evaluation_score = 0.6
        else:
            session.llm_evaluation_score = 0.3

        if self.agent_logger:
            self.agent_logger.info(
                f"📊 ExecutionEngine | Final Scores - "
                f"Completion: {session.completion_score:.2f}, "
                f"Tool Usage: {session.tool_usage_score:.2f}, "
                f"Progress: {session.progress_score:.2f}, "
                f"Answer Quality: {session.answer_quality_score:.2f}, "
                f"Overall: {session.overall_score:.2f}"
            )

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
