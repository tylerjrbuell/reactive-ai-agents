"""
Tests for ExecutionEngine.

Tests the core execution engine functionality including task execution,
context management, and engine lifecycle.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock, ANY
from reactive_agents.core.engine.execution_engine import ExecutionEngine
from reactive_agents.core.context.agent_context import AgentContext
from reactive_agents.core.types.status_types import TaskStatus
from reactive_agents.core.types.event_types import AgentStateEvent
from reactive_agents.core.types.execution_types import ExecutionResult
from reactive_agents.core.types.reasoning_types import (
    ReasoningContext,
    FinishTaskPayload,
    EvaluationPayload,
    ErrorPayload,
    ContinueThinkingPayload,
    StrategyAction,
    ReasoningStrategies,
)
from reactive_agents.core.types.strategy_types import StrategyResult
from reactive_agents.core.reasoning.state_machine import StrategyState
from reactive_agents.core.types.session_types import AgentSession


class TestExecutionEngine:
    """Test cases for ExecutionEngine."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock session."""
        session = Mock(spec=AgentSession)
        session.session_id = "test-session-123"
        session.iterations = 0
        session.final_answer = None
        session.current_task = "test task"
        session.initial_task = "test task"
        session.task_status = TaskStatus.INITIALIZED
        session.start_time = 1234567890.0
        session.end_time = None
        session.errors = []
        session.has_failed = False
        session.add_error = Mock()
        session.completion_score = 0.0
        session.tool_usage_score = 0.0
        session.progress_score = 0.0
        session.answer_quality_score = 0.0
        session.llm_evaluation_score = 0.0
        session.overall_score = 0.0
        return session

    @pytest.fixture
    def mock_context(self, mock_session):
        """Create a mock agent context."""
        context = Mock(spec=AgentContext)
        context.session = mock_session
        context.reasoning_engine = Mock()
        context.max_iterations = 20
        context.agent_name = "test-agent"
        context.enable_dynamic_strategy_switching = True
        context.reasoning_strategy = "reactive"
        context.emit_event = Mock()
        context.metrics_manager = Mock()
        context.metrics_manager.finalize_run_metrics = Mock()
        context.metrics_manager.get_metrics = Mock(return_value={})

        # Mock the context manager
        context.reasoning_engine.get_context_manager = Mock()
        mock_context_manager = Mock()
        mock_context_manager.set_active_strategy = Mock()
        mock_context_manager.add_nudge = Mock()
        mock_context_manager.summarize_and_prune = Mock()
        context.reasoning_engine.get_context_manager.return_value = mock_context_manager

        return context

    @pytest.fixture
    def mock_agent(self, mock_context):
        """Create a mock agent."""
        agent = Mock()
        agent.context = mock_context
        agent.agent_logger = Mock()
        agent.agent_logger.info = Mock()
        agent.agent_logger.warning = Mock()
        agent.agent_logger.error = Mock()
        return agent

    @pytest.fixture
    def mock_strategy_manager(self):
        """Create a mock strategy manager."""
        manager = Mock()
        manager.get_current_strategy_name = Mock(return_value="reactive")
        manager.get_current_strategy_enum = Mock(
            return_value=ReasoningStrategies.REACTIVE
        )
        manager.set_strategy = Mock()
        manager.reset = Mock()
        manager.initialize_active_strategy = AsyncMock()
        manager.execute_iteration = AsyncMock()
        manager.select_and_initialize_strategy = AsyncMock(return_value="reactive")
        return manager

    @pytest.fixture
    def mock_task_classifier(self):
        """Create a mock task classifier."""
        classifier = Mock()
        mock_classification = Mock()
        mock_classification.task_type = Mock()
        mock_classification.task_type.value = "general"
        mock_classification.confidence = 0.8
        classifier.classify_task = AsyncMock(return_value=mock_classification)
        return classifier

    @pytest.fixture
    def mock_state_machine(self):
        """Create a mock state machine."""
        machine = Mock()
        machine.current_state = StrategyState.INITIALIZING
        machine.transition_to = AsyncMock()
        machine.get_state_history = Mock(return_value=[])
        return machine

    @pytest.fixture
    def mock_error_recovery(self):
        """Create a mock error recovery orchestrator."""
        recovery = Mock()
        recovery_result = Mock()
        recovery_result.should_switch_strategy = False
        recovery_result.recommended_strategy = None
        recovery.handle_error = AsyncMock(return_value=recovery_result)
        return recovery

    @pytest.fixture
    def mock_performance_monitor(self):
        """Create a mock performance monitor."""
        monitor = Mock()
        monitor.start_execution_tracking = Mock()
        monitor.update_execution_progress = Mock()
        monitor.complete_execution_tracking = Mock()
        monitor.should_switch_strategy = Mock(return_value=None)
        monitor.get_strategy_rankings = Mock(return_value=[])
        return monitor

    @pytest.fixture
    def execution_engine(
        self,
        mock_agent,
        mock_strategy_manager,
        mock_task_classifier,
        mock_state_machine,
        mock_error_recovery,
        mock_performance_monitor,
    ):
        """Create an execution engine instance with mocked dependencies."""
        with patch(
            "reactive_agents.core.reasoning.strategy_manager.StrategyManager",
            return_value=mock_strategy_manager,
        ), patch(
            "reactive_agents.core.reasoning.task_classifier.TaskClassifier",
            return_value=mock_task_classifier,
        ), patch(
            "reactive_agents.core.reasoning.state_machine.StrategyStateMachine",
            return_value=mock_state_machine,
        ), patch(
            "reactive_agents.core.reasoning.recovery.ErrorRecoveryOrchestrator",
            return_value=mock_error_recovery,
        ), patch(
            "reactive_agents.core.reasoning.performance_monitor.StrategyPerformanceMonitor",
            return_value=mock_performance_monitor,
        ):

            engine = ExecutionEngine(agent=mock_agent)

            # Inject mocks for easier testing
            engine.strategy_manager = mock_strategy_manager
            engine.task_classifier = mock_task_classifier
            engine.state_machine = mock_state_machine
            engine.error_recovery = mock_error_recovery
            engine.performance_monitor = mock_performance_monitor

            return engine

    def test_initialization(self, execution_engine, mock_agent):
        """Test execution engine initialization."""
        assert execution_engine.agent == mock_agent
        assert execution_engine.context == mock_agent.context
        assert execution_engine.agent_logger == mock_agent.agent_logger
        assert not execution_engine._paused
        assert execution_engine._pause_event.is_set()
        assert not execution_engine._terminate_requested
        assert not execution_engine._stop_requested

    def test_setup_session(self, execution_engine):
        """Test session setup."""
        task = "Test task"
        execution_engine._setup_session(task)

        # Verify session was configured
        session = execution_engine.context.session
        assert session.current_task == task
        assert session.initial_task == task
        assert session.task_status == TaskStatus.RUNNING
        assert session.iterations == 0
        assert session.final_answer is None

        # Verify strategy manager was reset
        execution_engine.strategy_manager.reset.assert_called_once()

        # Verify context manager was configured
        execution_engine.context_manager.set_active_strategy.assert_called_once_with(
            None
        )

        # Verify event was emitted
        execution_engine.context.emit_event.assert_called_with(
            AgentStateEvent.SESSION_STARTED,
            {
                "initial_task": task,
                "session_id": session.session_id,
            },
        )

    @pytest.mark.asyncio
    async def test_strategy_selection_configured(self, execution_engine):
        """Test strategy selection with configured strategy."""
        task = "Test task"
        execution_engine.context.enable_dynamic_strategy_switching = False
        execution_engine.context.reasoning_strategy = "plan_execute_reflect"

        await execution_engine._select_strategy(task)

        execution_engine.strategy_manager.set_strategy.assert_called_once_with(
            "plan_execute_reflect"
        )
        execution_engine.context_manager.set_active_strategy.assert_called_once_with(
            "plan_execute_reflect"
        )

    @pytest.mark.asyncio
    async def test_strategy_selection_adaptive(self, execution_engine):
        """Test adaptive strategy selection."""
        task = "Complex research task"
        execution_engine.context.reasoning_strategy = "adaptive"

        await execution_engine._select_strategy(task)

        # Verify task classification was called
        execution_engine.task_classifier.classify_task.assert_called_once_with(task)

        # Verify strategy selection was called
        execution_engine.strategy_manager.select_and_initialize_strategy.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_simple_completion(self, execution_engine):
        """Test successful task execution with immediate completion."""
        task = "Simple test task"

        # Mock session creation
        with patch.object(execution_engine.context, "session", None):
            with patch(
                "reactive_agents.core.types.session_types.AgentSession"
            ) as mock_session_class:
                mock_session = Mock(spec=AgentSession)
                mock_session.session_id = "test-session"
                mock_session.iterations = 1
                mock_session.final_answer = "Test answer"
                mock_session.task_status = TaskStatus.COMPLETE
                mock_session.has_failed = False
                mock_session.errors = []
                mock_session.start_time = 1234567890.0
                mock_session.end_time = None
                mock_session.completion_score = 1.0
                mock_session.tool_usage_score = 0.8
                mock_session.progress_score = 0.9
                mock_session.answer_quality_score = 0.9
                mock_session.llm_evaluation_score = 1.0
                mock_session.overall_score = 0.92
                mock_session_class.return_value = mock_session
                execution_engine.context.session = mock_session

        # Mock successful strategy execution with completion
        finish_payload = FinishTaskPayload(
            final_answer="Test answer",
            action=StrategyAction.FINISH_TASK,
            evaluation=EvaluationPayload(
                action=StrategyAction.EVALUATE_COMPLETION,
                is_complete=True,
                reasoning="Task is complete",
                confidence=0.9,
            ),
        )
        strategy_result = StrategyResult(
            action="finish_task", payload=finish_payload, should_continue=False
        )
        execution_engine.strategy_manager.execute_iteration.return_value = (
            strategy_result
        )

        # Mock execution result preparation
        with patch.object(execution_engine, "_prepare_result") as mock_prepare:
            mock_result = Mock(spec=ExecutionResult)
            mock_prepare.return_value = mock_result

            result = await execution_engine.execute(task)

            # Verify result was prepared
            assert result == mock_result

            # Verify state transitions occurred
            assert execution_engine.state_machine.transition_to.call_count >= 3

            # Verify strategy was initialized
            execution_engine.strategy_manager.initialize_active_strategy.assert_called_once()

            # Verify iteration was executed
            execution_engine.strategy_manager.execute_iteration.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_multiple_iterations(self, execution_engine):
        """Test execution with multiple iterations before completion."""
        task = "Complex test task"

        # Mock session
        session = execution_engine.context.session
        session.final_answer = None
        session.has_failed = False
        session.task_status = TaskStatus.RUNNING

        # Mock multiple iterations then completion
        continue_payload = ContinueThinkingPayload(
            reasoning="Need more work", action=StrategyAction.CONTINUE_THINKING
        )
        continue_result = StrategyResult(
            action="continue_thinking", payload=continue_payload, should_continue=True
        )

        finish_payload = FinishTaskPayload(
            final_answer="Final answer",
            action=StrategyAction.FINISH_TASK,
            evaluation=EvaluationPayload(
                action=StrategyAction.EVALUATE_COMPLETION,
                is_complete=True,
                reasoning="Task is complete",
                confidence=0.9,
            ),
        )
        finish_result = StrategyResult(
            action="finish_task", payload=finish_payload, should_continue=False
        )

        execution_engine.strategy_manager.execute_iteration.side_effect = [
            continue_result,
            continue_result,
            finish_result,
        ]

        # Mock execution result
        with patch.object(execution_engine, "_prepare_result") as mock_prepare:
            mock_result = Mock(spec=ExecutionResult)
            mock_prepare.return_value = mock_result

            result = await execution_engine.execute(task)

            # Verify multiple iterations occurred
            assert execution_engine.strategy_manager.execute_iteration.call_count == 3

            # Verify final answer was set
            assert session.final_answer == "Final answer"
            assert session.task_status == TaskStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_execute_with_error_handling(self, execution_engine):
        """Test execution with error handling."""
        task = "Test task"

        # Mock strategy execution error
        test_error = Exception("Test error")
        execution_engine.strategy_manager.execute_iteration.side_effect = test_error

        # Mock error result preparation
        with patch.object(execution_engine, "_prepare_result") as mock_prepare:
            mock_result = Mock(spec=ExecutionResult)
            mock_prepare.return_value = mock_result

            result = await execution_engine.execute(task)

            # Verify error recovery was called
            execution_engine.error_recovery.handle_error.assert_called()

            # Verify result was still prepared
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_execute_with_strategy_error_payload(self, execution_engine):
        """Test execution with strategy reporting an error."""
        task = "Test task"

        # Mock session
        session = execution_engine.context.session
        session.errors = []

        # Mock strategy error payload
        error_payload = ErrorPayload(
            error_message="Strategy error", action=StrategyAction.ERROR
        )
        error_result = StrategyResult(
            action="error", payload=error_payload, should_continue=True
        )

        finish_payload = FinishTaskPayload(
            final_answer="Recovered answer",
            action=StrategyAction.FINISH_TASK,
            evaluation=EvaluationPayload(
                action=StrategyAction.EVALUATE_COMPLETION,
                is_complete=True,
                reasoning="Task is complete",
                confidence=0.9,
            ),
        )
        finish_result = StrategyResult(
            action="finish_task", payload=finish_payload, should_continue=False
        )

        execution_engine.strategy_manager.execute_iteration.side_effect = [
            error_result,
            finish_result,
        ]

        with patch.object(execution_engine, "_prepare_result") as mock_prepare:
            mock_result = Mock(spec=ExecutionResult)
            mock_prepare.return_value = mock_result

            result = await execution_engine.execute(task)

            # Verify error was handled
            execution_engine.error_recovery.handle_error.assert_called()

            # Verify session error was recorded
            session.add_error.assert_called()

    @pytest.mark.asyncio
    async def test_execute_with_evaluation_payload(self, execution_engine):
        """Test execution with task evaluation."""
        task = "Test task"

        # Mock evaluation payload
        eval_payload = EvaluationPayload(
            action=StrategyAction.EVALUATE_COMPLETION,
            is_complete=True,
            reasoning="Task is complete",
            confidence=0.9,
        )
        eval_result = StrategyResult(
            action="evaluation", payload=eval_payload, should_continue=True
        )

        finish_payload = FinishTaskPayload(
            final_answer="Final answer",
            action=StrategyAction.FINISH_TASK,
            evaluation=eval_payload,
        )
        finish_result = StrategyResult(
            action="finish_task", payload=finish_payload, should_continue=False
        )

        execution_engine.strategy_manager.execute_iteration.side_effect = [
            eval_result,
            finish_result,
        ]

        with patch.object(execution_engine, "_prepare_result") as mock_prepare:
            mock_result = Mock(spec=ExecutionResult)
            mock_prepare.return_value = mock_result

            result = await execution_engine.execute(task)

            # Verify nudge was added for completion
            execution_engine.context_manager.add_nudge.assert_called_with(
                "Task evaluation indicates completion. Please provide the final answer now."
            )

    @pytest.mark.asyncio
    async def test_execute_max_iterations(self, execution_engine):
        """Test execution stopping at max iterations."""
        task = "Test task"
        execution_engine.context.max_iterations = 2

        # Mock session
        session = execution_engine.context.session
        session.final_answer = None
        session.has_failed = False
        session.task_status = TaskStatus.RUNNING

        # Mock continuing payloads that never finish
        continue_payload = ContinueThinkingPayload(
            reasoning="Still working", action=StrategyAction.CONTINUE_THINKING
        )
        continue_result = StrategyResult(
            action="continue_thinking", payload=continue_payload, should_continue=True
        )
        execution_engine.strategy_manager.execute_iteration.return_value = (
            continue_result
        )

        with patch.object(execution_engine, "_prepare_result") as mock_prepare:
            mock_result = Mock(spec=ExecutionResult)
            mock_prepare.return_value = mock_result

            result = await execution_engine.execute(task)

            # Verify max iterations was reached
            assert session.iterations >= 2
            assert session.task_status == TaskStatus.MAX_ITERATIONS

    @pytest.mark.asyncio
    async def test_pause_resume_cycle(self, execution_engine):
        """Test pause and resume functionality."""
        # Test pause
        await execution_engine.pause()
        assert execution_engine.is_paused()
        assert not execution_engine._pause_event.is_set()

        # Verify pause events were emitted
        execution_engine.context.emit_event.assert_any_call(
            AgentStateEvent.PAUSE_REQUESTED, ANY
        )
        execution_engine.context.emit_event.assert_any_call(AgentStateEvent.PAUSED, ANY)

        # Test resume
        await execution_engine.resume()
        assert not execution_engine.is_paused()
        assert execution_engine._pause_event.is_set()

        # Verify resume events were emitted
        execution_engine.context.emit_event.assert_any_call(
            AgentStateEvent.RESUME_REQUESTED, ANY
        )
        execution_engine.context.emit_event.assert_any_call(
            AgentStateEvent.RESUMED, ANY
        )

    @pytest.mark.asyncio
    async def test_terminate(self, execution_engine):
        """Test termination functionality."""
        await execution_engine.terminate()

        assert execution_engine.is_terminating()

        # Verify terminate events were emitted
        execution_engine.context.emit_event.assert_any_call(
            AgentStateEvent.TERMINATE_REQUESTED, ANY
        )
        execution_engine.context.emit_event.assert_any_call(
            AgentStateEvent.TERMINATED, ANY
        )

    @pytest.mark.asyncio
    async def test_stop(self, execution_engine):
        """Test stop functionality."""
        await execution_engine.stop()

        assert execution_engine.is_stopping()

        # Verify stop events were emitted
        execution_engine.context.emit_event.assert_any_call(
            AgentStateEvent.STOP_REQUESTED, ANY
        )
        execution_engine.context.emit_event.assert_any_call(
            AgentStateEvent.STOPPED, ANY
        )

    def test_should_continue_conditions(self, execution_engine):
        """Test various conditions for continuing execution."""
        session = execution_engine.context.session

        # Test normal continuation
        session.has_failed = False
        session.final_answer = None
        session.task_status = TaskStatus.RUNNING
        session.iterations = 5
        assert execution_engine._should_continue()

        # Test with final answer
        session.final_answer = "Answer"
        assert not execution_engine._should_continue()

        # Test with failure
        session.final_answer = None
        session.has_failed = True
        assert not execution_engine._should_continue()

        # Test with completion status
        session.has_failed = False
        session.task_status = TaskStatus.COMPLETE
        assert not execution_engine._should_continue()

        # Test max iterations
        session.task_status = TaskStatus.RUNNING
        session.iterations = 25  # Over default max of 20
        assert not execution_engine._should_continue()
        assert session.task_status == TaskStatus.MAX_ITERATIONS

    def test_check_control_signals(self, execution_engine):
        """Test control signal checking."""
        session = execution_engine.context.session

        # Test normal state
        assert not execution_engine._check_control_signals()

        # Test terminate requested
        execution_engine._terminate_requested = True
        assert execution_engine._check_control_signals()
        assert session.task_status == TaskStatus.CANCELLED

        # Reset and test stop requested
        execution_engine._terminate_requested = False
        session.task_status = TaskStatus.RUNNING
        execution_engine._stop_requested = True
        assert execution_engine._check_control_signals()
        assert session.task_status == TaskStatus.CANCELLED

    def test_get_control_state(self, execution_engine):
        """Test getting control state."""
        state = execution_engine.get_control_state()

        assert isinstance(state, dict)
        assert "paused" in state
        assert "terminate_requested" in state
        assert "stop_requested" in state
        assert "pause_event_set" in state

        # Verify initial values
        assert state["paused"] is False
        assert state["terminate_requested"] is False
        assert state["stop_requested"] is False
        assert state["pause_event_set"] is True

    @pytest.mark.asyncio
    async def test_prepare_result(self, execution_engine):
        """Test result preparation."""
        session = execution_engine.context.session
        session.final_answer = "Test answer"
        session.task_status = TaskStatus.COMPLETE
        session.has_failed = False
        session.errors = []

        # Fix mock attributes that get formatted in to_pretty_string
        session.session_id = "test-session-123"
        session.duration = 1.5
        session.iterations = 3
        session.overall_score = 0.8
        session.end_time = None  # Will be set by the method

        execution_details = {
            "iterations": [{"result": "test"}],
            "total_iterations": 3,
            "strategy": "reactive",
            "state_history": [],
        }

        # Mock the execution result preparation
        with patch(
            "reactive_agents.core.engine.execution_engine.ExecutionResult"
        ) as mock_result_class:
            mock_result = Mock(spec=ExecutionResult)
            mock_result.session = session  # Add session attribute
            mock_result.generate_summary = AsyncMock()
            mock_result.model_dump = Mock(return_value={"test": "data"})
            mock_result.to_pretty_string = Mock(return_value="Pretty result")
            mock_result_class.return_value = mock_result

            result = await execution_engine._prepare_result(execution_details)

            # Verify result was created with correct parameters
            mock_result_class.assert_called_once()

            # Verify summary was generated
            mock_result.generate_summary.assert_called_once()

            # Verify session end event was emitted
            execution_engine.context.emit_event.assert_called_with(
                AgentStateEvent.SESSION_ENDED,
                {
                    "final_result": {"test": "data"},
                    "session_id": session.session_id,
                },
            )

    def test_update_session_scores(self, execution_engine):
        """Test session score calculation."""
        session = execution_engine.context.session
        session.final_answer = "This is a very detailed and comprehensive answer that provides substantial information about the topic being discussed. It includes multiple points and thorough explanations."
        session.task_status = TaskStatus.COMPLETE
        session.iterations = 5
        session.errors = []

        execution_details = {"total_iterations": 5}

        execution_engine._update_session_scores(session, execution_details)

        # Verify scores were calculated
        assert session.completion_score == 1.0  # Complete with answer
        assert session.tool_usage_score == 1.0  # No errors
        assert session.progress_score > 0.5  # Completed successfully
        assert session.answer_quality_score == 0.9  # Substantial answer (>50 chars)
        assert session.llm_evaluation_score == 1.0  # Complete without errors
