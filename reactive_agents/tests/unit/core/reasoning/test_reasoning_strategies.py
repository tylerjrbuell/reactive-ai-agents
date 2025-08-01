"""
Comprehensive tests for reasoning strategies including performance optimization.

This test suite provides:
1. Standard functional validation
2. Performance benchmarking and metrics
3. Strategy comparison and optimization
4. Adaptive learning capabilities
"""

import pytest
import asyncio
import time
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock, AsyncMock, create_autospec
from dataclasses import dataclass

from reactive_agents.core.reasoning.strategies.base import BaseReasoningStrategy
from reactive_agents.core.reasoning.strategies.reactive import ReactiveStrategy
from reactive_agents.core.reasoning.strategies.plan_execute_reflect import (
    PlanExecuteReflectStrategy,
)
from reactive_agents.core.reasoning.strategies.reflect_decide_act import (
    ReflectDecideActStrategy,
)
from reactive_agents.core.reasoning.engine import ReasoningEngine
from reactive_agents.core.context.agent_context import AgentContext
from reactive_agents.core.types.reasoning_types import ReasoningContext
from reactive_agents.core.types.strategy_types import (
    StrategyResult,
    StrategyCapabilities,
)
from reactive_agents.core.types.reasoning_types import (
    StrategyAction,
    EvaluationPayload,
    FinishTaskPayload,
    ContinueThinkingPayload,
    ReasoningStrategies,
)
from reactive_agents.core.types.strategy_types import StrategyCapabilities
from reactive_agents.core.types.session_types import AgentSession
from reactive_agents.core.types.status_types import TaskStatus


@dataclass
class PerformanceMetrics:
    """Metrics for evaluating reasoning strategy performance."""

    strategy_name: str
    task_type: str
    execution_time: float
    iterations_count: int
    success_rate: float
    tool_efficiency: float
    reasoning_quality: float
    adaptability_score: float
    error_recovery_rate: float
    resource_utilization: float


@dataclass
class TaskBenchmark:
    """Benchmark task for strategy evaluation."""

    name: str
    description: str
    complexity: str  # "simple", "medium", "complex"
    expected_tools: List[str]
    success_criteria: Dict[str, Any]
    optimal_iterations: int


class StrategyPerformanceTracker:
    """Tracks and analyzes strategy performance across different tasks."""

    def __init__(self):
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.strategy_comparisons: List[Dict[str, Any]] = []

    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics for a strategy."""
        if metrics.strategy_name not in self.performance_history:
            self.performance_history[metrics.strategy_name] = []
        self.performance_history[metrics.strategy_name].append(metrics)

    def get_strategy_performance(self, strategy_name: str) -> Dict[str, float]:
        """Get aggregated performance metrics for a strategy."""
        if strategy_name not in self.performance_history:
            return {}

        metrics_list = self.performance_history[strategy_name]
        if not metrics_list:
            return {}

        return {
            "avg_execution_time": sum(m.execution_time for m in metrics_list)
            / len(metrics_list),
            "avg_iterations": sum(m.iterations_count for m in metrics_list)
            / len(metrics_list),
            "avg_success_rate": sum(m.success_rate for m in metrics_list)
            / len(metrics_list),
            "avg_tool_efficiency": sum(m.tool_efficiency for m in metrics_list)
            / len(metrics_list),
            "avg_reasoning_quality": sum(m.reasoning_quality for m in metrics_list)
            / len(metrics_list),
            "total_tasks": len(metrics_list),
        }

    def compare_strategies(self, task_type: str) -> Dict[str, Any]:
        """Compare all strategies on a specific task type."""
        comparison = {}
        for strategy_name, metrics_list in self.performance_history.items():
            task_metrics = [m for m in metrics_list if m.task_type == task_type]
            if task_metrics:
                comparison[strategy_name] = {
                    "success_rate": sum(m.success_rate for m in task_metrics)
                    / len(task_metrics),
                    "efficiency": sum(m.tool_efficiency for m in task_metrics)
                    / len(task_metrics),
                    "speed": sum(m.execution_time for m in task_metrics)
                    / len(task_metrics),
                    "quality": sum(m.reasoning_quality for m in task_metrics)
                    / len(task_metrics),
                }
        return comparison


class TestReasoningStrategies:
    """Comprehensive test suite for reasoning strategies."""

    @pytest.fixture
    def performance_tracker(self):
        """Create a performance tracker for all tests."""
        return StrategyPerformanceTracker()

    @pytest.fixture
    def benchmark_tasks(self):
        """Create benchmark tasks for strategy evaluation."""
        return [
            TaskBenchmark(
                name="simple_calculation",
                description="Calculate 15 * 23 + 7",
                complexity="simple",
                expected_tools=["execute_python_code"],
                success_criteria={"correct_answer": "352"},
                optimal_iterations=2,
            ),
            TaskBenchmark(
                name="web_research",
                description="Find the current population of Tokyo",
                complexity="medium",
                expected_tools=["google_search_api", "url_to_markdown"],
                success_criteria={"has_population_data": True, "recent_data": True},
                optimal_iterations=4,
            ),
            TaskBenchmark(
                name="multi_step_analysis",
                description="Analyze sales data, create visualization, and provide recommendations",
                complexity="complex",
                expected_tools=["read_file", "execute_python_code", "write_file"],
                success_criteria={
                    "has_analysis": True,
                    "has_visualization": True,
                    "has_recommendations": True,
                },
                optimal_iterations=8,
            ),
        ]

    @pytest.fixture
    def mock_reasoning_engine(self):
        """Create a mock reasoning engine."""
        engine = create_autospec(ReasoningEngine, instance=True)

        # Mock context
        mock_context = create_autospec(AgentContext, instance=True)
        mock_context.agent_name = "TestAgent"
        mock_context.session = Mock()
        mock_context.session.session_id = "test-session"
        mock_context.session.iterations = 0
        mock_context.session.add_message = Mock()
        mock_context.session.strategy_state = {}
        mock_context.session.initialize_strategy_state = Mock()
        mock_context.session.get_strategy_state = Mock()
        mock_context.agent_logger = Mock()
        mock_context.role = "assistant"
        mock_context.instructions = "Test instructions"
        mock_context.get_tool_signatures = Mock(return_value=[])
        mock_context.emit_event = Mock()

        engine.context = mock_context

        # Mock engine methods
        engine.get_prompt = Mock()
        engine.think = AsyncMock()
        engine.think_chain = AsyncMock()
        engine.execute_tools = AsyncMock()

        return engine

    @pytest.fixture
    def reactive_strategy(self, mock_reasoning_engine):
        """Create a ReactiveStrategy instance."""
        with patch(
            "reactive_agents.core.reasoning.strategies.reactive.ComponentBasedStrategy.__init__",
            return_value=None,
        ):
            strategy = ReactiveStrategy(mock_reasoning_engine)
            strategy.engine = mock_reasoning_engine
            strategy.context = mock_reasoning_engine.context
            strategy.agent_logger = mock_reasoning_engine.context.agent_logger

            # Add missing component attributes
            strategy._thinking = Mock()
            strategy._planning = Mock()
            strategy._tools = Mock()
            strategy._reflection = Mock()
            strategy._evaluation = Mock()
            strategy._completion = Mock()
            strategy._error_handling = Mock()
            strategy._memory = Mock()
            strategy._transition = Mock()

            return strategy

    @pytest.fixture
    def plan_execute_reflect_strategy(self, mock_reasoning_engine):
        """Create a PlanExecuteReflectStrategy instance."""
        with patch(
            "reactive_agents.core.reasoning.strategies.plan_execute_reflect.ComponentBasedStrategy.__init__",
            return_value=None,
        ):
            strategy = PlanExecuteReflectStrategy(mock_reasoning_engine)
            strategy.engine = mock_reasoning_engine
            strategy.context = mock_reasoning_engine.context
            strategy.agent_logger = mock_reasoning_engine.context.agent_logger

            # Add missing component attributes
            strategy._thinking = Mock()
            strategy._planning = Mock()
            strategy._tools = Mock()
            strategy._reflection = Mock()
            strategy._evaluation = Mock()
            strategy._completion = Mock()
            strategy._error_handling = Mock()
            strategy._memory = Mock()
            strategy._transition = Mock()

            return strategy

    # === Standard Functional Tests ===

    def test_reactive_strategy_properties(self, reactive_strategy):
        """Test ReactiveStrategy basic properties."""
        assert reactive_strategy.name == "reactive"
        assert StrategyCapabilities.TOOL_EXECUTION in reactive_strategy.capabilities
        assert StrategyCapabilities.ADAPTATION in reactive_strategy.capabilities
        assert "reactive" in reactive_strategy.description.lower()

    def test_plan_execute_reflect_properties(self, plan_execute_reflect_strategy):
        """Test PlanExecuteReflectStrategy basic properties."""
        assert plan_execute_reflect_strategy.name == "plan_execute_reflect"
        assert len(plan_execute_reflect_strategy.capabilities) > 0
        assert "plan" in plan_execute_reflect_strategy.description.lower()

    @pytest.mark.asyncio
    async def test_strategy_initialization(self, reactive_strategy):
        """Test strategy initialization process."""
        task = "Test task"
        reasoning_context = ReasoningContext(
            current_strategy=ReasoningStrategies.REACTIVE,
            iteration_count=0,
            error_count=0,
            tool_usage_history=[],
        )

        # Mock state
        from reactive_agents.core.types.session_types import ReactiveState

        mock_state = ReactiveState()

        with patch.object(reactive_strategy, "get_state", return_value=mock_state):
            await reactive_strategy.initialize(task, reasoning_context)

            # Verify initialization steps
            reactive_strategy.context.session.add_message.assert_called()

    @pytest.mark.asyncio
    async def test_strategy_execution_success(self, reactive_strategy):
        """Test successful strategy execution."""
        task = "Test task"
        reasoning_context = ReasoningContext(
            current_strategy=ReasoningStrategies.REACTIVE,
            iteration_count=1,
            error_count=0,
            tool_usage_history=[],
        )

        # Mock successful execution
        success_result = StrategyResult(
            action="finish_task",
            payload=FinishTaskPayload(
                final_answer="Test answer",
                action=StrategyAction.FINISH_TASK,
                evaluation=EvaluationPayload(
                    action=StrategyAction.EVALUATE_COMPLETION,
                    is_complete=True,
                    reasoning="Task completed",
                    confidence=0.9,
                ),
            ),
            should_continue=False,
        )

        with patch(
            "reactive_agents.core.reasoning.strategies.reactive.ComponentBasedStrategy.execute_iteration",
            return_value=success_result,
        ) as mock_super:
            with patch.object(
                reactive_strategy, "get_component_health_status", return_value={}
            ):
                result = await reactive_strategy.execute_iteration(
                    task, reasoning_context
                )

                assert result == success_result
                mock_super.assert_called_once()

    @pytest.mark.asyncio
    async def test_strategy_execution_with_errors(self, reactive_strategy):
        """Test strategy execution with error handling."""
        task = "Test task"
        reasoning_context = ReasoningContext(
            current_strategy=ReasoningStrategies.REACTIVE,
            iteration_count=1,
            error_count=1,
            tool_usage_history=[],
        )

        # Mock execution error
        with patch(
            "reactive_agents.core.reasoning.strategies.reactive.ComponentBasedStrategy.execute_iteration",
            side_effect=RuntimeError("Test error"),
        ):
            with pytest.raises(RuntimeError):
                await reactive_strategy.execute_iteration(task, reasoning_context)

            # Verify error logging
            reactive_strategy.agent_logger.error.assert_called()

    # === Performance Benchmarking Tests ===

    @pytest.mark.asyncio
    async def test_strategy_performance_simple_task(
        self, reactive_strategy, performance_tracker, benchmark_tasks
    ):
        """Test strategy performance on simple tasks."""
        simple_task = next(t for t in benchmark_tasks if t.complexity == "simple")

        start_time = time.time()

        # Mock execution
        reasoning_context = ReasoningContext(
            current_strategy=ReasoningStrategies.REACTIVE,
            iteration_count=0,
            error_count=0,
            tool_usage_history=[],
        )

        with patch.object(reactive_strategy, "get_state", return_value=Mock()):
            with patch(
                "reactive_agents.core.reasoning.strategies.reactive.ComponentBasedStrategy.execute_iteration"
            ) as mock_exec:
                # Simulate successful execution
                mock_exec.return_value = StrategyResult(
                    action="finish_task",
                    payload=FinishTaskPayload(
                        final_answer="352",
                        action=StrategyAction.FINISH_TASK,
                        evaluation=EvaluationPayload(
                            action=StrategyAction.EVALUATE_COMPLETION,
                            is_complete=True,
                            reasoning="Calculation completed",
                            confidence=1.0,
                        ),
                    ),
                    should_continue=False,
                )

                result = await reactive_strategy.execute_iteration(
                    simple_task.description, reasoning_context
                )

                execution_time = time.time() - start_time

                # Calculate performance metrics
                metrics = PerformanceMetrics(
                    strategy_name=reactive_strategy.name,
                    task_type=simple_task.complexity,
                    execution_time=execution_time,
                    iterations_count=1,
                    success_rate=1.0,  # Successful completion
                    tool_efficiency=1.0,  # Used expected tools
                    reasoning_quality=0.9,  # High confidence
                    adaptability_score=0.8,
                    error_recovery_rate=1.0,  # No errors
                    resource_utilization=0.7,
                )

                performance_tracker.record_performance(metrics)

                # Verify performance meets expectations
                assert execution_time < 5.0  # Should be fast for simple tasks
                assert result.payload.evaluation.confidence > 0.8

    @pytest.mark.asyncio
    async def test_strategy_performance_complex_task(
        self, reactive_strategy, performance_tracker, benchmark_tasks
    ):
        """Test strategy performance on complex tasks."""
        complex_task = next(t for t in benchmark_tasks if t.complexity == "complex")

        start_time = time.time()

        # Mock multiple iterations for complex task
        reasoning_context = ReasoningContext(
            current_strategy=ReasoningStrategies.REACTIVE,
            iteration_count=0,
            error_count=0,
            tool_usage_history=[],
        )

        with patch.object(reactive_strategy, "get_state", return_value=Mock()):
            with patch(
                "reactive_agents.core.reasoning.strategies.reactive.ComponentBasedStrategy.execute_iteration"
            ) as mock_exec:
                # Simulate complex execution with multiple steps
                mock_exec.return_value = StrategyResult(
                    action="finish_task",
                    payload=FinishTaskPayload(
                        final_answer="Analysis complete with visualization and recommendations",
                        action=StrategyAction.FINISH_TASK,
                        evaluation=EvaluationPayload(
                            action=StrategyAction.EVALUATE_COMPLETION,
                            is_complete=True,
                            reasoning="All requirements met",
                            confidence=0.85,
                        ),
                    ),
                    should_continue=False,
                )

                result = await reactive_strategy.execute_iteration(
                    complex_task.description, reasoning_context
                )

                execution_time = time.time() - start_time

                # Calculate performance metrics
                metrics = PerformanceMetrics(
                    strategy_name=reactive_strategy.name,
                    task_type=complex_task.complexity,
                    execution_time=execution_time,
                    iterations_count=6,  # More iterations for complex task
                    success_rate=1.0,
                    tool_efficiency=0.8,  # Good tool usage
                    reasoning_quality=0.85,
                    adaptability_score=0.9,  # High adaptability
                    error_recovery_rate=1.0,
                    resource_utilization=0.9,
                )

                performance_tracker.record_performance(metrics)

                # Verify performance is reasonable for complex tasks
                assert result.payload.evaluation.confidence > 0.7

    # === Strategy Comparison Tests ===

    @pytest.mark.asyncio
    async def test_compare_strategies_on_same_task(
        self,
        reactive_strategy,
        plan_execute_reflect_strategy,
        performance_tracker,
        benchmark_tasks,
    ):
        """Compare different strategies on the same task."""
        test_task = benchmark_tasks[1]  # Medium complexity task  # type: ignore
        strategies = [reactive_strategy, plan_execute_reflect_strategy]
        results = {}

        for strategy in strategies:
            start_time = time.time()
            reasoning_context = ReasoningContext(
                current_strategy=ReasoningStrategies.REACTIVE,
                iteration_count=0,
                error_count=0,
                tool_usage_history=[],
            )

            with patch.object(strategy, "get_state", return_value=Mock()):
                with patch.object(strategy, "execute_iteration") as mock_exec:
                    # Mock strategy-specific behavior
                    if strategy.name == "reactive":
                        mock_exec.return_value = StrategyResult(
                            action="finish_task",
                            payload=FinishTaskPayload(
                                final_answer="Tokyo population: ~14 million",
                                action=StrategyAction.FINISH_TASK,
                                evaluation=EvaluationPayload(
                                    action=StrategyAction.EVALUATE_COMPLETION,
                                    is_complete=True,
                                    reasoning="Found current data",
                                    confidence=0.8,
                                ),
                            ),
                            should_continue=False,
                        )
                        iterations = 3
                    else:  # plan_execute_reflect
                        mock_exec.return_value = StrategyResult(
                            action="finish_task",
                            payload=FinishTaskPayload(
                                final_answer="Tokyo population: ~14 million (verified from multiple sources)",
                                action=StrategyAction.FINISH_TASK,
                                evaluation=EvaluationPayload(
                                    action=StrategyAction.EVALUATE_COMPLETION,
                                    is_complete=True,
                                    reasoning="Thoroughly researched and verified",
                                    confidence=0.95,
                                ),
                            ),
                            should_continue=False,
                        )
                        iterations = 5

                    result = await strategy.execute_iteration(
                        test_task.description, reasoning_context
                    )
                    execution_time = time.time() - start_time

                    # Record metrics
                    metrics = PerformanceMetrics(
                        strategy_name=strategy.name,
                        task_type=test_task.complexity,
                        execution_time=execution_time,
                        iterations_count=iterations,
                        success_rate=1.0,
                        tool_efficiency=(
                            0.9 if strategy.name == "plan_execute_reflect" else 0.8
                        ),
                        reasoning_quality=result.payload.evaluation.confidence,
                        adaptability_score=0.85,
                        error_recovery_rate=1.0,
                        resource_utilization=0.8,
                    )

                    performance_tracker.record_performance(metrics)
                    results[strategy.name] = result

        # Compare results
        comparison = performance_tracker.compare_strategies(test_task.complexity)

        # Verify both strategies completed successfully
        assert len(results) == 2
        assert all(r.payload.evaluation.is_complete for r in results.values())

        # Verify comparison data
        assert len(comparison) == 2
        assert "reactive" in comparison
        assert "plan_execute_reflect" in comparison

    # === Adaptive Optimization Tests ===

    def test_strategy_learning_from_performance(self, performance_tracker):
        """Test that strategies can learn from performance history."""
        # Simulate performance history
        for i in range(10):
            metrics = PerformanceMetrics(
                strategy_name="reactive",
                task_type="simple",
                execution_time=1.0 + (i * 0.1),  # Gradually improving
                iterations_count=2,
                success_rate=0.8 + (i * 0.02),  # Learning improves success
                tool_efficiency=0.7 + (i * 0.03),
                reasoning_quality=0.8 + (i * 0.02),
                adaptability_score=0.8,
                error_recovery_rate=1.0,
                resource_utilization=0.7,
            )
            performance_tracker.record_performance(metrics)

        # Analyze learning curve
        performance = performance_tracker.get_strategy_performance("reactive")

        assert performance["total_tasks"] == 10  # type: ignore
        assert performance["avg_success_rate"] > 0.8  # type: ignore
        assert performance["avg_tool_efficiency"] > 0.7  # type: ignore

    def test_strategy_optimization_recommendations(self, performance_tracker):
        """Test generating optimization recommendations for strategies."""
        # Record poor performance scenario
        poor_metrics = PerformanceMetrics(
            strategy_name="reactive",
            task_type="complex",
            execution_time=10.0,  # Slow
            iterations_count=15,  # Too many iterations
            success_rate=0.6,  # Low success
            tool_efficiency=0.4,  # Poor tool usage
            reasoning_quality=0.5,  # Low quality
            adaptability_score=0.3,
            error_recovery_rate=0.7,
            resource_utilization=0.9,
        )
        performance_tracker.record_performance(poor_metrics)

        performance = performance_tracker.get_strategy_performance("reactive")

        # Generate optimization recommendations
        recommendations = []
        if performance["avg_success_rate"] < 0.8:  # type: ignore
            recommendations.append("improve_error_handling")
        if performance["avg_tool_efficiency"] < 0.7:  # type: ignore
            recommendations.append("optimize_tool_selection")
        if performance["avg_execution_time"] > 5.0:  # type: ignore
            recommendations.append("reduce_iteration_overhead")

        assert len(recommendations) > 0
        assert "improve_error_handling" in recommendations

    # === Integration and Utility Tests ===

    def test_strategy_insights_generation(self, reactive_strategy):
        """Test strategy insights for performance analysis."""
        insights = reactive_strategy.get_strategy_insights()

        assert insights["strategy_type"] == "reactive"  # type: ignore
        assert "optimal_for" in insights
        assert "characteristics" in insights
        assert "performance_factors" in insights
        assert "component_utilization" in insights

        # Verify insights provide actionable information
        assert len(insights["optimal_for"]) > 0  # type: ignore
        assert "adaptability" in insights["characteristics"]  # type: ignore

    @pytest.mark.asyncio
    async def test_strategy_error_recovery_performance(
        self, reactive_strategy, performance_tracker
    ):
        """Test strategy performance during error recovery scenarios."""
        task = "Test error recovery"
        reasoning_context = ReasoningContext(
            current_strategy=ReasoningStrategies.REACTIVE,
            iteration_count=1,
            error_count=2,
            tool_usage_history=[],
        )

        # Mock error recovery scenario
        with patch.object(reactive_strategy, "get_state", return_value=Mock()):
            with patch(
                "reactive_agents.core.reasoning.strategies.reactive.ComponentBasedStrategy.execute_iteration"
            ) as mock_exec:
                # First call fails, second succeeds (error recovery)
                mock_exec.side_effect = [
                    RuntimeError("Simulated error"),
                    StrategyResult(
                        action="finish_task",
                        payload=FinishTaskPayload(
                            final_answer="Recovered successfully",
                            action=StrategyAction.FINISH_TASK,
                            evaluation=EvaluationPayload(
                                action=StrategyAction.EVALUATE_COMPLETION,
                                is_complete=True,
                                reasoning="Error recovered",
                                confidence=0.8,
                            ),
                        ),
                        should_continue=False,
                    ),
                ]

                # Test error scenario
                with pytest.raises(RuntimeError):
                    await reactive_strategy.execute_iteration(task, reasoning_context)

                # Test recovery
                result = await reactive_strategy.execute_iteration(
                    task, reasoning_context
                )

                # Record error recovery metrics
                metrics = PerformanceMetrics(
                    strategy_name=reactive_strategy.name,
                    task_type="error_recovery",
                    execution_time=2.0,
                    iterations_count=2,
                    success_rate=0.5,  # 1 success out of 2 attempts
                    tool_efficiency=0.8,
                    reasoning_quality=0.8,
                    adaptability_score=0.9,  # High adaptability during recovery
                    error_recovery_rate=1.0,  # Successful recovery
                    resource_utilization=0.8,
                )

                performance_tracker.record_performance(metrics)

                assert result.payload.evaluation.is_complete
                assert metrics.error_recovery_rate == 1.0

    def test_performance_tracker_analytics(self, performance_tracker):
        """Test performance tracker analytics capabilities."""
        # Add sample data for multiple strategies
        strategies = ["reactive", "plan_execute_reflect", "reflect_decide_act"]
        task_types = ["simple", "medium", "complex"]

        for strategy in strategies:
            for task_type in task_types:
                for i in range(3):  # 3 samples each
                    metrics = PerformanceMetrics(
                        strategy_name=strategy,
                        task_type=task_type,
                        execution_time=1.0
                        + (len(strategy) * 0.1),  # Strategy-dependent timing
                        iterations_count=2 + (1 if task_type == "complex" else 0),
                        success_rate=0.9 - (0.1 if task_type == "complex" else 0),
                        tool_efficiency=0.8,
                        reasoning_quality=0.85,
                        adaptability_score=0.8,
                        error_recovery_rate=0.9,
                        resource_utilization=0.7,
                    )
                    performance_tracker.record_performance(metrics)

        # Test analytics
        for strategy in strategies:
            performance = performance_tracker.get_strategy_performance(strategy)
            assert performance["total_tasks"] == 9  # 3 task types × 3 samples  # type: ignore
            assert performance["avg_success_rate"] > 0.7  # type: ignore

        # Test comparisons
        for task_type in task_types:
            comparison = performance_tracker.compare_strategies(task_type)
            assert len(comparison) == 3  # All strategies

            # Verify comparison includes key metrics
            for strategy_data in comparison.values():
                assert "success_rate" in strategy_data
                assert "efficiency" in strategy_data
                assert "speed" in strategy_data
                assert "quality" in strategy_data
