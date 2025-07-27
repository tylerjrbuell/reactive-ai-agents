from __future__ import annotations
from typing import List

from reactive_agents.core.types.reasoning_types import ReasoningContext
from reactive_agents.core.reasoning.strategies.base import StrategyCapabilities, StrategyResult
from reactive_agents.core.reasoning.strategy_components import ComponentBasedStrategy
from reactive_agents.core.reasoning.protocols import RetryStrategy
from reactive_agents.core.types.session_types import ReactiveState, register_strategy
from reactive_agents.core.reasoning.steps.base import BaseReasoningStep
from reactive_agents.core.reasoning.steps.reactive_steps import ReactiveActStep
from reactive_agents.core.reasoning.steps.common import EvaluateTaskCompletionStep


@register_strategy("reactive", ReactiveState)
class ReactiveStrategy(ComponentBasedStrategy):
    """
    An iterative strategy for reactive task execution.

    This strategy follows a simple "act, then evaluate" loop, allowing it to
    handle multi-step tasks without a formal plan. It acts, checks if it's
    done, and repeats, making it robust for dynamic tasks.
    """

    @property
    def name(self) -> str:
        return "reactive"

    @property
    def capabilities(self) -> List[StrategyCapabilities]:
        return [
            StrategyCapabilities.TOOL_EXECUTION,
            StrategyCapabilities.ADAPTATION,
        ]

    @property
    def description(self) -> str:
        return "An iterative, reactive strategy that acts and then evaluates progress."

    @property
    def steps(self) -> List[BaseReasoningStep]:
        """Defines the Act-Evaluate pipeline for this strategy."""
        return [
            ReactiveActStep(self.engine),
            EvaluateTaskCompletionStep(self.engine),
        ]

    async def initialize(self, task: str, reasoning_context: ReasoningContext) -> None:
        """Initialize the strategy for a new task."""
        state = self.get_state()
        if not isinstance(state, ReactiveState):
            raise TypeError(f"Expected ReactiveState, got {type(state)}")

        state.reset()

        self.context.session.add_message(
            role="system",
            content=f"Role: {self.context.role}\nInstructions: {self.context.instructions}",
        )
        self.context.session.add_message(
            role="user",
            content=f"Task: {task}",
        )

        if self.agent_logger:
            self.agent_logger.info("🚀 ReactiveStrategy | Initialized and ready for task execution")

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """
        Execute one iteration with comprehensive error handling and monitoring.
        
        Args:
            task: The current task description
            reasoning_context: Context about current reasoning state
            
        Returns:
            StrategyResult containing the strategy result
        """
        try:
            # Execute using the parent class iteration logic
            result = await super().execute_iteration(task, reasoning_context)
            
            # Check component health during execution
            health_status = self.get_component_health_status()
            failed_components = [
                name for name, status in health_status.items() 
                if status.get("status") == "error"
            ]
            
            if failed_components and self.agent_logger:
                self.agent_logger.warning(
                    f"🔧 ReactiveStrategy | Component failures detected: {failed_components}"
                )
            
            return result
            
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(
                    f"❌ ReactiveStrategy | Execution failed: {str(e)}"
                )
            # Re-raise the exception to let the engine handle it
            raise e

    def get_strategy_insights(self) -> dict:
        """
        Get insights about this strategy's current performance and capabilities.
        
        Returns:
            Dictionary containing strategy insights
        """
        return {
            "strategy_type": "reactive",
            "optimal_for": [
                "Simple tasks",
                "Dynamic problem solving", 
                "Tasks requiring quick adaptation",
                "Exploratory tasks with unknown requirements"
            ],
            "characteristics": {
                "planning_depth": "minimal",
                "adaptability": "high",
                "tool_usage": "dynamic",
                "reflection_frequency": "per_iteration"
            },
            "performance_factors": {
                "iteration_efficiency": "high",
                "complex_task_handling": "moderate",
                "error_recovery": "good",
                "resource_usage": "low"
            },
            "component_utilization": {
                "thinking": "high",
                "tool_execution": "high", 
                "evaluation": "high",
                "planning": "low",
                "reflection": "moderate"
            }
        }
