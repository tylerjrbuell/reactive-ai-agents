from __future__ import annotations
from typing import List

from reactive_agents.core.types.reasoning_types import ReasoningContext
from reactive_agents.core.reasoning.strategies.base import StrategyCapabilities, StrategyResult
from reactive_agents.core.reasoning.strategy_components import ComponentBasedStrategy
from reactive_agents.core.reasoning.protocols import RetryStrategy
from reactive_agents.core.types.session_types import (
    ReflectDecideActState,
    register_strategy,
)
from reactive_agents.core.reasoning.steps.base import BaseReasoningStep
from reactive_agents.core.reasoning.steps.reflect_decide_act_steps import (
    ReflectOnSituationStep,
    DecideNextActionStep,
    ExecuteActionStep,
)


@register_strategy("reflect_decide_act", ReflectDecideActState)
class ReflectDecideActStrategy(ComponentBasedStrategy):
    """
    A declarative implementation of the Reflect-Decide-Act strategy.

    This strategy follows a continuous loop of observing, orienting,
    deciding, and acting, making it suitable for dynamic and
    unpredictable tasks.
    """

    @property
    def name(self) -> str:
        return "reflect_decide_act"

    @property
    def capabilities(self) -> List[StrategyCapabilities]:
        return [
            StrategyCapabilities.REFLECTION,
            StrategyCapabilities.ADAPTATION,
            StrategyCapabilities.TOOL_EXECUTION,
        ]

    @property
    def description(self) -> str:
        return "A dynamic strategy that continuously reflects, decides, and acts."

    @property
    def steps(self) -> List[BaseReasoningStep]:
        """Defines the Reflect-Decide-Act pipeline."""
        return [
            ReflectOnSituationStep(self.engine),
            DecideNextActionStep(self.engine),
            ExecuteActionStep(self.engine),
        ]

    async def initialize(self, task: str, reasoning_context: ReasoningContext) -> None:
        """Initialize the strategy for a new task."""
        state = self.get_state()
        if not isinstance(state, ReflectDecideActState):
            raise TypeError(f"Expected ReflectDecideActState, got {type(state)}")

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
            self.agent_logger.info("🚀 ReflectDecideActStrategy | Initialized and ready for dynamic task execution")

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """
        Execute one iteration with reflection-specific monitoring.
        
        Args:
            task: The current task description
            reasoning_context: Context about current reasoning state
            
        Returns:
            StrategyResult containing the strategy result
        """
        try:
            # Get current state for reflection tracking
            state = self.get_state()
            if not isinstance(state, ReflectDecideActState):
                raise ValueError("Invalid state type for ReflectDecideAct strategy")
            
            # Execute using the parent class iteration logic
            result = await super().execute_iteration(task, reasoning_context)
            
            # Track reflection cycle progress
            reflection_progress = self._analyze_reflection_progress(state, reasoning_context)
            component_health = self.get_component_health_status()
            
            if self.agent_logger:
                self.agent_logger.debug(
                    f"🔄 ReflectDecideActStrategy | Reflection cycle {reasoning_context.iteration_count}, "
                    f"adaptability: {reflection_progress.get('adaptability_score', 0):.2f}"
                )
            
            return result
            
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(
                    f"❌ ReflectDecideActStrategy | Execution failed: {str(e)}"
                )
            # Re-raise the exception to let the engine handle it
            raise e

    def _analyze_reflection_progress(
        self, state: ReflectDecideActState, reasoning_context: ReasoningContext
    ) -> dict:
        """
        Analyze the current progress of the reflection cycle.
        
        Args:
            state: The current strategy state
            reasoning_context: Current reasoning context
            
        Returns:
            Dictionary containing reflection progress analysis
        """
        reflection_cycles = reasoning_context.iteration_count
        
        # Calculate reflection effectiveness based on error patterns
        error_trend = "improving" if reasoning_context.error_count < reflection_cycles * 0.1 else "stable"
        if reasoning_context.error_count > reflection_cycles * 0.3:
            error_trend = "concerning"
        
        # Assess decision quality over time
        decision_quality = "excellent" if reflection_cycles > 3 and error_trend == "improving" else "good"
        if error_trend == "concerning":
            decision_quality = "needs_improvement"
        
        return {
            "status": "active",
            "reflection_cycles": reflection_cycles,
            "error_trend": error_trend,
            "decision_quality": decision_quality,
            "adaptability_score": max(0.0, 1.0 - (reasoning_context.error_count / max(1, reflection_cycles))),
            "recent_errors": reasoning_context.error_count,
            "cycle_efficiency": 1.0 / max(1, reflection_cycles / 3)  # Prefer fewer cycles for simple tasks
        }

    def get_strategy_insights(self) -> dict:
        """
        Get insights about this strategy's current performance and capabilities.
        
        Returns:
            Dictionary containing strategy insights
        """
        return {
            "strategy_type": "reflect_decide_act",
            "optimal_for": [
                "Dynamic problem solving",
                "Uncertain or evolving tasks", 
                "Tasks requiring frequent adaptation",
                "Exploratory research tasks",
                "Real-time decision making"
            ],
            "characteristics": {
                "planning_depth": "minimal",
                "adaptability": "very_high",
                "tool_usage": "contextual",
                "reflection_frequency": "continuous"
            },
            "performance_factors": {
                "iteration_efficiency": "moderate",
                "complex_task_handling": "good",
                "error_recovery": "excellent",
                "resource_usage": "moderate"
            },
            "component_utilization": {
                "thinking": "very_high",
                "reflection": "very_high",
                "tool_execution": "moderate",
                "evaluation": "high",
                "planning": "minimal"
            }
        }
