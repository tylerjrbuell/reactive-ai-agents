from __future__ import annotations
from typing import List

from reactive_agents.core.reasoning.steps.base import BaseReasoningStep
from reactive_agents.core.types.reasoning_types import (
    ReasoningContext,
    ReasoningStrategies,
)
from reactive_agents.core.reasoning.strategies.base import (
    StrategyCapabilities,
    StrategyResult,
)
from reactive_agents.core.reasoning.strategy_components import (
    ComponentBasedStrategy,
)
from reactive_agents.core.reasoning.protocols import RetryStrategy
from reactive_agents.core.types.session_types import (
    PlanExecuteReflectState,
    register_strategy,
)
from reactive_agents.core.reasoning.steps.plan_execute_reflect_steps import (
    CheckPlanCompletionStep,
    ExecutePlanStep,
    ReflectStep,
)


@register_strategy("plan_execute_reflect", PlanExecuteReflectState)
class PlanExecuteReflectStrategy(ComponentBasedStrategy):
    """
    A declarative, step-based implementation of the Plan-Execute-Reflect strategy.

    This strategy defines its reasoning flow by composing a pipeline of reusable
    `BaseReasoningStep` objects, making the logic clear, composable, and easy to modify.
    """

    @property
    def name(self) -> str:
        return "plan_execute_reflect"

    @property
    def capabilities(self) -> List[StrategyCapabilities]:
        return [
            StrategyCapabilities.PLANNING,
            StrategyCapabilities.TOOL_EXECUTION,
            StrategyCapabilities.REFLECTION,
        ]

    @property
    def description(self) -> str:
        return "Declarative Plan-Execute-Reflect strategy using a step-based pipeline."

    @property
    def steps(self) -> List[BaseReasoningStep]:
        """Defines the reasoning pipeline for this strategy."""
        return [
            CheckPlanCompletionStep(self.engine),
            ExecutePlanStep(self.engine),
            ReflectStep(self.engine),
        ]

    async def initialize(self, task: str, reasoning_context: ReasoningContext) -> None:
        """Initialize the strategy for a new task by generating the initial plan."""
        state = self.get_state()
        if not isinstance(state, PlanExecuteReflectState):
            raise TypeError(f"Expected PlanExecuteReflectState, got {type(state)}")

        state.reset()

        # Use centralized system message creation
        self._add_centralized_system_message()
        
        self.context.session.add_message(
            role="user",
            content=f"Task: {task}",
        )

        # The first action is to create a plan using the planning component.
        state.current_plan = await self.plan(task, reasoning_context)

        if self.agent_logger:
            self.agent_logger.info(
                f"📋 PlanExecuteReflectStrategy | Generated plan with {len(state.current_plan.plan_steps)} steps"
            )
            for step in state.current_plan.plan_steps:
                self.agent_logger.info(
                    f"  📌 Step {step.index}: {step.description} - required_tools: {step.required_tools}"
                )

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """
        Execute one iteration with planning-specific monitoring.

        Args:
            task: The current task description
            reasoning_context: Context about current reasoning state

        Returns:
            StrategyResult containing the strategy result
        """
        try:
            # Get current state for plan tracking
            state = self.get_state()
            if not isinstance(state, PlanExecuteReflectState):
                raise ValueError("Invalid state type for PlanExecuteReflect strategy")

            # Execute using the parent class iteration logic
            result = await super().execute_iteration(task, reasoning_context)

            # Plan progress tracking
            plan_progress = self._analyze_plan_progress(state)
            component_health = self.get_component_health_status()
            
            if self.agent_logger:
                self.agent_logger.debug(
                    f"📋 PlanExecuteReflectStrategy | Plan progress: {plan_progress.get('progress', 0):.1%} complete"
                )

            return result

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(
                    f"❌ PlanExecuteReflectStrategy | Execution failed: {str(e)}"
                )
            # Re-raise the exception to let the engine handle it
            raise e

    def _analyze_plan_progress(self, state: PlanExecuteReflectState) -> dict:
        """
        Analyze the current progress of the plan execution.

        Args:
            state: The current strategy state

        Returns:
            Dictionary containing plan progress analysis
        """
        if not state.current_plan or not state.current_plan.plan_steps:
            return {"status": "no_plan", "progress": 0.0}

        total_steps = len(state.current_plan.plan_steps)
        completed_steps = sum(
            1 for step in state.current_plan.plan_steps if step.is_finished()
        )
        failed_steps = sum(
            1 for step in state.current_plan.plan_steps if step.status.value == "failed"
        )
        step = state.current_plan.get_next_step()

        return {
            "status": "in_progress",
            "progress": completed_steps / total_steps if total_steps > 0 else 0.0,
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "current_step": step.index if step else None,
            "efficiency": (completed_steps - failed_steps)
            / max(1, completed_steps + failed_steps),
        }

    def get_strategy_insights(self) -> dict:
        """
        Get insights about this strategy's current performance and capabilities.

        Returns:
            Dictionary containing strategy insights
        """
        return {
            "strategy_type": "plan_execute_reflect",
            "optimal_for": [
                "Complex multi-step tasks",
                "Tasks requiring systematic approach",
                "Long-running workflows",
                "Tasks with clear dependencies",
            ],
            "characteristics": {
                "planning_depth": "high",
                "adaptability": "moderate",
                "tool_usage": "structured",
                "reflection_frequency": "post_execution",
            },
            "performance_factors": {
                "iteration_efficiency": "moderate",
                "complex_task_handling": "excellent",
                "error_recovery": "very_good",
                "resource_usage": "moderate",
            },
            "component_utilization": {
                "thinking": "moderate",
                "planning": "very_high",
                "tool_execution": "high",
                "evaluation": "high",
                "reflection": "high",
            },
        }

    async def analyze_plan_effectiveness(self, task: str) -> dict:
        """
        Analyze how effective the current plan is for the given task.

        Args:
            task: The current task

        Returns:
            Plan effectiveness analysis
        """
        state = self.get_state()
        if not isinstance(state, PlanExecuteReflectState) or not state.current_plan:
            return {"status": "no_plan_available"}

        plan_progress = self._analyze_plan_progress(state)

        # Analysis using our components
        try:
            reflection_context = self.create_component_context(task, "plan_analysis")
            reflection_result = await self.reflect(
                task,
                {"plan_progress": plan_progress},
                ReasoningContext(
                    current_strategy=ReasoningStrategies.PLAN_EXECUTE_REFLECT
                ),
            )

            effectiveness_score = 0.7  # Base score
            if reflection_result and reflection_result.confidence > 0:
                # Adjust based on reflection insights
                effectiveness_score = reflection_result.confidence

            return {
                "effectiveness_score": effectiveness_score,
                "plan_progress": plan_progress,
                "reflection_insights": {
                    "progress_assessment": reflection_result.progress_assessment,
                    "goal_achieved": reflection_result.goal_achieved,
                    "confidence": reflection_result.confidence,
                    "next_action": reflection_result.next_action,
                    "blockers": reflection_result.blockers,
                    "success_indicators": reflection_result.success_indicators,
                } if reflection_result else None,
                "recommendations": self._generate_plan_recommendations(plan_progress),
            }

        except Exception as e:
            return {
                "status": "analysis_failed",
                "error": str(e),
                "plan_progress": plan_progress,
            }

    def _generate_plan_recommendations(self, plan_progress: dict) -> list:
        """Generate recommendations based on plan progress."""
        recommendations = []

        if plan_progress.get("failed_steps", 0) > 0:
            recommendations.append(
                "Consider revising failed steps or switching approach"
            )

        if (
            plan_progress.get("progress", 0) < 0.3
            and plan_progress.get("total_steps", 0) > 5
        ):
            recommendations.append(
                "Plan might be too complex - consider breaking into smaller tasks"
            )

        if plan_progress.get("efficiency", 1.0) < 0.5:
            recommendations.append(
                "Low execution efficiency - review step dependencies and requirements"
            )

        return recommendations
