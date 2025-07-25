from __future__ import annotations
from typing import List

from reactive_agents.core.reasoning.steps.base import BaseReasoningStep
from reactive_agents.core.types.reasoning_types import (
    ReasoningContext,
)
from reactive_agents.core.reasoning.strategies.base import (
    StrategyCapabilities,
)
from reactive_agents.core.reasoning.strategy_components import (
    ComponentBasedStrategy,
)
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

        self.context.session.add_message(
            role="system",
            content=f"Role: {self.context.role}\nInstructions: {self.context.instructions}",
        )
        self.context.session.add_message(
            role="user",
            content=f"Task: {task}",
        )

        # The first action is to create a plan using the planning component.
        state.current_plan = await self.plan(task, reasoning_context)

        if self.agent_logger:
            self.agent_logger.info(
                f"📋 Generated plan with {len(state.current_plan.plan_steps)} steps"
            )
            for step in state.current_plan.plan_steps:
                self.agent_logger.info(
                    f"Step {step.index}: {step.description} - required_tools: {step.required_tools}"
                )
