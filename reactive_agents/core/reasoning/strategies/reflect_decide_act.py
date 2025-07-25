from __future__ import annotations
from typing import List

from reactive_agents.core.types.reasoning_types import ReasoningContext
from reactive_agents.core.reasoning.strategies.base import StrategyCapabilities
from reactive_agents.core.reasoning.strategy_components import ComponentBasedStrategy
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
            self.agent_logger.info("Initialized Reflect-Decide-Act Strategy.")
