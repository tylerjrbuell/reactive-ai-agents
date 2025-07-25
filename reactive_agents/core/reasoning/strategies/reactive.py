from __future__ import annotations
from typing import List

from reactive_agents.core.types.reasoning_types import ReasoningContext
from reactive_agents.core.reasoning.strategies.base import StrategyCapabilities
from reactive_agents.core.reasoning.strategy_components import ComponentBasedStrategy
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
            self.agent_logger.info("Initialized Reactive Strategy.")
