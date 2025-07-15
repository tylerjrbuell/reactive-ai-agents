from __future__ import annotations
from typing import List

from reactive_agents.core.types.reasoning_types import ReasoningContext
from reactive_agents.core.reasoning.strategies.base import (
    BaseReasoningStrategy,
    StrategyResult,
    StrategyCapabilities,
)
from reactive_agents.core.types.session_types import ReactiveState, register_strategy


@register_strategy("reactive", ReactiveState)
class ReactiveStrategy(BaseReasoningStrategy):
    """
    Reactive strategy skeleton for custom implementation.
    """

    @property
    def name(self) -> str:
        # Return the unique name of this strategy
        return "reactive"  # placeholder

    @property
    def capabilities(self) -> List[StrategyCapabilities]:
        # Return a list of capabilities this strategy supports
        return []  # placeholder

    @property
    def description(self) -> str:
        # Return a short description of the strategy
        return "Reactive strategy (skeleton)"  # placeholder

    async def initialize(self, task: str, reasoning_context: ReasoningContext) -> None:
        """
        Initialize the strategy for a new task.
        - Set up any required context or state for the strategy.
        """
        pass  # TODO: Implement

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """
        Execute one iteration of the reactive strategy.
        - Execute the main action for the task (e.g., tool call or answer).
        - Optionally reflect and check for completion.
        - Return a StrategyResult indicating progress or completion.
        """
        return StrategyResult(
            action_taken="not_implemented",
            should_continue=False,
            status="not_implemented",
            evaluation=None,
            result=None,
        )
