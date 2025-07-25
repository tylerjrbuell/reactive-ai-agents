from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from reactive_agents.core.reasoning.engine import ReasoningEngine
    from reactive_agents.core.types.session_types import BaseStrategyState
    from reactive_agents.core.types.reasoning_types import ReasoningContext
    from reactive_agents.core.reasoning.strategies.base import StrategyResult
    from reactive_agents.core.reasoning.strategy_components import ComponentBasedStrategy


class BaseReasoningStep(ABC):
    """
    Represents a single, atomic step within a reasoning strategy.

    Each step is a self-contained unit of logic that can be composed
    with other steps to create a complete reasoning pipeline.
    """

    def __init__(self, engine: "ReasoningEngine"):
        """
        Initializes the reasoning step.

        Args:
            engine: The reasoning engine, providing access to shared components
                    like thinking, planning, and tool execution.
        """
        self.engine = engine
        self.context = engine.context
        self.agent_logger = engine.context.agent_logger
        self.strategy: Optional["ComponentBasedStrategy"] = None

    @abstractmethod
    async def execute(
        self,
        state: "BaseStrategyState",
        task: str,
        reasoning_context: "ReasoningContext",
    ) -> Optional["StrategyResult"]:
        """
        Executes the logic for this step.

        Args:
            state: The current state of the strategy.
            task: The overall task description.
            reasoning_context: The current reasoning context.

        Returns:
            - A `StrategyResult` if the step decides to terminate the current
              reasoning iteration (e.g., by finishing the task or needing to
              wait for tool output).
            - `None` if the reasoning flow should continue to the next step
              in the pipeline.
        """
        pass

