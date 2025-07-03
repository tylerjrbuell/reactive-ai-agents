from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING
from reactive_agents.core.types.reasoning_types import (
    ReasoningStrategies,
    ReasoningContext,
)
from .base import BaseReasoningStrategy

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class AdaptiveStrategy(BaseReasoningStrategy):
    """
    Adaptive reasoning strategy that dynamically selects and switches between
    other strategies based on task characteristics and performance.
    """

    def __init__(self, context: "AgentContext"):
        super().__init__(context)
        self.strategy_manager = None  # Will be set by external code

    def set_strategy_manager(self, strategy_manager):
        """Set the strategy manager for adaptive behavior."""
        self.strategy_manager = strategy_manager

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """
        Execute adaptive iteration by delegating to the strategy manager.
        The adaptive strategy itself doesn't implement specific logic,
        but rather coordinates between other strategies.
        """

        if not self.strategy_manager:
            # Fallback to reflect-decide-act if no strategy manager
            if self.agent_logger:
                self.agent_logger.warning("No strategy manager set, using fallback")

            # Import and use reflect-decide-act as fallback
            from .reflect_decide_act import ReflectDecideActStrategy

            fallback_strategy = ReflectDecideActStrategy(self.context)
            return await fallback_strategy.execute_iteration(task, reasoning_context)

        # Delegate to strategy manager
        return await self.strategy_manager.execute_iteration(task)

    def should_switch_strategy(
        self, reasoning_context: ReasoningContext
    ) -> Optional[Dict[str, Any]]:
        """
        Adaptive strategy manages switching itself, so no additional
        switching recommendations.
        """
        return None

    def get_strategy_name(self) -> ReasoningStrategies:
        """Return the strategy identifier."""
        return ReasoningStrategies.ADAPTIVE
