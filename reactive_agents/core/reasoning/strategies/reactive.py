from __future__ import annotations
from typing import List

from reactive_agents.core.types.reasoning_types import EvaluationPayload, FinishTaskPayload, ReasoningContext, StrategyAction
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
    
    def _get_state(self) -> ReactiveState:
        """Get the strategy state from session, initializing if needed."""
        state = self.get_state()
        if not isinstance(state, ReactiveState):
            raise TypeError(f"Expected ReactiveState, got {type(state)}")
        return state

    async def initialize(self, task: str, reasoning_context: ReasoningContext) -> None:
        """
        Initialize the strategy for a new task.
        - Use `self.context.session` to access and modify session state.
        - Example: self.context.session.add_message("assistant", "Initializing Reactive Strategy.")
        """
        state = self._get_state()
        state.reset()
        if self.agent_logger:
            self.agent_logger.info("Initialized Reactive Strategy.")

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """
        Execute one iteration of the reactive strategy.
        - Use `self.context.session` for state, e.g., `self.context.session.has_failed`.
        - Use `self._think_chain()` to call the LLM.
        - Return a StrategyResult indicating progress or completion.
        """
        session = self.context.session
        state = self._get_state()

        # Example: Add an error and finish if something goes wrong
        if state.error_count > state.max_errors:
            session.add_error("ReactiveStrategy", {"message": "Max errors reached"}, is_critical=True)
            return StrategyResult(
                action=StrategyAction.FINISH_TASK,
                payload=FinishTaskPayload(
                    action=StrategyAction.FINISH_TASK,
                    final_answer="Task failed due to excessive errors.",
                    evaluation=EvaluationPayload(
                        action=StrategyAction.EVALUATE_COMPLETION,
                        is_complete=False,
                        reasoning="Max errors reached",
                        confidence=0.0,
                    ),
                ),
                should_continue=False,
            )

        return StrategyResult(
            action=StrategyAction.FINISH_TASK,
            payload=FinishTaskPayload(
                action=StrategyAction.FINISH_TASK,
                final_answer="Task failed due to excessive errors.",
                evaluation=EvaluationPayload(
                    action=StrategyAction.EVALUATE_COMPLETION,
                    is_complete=False,
                    reasoning="Max errors reached",
                    confidence=0.0,
                ),
            ),
        )
