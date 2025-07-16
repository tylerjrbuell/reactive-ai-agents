from __future__ import annotations
from typing import List
from reactive_agents.core.types.reasoning_types import (
    EvaluationPayload,
    FinishTaskPayload,
    ReasoningContext,
    StrategyAction,
)
from reactive_agents.core.reasoning.strategies.base import (
    BaseReasoningStrategy,
    StrategyResult,
    StrategyCapabilities,
)
from reactive_agents.core.types.session_types import (
    ReflectDecideActState,
    register_strategy,
)


@register_strategy("reflect_decide_act", ReflectDecideActState)
class ReflectDecideActStrategy(BaseReasoningStrategy):
    """
    Reflect-Decide-Act strategy skeleton for custom implementation.
    """

    @property
    def name(self) -> str:
        # Return the unique name of this strategy
        return "reflect_decide_act"  # placeholder

    @property
    def capabilities(self) -> List[StrategyCapabilities]:
        # Return a list of capabilities this strategy supports
        return []  # placeholder

    @property
    def description(self) -> str:
        # Return a short description of the strategy
        return "Reflect-Decide-Act strategy (skeleton)"  # placeholder

    def _get_state(self) -> ReflectDecideActState:
        """Get the strategy state from session, initializing if needed."""
        state = self.get_state()
        if not isinstance(state, ReflectDecideActState):
            raise TypeError(f"Expected ReflectDecideActState, got {type(state)}")
        return state

    async def initialize(self, task: str, reasoning_context: ReasoningContext) -> None:
        """
        Initialize the strategy for a new task.
        - Use `self.get_state()` to get the strategy-specific state.
        - Use `self.context.session` to access shared session data.
        """
        state = self._get_state()
        state.reset()
        if self.agent_logger:
            self.agent_logger.info("Initialized Reflect-Decide-Act Strategy.")

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """
        Execute one iteration of the reflect-decide-act strategy.
        - 1. Reflect: Use `self.reflect_on_progress()`
        - 2. Decide: Use `self.decide_next_action()`
        - 3. Act: Use `self._think_chain()` or other components.
        - Use `self.context.session.add_error()` for robust error handling.
        """
        session = self.context.session
        state = self._get_state()

        # Example of RDA cycle
        # 1. Reflect
        # reflection = await self.reflect_on_progress(...)
        # state.record_reflection_result(reflection)

        # 2. Decide
        # decision = await self.decide_next_action(...)
        # state.record_decision_result(decision)

        # 3. Act
        # action_result = await self._think_chain(...)
        # state.record_action_result(action_result)

        # Check for completion
        if session.has_failed:
            return StrategyResult(
                action=StrategyAction.FINISH_TASK,
                payload=FinishTaskPayload(
                    action=StrategyAction.FINISH_TASK,
                    final_answer="Task failed.",
                    evaluation=EvaluationPayload(
                        action=StrategyAction.EVALUATE_COMPLETION,
                        is_complete=False,
                        reasoning="Task failed.",
                        confidence=0.0,
                    ),
                ),
                should_continue=False,
            )

        return StrategyResult(
            action=StrategyAction.FINISH_TASK,
            payload=FinishTaskPayload(
                action=StrategyAction.FINISH_TASK,
                final_answer="Task failed.",
                evaluation=EvaluationPayload(
                    action=StrategyAction.EVALUATE_COMPLETION,
                    is_complete=False,
                    reasoning="Task failed.",
                    confidence=0.0,
                ),
            ),
        )
