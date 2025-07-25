from __future__ import annotations
from typing import Optional, TYPE_CHECKING, cast

from reactive_agents.core.reasoning.steps.base import BaseReasoningStep
from reactive_agents.core.types.reasoning_types import (
    ContinueThinkingPayload,
    StrategyAction,
)

if TYPE_CHECKING:
    from reactive_agents.core.types.session_types import (
        ReactiveState,
        BaseStrategyState,
    )
    from reactive_agents.core.types.reasoning_types import ReasoningContext
    from reactive_agents.core.reasoning.strategies.base import StrategyResult


class ReactiveActStep(BaseReasoningStep):
    """
    A reasoning step that performs a single reactive action (thinking and/or tool use).
    This step does not terminate the iteration, allowing for an evaluation step to follow.
    """

    async def execute(
        self,
        state: "BaseStrategyState",
        task: str,
        reasoning_context: "ReasoningContext",
    ) -> Optional["StrategyResult"]:

        reactive_state = cast("ReactiveState", state)

        if self.agent_logger:
            self.agent_logger.info("⚡ Taking a reactive action...")

        # Use think_chain to get a direct response, potentially with tool calls
        think_result = await self.engine.think_chain(use_tools=True)

        if not think_result:
            # If thinking fails, we stop the iteration but allow the agent to continue
            return StrategyResult.create(
                payload=ContinueThinkingPayload(
                    action=StrategyAction.CONTINUE_THINKING,
                    reasoning="Thinking failed, stopping iteration.",
                ),
                should_continue=False,
            )

        # Record the result of the action
        reactive_state.record_response_result(think_result.model_dump())

        # Return None to allow the pipeline to proceed to the next step (evaluation)
        return None
