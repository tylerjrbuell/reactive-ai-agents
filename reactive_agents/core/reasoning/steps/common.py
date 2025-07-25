from __future__ import annotations
import time
from typing import Optional, TYPE_CHECKING, cast

from reactive_agents.core.reasoning.steps.base import BaseReasoningStep
from reactive_agents.core.types.reasoning_types import (
    ContinueThinkingPayload,
    FinishTaskPayload,
    EvaluationPayload,
    StrategyAction,
)
from reactive_agents.core.types.session_types import (
    PlanExecuteReflectState,
    ReactiveState,
    BaseStrategyState,
)
from reactive_agents.core.reasoning.strategies.base import StrategyResult

if TYPE_CHECKING:
    from reactive_agents.core.types.reasoning_types import ReasoningContext


class EvaluateTaskCompletionStep(BaseReasoningStep):
    """
    A reasoning step that evaluates the completion of the task.
    """

    async def execute(
        self,
        state: "BaseStrategyState",
        task: str,
        reasoning_context: "ReasoningContext",
    ) -> Optional["StrategyResult"]:

        if self.agent_logger:
            self.agent_logger.info("🔎 Evaluating task completion...")

        if not self.strategy:
            raise ValueError("Strategy not set on step")

        # Get a summary of what has been done.
        # This part is strategy-specific.
        summary = ""
        if isinstance(state, PlanExecuteReflectState):
            summary = state.current_plan.get_summary()
        elif isinstance(state, ReactiveState):
            summary = state.get_execution_summary().get("last_response", "")

        # Use the evaluation component
        evaluation_result = await self.strategy.evaluate(task, progress_summary=summary)

        if evaluation_result.is_complete:
            if self.agent_logger:
                self.agent_logger.info("Evaluation confirms task is complete.")

            completion = await self.strategy.complete_task(task, summary)
            return StrategyResult.create(
                payload=FinishTaskPayload(
                    action=StrategyAction.FINISH_TASK,
                    final_answer=completion.final_answer or "Task completed.",
                    evaluation=EvaluationPayload(
                        action=StrategyAction.EVALUATE_COMPLETION,
                        is_complete=True,
                        reasoning=completion.reasoning,
                        confidence=completion.confidence,
                    ),
                ),
                should_continue=False,
            )

        if self.agent_logger:
            self.agent_logger.info("Task not yet complete. Continuing iteration.")

        # If not complete, continue the loop
        return StrategyResult.create(
            payload=ContinueThinkingPayload(
                action=StrategyAction.CONTINUE_THINKING,
                reasoning=evaluation_result.reasoning,
            ),
            should_continue=True,
        )
