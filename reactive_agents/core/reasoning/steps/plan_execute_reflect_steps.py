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
from reactive_agents.core.types.status_types import StepStatus
from reactive_agents.core.reasoning.strategies.base import StrategyResult

if TYPE_CHECKING:
    from reactive_agents.core.types.session_types import (
        PlanExecuteReflectState,
        BaseStrategyState,
    )
    from reactive_agents.core.types.reasoning_types import ReasoningContext


class CheckPlanCompletionStep(BaseReasoningStep):
    """
    A reasoning step that checks if the current plan is complete.
    If it is, it generates the final answer and terminates the process.
    """

    async def execute(
        self,
        state: "BaseStrategyState",
        task: str,
        reasoning_context: "ReasoningContext",
    ) -> Optional["StrategyResult"]:

        state = cast("PlanExecuteReflectState", state)
        session = self.context.session
        summary = state.current_plan.get_summary()

        if self.agent_logger:
            self.agent_logger.info(f"Plan Summary: {summary}")

        if state.current_plan.is_finished():
            if not session.final_answer:
                session.final_answer = await state.current_plan.get_final_answer(
                    self.engine
                )

            successful = state.current_plan.is_successful()

            if self.agent_logger:
                self.agent_logger.info(f"🏁 Task Completed Successfully")

            return StrategyResult.create(
                payload=FinishTaskPayload(
                    action=StrategyAction.FINISH_TASK,
                    final_answer=session.final_answer,
                    evaluation=EvaluationPayload(
                        action=StrategyAction.EVALUATE_COMPLETION,
                        is_complete=successful,
                        reasoning=summary,
                        confidence=1.0 if successful else 0.0,
                    ),
                ),
                should_continue=False,
            )

        return None


class ExecutePlanStep(BaseReasoningStep):
    """
    A reasoning step that executes the next pending step in the plan.
    """

    async def execute(
        self,
        state: "BaseStrategyState",
        task: str,
        reasoning_context: "ReasoningContext",
    ) -> Optional["StrategyResult"]:

        state = cast("PlanExecuteReflectState", state)
        session = self.context.session

        current_step = state.current_plan.get_next_step()
        if not current_step:
            if self.agent_logger:
                self.agent_logger.warning("No current step available, skipping...")
            return StrategyResult.create(
                payload=ContinueThinkingPayload(
                    action=StrategyAction.CONTINUE_THINKING,
                    reasoning="No current step available, skipping...",
                ),
                should_continue=False,
            )

        if self.agent_logger:
            self.agent_logger.info(f"🔄 {current_step.get_summary()}")

        session.add_message(
            role="user",
            content=f"{current_step.description}",
        )

        use_tools = current_step.is_action or len(current_step.required_tools) > 0
        step_result = await self.engine.think_chain(use_tools=use_tools)
        step_result_content = step_result.content if step_result else None

        state.current_plan.update_step_status(
            current_step, step_result_content, state.max_retries_per_step
        )

        step_data = {
            "step_index": current_step.index,
            "step_description": current_step.description,
            "result": step_result_content,
            "success": (
                current_step.result.is_successful() if current_step.result else False
            ),
            "timestamp": time.time(),
        }
        state.record_step_result(step_data)

        # This step always continues to the next step in the pipeline (e.g., Reflection)
        return None


class ReflectStep(BaseReasoningStep):
    """
    A reasoning step that reflects on the progress and decides the next action.
    This step is crucial for the agent's ability to react to new information
    and correct its course.
    """

    async def execute(
        self,
        state: "BaseStrategyState",
        task: str,
        reasoning_context: "ReasoningContext",
    ) -> Optional["StrategyResult"]:

        per_state = cast("PlanExecuteReflectState", state)
        session = self.context.session

        if self.agent_logger:
            self.agent_logger.info("🤔 Reflecting on progress...")

        if not self.strategy:
            raise ValueError("Strategy not set on step")

        # Perform the reflection
        reflection_result = await self.strategy.reflect(
            task,
            {"messages": session.get_prompt_context(last_n_messages=10)},
            reasoning_context,
        )

        if self.agent_logger:
            self.agent_logger.info(f"Reflection result: {reflection_result}")

        if not reflection_result:
            if self.agent_logger:
                self.agent_logger.warning("Reflection produced no result. Continuing.")
            return StrategyResult.create(
                payload=ContinueThinkingPayload(
                    action=StrategyAction.CONTINUE_THINKING,
                    reasoning="Reflection failed, continuing.",
                ),
                should_continue=True,
            )

        # Record the reflection
        per_state.record_reflection_result(reflection_result)

        # Act on the reflection
        if reflection_result.goal_achieved:
            if self.agent_logger:
                self.agent_logger.info("Reflection suggests task is complete.")

            completion = await self.strategy.complete_task(
                task, per_state.current_plan.get_summary()
            )
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

        elif reflection_result.next_action == "retry":
            last_failed_step = per_state.current_plan.get_last_failed_step()
            if last_failed_step:
                if self.agent_logger:
                    self.agent_logger.info(
                        f"Reflection suggests retrying step {last_failed_step.index}."
                    )
                last_failed_step.status = StepStatus.PENDING
                last_failed_step.retries = 0  # Reset retries on manual retry

        # Default action is to continue to the next iteration
        return StrategyResult.create(
            payload=ContinueThinkingPayload(
                action=StrategyAction.CONTINUE_THINKING,
                reasoning=reflection_result.progress_assessment,
            ),
            should_continue=True,
        )
