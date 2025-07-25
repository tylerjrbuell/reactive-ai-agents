from __future__ import annotations
import json
from typing import Optional, TYPE_CHECKING, cast

from reactive_agents.core.reasoning.steps.base import BaseReasoningStep
from reactive_agents.core.reasoning.strategies.base import StrategyResult
from reactive_agents.core.types.reasoning_types import (
    ContinueThinkingPayload,
    EvaluationPayload,
    FinishTaskPayload,
    StrategyAction,
)

if TYPE_CHECKING:
    from reactive_agents.core.types.session_types import (
        ReflectDecideActState,
        BaseStrategyState,
    )
    from reactive_agents.core.types.reasoning_types import ReasoningContext


class ReflectOnSituationStep(BaseReasoningStep):
    """A step to reflect on the current situation and task progress."""

    async def execute(
        self,
        state: "BaseStrategyState",
        task: str,
        reasoning_context: "ReasoningContext",
    ) -> Optional["StrategyResult"]:

        rda_state = cast("ReflectDecideActState", state)
        session = self.context.session

        if self.agent_logger:
            self.agent_logger.info("🤔 Reflecting on the situation...")

        if not self.strategy:
            raise ValueError("Strategy not set on step")

        reflection_result = await self.strategy.reflect(
            task,
            {"messages": session.get_prompt_context(last_n_messages=10)},
            reasoning_context,
        )

        if reflection_result:
            rda_state.record_reflection_result(reflection_result.model_dump())

        if self.agent_logger:
            self.agent_logger.info(
                f"Reflection result: {json.dumps(reflection_result.model_dump(), indent=2)}"
            )

        if reflection_result.goal_achieved:
            if self.agent_logger:
                self.agent_logger.info("🎉 Goal achieved. Ending iteration.")
            return StrategyResult.create(
                payload=FinishTaskPayload(
                    action=StrategyAction.FINISH_TASK,
                    final_answer="Task Completed Successfully",
                    evaluation=EvaluationPayload(
                        action=StrategyAction.EVALUATE_COMPLETION,
                        is_complete=True,
                        reasoning=reflection_result.progress_assessment,
                        confidence=reflection_result.confidence,
                    ),
                ),
                should_continue=False,
            )

        return None  # Continue to Decide step


class DecideNextActionStep(BaseReasoningStep):
    """A step to decide on the next action based on reflection."""

    async def execute(
        self,
        state: "BaseStrategyState",
        task: str,
        reasoning_context: "ReasoningContext",
    ) -> Optional["StrategyResult"]:

        rda_state = cast("ReflectDecideActState", state)

        if self.agent_logger:
            self.agent_logger.info("🤔 Deciding on the next action...")

        if not self.strategy:
            raise ValueError("Strategy not set on step")

        # Create a prompt for the decision
        prompt = self.strategy.engine.get_prompt(
            "single_step_planning",
            task=task,
            reflection=(
                rda_state.reflection_history[-1]
                if rda_state.reflection_history
                else None
            ),
        )

        decision_result = await prompt.get_completion()

        if decision_result and decision_result.result_json:
            # print(json.dumps(decision_result.result_json, indent=2))
            rda_state.record_decision_result(decision_result.result_json)
            rda_state.current_action = decision_result.result_json

        return None  # Continue to Execute step


class ExecuteActionStep(BaseReasoningStep):
    """A step to execute the decided action."""

    async def execute(
        self,
        state: "BaseStrategyState",
        task: str,
        reasoning_context: "ReasoningContext",
    ) -> Optional["StrategyResult"]:

        rda_state = cast("ReflectDecideActState", state)
        # print(rda_state)
        if not rda_state.current_action:
            if self.agent_logger:
                self.agent_logger.warning("No action decided. Ending iteration.")
            return StrategyResult.create(
                payload=ContinueThinkingPayload(
                    action=StrategyAction.CONTINUE_THINKING,
                    reasoning="No action to execute.",
                ),
                should_continue=False,
            )

        action = rda_state.current_action.get("next_step")
        action_input = rda_state.current_action.get("parameters", {})
        rationale = rda_state.current_action.get("rationale")

        self.context.session.add_message(
            role="user",
            content=f"Executing action: {action} with input: {action_input} and rationale: {rationale}",
        )

        if self.agent_logger:
            self.agent_logger.info(
                f"⚡ Executing action: {action} with input: {action_input} and rationale: {rationale}"
            )

        # This is a simplified execution. A real implementation would
        # use the tool execution component or thinking component based on the action.
        result = await self.engine.think_chain(use_tools=True)

        rda_state.record_action_result(result.model_dump() if result else {})

        return StrategyResult.create(
            payload=ContinueThinkingPayload(
                action=StrategyAction.CONTINUE_THINKING,
                reasoning="Action executed.",
            ),
            should_continue=True,
        )
