from __future__ import annotations
from typing import List
import time

from reactive_agents.core.types.reasoning_types import (
    ContinueThinkingPayload,
    EvaluationPayload,
    FinishTaskPayload,
    ReasoningContext,
    StrategyAction,
)
from reactive_agents.core.reasoning.strategies.base import (
    StrategyResult,
    StrategyCapabilities,
)
from reactive_agents.core.reasoning.strategy_components import (
    ComponentBasedStrategy,
)
from reactive_agents.core.types.session_types import (
    PlanExecuteReflectState,
    register_strategy,
)
from reactive_agents.core.types.status_types import StepStatus


@register_strategy("plan_execute_reflect", PlanExecuteReflectState)
class PlanExecuteReflectStrategy(ComponentBasedStrategy):
    """
    Plan-Execute-Reflect strategy using session-based state management.
    """

    @property
    def name(self) -> str:
        return "plan_execute_reflect"

    @property
    def capabilities(self) -> List[StrategyCapabilities]:
        return [
            StrategyCapabilities.PLANNING,
            StrategyCapabilities.TOOL_EXECUTION,
            StrategyCapabilities.REFLECTION,
        ]

    @property
    def description(self) -> str:
        return "Plan-Execute-Reflect strategy with persistent state management"

    def _get_state(self) -> PlanExecuteReflectState:
        """Get the strategy state from session, initializing if needed."""
        state = self.get_state()
        if not isinstance(state, PlanExecuteReflectState):
            raise TypeError(f"Expected PlanExecuteReflectState, got {type(state)}")
        return state

    async def initialize(self, task: str, reasoning_context: ReasoningContext) -> None:
        """Initialize the strategy for a new task."""
        state = self._get_state()
        state.reset()
        self.context.session.add_message(
            role="system",
            content=f"Role: {self.context.role}\nInstructions: {self.context.instructions}",
        )
        self.context.session.add_message(
            role="user",
            content=f"Task: {task}",
        )
        state.current_plan = await self.plan(task, reasoning_context)
        if self.agent_logger:
            self.agent_logger.info(
                f"📋 Generated plan with {len(state.current_plan.plan_steps)} steps"
            )

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """Execute one iteration of the plan-execute-reflect strategy."""
        state = self._get_state()
        session = self.context.session

        if not state.current_plan.plan_steps:
            if self.agent_logger:
                self.agent_logger.warning("No plan available, generating one...")
            state.current_plan = await self.plan(task, reasoning_context)
        for step in state.current_plan.plan_steps:
            print(
                f"Step {step.index}: {step.description} - action: {step.is_action} - status: {'✅' if step.status == StepStatus.COMPLETED else '❌' if step.status == StepStatus.FAILED else '⏳'}"
            )
        if state.current_plan.is_finished():
            if not session.final_answer:
                session.final_answer = await state.current_plan.get_final_answer(
                    self.engine
                )
            successful = state.current_plan.is_successful()
            if self.agent_logger:
                self.agent_logger.info(f"🏁 {state.current_plan.get_summary()}")

            return StrategyResult(
                action=StrategyAction.FINISH_TASK,
                payload=FinishTaskPayload(
                    action=StrategyAction.FINISH_TASK,
                    final_answer=session.final_answer,
                    evaluation=EvaluationPayload(
                        action=StrategyAction.EVALUATE_COMPLETION,
                        is_complete=successful,
                        reasoning=state.current_plan.get_summary(),
                        confidence=1.0 if successful else 0.0,
                    ),
                ),
                should_continue=False,
            )

        current_step = state.current_plan.get_next_step()
        if not current_step:
            if self.agent_logger:
                self.agent_logger.warning("No current step available, skipping...")
            return StrategyResult(
                action=StrategyAction.CONTINUE_THINKING,
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
        step_result = await self._think_chain(use_tools=use_tools)
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

        reflection_result = await self.reflect_on_progress(
            task,
            {"messages": session.get_prompt_context(last_n_messages=10)},
            reasoning_context,
        )

        if reflection_result:
            state.record_reflection_result(reflection_result)

        return StrategyResult(
            action=StrategyAction.CONTINUE_THINKING,
            payload=ContinueThinkingPayload(
                action=StrategyAction.CONTINUE_THINKING,
                reasoning=current_step.get_summary(),
            ),
            should_continue=True,
        )
