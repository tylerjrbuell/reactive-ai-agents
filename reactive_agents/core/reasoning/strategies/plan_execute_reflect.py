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


@register_strategy("plan_execute_reflect", PlanExecuteReflectState)
class PlanExecuteReflectStrategy(ComponentBasedStrategy):
    """
    Plan-Execute-Reflect strategy using session-based state management.

    Benefits of session.strategy_state:
    - Persistent across iterations and strategy switches
    - Type-safe with Pydantic models
    - Built-in state management methods
    - Survives context resets
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

        # Reset state for new task
        state.current_step = 0
        state.execution_history.clear()
        state.error_count = 0
        state.completed_actions.clear()
        state.reflection_count = 0
        state.reflection_history.clear()

        # Generate initial plan
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

        # Check if we have a plan
        if not state.current_plan.plan_steps:
            if self.agent_logger:
                self.agent_logger.warning("No plan available, generating one...")
            state.current_plan = await self.plan(task, reasoning_context)
            state.current_step = 0

        # Check if we've completed all steps
        if state.current_step >= len(state.current_plan.plan_steps):
            if self.agent_logger:
                self.agent_logger.info("✅ All plan steps completed")
            return StrategyResult(
                action=StrategyAction.FINISH_TASK,
                payload=FinishTaskPayload(
                    action=StrategyAction.FINISH_TASK,
                    final_answer=self.context.session.final_answer
                    or "Task Plan completed successfully",
                    evaluation=EvaluationPayload(
                        action=StrategyAction.EVALUATE_COMPLETION,
                        is_complete=True,
                        reasoning="\n".join(
                            state.last_reflection_result.learning_insights
                            if state.last_reflection_result
                            else []
                        ),
                        confidence=state.last_reflection_result.completion_score
                        if state.last_reflection_result
                        else 1.0,
                    ),
                ),
                should_continue=False,
            )

        # Execute current step
        current_step = state.current_plan.plan_steps[state.current_step]

        if self.agent_logger:
            self.agent_logger.info(
                f"🔄 Executing step {state.current_step + 1}: {current_step.description}"
            )

            # Add step to context
            self.context_manager.add_message(
                role="user",
                content=f"Step {state.current_step + 1}: {current_step.description}",
            )

        # Execute the step
        step_result = await self._think_chain(use_tools=current_step.is_action)

        # Record step result
        step_data = {
            "step_index": state.current_step,
            "step_description": current_step.description,
            "result": step_result.content if step_result else None,
            "success": step_result is not None,
            "timestamp": time.time(),
        }
        state.record_step_result(step_data)

        # Reflect on progress
        reflection_result = await self.reflect_on_progress(
            task,
            {"messages": self.context_manager.get_latest_n_messages(10)},
            reasoning_context,
        )

        # Record reflection
        if reflection_result:
            state.record_reflection_result(reflection_result)

        # Determine next action
        goal_achieved = (
            reflection_result.goal_achieved
            if reflection_result
            else False
        )

        if goal_achieved:
            if self.agent_logger:
                self.agent_logger.info("✅ Goal achieved according to reflection")
            return StrategyResult(
                action=StrategyAction.FINISH_TASK,
                payload=FinishTaskPayload(
                    action=StrategyAction.FINISH_TASK,
                    final_answer=self.context.session.final_answer
                    or "Goal achieved according to reflection",
                    evaluation=EvaluationPayload(
                        action=StrategyAction.EVALUATE_COMPLETION,
                        is_complete=True,
                        reasoning="\n".join(
                            reflection_result.learning_insights
                            if reflection_result
                            else []
                        ),
                        confidence=reflection_result.completion_score
                        if reflection_result
                        else 1.0,
                    ),
                ),
                should_continue=False,
            )
        else:
            # Move to next step
            state.current_step += 1
            state.completed_actions.append(current_step.description)

            if self.agent_logger:
                self.agent_logger.info(f"📈 Moving to step {state.current_step + 1}")

        return StrategyResult(
            action=StrategyAction.CONTINUE_THINKING,
            payload=ContinueThinkingPayload(
                action=StrategyAction.CONTINUE_THINKING,
                reasoning="Continuing to next step",
            ),
            should_continue=True,
        )
