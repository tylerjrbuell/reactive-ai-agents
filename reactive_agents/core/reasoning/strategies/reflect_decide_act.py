from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING
from reactive_agents.core.types.reasoning_types import (
    ReasoningStrategies,
    ReasoningContext,
)
from .base import BaseReasoningStrategy, StrategyResult, StrategyCapabilities
from ..infrastructure import Infrastructure

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class ReflectDecideActStrategy(BaseReasoningStrategy):
    """
    Simple Reflect-Decide-Act strategy.
    Each iteration: reflect on progress, decide next action, then execute it.
    Uses the simplified infrastructure for all operations.
    """

    @property
    def name(self) -> str:
        return "reflect_decide_act"

    @property
    def capabilities(self) -> list[StrategyCapabilities]:
        return [
            StrategyCapabilities.REFLECTION,
            StrategyCapabilities.PLANNING,
            StrategyCapabilities.TOOL_EXECUTION,
        ]

    def __init__(self, infrastructure: "Infrastructure"):
        super().__init__(infrastructure)

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """Execute reflect-decide-act iteration."""
        try:
            if self.agent_logger:
                self.agent_logger.debug("🔄 Reflect-Decide-Act iteration")

            # Phase 1: Reflect on current progress
            reflection = await self._reflect_phase(task, reasoning_context)

            # Phase 2: Decide next action based on reflection
            decision = await self._decide_phase(task, reflection)

            # Phase 3: Act on the decision
            action_result = await self._act_phase(decision)

            # Preserve the iteration context
            self._preserve_context(
                f"rda_iteration_{reasoning_context.iteration_count}",
                {
                    "reflection": reflection,
                    "decision": decision,
                    "action": action_result,
                },
            )

            return StrategyResult(
                action_taken="reflect_decide_act",
                result={
                    "reflection": reflection,
                    "decision": decision,
                    "action": action_result,
                },
                should_continue=action_result.get("should_continue", True),
                strategy_used="reflect_decide_act",
                final_answer=action_result.get("final_answer"),
            )

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Reflect-Decide-Act iteration failed: {e}")
            return self._format_error_result(e, "reflect_decide_act")

    async def _reflect_phase(
        self, task: str, reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """Reflect on current progress and state."""
        if self.agent_logger:
            self.agent_logger.debug("🤔 Phase 1: Reflection")

        # Use infrastructure's enhanced reflection method with full context
        reflection_text = await self.infrastructure.reflect(
            task=task,
            progress=f"Iteration {reasoning_context.iteration_count}",
            last_result=str(reasoning_context.last_action_result or "None"),
        )

        if not reflection_text:
            return {
                "progress_assessment": "Unable to generate reflection",
                "completion_score": 0.5,
                "next_action_type": "continue",
                "reasoning": "Reflection generation failed, continuing",
            }

        # Use centralized completion check
        completion_info = await self.check_task_completion(task)
        is_complete = completion_info.get("is_complete", False)

        return {
            "progress_assessment": reflection_text,
            "completion_score": completion_info.get("confidence", 0.5),
            "next_action_type": "finalize" if is_complete else "continue",
            "reasoning": "Based on current progress analysis",
        }

    async def _decide_phase(
        self, task: str, reflection: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decide on the next best action based on reflection."""
        if self.agent_logger:
            self.agent_logger.debug("🎯 Phase 2: Decision")

        # If reflection suggests completion
        if reflection.get("next_action_type") == "finalize":
            return {
                "next_step": "Provide final answer",
                "rationale": "Task appears complete based on reflection",
                "action_type": "finalize",
                "confidence": reflection.get("completion_score", 0.8),
            }

        # Otherwise, continue with next step
        return {
            "next_step": "Continue task execution",
            "rationale": "More work needed based on reflection",
            "action_type": "continue",
            "confidence": 0.7,
        }

    async def _act_phase(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the decided action."""
        if self.agent_logger:
            self.agent_logger.debug("⚡ Phase 3: Action")

        action_type = decision.get("action_type", "continue")

        if action_type == "finalize":
            # Use centralized final answer generation
            final_answer = await self.generate_final_answer(
                task=decision.get("next_step", "the current task"),
                execution_summary=f"Reflect-Decide-Act strategy finalization after reflection analysis",
            )

            if final_answer:
                return {
                    "action_type": "finalize",
                    "result": final_answer,
                    "should_continue": False,
                    "final_answer": final_answer,
                }
            else:
                # Fallback if centralized generation fails
                return {
                    "action_type": "finalize_failed",
                    "result": "Failed to generate final answer",
                    "should_continue": True,
                }

        # Continue with task execution
        prompt = f"Continue working on the task. {decision.get('rationale', '')}"
        result = await self._think_chain(use_tools=True)

        return {
            "action_type": "continue",
            "result": (
                result.get("content", "Continued task execution")
                if result
                else "Action failed"
            ),
            "should_continue": True,
        }

    def should_switch_strategy(
        self, reasoning_context: ReasoningContext
    ) -> Optional[str]:
        """Consider switching strategy based on context."""
        # Switch to planning if many iterations without progress
        if (
            reasoning_context.iteration_count >= 5
            and reasoning_context.stagnation_count >= 3
        ):
            return "plan_execute_reflect"

        # Switch to reactive if task seems simple
        if (
            reasoning_context.iteration_count >= 2
            and reasoning_context.error_count == 0
        ):
            completion_hints = ["simple", "quick", "direct", "straightforward"]
            last_result = str(reasoning_context.last_action_result or "")
            if any(hint in last_result.lower() for hint in completion_hints):
                return "reactive"

        return None

    def get_strategy_name(self) -> ReasoningStrategies:
        """Return the strategy identifier."""
        return ReasoningStrategies.REFLECT_DECIDE_ACT
