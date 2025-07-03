from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING
from reactive_agents.core.types.reasoning_types import (
    ReasoningStrategies,
    ReasoningContext,
)
from .base import BaseReasoningStrategy

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class PlanExecuteReflectStrategy(BaseReasoningStrategy):
    """
    Plan-Execute-Reflect strategy: Traditional static planning approach.
    Generates a plan upfront and executes it step by step.
    Maintains backwards compatibility with existing system.
    """

    def __init__(self, context: "AgentContext"):
        super().__init__(context)
        self.plan_generated = False

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """Execute plan-execute-reflect iteration."""

        if self.agent_logger:
            self.agent_logger.info("🔄 Plan-Execute-Reflect Strategy")

        # Generate plan if not already done
        if not self.plan_generated:
            await self._generate_plan(task)
            self.plan_generated = True

        # Execute current step in plan
        step_result = await self._execute_current_step()

        # Reflect on progress
        reflection = await self._reflect_on_plan_progress()

        return {
            "action_taken": "plan_execute_reflect",
            "result": {"step_result": step_result, "reflection": reflection},
            "should_continue": self._should_continue_plan(),
            "strategy_used": "plan_execute_reflect",
        }

    async def _generate_plan(self, task: str):
        """Generate initial plan using the existing plan manager."""
        if hasattr(self.context, "session") and hasattr(
            self.context.session, "plan_steps"
        ):
            # Use existing plan manager if available
            plan_manager = getattr(self.context, "plan_manager", None)
            if plan_manager:
                await plan_manager.generate_and_store_plan(task)
            else:
                # Fallback plan generation
                self.context.session.plan = [
                    {
                        "description": "Analyze the task requirements",
                        "is_action": False,
                    },
                    {"description": "Execute necessary actions", "is_action": True},
                    {"description": "Provide final answer", "is_action": True},
                ]

        if self.agent_logger:
            self.agent_logger.info("📋 Generated plan for execution")

    async def _execute_current_step(self) -> Dict[str, Any]:
        """Execute the current step in the plan."""
        if not hasattr(self.context, "session") or not self.context.session.plan_steps:
            return {"error": "No plan steps available"}

        # Get current step
        current_index = getattr(self.context.session, "current_step_index", 0)
        if current_index >= len(self.context.session.plan_steps):
            return {"completed": True, "message": "All plan steps completed"}

        current_step = self.context.session.plan_steps[current_index]

        if self.agent_logger:
            self.agent_logger.info(
                f"📝 Executing step {current_index + 1}: {current_step.description}"
            )

        # Execute step based on type
        if getattr(current_step, "is_action", False):
            # Action step - may require tool use
            tool_result = await self._execute_tool_if_needed(
                {
                    "tool_needed": (
                        "final_answer"
                        if "final" in current_step.description.lower()
                        else None
                    ),
                    "parameters": {"answer": "Step completed"},
                }
            )
            return {"tool_result": tool_result, "step_index": current_index}
        else:
            # Reasoning step - add to messages
            self.context.session.messages.append(
                {
                    "role": "assistant",
                    "content": f"Planning step: {current_step.description}",
                }
            )
            return {
                "reasoning_step": current_step.description,
                "step_index": current_index,
            }

    async def _reflect_on_plan_progress(self) -> Dict[str, Any]:
        """Reflect on plan execution progress."""
        if not hasattr(self.context, "session") or not self.context.session.plan_steps:
            return {"error": "No plan to reflect on"}

        total_steps = len(self.context.session.plan_steps)
        current_index = getattr(self.context.session, "current_step_index", 0)

        progress = (current_index + 1) / total_steps if total_steps > 0 else 0

        return {
            "progress": progress,
            "steps_completed": current_index + 1,
            "total_steps": total_steps,
            "assessment": f"Completed {current_index + 1} of {total_steps} steps",
        }

    def _should_continue_plan(self) -> bool:
        """Determine if plan execution should continue."""
        if not hasattr(self.context, "session"):
            return False

        # Check if final answer is set
        if getattr(self.context.session, "final_answer", None):
            return False

        # Check if all steps completed
        if hasattr(self.context.session, "plan_steps"):
            current_index = getattr(self.context.session, "current_step_index", 0)
            return current_index < len(self.context.session.plan_steps)

        return True

    def should_switch_strategy(
        self, reasoning_context: ReasoningContext
    ) -> Optional[Dict[str, Any]]:
        """Plan-execute-reflect rarely switches, but may if plan fails repeatedly."""

        # Switch to reflect-decide-act if many errors
        if reasoning_context.error_count >= 3:
            return {
                "to_strategy": ReasoningStrategies.REFLECT_DECIDE_ACT,
                "reason": "Plan execution encountering too many errors, switching to adaptive approach",
                "confidence": 0.8,
                "trigger": "plan_execution_failures",
            }

        return None

    def get_strategy_name(self) -> ReasoningStrategies:
        """Return the strategy identifier."""
        return ReasoningStrategies.PLAN_EXECUTE_REFLECT
