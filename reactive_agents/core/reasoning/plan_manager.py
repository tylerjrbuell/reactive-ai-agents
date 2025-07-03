from typing import Any, List, Optional
import json
from reactive_agents.core.context.agent_context import AgentContext
from reactive_agents.core.reasoning.prompts.agent_prompts import HYBRID_TASK_PLANNING_SYSTEM_PROMPT
import time
from reactive_agents.core.types.session_types import PlanStep, StepStatus


class PlanManager:
    """
    Handles plan generation and storage for the agent.
    """

    def __init__(self, context: AgentContext):
        self.context = context
        self.agent_logger = context.get_logger()
        self.model_provider = context.get_model_provider()

    async def generate_and_store_plan(
        self, task: str, failure_context: Optional[str] = None
    ):
        """
        Generate a stepwise plan using the planning prompt and store it in the session.
        Optionally include failure context to help the planner adapt.
        """
        self.agent_logger.info("📝 Generating stepwise plan for the task.")
        available_tools = [
            f"Name: {tool['function']['name']} - Description: {tool['function']['description'][:100]} - Parameters: {tool['function']['parameters']}"
            for tool in self.context.get_tool_signatures()
        ]
        required_tools = list(self.context.session.min_required_tools or [])
        plan_prompt = (
            HYBRID_TASK_PLANNING_SYSTEM_PROMPT
            + f"\n\nTask: {task}\nAvailable Tools: {available_tools}\nRequired Tools: {required_tools}"
        )
        if failure_context:
            plan_prompt += f"\n\nPrevious failure or error: {failure_context}"
        response = await self.model_provider.get_completion(
            system=plan_prompt,
            prompt="Decompose the task into a minimal, stepwise plan.",
            options=self.context.model_provider_options,
        )
        plan = []
        if response and response.message.content:
            try:
                plan_json = json.loads(response.message.content)
                plan = plan_json.get("plan", [])
            except Exception as e:
                self.agent_logger.warning(f"Failed to parse plan JSON: {e}")
        if not plan:
            self.agent_logger.warning(
                "No plan generated, using fallback single-step plan."
            )
            plan = [
                {
                    "description": "Provide the final answer to the user.",
                    "is_action": True,
                }
            ]
        self.context.session.plan = plan
        self.context.session.plan_last_modified = time.time()
        # Initialize plan_steps with new format
        plan_steps = []
        for i, step in enumerate(plan):
            if isinstance(step, dict):
                plan_steps.append(
                    PlanStep(
                        index=i,
                        description=step.get("description", str(step)),
                        is_action=step.get("is_action", None),
                    )
                )
            else:
                plan_steps.append(
                    PlanStep(index=i, description=str(step), is_action=None)
                )
        self.context.session.plan_steps = plan_steps
        self.context.session.current_step_index = 0
        try:
            plan_str = "\n".join(
                [
                    json.dumps(
                        step if isinstance(step, dict) else {"description": str(step)},
                        ensure_ascii=False,
                    )
                    for step in plan
                ]
            )
        except Exception as e:
            plan_str = f"<Failed to stringify plan: {e}>"

        self.agent_logger.info(f"Generated plan: {plan_str}")
        self.context.session.messages.append(
            {
                "role": "assistant",
                "content": f"I have generated the following plan:\n{plan_str}",
            }
        )

    def get_current_step(self) -> Optional[PlanStep]:
        idx = self.context.session.current_step_index
        if idx is None or idx < 0:
            idx = 0
        if 0 <= idx < len(self.context.session.plan_steps):
            return self.context.session.plan_steps[idx]
        # Fallback: return next pending step if any
        for step in self.context.session.plan_steps:
            if step.status == StepStatus.PENDING:
                self.context.session.current_step_index = step.index
                return step
        return None

    def get_next_pending_step(self) -> Optional[PlanStep]:
        for step in self.context.session.plan_steps:
            if step.status == StepStatus.PENDING:
                return step
        return None

    def mark_step_status(
        self, step_index: int, status: StepStatus, error: Optional[str] = None
    ):
        if 0 <= step_index < len(self.context.session.plan_steps):
            step = self.context.session.plan_steps[step_index]
            step.status = status
            if error:
                step.error = error

    def reset_plan_steps(self):
        for step in self.context.session.plan_steps:
            step.status = StepStatus.PENDING
            step.retry_count = 0
            step.result = None
            step.error = None
            step.completed_at = None
            step.tool_used = None
            step.parameters = None

    def is_plan_complete(self) -> bool:
        if not self.context.session.plan_steps:
            return False
        return all(
            step.status == StepStatus.COMPLETED
            for step in self.context.session.plan_steps
        )

    def get_step(self, index: int) -> Optional[PlanStep]:
        if 0 <= index < len(self.context.session.plan_steps):
            return self.context.session.plan_steps[index]
        return None
