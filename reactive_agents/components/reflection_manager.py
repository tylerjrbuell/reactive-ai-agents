from __future__ import annotations
import json
import traceback
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime  # Added for timestamp
import time

from pydantic import BaseModel, Field

# --- Import centralized prompts ---
from reactive_agents.prompts.agent_prompts import (
    REFLECTION_CONTEXT_PROMPT,
    STEP_REFLECTION_SYSTEM_PROMPT,
)

# --- End Import ---

from reactive_agents.common.types.event_types import AgentStateEvent
from reactive_agents.common.types.session_types import StepStatus, PlanStep


# Assuming ReflectionFormat is defined here or imported
class ReflectionFormat(BaseModel):
    # completion_score: float = Field(..., ge=0.0, le=1.0) # Removed
    next_step: str  # What is the single next concrete action?
    reason: str  # Why is this the next step? Mention errors if any.
    # required_tools: List[str] = []  # Removed - let planning handle this maybe?
    completed_tools: List[str] = Field(
        ..., description="MUST exactly match the tools_used_successfully input."
    )


if TYPE_CHECKING:
    from reactive_agents.context.agent_context import AgentContext
    from reactive_agents.loggers.base import Logger
    from reactive_agents.model_providers.base import BaseModelProvider
    from reactive_agents.components.tool_manager import ToolManager


class ReflectionManager(BaseModel):
    """Handles the generation and storage of task reflections."""

    context: AgentContext = Field(exclude=True)  # Reference back to the main context

    # State
    reflections: List[Dict[str, Any]] = []  # Stores reflection dictionaries

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.reset()  # Clear reflections on init before loading
        # Load initial reflections maybe from memory manager?
        if self.context.memory_manager:
            try:
                initial_reflections = self.context.memory_manager.get_reflections()
                if initial_reflections:
                    # Check if these reflections are relevant? For now, load them.
                    # A reset might be better called externally at the start of a *run*.
                    self.reflections = initial_reflections
                    self.agent_logger.info(
                        f"Loaded {len(self.reflections)} reflections from memory."
                    )
            except Exception as e:
                self.agent_logger.warning(
                    f"Could not load initial reflections from memory: {e}"
                )

    def reset(self):
        """Clears the current list of reflections."""
        self.agent_logger.debug("Resetting reflections list.")
        self.reflections = []

    @property
    def agent_logger(self) -> Logger:
        assert self.context.agent_logger is not None
        return self.context.agent_logger

    @property
    def model_provider(self) -> BaseModelProvider:
        assert self.context.model_provider is not None
        return self.context.model_provider

    @property
    def tool_manager(self) -> ToolManager:
        assert self.context.tool_manager is not None
        return self.context.tool_manager

    def get_last_reflection(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent reflection, if any."""
        return self.reflections[-1] if self.reflections else None

    async def reflect_and_evaluate_steps(self, step_result: Any) -> dict:
        """
        Reflect on the current step result and evaluate the completion status of plan steps.
        Returns step updates and next step information.
        """
        if not self.context.reflect_enabled or not self.context.model_provider:
            return {
                "step_updates": [],
                "next_step_index": self.context.session.current_step_index,
                "plan_complete": False,
                "reason": "Reflection disabled or no model provider.",
            }

        if self.context.session.final_answer:
            return {
                "step_updates": [],
                "next_step_index": self.context.session.current_step_index,
                "plan_complete": True,
                "reason": "Final answer set, plan complete.",
            }

        # Get tool history for context
        tool_history = []
        if self.context.tool_manager:
            tool_history = self.context.tool_manager.tool_history or []

        # Prepare step information for reflection
        plan_steps = []
        for i, step in enumerate(self.context.session.plan_steps):
            step_info = {
                "index": step.index,
                "description": step.description,
                "status": step.status.value,
                "result": step.result,
                "error": step.error,
                "tool_used": step.tool_used,
                "parameters": step.parameters,
            }
            plan_steps.append(step_info)

        # Prepare reflection context
        reflection_context = {
            "task": self.context.session.current_task,
            "instructions": self.context.instructions,
            "plan_steps": json.dumps(plan_steps, indent=2),
            "current_step_index": self.context.session.current_step_index,
            "step_result": (
                json.dumps(step_result) if step_result is not None else "null"
            ),
            "tool_history": json.dumps(tool_history, indent=2),
        }

        # Format the reflection prompt
        system_prompt_formatted = STEP_REFLECTION_SYSTEM_PROMPT.format(
            **reflection_context
        )

        # Call the LLM
        response = await self.context.model_provider.get_completion(
            system=system_prompt_formatted,
            prompt=f"Evaluate the completion status of the current step and update step tracking accordingly.",
            options=self.context.model_provider_options,
        )

        step_updates = []
        next_step_index = self.context.session.current_step_index
        plan_complete = False
        reason = "No step updates."

        if response and response.message.content:
            try:
                reflection_json = json.loads(response.message.content)
                step_updates = reflection_json.get("step_updates", [])
                next_step_index = reflection_json.get(
                    "next_step_index", self.context.session.current_step_index
                )
                plan_complete = reflection_json.get("plan_complete", False)
                reason = reflection_json.get("reason", reason)
            except Exception as e:
                agent_logger = getattr(self.context, "agent_logger", None)
                if agent_logger is not None:
                    agent_logger.warning(f"Failed to parse step reflection JSON: {e}")

        return {
            "step_updates": step_updates,
            "next_step_index": next_step_index,
            "plan_complete": plan_complete,
            "reason": reason,
        }
