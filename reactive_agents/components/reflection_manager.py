from __future__ import annotations
import json
import traceback
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime  # Added for timestamp

from pydantic import BaseModel, Field

# --- Import centralized prompts ---
from reactive_agents.prompts.agent_prompts import (
    REFLECTION_SYSTEM_PROMPT,
    REFLECTION_CONTEXT_PROMPT,
)

# --- End Import ---

from reactive_agents.common.types.event_types import AgentStateEvent


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

    async def generate_reflection(
        self, think_act_result: Any
    ) -> Optional[Dict[str, Any]]:
        """Generate reflection on the current state."""
        if not self.context.reflect_enabled or not self.context.model_provider:
            return None

        try:
            # Get tool history
            tool_history = (
                self.context.tool_manager.tool_history
                if self.context.tool_manager
                else []
            )
            tools_used = [t.get("name") for t in tool_history if t.get("name")]
            tools_used_successfully = [
                t.get("name")
                for t in tool_history
                if t.get("name") and not t.get("error", False)
            ]

            # Get last tool action
            last_tool_action = ""
            if tool_history:
                last_tool = tool_history[-1]
                last_tool_action = f"Tool: {last_tool.get('name')}\nParams: {json.dumps(last_tool.get('params', {}), indent=2)}\nResult: {last_tool.get('result', 'No result')}"

            # Get min_required_tools from session or use empty list
            min_required_tools = (
                getattr(self.context.session, "min_required_tools", set()) or set()
            )

            # Format the reflection prompt
            reflection_context = {
                "task": self.context.session.current_task,
                "instructions": self.context.instructions,
                "min_required_tools": list(min_required_tools),
                "tools_used": tools_used,
                "last_result": think_act_result,
                "last_tool_action": last_tool_action,
            }

            # Get reflection from model
            system_prompt_formatted = REFLECTION_SYSTEM_PROMPT.format(
                **reflection_context
            )
            response = await self.context.model_provider.get_completion(
                system=system_prompt_formatted,
                prompt="Evaluate the current state and determine the next step.",
            )

            if response and response.get("response"):
                try:
                    reflection_data = json.loads(response["response"])
                    self.reflections.append(reflection_data)
                    # Emit REFLECTION_GENERATED event
                    self.context.emit_event(
                        AgentStateEvent.REFLECTION_GENERATED,
                        {"reflection": reflection_data},
                    )
                    return reflection_data
                except json.JSONDecodeError as e:
                    if self.context.agent_logger:
                        self.context.agent_logger.warning(
                            f"Reflection LLM did not return valid JSON. Raw response: {response['response']!r}. Error: {e}"
                        )
                    # Fallback: return a minimal default reflection
                    return {
                        "next_step": "No valid reflection generated. Proceed to next step or retry.",
                        "reason": "LLM did not return valid JSON for reflection.",
                        "completed_tools": tools_used_successfully,
                    }
            else:
                if self.context.agent_logger:
                    self.context.agent_logger.warning(
                        f"Reflection LLM returned empty or no response."
                    )
                return {
                    "next_step": "No reflection response. Proceed to next step or retry.",
                    "reason": "LLM returned empty response for reflection.",
                    "completed_tools": tools_used_successfully,
                }

        except Exception as e:
            if self.context.agent_logger:
                self.context.agent_logger.error(
                    f"Unexpected error during reflection generation: {e}"
                )
            traceback.print_exc()
            return None

    def get_last_reflection(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent reflection, if any."""
        return self.reflections[-1] if self.reflections else None
