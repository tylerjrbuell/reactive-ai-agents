from __future__ import annotations
import json
import traceback
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime  # Added for timestamp

from pydantic import BaseModel, Field

# --- Import centralized prompts ---
from prompts.agent_prompts import (
    REFLECTION_SYSTEM_PROMPT,
    REFLECTION_CONTEXT_PROMPT,
)

# --- End Import ---


# Assuming ReflectionFormat is defined here or imported
class ReflectionFormat(BaseModel):
    completion_score: float = Field(..., ge=0.0, le=1.0)
    next_step: str
    required_tools: List[str] = []  # Default to empty list
    completed_tools: List[str] = []  # Default to empty list
    reason: str


if TYPE_CHECKING:
    from context.agent_context import AgentContext
    from loggers.base import Logger
    from model_providers.base import BaseModelProvider
    from components.tool_manager import ToolManager


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
        self, last_step_result: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Generates a reflection on the current task progress based on the last result.

        Args:
            last_step_result: The dictionary result from the last agent execution step (e.g., model response).

        Returns:
            A dictionary containing reflection data (score, next_step, etc.), or None if reflection fails.
        """
        if not self.context.reflect_enabled:
            self.agent_logger.debug("Reflection disabled, skipping generation.")
            return None

        self.agent_logger.info("🤔 Reflecting on task progress...")

        try:
            # --- Prepare Context for Reflection ---
            result_content = ""
            if last_step_result:
                # Extract content intelligently from various possible structures
                if isinstance(last_step_result, dict):
                    message = last_step_result.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content")
                        if content:
                            result_content = str(content)
                        else:  # Check for tool calls as result proxy? Unlikely useful here.
                            result_content = (
                                f"Assistant proposed actions: {message.get('tool_calls')}"
                                if message.get("tool_calls")
                                else str(message)
                            )
                    else:  # If message is not a dict
                        result_content = str(message)
                elif isinstance(last_step_result, str):
                    result_content = last_step_result
                else:  # Fallback for other types
                    result_content = str(last_step_result)

            # Get tool info from ToolManager
            available_tool_names = [tool.name for tool in self.tool_manager.tools]

            reflection_input_context = {
                "task": self.context.current_task,  # Use current (possibly rescoped) task
                "last_result": (
                    result_content[:3000] + "..."
                    if len(result_content) > 3000
                    else result_content
                ),  # Limit result size
                "tools_available": available_tool_names,
                "tools_used_successfully": list(
                    set(self.context.successful_tools)
                ),  # Unique successful tools
                "completion_criteria_score": self.context.min_completion_score,
                "current_iteration": self.context.iterations,
                "max_iterations": self.context.max_iterations,
                "previous_reflections": self.reflections[-3:],  # Provide recent context
                "task_progress_summary": self.context.task_progress[
                    -1000:
                ],  # Summary of actions so far
            }
            self.agent_logger.debug(
                f"Reflection Tool Use context: {reflection_input_context['tools_used_successfully']}"
            )
            # --- Call Model using centralized prompts ---
            reflection_context = REFLECTION_CONTEXT_PROMPT.format(
                reflection_input_context_json=json.dumps(
                    reflection_input_context, indent=2, default=str
                )
            )
            model_response = await self.model_provider.get_completion(
                system=REFLECTION_SYSTEM_PROMPT,  # Use imported constant
                prompt=reflection_context,
                format=(
                    ReflectionFormat.model_json_schema()
                    if self.model_provider.name == "ollama"
                    else "json"  # Adapt based on provider capabilities
                ),
            )

            if not model_response or not model_response.get("response"):
                self.agent_logger.warning(
                    "Reflection model call failed or returned empty response."
                )
                return None

            # --- Parse and Store Reflection ---
            try:
                response_data = model_response["response"]
                reflection_data = (
                    json.loads(response_data)
                    if isinstance(response_data, str)
                    else response_data
                )

                # Validate with Pydantic model
                validated_reflection = ReflectionFormat(**reflection_data).dict()
                if validated_reflection.get("next_step"):
                    self.context.task_nudges.append(
                        validated_reflection.get("next_step", "")
                    )
                # Add timestamp?
                validated_reflection["timestamp"] = datetime.now().isoformat()

                self.reflections.append(validated_reflection)
                self.agent_logger.info(
                    f"Reflection generated. Score: {validated_reflection['completion_score']:.2f}"
                )
                self.agent_logger.debug(f"Reflection details: {validated_reflection}")

                # Sync with memory manager if needed (optional, could be done periodically)
                # if self.context.memory_manager and self.context.memory_manager.agent_memory:
                #     self.context.memory_manager.agent_memory.reflections = self.reflections
                #     self.context.memory_manager.save_memory() # Save immediately?

                return validated_reflection

            except (json.JSONDecodeError, TypeError) as e:
                self.agent_logger.error(
                    f"Error parsing reflection JSON response: {e}\nResponse: {model_response.get('response')}"
                )
                return None
            except Exception as e:  # Catch Pydantic validation errors etc.
                self.agent_logger.error(
                    f"Error validating reflection data: {e}\nData: {reflection_data}"
                )
                # Still add the raw data? Maybe not.
                return None

        except Exception as e:
            tb_str = traceback.format_exc()
            self.agent_logger.error(
                f"Unexpected error during reflection generation: {e}\n{tb_str}"
            )
            return None

    def get_last_reflection(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent reflection, if any."""
        return self.reflections[-1] if self.reflections else None
