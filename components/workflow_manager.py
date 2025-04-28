from __future__ import annotations
import traceback  # Added for detailed error logging
from datetime import datetime  # Added for timestamp
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING

from pydantic import BaseModel, Field

# Import shared types from the new location
from common.types import TaskStatus

# Assuming TaskStatus is available (e.g., from react_agent or a common types file)
# try:
#     from agents.react_agent import TaskStatus
# except ImportError:
#      # Basic placeholder if import fails
#      class TaskStatus:
#          INITIALIZED = "initialized"
#          WAITING_DEPENDENCIES = "waiting_for_dependencies"
#          RUNNING = "running"
#          MISSING_TOOLS = "missing_tools"
#          COMPLETE = "complete"
#          RESCOPED_COMPLETE = "rescoped_complete"
#          MAX_ITERATIONS = "max_iterations_reached"
#          ERROR = "error"
#          CANCELLED = "cancelled"
#          def __str__(self): return self.value # type: ignore


if TYPE_CHECKING:
    from context.agent_context import AgentContext
    from loggers.base import Logger


class WorkflowManager(BaseModel):
    """Manages workflow dependencies and updates the shared workflow context."""

    context: AgentContext = Field(exclude=True)  # Reference back to the main context

    # State / Config
    workflow_context: Optional[Dict[str, Any]] = (
        None  # Shared dict passed from orchestration layer
    )
    workflow_dependencies: List[str] = []  # Dependencies for *this* agent

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize self in workflow context if needed
        if (
            self.workflow_context is not None
            and self.context.agent_name not in self.workflow_context
        ):
            self.workflow_context[self.context.agent_name] = {
                "status": str(TaskStatus.INITIALIZED),  # Use string value
                "current_progress": "",
                "iterations": 0,
                "dependencies_met": True,  # Assume true initially
                "reflections": [],
                "last_updated": "",
            }
            self.agent_logger.debug(
                f"Initialized workflow context entry for {self.context.agent_name}"
            )

    @property
    def agent_logger(self) -> Logger:
        assert self.context.agent_logger is not None
        return self.context.agent_logger

    def check_dependencies(self) -> bool:
        """Checks if all defined workflow dependencies for this agent are met."""
        if not self.workflow_dependencies or self.workflow_context is None:
            return True  # No dependencies or no context = dependencies met

        all_met = True
        for dep_agent_name in self.workflow_dependencies:
            dep_status = self.workflow_context.get(dep_agent_name, {}).get("status")

            # Consider dependency met if status is COMPLETE or RESCOPED_COMPLETE
            met_statuses = [str(TaskStatus.COMPLETE), str(TaskStatus.RESCOPED_COMPLETE)]
            if dep_status not in met_statuses:
                self.agent_logger.info(
                    f"Dependency '{dep_agent_name}' not met. Status: '{dep_status}'. Waiting..."
                )
                all_met = False
                # Optionally break early if you only care if *any* dependency is unmet
                # break

        # Update own status in workflow context if waiting
        if not all_met:
            self.update_context(TaskStatus.WAITING_DEPENDENCIES)
        elif self.context.task_status == TaskStatus.WAITING_DEPENDENCIES:
            # If dependencies were previously unmet but now are, reset status
            self.update_context(TaskStatus.INITIALIZED)  # Or RUNNING if appropriate?

        return all_met

    def update_context(
        self, status: "TaskStatus", result: Optional[str] = None, **kwargs
    ):
        """
        Updates this agent's entry in the shared workflow context.

        Args:
            status: The new TaskStatus (or its string value).
            result: The current progress or final result.
            **kwargs: Additional data to add (e.g., error, completion_score, reflection_data).
        """
        if self.workflow_context is None:
            self.agent_logger.debug("No workflow context provided, skipping update.")
            return

        agent_name = self.context.agent_name
        if agent_name not in self.workflow_context:
            self.agent_logger.warning(
                f"Cannot update workflow context: Agent '{agent_name}' not found."
            )
            return

        # --- Prepare Update Data ---
        try:
            # Ensure status is a string
            status_str = str(status)

            # Basic info updated on most calls
            context_update: Dict[str, Any] = {
                "status": status_str,
                "iterations": self.context.iterations,
                "dependencies_met": self.check_dependencies(),  # Re-check for current status
                "last_updated": datetime.now().isoformat(),
            }

            # Add result/progress
            if result is not None:
                result_str = str(result)  # Ensure string representation
                result_key = (
                    "final_result"
                    if status_str
                    in [str(TaskStatus.COMPLETE), str(TaskStatus.RESCOPED_COMPLETE)]
                    else "current_progress"
                )
                # Truncate long results
                context_update[result_key] = (
                    result_str[:2000] + "..." if len(result_str) > 2000 else result_str
                )

            # Add specific fields based on status or kwargs
            status_enum = TaskStatus(status)
            if status_enum == TaskStatus.ERROR:
                error_info = kwargs.get("error", "Unknown error")
                context_update["error"] = str(error_info)[
                    :1000
                ]  # Limit error message length
            elif status_enum == TaskStatus.MISSING_TOOLS:
                context_update["missing_tools"] = kwargs.get("missing_tools", [])

            # Add optional data from kwargs
            if "completion_score" in kwargs:
                context_update["completion_score"] = kwargs["completion_score"]
            if "plan" in kwargs:  # Maybe last action plan?
                context_update["last_action_plan"] = kwargs["plan"]
            if "reflection_data" in kwargs:
                context_update["last_reflection"] = kwargs[
                    "reflection_data"
                ]  # Store latest reflection dict
            if "rescoped" in kwargs:
                context_update["rescoped"] = kwargs["rescoped"]
                if kwargs.get("original_task"):
                    context_update["original_task"] = kwargs["original_task"]
                if kwargs.get("rescoped_task"):
                    context_update["rescoped_task"] = kwargs["rescoped_task"]

            # Ensure reflections list exists and add latest reflections (optional, can be large)
            # Maybe only store the *last* reflection?
            # if self.context.reflection_manager:
            #    context_update["reflections"] = self.context.reflection_manager.reflections # Could be large

            # --- Apply Update ---
            if agent_name in self.workflow_context:
                self.workflow_context[agent_name].update(context_update)
                self.agent_logger.debug(
                    f"Updated workflow context for {agent_name}: Status={status_str}, Iter={self.context.iterations}"
                )
            else:
                # This case should ideally be handled by the init, but as a safeguard:
                self.agent_logger.warning(
                    f"Agent {agent_name} context missing, re-initializing."
                )
                self.workflow_context[agent_name] = context_update  # Create entry

        except Exception as e:
            tb_str = traceback.format_exc()
            self.agent_logger.error(
                f"Failed to update workflow context for {agent_name}: {e}\n{tb_str}"
            )

    def get_agent_context(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves the context data for a specific agent from the shared workflow context."""
        if self.workflow_context:
            return self.workflow_context.get(agent_name)
        return None

    def get_full_context(self) -> Optional[Dict[str, Any]]:
        """Returns the entire shared workflow context dictionary."""
        return self.workflow_context
