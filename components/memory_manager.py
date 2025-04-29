from __future__ import annotations
import os
import json
import traceback  # Added for detailed error logging
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from pydantic import BaseModel, Field

# Import shared types from the new location
from common.types import AgentMemory, TaskStatus
from context.session import AgentSession

# Need to import AgentMemory from its original location or move it
# Assuming AgentMemory is defined in react_agent for now
# try:
#     from agents.react_agent import AgentMemory, TaskStatus
# except ImportError:
#     # Fallback if react_agent cannot be imported directly (e.g., during initial setup)
#     class AgentMemory(BaseModel): # Basic placeholder
#         agent_name: str
#         session_history: List[Dict[str, Any]] = []
#         tool_preferences: Dict[str, Any] = {}
#         reflections: List[Dict[str, Any]] = [] # Keep reflections sync'd
#         last_updated: datetime = Field(default_factory=datetime.now)
#     class TaskStatus: # Placeholder
#         COMPLETE = "complete"
#         RESCOPED_COMPLETE = "rescoped_complete"


if TYPE_CHECKING:
    from context.agent_context import AgentContext
    from loggers.base import Logger


class MemoryManager(BaseModel):
    """Manages agent's persistent memory (session history, preferences, reflections)."""

    context: AgentContext = Field(exclude=True)  # Reference back to the main context

    # State
    agent_memory: Optional[AgentMemory] = None
    memory_file_path: Optional[str] = None
    memory_enabled: bool = True  # Controlled by context

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.memory_enabled = self.context.use_memory_enabled
        if self.memory_enabled:
            self._initialize_memory()

    @property
    def agent_logger(self) -> Logger:
        assert self.context.agent_logger is not None
        return self.context.agent_logger

    def _initialize_memory(self):
        """Loads or initializes the agent's persistent memory state."""
        if not self.memory_enabled:
            self.agent_logger.debug("Memory is disabled, skipping initialization.")
            return

        try:
            # Define memory directory relative to this file or project root
            # Using __file__ assumes a standard project structure
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            memory_dir = os.path.join(base_dir, "memory")
            os.makedirs(memory_dir, exist_ok=True)

            safe_agent_name = self.context.agent_name.replace(" ", "_").replace(
                "/", "_"
            )  # Sanitize name
            self.memory_file_path = os.path.join(
                memory_dir, f"{safe_agent_name}_memory.json"
            )
            self.agent_logger.debug(f"Memory file path set to: {self.memory_file_path}")

            if os.path.exists(self.memory_file_path):
                try:
                    with open(self.memory_file_path, "r") as f:
                        memory_data = json.load(f)
                        # Validate data before loading
                        if memory_data.get("agent_name") == self.context.agent_name:
                            self.agent_memory = AgentMemory(**memory_data)
                            self.agent_logger.info(
                                f"Loaded memory for {self.context.agent_name} from {self.memory_file_path}"
                            )
                        else:
                            self.agent_logger.warning(
                                f"Memory file agent name mismatch ('{memory_data.get('agent_name')}' != '{self.context.agent_name}'). Initializing new memory."
                            )
                            self.agent_memory = AgentMemory(
                                agent_name=self.context.agent_name
                            )

                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    self.agent_logger.warning(
                        f"Error loading or validating memory file {self.memory_file_path}: {e}. Initializing new memory."
                    )
                    self.agent_memory = AgentMemory(agent_name=self.context.agent_name)
                except Exception as e:  # Catch other potential file/Pydantic errors
                    self.agent_logger.error(
                        f"Unexpected error loading memory file {self.memory_file_path}: {e}. Initializing new memory."
                    )
                    self.agent_memory = AgentMemory(agent_name=self.context.agent_name)
            else:
                self.agent_logger.info(
                    f"No existing memory file found for {self.context.agent_name}. Creating new memory."
                )
                self.agent_memory = AgentMemory(agent_name=self.context.agent_name)

        except Exception as e:
            self.agent_logger.error(f"Critical error initializing memory system: {e}")
            # Fallback to in-memory only (non-persistent)
            self.agent_memory = AgentMemory(agent_name=self.context.agent_name)
            self.memory_file_path = None  # Ensure we don't try to save later
            self.memory_enabled = False  # Disable persistence if init failed badly
            self.agent_logger.warning(
                "Memory persistence disabled due to initialization error."
            )

    def save_memory(self):
        """Saves the current agent memory state to the file."""
        if (
            not self.memory_enabled
            or not self.memory_file_path
            or not self.agent_memory
        ):
            self.agent_logger.debug(
                "Memory saving skipped (disabled, no path, or no memory object)."
            )
            return

        try:
            self.agent_memory.last_updated = datetime.now()

            # Sync reflections from reflection manager if available
            if self.context.reflection_manager:
                self.agent_memory.reflections = (
                    self.context.reflection_manager.reflections
                )

            memory_dict = self.agent_memory.dict()
            memory_json = json.dumps(
                memory_dict, indent=2, default=str
            )  # Use default=str for datetime etc.

            with open(self.memory_file_path, "w") as f:
                f.write(memory_json)
            self.agent_logger.debug(f"💾 Saved memory to {self.memory_file_path}")

        except TypeError as e:
            self.agent_logger.error(
                f"Error serializing memory to JSON: {e}. Memory may not be saved."
            )
        except IOError as e:
            self.agent_logger.error(
                f"Error writing memory file {self.memory_file_path}: {e}. Memory may not be saved."
            )
        except Exception as e:
            self.agent_logger.error(f"Unexpected error saving memory: {e}")

    def update_session_history(self, session_data: AgentSession):
        """Adds a summary of the completed agent run to the session history."""
        if not self.memory_enabled or not self.agent_memory:
            return

        try:
            # Extract relevant info directly from the session object
            status_str = str(session_data.task_status)
            iterations = session_data.iterations
            evaluation = session_data.evaluation
            adherence_score = evaluation.get("adherence_score")
            initial_task = session_data.initial_task
            final_result_summary = (
                str(session_data.final_answer)[:200] + "..."
                if session_data.final_answer
                else None
            )
            rescoped = (
                initial_task != session_data.current_task
            )  # Simple check if task changed

            # Determine success based on status
            success_statuses = [
                str(TaskStatus.COMPLETE),
                str(TaskStatus.RESCOPED_COMPLETE),
            ]
            is_success = status_str in success_statuses

            # Get tools used from session object
            tools_used_names = list(set(session_data.successful_tools))

            session_entry = {
                "session_id": session_data.session_id,  # Log session ID
                "timestamp": datetime.now().isoformat(),
                "task": initial_task,
                "status": status_str,
                "success": is_success,
                "iterations": iterations,
                "tools_used": tools_used_names,
                "adherence_score": adherence_score,
                "rescoped": rescoped,
                "final_result_summary": final_result_summary,
            }

            self.agent_memory.session_history.append(session_entry)

            # Limit history size
            max_history = 20
            if len(self.agent_memory.session_history) > max_history:
                self.agent_memory.session_history = self.agent_memory.session_history[
                    -max_history:
                ]

            # Optionally save immediately after update
            # self.save_memory() # Consider saving less frequently if performance is an issue

        except Exception as e:
            self.agent_logger.error(f"Error updating session history: {e}")

    def update_tool_preferences(
        self, tool_name: str, success: bool, feedback: Optional[str] = None
    ):
        """Updates usage statistics and feedback for a specific tool."""
        if not self.memory_enabled or not self.agent_memory:
            return

        try:
            if tool_name not in self.agent_memory.tool_preferences:
                self.agent_memory.tool_preferences[tool_name] = {
                    "success_count": 0,
                    "failure_count": 0,
                    "total_usage": 0,
                    "feedback": [],
                    "success_rate": 0.0,  # Calculated field
                }

            prefs = self.agent_memory.tool_preferences[tool_name]
            prefs["total_usage"] += 1

            if success:
                prefs["success_count"] += 1
            else:
                prefs["failure_count"] += 1

            # Recalculate success rate
            prefs["success_rate"] = (
                (prefs["success_count"] / prefs["total_usage"])
                if prefs["total_usage"] > 0
                else 0.0
            )

            if feedback:
                prefs["feedback"].append(
                    {"timestamp": datetime.now().isoformat(), "message": feedback}
                )
                # Limit feedback history
                max_feedback = 5
                if len(prefs["feedback"]) > max_feedback:
                    prefs["feedback"] = prefs["feedback"][-max_feedback:]

            # Optionally save immediately
            # self.save_memory()

        except Exception as e:
            self.agent_logger.error(
                f"Error updating tool preferences for '{tool_name}': {e}"
            )

    def get_reflections(self) -> List[Dict[str, Any]]:
        """Retrieves the list of reflections stored in memory."""
        if self.agent_memory:
            return self.agent_memory.reflections
        return []

    def get_tool_preferences(self) -> Dict[str, Any]:
        """Retrieves the tool preferences dictionary."""
        if self.agent_memory:
            return self.agent_memory.tool_preferences
        return {}

    def get_session_history(self) -> List[Dict[str, Any]]:
        """Retrieves the session history list."""
        if self.agent_memory:
            return self.agent_memory.session_history
        return []
