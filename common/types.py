from __future__ import annotations
import json
from datetime import datetime
from enum import Enum
from typing import List, Any, Optional, Dict

from pydantic import BaseModel, Field


class TaskStatus(Enum):
    """Standardized task status values"""

    INITIALIZED = "initialized"
    WAITING_DEPENDENCIES = "waiting_for_dependencies"
    RUNNING = "running"
    MISSING_TOOLS = "missing_tools"
    COMPLETE = "complete"
    RESCOPED_COMPLETE = "rescoped_complete"
    MAX_ITERATIONS = "max_iterations_reached"
    ERROR = "error"
    CANCELLED = "cancelled"

    def __str__(self):
        return self.value


class AgentMemory(BaseModel):
    """Model for agent memory storage"""

    agent_name: str
    session_history: List[Dict[str, Any]] = []
    tool_preferences: Dict[str, Any] = {}
    user_preferences: Dict[str, Any] = {}
    reflections: List[Dict[str, Any]] = []
    last_updated: datetime = Field(default_factory=datetime.now)
