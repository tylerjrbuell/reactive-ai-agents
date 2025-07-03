from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class AgentMemory(BaseModel):
    """Model for agent memory storage"""

    agent_name: str
    session_history: List[Dict[str, Any]] = []
    tool_preferences: Dict[str, Any] = {}
    user_preferences: Dict[str, Any] = {}
    reflections: List[Dict[str, Any]] = []
    last_updated: datetime = Field(default_factory=datetime.now)
