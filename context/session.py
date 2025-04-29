from __future__ import annotations
import uuid
import time
from typing import List, Dict, Any, Optional, Set

from pydantic import BaseModel, Field

# Import common types needed
from common.types import TaskStatus


class AgentSession(BaseModel):
    """Represents the state of a single agent run."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None

    # Task definition for this run
    initial_task: str = ""
    current_task: str = ""  # Potentially rescoped
    min_required_tools: Optional[Set[str]] = None

    # Execution state for this run
    task_status: TaskStatus = TaskStatus.INITIALIZED  # Default to initialized
    iterations: int = 0
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    successful_tools: List[str] = Field(default_factory=list)
    task_progress: str = ""
    task_nudges: List[str] = Field(default_factory=list)
    reasoning_log: List[str] = Field(default_factory=list)

    # Final outcome of this run
    final_answer: Optional[str] = None
    completion_score: float = 0.0  # Deterministic score calculated during run
    summary: str = ""
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(
        default_factory=dict
    )  # Populated by MetricsManager at end of run
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True  # Allow TaskStatus
