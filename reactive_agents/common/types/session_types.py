from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from .status_types import TaskStatus
from .agent_types import TaskSuccessCriteria
import uuid


class AgentSession(BaseModel):
    """Session data for a single agent run."""

    # Core session data
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    initial_task: str
    current_task: str
    start_time: float
    end_time: Optional[float] = None
    task_status: TaskStatus = TaskStatus.INITIALIZED
    error: Optional[str] = None

    # Message history and reasoning
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_log: List[str] = Field(default_factory=list)
    thinking_log: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # Store thinking with call context
    task_progress: List[str] = Field(default_factory=list)
    task_nudges: List[str] = Field(default_factory=list)

    # Tool usage tracking
    successful_tools: List[str] = Field(default_factory=list)
    min_required_tools: Optional[Set[str]] = None

    # Metrics and scoring
    metrics: Dict[str, Any] = Field(default_factory=dict)
    completion_score: float = 0.0
    tool_usage_score: float = 0.0
    progress_score: float = 0.0
    answer_quality_score: float = 0.0
    llm_evaluation_score: float = 0.0
    instruction_adherence_score: float = 0.0

    # Evaluation and improvement
    evaluation: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    final_answer: Optional[str] = None
    success_criteria: Optional[TaskSuccessCriteria] = None

    # Error tracking
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    # Iteration tracking
    iterations: int = 0

    # Scoring weights
    tool_usage_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    progress_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    answer_quality_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    llm_evaluation_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    completion_score_weight: float = Field(default=0.4, ge=0.0, le=1.0)

    # Add next step tracking
    current_next_step: Optional[str] = None
    next_step_source: Optional[str] = None  # "reflection", "planning", "manual"
    next_step_timestamp: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True
