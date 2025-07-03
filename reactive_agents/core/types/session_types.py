from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from .status_types import TaskStatus
from .agent_types import TaskSuccessCriteria
import uuid
from enum import Enum
import time


class StepStatus(str, Enum):
    """Status of individual plan steps."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    """Represents a single step in the plan with tracking information."""

    index: int
    description: str
    is_action: Optional[bool] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    completed_at: Optional[float] = None
    tool_used: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    retry_count: int = 0


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
    successful_tools: Set[str] = Field(default_factory=set)
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

    # Add last result tracking
    last_result: Optional[str] = None
    last_result_timestamp: Optional[float] = None
    last_result_iteration: Optional[int] = None

    # Index of the last summary message in the session (for incremental summarization)
    last_summary_index: int = 0

    # Plan tracking for hybrid planning-reflection
    plan: Optional[list] = Field(default_factory=list)
    plan_last_modified: Optional[float] = None

    # New step tracking system
    plan_steps: List[PlanStep] = Field(default_factory=list)
    current_step_index: int = 0

    def set_min_required_tools(self, required_tools) -> None:
        """
        Set the min_required_tools for the session, ensuring 'final_answer' is always included.
        Accepts any iterable of tool names.
        """
        tools_set = set(required_tools)
        tools_set.add("final_answer")
        self.min_required_tools = tools_set

    def get_current_step(self) -> Optional[PlanStep]:
        if self.current_step_index is None or self.current_step_index < 0:
            self.current_step_index = 0
        if 0 <= self.current_step_index < len(self.plan_steps):
            return self.plan_steps[self.current_step_index]
        # Fallback: return next pending step if any
        for step in self.plan_steps:
            if step.status == StepStatus.PENDING:
                self.current_step_index = step.index
                return step
        return None

    def get_next_pending_step(self) -> Optional[PlanStep]:
        """Get the next pending step in the plan."""
        for step in self.plan_steps:
            if step.status == StepStatus.PENDING:
                return step
        return None

    def mark_step_completed(
        self,
        step_index: int,
        result: str,
        tool_used: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a step as completed with its result."""
        if 0 <= step_index < len(self.plan_steps):
            step = self.plan_steps[step_index]
            step.status = StepStatus.COMPLETED
            step.result = result
            step.completed_at = time.time()
            step.tool_used = tool_used
            step.parameters = parameters

    def mark_step_failed(self, step_index: int, error: str) -> None:
        """Mark a step as failed with an error message."""
        if 0 <= step_index < len(self.plan_steps):
            step = self.plan_steps[step_index]
            step.status = StepStatus.FAILED
            step.error = error

    def is_plan_complete(self) -> bool:
        """Check if all steps in the plan are completed."""
        if not self.plan_steps:
            return False
        return all(step.status == StepStatus.COMPLETED for step in self.plan_steps)

    def get_completion_percentage(self) -> float:
        """Get the percentage of completed steps."""
        if self.final_answer:
            return 100.0
        if not self.plan_steps:
            return 0.0
        completed = sum(
            1 for step in self.plan_steps if step.status == StepStatus.COMPLETED
        )
        return (completed / len(self.plan_steps)) * 100.0

    class Config:
        arbitrary_types_allowed = True
