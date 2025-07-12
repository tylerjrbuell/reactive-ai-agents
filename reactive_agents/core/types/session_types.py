from __future__ import annotations
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
import uuid
import time
from .status_types import TaskStatus, StepStatus
from .agent_types import TaskSuccessCriteria
from .task_types import PlanStep
from .reasoning_types import ReflectDecideActState


class ReactiveState(BaseModel):
    """State tracking specific to Reactive strategy."""

    # Execution state
    execution_history: List[Dict[str, Any]] = Field(default_factory=list)
    error_count: int = 0
    max_errors: int = 3
    last_response: str = ""
    tool_responses: List[str] = Field(default_factory=list)

    # Metrics
    tool_success_rate: float = 0.0
    response_quality_score: float = 0.0

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a structured summary of execution progress."""
        successful_responses = [
            r for r in self.execution_history if r.get("success", False)
        ]
        failed_responses = [
            r for r in self.execution_history if not r.get("success", False)
        ]

        return {
            "total_responses": len(self.execution_history),
            "successful_responses": len(successful_responses),
            "failed_responses": len(failed_responses),
            "error_count": self.error_count,
            "tool_success_rate": self.tool_success_rate,
            "response_quality_score": self.response_quality_score,
        }

    def record_response_result(self, response_result: Dict[str, Any]) -> None:
        """Record a response result and update metrics."""
        self.execution_history.append(response_result)
        self.last_response = response_result.get("content", "")

        # Update success rates
        total_responses = len(self.execution_history)
        successful_responses = len(
            [r for r in self.execution_history if r.get("success", False)]
        )
        self.tool_success_rate = (
            successful_responses / total_responses if total_responses > 0 else 0.0
        )

        # Track tool responses
        if "tool_calls" in response_result:
            for call in response_result["tool_calls"]:
                if isinstance(call, dict) and "result" in call:
                    self.tool_responses.append(str(call["result"]))


class PlanExecuteReflectState(BaseModel):
    """State tracking specific to Plan-Execute-Reflect strategy."""

    # Plan state
    current_plan: Dict[str, Any] = Field(default_factory=lambda: {"steps": []})
    current_step: int = 0
    execution_history: List[Dict[str, Any]] = Field(default_factory=list)
    error_count: int = 0
    completed_actions: List[str] = Field(default_factory=list)
    max_errors: int = 3
    last_step_output: str = ""
    tool_responses: List[str] = Field(default_factory=list)

    # Reflection state
    reflection_count: int = 0
    last_reflection_result: Dict[str, Any] = Field(default_factory=dict)
    reflection_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Strategy metrics
    plan_success_rate: float = 0.0
    step_success_rate: float = 0.0
    recovery_success_rate: float = 0.0

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a structured summary of execution progress."""
        successful_steps = [
            s for s in self.execution_history if s.get("success", False)
        ]
        failed_steps = [
            s for s in self.execution_history if not s.get("success", False)
        ]

        return {
            "total_steps": len(self.execution_history),
            "successful_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "error_count": self.error_count,
            "reflection_count": self.reflection_count,
            "current_step": self.current_step,
            "plan_success_rate": self.plan_success_rate,
            "step_success_rate": self.step_success_rate,
            "recovery_success_rate": self.recovery_success_rate,
        }

    def record_step_result(self, step_result: Dict[str, Any]) -> None:
        """Record a step execution result and update metrics."""
        self.execution_history.append(step_result)
        self.last_step_output = step_result.get("result", "")

        # Update success rates
        total_steps = len(self.execution_history)
        successful_steps = len(
            [s for s in self.execution_history if s.get("success", False)]
        )
        self.step_success_rate = (
            successful_steps / total_steps if total_steps > 0 else 0.0
        )

        # Track tool responses
        if "tool_calls" in step_result:
            for call in step_result["tool_calls"]:
                if isinstance(call, dict) and "result" in call:
                    self.tool_responses.append(str(call["result"]))

    def record_reflection_result(self, reflection_result: Dict[str, Any]) -> None:
        """Record a reflection result and update metrics."""
        self.reflection_count += 1
        self.last_reflection_result = reflection_result
        self.reflection_history.append(reflection_result)


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

    # Strategy-specific state - updated to support multiple strategies
    per_strategy_state: Optional[
        PlanExecuteReflectState | ReactiveState | ReflectDecideActState
    ] = None

    def initialize_strategy_state(self, strategy_name: str) -> None:
        """Initialize state tracking for a specific strategy."""
        if strategy_name == "plan_execute_reflect":
            self.per_strategy_state = PlanExecuteReflectState()
        elif strategy_name == "reactive":
            self.per_strategy_state = ReactiveState()
        elif strategy_name == "reflect_decide_act":
            self.per_strategy_state = ReflectDecideActState()
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    def get_strategy_state(
        self,
    ) -> Optional[PlanExecuteReflectState | ReactiveState | ReflectDecideActState]:
        """Get the current strategy state."""
        return self.per_strategy_state

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
