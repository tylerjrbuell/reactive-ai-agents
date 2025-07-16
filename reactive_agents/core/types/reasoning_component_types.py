from __future__ import annotations
from typing import Dict, Any, List, Optional, Union, Callable, Awaitable, TYPE_CHECKING
from enum import Enum
from pydantic import BaseModel

from reactive_agents.core.types.status_types import StepStatus


class ComponentType(Enum):
    """Types of strategy components."""

    THINKING = "thinking"
    PLANNING = "planning"
    TOOL_EXECUTION = "tool_execution"
    REFLECTION = "reflection"
    EVALUATION = "evaluation"
    COMPLETION = "completion"
    ERROR_HANDLING = "error_handling"
    MEMORY_INTEGRATION = "memory_integration"
    STRATEGY_TRANSITION = "strategy_transition"


class ComponentMetadata(BaseModel):
    """Metadata for strategy components."""

    name: str
    description: str
    component_type: ComponentType
    version: str = "1.0.0"
    author: str = "unknown"
    required_components: List[str] = []
    config_schema: Dict[str, Any] = {}


class StepResult(BaseModel):
    """Result of a single step in a strategy."""

    success: bool = True
    step_name: str
    output: Dict[str, Any] = {}
    error: Optional[str] = None
    next_step: Optional[str] = None
    should_continue: bool = True
    confidence: float = 0.5
    metadata: Dict[str, Any] = {}

    def is_successful(self) -> bool:
        """Returns True if the step was successful."""
        return self.success


class PlanStep(BaseModel):
    """A step in a plan."""

    index: int
    description: str
    required_tools: List[str]
    purpose: str
    is_action: bool
    success_criteria: str
    status: StepStatus = StepStatus.PENDING
    result: Optional[StepResult] = None
    retries: int = 0
    is_crucial: bool = True

    def is_finished(self) -> bool:
        """Returns True if the step is completed or failed."""
        return self.status in [StepStatus.COMPLETED, StepStatus.FAILED]

    def get_summary(self) -> str:
        """Returns a one-line summary of the step's status."""
        summary = f"Step {self.index + 1} ({self.description[:30]}...): {self.status.value}"
        if self.retries > 0:
            summary += f" (Retries: {self.retries})"
        return summary


class Plan(BaseModel):
    """A plan with steps."""

    plan_steps: List[PlanStep] = []
    metadata: Dict[str, Any] = {}

    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next step to execute."""
        for step in self.plan_steps:
            if step.status == StepStatus.PENDING:
                return step
        return None

    def update_step_status(
        self, step: PlanStep, result_content: Optional[str], max_retries: int
    ):
        """Evaluate the result of a step and update its status."""
        success = result_content is not None and "error" not in result_content.lower()

        if success:
            step.status = StepStatus.COMPLETED
            step.result = StepResult(
                success=True,
                step_name=step.description,
                output={"result": result_content},
            )
        else:
            step.retries += 1
            if step.retries >= max_retries:
                step.status = StepStatus.FAILED
                step.result = StepResult(
                    success=False,
                    step_name=step.description,
                    error="Max retries reached",
                )
            else:
                step.status = StepStatus.PENDING

    def is_finished(self) -> bool:
        """Check if the plan is finished."""
        return self.get_next_step() is None

    def is_successful(self) -> bool:
        """Check if the plan was successful."""
        for step in self.plan_steps:
            if step.status == StepStatus.FAILED and step.is_crucial:
                return False
        return True

    def get_summary(self) -> str:
        """Returns a summary of the plan's status."""
        completed = len(
            [s for s in self.plan_steps if s.status == StepStatus.COMPLETED]
        )
        failed = len([s for s in self.plan_steps if s.status == StepStatus.FAILED])
        total = len(self.plan_steps)
        return f"Plan {completed}/{total} steps completed. {failed} failed."


class ReflectionResult(BaseModel):
    """Result of a reflection."""

    progress_assessment: str = ""
    goal_achieved: bool = False
    completion_score: float = 0.0
    next_action: str = "continue"  # continue, retry, complete, extend_plan
    confidence: float = 0.5
    blockers: List[str] = []
    success_indicators: List[str] = []
    learning_insights: List[str] = []
    error: Optional[str] = None


class ToolExecutionResult(BaseModel):
    """Result of a tool execution."""

    tool_calls: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    reasoning: str = ""
    error: Optional[str] = None


class CompletionResult(BaseModel):
    """Result of a task completion evaluation."""

    is_complete: bool = False
    should_complete: bool = False
    completion_score: float = 0.0
    final_answer: Optional[str] = None
    reasoning: str = ""
    missing_requirements: List[str] = []
    confidence: float = 0.5


class ErrorRecoveryResult(BaseModel):
    """Result of error recovery."""

    recovery_action: str
    rationale: str = ""
    alternative_approach: Optional[str] = None
    confidence: float = 0.5
    error_analysis: str = ""
    prevention_measures: List[str] = []
    tool_adjustments: Dict[str, str] = {}


class StrategyTransitionResult(BaseModel):
    """Result of a strategy transition decision."""

    should_switch: bool = False
    recommended_strategy: Optional[str] = None
    reasoning: str = ""
    confidence: float = 0.5
    trigger: str = ""
    expected_benefits: List[str] = []
    risks: List[str] = []
