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
    

class PlanStep(BaseModel):
    """A step in a plan."""

    index: int = 0
    description: str
    required_tools: List[str] = []
    purpose: str = ""
    is_action: bool = True
    success_criteria: str = ""
    status: StepStatus = StepStatus.PENDING
    result: Optional[StepResult] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    completed_at: Optional[float] = None
    tool_used: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}


class Plan(BaseModel):
    """A plan with steps."""

    plan_steps: List[PlanStep] = []
    plan_id: str = ""
    metadata: Dict[str, Any] = {}


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
