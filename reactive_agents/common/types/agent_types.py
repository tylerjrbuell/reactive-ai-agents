from typing import List, Optional, Dict, Any, Set
from pydantic import BaseModel, Field


class PlanFormat(BaseModel):
    """Format for agent planning data."""

    next_step: str
    rationale: str
    suggested_tools: List[str] = []


class ToolAnalysisFormat(BaseModel):
    """Format for tool analysis data."""

    required_tools: List[str] = Field(
        ..., description="List of tools essential for this task"
    )
    optional_tools: List[str] = Field(
        [], description="List of tools helpful but not essential"
    )
    explanation: str = Field(
        ..., description="Brief explanation of the tool requirements"
    )


class RescopeFormat(BaseModel):
    """Format for task rescoping data."""

    rescoped_task: Optional[str] = Field(
        None,
        description="A simplified, achievable task, or null if no rescope possible.",
    )
    explanation: str = Field(
        ..., description="Why this task was/wasn't rescoped and justification."
    )
    expected_tools: List[str] = Field(
        [], description="Tools expected for the rescoped task (if any)."
    )


class EvaluationFormat(BaseModel):
    """Format for goal evaluation data."""

    adherence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score 0.0-1.0: how well the result matches the goal",
    )
    strengths: List[str] = Field(
        [], description="Ways the result successfully addressed the goal"
    )
    weaknesses: List[str] = Field(
        [], description="Ways the result fell short of the goal"
    )
    explanation: str = Field(..., description="Overall explanation of the rating")
    matches_intent: bool = Field(
        ...,
        description="Whether the result fundamentally addresses the user's core intent",
    )


class TaskSuccessCriteria(BaseModel):
    """Model for task-specific success criteria."""

    required_tools: Set[str] = Field(
        default_factory=set,
        description="Set of tools that must be used for successful completion",
    )
    min_completion_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum completion score required for success",
    )
    required_answer_format: Optional[str] = Field(
        default=None,
        description="Expected format of the final answer (e.g., 'json', 'list', 'number')",
    )
    required_answer_content: Optional[List[str]] = Field(
        default=None, description="Required content elements in the final answer"
    )
    max_iterations: Optional[int] = Field(
        default=None, description="Maximum number of iterations allowed"
    )
    time_limit: Optional[float] = Field(
        default=None, description="Maximum time allowed for task completion in seconds"
    )
    success_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Overall success threshold combining all criteria",
    )
