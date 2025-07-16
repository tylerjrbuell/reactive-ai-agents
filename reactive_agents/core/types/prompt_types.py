from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal


class TaskPlanningOutput(BaseModel):
    next_step: str = Field(description="Specific action to take")
    rationale: str = Field(description="Reasoning for this step")
    tool_needed: Optional[str] = Field(description="Tool name if required, null otherwise")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    memory_influence: str = Field(description="How past experiences influenced this decision")
    avoid_patterns: List[str] = Field(description="Patterns from memory to avoid")


class ReflectionOutput(BaseModel):
    progress_assessment: str = Field(
        default="", description="Summary of current progress"
    )
    goal_achieved: bool = Field(
        default=False, description="True ONLY if ALL steps succeeded AND no errors"
    )
    completion_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Based on successful_steps/total_steps"
    )
    next_action: Literal["continue", "retry", "complete"] = "continue"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    blockers: List[str] = Field(
        default_factory=list, description="List of current blockers"
    )
    success_indicators: List[str] = Field(
        default_factory=list, description="List of positive indicators"
    )
    learning_insights: List[str] = Field(
        default_factory=list, description="Insights from execution"
    )

    def to_prompt_format(self) -> str:
        """Formats the reflection into a concise string for the next LLM call."""
        blockers = f"Blockers: {', '.join(self.blockers)}" if self.blockers else "No blockers."
        return (
            f"Self-reflection: {self.progress_assessment}. "
            f"Confidence: {self.confidence:.2f}. {blockers} "
            f"Next action: {self.next_action}."
        )


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class FunctionCall(BaseModel):
    function: ToolCall


class ToolSelectionOutput(BaseModel):
    tool_calls: List[FunctionCall]
    reasoning: str = Field(description="Why this tool and these parameters")

    def to_prompt_format(self) -> str:
        """Formats the tool selection into the syntax expected by the LLM."""
        calls = []
        for tool_call in self.tool_calls:
            args = ", ".join(
                f"{k}={v}" for k, v in tool_call.function.arguments.items()
            )
            calls.append(f"{tool_call.function.name}({args})")
        return f"Tool selection: {', '.join(calls)}. Reasoning: {self.reasoning}"


class FinalAnswerOutput(BaseModel):
    final_answer: str = Field(description="Comprehensive answer that directly addresses the original task")
    summary: str = Field(description="Brief summary of what was accomplished")
    key_findings: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    methodology: str = Field(description="How the task was approached")
    limitations: List[str] = Field(description="Any limitations or caveats")


class StrategyTransitionOutput(BaseModel):
    should_switch: bool
    recommended_strategy: Optional[str] = Field(description="Strategy name or null")
    reasoning: str = Field(description="Detailed reasoning for the recommendation")
    confidence: float = Field(ge=0.0, le=1.0)
    trigger: str = Field(description="What triggered this recommendation")
    expected_benefits: List[str]
    risks: List[str]


class ErrorRecoveryOutput(BaseModel):
    recovery_action: str = Field(description="Specific action to take")
    rationale: str = Field(description="Why this action should work")
    alternative_approach: str = Field(description="Alternative if primary action fails")
    confidence: float = Field(ge=0.0, le=1.0)
    error_analysis: str = Field(description="Analysis of what went wrong")
    prevention_measures: List[str]
    tool_adjustments: Dict[str, str] = Field(description="<tool_name>: <adjustment_needed>")


class TaskCompletionValidationOutput(BaseModel):
    is_complete: bool = Field(description="True if ALL required components are DONE")
    completion_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="Specific reason based on execution results")
    confidence: float = Field(ge=0.0, le=1.0)


class PlanProgressReflectionOutput(BaseModel):
    progress_assessment: str = Field(description="Detailed evaluation of current progress")
    current_step_status: Literal["pending", "in_progress", "completed", "failed"]
    overall_completion_score: float = Field(ge=0.0, le=1.0)
    blockers: List[str]
    next_action: Literal["continue", "extend_plan", "complete_task", "retry"]
    confidence: float = Field(ge=0.0, le=1.0)
    learning_insights: List[str]
    recommendations: List[str]


class PlanStep(BaseModel):
    step_number: int
    description: str = Field(description="Specific action to take")
    purpose: str = Field(description="What this step accomplishes")
    is_action: bool
    required_tools: List[str]
    success_criteria: str = Field(description="How to know this step is complete")
    addresses_gap: str = Field(description="Which completion gap this addresses")


class PlanExtensionOutput(BaseModel):
    additional_steps: List[PlanStep]
    rationale: str = Field(description="Why these steps are needed")
    confidence: float = Field(ge=0.0, le=1.0)


class TaskGoalEvaluationOutput(BaseModel):
    completion: bool
    completion_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    missing_requirements: List[str]


class ToolCallSystemOutput(BaseModel):
    tool_calls: List[FunctionCall]
