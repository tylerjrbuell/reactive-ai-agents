from enum import Enum
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
    Literal,
)
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class ReasoningStrategies(str, Enum):
    """Available reasoning strategies for agents."""

    REACTIVE = (
        "reactive"  # Default strategy - direct prompt-response with state tracking
    )
    REFLECT_DECIDE_ACT = "reflect_decide_act"  # Reflection-based approach
    PLAN_EXECUTE_REFLECT = "plan_execute_reflect"  # Static planning upfront
    SELF_ASK = "self_ask"  # Question decomposition
    GOAL_ACTION_FEEDBACK = "goal_action_feedback"  # GAF pattern
    ADAPTIVE = "adaptive"  # Dynamic strategy switching (starts with reactive)


class ReasoningContext(BaseModel):
    """Context for reasoning strategies."""

    current_strategy: ReasoningStrategies
    iteration_count: int = 0
    error_count: int = 0
    tool_usage_history: List[str] = Field(default_factory=list)
    strategy_metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskGoalEvaluationContext(BaseModel):
    """Context for task goal evaluation."""

    task_description: str
    progress_summary: str = ""
    latest_output: str = ""
    execution_log: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)


class FinalAnswer(BaseModel):
    """Standardized final answer format for all strategies."""

    answer: str = Field(description="Clear, concise answer to the task")
    confidence: float = Field(
        description="Confidence score between 0 and 1", ge=0.0, le=1.0
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional task-specific details"
    )
    steps_completed: List[str] = Field(
        default_factory=list, description="List of completed steps or actions"
    )
    tools_used: List[str] = Field(
        default_factory=list, description="List of tools used during execution"
    )
    limitations: List[str] = Field(
        default_factory=list, description="List of limitations or caveats"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific metadata"
    )

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-compatible dictionary."""
        return {
            "final_answer": self.answer,
            "confidence": self.confidence,
            "details": {
                "steps_completed": self.steps_completed,
                "tools_used": self.tools_used,
                "limitations": self.limitations,
                **self.details,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "FinalAnswer":
        """Create FinalAnswer from JSON data."""
        details = data.get("details", {})
        return cls(
            answer=data.get("final_answer", ""),
            confidence=data.get("confidence", 0.0),
            steps_completed=details.get("steps_completed", []),
            tools_used=details.get("tools_used", []),
            limitations=details.get("limitations", []),
            details={
                k: v
                for k, v in details.items()
                if k not in ["steps_completed", "tools_used", "limitations"]
            },
            metadata=data.get("metadata", {}),
        )


@runtime_checkable
class ReasoningStrategyProtocol(Protocol):
    """Protocol for reasoning strategy implementations."""

    async def execute_iteration(
        self, task: str, context: "AgentContext", reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """Execute one iteration of this reasoning strategy."""
        ...

    def should_switch_strategy(
        self, context: "AgentContext", reasoning_context: ReasoningContext
    ) -> Optional[ReasoningStrategies]:
        """Determine if strategy should be switched."""
        ...

    def get_strategy_name(self) -> ReasoningStrategies:
        """Get the strategy enum identifier."""
        ...


class StrategySwitch(BaseModel):
    """Information about a strategy switch decision."""

    from_strategy: ReasoningStrategies
    to_strategy: ReasoningStrategies
    reason: str
    confidence: float  # 0.0 to 1.0
    trigger: str  # What triggered the switch


def flatten_and_extract_fields(data: dict, required_fields: list[str]) -> dict:
    """
    Recursively search nested dicts/lists and extract the first occurrence of each required field.
    Returns a dict mapping field names to their found values (or None if not found).
    """

    def _find_field(d, field):
        if isinstance(d, dict):
            if field in d:
                return d[field]
            for v in d.values():
                found = _find_field(v, field)
                if found is not None:
                    return found
        elif isinstance(d, list):
            for item in d:
                found = _find_field(item, field)
                if found is not None:
                    return found
        return None

    return {f: _find_field(data, f) for f in required_fields}


class TaskGoalEvaluationResult(BaseModel):
    """
    Structured result from LLM-powered task completion evaluation.
    """

    completion: bool
    completion_score: float
    reasoning: str
    missing_requirements: List[str] = []


class StrategyAction(str, Enum):
    """Defines the explicit actions a strategy can take in an iteration."""

    CONTINUE_THINKING = "continue_thinking"
    CALL_TOOLS = "call_tools"
    EVALUATE_COMPLETION = "evaluate_completion"
    FINISH_TASK = "finish_task"
    ERROR = "error"


# --- Payload Models ---


class ContinueThinkingPayload(BaseModel):
    """Payload for when the strategy needs to continue its thought process."""

    action: Literal[StrategyAction.CONTINUE_THINKING]
    reasoning: str = Field(description="The reasoning for the next step.")


class ToolCallPayload(BaseModel):
    """Payload for when the strategy decides to call one or more tools."""

    action: Literal[StrategyAction.CALL_TOOLS]
    tool_calls: List[Dict[str, Any]] = Field(
        description="The tool calls to be executed."
    )
    reasoning: str = Field(description="The agent's reasoning for calling these tools.")


class EvaluationPayload(BaseModel):
    """Represents the structured result of a task completion evaluation."""

    action: Literal[StrategyAction.EVALUATE_COMPLETION]
    is_complete: bool = Field(description="True if the task is considered complete.")
    reasoning: str = Field(description="The reasoning behind the completion assessment.")
    confidence: float = Field(ge=0.0, le=1.0)


class FinishTaskPayload(BaseModel):
    """Payload for when the strategy provides the final answer to the task."""

    action: Literal[StrategyAction.FINISH_TASK]
    final_answer: str = Field(description="The final answer for the user.")
    evaluation: EvaluationPayload = Field(
        description="The final evaluation that led to this answer."
    )


class ErrorPayload(BaseModel):
    """Payload for reporting an error within a strategy."""

    action: Literal[StrategyAction.ERROR]
    error_message: str
    details: Dict[str, Any] = Field(default_factory=dict)

    def is_critical(self) -> bool:
        """Returns True if the error is critical."""
        return self.details.get("is_critical", False)


# --- Discriminated Union for all Payloads ---
from typing import Union

ActionPayload = Union[
    ContinueThinkingPayload,
    ToolCallPayload,
    EvaluationPayload,
    FinishTaskPayload,
    ErrorPayload,
]
