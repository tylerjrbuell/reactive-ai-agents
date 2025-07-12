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
    """Context information for reasoning strategy selection and execution."""

    current_strategy: ReasoningStrategies
    task_classification: Optional[Dict[str, Any]] = None
    iteration_count: int = 0
    last_action_result: Optional[Dict[str, Any]] = None
    stagnation_count: int = 0
    tool_usage_history: List[str] = []
    error_count: int = 0
    success_indicators: List[str] = []
    strategy_switches: List[Dict[str, Any]] = []


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


class ReflectionResult(BaseModel):
    """Unified reflection result schema for all strategies."""

    # Core fields (required by most strategies)
    progress_assessment: Optional[str] = Field(
        None, description="Assessment of current progress"
    )
    completion_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Completion score (0-1)"
    )
    next_action: Optional[str] = Field(
        None, description="Next action to take (continue, retry, complete, etc.)"
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence in assessment (0-1)"
    )

    # PER/other fields
    goal_achieved: Optional[bool] = Field(
        None, description="Whether the goal is achieved"
    )
    blockers: Optional[List[str]] = Field(
        default_factory=list, description="List of blockers or issues"
    )
    success_indicators: Optional[List[str]] = Field(
        default_factory=list, description="Indicators of success"
    )
    learning_insights: Optional[List[str]] = Field(
        default_factory=list, description="Insights or lessons learned"
    )

    # For extensibility
    extra: Dict[str, Any] = Field(
        default_factory=dict, description="Additional strategy-specific fields"
    )

    @staticmethod
    def _deep_parse_json(data: Any) -> Any:
        """Recursively parse stringified JSON dicts until a dict is reached or parsing fails."""
        import json

        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                return ReflectionResult._deep_parse_json(parsed)
            except Exception:
                return data
        if isinstance(data, dict):
            return {k: ReflectionResult._deep_parse_json(v) for k, v in data.items()}
        if isinstance(data, list):
            return [ReflectionResult._deep_parse_json(v) for v in data]
        return data

    @classmethod
    def _find_field(cls, data: Any, field: str) -> Any:
        """Recursively search for the first occurrence of a field in a nested dict/list."""
        if isinstance(data, dict):
            if field in data:
                return data[field]
            for v in data.values():
                found = cls._find_field(v, field)
                if found is not None:
                    return found
        elif isinstance(data, list):
            for item in data:
                found = cls._find_field(item, field)
                if found is not None:
                    return found
        return None

    @classmethod
    def from_raw(cls, data: Dict[str, Any]) -> "ReflectionResult":
        """Parse and coerce a raw dict (possibly from LLM) into a ReflectionResult, handling missing/extra fields, deep parsing, and flattening."""
        # Recursively parse any stringified JSON
        data = cls._deep_parse_json(data)
        # Promote required fields from any nested dicts
        known = {}
        for f in cls.model_fields.keys():
            if f == "extra":
                continue
            val = cls._find_field(data, f)
            known[f] = val
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(**known, extra=extra)

    def is_valid(self, require_core: bool = True) -> bool:
        """Check if the reflection result has the minimum required fields for most strategies."""
        core_fields = [
            self.progress_assessment,
            self.completion_score,
            self.next_action,
            self.confidence,
        ]
        if require_core:
            return all(f is not None for f in core_fields)
        return True


class ReflectDecideActState(BaseModel):
    """State tracking for Reflect-Decide-Act strategy."""

    cycle_count: int = Field(default=0, description="Number of RDA cycles completed")
    reflection_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of reflection results"
    )
    decision_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of decisions made"
    )
    action_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of actions taken"
    )
    current_action: Dict[str, Any] = Field(
        default_factory=dict, description="Current action being executed"
    )
    error_count: int = Field(default=0, description="Number of errors encountered")
    max_errors: int = Field(
        default=3, description="Maximum allowed errors before strategy switch"
    )
    last_goal_evaluation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Most recent LLM-powered goal evaluation feedback for this strategy.",
    )

    def record_reflection_result(self, result: Dict[str, Any]) -> None:
        """Record a reflection result in history."""
        self.reflection_history.append(result)

    def record_decision_result(self, result: Dict[str, Any]) -> None:
        """Record a decision result in history."""
        self.decision_history.append(result)

    def record_action_result(self, result: Dict[str, Any]) -> None:
        """Record an action result in history."""
        self.action_history.append(result)
        self.current_action = result
        self.cycle_count += 1

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution progress."""
        return {
            "total_cycles": self.cycle_count,
            "reflections_generated": len(self.reflection_history),
            "decisions_made": len(self.decision_history),
            "actions_attempted": len(self.action_history),
            "successful_actions": sum(
                1 for a in self.action_history if a.get("success", False)
            ),
            "failed_actions": sum(
                1 for a in self.action_history if not a.get("success", False)
            ),
            "error_count": self.error_count,
            "reflection_quality_score": sum(
                r.get("validation", {}).get("quality_score", 0)
                for r in self.reflection_history
            )
            / max(len(self.reflection_history), 1),
            "decision_quality_score": sum(
                d.get("validation", {}).get("quality_score", 0)
                for d in self.decision_history
            )
            / max(len(self.decision_history), 1),
            "action_success_rate": sum(
                1 for a in self.action_history if a.get("success", False)
            )
            / max(len(self.action_history), 1),
        }


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


class TaskGoalEvaluationContext(BaseModel):
    """
    Context for LLM-powered task completion evaluation.
    """

    task_description: str
    progress_summary: str
    latest_output: str
    execution_log: str
    meta: Dict[str, Any] = {}
    success_criteria: Optional[str] = None


class TaskGoalEvaluationResult(BaseModel):
    """
    Structured result from LLM-powered task completion evaluation.
    """

    completion: bool
    completion_score: float
    reasoning: str
    missing_requirements: List[str] = []
