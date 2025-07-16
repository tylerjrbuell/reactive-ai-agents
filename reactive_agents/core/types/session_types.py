from __future__ import annotations
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
import uuid
import time

from reactive_agents.core.types.prompt_types import ReflectionOutput
from .status_types import TaskStatus, StepStatus
from .agent_types import TaskSuccessCriteria
from .reasoning_component_types import Plan, PlanStep, StepResult


class BaseStrategyState(BaseModel):
    """Base class for all strategy-specific state models."""

    pass


class ReactiveState(BaseStrategyState):
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


class PlanExecuteReflectState(BaseStrategyState):
    """State tracking specific to Plan-Execute-Reflect strategy."""

    # Plan state
    current_plan: Plan = Field(default_factory=Plan)
    current_step: int = 0
    execution_history: List[Dict[str, Any]] = Field(default_factory=list)
    error_count: int = 0
    completed_actions: List[str] = Field(default_factory=list)
    max_errors: int = 3
    last_step_output: str = ""
    tool_responses: List[str] = Field(default_factory=list)

    # Reflection state
    reflection_count: int = 0
    last_reflection_result: Optional[ReflectionOutput] = None
    reflection_history: List[ReflectionOutput] = Field(default_factory=list)

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

    def record_reflection_result(self, reflection_result: ReflectionOutput) -> None:
        """Record a reflection result and update metrics."""
        self.reflection_count += 1
        self.last_reflection_result = reflection_result
        self.reflection_history.append(reflection_result)


class ReflectDecideActState(BaseStrategyState):
    """State tracking specific to Reflect-Decide-Act strategy."""

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
        default=None, description="Last goal evaluation result"
    )

    def record_reflection_result(self, result: Dict[str, Any]) -> None:
        self.reflection_history.append(result)

    def record_decision_result(self, result: Dict[str, Any]) -> None:
        self.decision_history.append(result)

    def record_action_result(self, result: Dict[str, Any]) -> None:
        self.action_history.append(result)

    def get_execution_summary(self) -> Dict[str, Any]:
        return {
            "cycle_count": self.cycle_count,
            "reflection_count": len(self.reflection_history),
            "decision_count": len(self.decision_history),
            "action_count": len(self.action_history),
            "error_count": self.error_count,
        }


# --- Unified Strategy & State Registration ---
STRATEGY_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_strategy(name: str, state_cls: type, **metadata):
    """Register a strategy and its state class in the global registry."""

    def decorator(strategy_cls):
        STRATEGY_REGISTRY[name] = {
            "strategy_cls": strategy_cls,
            "state_cls": state_cls,
            "metadata": metadata,
        }
        return strategy_cls

    return decorator


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

    # Strategy-specific state - updated to support multiple strategies
    strategy_state: Dict[str, BaseStrategyState] = Field(default_factory=dict)
    active_strategy: Optional[str] = None

    def initialize_strategy_state(self, strategy_name: str) -> None:
        """
        Initialize state tracking for a specific strategy if not already present.
        Uses the STRATEGY_REGISTRY for dynamic state instantiation.
        Falls back to manual mapping for legacy strategies.
        """
        if strategy_name not in self.strategy_state:
            entry = STRATEGY_REGISTRY.get(strategy_name)
            if entry and "state_cls" in entry:
                self.strategy_state[strategy_name] = entry["state_cls"]()
            elif strategy_name == "plan_execute_reflect":
                self.strategy_state[strategy_name] = PlanExecuteReflectState()
            elif strategy_name == "reactive":
                self.strategy_state[strategy_name] = ReactiveState()
            elif strategy_name == "reflect_decide_act":
                self.strategy_state[strategy_name] = ReflectDecideActState()
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
        self.active_strategy = strategy_name

    def get_strategy_state(
        self, strategy_name: Optional[str] = None
    ) -> Optional[BaseStrategyState]:
        """Get the state for a specific strategy, or the active one if not specified."""
        name = strategy_name or self.active_strategy
        if name is None:
            return None
        return self.strategy_state.get(name)
