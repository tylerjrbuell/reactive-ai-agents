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
from pydantic import BaseModel
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class ReasoningStrategies(str, Enum):
    """Available reasoning strategies for agents."""

    REACTIVE = "reactive"  # No planning, pure prompt-response
    REFLECT_DECIDE_ACT = "reflect_decide_act"  # Current default
    PLAN_EXECUTE_REFLECT = "plan_execute_reflect"  # Static planning upfront
    SELF_ASK = "self_ask"  # Question decomposition
    GOAL_ACTION_FEEDBACK = "goal_action_feedback"  # GAF pattern
    ADAPTIVE = "adaptive"  # Dynamic strategy switching


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
