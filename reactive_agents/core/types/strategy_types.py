"""Types for reasoning strategies and strategy management."""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
from dataclasses import dataclass, field


class StrategyCapabilities(Enum):
    """Capabilities that a strategy can declare."""
    
    TOOL_EXECUTION = "tool_execution"
    PLANNING = "planning"
    REFLECTION = "reflection"
    MEMORY_USAGE = "memory_usage"
    ADAPTATION = "adaptation"
    COLLABORATION = "collaboration"


class StrategyState(Enum):
    """States that a strategy can be in during execution."""
    
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    EVALUATING = "evaluating"
    COMPLETING = "completing"
    ERROR_RECOVERY = "error_recovery"
    TRANSITIONING = "transitioning"
    PAUSED = "paused"
    TERMINATED = "terminated"
    FAILED = "failed"


class StateTransitionTrigger(Enum):
    """Triggers that can cause state transitions."""
    
    START_EXECUTION = "start_execution"
    PLANNING_COMPLETE = "planning_complete"
    EXECUTION_STEP_COMPLETE = "execution_step_complete"
    REFLECTION_COMPLETE = "reflection_complete"
    EVALUATION_COMPLETE = "evaluation_complete"
    TASK_COMPLETE = "task_complete"
    ERROR_OCCURRED = "error_occurred"
    RECOVERY_COMPLETE = "recovery_complete"
    STRATEGY_SWITCH_REQUESTED = "strategy_switch_requested"
    PAUSE_REQUESTED = "pause_requested"
    RESUME_REQUESTED = "resume_requested"
    TERMINATION_REQUESTED = "termination_requested"
    RETRY_REQUESTED = "retry_requested"


class RetryStrategy(Enum):
    """Available retry strategies for failed operations."""
    
    NONE = "none"
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"


class ComponentCapability(Enum):
    """Standard capabilities that components can provide."""
    
    THINKING = "thinking"
    PLANNING = "planning"
    TOOL_EXECUTION = "tool_execution"
    REFLECTION = "reflection"
    EVALUATION = "evaluation"
    COMPLETION = "completion"
    ERROR_HANDLING = "error_handling"
    MEMORY_INTEGRATION = "memory_integration"
    STRATEGY_TRANSITION = "strategy_transition"
    CONTEXT_MANAGEMENT = "context_management"
    PERFORMANCE_MONITORING = "performance_monitoring"


class ComponentStatus(Enum):
    """Component operational status."""
    
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"
    INITIALIZING = "initializing"


class PerformanceMetric(Enum):
    """Available performance metrics."""
    
    SUCCESS_RATE = "success_rate"
    AVERAGE_ITERATIONS = "average_iterations"
    EXECUTION_TIME = "execution_time"
    ERROR_RATE = "error_rate"
    COMPLETION_SCORE = "completion_score"
    EFFICIENCY_SCORE = "efficiency_score"
    RESOURCE_USAGE = "resource_usage"
    CONTEXT_EFFICIENCY = "context_efficiency"
    TOOL_USAGE_EFFECTIVENESS = "tool_usage_effectiveness"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for pattern matching."""
    
    NETWORK = "network"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    LOGIC = "logic"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Available recovery actions."""
    
    RETRY = "retry"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    SWITCH_STRATEGY = "switch_strategy"
    SIMPLIFY_APPROACH = "simplify_approach"
    REQUEST_USER_INPUT = "request_user_input"
    FALLBACK_MODE = "fallback_mode"
    TERMINATE = "terminate"
    ESCALATE = "escalate"
    RESET_CONTEXT = "reset_context"
    REDUCE_SCOPE = "reduce_scope"


class StrategyResult(BaseModel):
    """
    A strongly-typed result from a strategy iteration.
    It contains a specific action and a corresponding payload with the required data.
    """
    
    # Import ActionPayload and StrategyAction from reasoning_types when needed
    action: Any  # Will be StrategyAction 
    payload: Any = Field(..., description="ActionPayload with discriminator='action'")
    should_continue: bool = True
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    execution_time_ms: Optional[float] = None
    iteration_count: int = 0
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Error handling
    error_message: Optional[str] = None
    retry_strategy: RetryStrategy = RetryStrategy.NONE
    
    # This allows creating the model with a simplified syntax
    @classmethod
    def create(
        cls, payload: Any, should_continue: bool = True
    ) -> "StrategyResult":
        return cls(
            action=payload.action, payload=payload, should_continue=should_continue
        )
    
    @classmethod
    def success_result(cls, payload: Any, should_continue: bool = True, **kwargs) -> "StrategyResult":
        """Create a successful strategy result."""
        return cls(
            action=payload.action if hasattr(payload, 'action') else None,
            payload=payload,
            should_continue=should_continue,
            **kwargs
        )
    
    @classmethod
    def error_result(
        cls, 
        error_message: str, 
        retry_strategy: RetryStrategy = RetryStrategy.NONE,
        **kwargs
    ) -> "StrategyResult":
        """Create an error strategy result."""
        return cls(
            action=None,
            payload=None,
            should_continue=False,
            error_message=error_message,
            retry_strategy=retry_strategy,
            **kwargs
        )


# ComponentContext is defined in protocols.py - use that instead


class ComponentHealthCheck(BaseModel):
    """Health status of a component."""
    
    component_name: str
    status: ComponentStatus
    last_check: datetime = Field(default_factory=datetime.now)
    
    # Health metrics
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    average_response_time_ms: float = 0.0
    total_executions: int = 0
    error_count: int = 0
    
    # Additional health data
    last_error: Optional[str] = None
    last_success: Optional[datetime] = None
    uptime_seconds: float = 0.0
    
    # Resource metrics
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return (
            self.status in [ComponentStatus.READY, ComponentStatus.BUSY] and
            self.success_rate >= 0.8 and
            self.average_response_time_ms < 30000  # 30 seconds
        )


class PerformanceThreshold(BaseModel):
    """Performance thresholds for strategy evaluation."""
    
    min_success_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    max_average_iterations: int = Field(default=10, ge=1)
    max_execution_time_ms: float = Field(default=300000, ge=0)  # 5 minutes
    max_error_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    min_completion_score: float = Field(default=0.6, ge=0.0, le=1.0)
    min_efficiency_score: float = Field(default=0.5, ge=0.0, le=1.0)


@dataclass
class StateTransition:
    """Represents a state transition with metadata."""
    
    from_state: StrategyState
    to_state: StrategyState
    trigger: StateTransitionTrigger
    timestamp: datetime
    context: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


class TransitionCondition(BaseModel):
    """Defines conditions that must be met for a transition."""
    
    required_context_keys: Set[str] = Field(default_factory=set)
    forbidden_context_keys: Set[str] = Field(default_factory=set)
    custom_validator: Optional[str] = None  # Name of custom validation function
    max_retry_count: Optional[int] = None
    min_execution_time_ms: Optional[float] = None


class StateDefinition(BaseModel):
    """Defines a strategy state with its properties and valid transitions."""
    
    state: StrategyState
    description: str = ""
    is_terminal: bool = False
    max_duration_ms: Optional[float] = None
    
    # Valid transitions from this state
    valid_transitions: Dict[StateTransitionTrigger, StrategyState] = Field(default_factory=dict)
    
    # Conditions for transitions
    transition_conditions: Dict[StateTransitionTrigger, TransitionCondition] = Field(default_factory=dict)
    
    # Callbacks
    on_enter: Optional[str] = None  # Method name to call on entering state
    on_exit: Optional[str] = None   # Method name to call on exiting state
    on_timeout: Optional[str] = None  # Method name to call on timeout


@dataclass
class ExecutionRecord:
    """Record of a single strategy execution."""
    
    strategy_name: str
    task_description: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    iterations: int = 0
    completion_score: float = 0.0
    error_count: int = 0
    state_transitions: List[Tuple[StrategyState, StrategyState]] = field(default_factory=list)
    tool_calls: int = 0
    context_tokens_used: int = 0
    resource_metrics: Dict[str, float] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def execution_time_ms(self) -> float:
        """Get execution time in milliseconds."""
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score based on completion vs. resources used."""
        if self.iterations == 0:
            return 0.0
        
        # Base efficiency on completion score per iteration
        base_efficiency = self.completion_score / max(1, self.iterations)
        
        # Adjust for error rate
        error_penalty = self.error_count * 0.1
        
        return max(0.0, min(1.0, base_efficiency - error_penalty))


@dataclass
class ErrorContext:
    """Context information for an error occurrence."""
    
    task: str
    component: str
    operation: str
    iteration: int
    strategy: str
    error_message: str
    error_type: str
    stack_trace: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RecoveryPattern:
    """A learned pattern for error recovery."""
    
    error_signature: str  # Hash of error characteristics
    error_category: ErrorCategory
    severity: ErrorSeverity
    recovery_action: RecoveryAction
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None
    context_requirements: Set[str] = field(default_factory=set)
    effectiveness_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureRecord:
    """Record of a failure and recovery attempt."""
    
    error_context: ErrorContext
    recovery_pattern: Optional[RecoveryPattern]
    recovery_action_taken: RecoveryAction
    success: bool
    recovery_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""


class RecoveryResult(BaseModel):
    """Result of an error recovery attempt."""
    
    success: bool
    recovery_action: RecoveryAction
    should_switch_strategy: bool = False
    recommended_strategy: Optional[str] = None
    
    # Recovery details
    recovery_time_ms: float = 0.0
    attempts_made: int = 1
    
    # Context changes
    context_modifications: Dict[str, Any] = Field(default_factory=dict)
    
    # Learning data
    pattern_effectiveness: Optional[float] = None
    should_update_pattern: bool = False
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    notes: str = ""