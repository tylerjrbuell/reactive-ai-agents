"""
Protocols and interfaces for reasoning system components.

This module defines the standard interfaces and result wrappers for all
reasoning components to ensure consistency and reliability across the system.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic, Any, Dict, Optional, Set, Union
from enum import Enum
from pydantic import BaseModel, Field

from reactive_agents.core.types.reasoning_component_types import (
    ReflectionResult,
    ToolExecutionResult,
    CompletionResult,
    ErrorRecoveryResult,
    StrategyTransitionResult,
    Plan,
    StepResult,
    ComponentType,
)


# Type variable for generic result data
T = TypeVar('T', bound=BaseModel)


class RetryStrategy(Enum):
    """Available retry strategies for failed operations."""
    NONE = "none"
    IMMEDIATE = "immediate"  
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"


class ComponentResult(BaseModel, Generic[T]):
    """
    Standardized result wrapper for all component operations.
    
    This provides a consistent interface for component returns with
    built-in error handling, retry logic, and metadata tracking.
    """
    success: bool = Field(description="Whether the operation succeeded")
    data: Optional[T] = Field(default=None, description="The actual result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context information")
    retry_strategy: RetryStrategy = Field(default=RetryStrategy.NONE, description="Recommended retry approach")
    retry_count: int = Field(default=0, description="Number of times this operation has been retried")
    execution_time_ms: Optional[float] = Field(default=None, description="Execution time in milliseconds")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in the result")
    
    @classmethod
    def success_result(cls, data: T, confidence: float = 1.0, **metadata) -> ComponentResult[T]:
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            confidence=confidence,
            metadata=metadata
        )
    
    @classmethod
    def error_result(
        cls, 
        error: str, 
        retry_strategy: RetryStrategy = RetryStrategy.NONE,
        confidence: float = 0.0,
        **metadata
    ) -> ComponentResult[T]:
        """Create an error result."""
        return cls(
            success=False,
            error=error,
            retry_strategy=retry_strategy,
            confidence=confidence,
            metadata=metadata
        )
    
    def is_retryable(self) -> bool:
        """Check if this result indicates a retryable operation."""
        return not self.success and self.retry_strategy != RetryStrategy.NONE
    
    def requires_recovery(self) -> bool:
        """Check if this result requires error recovery."""
        return not self.success
    
    def with_retry_increment(self) -> ComponentResult[T]:
        """Return a copy with incremented retry count."""
        return self.model_copy(update={'retry_count': self.retry_count + 1})


class ComponentContext(BaseModel):
    """Context passed to component operations."""
    task: str = Field(description="The current task being executed")
    session_id: str = Field(description="Unique session identifier")
    iteration: int = Field(default=1, description="Current iteration number")
    previous_results: Dict[str, Any] = Field(default_factory=dict, description="Results from previous steps")
    available_tools: Set[str] = Field(default_factory=set, description="Available tool names")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")


class ReasoningComponent(Protocol):
    """
    Standard protocol interface for all reasoning components.
    
    This ensures consistency across all component implementations and
    provides a foundation for robust error handling and retry logic.
    """
    
    @abstractmethod
    async def execute(self, context: ComponentContext) -> ComponentResult[BaseModel]:
        """
        Execute the component's primary logic.
        
        Args:
            context: The execution context containing task and state information
            
        Returns:
            ComponentResult containing the operation result and metadata
        """
        pass
    
    @abstractmethod
    def can_retry(self, error: Exception, context: ComponentContext) -> bool:
        """
        Determine if the operation can be retried given the error and context.
        
        Args:
            error: The exception that occurred
            context: The execution context
            
        Returns:
            True if the operation should be retried
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Set[str]:
        """
        Return the set of capabilities this component provides.
        
        Returns:
            Set of capability names (e.g., {'planning', 'tool_execution'})
        """
        pass
    
    @abstractmethod
    def get_retry_strategy(self, error: Exception, context: ComponentContext) -> RetryStrategy:
        """
        Determine the appropriate retry strategy for a given error.
        
        Args:
            error: The exception that occurred
            context: The execution context
            
        Returns:
            The recommended retry strategy
        """
        pass


# Specialized result types that maintain backward compatibility
ComponentReflectionResult = ComponentResult[ReflectionResult]
ComponentToolExecutionResult = ComponentResult[ToolExecutionResult]
ComponentCompletionResult = ComponentResult[CompletionResult]
ComponentErrorRecoveryResult = ComponentResult[ErrorRecoveryResult]
ComponentStrategyTransitionResult = ComponentResult[StrategyTransitionResult]
ComponentPlanResult = ComponentResult[Plan]
ComponentStepResult = ComponentResult[StepResult]


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


class ComponentHealthCheck(BaseModel):
    """Health check result for components."""
    status: ComponentStatus
    last_success: Optional[float] = None  # timestamp
    error_count: int = 0
    performance_score: float = 1.0
    dependencies_healthy: bool = True
    message: Optional[str] = None