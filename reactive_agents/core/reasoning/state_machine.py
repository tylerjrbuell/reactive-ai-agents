"""
Strategy State Machine for formal state transition management.

This module provides a robust state machine implementation for managing
strategy execution states and transitions with validation and rollback capabilities.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, Optional, Callable, List, Any, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
from pydantic import BaseModel, Field

from reactive_agents.core.types.reasoning_types import ReasoningContext


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
    """Defines a state and its allowed transitions."""
    state: StrategyState
    description: str
    allowed_transitions: Dict[StateTransitionTrigger, StrategyState] = Field(default_factory=dict)
    entry_actions: List[str] = Field(default_factory=list)  # Names of functions to call on entry
    exit_actions: List[str] = Field(default_factory=list)   # Names of functions to call on exit
    timeout_ms: Optional[float] = None
    is_terminal: bool = False
    requires_confirmation: bool = False


class StrategyStateMachine:
    """
    Manages strategy execution state transitions with validation and rollback.
    
    This state machine provides formal state management for reasoning strategies,
    ensuring valid transitions and maintaining execution history for debugging
    and rollback purposes.
    """

    def __init__(self, initial_state: StrategyState = StrategyState.INITIALIZING):
        """
        Initialize the state machine.
        
        Args:
            initial_state: The starting state for the machine
        """
        self.current_state = initial_state
        self.previous_state: Optional[StrategyState] = None
        self.state_history: List[StateTransition] = []
        self.transition_handlers: Dict[Tuple[StrategyState, StrategyState], Callable] = {}
        self.state_definitions = self._initialize_state_definitions()
        self.transition_conditions: Dict[Tuple[StrategyState, StrategyState], TransitionCondition] = {}
        self.custom_validators: Dict[str, Callable] = {}
        self.state_entry_time: Optional[datetime] = datetime.now()
        self._locks: Dict[str, asyncio.Lock] = {}

    def _initialize_state_definitions(self) -> Dict[StrategyState, StateDefinition]:
        """Initialize the standard state definitions."""
        return {
            StrategyState.INITIALIZING: StateDefinition(
                state=StrategyState.INITIALIZING,
                description="Strategy is being initialized",
                allowed_transitions={
                    StateTransitionTrigger.START_EXECUTION: StrategyState.PLANNING,
                    StateTransitionTrigger.ERROR_OCCURRED: StrategyState.ERROR_RECOVERY,
                    StateTransitionTrigger.TERMINATION_REQUESTED: StrategyState.TERMINATED,
                },
                timeout_ms=30000  # 30 seconds max for initialization
            ),
            StrategyState.PLANNING: StateDefinition(
                state=StrategyState.PLANNING,
                description="Strategy is planning the approach",
                allowed_transitions={
                    StateTransitionTrigger.PLANNING_COMPLETE: StrategyState.EXECUTING,
                    StateTransitionTrigger.ERROR_OCCURRED: StrategyState.ERROR_RECOVERY,
                    StateTransitionTrigger.PAUSE_REQUESTED: StrategyState.PAUSED,
                    StateTransitionTrigger.TERMINATION_REQUESTED: StrategyState.TERMINATED,
                }
            ),
            StrategyState.EXECUTING: StateDefinition(
                state=StrategyState.EXECUTING,
                description="Strategy is executing steps",
                allowed_transitions={
                    StateTransitionTrigger.EXECUTION_STEP_COMPLETE: StrategyState.REFLECTING,
                    StateTransitionTrigger.TASK_COMPLETE: StrategyState.COMPLETING,
                    StateTransitionTrigger.ERROR_OCCURRED: StrategyState.ERROR_RECOVERY,
                    StateTransitionTrigger.PAUSE_REQUESTED: StrategyState.PAUSED,
                    StateTransitionTrigger.STRATEGY_SWITCH_REQUESTED: StrategyState.TRANSITIONING,
                    StateTransitionTrigger.TERMINATION_REQUESTED: StrategyState.TERMINATED,
                }
            ),
            StrategyState.REFLECTING: StateDefinition(
                state=StrategyState.REFLECTING,
                description="Strategy is reflecting on progress",
                allowed_transitions={
                    StateTransitionTrigger.REFLECTION_COMPLETE: StrategyState.EVALUATING,
                    StateTransitionTrigger.ERROR_OCCURRED: StrategyState.ERROR_RECOVERY,
                    StateTransitionTrigger.PAUSE_REQUESTED: StrategyState.PAUSED,
                    StateTransitionTrigger.TERMINATION_REQUESTED: StrategyState.TERMINATED,
                }
            ),
            StrategyState.EVALUATING: StateDefinition(
                state=StrategyState.EVALUATING,
                description="Strategy is evaluating completion status",
                allowed_transitions={
                    StateTransitionTrigger.EVALUATION_COMPLETE: StrategyState.EXECUTING,
                    StateTransitionTrigger.TASK_COMPLETE: StrategyState.COMPLETING,
                    StateTransitionTrigger.ERROR_OCCURRED: StrategyState.ERROR_RECOVERY,
                    StateTransitionTrigger.STRATEGY_SWITCH_REQUESTED: StrategyState.TRANSITIONING,
                    StateTransitionTrigger.TERMINATION_REQUESTED: StrategyState.TERMINATED,
                }
            ),
            StrategyState.COMPLETING: StateDefinition(
                state=StrategyState.COMPLETING,
                description="Strategy is generating final results",
                allowed_transitions={
                    StateTransitionTrigger.ERROR_OCCURRED: StrategyState.ERROR_RECOVERY,
                    StateTransitionTrigger.TERMINATION_REQUESTED: StrategyState.TERMINATED,
                },
                is_terminal=True
            ),
            StrategyState.ERROR_RECOVERY: StateDefinition(
                state=StrategyState.ERROR_RECOVERY,
                description="Strategy is recovering from an error",
                allowed_transitions={
                    StateTransitionTrigger.RECOVERY_COMPLETE: StrategyState.EXECUTING,
                    StateTransitionTrigger.RETRY_REQUESTED: StrategyState.PLANNING,
                    StateTransitionTrigger.TERMINATION_REQUESTED: StrategyState.FAILED,
                }
            ),
            StrategyState.TRANSITIONING: StateDefinition(
                state=StrategyState.TRANSITIONING,
                description="Strategy is transitioning to another strategy",
                allowed_transitions={
                    StateTransitionTrigger.TERMINATION_REQUESTED: StrategyState.TERMINATED,
                },
                is_terminal=True,
                requires_confirmation=True
            ),
            StrategyState.PAUSED: StateDefinition(
                state=StrategyState.PAUSED,
                description="Strategy execution is paused",
                allowed_transitions={
                    StateTransitionTrigger.RESUME_REQUESTED: StrategyState.EXECUTING,
                    StateTransitionTrigger.TERMINATION_REQUESTED: StrategyState.TERMINATED,
                }
            ),
            StrategyState.TERMINATED: StateDefinition(
                state=StrategyState.TERMINATED,
                description="Strategy execution was terminated",
                allowed_transitions={},
                is_terminal=True
            ),
            StrategyState.FAILED: StateDefinition(
                state=StrategyState.FAILED,
                description="Strategy execution failed",
                allowed_transitions={},
                is_terminal=True
            ),
        }

    async def transition_to(
        self, 
        new_state: StrategyState, 
        trigger: StateTransitionTrigger,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Attempt to transition to a new state.
        
        Args:
            new_state: The target state
            trigger: The trigger causing this transition
            context: Additional context for the transition
            
        Returns:
            True if the transition was successful
        """
        context = context or {}
        
        # Validate the transition is allowed
        if not self._can_transition(self.current_state, new_state, trigger):
            return False

        # Check transition conditions
        if not await self._check_transition_conditions(self.current_state, new_state, context):
            return False

        # Execute pre-transition logic
        if not await self._execute_transition_handler(self.current_state, new_state, context):
            return False

        # Record the transition
        transition = StateTransition(
            from_state=self.current_state,
            to_state=new_state,
            trigger=trigger,
            timestamp=datetime.now(),
            context=context.copy(),
            success=True
        )

        # Update state
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_history.append(transition)
        self.state_entry_time = datetime.now()

        return True

    def _can_transition(
        self, 
        from_state: StrategyState, 
        to_state: StrategyState,
        trigger: StateTransitionTrigger
    ) -> bool:
        """Check if a transition is valid according to state definitions."""
        state_def = self.state_definitions.get(from_state)
        if not state_def:
            return False

        allowed_target = state_def.allowed_transitions.get(trigger)
        return allowed_target == to_state

    async def _check_transition_conditions(
        self,
        from_state: StrategyState,
        to_state: StrategyState,
        context: Dict[str, Any]
    ) -> bool:
        """Check if transition conditions are met."""
        condition = self.transition_conditions.get((from_state, to_state))
        if not condition:
            return True  # No conditions means transition is allowed

        # Check required context keys
        if not condition.required_context_keys.issubset(context.keys()):
            return False

        # Check forbidden context keys
        if condition.forbidden_context_keys.intersection(context.keys()):
            return False

        # Check retry count
        if condition.max_retry_count is not None:
            retry_count = context.get('retry_count', 0)
            if retry_count > condition.max_retry_count:
                return False

        # Check minimum execution time
        if condition.min_execution_time_ms is not None and self.state_entry_time:
            elapsed = (datetime.now() - self.state_entry_time).total_seconds() * 1000
            if elapsed < condition.min_execution_time_ms:
                return False

        # Run custom validator if specified
        if condition.custom_validator:
            validator = self.custom_validators.get(condition.custom_validator)
            if validator and not await validator(from_state, to_state, context):
                return False

        return True

    async def _execute_transition_handler(
        self,
        from_state: StrategyState,
        to_state: StrategyState,
        context: Dict[str, Any]
    ) -> bool:
        """Execute any registered transition handler."""
        handler = self.transition_handlers.get((from_state, to_state))
        if handler:
            try:
                result = await handler(context)
                return result if isinstance(result, bool) else True
            except Exception:
                return False
        return True

    def rollback_to_previous_state(self) -> bool:
        """
        Rollback to the previous stable state.
        
        Returns:
            True if rollback was successful
        """
        if not self.previous_state:
            return False

        # Create rollback transition record
        transition = StateTransition(
            from_state=self.current_state,
            to_state=self.previous_state,
            trigger=StateTransitionTrigger.RETRY_REQUESTED,  # Use retry as rollback trigger
            timestamp=datetime.now(),
            context={'rollback': True},
            success=True
        )

        # Rollback state
        rollback_to = self.previous_state
        self.previous_state = None  # Clear previous since we're rolling back
        self.current_state = rollback_to
        self.state_history.append(transition)
        self.state_entry_time = datetime.now()

        return True

    def register_transition_handler(
        self,
        from_state: StrategyState,
        to_state: StrategyState,
        handler: Callable
    ) -> None:
        """Register a handler for a specific transition."""
        self.transition_handlers[(from_state, to_state)] = handler

    def register_transition_condition(
        self,
        from_state: StrategyState,
        to_state: StrategyState,
        condition: TransitionCondition
    ) -> None:
        """Register conditions for a specific transition."""
        self.transition_conditions[(from_state, to_state)] = condition

    def register_custom_validator(self, name: str, validator: Callable) -> None:
        """Register a custom validation function."""
        self.custom_validators[name] = validator

    def get_allowed_transitions(self) -> List[Tuple[StateTransitionTrigger, StrategyState]]:
        """Get all allowed transitions from the current state."""
        state_def = self.state_definitions.get(self.current_state)
        if not state_def:
            return []
        
        return [(trigger, target) for trigger, target in state_def.allowed_transitions.items()]

    def get_state_history(self) -> List[StateTransition]:
        """Get the complete state transition history."""
        return self.state_history.copy()

    def is_terminal_state(self) -> bool:
        """Check if the current state is terminal (no further transitions possible)."""
        state_def = self.state_definitions.get(self.current_state)
        return state_def.is_terminal if state_def else False

    def get_execution_time_in_state(self) -> float:
        """Get the time spent in the current state in milliseconds."""
        if not self.state_entry_time:
            return 0.0
        return (datetime.now() - self.state_entry_time).total_seconds() * 1000

    def has_state_timeout_exceeded(self) -> bool:
        """Check if the current state has exceeded its timeout."""
        state_def = self.state_definitions.get(self.current_state)
        if not state_def or not state_def.timeout_ms:
            return False
        
        return self.get_execution_time_in_state() > state_def.timeout_ms

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state machine status."""
        state_def = self.state_definitions.get(self.current_state)
        return {
            'current_state': self.current_state.value,
            'previous_state': self.previous_state.value if self.previous_state else None,
            'description': state_def.description if state_def else "Unknown state",
            'time_in_state_ms': self.get_execution_time_in_state(),
            'is_terminal': self.is_terminal_state(),
            'timeout_exceeded': self.has_state_timeout_exceeded(),
            'allowed_transitions': [
                {'trigger': trigger.value, 'target': target.value}
                for trigger, target in self.get_allowed_transitions()
            ],
            'transition_count': len(self.state_history)
        }