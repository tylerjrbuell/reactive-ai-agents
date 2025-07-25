"""
Event Type Mappings and Utilities

This module provides type mappings and utilities for the ReactiveAgent event system,
enabling precise type inference and validation for dynamic event methods.
"""

from typing import Dict, Type, TypeVar, Union, get_type_hints, Callable, Any
from typing_extensions import Literal, TypedDict

from reactive_agents.core.types.event_types import AgentStateEvent
from reactive_agents.core.types.event_types import (
    BaseEventData,
    SessionStartedEventData,
    SessionEndedEventData,
    TaskStatusChangedEventData,
    IterationStartedEventData,
    IterationCompletedEventData,
    ToolCalledEventData,
    ToolCompletedEventData,
    ToolFailedEventData,
    ReflectionGeneratedEventData,
    FinalAnswerSetEventData,
    MetricsUpdatedEventData,
    ErrorOccurredEventData,
    PauseRequestedEventData,
    PausedEventData,
    ResumeRequestedEventData,
    ResumedEventData,
    TerminateRequestedEventData,
    TerminatedEventData,
    StopRequestedEventData,
    StoppedEventData,
    CancelledEventData,
)
from reactive_agents.core.events.event_bus import EventCallback

# Type variable for event data
T = TypeVar("T", bound=BaseEventData)

# Union type for all possible event data types
EventData = Union[
    SessionStartedEventData,
    SessionEndedEventData,
    TaskStatusChangedEventData,
    IterationStartedEventData,
    IterationCompletedEventData,
    ToolCalledEventData,
    ToolCompletedEventData,
    ToolFailedEventData,
    ReflectionGeneratedEventData,
    FinalAnswerSetEventData,
    MetricsUpdatedEventData,
    ErrorOccurredEventData,
    PauseRequestedEventData,
    PausedEventData,
    ResumeRequestedEventData,
    ResumedEventData,
    TerminateRequestedEventData,
    TerminatedEventData,
    StopRequestedEventData,
    StoppedEventData,
    CancelledEventData,
]

# Mapping from event types to their corresponding data types
EVENT_DATA_TYPE_MAP: Dict[AgentStateEvent, Type[BaseEventData]] = {
    AgentStateEvent.SESSION_STARTED: SessionStartedEventData,
    AgentStateEvent.SESSION_ENDED: SessionEndedEventData,
    AgentStateEvent.TASK_STATUS_CHANGED: TaskStatusChangedEventData,
    AgentStateEvent.ITERATION_STARTED: IterationStartedEventData,
    AgentStateEvent.ITERATION_COMPLETED: IterationCompletedEventData,
    AgentStateEvent.TOOL_CALLED: ToolCalledEventData,
    AgentStateEvent.TOOL_COMPLETED: ToolCompletedEventData,
    AgentStateEvent.TOOL_FAILED: ToolFailedEventData,
    AgentStateEvent.REFLECTION_GENERATED: ReflectionGeneratedEventData,
    AgentStateEvent.FINAL_ANSWER_SET: FinalAnswerSetEventData,
    AgentStateEvent.METRICS_UPDATED: MetricsUpdatedEventData,
    AgentStateEvent.ERROR_OCCURRED: ErrorOccurredEventData,
    AgentStateEvent.PAUSE_REQUESTED: PauseRequestedEventData,
    AgentStateEvent.PAUSED: PausedEventData,
    AgentStateEvent.RESUME_REQUESTED: ResumeRequestedEventData,
    AgentStateEvent.RESUMED: ResumedEventData,
    AgentStateEvent.STOP_REQUESTED: StopRequestedEventData,
    AgentStateEvent.STOPPED: StoppedEventData,
    AgentStateEvent.TERMINATE_REQUESTED: TerminateRequestedEventData,
    AgentStateEvent.TERMINATED: TerminatedEventData,
    AgentStateEvent.CANCELLED: CancelledEventData,
}

# Mapping from event names to their corresponding data types
EVENT_NAME_TO_DATA_TYPE_MAP: Dict[str, Type[BaseEventData]] = {
    event.value: data_type for event, data_type in EVENT_DATA_TYPE_MAP.items()
}

# Mapping from method names to their corresponding data types
METHOD_NAME_TO_DATA_TYPE_MAP: Dict[str, Type[BaseEventData]] = {
    f"on_{event.value}": data_type for event, data_type in EVENT_DATA_TYPE_MAP.items()
}

# Literal types for each event method name
EventMethodName = Literal[
    "on_session_started",
    "on_session_ended",
    "on_task_status_changed",
    "on_iteration_started",
    "on_iteration_completed",
    "on_tool_called",
    "on_tool_completed",
    "on_tool_failed",
    "on_reflection_generated",
    "on_final_answer_set",
    "on_metrics_updated",
    "on_error_occurred",
    "on_pause_requested",
    "on_paused",
    "on_resume_requested",
    "on_resumed",
    "on_stop_requested",
    "on_stopped",
    "on_terminate_requested",
    "on_terminated",
    "on_cancelled",
]


# Event categories for grouping
class EventCategory(TypedDict):
    """Categorization of events for better organization"""

    name: str
    description: str
    events: list[str]


EVENT_CATEGORIES: Dict[str, EventCategory] = {
    "session": {
        "name": "Session Events",
        "description": "Events related to agent session lifecycle",
        "events": ["session_started", "session_ended"],
    },
    "task": {
        "name": "Task Events",
        "description": "Events related to task processing",
        "events": ["task_status_changed"],
    },
    "iteration": {
        "name": "Iteration Events",
        "description": "Events related to agent iteration cycles",
        "events": ["iteration_started", "iteration_completed"],
    },
    "tool": {
        "name": "Tool Events",
        "description": "Events related to tool execution",
        "events": ["tool_called", "tool_completed", "tool_failed"],
    },
    "reflection": {
        "name": "Reflection Events",
        "description": "Events related to agent reasoning and reflection",
        "events": ["reflection_generated", "final_answer_set"],
    },
    "metrics": {
        "name": "Metrics Events",
        "description": "Events related to performance metrics",
        "events": ["metrics_updated"],
    },
    "error": {
        "name": "Error Events",
        "description": "Events related to error handling",
        "events": ["error_occurred"],
    },
    "control": {
        "name": "Control Events",
        "description": "Events related to agent control operations",
        "events": [
            "pause_requested",
            "paused",
            "resume_requested",
            "resumed",
            "stop_requested",
            "stopped",
            "terminate_requested",
            "terminated",
            "cancelled",
        ],
    },
}


# Utility functions
def get_event_data_type(event_name: str) -> Type[BaseEventData] | None:
    """
    Get the data type for a given event name.

    Args:
        event_name: The event name (e.g., 'session_started')

    Returns:
        The corresponding data type class, or None if not found
    """
    return EVENT_NAME_TO_DATA_TYPE_MAP.get(event_name)


def get_method_data_type(method_name: str) -> Type[BaseEventData] | None:
    """
    Get the data type for a given method name.

    Args:
        method_name: The method name (e.g., 'on_session_started')

    Returns:
        The corresponding data type class, or None if not found
    """
    return METHOD_NAME_TO_DATA_TYPE_MAP.get(method_name)


def is_valid_event_method(method_name: str) -> bool:
    """
    Check if a method name is a valid event handler method.

    Args:
        method_name: The method name to check

    Returns:
        True if the method name is a valid event handler
    """
    return method_name in METHOD_NAME_TO_DATA_TYPE_MAP


def get_available_event_methods() -> list[str]:
    """
    Get a list of all available event handler method names.

    Returns:
        List of method names (e.g., ['on_session_started', ...])
    """
    return list(METHOD_NAME_TO_DATA_TYPE_MAP.keys())


def get_available_events() -> list[str]:
    """
    Get a list of all available event names.

    Returns:
        List of event names (e.g., ['session_started', ...])
    """
    return list(EVENT_NAME_TO_DATA_TYPE_MAP.keys())


def get_events_by_category(category: str) -> list[str]:
    """
    Get events belonging to a specific category.

    Args:
        category: The category name (e.g., 'session', 'tool', 'control')

    Returns:
        List of event names in the category
    """
    return EVENT_CATEGORIES.get(category, {}).get("events", [])


def get_event_categories() -> Dict[str, EventCategory]:
    """
    Get all event categories with their descriptions.

    Returns:
        Dictionary mapping category names to category information
    """
    return EVENT_CATEGORIES.copy()


def validate_callback_signature(callback: Callable[..., Any], event_name: str) -> bool:
    """
    Validate that a callback function has the correct signature for an event.

    Args:
        callback: The callback function to validate
        event_name: The event name the callback is for

    Returns:
        True if the callback signature is valid
    """
    try:
        # Get the expected data type
        expected_type = get_event_data_type(event_name)
        if not expected_type:
            return False

        # Get type hints from the callback
        hints = get_type_hints(callback)

        # Basic validation - callback should accept at least one parameter
        import inspect

        sig = inspect.signature(callback)
        params = list(sig.parameters.values())

        # Should have at least one parameter for the event data
        if len(params) < 1:
            return False

        return True

    except Exception:
        # If we can't validate, assume it's valid
        return True
