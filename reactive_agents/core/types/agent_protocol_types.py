"""
Agent Protocol Definitions

This module contains Protocol definitions for better type checking and IntelliSense
support for dynamic agent methods.
"""

from typing import Protocol, Callable, Any, Union, List, overload, Literal
from typing_extensions import ParamSpec
from reactive_agents.core.types.event_types import (
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

# Type variable for flexible callback typing
P = ParamSpec("P")


class EventHandlerProtocol(Protocol):
    """
    Protocol defining the dynamic event handler interface for ReactiveAgent.

    This protocol provides precise type safety for dynamically generated event handler
    methods using overloads and literal types. Each event method is precisely typed
    to ensure callbacks receive the correct event data type.

    The agent dynamically creates event handler methods following the pattern:
    on_<event_name>(callback: EventCallback[SpecificEventData]) -> Any

    This protocol ensures full IntelliSense support while preserving the dynamic nature
    of the event system.
    """

    # === Precise overloads for each event method ===

    # Session Events
    @overload
    def __getattr__(
        self, name: Literal["on_session_started"]
    ) -> Callable[[EventCallback[SessionStartedEventData]], Any]: ...

    @overload
    def __getattr__(
        self, name: Literal["on_session_ended"]
    ) -> Callable[[EventCallback[SessionEndedEventData]], Any]: ...

    # Task Events
    @overload
    def __getattr__(
        self, name: Literal["on_task_status_changed"]
    ) -> Callable[[EventCallback[TaskStatusChangedEventData]], Any]: ...

    # Iteration Events
    @overload
    def __getattr__(
        self, name: Literal["on_iteration_started"]
    ) -> Callable[[EventCallback[IterationStartedEventData]], Any]: ...

    @overload
    def __getattr__(
        self, name: Literal["on_iteration_completed"]
    ) -> Callable[[EventCallback[IterationCompletedEventData]], Any]: ...

    # Tool Events
    @overload
    def __getattr__(
        self, name: Literal["on_tool_called"]
    ) -> Callable[[EventCallback[ToolCalledEventData]], Any]: ...

    @overload
    def __getattr__(
        self, name: Literal["on_tool_completed"]
    ) -> Callable[[EventCallback[ToolCompletedEventData]], Any]: ...

    @overload
    def __getattr__(
        self, name: Literal["on_tool_failed"]
    ) -> Callable[[EventCallback[ToolFailedEventData]], Any]: ...

    # Reflection Events
    @overload
    def __getattr__(
        self, name: Literal["on_reflection_generated"]
    ) -> Callable[[EventCallback[ReflectionGeneratedEventData]], Any]: ...

    @overload
    def __getattr__(
        self, name: Literal["on_final_answer_set"]
    ) -> Callable[[EventCallback[FinalAnswerSetEventData]], Any]: ...

    # Metrics Events
    @overload
    def __getattr__(
        self, name: Literal["on_metrics_updated"]
    ) -> Callable[[EventCallback[MetricsUpdatedEventData]], Any]: ...

    # Error Events
    @overload
    def __getattr__(
        self, name: Literal["on_error_occurred"]
    ) -> Callable[[EventCallback[ErrorOccurredEventData]], Any]: ...

    # Control Events - Pause
    @overload
    def __getattr__(
        self, name: Literal["on_pause_requested"]
    ) -> Callable[[EventCallback[PauseRequestedEventData]], Any]: ...

    @overload
    def __getattr__(
        self, name: Literal["on_paused"]
    ) -> Callable[[EventCallback[PausedEventData]], Any]: ...

    # Control Events - Resume
    @overload
    def __getattr__(
        self, name: Literal["on_resume_requested"]
    ) -> Callable[[EventCallback[ResumeRequestedEventData]], Any]: ...

    @overload
    def __getattr__(
        self, name: Literal["on_resumed"]
    ) -> Callable[[EventCallback[ResumedEventData]], Any]: ...

    # Control Events - Stop
    @overload
    def __getattr__(
        self, name: Literal["on_stop_requested"]
    ) -> Callable[[EventCallback[StopRequestedEventData]], Any]: ...

    @overload
    def __getattr__(
        self, name: Literal["on_stopped"]
    ) -> Callable[[EventCallback[StoppedEventData]], Any]: ...

    # Control Events - Terminate
    @overload
    def __getattr__(
        self, name: Literal["on_terminate_requested"]
    ) -> Callable[[EventCallback[TerminateRequestedEventData]], Any]: ...

    @overload
    def __getattr__(
        self, name: Literal["on_terminated"]
    ) -> Callable[[EventCallback[TerminatedEventData]], Any]: ...

    # Control Events - Cancel
    @overload
    def __getattr__(
        self, name: Literal["on_cancelled"]
    ) -> Callable[[EventCallback[CancelledEventData]], Any]: ...

    # Generic fallback for unknown attributes
    @overload
    def __getattr__(self, name: str) -> Any: ...

    def __getattr__(self, name: str) -> Any:
        """
        Dynamic event handler registration with precise type inference.

        Enables dynamic event subscription using the pattern: agent.on_<event_name>(callback)

        Each event method is precisely typed to ensure the callback receives
        the correct event data type, providing full IntelliSense support.

        Args:
            name: The attribute name (should start with 'on_')

        Returns:
            A function that accepts a callback and registers it for the event

        Raises:
            AttributeError: If the attribute is not a valid event handler
        """
        ...

    def get_available_events(self) -> List[str]:
        """Get list of all available event types for subscription."""
        ...

    def subscribe_to_all_events(self, callback: EventCallback[EventData]) -> Any:
        """Subscribe to all available events with a single callback."""
        ...
