from __future__ import annotations
from typing import (
    TypedDict,
    Optional,
    Dict,
    Any,
    List,
    Callable,
    Generic,
    TypeVar,
    Protocol,
    runtime_checkable,
    Awaitable,
    cast,
)
import asyncio

from context.agent_observer import AgentStateEvent


# Type definitions for different event data structures
class BaseEventData(TypedDict):
    """Base structure for all event data"""

    timestamp: float
    event_type: str
    agent_name: str
    session_id: str
    task: Optional[str]
    task_status: str
    iterations: int


class SessionStartedEventData(BaseEventData):
    """Data for session started events"""

    initial_task: str


class SessionEndedEventData(BaseEventData):
    """Data for session ended events"""

    final_status: str
    elapsed_time: float


class TaskStatusChangedEventData(BaseEventData):
    """Data for task status changed events"""

    previous_status: str
    new_status: str
    rescoped_task: Optional[str]
    explanation: Optional[str]


class IterationStartedEventData(BaseEventData):
    """Data for iteration started events"""

    iteration: int
    max_iterations: Optional[int]


class IterationCompletedEventData(BaseEventData):
    """Data for iteration completed events"""

    iteration: int
    has_result: bool
    has_plan: bool


class ToolCalledEventData(BaseEventData):
    """Data for tool called events"""

    tool_name: str
    tool_id: str
    parameters: Dict[str, Any]


class ToolCompletedEventData(BaseEventData):
    """Data for tool completed events"""

    tool_name: str
    tool_id: str
    result: Any
    execution_time: float


class ToolFailedEventData(BaseEventData):
    """Data for tool failed events"""

    tool_name: str
    tool_id: str
    error: str
    details: Optional[str]


class ReflectionGeneratedEventData(BaseEventData):
    """Data for reflection generated events"""

    reason: str
    next_step: Optional[str]
    required_tools: List[str]


class FinalAnswerSetEventData(BaseEventData):
    """Data for final answer set events"""

    answer: str


class MetricsUpdatedEventData(BaseEventData):
    """Data for metrics updated events"""

    metrics: Dict[str, Any]


class ErrorOccurredEventData(BaseEventData):
    """Data for error occurred events"""

    error: str
    details: Optional[str]


# Map event types to their corresponding data types
EventDataMapping = {
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
}


# Define generic type for event data
T = TypeVar("T", bound=BaseEventData)


@runtime_checkable
class EventCallback(Protocol, Generic[T]):
    """Protocol for event callbacks with proper typing"""

    def __call__(self, event: T) -> None: ...


@runtime_checkable
class AsyncEventCallback(Protocol, Generic[T]):
    """Protocol for async event callbacks with proper typing"""

    def __call__(self, event: T) -> Awaitable[None]: ...


# Subscription objects for fluent API
class EventSubscription(Generic[T]):
    """
    A subscription to a specific event type with type-safe callbacks.

    This provides methods to manage callbacks for a specific event type
    with proper type checking.
    """

    def __init__(
        self,
        event_type: AgentStateEvent,
        register_callback: Callable[
            [AgentStateEvent, Callable[[Dict[str, Any]], None]], None
        ],
        register_async_callback: Callable[
            [AgentStateEvent, Callable[[Dict[str, Any]], asyncio.Future]], None
        ],
        unregister_callback: Callable[
            [AgentStateEvent, Callable[[Dict[str, Any]], None]], None
        ],
        unregister_async_callback: Callable[
            [AgentStateEvent, Callable[[Dict[str, Any]], asyncio.Future]], None
        ],
    ):
        """
        Initialize an event subscription.

        Args:
            event_type: The event type this subscription is for
            register_callback: Function to register a callback
            register_async_callback: Function to register an async callback
            unregister_callback: Function to unregister a callback
            unregister_async_callback: Function to unregister an async callback
        """
        self.event_type = event_type
        self._register_callback = register_callback
        self._register_async_callback = register_async_callback
        self._unregister_callback = unregister_callback
        self._unregister_async_callback = unregister_async_callback
        self._callbacks: List[EventCallback[T]] = []
        self._async_callbacks: List[AsyncEventCallback[T]] = []

    def subscribe(self, callback: EventCallback[T]) -> "EventSubscription[T]":
        """
        Subscribe a callback to this event.

        Args:
            callback: The callback function to call when this event occurs
                      The callback will receive the event data with proper typing

        Returns:
            self for method chaining
        """
        # Store the callback for later unsubscription
        self._callbacks.append(callback)

        # We need a wrapper to handle the type conversion
        def callback_wrapper(event_data: Dict[str, Any]) -> None:
            callback(event_data)  # type: ignore[reportGeneralTypeIssues]

        self._register_callback(self.event_type, callback_wrapper)
        return self

    async def subscribe_async(
        self, callback: AsyncEventCallback[T]
    ) -> "EventSubscription[T]":
        """
        Subscribe an async callback to this event.

        Args:
            callback: The async callback function to call when this event occurs
                      The callback will receive the event data with proper typing

        Returns:
            self for method chaining
        """
        # Store the callback for later unsubscription
        self._async_callbacks.append(callback)

        # We need a wrapper to handle the type conversion
        def async_wrapper(event_data: Dict[str, Any]) -> asyncio.Future:
            # Create a future to return
            future = asyncio.get_event_loop().create_future()

            # Create a task that will set the future result
            async def run_callback():
                try:
                    # Type cast event_data to T to satisfy type checker
                    result = await callback(
                        cast(T, event_data)
                    )  # This will await the coroutine
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)

            # Start the task
            asyncio.create_task(run_callback())

            # Return the future
            return future

        self._register_async_callback(self.event_type, async_wrapper)
        return self

    def unsubscribe(self, callback: EventCallback[T]) -> "EventSubscription[T]":
        """
        Unsubscribe a callback from this event.

        Args:
            callback: The callback function to unsubscribe

        Returns:
            self for method chaining
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            # We need to unregister it from the observer
            # This would need a more sophisticated approach to find the wrapper
            # For now, just let users know this limitation
            raise NotImplementedError(
                "Unsubscribing individual callbacks is not yet supported. "
                "Please use unsubscribe_all() instead."
            )
        return self

    def unsubscribe_all(self) -> None:
        """
        Unsubscribe all callbacks from this event.
        """
        # This would need access to the actual observer instance
        # to remove all callbacks for this event type
        raise NotImplementedError(
            "The unsubscribe_all() method requires implementation in a class "
            "with direct access to the AgentStateObserver instance."
        )

    def __str__(self) -> str:
        return f"EventSubscription[{self.event_type}]"

    def __repr__(self) -> str:
        return f"EventSubscription[{self.event_type}]({len(self._callbacks)} callbacks, {len(self._async_callbacks)} async callbacks)"


class AgentEventManager:
    """
    Manager for agent events with a type-safe interface.

    This class is intended to be a base class for ReactAgent to provide
    a fluent API for event subscriptions.
    """

    def __init__(self, observer):
        """
        Initialize the event manager with an AgentStateObserver.

        Args:
            observer: The AgentStateObserver instance to use
        """
        self._observer = observer

    def __call__(self, event_type: AgentStateEvent) -> EventSubscription:
        """
        Get a subscription for a specific event type.

        This allows using the manager directly as a callable:
        agent.events(AgentStateEvent.TOOL_CALLED).subscribe(callback)

        Args:
            event_type: The event type to subscribe to

        Returns:
            An EventSubscription instance for the specified event type
        """
        return self.events(event_type)

    def events(self, event_type: AgentStateEvent) -> EventSubscription:
        """
        Get a subscription for a specific event type.

        Args:
            event_type: The event type to subscribe to

        Returns:
            An EventSubscription instance for the specified event type
        """
        # Create a new subscription with the proper event type
        # and register/unregister functions from the observer
        subscription = EventSubscription(
            event_type=event_type,
            register_callback=(
                self._observer.register_callback
                if self._observer
                else lambda *args: None
            ),
            register_async_callback=(
                self._observer.register_async_callback
                if self._observer
                else lambda *args: None
            ),
            unregister_callback=(
                self._observer.unregister_callback
                if self._observer
                else lambda *args: None
            ),
            unregister_async_callback=(
                self._observer.unregister_async_callback
                if self._observer
                else lambda *args: None
            ),
        )

        # Add unsubscribe_all implementation
        def unsubscribe_all():
            # Remove all callbacks for this event type
            if self._observer:
                self._observer._callbacks[event_type] = []
                self._observer._async_callbacks[event_type] = []

        subscription.unsubscribe_all = unsubscribe_all  # type: ignore[reportGeneralTypeIssues]

        return subscription

    def on_session_started(self) -> EventSubscription[SessionStartedEventData]:
        """Subscribe to session started events"""
        return self.events(AgentStateEvent.SESSION_STARTED)  # type: ignore[reportGeneralTypeIssues]

    def on_session_ended(self) -> EventSubscription[SessionEndedEventData]:
        """Subscribe to session ended events"""
        return self.events(AgentStateEvent.SESSION_ENDED)  # type: ignore[reportGeneralTypeIssues]

    def on_task_status_changed(self) -> EventSubscription[TaskStatusChangedEventData]:
        """Subscribe to task status changed events"""
        return self.events(AgentStateEvent.TASK_STATUS_CHANGED)  # type: ignore[reportGeneralTypeIssues]

    def on_iteration_started(self) -> EventSubscription[IterationStartedEventData]:
        """Subscribe to iteration started events"""
        return self.events(AgentStateEvent.ITERATION_STARTED)  # type: ignore[reportGeneralTypeIssues]

    def on_iteration_completed(self) -> EventSubscription[IterationCompletedEventData]:
        """Subscribe to iteration completed events"""
        return self.events(AgentStateEvent.ITERATION_COMPLETED)  # type: ignore[reportGeneralTypeIssues]

    def on_tool_called(self) -> EventSubscription[ToolCalledEventData]:
        """Subscribe to tool called events"""
        return self.events(AgentStateEvent.TOOL_CALLED)  # type: ignore[reportGeneralTypeIssues]

    def on_tool_completed(self) -> EventSubscription[ToolCompletedEventData]:
        """Subscribe to tool completed events"""
        return self.events(AgentStateEvent.TOOL_COMPLETED)  # type: ignore[reportGeneralTypeIssues]

    def on_tool_failed(self) -> EventSubscription[ToolFailedEventData]:
        """Subscribe to tool failed events"""
        return self.events(AgentStateEvent.TOOL_FAILED)  # type: ignore[reportGeneralTypeIssues]

    def on_reflection_generated(
        self,
    ) -> EventSubscription[ReflectionGeneratedEventData]:
        """Subscribe to reflection generated events"""
        return self.events(AgentStateEvent.REFLECTION_GENERATED)  # type: ignore[reportGeneralTypeIssues]

    def on_final_answer_set(self) -> EventSubscription[FinalAnswerSetEventData]:
        """Subscribe to final answer set events"""
        return self.events(AgentStateEvent.FINAL_ANSWER_SET)  # type: ignore[reportGeneralTypeIssues]

    def on_metrics_updated(self) -> EventSubscription[MetricsUpdatedEventData]:
        """Subscribe to metrics updated events"""
        return self.events(AgentStateEvent.METRICS_UPDATED)  # type: ignore[reportGeneralTypeIssues]

    def on_error_occurred(self) -> EventSubscription[ErrorOccurredEventData]:
        """Subscribe to error occurred events"""
        return self.events(AgentStateEvent.ERROR_OCCURRED)  # type: ignore[reportGeneralTypeIssues]
