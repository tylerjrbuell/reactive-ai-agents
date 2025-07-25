from __future__ import annotations
from typing import (
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
from reactive_agents.core.types.event_types import AgentStateEvent
from reactive_agents.core.events.event_bus import EventBus


# Define generic type for event data
T = TypeVar("T", bound=Dict[str, Any], contravariant=True)


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
