"""
Unified Event System for Reactive Agents

This module provides a single, comprehensive event system that replaces:
- AgentStateObserver
- EventManager
- AgentEventManager
- EventBus (simplified)

Features:
- Type-safe event subscriptions
- Synchronous and asynchronous callbacks
- Event statistics and debugging
- Simple, clean API
- Backward compatibility with existing code
"""

from __future__ import annotations
import asyncio
import time
import uuid
from typing import (
    Dict,
    List,
    Any,
    Callable,
    Optional,
    Set,
    Union,
    Protocol,
    runtime_checkable,
    Awaitable,
    TypeVar,
    Generic,
    cast,
)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

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
    StopRequestedEventData,
    StoppedEventData,
    TerminateRequestedEventData,
    TerminatedEventData,
    CancelledEventData,
    EventDataMapping,
)

# Type definitions
T = TypeVar("T", bound=BaseEventData, contravariant=True)


# Protocol definitions for callbacks
@runtime_checkable
class EventCallback(Protocol, Generic[T]):
    """Protocol for event callbacks with proper typing"""

    def __call__(self, event: T) -> None: ...


@runtime_checkable
class AsyncEventCallback(Protocol, Generic[T]):
    """Protocol for async event callbacks with proper typing"""

    def __call__(self, event: T) -> Awaitable[None]: ...


# Event data mapping is now imported from event_types.py


@dataclass
class Event:
    """Enhanced event structure with metadata"""

    type: AgentStateEvent = field()
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure timestamp is set if not provided"""
        if not self.timestamp:
            self.timestamp = time.time()


class EventSubscription(Generic[T]):
    """
    A subscription to a specific event type with type-safe callbacks.
    """

    def __init__(
        self,
        event_type: AgentStateEvent,
        event_bus: "EventBus",
    ):
        self.event_type = event_type
        self.event_bus = event_bus
        self._callbacks: List[EventCallback[T]] = []
        self._async_callbacks: List[AsyncEventCallback[T]] = []

    def subscribe(self, callback: EventCallback[T]) -> "EventSubscription[T]":
        """Subscribe a callback to this event."""
        self._callbacks.append(callback)
        self.event_bus._register_callback(self.event_type, callback)
        return self

    async def subscribe_async(
        self, callback: AsyncEventCallback[T]
    ) -> "EventSubscription[T]":
        """Subscribe an async callback to this event."""
        self._async_callbacks.append(callback)
        self.event_bus._register_async_callback(self.event_type, callback)
        return self

    def unsubscribe_all(self) -> None:
        """Unsubscribe all callbacks from this event."""
        for callback in self._callbacks:
            self.event_bus._unregister_callback(self.event_type, callback)
        for callback in self._async_callbacks:
            self.event_bus._unregister_async_callback(self.event_type, callback)
        self._callbacks.clear()
        self._async_callbacks.clear()


class EventBus:
    """
    Unified event system that combines the best features of all previous event systems.

    This replaces:
    - AgentStateObserver
    - EventManager
    - AgentEventManager
    - Previous EventBus (simplified)
    """

    def __init__(self, agent_name: str = "UnknownAgent"):
        self.agent_name = agent_name

        # Callback storage
        self._callbacks: Dict[AgentStateEvent, List[EventCallback]] = {
            event_type: [] for event_type in AgentStateEvent
        }
        self._async_callbacks: Dict[AgentStateEvent, List[AsyncEventCallback]] = {
            event_type: [] for event_type in AgentStateEvent
        }

        # Session tracking
        self._active_sessions: Set[str] = set()

        # Statistics
        self._stats = {
            "events_emitted": 0,
            "callbacks_invoked": 0,
            "events_by_type": {event_type.value: 0 for event_type in AgentStateEvent},
            "last_event_time": None,
            "errors": 0,
        }

    # === Core Event Methods ===

    def emit(self, event_type: AgentStateEvent, data: Dict[str, Any]) -> None:
        """Emit a synchronous event."""
        self._emit_event(event_type, data, is_async=False)

    async def emit_async(
        self, event_type: AgentStateEvent, data: Dict[str, Any]
    ) -> None:
        """Emit an asynchronous event."""
        await self._emit_event_async(event_type, data)

    def _emit_event(
        self, event_type: AgentStateEvent, data: Dict[str, Any], is_async: bool = False
    ) -> None:
        """Internal method to emit events."""
        # Update stats
        self._stats["events_emitted"] += 1
        self._stats["events_by_type"][event_type.value] += 1
        self._stats["last_event_time"] = time.time()

        # Add timestamp and agent context
        event_data = {
            "timestamp": time.time(),
            "event_type": event_type.value,
            "agent_name": self.agent_name,
            **data,
        }

        # Special handling for session events
        if event_type == AgentStateEvent.SESSION_STARTED:
            session_id = data.get("session_id")
            if session_id in self._active_sessions:
                return
            if session_id:
                self._active_sessions.add(session_id)
        elif event_type == AgentStateEvent.SESSION_ENDED:
            session_id = data.get("session_id")
            if session_id:
                self._active_sessions.discard(session_id)

        # Call synchronous callbacks
        for callback in self._callbacks[event_type]:
            try:
                callback(event_data)
                self._stats["callbacks_invoked"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                print(f"Error in event callback: {e}")

    async def _emit_event_async(
        self, event_type: AgentStateEvent, data: Dict[str, Any]
    ) -> None:
        """Internal method to emit async events."""
        # Update stats
        self._stats["events_emitted"] += 1
        self._stats["events_by_type"][event_type.value] += 1
        self._stats["last_event_time"] = time.time()

        # Add timestamp and agent context
        event_data = {
            "timestamp": time.time(),
            "event_type": event_type.value,
            "agent_name": self.agent_name,
            **data,
        }

        # Special handling for session events
        if event_type == AgentStateEvent.SESSION_STARTED:
            session_id = data.get("session_id")
            if session_id in self._active_sessions:
                return
            if session_id:
                self._active_sessions.add(session_id)
        elif event_type == AgentStateEvent.SESSION_ENDED:
            session_id = data.get("session_id")
            if session_id:
                self._active_sessions.discard(session_id)

        # Call async callbacks concurrently
        if self._async_callbacks[event_type]:
            tasks = []
            for callback in self._async_callbacks[event_type]:
                try:
                    tasks.append(callback(event_data))
                except Exception as e:
                    self._stats["errors"] += 1
                    print(f"Error in async event callback: {e}")

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                self._stats["callbacks_invoked"] += len(tasks)

    # === Registration Methods ===

    def _register_callback(
        self, event_type: AgentStateEvent, callback: EventCallback
    ) -> None:
        """Register a synchronous callback."""
        self._callbacks[event_type].append(callback)

    def _register_async_callback(
        self, event_type: AgentStateEvent, callback: AsyncEventCallback
    ) -> None:
        """Register an asynchronous callback."""
        self._async_callbacks[event_type].append(callback)

    def _unregister_callback(
        self, event_type: AgentStateEvent, callback: EventCallback
    ) -> None:
        """Unregister a synchronous callback."""
        if callback in self._callbacks[event_type]:
            self._callbacks[event_type].remove(callback)

    def _unregister_async_callback(
        self, event_type: AgentStateEvent, callback: AsyncEventCallback
    ) -> None:
        """Unregister an asynchronous callback."""
        if callback in self._async_callbacks[event_type]:
            self._async_callbacks[event_type].remove(callback)

    # === Fluent API Methods ===

    def events(self, event_type: AgentStateEvent) -> EventSubscription:
        """Get a subscription for a specific event type."""
        return EventSubscription(event_type, self)

    # === Convenience Methods for Common Events ===

    def on_session_started(self) -> EventSubscription[SessionStartedEventData]:
        """Subscribe to session started events."""
        return self.events(AgentStateEvent.SESSION_STARTED)

    def on_session_ended(self) -> EventSubscription[SessionEndedEventData]:
        """Subscribe to session ended events."""
        return self.events(AgentStateEvent.SESSION_ENDED)

    def on_task_status_changed(self) -> EventSubscription[TaskStatusChangedEventData]:
        """Subscribe to task status changed events."""
        return self.events(AgentStateEvent.TASK_STATUS_CHANGED)

    def on_iteration_started(self) -> EventSubscription[IterationStartedEventData]:
        """Subscribe to iteration started events."""
        return self.events(AgentStateEvent.ITERATION_STARTED)

    def on_iteration_completed(self) -> EventSubscription[IterationCompletedEventData]:
        """Subscribe to iteration completed events."""
        return self.events(AgentStateEvent.ITERATION_COMPLETED)

    def on_tool_called(self) -> EventSubscription[ToolCalledEventData]:
        """Subscribe to tool called events."""
        return self.events(AgentStateEvent.TOOL_CALLED)

    def on_tool_completed(self) -> EventSubscription[ToolCompletedEventData]:
        """Subscribe to tool completed events."""
        return self.events(AgentStateEvent.TOOL_COMPLETED)

    def on_tool_failed(self) -> EventSubscription[ToolFailedEventData]:
        """Subscribe to tool failed events."""
        return self.events(AgentStateEvent.TOOL_FAILED)

    def on_reflection_generated(
        self,
    ) -> EventSubscription[ReflectionGeneratedEventData]:
        """Subscribe to reflection generated events."""
        return self.events(AgentStateEvent.REFLECTION_GENERATED)

    def on_final_answer_set(self) -> EventSubscription[FinalAnswerSetEventData]:
        """Subscribe to final answer set events."""
        return self.events(AgentStateEvent.FINAL_ANSWER_SET)

    def on_metrics_updated(self) -> EventSubscription[MetricsUpdatedEventData]:
        """Subscribe to metrics updated events."""
        return self.events(AgentStateEvent.METRICS_UPDATED)

    def on_error_occurred(self) -> EventSubscription[ErrorOccurredEventData]:
        """Subscribe to error occurred events."""
        return self.events(AgentStateEvent.ERROR_OCCURRED)

    def on_pause_requested(self) -> EventSubscription[PauseRequestedEventData]:
        """Subscribe to pause requested events."""
        return self.events(AgentStateEvent.PAUSE_REQUESTED)

    def on_paused(self) -> EventSubscription[PausedEventData]:
        """Subscribe to paused events."""
        return self.events(AgentStateEvent.PAUSED)

    def on_resume_requested(self) -> EventSubscription[ResumeRequestedEventData]:
        """Subscribe to resume requested events."""
        return self.events(AgentStateEvent.RESUME_REQUESTED)

    def on_resumed(self) -> EventSubscription[ResumedEventData]:
        """Subscribe to resumed events."""
        return self.events(AgentStateEvent.RESUMED)

    def on_stop_requested(self) -> EventSubscription[StopRequestedEventData]:
        """Subscribe to stop requested events."""
        return self.events(AgentStateEvent.STOP_REQUESTED)

    def on_stopped(self) -> EventSubscription[StoppedEventData]:
        """Subscribe to stopped events."""
        return self.events(AgentStateEvent.STOPPED)

    def on_terminate_requested(self) -> EventSubscription[TerminateRequestedEventData]:
        """Subscribe to terminate requested events."""
        return self.events(AgentStateEvent.TERMINATE_REQUESTED)

    def on_terminated(self) -> EventSubscription[TerminatedEventData]:
        """Subscribe to terminated events."""
        return self.events(AgentStateEvent.TERMINATED)

    def on_cancelled(self) -> EventSubscription[CancelledEventData]:
        """Subscribe to cancelled events."""
        return self.events(AgentStateEvent.CANCELLED)

    # === Backward Compatibility Methods ===

    def register_callback(
        self, event_type: AgentStateEvent, callback: EventCallback
    ) -> None:
        """Backward compatibility method."""
        self._register_callback(event_type, callback)

    def register_async_callback(
        self, event_type: AgentStateEvent, callback: AsyncEventCallback
    ) -> None:
        """Backward compatibility method."""
        self._register_async_callback(event_type, callback)

    def unregister_callback(
        self, event_type: AgentStateEvent, callback: EventCallback
    ) -> None:
        """Backward compatibility method."""
        self._unregister_callback(event_type, callback)

    def unregister_async_callback(
        self, event_type: AgentStateEvent, callback: AsyncEventCallback
    ) -> None:
        """Backward compatibility method."""
        self._unregister_async_callback(event_type, callback)

    # === Statistics and Debugging ===

    def get_stats(self) -> Dict[str, Any]:
        """Get event system statistics."""
        return {
            **self._stats,
            "active_sessions": len(self._active_sessions),
            "total_callbacks": sum(
                len(callbacks) for callbacks in self._callbacks.values()
            ),
            "total_async_callbacks": sum(
                len(callbacks) for callbacks in self._async_callbacks.values()
            ),
        }

    def clear_stats(self) -> None:
        """Clear event statistics."""
        self._stats = {
            "events_emitted": 0,
            "callbacks_invoked": 0,
            "events_by_type": {event_type.value: 0 for event_type in AgentStateEvent},
            "last_event_time": None,
            "errors": 0,
        }

    # === Utility Methods ===

    def get_subscriber_count(self, event_type: AgentStateEvent) -> int:
        """Get number of subscribers for an event type."""
        return len(self._callbacks[event_type]) + len(self._async_callbacks[event_type])

    def has_subscribers(self, event_type: AgentStateEvent) -> bool:
        """Check if an event type has any subscribers."""
        return bool(self._callbacks[event_type] or self._async_callbacks[event_type])

    def get_all_event_types(self) -> List[AgentStateEvent]:
        """Get all available event types."""
        return list(AgentStateEvent)

    def clear_all_subscriptions(self) -> None:
        """Clear all event subscriptions."""
        for event_type in AgentStateEvent:
            self._callbacks[event_type].clear()
            self._async_callbacks[event_type].clear()
        self._active_sessions.clear()
