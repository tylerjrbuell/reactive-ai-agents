from __future__ import annotations
from typing import Dict, List, Any, Callable, Optional, Set
from enum import Enum
import asyncio
import time

from pydantic import BaseModel, Field
from common.types import TaskStatus


class AgentStateEvent(str, Enum):
    """Events that can be observed in the agent's lifecycle"""

    # Session lifecycle events
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"

    # Task status events
    TASK_STATUS_CHANGED = "task_status_changed"

    # Iteration events
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"

    # Tool events
    TOOL_CALLED = "tool_called"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"

    # Reflection events
    REFLECTION_GENERATED = "reflection_generated"

    # Result events
    FINAL_ANSWER_SET = "final_answer_set"

    # Metrics events
    METRICS_UPDATED = "metrics_updated"

    # Error events
    ERROR_OCCURRED = "error_occurred"


# Define callback type
ObserverCallback = Callable[[Dict[str, Any]], None]
AsyncObserverCallback = Callable[[Dict[str, Any]], asyncio.Future]


class AgentStateObserver:
    """
    Observer for tracking ReactAgent state in real-time.

    This class manages callbacks for different agent state events,
    allowing external systems to hook into agent state changes.
    """

    def __init__(self):
        """Initialize the observer with empty callback registries"""
        # Mapping of event types to list of registered callbacks
        self._callbacks: Dict[AgentStateEvent, List[ObserverCallback]] = {
            event_type: [] for event_type in AgentStateEvent
        }

        # Mapping of event types to list of registered async callbacks
        self._async_callbacks: Dict[AgentStateEvent, List[AsyncObserverCallback]] = {
            event_type: [] for event_type in AgentStateEvent
        }

        # Set to track session IDs that have started to prevent duplicate notifications
        self._active_sessions: Set[str] = set()

        # Stats tracking
        self._stats = {
            "events_emitted": 0,
            "callbacks_invoked": 0,
            "events_by_type": {event_type.value: 0 for event_type in AgentStateEvent},
            "last_event_time": None,
        }

    def register_callback(
        self, event_type: AgentStateEvent, callback: ObserverCallback
    ) -> None:
        """
        Register a synchronous callback for a specific event type.

        Args:
            event_type: The event type to listen for
            callback: The callback function to invoke when the event occurs
        """
        self._callbacks[event_type].append(callback)

    def register_async_callback(
        self, event_type: AgentStateEvent, callback: AsyncObserverCallback
    ) -> None:
        """
        Register an asynchronous callback for a specific event type.

        Args:
            event_type: The event type to listen for
            callback: The async callback function to invoke when the event occurs
        """
        self._async_callbacks[event_type].append(callback)

    def unregister_callback(
        self, event_type: AgentStateEvent, callback: ObserverCallback
    ) -> None:
        """
        Unregister a previously registered callback.

        Args:
            event_type: The event type the callback was registered for
            callback: The callback function to remove
        """
        if callback in self._callbacks[event_type]:
            self._callbacks[event_type].remove(callback)

    def unregister_async_callback(
        self, event_type: AgentStateEvent, callback: AsyncObserverCallback
    ) -> None:
        """
        Unregister a previously registered async callback.

        Args:
            event_type: The event type the callback was registered for
            callback: The async callback function to remove
        """
        if callback in self._async_callbacks[event_type]:
            self._async_callbacks[event_type].remove(callback)

    def emit(self, event_type: AgentStateEvent, data: Dict[str, Any]) -> None:
        """
        Emit an event to all registered callbacks.

        Args:
            event_type: The type of event being emitted
            data: The data associated with the event
        """
        # Update stats
        self._stats["events_emitted"] += 1
        self._stats["events_by_type"][event_type.value] += 1
        self._stats["last_event_time"] = time.time()

        # Add timestamp to the event data
        event_data = {"timestamp": time.time(), "event_type": event_type.value, **data}

        # Special handling for session events to avoid duplicates
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

        # Call sync callbacks
        for callback in self._callbacks[event_type]:
            try:
                callback(event_data)
                self._stats["callbacks_invoked"] += 1
            except Exception as e:
                print(f"Error in observer callback: {e}")

    async def emit_async(
        self, event_type: AgentStateEvent, data: Dict[str, Any]
    ) -> None:
        """
        Emit an event to all registered async callbacks.

        Args:
            event_type: The type of event being emitted
            data: The data associated with the event
        """
        # Update stats
        self._stats["events_emitted"] += 1
        self._stats["events_by_type"][event_type.value] += 1
        self._stats["last_event_time"] = time.time()

        # Add timestamp to the event data
        event_data = {"timestamp": time.time(), "event_type": event_type.value, **data}

        # Special handling for session events to avoid duplicates
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
                    print(f"Error in async observer callback: {e}")

            if tasks:
                # Wait for all callbacks to complete
                await asyncio.gather(*tasks, return_exceptions=True)
                self._stats["callbacks_invoked"] += len(tasks)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the observer's activity"""
        return self._stats
