from __future__ import annotations
from typing import Dict, List, Any, Callable, Set, Optional
from reactive_agents.core.types.event_types import AgentStateEvent
import asyncio
import time
import json
from datetime import datetime


# Define callback type
ObserverCallback = Callable[[Dict[str, Any]], None]
AsyncObserverCallback = Callable[[Dict[str, Any]], asyncio.Future[Any]]


class AgentStateObserver:
    """
    Enhanced Observer for tracking ReactAgent state and context in real-time.

    This class manages callbacks for different agent state events and context observability,
    allowing external systems to hook into agent state changes and context management.
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

        # Context observability tracking
        self._context_history: List[Dict[str, Any]] = []
        self._operation_timings: Dict[str, List[float]] = {}
        self._token_usage: List[Dict[str, Any]] = []
        self._context_snapshots: List[Dict[str, Any]] = []
        self._context_events: List[Dict[str, Any]] = []
        self._operation_events: List[Dict[str, Any]] = []
        self._token_events: List[Dict[str, Any]] = []
        self._snapshot_events: List[Dict[str, Any]] = []

        # Stats tracking
        self._stats = {
            "events_emitted": 0,
            "callbacks_invoked": 0,
            "events_by_type": {event_type.value: 0 for event_type in AgentStateEvent},
            "last_event_time": None,
            # Context observability stats
            "context_changes": 0,
            "operations_tracked": 0,
            "tokens_tracked": 0,
            "snapshots_taken": 0,
        }

    # === Existing Methods (Unchanged) ===
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

    # === Context Observability Methods ===

    def track_context_change(
        self,
        change_type: str,
        message_count_before: int,
        message_count_after: int,
        token_estimate_before: int,
        token_estimate_after: int,
        operation_duration: float,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Track a context change event.

        Args:
            change_type: Type of context change (pruning, summarization, etc.)
            message_count_before: Number of messages before the change
            message_count_after: Number of messages after the change
            token_estimate_before: Estimated tokens before the change
            token_estimate_after: Estimated tokens after the change
            operation_duration: Duration of the operation in seconds
            additional_data: Additional context data
        """
        change_data = {
            "timestamp": time.time(),
            "change_type": change_type,
            "message_count_before": message_count_before,
            "message_count_after": message_count_after,
            "token_estimate_before": token_estimate_before,
            "token_estimate_after": token_estimate_after,
            "operation_duration": operation_duration,
            "token_delta": token_estimate_after - token_estimate_before,
            "message_delta": message_count_after - message_count_before,
            "additional_data": additional_data or {},
        }

        self._context_history.append(change_data)
        self._context_events.append(change_data)
        self._stats["context_changes"] += 1

        # Emit context change event
        self.emit(AgentStateEvent.CONTEXT_CHANGED, change_data)

    def track_operation(
        self,
        operation_name: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Track an operation timing.

        Args:
            operation_name: Name of the operation
            duration: Duration in seconds
            success: Whether the operation was successful
            metadata: Additional operation metadata
        """
        operation_data = {
            "timestamp": time.time(),
            "operation_name": operation_name,
            "duration": duration,
            "success": success,
            "metadata": metadata or {},
        }

        if operation_name not in self._operation_timings:
            self._operation_timings[operation_name] = []
        self._operation_timings[operation_name].append(duration)
        self._operation_events.append(operation_data)
        self._stats["operations_tracked"] += 1

        # Emit operation completed event
        self.emit(AgentStateEvent.OPERATION_COMPLETED, operation_data)

    def track_token_usage(
        self,
        tokens_used: int,
        operation_type: str,
        efficiency_ratio: float = 1.0,
        context_size: Optional[int] = None,
    ) -> None:
        """
        Track token usage for operations.

        Args:
            tokens_used: Number of tokens used
            operation_type: Type of operation (summarization, pruning, etc.)
            efficiency_ratio: Efficiency ratio (1.0 = optimal)
            context_size: Size of context when tokens were used
        """
        token_data = {
            "timestamp": time.time(),
            "tokens_used": tokens_used,
            "operation_type": operation_type,
            "efficiency_ratio": efficiency_ratio,
            "context_size": context_size,
        }

        self._token_usage.append(token_data)
        self._token_events.append(token_data)
        self._stats["tokens_tracked"] += 1

        # Emit token usage event
        self.emit(AgentStateEvent.TOKENS_USED, token_data)

    def take_context_snapshot(
        self,
        snapshot_type: str,
        context_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Take a snapshot of current context state.

        Args:
            snapshot_type: Type of snapshot (before_pruning, after_summarization, etc.)
            context_data: Current context data
            metadata: Additional snapshot metadata
        """
        snapshot_data = {
            "timestamp": time.time(),
            "snapshot_type": snapshot_type,
            "context_data": context_data,
            "metadata": metadata or {},
        }

        self._context_snapshots.append(snapshot_data)
        self._snapshot_events.append(snapshot_data)
        self._stats["snapshots_taken"] += 1

        # Emit snapshot taken event
        self.emit(AgentStateEvent.SNAPSHOT_TAKEN, snapshot_data)

    # === Context Observability Analysis Methods ===

    def get_context_insights(self) -> Dict[str, Any]:
        """Get insights about context management."""
        insights = {
            "context_changes": self._analyze_context_changes(),
            "operation_timings": self._analyze_operation_timings(),
            "token_usage": self._analyze_token_usage(),
            "snapshots_count": len(self._context_snapshots),
        }
        return insights

    def _analyze_context_changes(self) -> Dict[str, Any]:
        """Analyze context change patterns."""
        if not self._context_history:
            return {"total_changes": 0, "total_tokens_used": 0, "avg_message_change": 0}

        total_changes = len(self._context_history)
        total_tokens_used = sum(
            change["token_delta"] for change in self._context_history
        )
        avg_message_change = (
            sum(abs(change["message_delta"]) for change in self._context_history)
            / total_changes
        )

        return {
            "total_changes": total_changes,
            "total_tokens_used": total_tokens_used,
            "avg_message_change": avg_message_change,
        }

    def _analyze_operation_timings(self) -> Dict[str, Dict[str, Any]]:
        """Analyze operation timing patterns."""
        analysis = {}
        for operation_name, timings in self._operation_timings.items():
            if timings:
                analysis[operation_name] = {
                    "count": len(timings),
                    "avg_duration": sum(timings) / len(timings),
                    "min_duration": min(timings),
                    "max_duration": max(timings),
                }
        return analysis

    def _analyze_token_usage(self) -> Dict[str, Any]:
        """Analyze token usage patterns."""
        if not self._token_usage:
            return {}

        total_tokens_used = sum(usage["tokens_used"] for usage in self._token_usage)
        avg_efficiency_ratio = sum(
            usage["efficiency_ratio"] for usage in self._token_usage
        ) / len(self._token_usage)

        return {
            "total_tokens_used": total_tokens_used,
            "avg_efficiency_ratio": avg_efficiency_ratio,
            "usage_by_operation": self._group_token_usage_by_operation(),
        }

    def _group_token_usage_by_operation(self) -> Dict[str, int]:
        """Group token usage by operation type."""
        usage_by_operation = {}
        for usage in self._token_usage:
            operation = usage["operation_type"]
            usage_by_operation[operation] = (
                usage_by_operation.get(operation, 0) + usage["tokens_used"]
            )
        return usage_by_operation

    def export_debug_data(self) -> Dict[str, Any]:
        """Export all debug data for analysis."""
        return {
            "context_history": self._context_history,
            "operation_timings": self._operation_timings,
            "token_usage": self._token_usage,
            "context_snapshots": self._context_snapshots,
            "stats": self._stats,
            "export_timestamp": datetime.now().isoformat(),
        }

    def clear_context_data(self) -> None:
        """Clear all context observability data."""
        self._context_history.clear()
        self._operation_timings.clear()
        self._token_usage.clear()
        self._context_snapshots.clear()
        self._context_events.clear()
        self._operation_events.clear()
        self._token_events.clear()
        self._snapshot_events.clear()

    # === Event Access Methods ===

    @property
    def context_events(self) -> List[Dict[str, Any]]:
        """Get all context change events."""
        return self._context_events.copy()

    @property
    def operation_events(self) -> List[Dict[str, Any]]:
        """Get all operation events."""
        return self._operation_events.copy()

    @property
    def token_events(self) -> List[Dict[str, Any]]:
        """Get all token usage events."""
        return self._token_events.copy()

    @property
    def snapshot_events(self) -> List[Dict[str, Any]]:
        """Get all snapshot events."""
        return self._snapshot_events.copy()
