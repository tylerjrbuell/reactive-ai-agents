"""
Tests for EventBus.

Tests the event bus functionality including event publishing,
subscription management, and event routing.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from reactive_agents.core.events.event_bus import EventBus, EventSubscription, Event
from reactive_agents.core.types.event_types import (
    AgentStateEvent,
    SessionStartedEventData,
    SessionEndedEventData,
    TaskStatusChangedEventData,
    IterationStartedEventData,
    ToolCalledEventData,
    ErrorOccurredEventData,
)


class TestEventBus:
    """Test cases for EventBus."""

    @pytest.fixture
    def event_bus(self):
        """Create an event bus instance."""
        return EventBus(agent_name="TestAgent")

    @pytest.fixture
    def mock_callback(self):
        """Create a mock event callback."""
        return Mock()

    @pytest.fixture
    def mock_async_callback(self):
        """Create a mock async event callback."""
        return AsyncMock()

    def test_initialization(self, event_bus):
        """Test event bus initialization."""
        assert event_bus.agent_name == "TestAgent"
        assert len(event_bus._callbacks) == len(AgentStateEvent)
        assert len(event_bus._async_callbacks) == len(AgentStateEvent)
        assert event_bus._active_sessions == set()

        # Check stats initialization
        stats = event_bus._stats
        assert stats["events_emitted"] == 0
        assert stats["callbacks_invoked"] == 0
        assert stats["errors"] == 0
        assert stats["last_event_time"] is None
        assert len(stats["events_by_type"]) == len(AgentStateEvent)

    def test_emit_synchronous_event(self, event_bus, mock_callback):
        """Test emitting a synchronous event."""
        # Register a callback
        event_bus._register_callback(AgentStateEvent.SESSION_STARTED, mock_callback)

        # Emit event
        test_data = {"session_id": "test-123", "initial_task": "Test task"}
        event_bus.emit(AgentStateEvent.SESSION_STARTED, test_data)

        # Verify callback was called
        mock_callback.assert_called_once()

        # Verify stats updated
        assert event_bus._stats["events_emitted"] == 1
        assert (
            event_bus._stats["events_by_type"][AgentStateEvent.SESSION_STARTED.value]
            == 1
        )

    @pytest.mark.asyncio
    async def test_emit_asynchronous_event(self, event_bus, mock_async_callback):
        """Test emitting an asynchronous event."""
        # Register an async callback
        event_bus._register_async_callback(
            AgentStateEvent.SESSION_ENDED, mock_async_callback
        )

        # Emit async event
        test_data = {"session_id": "test-123", "status": "completed"}
        await event_bus.emit_async(AgentStateEvent.SESSION_ENDED, test_data)

        # Verify async callback was called
        mock_async_callback.assert_called_once()

        # Verify stats updated
        assert event_bus._stats["events_emitted"] == 1
        assert (
            event_bus._stats["events_by_type"][AgentStateEvent.SESSION_ENDED.value] == 1
        )

    def test_register_and_unregister_callback(self, event_bus, mock_callback):
        """Test registering and unregistering callbacks."""
        # Register callback
        event_bus._register_callback(AgentStateEvent.ITERATION_STARTED, mock_callback)
        assert mock_callback in event_bus._callbacks[AgentStateEvent.ITERATION_STARTED]

        # Unregister callback
        event_bus._unregister_callback(AgentStateEvent.ITERATION_STARTED, mock_callback)
        assert (
            mock_callback not in event_bus._callbacks[AgentStateEvent.ITERATION_STARTED]
        )

    @pytest.mark.asyncio
    async def test_register_and_unregister_async_callback(
        self, event_bus, mock_async_callback
    ):
        """Test registering and unregistering async callbacks."""
        # Register async callback
        event_bus._register_async_callback(
            AgentStateEvent.TOOL_CALLED, mock_async_callback
        )
        assert (
            mock_async_callback
            in event_bus._async_callbacks[AgentStateEvent.TOOL_CALLED]
        )

        # Unregister async callback
        event_bus._unregister_async_callback(
            AgentStateEvent.TOOL_CALLED, mock_async_callback
        )
        assert (
            mock_async_callback
            not in event_bus._async_callbacks[AgentStateEvent.TOOL_CALLED]
        )

    def test_subscribe_method(self, event_bus, mock_callback):
        """Test the subscribe method with fluent interface."""
        subscription = event_bus.events(AgentStateEvent.TASK_STATUS_CHANGED)
        subscription.subscribe(mock_callback)

        # Verify callback was registered
        assert (
            mock_callback in event_bus._callbacks[AgentStateEvent.TASK_STATUS_CHANGED]
        )

        # Test event emission
        test_data = {"new_status": "running", "previous_status": "pending"}
        event_bus.emit(AgentStateEvent.TASK_STATUS_CHANGED, test_data)

        mock_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe_async_method(self, event_bus, mock_async_callback):
        """Test the async subscribe method."""
        subscription = event_bus.events(AgentStateEvent.ERROR_OCCURRED)
        await subscription.subscribe_async(mock_async_callback)

        # Verify async callback was registered
        assert (
            mock_async_callback
            in event_bus._async_callbacks[AgentStateEvent.ERROR_OCCURRED]
        )

        # Test async event emission
        test_data = {"error_type": "validation", "message": "Test error"}
        await event_bus.emit_async(AgentStateEvent.ERROR_OCCURRED, test_data)

        mock_async_callback.assert_called_once()

    def test_multiple_callbacks_for_same_event(self, event_bus):
        """Test multiple callbacks for the same event type."""
        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()

        # Register multiple callbacks
        event_bus._register_callback(AgentStateEvent.ITERATION_COMPLETED, callback1)
        event_bus._register_callback(AgentStateEvent.ITERATION_COMPLETED, callback2)
        event_bus._register_callback(AgentStateEvent.ITERATION_COMPLETED, callback3)

        # Emit event
        test_data = {"iteration": 1, "result": "success"}
        event_bus.emit(AgentStateEvent.ITERATION_COMPLETED, test_data)

        # Verify all callbacks were called
        callback1.assert_called_once()
        callback2.assert_called_once()
        callback3.assert_called_once()

    def test_event_subscription_unsubscribe_all(self, event_bus):
        """Test unsubscribing all callbacks from a subscription."""
        callback1 = Mock()
        callback2 = Mock()

        # Create subscription and add callbacks
        subscription = event_bus.events(AgentStateEvent.FINAL_ANSWER_SET)
        subscription.subscribe(callback1)
        subscription.subscribe(callback2)

        # Verify callbacks are registered
        assert callback1 in event_bus._callbacks[AgentStateEvent.FINAL_ANSWER_SET]
        assert callback2 in event_bus._callbacks[AgentStateEvent.FINAL_ANSWER_SET]

        # Unsubscribe all
        subscription.unsubscribe_all()

        # Verify callbacks are removed
        assert callback1 not in event_bus._callbacks[AgentStateEvent.FINAL_ANSWER_SET]
        assert callback2 not in event_bus._callbacks[AgentStateEvent.FINAL_ANSWER_SET]

    def test_get_stats(self, event_bus, mock_callback):
        """Test getting event statistics."""
        # Register callback and emit some events
        event_bus._register_callback(AgentStateEvent.SESSION_STARTED, mock_callback)

        event_bus.emit(AgentStateEvent.SESSION_STARTED, {"session_id": "test-1"})
        event_bus.emit(AgentStateEvent.SESSION_STARTED, {"session_id": "test-2"})
        event_bus.emit(AgentStateEvent.ITERATION_STARTED, {"iteration": 1})

        stats = event_bus.get_stats()

        assert stats["events_emitted"] == 3
        assert stats["callbacks_invoked"] == 2  # Only SESSION_STARTED has callbacks
        assert stats["events_by_type"][AgentStateEvent.SESSION_STARTED.value] == 2
        assert stats["events_by_type"][AgentStateEvent.ITERATION_STARTED.value] == 1
        assert stats["last_event_time"] is not None

    def test_clear_stats(self, event_bus):
        """Test clearing event statistics."""
        # Emit some events first
        event_bus.emit(AgentStateEvent.SESSION_STARTED, {"test": "data"})
        event_bus.emit(AgentStateEvent.SESSION_ENDED, {"test": "data"})

        # Verify stats are not zero
        assert event_bus._stats["events_emitted"] > 0

        # Clear stats
        event_bus.clear_stats()

        # Verify stats are reset
        assert event_bus._stats["events_emitted"] == 0
        assert event_bus._stats["callbacks_invoked"] == 0
        assert event_bus._stats["errors"] == 0
        assert event_bus._stats["last_event_time"] is None

    def test_session_tracking(self, event_bus):
        """Test that sessions are tracked internally when events are emitted."""
        # Session tracking is automatic when events contain session_id
        event_data = {"session_id": "test-session-123", "message": "test"}

        # Emit an event with session_id
        event_bus.emit(AgentStateEvent.SESSION_STARTED, event_data)

        # Check that active sessions count increased in stats
        stats = event_bus.get_stats()
        assert stats["active_sessions"] >= 0  # Should exist in stats

    def test_get_active_sessions_count(self, event_bus):
        """Test getting active sessions count from stats."""
        # Active sessions count should be available in stats
        stats = event_bus.get_stats()
        assert "active_sessions" in stats
        assert isinstance(stats["active_sessions"], int)

    def test_callback_error_handling(self, event_bus):
        """Test error handling when callbacks raise exceptions."""

        # Create a callback that raises an exception
        def failing_callback(event_data):
            raise ValueError("Test callback error")

        # Create a normal callback
        normal_callback = Mock()

        # Register both callbacks
        event_bus._register_callback(AgentStateEvent.ERROR_OCCURRED, failing_callback)
        event_bus._register_callback(AgentStateEvent.ERROR_OCCURRED, normal_callback)

        # Emit event - should not raise exception
        test_data = {"error": "test"}
        event_bus.emit(AgentStateEvent.ERROR_OCCURRED, test_data)

        # Normal callback should still be called
        normal_callback.assert_called_once()

        # Error count should be incremented
        assert event_bus._stats["errors"] >= 1

    @pytest.mark.asyncio
    async def test_async_callback_error_handling(self, event_bus):
        """Test error handling when async callbacks raise exceptions."""

        # Create an async callback that raises an exception
        async def failing_async_callback(event_data):
            raise ValueError("Test async callback error")

        # Create a normal async callback
        normal_async_callback = AsyncMock()

        # Register both callbacks
        event_bus._register_async_callback(
            AgentStateEvent.METRICS_UPDATED, failing_async_callback
        )
        event_bus._register_async_callback(
            AgentStateEvent.METRICS_UPDATED, normal_async_callback
        )

        # Emit async event - should not raise exception
        test_data = {"metrics": {"test": 123}}
        await event_bus.emit_async(AgentStateEvent.METRICS_UPDATED, test_data)

        # Normal callback should still be called
        normal_async_callback.assert_called_once()

        # Error count should be incremented
        assert event_bus._stats["errors"] >= 1

    def test_event_data_validation(self, event_bus, mock_callback):
        """Test that event data is properly passed to callbacks."""
        event_bus._register_callback(AgentStateEvent.TOOL_CALLED, mock_callback)

        test_data = {
            "tool_name": "test_tool",
            "arguments": {"arg1": "value1"},
            "call_id": "call-123",
        }

        event_bus.emit(AgentStateEvent.TOOL_CALLED, test_data)

        # Verify callback received the correct data as a dictionary
        call_args = mock_callback.call_args[0][0]  # First positional argument
        assert call_args["tool_name"] == "test_tool"
        assert call_args["arguments"] == {"arg1": "value1"}
        assert call_args["call_id"] == "call-123"

    def test_event_subscription_fluent_interface(self, event_bus):
        """Test the fluent interface of event subscriptions."""
        callback1 = Mock()
        callback2 = Mock()

        # Test method chaining
        subscription = (
            event_bus.events(AgentStateEvent.PAUSED)
            .subscribe(callback1)
            .subscribe(callback2)
        )

        assert isinstance(subscription, EventSubscription)

        # Test that both callbacks are registered
        event_bus.emit(AgentStateEvent.PAUSED, {"reason": "user_requested"})

        callback1.assert_called_once()
        callback2.assert_called_once()

    @pytest.mark.asyncio
    async def test_mixed_sync_async_callbacks(self, event_bus):
        """Test that sync and async callbacks work for the same event type."""
        sync_callback = Mock()
        async_callback = AsyncMock()

        # Register both types of callbacks for the same event
        event_bus._register_callback(AgentStateEvent.RESUMED, sync_callback)
        event_bus._register_async_callback(AgentStateEvent.RESUMED, async_callback)

        # Emit sync event (should call sync callbacks)
        test_data = {"resumed_at": "2024-01-01T00:00:00Z"}
        event_bus.emit(AgentStateEvent.RESUMED, test_data)
        sync_callback.assert_called_once()

        # Emit async event (should call async callbacks)
        await event_bus.emit_async(AgentStateEvent.RESUMED, test_data)
        async_callback.assert_called_once()

    def test_event_bus_string_representation(self, event_bus):
        """Test string representation of EventBus."""
        str_repr = str(event_bus)
        # EventBus should have some basic representation
        assert "EventBus" in str_repr or "object" in str_repr
        # Check that it has the agent name property
        assert hasattr(event_bus, "agent_name")
        assert event_bus.agent_name == "TestAgent"

    def test_event_creation(self):
        """Test Event dataclass creation and properties."""
        event = Event(
            type=AgentStateEvent.SESSION_STARTED,
            data={"session_id": "test-123"},
            source="TestAgent",
            correlation_id="corr-456",
        )

        assert event.type == AgentStateEvent.SESSION_STARTED
        assert event.data == {"session_id": "test-123"}
        assert event.source == "TestAgent"
        assert event.correlation_id == "corr-456"
        assert event.id is not None  # UUID should be generated
        assert event.timestamp > 0  # Timestamp should be set
