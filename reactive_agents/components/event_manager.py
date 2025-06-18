"""
Event management module for reactive-ai-agent framework.
Handles event subscription, emission, and management.
"""

from typing import Callable, Dict, Any, Optional, Protocol, Generic, TypeVar
from reactive_agents.common.types.event_types import AgentStateEvent
from reactive_agents.loggers.base import Logger
from reactive_agents.context.agent_events import (
    AgentEventManager,
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
    EventCallback,
)


class EventManager:
    """
    Handles event subscription, emission, and management for agents.
    """

    def __init__(self, agent):
        """Initialize the EventManager with an agent reference."""
        self.agent = agent
        self.context = agent.context
        self.agent_logger = Logger("EventManager", "events", self.context.log_level)
        self._event_manager = AgentEventManager(self.context.state_observer)

    def on_session_started(
        self, callback: EventCallback[SessionStartedEventData]
    ) -> Any:
        """Subscribe to session started events."""
        return self._event_manager.on_session_started().subscribe(callback)

    def on_session_ended(self, callback: EventCallback[SessionEndedEventData]) -> Any:
        """Subscribe to session ended events."""
        return self._event_manager.on_session_ended().subscribe(callback)

    def on_task_status_changed(
        self, callback: EventCallback[TaskStatusChangedEventData]
    ) -> Any:
        """Subscribe to task status changed events."""
        return self._event_manager.on_task_status_changed().subscribe(callback)

    def on_iteration_started(
        self, callback: EventCallback[IterationStartedEventData]
    ) -> Any:
        """Subscribe to iteration started events."""
        return self._event_manager.on_iteration_started().subscribe(callback)

    def on_iteration_completed(
        self, callback: EventCallback[IterationCompletedEventData]
    ) -> Any:
        """Subscribe to iteration completed events."""
        return self._event_manager.on_iteration_completed().subscribe(callback)

    def on_tool_called(self, callback: EventCallback[ToolCalledEventData]) -> Any:
        """Subscribe to tool called events."""
        return self._event_manager.on_tool_called().subscribe(callback)

    def on_tool_completed(self, callback: EventCallback[ToolCompletedEventData]) -> Any:
        """Subscribe to tool completed events."""
        return self._event_manager.on_tool_completed().subscribe(callback)

    def on_tool_failed(self, callback: EventCallback[ToolFailedEventData]) -> Any:
        """Subscribe to tool failed events."""
        return self._event_manager.on_tool_failed().subscribe(callback)

    def on_reflection_generated(
        self, callback: EventCallback[ReflectionGeneratedEventData]
    ) -> Any:
        """Subscribe to reflection generated events."""
        return self._event_manager.on_reflection_generated().subscribe(callback)

    def on_final_answer_set(
        self, callback: EventCallback[FinalAnswerSetEventData]
    ) -> Any:
        """Subscribe to final answer set events."""
        return self._event_manager.on_final_answer_set().subscribe(callback)

    def on_metrics_updated(
        self, callback: EventCallback[MetricsUpdatedEventData]
    ) -> Any:
        """Subscribe to metrics updated events."""
        return self._event_manager.on_metrics_updated().subscribe(callback)

    def on_error_occurred(self, callback: EventCallback[ErrorOccurredEventData]) -> Any:
        """Subscribe to error occurred events."""
        return self._event_manager.on_error_occurred().subscribe(callback)

    def on_pause_requested(
        self, callback: EventCallback[PauseRequestedEventData]
    ) -> Any:
        """Subscribe to pause requested events."""
        return self._event_manager.on_pause_requested().subscribe(callback)

    def on_paused(self, callback: EventCallback[PausedEventData]) -> Any:
        """Subscribe to paused events."""
        return self._event_manager.on_paused().subscribe(callback)

    def on_resume_requested(
        self, callback: EventCallback[ResumeRequestedEventData]
    ) -> Any:
        """Subscribe to resume requested events."""
        return self._event_manager.on_resume_requested().subscribe(callback)

    def on_resumed(self, callback: EventCallback[ResumedEventData]) -> Any:
        """Subscribe to resumed events."""
        return self._event_manager.on_resumed().subscribe(callback)

    def on_terminate_requested(
        self, callback: EventCallback[TerminateRequestedEventData]
    ) -> Any:
        """Subscribe to terminate requested events."""
        return self._event_manager.on_terminate_requested().subscribe(callback)

    def on_terminated(self, callback: EventCallback[TerminatedEventData]) -> Any:
        """Subscribe to terminated events."""
        return self._event_manager.on_terminated().subscribe(callback)

    def on_stop_requested(self, callback: EventCallback[StopRequestedEventData]) -> Any:
        """Subscribe to stop requested events."""
        return self._event_manager.on_stop_requested().subscribe(callback)

    def on_stopped(self, callback: EventCallback[StoppedEventData]) -> Any:
        """Subscribe to stopped events."""
        return self._event_manager.on_stopped().subscribe(callback)

    def on_cancelled(self, callback: EventCallback[CancelledEventData]) -> Any:
        """Subscribe to cancelled events."""
        return self._event_manager.on_cancelled().subscribe(callback)

    def emit_event(self, event_type: AgentStateEvent, data: Dict[str, Any]) -> None:
        """
        Emit an event with the given type and data.

        Args:
            event_type: The type of event to emit
            data: The data to include with the event
        """
        self.context.emit_event(event_type, data)
