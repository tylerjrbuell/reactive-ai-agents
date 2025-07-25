from enum import Enum
from typing import TypedDict, Optional, Dict, Any, List


class AgentStateEvent(str, Enum):
    """Events that can be observed in the agent's lifecycle."""

    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    TASK_STATUS_CHANGED = "task_status_changed"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    TOOL_CALLED = "tool_called"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    REFLECTION_GENERATED = "reflection_generated"
    FINAL_ANSWER_SET = "final_answer_set"
    METRICS_UPDATED = "metrics_updated"
    ERROR_OCCURRED = "error_occurred"
    PAUSE_REQUESTED = "pause_requested"
    PAUSED = "paused"
    RESUME_REQUESTED = "resume_requested"
    RESUMED = "resumed"
    STOP_REQUESTED = "stop_requested"
    STOPPED = "stopped"
    TERMINATE_REQUESTED = "terminate_requested"
    TERMINATED = "terminated"
    CANCELLED = "cancelled"

    # Context Observability Events
    CONTEXT_CHANGED = "context_changed"
    OPERATION_COMPLETED = "operation_completed"
    TOKENS_USED = "tokens_used"
    SNAPSHOT_TAKEN = "snapshot_taken"


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


# --- Agent control event data (pause/resume/stop/terminate) ---
class PauseRequestedEventData(BaseEventData):
    pass


class PausedEventData(BaseEventData):
    pass


class ResumeRequestedEventData(BaseEventData):
    pass


class ResumedEventData(BaseEventData):
    pass


# --- Graceful stop event data ---
class StopRequestedEventData(BaseEventData):
    pass


class StoppedEventData(BaseEventData):
    pass


# --- Forceful terminate event data ---
class TerminateRequestedEventData(BaseEventData):
    pass


class TerminatedEventData(BaseEventData):
    pass


# --- Cancellation event data ---
class CancelledEventData(BaseEventData):
    pass


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
    # --- Agent control events ---
    AgentStateEvent.PAUSE_REQUESTED: PauseRequestedEventData,
    AgentStateEvent.PAUSED: PausedEventData,
    AgentStateEvent.RESUME_REQUESTED: ResumeRequestedEventData,
    AgentStateEvent.RESUMED: ResumedEventData,
    # --- Graceful stop events ---
    AgentStateEvent.STOP_REQUESTED: StopRequestedEventData,
    AgentStateEvent.STOPPED: StoppedEventData,
    # --- Forceful terminate events ---
    AgentStateEvent.TERMINATE_REQUESTED: TerminateRequestedEventData,
    AgentStateEvent.TERMINATED: TerminatedEventData,
    # --- Cancellation events ---
    AgentStateEvent.CANCELLED: CancelledEventData,
}
