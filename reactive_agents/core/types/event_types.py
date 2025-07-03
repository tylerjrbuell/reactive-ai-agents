from enum import Enum


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
