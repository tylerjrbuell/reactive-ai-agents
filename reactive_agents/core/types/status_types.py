from enum import Enum


class TaskStatus(Enum):
    """Standardized task status values for agent execution lifecycle."""

    INITIALIZED = "initialized"
    WAITING_DEPENDENCIES = "waiting_for_dependencies"
    RUNNING = "running"
    MISSING_TOOLS = "missing_tools"
    COMPLETE = "complete"
    RESCOPED_COMPLETE = "rescoped_complete"
    MAX_ITERATIONS = "max_iterations_reached"
    ERROR = "error"
    CANCELLED = "cancelled"

    def __str__(self):
        return self.value


class StepStatus(Enum):
    """Status values for individual execution steps."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    RETRYING = "retrying"

    def __str__(self):
        return self.value
