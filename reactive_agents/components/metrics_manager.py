from __future__ import annotations
import time
from typing import Dict, Any, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from context.agent_context import AgentContext
    from loggers.base import Logger


class MetricsManager(BaseModel):
    """Manages the collection and retrieval of agent execution metrics."""

    context: AgentContext = Field(exclude=True)  # Reference back to the main context

    # Metrics State
    metrics: Dict[str, Any] = Field(
        default_factory=lambda: {
            "start_time": time.time(),
            "end_time": None,
            "total_time": 0,
            "status": "initialized",  # Will be updated via context.task_status
            "tool_calls": 0,
            "tool_errors": 0,
            "iterations": 0,  # Will be updated via context.iterations
            "tokens": {
                "prompt": 0,
                "completion": 0,
                "total": 0,
            },
            "model_calls": 0,
            "tools": {},  # {tool_name: {"calls": 0, "errors": 0, "total_time": 0}}
            "cache": {  # Managed by ToolManager, but reported here
                "hits": 0,
                "misses": 0,
                "ratio": 0.0,
            },
            "latency": {  # Average/Max latencies can be calculated in get_metrics
                "tool_time": 0,  # Sum of successful tool execution times
                "model_time": 0,  # Sum of model call times
            },
            # Add other specific metrics as needed
        }
    )

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize start time if not already set by default_factory
        if "start_time" not in self.metrics or not self.metrics["start_time"]:
            self.metrics["start_time"] = time.time()

    @property
    def agent_logger(self) -> Logger:
        assert self.context.agent_logger is not None
        return self.context.agent_logger

    def reset(self):
        """Resets metrics for a new run."""
        self.agent_logger.debug("Resetting metrics.")
        start_time = time.time()
        self.metrics = {
            "start_time": start_time,
            "end_time": None,
            "total_time": 0,
            "status": str(self.context.session.task_status),  # Use session status
            "tool_calls": 0,
            "tool_errors": 0,
            "iterations": self.context.session.iterations,  # Use session iterations
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "model_calls": 0,
            "tools": {},
            "cache": {"hits": 0, "misses": 0, "ratio": 0.0},
            "latency": {"tool_time": 0, "model_time": 0},
        }

    def update_tool_metrics(self, tool_history_entry: Dict[str, Any]):
        """Updates metrics based on a tool execution entry from ToolManager history."""
        if not self.context.collect_metrics_enabled:
            return

        self.metrics["tool_calls"] += 1
        tool_name = tool_history_entry.get("name", "unknown")
        is_error = tool_history_entry.get("error", False)
        execution_time = tool_history_entry.get(
            "execution_time"
        )  # Can be None or 0.0 for errors/cache

        if is_error:
            self.metrics["tool_errors"] += 1

        # Initialize tool-specific entry if needed
        if tool_name not in self.metrics["tools"]:
            self.metrics["tools"][tool_name] = {
                "calls": 0,
                "errors": 0,
                "total_time": 0,
            }

        self.metrics["tools"][tool_name]["calls"] += 1
        if is_error:
            self.metrics["tools"][tool_name]["errors"] += 1

        # Only add execution time if the tool actually ran (not cached, not cancelled, not error before execution)
        if (
            execution_time is not None
            and not tool_history_entry.get("cached")
            and not tool_history_entry.get("cancelled")
            and not is_error
        ):
            self.metrics["tools"][tool_name]["total_time"] += execution_time
            self.metrics["latency"]["tool_time"] += execution_time

    def update_model_metrics(self, model_call_data: Dict[str, Any]):
        """Updates metrics after a model call."""
        if not self.context.collect_metrics_enabled:
            return

        self.metrics["model_calls"] += 1

        # Update token counts
        prompt_tokens = model_call_data.get("prompt_tokens", 0)
        completion_tokens = model_call_data.get("completion_tokens", 0)

        self.metrics["tokens"]["prompt"] += prompt_tokens
        self.metrics["tokens"]["completion"] += completion_tokens
        self.metrics["tokens"]["total"] = (
            self.metrics["tokens"]["prompt"] + self.metrics["tokens"]["completion"]
        )

        # Update model latency
        if "time" in model_call_data:
            self.metrics["latency"]["model_time"] += model_call_data["time"]

    def finalize_run_metrics(self):
        """Updates final metrics like end_time, total_time, and status."""
        if not self.context.collect_metrics_enabled:
            return

        end_time = time.time()
        self.metrics["end_time"] = end_time
        self.metrics["total_time"] = end_time - self.metrics["start_time"]
        self.metrics["status"] = str(
            self.context.session.task_status
        )  # Use session status
        self.metrics["iterations"] = (
            self.context.session.iterations  # Use session iterations
        )  # Ensure latest iteration count

        # Update cache metrics from ToolManager
        if self.context.tool_manager:
            self.metrics["cache"]["hits"] = self.context.tool_manager.cache_hits
            self.metrics["cache"]["misses"] = self.context.tool_manager.cache_misses
            total_cache_calls = (
                self.metrics["cache"]["hits"] + self.metrics["cache"]["misses"]
            )
            if total_cache_calls > 0:
                self.metrics["cache"]["ratio"] = (
                    self.metrics["cache"]["hits"] / total_cache_calls
                )
            else:
                self.metrics["cache"]["ratio"] = 0.0

        self.agent_logger.debug("📊📈 Finalized run metrics.")

    def get_metrics(self) -> Dict[str, Any]:
        """Returns the current metrics dictionary."""
        if not self.context.collect_metrics_enabled:
            return {}

        # Ensure metrics are up-to-date before returning
        # If the run hasn't officially ended, calculate current duration
        if self.metrics["end_time"] is None:
            self.metrics["total_time"] = time.time() - self.metrics["start_time"]
            self.metrics["status"] = str(
                self.context.session.task_status
            )  # Use session status
            self.metrics["iterations"] = (
                self.context.session.iterations
            )  # Use session iterations
            # Update cache stats dynamically if run is ongoing
            if self.context.tool_manager:
                self.metrics["cache"]["hits"] = self.context.tool_manager.cache_hits
                self.metrics["cache"]["misses"] = self.context.tool_manager.cache_misses
                total_cache_calls = (
                    self.metrics["cache"]["hits"] + self.metrics["cache"]["misses"]
                )
                self.metrics["cache"]["ratio"] = (
                    (self.metrics["cache"]["hits"] / total_cache_calls)
                    if total_cache_calls > 0
                    else 0.0
                )

        # Calculate aggregate latency metrics (optional, could be done on retrieval)
        # avg_tool_latency = self.metrics["latency"]["tool_time"] / self.metrics["tool_calls"] if self.metrics["tool_calls"] > 0 else 0
        # avg_model_latency = self.metrics["latency"]["model_time"] / self.metrics["model_calls"] if self.metrics["model_calls"] > 0 else 0
        # self.metrics["latency"]["avg_tool_latency"] = avg_tool_latency
        # self.metrics["latency"]["avg_model_latency"] = avg_model_latency

        return self.metrics
