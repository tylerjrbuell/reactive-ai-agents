from __future__ import annotations
import time
from typing import Dict, Any, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext
    from reactive_agents.utils.logging import Logger


class ToolMetrics(BaseModel):
    """Metrics for a single tool."""
    calls: int = 0
    errors: int = 0
    total_time: float = 0.0


class MetricsManager(BaseModel):
    """A self-contained, active model for managing and calculating agent execution metrics."""

    context: AgentContext = Field(exclude=True)

    # Core Metrics
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "initialized"
    tool_calls: int = 0
    tool_errors: int = 0
    iterations: int = 0
    model_calls: int = 0

    # Detailed Metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tools: Dict[str, ToolMetrics] = Field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    tool_latency: float = 0.0
    model_latency: float = 0.0

    class Config:
        arbitrary_types_allowed = True

    @property
    def agent_logger(self) -> Logger:
        assert self.context.agent_logger is not None
        return self.context.agent_logger

    @property
    def total_time(self) -> float:
        """Calculates the total execution time."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def total_tokens(self) -> int:
        """Calculates the total tokens used."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_ratio(self) -> float:
        """Calculates the cache hit ratio."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def reset(self):
        """Resets metrics for a new run."""
        self.agent_logger.debug("Resetting metrics.")
        self.start_time = time.time()
        self.end_time = None
        self.status = str(self.context.session.task_status)
        self.tool_calls = 0
        self.tool_errors = 0
        self.iterations = self.context.session.iterations
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model_calls = 0
        self.tools = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.tool_latency = 0.0
        self.model_latency = 0.0

    def update_tool_metrics(self, tool_history_entry: Dict[str, Any]):
        """Updates metrics based on a tool execution entry from ToolManager history."""
        if not self.context.collect_metrics_enabled:
            return

        self.tool_calls += 1
        tool_name = tool_history_entry.get("name", "unknown")
        is_error = tool_history_entry.get("error", False)
        execution_time = tool_history_entry.get("execution_time")

        if is_error:
            self.tool_errors += 1

        if tool_name not in self.tools:
            self.tools[tool_name] = ToolMetrics()

        self.tools[tool_name].calls += 1
        if is_error:
            self.tools[tool_name].errors += 1

        if execution_time is not None and not tool_history_entry.get("cached") and not is_error:
            self.tools[tool_name].total_time += execution_time
            self.tool_latency += execution_time

    def update_model_metrics(self, model_call_data: Dict[str, Any]):
        """Updates metrics after a model call."""
        if not self.context.collect_metrics_enabled:
            return

        self.model_calls += 1
        self.prompt_tokens += model_call_data.get("prompt_tokens", 0) or 0
        self.completion_tokens += model_call_data.get("completion_tokens", 0) or 0
        self.model_latency += model_call_data.get("time", 0) or 0

    def finalize_run_metrics(self):
        """Updates final metrics like end_time, total_time, and status."""
        if not self.context.collect_metrics_enabled:
            return

        self.end_time = time.time()
        self.status = str(self.context.session.task_status)
        self.iterations = self.context.session.iterations

        if self.context.tool_manager:
            self.cache_hits = self.context.tool_manager.cache_hits
            self.cache_misses = self.context.tool_manager.cache_misses

        self.agent_logger.debug("📊📈 Finalized run metrics.")

    def get_metrics(self) -> Dict[str, Any]:
        """Returns a comprehensive dictionary report of all metrics."""
        if not self.context.collect_metrics_enabled:
            return {}
        
        # Ensure dynamic properties are calculated for the report
        report = self.model_dump(exclude={"context"})
        report["total_time"] = self.total_time
        report["total_tokens"] = self.total_tokens
        report["cache_ratio"] = self.cache_ratio
        return report

