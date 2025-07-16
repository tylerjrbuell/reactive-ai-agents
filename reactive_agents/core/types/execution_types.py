from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from reactive_agents.core.types.status_types import TaskStatus
from reactive_agents.core.types.session_types import AgentSession

class ExecutionResult(BaseModel):
    """A structured, self-contained result of a full agent execution run."""

    session: AgentSession
    status: TaskStatus = Field(description="The final status of the task.")
    final_answer: Optional[str] = Field(description="The final answer, if one was produced.")
    strategy_used: str = Field(description="The primary strategy used for the execution.")
    execution_details: Dict[str, Any] = Field(description="Raw details from the execution loop.")
    task_metrics: Dict[str, Any] = Field(default_factory=dict, description="Metrics collected during the run.")

    def was_successful(self) -> bool:
        """Returns True if the task completed successfully."""
        return self.status == TaskStatus.COMPLETE and not self.session.has_failed

    def to_pretty_string(self) -> str:
        """Generates a beautiful, human-readable summary of the execution result."""
        header = f"🚀 Execution Result: {self.status.value} 🚀"
        divider = "=" * len(header)
        
        summary = (
            f"{header}\n{divider}\n"
            f"🔹 Session ID: {self.session.session_id}\n"
            f"🔹 Strategy: {self.strategy_used}\n"
            f"🔹 Duration: {self.session.duration:.2f}s\n"
            f"🔹 Iterations: {self.session.iterations}\n"
            f"🔹 Final Score: {self.session.overall_score:.2f}\n"
        )

        if self.final_answer:
            summary += f"\n📝 Final Answer:\n---\n{self.final_answer}\n---\n"
        
        if self.session.errors:
            summary += f"\n❗ Errors ({len(self.session.errors)}):\n"
            for error in self.session.errors:
                summary += f"  - [{error.get('source', 'Unknown')}]: {error.get('details', {}).get('error', 'N/A')}\n"

        return summary

    def to_json(self, **kwargs) -> str:
        """Serializes the result to a JSON string."""
        return self.model_dump_json(indent=2, **kwargs)
