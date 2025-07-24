from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import json
import textwrap

from reactive_agents.core.types.status_types import TaskStatus
from reactive_agents.core.types.session_types import AgentSession

if TYPE_CHECKING:
    from reactive_agents.core.reasoning.engine import ReasoningEngine


class ExecutionResult(BaseModel):
    """A structured, self-contained result of a full agent execution run."""

    session: AgentSession
    status: TaskStatus = Field(description="The final status of the task.")
    final_answer: Optional[str] = Field(
        description="The final answer, if one was produced."
    )
    summary: Optional[str] = Field(
        default="", description="The summary of the execution result."
    )
    strategy_used: str = Field(
        description="The primary strategy used for the execution."
    )
    execution_details: Dict[str, Any] = Field(
        description="Raw details from the execution loop."
    )
    task_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Metrics collected during the run."
    )

    def was_successful(self) -> bool:
        """Returns True if the task completed successfully."""
        return self.status == TaskStatus.COMPLETE and not self.session.has_failed

    async def generate_summary(self, engine: "ReasoningEngine") -> str:
        """Generates a narrative summary of the execution result using a reasoning engine."""
        try:
            summary_prompt = engine.get_prompt("execution_result_summary")
            result = await summary_prompt.get_completion(execution_result=self)

            if result and result.content:
                self.summary = result.content
                return result.content
            return "Could not generate a summary."
        except Exception as e:
            return f"Failed to generate summary: {e}"

    def to_prompt_string(self) -> str:
        """Generates a string for the prompt."""
        thinking_log_summary = "\n".join(
            [json.dumps(thought) for thought in self.session.thinking_log[-5:]]
        )
        task_metrics_summary = json.dumps(self.task_metrics, indent=2)
        error_log_summary = "\n".join(
            [json.dumps(error) for error in self.session.errors[-5:]]
        )

        return f"""
        As the agent named '{self.session.agent_name}', provide a concise, narrative summary of your performance and actions based on the following execution data.
        The summary should answer the questions: "What did I do?" and "How did I perform?".
        Focus on the key outcomes, decisions, and any notable events like errors or strategy choices.

        **Execution Data:**
        - **Final Status:** {self.status.value}
        - **Strategy Used:** {self.strategy_used}
        - **Successful:** {'Yes' if self.was_successful() else 'No'}
        - **Duration:** {self.session.duration:.2f} seconds
        - **Iterations:** {self.session.iterations}
        - **Final Answer:** {'Provided' if self.final_answer else 'Not provided'}

        **Key Metrics:**
        {task_metrics_summary}

        **Recent Thoughts (Last 5):**
        {thinking_log_summary}

        **Errors Encountered (Last 5):**
        {error_log_summary}
    """

    def to_pretty_string(self, **kwargs) -> str:
        """Generates a beautiful, human-readable summary of the execution result."""
        header = f"🚀 Execution Result: {self.status.value} 🚀"
        divider = "=" * len(header)

        summary = (
            f"{header}\n{divider}\n"
            f"🔹 Session ID: {self.session.session_id}\n"
            f"🔹 Strategy: {self.strategy_used}\n"
            f"🔹 Duration: {self.session.duration:.2f}s\n"
            f"🔹 Successful: {'Yes' if self.was_successful() else 'No'}\n"
            f"🔹 Iterations: {self.session.iterations}\n"
            f"🔹 Final Score: {self.session.overall_score:.2f}\n"
            f"🔹 Summary: {self.summary}\n"
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
