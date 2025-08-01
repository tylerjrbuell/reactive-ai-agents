"""Final answer tool implementation."""

from typing import Any, Dict, TYPE_CHECKING
from reactive_agents.core.tools.base import Tool
from reactive_agents.core.tools.abstractions import ToolResult

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class FinalAnswerTool(Tool):
    """Wrapper class to make the final_answer function ToolProtocol compatible."""

    name = "final_answer"
    description = (
        "Provides the final answer to the user's query and concludes the task."
    )
    tool_definition = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final textual answer to the user's query as a complete response to the original task.",
                    }
                },
                "required": ["answer"],
            },
        },
    }

    def __init__(self, context: "AgentContext"):
        self.context = context

    async def use(self, params: Dict[str, Any]) -> ToolResult:
        """Executes the final answer tool with the provided parameters."""
        answer = params.get("answer")
        if answer is None:
            return ToolResult("Error: Missing required parameter 'answer'.")

        # Set the final answer in the context
        self.context.session.final_answer = answer

        # Log final answer setting
        if hasattr(self.context, "agent_logger") and self.context.agent_logger:
            self.context.agent_logger.info(
                f"🔧 FinalAnswerTool: Set session.final_answer = {answer[:50] if answer else 'None'}..."
            )

        return ToolResult(answer)