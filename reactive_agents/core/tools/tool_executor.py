"""Tool execution engine with comprehensive error handling."""

import time
import json
import traceback
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from reactive_agents.core.tools.abstractions import ToolResult, ToolProtocol
from reactive_agents.core.types.event_types import AgentStateEvent
from reactive_agents.core.reasoning.prompts.agent_prompts import (
    TOOL_ACTION_SUMMARY_PROMPT,
    TOOL_SUMMARY_CONTEXT_PROMPT,
)
from reactive_agents.utils.logging import Logger

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class ToolExecutor:
    """Handles the actual execution of tools with comprehensive error handling."""

    def __init__(self, context: "AgentContext"):
        self.context = context

    async def execute_tool(
        self, tool: ToolProtocol, tool_name: str, params: Dict[str, Any]
    ) -> Union[str, List[str], None]:
        """Execute a tool and handle all aspects of the execution."""
        try:
            tool_start_time = time.time()

            # Log execution start
            if hasattr(self.context, "tool_logger") and self.context.tool_logger:
                self.context.tool_logger.info(f"Using tool: {tool_name} with {params}")

            # Execute the tool
            result_obj = await tool.use(params)
            tool_execution_time = time.time() - tool_start_time

            # Ensure result is properly wrapped
            if not isinstance(result_obj, ToolResult):
                result_obj = ToolResult.wrap(result_obj)

            result_list = result_obj.to_list()
            result_str = str(result_list)

            # Log completion
            if hasattr(self.context, "tool_logger") and self.context.tool_logger:
                self.context.tool_logger.info(
                    f"Tool {tool_name} completed in {tool_execution_time:.2f}s"
                )
                if len(result_str) > 500:
                    self.context.tool_logger.debug(
                        f"Result Preview: {result_str[:500]}... (truncated)"
                    )
                else:
                    self.context.tool_logger.debug(f"Result: {result_str}")

            # Generate summary
            summary = await self._generate_tool_summary(tool_name, params, result_list)

            # Handle final answer tool specially
            if tool_name == "final_answer":
                self._handle_final_answer_tool(result_list)

            # Track successful tool usage
            if tool_name != "final_answer":
                self.context.session.successful_tools.add(tool_name)

            # Emit completion event
            self.context.emit_event(
                AgentStateEvent.TOOL_COMPLETED,
                {
                    "tool_name": tool_name,
                    "parameters": params,
                    "result": result_list,
                    "execution_time": tool_execution_time,
                },
            )

            # Emit final answer event if applicable
            if tool_name == "final_answer":
                self.context.emit_event(
                    AgentStateEvent.FINAL_ANSWER_SET,
                    {
                        "tool_name": tool_name,
                        "answer": result_list,
                        "parameters": params,
                    },
                )

            return result_list

        except Exception as e:
            return await self._handle_execution_error(tool_name, params, e)

    def _handle_final_answer_tool(self, result_list: Union[str, List[str]]) -> None:
        """Handle final answer tool execution."""
        if hasattr(self.context, "tool_logger") and self.context.tool_logger:
            self.context.tool_logger.info(
                "🔧 ToolExecutor: final_answer tool completed successfully"
            )

    async def _handle_execution_error(
        self, tool_name: str, params: Dict[str, Any], error: Exception
    ) -> str:
        """Handle tool execution errors with detailed logging."""
        tb_str = traceback.format_exc()
        error_message = f"Error using tool {tool_name}: {str(error)}"

        # Log error details
        if hasattr(self.context, "tool_logger") and self.context.tool_logger:
            self.context.tool_logger.error(error_message)
            self.context.tool_logger.debug(f"Traceback:\n{tb_str}")

        # Emit error event
        self.context.emit_event(
            AgentStateEvent.TOOL_FAILED,
            {
                "tool_name": tool_name,
                "parameters": params,
                "error": error_message,
                "traceback": tb_str,
            },
        )

        return error_message

    async def _generate_tool_summary(
        self, tool_name: str, params: Dict[str, Any], result: Any
    ) -> str:
        """Generate a summary of the tool action."""
        try:
            result_str = str(result)
            # Limit result string size for the prompt
            result_for_prompt = (
                result_str[:2000] + "..." if len(result_str) > 2000 else result_str
            )

            # Use the enhanced TOOL_SUMMARY_CONTEXT_PROMPT
            summary_context_prompt = TOOL_SUMMARY_CONTEXT_PROMPT.format(
                tool_name=tool_name,
                params=str(params),
                result_str=result_for_prompt,
            )

            # Get summary from model provider
            if hasattr(self.context, "model_provider") and self.context.model_provider:
                summary_result = await self.context.model_provider.get_completion(
                    system=TOOL_ACTION_SUMMARY_PROMPT,
                    prompt=summary_context_prompt,
                    options=getattr(self.context, "model_provider_options", {}),
                )
                tool_action_summary = (
                    summary_result.message.content.strip()
                    or f"Executed tool {tool_name}."
                )
            else:
                tool_action_summary = f"Executed tool {tool_name}."

            # Ensure tool summary is clearly marked
            if not tool_action_summary.startswith("[TOOL SUMMARY]"):
                tool_action_summary = f"[TOOL SUMMARY] {tool_action_summary}"
                self.context.session.add_message(
                    role="assistant",
                    content=tool_action_summary,
                )

            # Enhanced logging for debugging
            if hasattr(self.context, "tool_logger") and self.context.tool_logger:
                self.context.tool_logger.debug(
                    f"Tool Action Summary: {tool_action_summary}"
                )

            # Add summary to context logs
            self.context.session.reasoning_log.append(tool_action_summary)
            self.context.session.task_progress.append(tool_action_summary)

            return tool_action_summary

        except Exception as e:
            # Fallback summary on error
            if hasattr(self.context, "tool_logger") and self.context.tool_logger:
                self.context.tool_logger.error(
                    f"Failed to generate tool action summary for {tool_name}: {e}"
                )

            fallback_summary = f"Successfully executed tool '{tool_name}'."
            self.context.session.reasoning_log.append(fallback_summary)
            self.context.session.task_progress.append(fallback_summary)
            return fallback_summary

    def parse_tool_arguments(
        self, tool_call: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Parse tool call arguments with robust error handling."""
        tool_name = tool_call.get("function", {}).get("name")
        if not tool_name:
            raise ValueError("Tool call missing function name")

        # Parse arguments
        arguments_raw = tool_call.get("function", {}).get("arguments")
        if isinstance(arguments_raw, str):
            try:
                params = json.loads(arguments_raw)
            except json.JSONDecodeError as json_err:
                raise ValueError(
                    f"Invalid JSON arguments for tool {tool_name}: {json_err}. "
                    f"Arguments: '{arguments_raw}'"
                )
        elif isinstance(arguments_raw, dict):
            params = arguments_raw
        else:
            raise ValueError(
                f"Invalid argument type for tool {tool_name}: {type(arguments_raw)}. "
                "Expected dict or JSON string."
            )

        return tool_name, params

    def add_reasoning_to_context(self, params: Dict[str, Any]) -> None:
        """Add reasoning from tool parameters to the main context log."""
        if "reasoning" in params:
            self.context.session.reasoning_log.append(params["reasoning"])
