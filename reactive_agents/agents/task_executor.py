"""
Task execution module for reactive-ai-agent framework.
Handles the core task execution logic for agents.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Set, Tuple, TYPE_CHECKING
import asyncio
import json
import time
import traceback
from reactive_agents.common.types.status_types import TaskStatus
from reactive_agents.common.types.event_types import AgentStateEvent
from reactive_agents.loggers.base import Logger
from reactive_agents.agents.base import Agent

if TYPE_CHECKING:
    from reactive_agents.agents.execution_engine import AgentExecutionEngine


class TaskExecutor:
    """
    Handles the core task execution logic for agents.
    Manages the execution flow, reflection, and planning steps.
    """

    def __init__(self, agent: Agent):
        """Initialize the task executor with an agent reference."""
        self.agent = agent
        self.context = agent.context
        self.agent_logger = agent.agent_logger
        self.tool_logger = agent.tool_logger
        self.result_logger = agent.result_logger
        self.model_provider = agent.model_provider
        self._current_async_task: Optional[asyncio.Task] = None
        self._pending_tool_calls: List[Dict[str, Any]] = []
        self._tool_call_lock = asyncio.Lock()

    @property
    def execution_engine(self) -> "AgentExecutionEngine":
        """Get the execution engine from the agent."""
        if not self.agent.execution_engine:
            raise RuntimeError("Execution engine not initialized")
        return self.agent.execution_engine

    async def execute_iteration(self, task: str) -> Optional[Dict[str, Any]]:
        """
        Executes one iteration of the task execution loop.

        Args:
            task: The current task to execute

        Returns:
            Optional[Dict[str, Any]]: The result of the iteration
        """
        # --- 1. Think/Act Step ---
        self.agent_logger.info("🧠 Thinking/Acting...")
        think_act_result = await self._execute_think_act(task)

        # --- 2. Reflection Step ---
        reflection = await self._execute_reflection(think_act_result)

        # --- 3. Planning Step ---
        plan = await self._execute_planning(reflection)

        return {"think_act": think_act_result, "reflection": reflection, "plan": plan}

    async def _execute_think_act(self, task: str) -> Optional[Dict[str, Any]]:
        """Execute the think/act step of the iteration."""
        # Get tool signatures from context
        tool_signatures = self.context.get_tool_signatures()
        use_tools = self.context.tool_use_enabled and bool(tool_signatures)

        # Get chat completion from model provider
        result = await self.model_provider.get_chat_completion(
            tools=tool_signatures if use_tools else [],
            messages=self.context.session.messages,
            options=self.context.model_provider_options,
        )
        result = result.model_dump()

        if not result or "message" not in result:
            return None

        message = result["message"]
        content = message.get("content")

        if content:
            self.context.session.reasoning_log.append(
                f"Assistant Thought/Response: {content[:200]}..."
            )
            # Add assistant response to context's message history
            self.context.session.messages.append(
                {"role": "assistant", "content": content}
            )
        if message.get("tool_calls"):
            # --- DEBUG LOGGING ---
            self.agent_logger.debug(f"Tool calls: {len(message.get('tool_calls', []))}")
            self.context.session.reasoning_log.append(
                f"Assistant Action: Called tools."
            )
            # Add the assistant message with tool calls
            self.context.session.messages.append(message)

            # --- BATCH TOOL CALL EXECUTION ---
            for tool_call in message.get("tool_calls", []):
                self.context.emit_event(
                    AgentStateEvent.TOOL_CALLED,
                    {
                        "tool_name": tool_call.get("function", {}).get(
                            "name", "unknown"
                        ),
                        "tool_id": tool_call.get("id", "unknown"),
                        "parameters": tool_call.get("function", {}).get(
                            "parameters", {}
                        ),
                    },
                )
                # Execute the tool call through the execution engine (pause/stop/terminate respected)
                tool_result = await self.execution_engine._execute_tool_call(tool_call)
                if tool_result is not None:
                    # Add tool result to messages
                    self.context.session.messages.append(
                        {
                            "role": "tool",
                            "name": tool_call.get("function", {}).get(
                                "name", "unknown"
                            ),
                            "content": str(tool_result),
                            **(
                                {"tool_call_id": tool_call.get("id")}
                                if tool_call.get("id")
                                else {}
                            ),
                        }
                    )

        return result

    async def _execute_reflection(
        self, think_act_result: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Execute the reflection step of the iteration."""
        if not self.context.reflect_enabled or not self.context.reflection_manager:
            return None

        reflection_input = think_act_result
        if isinstance(reflection_input, str):
            reflection_input = {"message": {"content": reflection_input}}

        reflection = await self.context.reflection_manager.generate_reflection(
            reflection_input
        )

        if reflection:
            self.context.session.reasoning_log.append(
                f"Reflection: Reason={reflection['reason']}"
            )

            # Emit reflection generated event
            self.context.emit_event(
                AgentStateEvent.REFLECTION_GENERATED,
                {
                    "reason": reflection["reason"],
                    "next_step": reflection.get("next_step", "None"),
                    "required_tools": reflection.get("required_tools", []),
                },
            )

            # Make next_step more prominent in the context
            if reflection.get("next_step"):
                next_step = reflection["next_step"]
                self.context.session.reasoning_log.append(
                    f"REFLECTION NEXT STEP: {next_step}"
                )
                # Prune old system messages, keep only the first (original) and the latest next_step
                messages = self.context.session.messages
                if messages:
                    # Keep only the first system message (original prompt)
                    pruned = [messages[0]] if messages[0]["role"] == "system" else []
                    # Remove all other system messages
                    pruned += [m for m in messages[1:] if m["role"] != "system"]
                    self.context.session.messages = pruned
                # Add next_step as the last system message
                self.context.session.messages.append(
                    {
                        "role": "system",
                        "content": f"CRITICAL: You MUST execute this exact next step: {next_step}",
                    }
                )
                # Add a strong user nudge right before the model's turn
                self.context.session.messages.append(
                    {
                        "role": "user",
                        "content": f"REMINDER: The next_step is: {next_step}. You must execute this now.",
                    }
                )
                # Update system prompt to emphasize next_step
                self.context.update_system_prompt()

        return reflection

    async def _execute_planning(
        self, reflection: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Execute the planning step of the iteration."""
        if not reflection or not reflection.get("next_step"):
            return None

        next_step = reflection.get("next_step")
        required_tools = reflection.get("required_tools", [])

        # Update session with required tools if provided
        if required_tools:
            self.context.session.min_required_tools = set(required_tools)

        return next_step

    def should_continue(self) -> bool:
        """Determine if the execution loop should continue."""
        # Check terminal statuses
        if self.context.session.task_status in [
            TaskStatus.COMPLETE,
            TaskStatus.ERROR,
            TaskStatus.CANCELLED,
            TaskStatus.RESCOPED_COMPLETE,
            TaskStatus.MAX_ITERATIONS,
            TaskStatus.MISSING_TOOLS,
        ]:
            self.agent_logger.debug(
                f"Stopping loop: Terminal status {self.context.session.task_status}."
            )
            return False

        # Check max iterations
        if (
            self.context.max_iterations is not None
            and self.context.session.iterations >= self.context.max_iterations
        ):
            self.agent_logger.info(
                f"Stopping loop: Max iterations ({self.context.max_iterations}) reached."
            )
            self.context.session.task_status = TaskStatus.MAX_ITERATIONS
            return False

        # Check dependencies
        if not self._check_dependencies():
            self.agent_logger.info("Stopping loop: Dependencies not met.")
            return False

        # Check required tools completion
        if (
            self.context.session.min_required_tools is not None
            and len(self.context.session.min_required_tools) > 0
        ):
            tools_completed = all(
                tool in self.context.session.successful_tools
                for tool in self.context.session.min_required_tools
            )
            if tools_completed:
                self.agent_logger.info("All required tools completed successfully.")
                return False

        return True

    def _check_dependencies(self) -> bool:
        """Check if all dependencies are met."""
        if self.context.workflow_manager:
            return self.context.workflow_manager.check_dependencies()
        self.agent_logger.debug("No workflow manager, skipping dependency check.")
        return True

    async def execute_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        """
        Execute a single tool call.

        Args:
            tool_call: Dictionary containing tool call information including name and parameters

        Returns:
            The result of the tool execution
        """
        if not self.context.tool_manager:
            raise RuntimeError("Tool manager not initialized")

        tool_name = tool_call.get("function", {}).get("name")
        parameters = tool_call.get("function", {}).get("parameters", {})

        if not tool_name:
            raise ValueError("Tool call missing name")

        return await self.context.tool_manager.use_tool(tool_call)
