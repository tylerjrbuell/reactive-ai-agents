from __future__ import annotations
import json
import traceback
import asyncio
from typing import List, Dict, Any, Optional, Union, Sequence, Callable, TYPE_CHECKING

# Removed Logger, BaseModelProvider, MCPClient, ToolProtocol, ToolResult, MCPToolWrapper etc.
# Removed ModelProviderFactory, prompts

# Import the AgentContext
from context.agent_context import AgentContext
from tools.abstractions import (
    ToolResult,
)  # Keep ToolResult potentially for type hints if needed later

# Keep for type hinting if needed
from loggers.base import Logger
from model_providers.base import BaseModelProvider
import time  # Keep time for metric tracking

# Removed time import if not used directly

# Import shared types from the new location
from common.types import TaskStatus


class Agent:
    """
    Base class for AI agents, using AgentContext for state and component management.
    """

    context: AgentContext  # Agent now holds a context instance

    def __init__(self, context: AgentContext):
        """
        Initializes the Agent with a pre-configured AgentContext.

        Args:
            context: The AgentContext instance holding configuration, state, and managers.
        """
        self.context = context
        # Ensure logger is initialized before using it
        if not self.context.agent_logger:
            self.context._initialize_loggers()  # Call initialization if needed
        assert self.context.agent_logger is not None  # Assertion for type checker
        self.agent_logger.info(
            f"Base Agent '{self.context.agent_name}' initialized with context."
        )

    # --- Convenience properties to access context components ---
    @property
    def agent_logger(self) -> Logger:
        # Logger should be initialized by context or __init__
        assert self.context.agent_logger is not None
        return self.context.agent_logger

    @property
    def tool_logger(self) -> Logger:
        assert self.context.tool_logger is not None
        return self.context.tool_logger

    @property
    def result_logger(self) -> Logger:
        assert self.context.result_logger is not None
        return self.context.result_logger

    @property
    def model_provider(self) -> BaseModelProvider:
        assert self.context.model_provider is not None
        return self.context.model_provider

    # --- End Convenience properties ---

    async def _think(self, **kwargs) -> dict | None:
        """Directly calls the model provider for a simple completion."""
        start_time = time.time()
        try:
            self.agent_logger.info("Thinking (direct completion)...")
            # Pass messages directly if provided, otherwise use context messages
            kwargs.setdefault("messages", self.context.session.messages)
            result = await self.model_provider.get_completion(**kwargs)

            # Metric Tracking
            execution_time = time.time() - start_time
            if self.context.metrics_manager:
                usage = result.get("usage", {}) if result else {}
                self.context.metrics_manager.update_model_metrics(
                    {
                        "time": execution_time,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    }
                )
            return result
        except Exception as e:
            self.agent_logger.error(f"Direct Completion Error: {e}")
            # Track error? Maybe implicitly tracked by lack of successful result.
            return None

    async def _think_chain(
        self,
        remember_messages: bool = True,
        use_tools: bool = True,
        **kwargs,
    ) -> dict | None:
        """
        Performs a chat completion, potentially using tools and managing message history via context.
        """
        start_time = time.time()
        try:
            # Use tool signatures from the ToolManager via context
            tool_signatures = self.context.get_tool_signatures()
            use_tools = (
                self.context.tool_use_enabled and bool(tool_signatures) and use_tools
            )

            # Default to context messages
            kwargs.setdefault("messages", self.context.session.messages)
            self.agent_logger.debug(
                f"Thinking (chat completion, tool_use={use_tools})..."
            )

            result = await self.model_provider.get_chat_completion(
                tools=tool_signatures if use_tools else [],
                **kwargs,
            )

            # Metric Tracking (always track call, even if it fails below)
            execution_time = time.time() - start_time
            if self.context.metrics_manager:
                usage = result.get("usage", {}) if result else {}
                self.context.metrics_manager.update_model_metrics(
                    {
                        "time": execution_time,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    }
                )

            if not result or "message" not in result:
                self.agent_logger.warning(
                    "Chat completion did not return a valid message structure."
                )
                return None

            message = result["message"]
            message_content = message.get("content")
            tool_calls = message.get("tool_calls")

            if message_content and remember_messages:
                # Add assistant response to context's message history
                self.context.session.messages.append(
                    {"role": "assistant", "content": message_content}
                )
                self.agent_logger.debug(
                    f"Added assistant message to context: {message_content[:100]}..."
                )

            elif tool_calls and use_tools:
                self.agent_logger.info(f"Received {len(tool_calls)} tool calls.")
                # Add the assistant message with tool calls before processing results
                if remember_messages:
                    self.context.session.messages.append(
                        message
                    )  # Store the message with tool_calls
                    self.agent_logger.debug(
                        "Added assistant tool call message to context."
                    )

                # Process tools using ToolManager via context
                await self._process_tool_calls(tool_calls=tool_calls)

                # Recursive call to let the model respond to tool results
                # Pass remember_messages=False for the recursive call if we only want the *final* assistant message
                return await self._think_chain(
                    remember_messages=remember_messages, use_tools=False, **kwargs
                )

            # Return the result which might contain content or tool_calls
            return result

        except Exception as e:
            # Capture detailed traceback
            tb_str = traceback.format_exc()
            self.agent_logger.error(f"Chat Completion Error: {e}\n{tb_str}")
            return None

    async def _process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """
        Processes tool calls using the ToolManager from the context and appends results to context messages.
        """
        if not self.context.tool_manager:
            self.agent_logger.error(
                "Tool processing requested but ToolManager is not available in context."
            )
            return

        # Could potentially run in parallel if ToolManager supports it and tools are independent
        # For now, process sequentially
        processed_tool_call_ids = set()  # Track by ID if available

        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id")
            # Simple check to avoid reprocessing identical calls in the same batch (if IDs aren't present)
            # A more robust check might involve hashing the call details
            call_repr = str(tool_call)  # Simple representation for de-duplication
            if tool_call_id and tool_call_id in processed_tool_call_ids:
                self.tool_logger.debug(
                    f"Tool call ID {tool_call_id} already processed in this batch."
                )
                continue
            elif not tool_call_id and call_repr in processed_tool_call_ids:
                self.tool_logger.debug(
                    f"Duplicate tool call signature processed in this batch: {tool_call.get('function', {}).get('name')}"
                )
                continue

            # Use the ToolManager to execute the tool
            tool_result = await self.context.tool_manager.use_tool(tool_call=tool_call)

            # Append result to context message list
            if tool_result is not None:
                # Result should be formatted as string for the model
                result_content = str(tool_result)

                tool_result_message = {
                    "role": "tool",
                    "name": tool_call.get("function", {}).get("name", "unknown_tool"),
                    "content": result_content,
                    **({"tool_call_id": tool_call_id} if tool_call_id else {}),
                }
                self.context.session.messages.append(tool_result_message)
                self.agent_logger.debug(
                    f"Added tool result message to context for {tool_result_message['name']}."
                )

                if tool_call_id:
                    processed_tool_call_ids.add(tool_call_id)
                else:
                    processed_tool_call_ids.add(call_repr)
            else:
                self.tool_logger.warning(
                    f"Tool call {tool_call.get('function', {}).get('name')} produced None result. Not adding to messages."
                )

    async def _run_task(self, task: str) -> dict | None:
        """Adds the main task to the message list and initiates the thinking chain."""
        # Ensure initial task is set in context if not already
        if not self.context.session.initial_task:
            self.context.session.initial_task = task
        if not self.context.session.current_task:
            self.context.session.current_task = task

        # Append the user's task message
        task_nudges_joined = "\n".join(self.context.session.task_nudges)
        self.context.session.messages.append(
            {
                "role": "user",
                "content": f"""
                {task}
                {task_nudges_joined}
                """,
            }
        )
        # Start the thinking process
        result = await self._think_chain(
            remember_messages=True
        )  # Remember intermediate steps by default
        return result

    async def _run_task_iteration(self, task: str) -> Optional[str]:
        """
        Runs a single iteration or loop for the task, potentially simplified in base Agent.
        ReactAgent will override this with more complex logic (reflection, planning).
        """
        self.context.session.iterations = 0
        max_iterations = (
            self.context.max_iterations or 1
        )  # Default to 1 iteration for base agent

        final_result_content = None

        try:
            self.agent_logger.info(f"Starting task: {task}")
            while self.context.session.iterations < max_iterations:
                self.context.session.iterations += 1
                self.agent_logger.info(
                    f"Running Iteration: {self.context.session.iterations}/{max_iterations}"
                )

                # In base agent, just run the task directly
                # Use current_task from context, which might be updated by subclasses
                result = await self._run_task(task=self.context.session.current_task)

                if not result or not result.get("message"):
                    self.agent_logger.warning(
                        f"Iteration {self.context.session.iterations} failed or produced no result."
                    )
                    break  # Stop if an iteration fails

                message = result["message"]
                final_result_content = message.get(
                    "content"
                )  # Store the last content message

                # Base agent doesn't handle complex stopping conditions like reflections or final_answer tool
                # It just runs for the allowed iterations. Subclasses override this.
                if (
                    self.context.session.final_answer
                ):  # Check if a tool set the final answer
                    self.agent_logger.info("Final answer detected in context.")
                    final_result_content = self.context.session.final_answer
                    break

                # Simple check for no progress (if assistant repeats itself) - maybe less relevant in base agent
                if (
                    self.context.session.iterations > 1
                    and len(self.context.session.messages) > 2
                ):
                    last_assistant_message = (
                        self.context.session.messages[-1]
                        if self.context.session.messages[-1]["role"] == "assistant"
                        else None
                    )
                    prev_assistant_message = (
                        self.context.session.messages[-3]
                        if len(self.context.session.messages) > 3
                        and self.context.session.messages[-3]["role"] == "assistant"
                        else None
                    )
                    if (
                        last_assistant_message
                        and prev_assistant_message
                        and last_assistant_message.get("content")
                        == prev_assistant_message.get("content")
                    ):
                        self.agent_logger.info(
                            "Assistant message repeated, stopping iteration."
                        )
                        break

            if self.context.session.iterations >= max_iterations:
                self.agent_logger.info(
                    f"Reached maximum iterations ({max_iterations}) for base agent run."
                )

            return (
                self.context.session.final_answer or final_result_content
            )  # Return final answer if set, else last content

        except Exception as e:
            tb_str = traceback.format_exc()
            self.agent_logger.error(f"Error during task iteration: {e}\n{tb_str}")
            return None  # Indicate error

    async def run(self, initial_task: str) -> Optional[str]:
        """
        Basic run method for the agent. Sets the initial task and runs the iteration loop.
        Subclasses (like ReactAgent) should override this for more complex execution flows.
        """
        # Reset context state for a new run? Or assume context is fresh?
        # Let's assume context might be reused, so reset relevant parts.
        self.context.session.initial_task = initial_task
        self.context.session.current_task = initial_task
        self.context.session.iterations = 0
        self.context.session.final_answer = None
        # Clear messages except system prompt? Be careful if context is shared.
        # For now, assume messages are managed externally or cleared before run if needed.
        # self.context.messages = [msg for msg in self.context.messages if msg['role'] == 'system']

        self.agent_logger.info(
            f"Agent '{self.context.agent_name}' starting run for task: {initial_task}"
        )

        # Reset metrics if manager exists
        if self.context.metrics_manager:
            self.context.metrics_manager.reset()

        final_result = None
        try:
            # The base agent runs a simple iteration loop.
            final_result = await self._run_task_iteration(task=initial_task)

            if final_result:
                self.result_logger.info(
                    f"Task completed. Result: {str(final_result)[:500]}..."
                )
            else:
                self.agent_logger.warning(
                    "Task completed without a final string result."
                )

            # Update final metrics
            if self.context.metrics_manager:
                # Assume status is COMPLETE if we finished without error in base agent
                if self.context.session.task_status not in [
                    TaskStatus.ERROR
                ]:  # Use TaskStatus from common
                    self.context.session.task_status = (
                        TaskStatus.COMPLETE
                    )  # Use TaskStatus from common
                self.context.metrics_manager.finalize_run_metrics()

            # Save memory if enabled
            if self.context.memory_manager:
                # Need to construct a result dict for memory manager
                self.context.memory_manager.update_session_history(self.context.session)
                self.context.memory_manager.save_memory()

            return final_result

        except Exception as e:
            self.agent_logger.error(f"Agent run failed: {e}")
            self.context.session.task_status = (
                TaskStatus.ERROR
            )  # Use TaskStatus from common
            if self.context.metrics_manager:
                self.context.metrics_manager.finalize_run_metrics()
            return None
        # finally:
        # Context closure should be handled by the caller who created the context
        # await self.context.close() # Don't close context here

    # --- Methods removed as logic moved to Context/Managers ---
    # _use_tool -> Now handled by ToolManager
    # _safe_close_mcp_client -> Now handled by AgentContext.close
    # ask_user_confirmation -> Now handled by ToolManager
    # _update_metrics -> Now handled by MetricsManager
    # set_model_provider -> Configuration should happen at context creation
    # _plan -> Specific logic, likely belongs in ReactAgent or similar
