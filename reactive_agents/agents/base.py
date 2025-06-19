from __future__ import annotations
import traceback
from typing import List, Dict, Any, Optional, TYPE_CHECKING

# Import the AgentContext
from reactive_agents.context.agent_context import AgentContext

# Keep for type hinting if needed
from reactive_agents.loggers.base import Logger
from reactive_agents.model_providers.base import BaseModelProvider
import time  # Keep time for metric tracking

# Import shared types from the new location
from reactive_agents.common.types.status_types import TaskStatus
from reactive_agents.common.types.session_types import AgentSession

if TYPE_CHECKING:
    from reactive_agents.components.execution_engine import AgentExecutionEngine


class Agent:
    """
    Base class for AI agents, using AgentContext for state and component management.
    """

    context: AgentContext  # Agent now holds a context instance
    execution_engine: Optional["AgentExecutionEngine"] = (
        None  # Will be set by concrete implementations
    )

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

    def _extract_thinking_and_clean_content(
        self, content: str, call_context: str = "unknown"
    ) -> tuple[str, Optional[str]]:
        """
        Extract thinking content from <think> tags and clean the content.

        Args:
            content: The raw content from the model response
            call_context: The context of the call (e.g., "summary_generation", "reflection", etc.)

        Returns:
            tuple: (cleaned_content, thinking_content)
        """
        if not content:
            return "", None

        # Look for <think> tags
        think_start = content.find("<think>")
        think_end = content.find("</think>")

        if think_start != -1 and think_end != -1 and think_end > think_start:
            # Extract thinking content
            thinking_content = content[think_start + 7 : think_end].strip()

            # Remove thinking tags from content
            cleaned_content = content[:think_start] + content[think_end + 8 :]
            cleaned_content = cleaned_content.strip()

            return cleaned_content, thinking_content
        else:
            # No thinking tags found, return content as is
            return content.strip(), None

    def _store_thinking(self, thinking_content: str, call_context: str) -> None:
        """
        Store thinking content in the session with call context.

        Args:
            thinking_content: The extracted thinking content
            call_context: The context of the call
        """
        if thinking_content and hasattr(self.context, "session"):
            thinking_entry = {
                "timestamp": time.time(),
                "call_context": call_context,
                "thinking": thinking_content,
            }
            self.context.session.thinking_log.append(thinking_entry)
            self.agent_logger.debug(
                f"Stored thinking for {call_context}: {thinking_content[:100]}..."
            )

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

    def _should_use_tools_in_recursion(self, tool_calls: List[Dict[str, Any]]) -> bool:
        """
        Dynamically determine if the recursive think_chain should use tools.

        Args:
            tool_calls: The tool calls that were just processed

        Returns:
            bool: True if tools should be enabled in the recursive call, False otherwise
        """
        if not self.context.tool_manager:
            return False

        # Check if we have required tools and if they've been used
        if self.context.session.min_required_tools:
            successful_tools = set(self.context.session.successful_tools)
            required_tools = self.context.session.min_required_tools

            # If we've used all required tools, disable tools to force final response
            if required_tools.issubset(successful_tools):
                self.agent_logger.info(
                    "✅ All required tools used, disabling tools to force final response"
                )
                return False

        # Default: allow tools for proactive behavior
        self.agent_logger.info("🛠️ Enabling tools in recursion for proactive tool usage")
        return True

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
            if self.context.model_provider_options:
                self.agent_logger.debug(
                    f"Using model provider options: {self.context.model_provider_options}"
                )
                kwargs["options"] = self.context.model_provider_options

            result = await self.model_provider.get_chat_completion(
                tools=tool_signatures if use_tools else [],
                **kwargs,
            )
            result = result.model_dump()
            self.agent_logger.debug(f"Chat completion result: {result}")
            # Metric Tracking (always track call, even if it fails below)
            execution_time = time.time() - start_time
            if self.context.metrics_manager:
                self.context.metrics_manager.update_model_metrics(
                    {
                        "time": result.get("total_duration", execution_time),
                        "prompt_tokens": result.get("prompt_tokens", 0),
                        "completion_tokens": result.get("completion_tokens", 0),
                    }
                )

            if not result or "message" not in result.keys():
                self.agent_logger.warning(
                    "Chat completion did not return a valid message structure."
                )
                return None

            message = result["message"]
            message_content: str = message.get("content")
            tool_calls: List[Dict[str, Any]] = message.get("tool_calls")

            # Check if thinking is enabled and extract thinking content
            thinking_enabled = (
                self.context.model_provider_options.get("think", False)
                if self.context.model_provider_options
                else False
            )
            call_context = "think_chain"

            if thinking_enabled and message_content:
                cleaned_content, thinking_content = (
                    self._extract_thinking_and_clean_content(
                        message_content, call_context
                    )
                )
                if thinking_content:
                    self._store_thinking(thinking_content, call_context)
                message_content = cleaned_content
                # Update the message content with cleaned version
                message["content"] = cleaned_content

            if message_content.strip() and remember_messages:
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

                # Dynamically determine if tools should be enabled in the recursive call
                should_use_tools = self._should_use_tools_in_recursion(tool_calls)

                # Recursive call to let the model respond to tool results
                return await self._think_chain(
                    remember_messages=remember_messages,
                    use_tools=should_use_tools,
                    **kwargs,
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
        Processes tool calls using the execution engine to handle pause/stop/terminate states.
        Now runs tool calls in parallel for speed optimization.
        """
        import asyncio

        if not self.execution_engine:
            self.agent_logger.error(
                "Tool processing requested but execution engine is not available."
            )
            return

        processed_tool_call_ids = set()  # Track by ID if available
        tool_call_tasks = []
        tool_call_contexts = []  # To keep track of tool_call and its result

        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id")
            call_repr = str(tool_call)
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
            # Schedule tool call for parallel execution
            tool_call_tasks.append(self.execution_engine._execute_tool_call(tool_call))
            tool_call_contexts.append((tool_call, tool_call_id, call_repr))

        # Run all tool calls in parallel
        results = await asyncio.gather(*tool_call_tasks, return_exceptions=True)

        for idx, tool_result in enumerate(results):
            tool_call, tool_call_id, call_repr = tool_call_contexts[idx]
            if isinstance(tool_result, Exception):
                self.tool_logger.warning(
                    f"Tool call {tool_call.get('function', {}).get('name')} raised exception: {tool_result}"
                )
                continue
            if tool_result is not None:
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
