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
                execution_time = time.time() - start_time
                if self.context.metrics_manager:
                    self.context.metrics_manager.update_model_metrics(
                        {
                            "time": result.total_duration or execution_time,
                            "prompt_tokens": result.prompt_tokens or 0,
                            "completion_tokens": result.completion_tokens or 0,
                        }
                    )
            return result.model_dump()
        except Exception as e:
            self.agent_logger.error(f"Direct Completion Error: {e}")
            return None

    def _should_use_tools(self, tool_calls: List[Dict[str, Any]]) -> bool:
        """
        Decide whether to allow tool use in the next think_chain call, based on context.tool_use_policy and plan progress.
        - If the final answer has been set, do not allow tool use.
        - If the plan is complete (all steps completed), do not allow tool use.
        - Otherwise, use the existing tool_use_policy logic.
        """
        # --- Final answer check: if final answer is set, do not allow tool use ---
        if self.context.session.final_answer is not None:
            self.agent_logger.debug("Final answer already set, not allowing tool use.")
            return False

        # --- Step-based plan completion check: if plan is complete, do not allow tool use ---
        if self.context.session.is_plan_complete():
            self.agent_logger.debug("Plan is complete, not allowing tool use.")
            return False

        # --- Existing tool_use_policy logic ---
        policy = getattr(self.context, "tool_use_policy", "adaptive")
        session = self.context.session

        if policy == "always":
            return True
        if policy == "never":
            return False
        if policy == "required_only":
            if session.min_required_tools:
                tools_completed, _ = self.context.has_completed_required_tools()
                return not tools_completed
            return False

        # Adaptive/heuristic
        # 1. If required tools left, use tools
        if session.min_required_tools:
            tools_completed, _ = self.context.has_completed_required_tools()
            if not tools_completed:
                return True
        # 2. If just used a tool, consider pausing for reflection
        if tool_calls and tool_calls[-1].get("name") in session.successful_tools:
            return False
        # 3. If too many tool calls in a row, force reflection
        max_calls = getattr(self.context, "tool_use_max_consecutive_calls", 3)
        if len(tool_calls) > max_calls:
            return False
        # 4. Otherwise, allow tool use
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
                tool_use_required=use_tools,
                **kwargs,
            )
            self.agent_logger.debug(f"Chat completion result: {result}")
            # Metric Tracking (always track call, even if it fails below)
            execution_time = time.time() - start_time
            if self.context.metrics_manager:
                self.context.metrics_manager.update_model_metrics(
                    {
                        "time": result.total_duration or execution_time,
                        "prompt_tokens": result.prompt_tokens or 0,
                        "completion_tokens": result.completion_tokens or 0,
                    }
                )
            if not result or not result.message:
                self.agent_logger.warning(
                    "Chat completion did not return a valid message structure."
                )
                return None
            message = result.message
            message_content: str = message.content or ""
            tool_calls: List[Dict[str, Any]] = message.tool_calls or []
            if message_content.strip() and remember_messages:
                # Add assistant response to context's message history
                self.context.session.messages.append(
                    {"role": "assistant", "content": message_content}
                )
                self.agent_logger.debug(
                    f"Added assistant message to context: {message_content[:100]}..."
                )
            if tool_calls and use_tools:
                self.agent_logger.info(f"Received {len(tool_calls)} tool calls.")
                # Add the assistant message with tool calls before processing results
                if remember_messages:
                    self.context.session.messages.append(
                        message.model_dump()
                    )  # Store the message with tool_calls
                    self.agent_logger.debug(
                        "Added assistant tool call message to context."
                    )

                # Process tools using ToolManager via context
                await self._process_tool_calls(tool_calls=tool_calls)

                # Check if final answer was set during tool processing
                if self.context.session.final_answer is not None:
                    self.agent_logger.info(
                        "Final answer set during tool processing, stopping tool chain."
                    )
                    return result.model_dump()

                # Dynamically determine if tools should be enabled in the recursive call
                should_use_tools = self._should_use_tools(tool_calls)

                # Recursive call to let the model respond to tool results
                return await self._think_chain(
                    remember_messages=remember_messages,
                    use_tools=False,
                    **kwargs,
                )
            return result.model_dump()
        except Exception as e:
            tb_str = traceback.format_exc()
            self.agent_logger.error(f"Chat Completion Error: {e}\n{tb_str}")
            return None

    async def _process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> list | None:
        """
        Processes tool calls using the execution engine to handle pause/stop/terminate states.
        Now runs tool calls in parallel for speed optimization.
        Stops processing immediately if final_answer tool is called.
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
                tool_name = tool_call.get("function", {}).get("name", "unknown_tool")
                tool_result_message = {
                    "role": "tool",
                    "name": tool_name,
                    "content": result_content,
                    **({"tool_call_id": tool_call_id} if tool_call_id else {}),
                }
                self.context.session.messages.append(tool_result_message)
                self.agent_logger.debug(
                    f"Added tool result message to context for {tool_result_message['name']}."
                )

                # If this was a final_answer tool
                if tool_name == "final_answer":
                    self.agent_logger.info(
                        f"✅ Final answer set: {result_content[:100]}..."
                    )

                if tool_call_id:
                    processed_tool_call_ids.add(tool_call_id)
                else:
                    processed_tool_call_ids.add(call_repr)
            else:
                self.tool_logger.warning(
                    f"Tool call {tool_call.get('function', {}).get('name')} produced None result. Not adding to messages."
                )
        return results

    async def _run_task(self, task: str) -> dict | None:
        """Adds the main task to the message list and initiates the thinking chain."""
        # Ensure initial task is set in context if not already
        if not self.context.session.initial_task:
            self.context.session.initial_task = task
        if not self.context.session.current_task:
            self.context.session.current_task = task

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
