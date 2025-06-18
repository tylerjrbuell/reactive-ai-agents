"""
Task execution module for reactive-ai-agent framework.
Handles the core task execution logic for agents.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import asyncio
import time
from reactive_agents.common.types.status_types import TaskStatus
from reactive_agents.agents.base import Agent

if TYPE_CHECKING:
    from reactive_agents.components.execution_engine import AgentExecutionEngine


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
        self.agent_logger.info("🧠 Phase 1: Think & Act")
        think_act_result = await self._execute_think_act(task)

        # --- 2. Reflection Step ---
        self.agent_logger.info("🤔 Phase 2: Reflection & Analysis")
        reflection = await self._execute_reflection(think_act_result)

        # --- 3. Planning Step ---
        self.agent_logger.info("📋 Phase 3: Planning Next Steps")
        plan = await self._execute_planning(reflection)

        self.agent_logger.info("✅ Iteration Complete")

        return {"think_act": think_act_result, "reflection": reflection, "plan": plan}

    async def _execute_think_act(self, task: str) -> Optional[Dict[str, Any]]:
        """Execute the think/act step with dynamic context."""
        # Get current next step from session or reflection manager
        current_next_step = self._get_current_next_step()
        # Update system prompt with dynamic context
        self.context.update_system_prompt(current_next_step)
        # Prune context if needed
        self.context.prune_context_if_needed()
        # Add task message (avoid duplication)
        task = current_next_step or task
        if not self._has_recent_task_message(task):
            self.context.session.messages.append(
                {
                    "role": "user",
                    "content": f"Execute the current task: {task}",
                }
            )
        self.agent_logger.info(f"🧠 Executing Task Step: {task}")
        # Execute thinking with updated context
        result = await self.agent._think_chain(
            remember_messages=True,
            use_tools=self.context.tool_use_enabled,
        )
        if result and "message" in result:
            message = result["message"]
            content = message.get("content")
            if content:
                self.agent_logger.info("🧠 Think/act phase completed with response")
                self.context.session.reasoning_log.append(
                    f"Assistant Response: {content[:200]}..."
                )
            else:
                self.agent_logger.info(
                    "🧠 Think/act phase completed but no content generated"
                )
        else:
            self.agent_logger.warning("🧠 Think/act phase failed or returned no result")
        return result

    def _get_current_next_step(self) -> Optional[str]:
        """Get the current next step from various sources."""
        # First check session storage
        if self.context.session.current_next_step:
            return self.context.session.current_next_step
        # Check reflection manager for latest reflection
        if self.context.reflection_manager:
            last_reflection = self.context.reflection_manager.get_last_reflection()
            if last_reflection and last_reflection.get("next_step"):
                return last_reflection["next_step"]
        return None

    def _has_recent_task_message(self, task: str) -> bool:
        """Check if we already have a recent task message to avoid duplication."""
        if not self.context.session.messages:
            return False
        recent_messages = self.context.session.messages[-3:]
        for message in recent_messages:
            if message.get("role") == "user" and task in message.get("content", ""):
                return True
        return False

    async def _execute_reflection(
        self, think_act_result: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Execute the reflection step of the iteration."""
        if not self.context.reflect_enabled or not self.context.reflection_manager:
            self.agent_logger.info(
                "🤔 Reflection disabled or no reflection manager available"
            )
            return None
        self.agent_logger.info("🤔 Generating reflection on current state...")
        reflection_input = think_act_result
        if isinstance(reflection_input, str):
            reflection_input = {"message": {"content": reflection_input}}
        reflection = await self.context.reflection_manager.generate_reflection(
            reflection_input
        )
        if reflection and reflection.get("next_step"):
            next_step = reflection["next_step"]
            # Store next step in session with source
            self.context.session.current_next_step = next_step
            self.context.session.next_step_source = "reflection"
            self.context.session.next_step_timestamp = time.time()
            # Update system prompt with new next step
            self.context.update_system_prompt(next_step)
            self.agent_logger.info(f"🎯 Next Step Set: {next_step}")
        elif reflection:
            self.agent_logger.info("🤔 No next_step in reflection")
        else:
            self.agent_logger.info("🤔 No reflection generated")
        return reflection

    async def _execute_planning(
        self, reflection: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Execute the planning step of the iteration. Ensure final_answer tool is required."""
        if not reflection or not reflection.get("next_step"):
            return None
        next_step = reflection.get("next_step")
        required_tools = reflection.get("required_tools", [])
        # Ensure final_answer is always required
        if "final_answer" not in required_tools:
            required_tools.append("final_answer")
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

        # Check required tools completion with consistent logic from ReactAgent
        tools_completed = False
        deterministic_score = 0.0

        if (
            self.context.session.min_required_tools is not None
            and len(self.context.session.min_required_tools) > 0
        ):
            successful_tools_set = set(self.context.session.successful_tools)
            successful_intersection = (
                self.context.session.min_required_tools.intersection(
                    successful_tools_set
                )
            )
            deterministic_score = len(successful_intersection) / len(
                self.context.session.min_required_tools
            )
            tools_completed = self.context.session.min_required_tools.issubset(
                successful_tools_set
            )
            self.agent_logger.info(
                f"Required tools check: Required={self.context.session.min_required_tools}, "
                f"Successful={successful_tools_set}, Completed={tools_completed}, Score={deterministic_score:.2f}"
            )
            if not tools_completed:
                missing_tools = (
                    self.context.session.min_required_tools - successful_tools_set
                )
                nudge = f"**Task requires completion of these tools: {missing_tools}**"
                if nudge not in self.context.session.task_nudges:
                    self.context.session.task_nudges.append(nudge)
        else:
            tools_completed = True
            deterministic_score = 1.0
            self.agent_logger.info(
                "No minimum required tools set. Score=1.0, Tools Completed=True."
            )

        # Store the calculated score in session
        self.context.session.completion_score = deterministic_score

        score_met = deterministic_score >= self.context.min_completion_score

        # Check if ALL conditions are met: Score Threshold Met, All Required Tools Used, AND Final Answer Provided
        if (
            score_met
            and tools_completed
            and self.context.session.final_answer is not None
        ):
            self.agent_logger.info(
                f"Stopping loop: Score threshold met ({deterministic_score:.2f} >= {self.context.min_completion_score}), "
                f"all required tools used, and final answer provided."
            )
            self.context.session.task_status = TaskStatus.COMPLETE
            return False
        elif tools_completed and self.context.session.final_answer is None:
            nudge = "**All required tools used, but requires 'final_answer(<answer>)' tool call.**"
            if nudge not in self.context.session.task_nudges:
                self.context.session.task_nudges.append(nudge)
        elif score_met and not tools_completed:
            nudge = f"**Score threshold ({self.context.min_completion_score}) met, but required tools still missing.**"
            if nudge not in self.context.session.task_nudges:
                self.context.session.task_nudges.append(nudge)

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
