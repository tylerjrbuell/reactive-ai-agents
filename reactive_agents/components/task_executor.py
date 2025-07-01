"""
Task execution module for reactive-ai-agent framework.
Handles the core task execution logic for agents.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import asyncio
import time
from reactive_agents.common.types.status_types import TaskStatus
from reactive_agents.prompts.agent_prompts import HYBRID_TASK_PLANNING_SYSTEM_PROMPT
import json
from reactive_agents.components.plan_manager import PlanManager
from reactive_agents.common.types.session_types import PlanStep, StepStatus

if TYPE_CHECKING:
    from reactive_agents.components.execution_engine import AgentExecutionEngine
    from reactive_agents.agents.base import Agent


class TaskExecutor:
    """
    Handles the core task execution logic for agents.
    Manages the execution flow, reflection, and planning steps.
    """

    def __init__(self, agent: "Agent"):
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
        self.plan_manager = PlanManager(self.context)
        self.plan_regeneration_count = 0
        self.max_plan_regenerations = 3

    @property
    def execution_engine(self) -> "AgentExecutionEngine":
        """Get the execution engine from the agent."""
        if not self.agent.execution_engine:
            raise RuntimeError("Execution engine not initialized")
        return self.agent.execution_engine

    async def _force_tool_call(self, step: PlanStep) -> Any:
        """
        Force a tool call for the given step using the model provider's get_tool_calls method.
        """
        self.agent_logger.info(f"🔧 Forcing tool call for step: {step.description}")
        # Prepare tool signatures and call get_tool_calls
        tool_signatures = self.context.get_tool_signatures()
        tool_calls = []
        results = []
        get_tool_calls_fn = getattr(self.model_provider, "get_tool_calls", None)
        if not callable(get_tool_calls_fn):
            self.agent_logger.error(
                f"Model provider {self.model_provider.name} does not support forced tool calls."
            )
            return {
                "error": f"Model provider {self.model_provider.name} does not support forced tool calls."
            }
        try:
            tool_calls = await get_tool_calls_fn(
                task=step.description,
                context=self.context.session.messages,
                model=getattr(self.model_provider, "model", None),
                tools=tool_signatures,
                options=self.context.model_provider_options,
            )
        except Exception as e:
            self.agent_logger.error(
                f"Failed to generate tool call for step '{step.description}': {e}"
            )
            return {"error": str(e)}
        for tool_call in tool_calls:
            results.append(await self.execute_tool_call(tool_call.model_dump()))
        return results

    def _should_execute_step(self, step: Optional[PlanStep]) -> bool:
        """Return True if the step should be executed (pending, failed, or skipped)."""
        return step is not None and step.status in [
            StepStatus.PENDING,
            StepStatus.FAILED,
            StepStatus.SKIPPED,
        ]

    async def execute_iteration(self, task: str) -> Optional[Dict[str, Any]]:
        """
        Executes one iteration of the simplified agent reasoning loop:
        - Execute current step
        - Evaluate step completion
        - Move to next step or complete
        """
        # --- Context management: summarize/prune context as needed ---
        await self.context.manage_context()

        # --- 0. Ensure plan and plan steps exist ---
        if not self.context.session.plan or not self.context.session.plan_steps:
            await self.plan_manager.generate_and_store_plan(task)
            # PlanManager already initializes plan steps and index

        # --- 1. Execute current step ---
        current_step = self.plan_manager.get_current_step()
        self.agent_logger.debug(f"Current step: {current_step}")

        if current_step is None:
            self.agent_logger.error("No current step found in session")
            return None
        step_result = None
        if self._should_execute_step(current_step):
            # Always reset status to PENDING before execution
            self.plan_manager.mark_step_status(current_step.index, StepStatus.PENDING)

            # Insert user message for the current step (deduplicated)
            messages = self.context.session.messages
            if not (
                messages
                and messages[-1].get("role") == "user"
                and messages[-1].get("content") == current_step.description
            ):
                messages.append({"role": "user", "content": current_step.description})

            self.agent_logger.info(
                f"📝 Executing step {current_step.index + 1}: {current_step.description}"
            )

            # Use is_action to decide how to execute the step
            if getattr(current_step, "is_action", False):
                step_result = await self._force_tool_call(current_step)
            else:
                step_result = await self._think_and_act()
            # step_result = await self._think_and_act()
            # Check if plan is exhausted
            if not current_step and not self.plan_manager.is_plan_complete():
                await self._handle_plan_exhaustion()
                if self.context.session.plan:
                    self.plan_manager.reset_plan_steps()

        # --- 2. Step Evaluation: evaluate step completion ---
        reflection_manager = self.context.get_reflection_manager()
        self.agent_logger.info("🤔 Phase 2: Step Evaluation")
        step_evaluation = await reflection_manager.reflect_and_evaluate_steps(
            step_result
        )

        if step_evaluation:
            # Update step statuses based on evaluation
            step_updates = step_evaluation.get("step_updates", [])
            self._update_step_status(step_updates)

            # Update current step index
            next_step_index = step_evaluation.get(
                "next_step_index", self.context.session.current_step_index
            )
            if next_step_index != self.context.session.current_step_index:
                self.agent_logger.info(
                    f"📝 Moving from step {self.context.session.current_step_index} to step {next_step_index}"
                )
                self.context.session.current_step_index = next_step_index

            # Check if plan is complete
            plan_complete = step_evaluation.get("plan_complete", False)
            if plan_complete:
                self.agent_logger.info("✅ Plan execution complete")
                self.context.session.task_status = TaskStatus.COMPLETE

        # --- 3. Summarize/log progress ---
        completion_percentage = self.context.session.get_completion_percentage()
        self.agent_logger.info(
            f"✅ Iteration Complete. Plan Progress: {completion_percentage:.1f}%"
        )

        return {
            "current_step": current_step.description if current_step else None,
            "step_result": step_result,
            "step_evaluation": step_evaluation,
            "completion_percentage": completion_percentage,
        }

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

        # Check if plan is complete using step-based system
        if self.plan_manager.is_plan_complete():
            self.agent_logger.info("Stopping loop: All plan steps completed.")
            self.context.session.task_status = TaskStatus.COMPLETE
            return False

        # Check required tools completion with consistent logic from ReactAgent
        min_required_tools = self.context.session.min_required_tools or set()
        successful_tools = self.context.session.successful_tools
        if min_required_tools:
            tools_completed, missing_tools = self.context.has_completed_required_tools()
            successful_intersection = min_required_tools.intersection(successful_tools)
            deterministic_score = len(successful_intersection) / len(min_required_tools)
            self.agent_logger.info(
                f"Required tools check: Required={min_required_tools}, "
                f"Successful={successful_tools}, Completed={tools_completed}, Score={deterministic_score:.2f}"
            )
            if not tools_completed:
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

        # Check for repeated assistant messages (use session messages)
        if (
            self.context.session.iterations > 1
            and len(self.context.session.messages) > 3
        ):
            if (
                self.context.session.messages[-1].get("role") == "tool"
                and self.context.session.messages[-2].get("role") == "assistant"
            ):
                last_assistant_msg = self.context.session.messages[-2]
                if (
                    len(self.context.session.messages) > 4
                    and self.context.session.messages[-3].get("role") == "tool"
                    and self.context.session.messages[-4].get("role") == "assistant"
                ):
                    prev_assistant_msg = self.context.session.messages[-4]
                    if last_assistant_msg.get("content") and last_assistant_msg.get(
                        "content"
                    ) == prev_assistant_msg.get("content"):
                        if not last_assistant_msg.get("tool_calls"):
                            self.agent_logger.warning(
                                "Stopping loop: Assistant content repeated between iterations."
                            )
                            self.context.session.task_status = (
                                TaskStatus.COMPLETE
                            )  # Update session status
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

    async def _think_and_act(self) -> Any:
        """
        Calls the agent's _think_chain method to let the LLM generate the next tool call or response.
        Returns the result of the model's reasoning and tool use.
        """
        result = await self.agent._think_chain(remember_messages=True, use_tools=True)
        if result is None:
            return
        self.context.session.last_result = result.get("message", {}).get("content")
        return result

    async def _handle_plan_exhaustion(self):
        """
        Handles the case where the plan is missing or exhausted by regenerating the plan.
        Resets the current step index to 0.
        """
        current_task = (
            self.context.session.current_task or self.context.session.initial_task
        )
        await self.plan_manager.generate_and_store_plan(current_task)
        # The _generate_and_store_plan method will initialize plan steps and set current_step_index to 0

    def _ensure_final_answer_in_plan(self, plan: List[Any]) -> List[Any]:
        """
        Ensures the plan always ends with the final_answer tool.
        If the plan doesn't end with final_answer, adds it as the last step.
        Handles both dict (new format) and str (legacy format) plan steps.
        """
        if not plan:
            return [
                {
                    "description": "Use final_answer with the complete answer",
                    "is_action": True,
                }
            ]

        last_step = plan[-1]
        if isinstance(last_step, dict):
            last_desc = last_step.get("description", "")
        else:
            last_desc = str(last_step)
        if "final_answer" in last_desc:
            return plan
        # Add final_answer as the last step in new format
        return plan + [
            {
                "description": "Use final_answer with the complete answer",
                "is_action": True,
            }
        ]

    def _update_step_status(self, step_updates: List[Dict[str, Any]]) -> None:
        """
        Update step status based on reflection results.
        """
        for update in step_updates:
            step_index = update.get("step_index")
            status = update.get("status")
            result = update.get("result")
            error = update.get("error")
            tool_used = update.get("tool_used")
            parameters = update.get("parameters")

            if step_index is not None and 0 <= step_index < len(
                self.context.session.plan_steps
            ):
                step = self.context.session.plan_steps[step_index]
                if status:
                    step.status = StepStatus(status)
                if result is not None:
                    step.result = result
                if error is not None:
                    step.error = error
                if tool_used is not None:
                    step.tool_used = tool_used
                if parameters is not None:
                    step.parameters = parameters

                self.agent_logger.info(
                    f"Updated step {step_index}: {step.status.value}"
                )
