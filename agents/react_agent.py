from __future__ import annotations
import json
import traceback
import asyncio
import os
from datetime import datetime
from enum import Enum, auto
from typing import List, Any, Optional, Dict, Set, Union, Tuple, Callable
from collections.abc import Awaitable
import time
import random

from pydantic import BaseModel, Field, model_validator
from prompts.agent_prompts import (
    AGENT_ACTION_PLAN_PROMPT,
    MISSING_TOOLS_PROMPT,
    TOOL_FEASIBILITY_CONTEXT_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_CONTEXT_PROMPT,
    RESCOPE_SYSTEM_PROMPT,
    RESCOPE_CONTEXT_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    EVALUATION_CONTEXT_PROMPT,
    PLANNING_CONTEXT_PROMPT,
)
from agents.base import Agent
from context.agent_context import AgentContext
from agent_mcp.client import MCPClient

# Import shared types from the new location
from common.types import TaskStatus


# --- Agent Configuration Model ---
class ReactAgentConfig(BaseModel):
    agent_name: str = Field("ReactAgent", description="Name of the agent.")
    role: str = Field("Task Executor", description="Role of the agent.")
    provider_model_name: str = Field(
        "ollama:qwen2:7b", description="Name of the LLM provider and model."
    )
    mcp_client: Optional[MCPClient] = Field(
        None, description="An initialized MCPClient instance."
    )
    min_completion_score: float = Field(
        1.0, ge=0.0, le=1.0, description="Minimum score for task completion evaluation."
    )
    instructions: str = Field(
        "Solve the given task.", description="High-level instructions for the agent."
    )
    max_iterations: Optional[int] = Field(
        10, description="Maximum number of iterations allowed."
    )
    reflect_enabled: bool = Field(
        True, description="Whether reflection mechanism is enabled."
    )
    log_level: str = Field(
        "info", description="Logging level ('debug', 'info', 'warning', 'error')."
    )
    initial_task: Optional[str] = Field(
        None, description="The initial task description (can also be passed to run)."
    )
    tool_use_enabled: bool = Field(True, description="Whether the agent can use tools.")
    use_memory_enabled: bool = Field(
        True, description="Whether the agent uses long-term memory."
    )
    collect_metrics_enabled: bool = Field(
        True, description="Whether to collect performance metrics."
    )
    check_tool_feasibility: bool = Field(
        True, description="Whether to check tool feasibility before starting."
    )
    enable_caching: bool = Field(
        True, description="Whether to enable LLM response caching."
    )
    confirmation_callback: Optional[
        Callable[[str, Dict[str, Any]], Awaitable[bool]]
    ] = Field(None, description="Async callback for confirming tool use.")
    # Store extra kwargs passed, e.g. for specific context managers
    kwargs: Dict[str, Any] = Field(
        {}, description="Additional keyword arguments passed to AgentContext."
    )

    class Config:
        arbitrary_types_allowed = True  # Allow MCPClient etc.

    @model_validator(mode="before")
    def capture_extra_kwargs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        known_fields = {f for f in cls.model_fields if f != "kwargs"}
        extra_kwargs = {}
        processed_values = {}
        for key, value in values.items():
            if key in known_fields:
                processed_values[key] = value
            else:
                extra_kwargs[key] = value
        processed_values["kwargs"] = extra_kwargs
        return processed_values


# --- End Agent Configuration Model ---


class ReactAgent(Agent):
    """
    A ReAct-style agent that uses reflection and planning within an AgentContext.
    """

    class PlanFormat(BaseModel):
        next_step: str
        rationale: str
        suggested_tools: List[str] = []

    class ToolAnalysisFormat(BaseModel):
        required_tools: List[str] = Field(
            ..., description="List of tools essential for this task"
        )
        optional_tools: List[str] = Field(
            [], description="List of tools helpful but not essential"
        )
        explanation: str = Field(
            ..., description="Brief explanation of the tool requirements"
        )

    class RescopeFormat(BaseModel):
        rescoped_task: Optional[str] = Field(
            None,
            description="A simplified, achievable task, or null if no rescope possible.",
        )
        explanation: str = Field(
            ..., description="Why this task was/wasn't rescoped and justification."
        )
        expected_tools: List[str] = Field(
            [], description="Tools expected for the rescoped task (if any)."
        )

    class EvaluationFormat(BaseModel):
        adherence_score: float = Field(
            ...,
            ge=0.0,
            le=1.0,
            description="Score 0.0-1.0: how well the result matches the goal",
        )
        strengths: List[str] = Field(
            [], description="Ways the result successfully addressed the goal"
        )
        weaknesses: List[str] = Field(
            [], description="Ways the result fell short of the goal"
        )
        explanation: str = Field(..., description="Overall explanation of the rating")
        matches_intent: bool = Field(
            ...,
            description="Whether the result fundamentally addresses the user's core intent",
        )

    # --- End Pydantic models in class ---

    def __init__(
        self,
        config: ReactAgentConfig,
    ):
        """
        Initializes the ReactAgent using a configuration object.

        Args:
            config: The ReactAgentConfig object containing all settings.
        """
        context = AgentContext(
            agent_name=config.agent_name,
            role=config.role,
            provider_model_name=config.provider_model_name,
            mcp_client=config.mcp_client,
            min_completion_score=config.min_completion_score,
            instructions=config.instructions,
            max_iterations=config.max_iterations,
            reflect_enabled=config.reflect_enabled,
            log_level=config.log_level,
            initial_task=config.initial_task or "",
            tool_use_enabled=config.tool_use_enabled,
            use_memory_enabled=config.use_memory_enabled,
            collect_metrics_enabled=config.collect_metrics_enabled,
            check_tool_feasibility=config.check_tool_feasibility,
            enable_caching=config.enable_caching,
            confirmation_callback=config.confirmation_callback,
            **config.kwargs,
        )
        super().__init__(context)
        # Store the config for potential reference, though context holds the state
        self.config = config
        self.agent_logger.info(
            f"ReactAgent '{self.context.agent_name}' initialized with internal context via config."
        )

    async def run(
        self,
        initial_task: Optional[str] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Run the ReAct agent loop for the given task.

        Args:
            initial_task: The task description. If None, uses the initial_task passed during initialization.
            cancellation_event: An optional asyncio.Event to signal cancellation.

        Returns:
            A dictionary containing the final status, result, and other execution details.
        """
        # --- Use initial_task from run() if provided, otherwise from context init ---
        current_initial_task = initial_task or self.context.initial_task
        if not current_initial_task:
            raise ValueError(
                "An initial task must be provided either during initialization or when calling run()."
            )

        # 1. --- Initialization and Context Setup ---
        self.context.start_time = time.time()
        self.context.initial_task = current_initial_task  # Use the determined task
        self.context.current_task = current_initial_task  # Use the determined task
        self.context.iterations = 0
        self.context.final_answer = None
        self.context.task_progress = ""
        self.context.reasoning_log = []
        self.context.task_status = TaskStatus.INITIALIZED

        # Reset metrics
        if self.context.metrics_manager:
            self.context.metrics_manager.reset()

        # Reset reflections for the new run
        if self.context.reflection_manager:
            self.context.reflection_manager.reset()
            # We might want to load relevant reflections *after* reset if needed
            # E.g., self.context.reflection_manager.load_relevant_reflections(initial_task)

        self.agent_logger.info(
            f"ReactAgent run starting for task: {self.context.initial_task}"
        )  # Log correct task

        # Local state for run management
        last_error: Optional[str] = None
        rescoped_task: Optional[str] = None
        final_result_content: Optional[str] = None
        max_failures = self.context.max_task_retries
        failure_count = 0
        # Use a set to track active asyncio tasks spawned by the agent during the run
        active_tasks: Set[asyncio.Task] = set()

        try:
            # 2. --- Pre-run Checks (Dependencies, Tool Feasibility) ---
            if not self._check_dependencies():
                return self._prepare_final_result(rescoped_task)

            self.context.min_required_tools = None  # Initialize/reset for the run
            if self.context.check_tool_feasibility:
                feasibility = await self.check_tool_feasibility(
                    self.context.initial_task
                )  # Use context's initial_task
                # Store the initial required tools if feasibility check ran and succeeded
                if feasibility and feasibility.get("required_tools") is not None:
                    self.context.min_required_tools = set(feasibility["required_tools"])
                    self.agent_logger.debug(
                        f"Stored initial required tools: {self.context.min_required_tools}"
                    )

                if not feasibility["feasible"]:
                    self.agent_logger.warning(
                        f"Missing required tools: {feasibility['missing_tools']}"
                    )
                    self.context.reasoning_log.append(
                        f"Cannot complete task: Missing Tools - {feasibility.get('explanation', '')}"
                    )
                    self.context.task_status = TaskStatus.MISSING_TOOLS
                    if self.context.workflow_manager:
                        self.context.workflow_manager.update_context(
                            TaskStatus.MISSING_TOOLS,
                            missing_tools=feasibility["missing_tools"],
                            explanation=feasibility["explanation"],
                        )
                    return self._prepare_final_result(
                        rescoped_task, feasibility=feasibility
                    )

            # 3. --- Set Status to Running ---
            self.context.task_status = TaskStatus.RUNNING
            if self.context.workflow_manager:
                self.context.workflow_manager.update_context(TaskStatus.RUNNING)

            # Initialize last_plan outside the loop
            last_plan = None

            # 4. --- Execution Loop ---
            while self._should_continue():
                # Check for cancellation
                if cancellation_event and cancellation_event.is_set():
                    self.agent_logger.info("Task execution cancelled by user.")
                    self.context.task_status = TaskStatus.CANCELLED
                    break

                self.context.iterations += 1
                self.agent_logger.info(
                    f"🔄 ITERATION {self.context.iterations}/{self.context.max_iterations or 'unlimited'}"
                )
                # Update workflow context iteration count
                if self.context.workflow_manager:
                    self.context.workflow_manager.update_context(
                        self.context.task_status
                    )

                # Inject guidance from the previous iteration's plan
                if (
                    last_plan
                    and last_plan.get("next_step")
                    and last_plan.get("rationale")
                ):
                    guidance_message = {
                        "role": "user",
                        "content": f"Based on the previous plan, the next concrete step to take is: {last_plan['next_step']}. Rationale: {last_plan['rationale']}. Please proceed with this action.",
                    }
                    self.context.messages.append(guidance_message)
                    self.agent_logger.debug(
                        f"Injecting plan guidance: {guidance_message['content']}"
                    )

                current_task_for_iteration = rescoped_task or self.context.current_task

                try:
                    # --- Run the full React iteration (Think/Act -> Reflect -> Plan) ---
                    # This method now handles internal logic and returns (content, plan)
                    self.agent_logger.info(
                        f"🔄 Iteration {self.context.iterations} current task: {current_task_for_iteration}"
                    )
                    # Capture both content and the new plan for the *next* iteration
                    iteration_content, new_plan = await self._run_task_iteration(
                        task=current_task_for_iteration
                    )
                    self.agent_logger.info(f"🔄 Content Preview: {iteration_content}")

                    # Update last_plan for the next iteration
                    last_plan = new_plan

                    # Update final_result_content if the iteration produced text
                    if iteration_content:
                        final_result_content = iteration_content

                    # Check if a tool inside the iteration set the final answer
                    if self.context.final_answer:
                        self.agent_logger.info(
                            "✅ Final answer received during task execution."
                        )
                        self.context.task_status = (
                            TaskStatus.COMPLETE
                        )  # Ensure status is updated
                        break  # Exit loop immediately

                    # Update system prompt if needed
                    self.context.update_system_prompt()

                    # Reset failure counter on successful iteration
                    failure_count = 0

                except Exception as iter_error:
                    failure_count += 1
                    last_error = str(iter_error)
                    tb_str = traceback.format_exc()
                    self.agent_logger.error(
                        f"❌ Iteration {self.context.iterations} Error: {last_error}\n{tb_str}"
                    )
                    self.context.reasoning_log.append(
                        f"ERROR in iteration {self.context.iterations}: {last_error}"
                    )

                    if failure_count >= max_failures:
                        self.agent_logger.warning(
                            f"Reached max failures ({max_failures}). Attempting to rescope task."
                        )
                        error_context = f"Multiple ({failure_count}) failures during execution. Last error: {last_error}"
                        rescope_result = await self.rescope_goal(
                            self.context.initial_task, error_context
                        )

                        if rescope_result["rescoped_task"]:
                            rescoped_task = rescope_result["rescoped_task"]
                            # Check type before assignment to satisfy linter
                            if isinstance(rescoped_task, str):
                                self.context.current_task = rescoped_task
                                self.agent_logger.info(
                                    f"Task rescoped to: {rescoped_task}"
                                )
                                self.context.reasoning_log.append(
                                    f"Task rescoped: {rescope_result['explanation']}"
                                )
                                failure_count = (
                                    0  # Reset failure count for the new task
                                )

                                # <<< UPDATE min_required_tools based on rescope >>>
                                if rescope_result.get("expected_tools") is not None:
                                    self.context.min_required_tools = set(
                                        rescope_result["expected_tools"]
                                    )
                                    self.agent_logger.debug(
                                        f"Updated required tools for rescoped task: {self.context.min_required_tools}"
                                    )
                                else:
                                    # If rescope doesn't specify tools, maybe reset or keep old? Reset seems safer.
                                    self.context.min_required_tools = None
                                    self.agent_logger.debug(
                                        "Reset required tools as rescope did not specify expected tools."
                                    )
                                # <<< END UPDATE >>>

                                if self.context.workflow_manager:
                                    self.context.workflow_manager.update_context(
                                        self.context.task_status,
                                        rescoped=True,
                                        original_task=self.context.initial_task,
                                        rescoped_task=rescoped_task,
                                    )
                                # Only continue if task was successfully rescoped and assigned
                                continue
                            else:
                                # This case should ideally not happen if rescope_result["rescoped_task"] is truthy
                                self.agent_logger.error(
                                    "Rescoping failed unexpectedly: rescoped_task was not a string."
                                )
                                self.context.task_status = TaskStatus.ERROR
                                break  # Exit loop on error

                        # If rescoping failed or wasn't possible
                        self.agent_logger.error(
                            "Could not rescope task after multiple failures."
                        )
                        self.context.task_status = TaskStatus.ERROR
                        self.context.reasoning_log.append(
                            "Failed to rescope task after multiple errors."
                        )
                        break

            # 5. --- Loop End: Determine Final Status ---
            self.agent_logger.info(
                f"🏁 React loop finished after {self.context.iterations} iterations."
            )
            if self.context.task_status not in [TaskStatus.ERROR, TaskStatus.CANCELLED]:
                if self.context.final_answer:
                    self.context.task_status = TaskStatus.COMPLETE
                elif rescoped_task:
                    last_reflection = (
                        self.context.reflection_manager.get_last_reflection()
                        if self.context.reflection_manager
                        else None
                    )
                    if (
                        last_reflection
                        and last_reflection.get("completion_score", 0)
                        >= self.context.min_completion_score
                    ):
                        self.context.task_status = TaskStatus.RESCOPED_COMPLETE
                    else:
                        self.context.task_status = (
                            TaskStatus.MAX_ITERATIONS
                            if self.context.iterations
                            >= (self.context.max_iterations or float("inf"))
                            else TaskStatus.ERROR
                        )
                        if not final_result_content and last_error:
                            final_result_content = f"Error: {last_error}"
                elif self.context.iterations >= (
                    self.context.max_iterations or float("inf")
                ):
                    self.context.task_status = TaskStatus.MAX_ITERATIONS
                else:
                    self.context.task_status = TaskStatus.COMPLETE

            if (
                self.context.task_status == TaskStatus.ERROR
                and not final_result_content
                and last_error
            ):
                final_result_content = f"Error: {last_error}"
            elif self.context.task_status == TaskStatus.CANCELLED:
                final_result_content = "Task cancelled by user."

            # 6. --- Post-run Processing (Summary, Evaluation) ---
            self.context.session.summary = await self.generate_summary()
            self.context.session.evaluation = (
                await self.generate_goal_result_evaluation()
            )

        except Exception as run_error:
            tb_str = traceback.format_exc()
            self.agent_logger.error(
                f"Unhandled error during agent run: {run_error}\n{tb_str}"
            )
            self.context.task_status = TaskStatus.ERROR
            final_result_content = f"Critical error during agent run: {run_error}"
            self.context.session.evaluation = {
                "adherence_score": 0.0,
                "matches_intent": False,
                "explanation": "Critical error.",
            }

        finally:
            # 7. --- Cleanup and Result Preparation ---
            # Cancel any lingering tasks spawned during the run
            if active_tasks:
                self.agent_logger.debug(
                    f"Cleaning up {len(active_tasks)} background tasks..."
                )
                for task in active_tasks:
                    if not task.done():
                        task.cancel()
                active_tasks.clear()

            result_to_return = (
                self.context.final_answer
                or final_result_content
                or "Task completed without explicit result."
            )

            # Prepare the final result dictionary
            final_result_package = self._prepare_final_result(
                rescoped_task=rescoped_task,
                result_content=result_to_return,
                summary=self.context.session.summary,
                evaluation=self.context.session.evaluation,
            )

            # Update workflow context with final status
            if self.context.workflow_manager:
                self.context.workflow_manager.update_context(
                    self.context.task_status,
                    result_to_return,
                    adherence_score=self.context.session.evaluation.get(
                        "adherence_score"
                    ),
                    matches_intent=self.context.session.evaluation.get(
                        "matches_intent"
                    ),
                    rescoped=(rescoped_task is not None),
                    error=(
                        last_error
                        if self.context.task_status == TaskStatus.ERROR
                        else None
                    ),
                )

            # Finalize metrics
            if self.context.metrics_manager:
                self.context.metrics_manager.finalize_run_metrics()
                final_result_package["metrics"] = (
                    self.context.metrics_manager.get_metrics()
                )

            # Save memory
            if self.context.memory_manager:
                self.context.memory_manager.update_session_history(final_result_package)
                self.context.memory_manager.save_memory()

            self.agent_logger.info(
                f"ReactAgent run finished with status: {self.context.task_status}"
            )

        return final_result_package

    async def close(self):
        """Closes the agent's context and associated resources."""
        self.agent_logger.info(
            f"Closing context for agent '{self.context.agent_name}'..."
        )
        await self.context.close()
        self.agent_logger.info(f"Context closed for agent '{self.context.agent_name}'.")

    def _prepare_final_result(
        self,
        rescoped_task: Optional[str],
        result_content: Optional[str] = None,
        summary: Optional[str] = None,
        evaluation: Optional[Dict] = None,
        feasibility: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Helper to construct the final return dictionary."""
        # Convert set to list for serialization if needed
        min_req_tools_list = (
            list(self.context.min_required_tools)
            if self.context.min_required_tools is not None
            else None
        )
        return {
            "status": str(self.context.task_status),
            "result": result_content
            or self.context.final_answer
            or "No textual result produced.",
            "iterations": self.context.iterations,
            "summary": summary or "Summary not generated.",
            "reasoning_log": self.context.reasoning_log,
            "evaluation": evaluation
            or {
                "adherence_score": 0.0,
                "matches_intent": False,
                "explanation": "Evaluation not performed.",
            },
            "rescoped": rescoped_task is not None,
            "original_task": self.context.initial_task,
            "rescoped_task": rescoped_task,
            "min_required_tools": min_req_tools_list,
            "metrics": (
                self.context.get_metrics() if self.context.metrics_manager else None
            ),
            **(feasibility if feasibility else {}),
        }

    def _check_dependencies(self) -> bool:
        """Delegates dependency check to the WorkflowManager."""
        if self.context.workflow_manager:
            return self.context.workflow_manager.check_dependencies()
        self.agent_logger.debug("No workflow manager, skipping dependency check.")
        return True

    def _should_continue(self) -> bool:
        """Determines if the ReAct loop should continue."""
        if self.context.task_status in [
            TaskStatus.COMPLETE,
            TaskStatus.ERROR,
            TaskStatus.CANCELLED,
            TaskStatus.RESCOPED_COMPLETE,
            TaskStatus.MAX_ITERATIONS,
            TaskStatus.MISSING_TOOLS,
        ]:
            self.agent_logger.debug(
                f"Stopping loop: Terminal status {self.context.task_status}."
            )
            return False

        if (
            self.context.max_iterations is not None
            and self.context.iterations >= self.context.max_iterations
        ):
            self.agent_logger.info(
                f"Stopping loop: Max iterations ({self.context.max_iterations}) reached."
            )
            self.context.task_status = TaskStatus.MAX_ITERATIONS
            return False

        if self.context.final_answer is not None:
            self.agent_logger.info("Stopping loop: Final answer set.")
            if self.context.task_status == TaskStatus.RUNNING:
                self.context.task_status = TaskStatus.COMPLETE
            return False

        if not self._check_dependencies():
            self.agent_logger.info("Stopping loop: Dependencies not met.")
            return False

        if self.context.reflect_enabled and self.context.reflection_manager:
            # We still need reflection to potentially suggest next steps or identify loops,
            # but completion decision is now based purely on tools and final_answer.
            pass  # Placeholder, maybe add check for infinite loops based on reflection?

        # Check required tools completion against actual successful tools context
        tools_completed = False
        deterministic_score = 0.0
        if (
            self.context.min_required_tools is not None
            and len(self.context.min_required_tools) > 0
        ):
            successful_tools_set = set(self.context.successful_tools)
            successful_intersection = self.context.min_required_tools.intersection(
                successful_tools_set
            )
            deterministic_score = len(successful_intersection) / len(
                self.context.min_required_tools
            )
            tools_completed = self.context.min_required_tools.issubset(
                successful_tools_set
            )
            self.agent_logger.info(
                f"Required tools check: Required={self.context.min_required_tools}, "
                f"Successful={successful_tools_set}, Completed={tools_completed}, Score={deterministic_score:.2f}"
            )
            if not tools_completed:
                missing_tools = self.context.min_required_tools - successful_tools_set
                # Avoid adding duplicate nudges
                nudge = f"**Task requires completion of these tools: {missing_tools}**"
                if nudge not in self.context.task_nudges:
                    self.context.task_nudges.append(nudge)
        else:
            # If no minimum required tools were identified, consider task complete from tool perspective
            tools_completed = True
            deterministic_score = 1.0  # Score is 1.0 if no tools required
            self.agent_logger.info(
                "No minimum required tools set. Score=1.0, Tools Completed=True."
            )

        # Check if completion score threshold is met
        score_met = deterministic_score >= self.context.min_completion_score

        # Check if ALL conditions are met: Score Threshold Met, All Required Tools Used, AND Final Answer Provided
        if score_met and tools_completed and self.context.final_answer is not None:
            self.agent_logger.info(
                f"Stopping loop: Score threshold met ({deterministic_score:.2f} >= {self.context.min_completion_score}), "
                f"all required tools used, and final answer provided."
            )
            self.context.task_status = TaskStatus.COMPLETE  # Ensure status is COMPLETE
            return False
        elif tools_completed and self.context.final_answer is None:
            # Add nudge if tools are done, but answer missing (score doesn't matter here)
            nudge = "**All required tools used, but requires final_answer(<answer>) tool call.**"
            if nudge not in self.context.task_nudges:
                self.context.task_nudges.append(nudge)
        elif score_met and not tools_completed:
            # Add nudge if score threshold met, but tools still missing (relevant if score < 1.0)
            nudge = f"**Score threshold ({self.context.min_completion_score}) met, but required tools still missing.**"
            if nudge not in self.context.task_nudges:
                self.context.task_nudges.append(nudge)

        if self.context.iterations > 1 and len(self.context.messages) > 3:
            if (
                self.context.messages[-1].get("role") == "tool"
                and self.context.messages[-2].get("role") == "assistant"
            ):
                last_assistant_msg = self.context.messages[-2]
                if (
                    len(self.context.messages) > 4
                    and self.context.messages[-3].get("role") == "tool"
                    and self.context.messages[-4].get("role") == "assistant"
                ):
                    prev_assistant_msg = self.context.messages[-4]
                    if last_assistant_msg.get("content") and last_assistant_msg.get(
                        "content"
                    ) == prev_assistant_msg.get("content"):
                        if not last_assistant_msg.get("tool_calls"):
                            self.agent_logger.warning(
                                "Stopping loop: Assistant content repeated between iterations."
                            )
                            self.context.task_status = TaskStatus.COMPLETE
                            return False

        return True

    async def check_tool_feasibility(self, task: str) -> Dict[str, Any]:
        """Checks if required tools are available using the ToolManager."""
        self.agent_logger.info(f"Checking tool feasibility for task: {task[:100]}...")
        if not self.context.tool_manager:
            self.agent_logger.error(
                "Cannot check tool feasibility: ToolManager missing."
            )
            return {
                "feasible": True,
                "missing_tools": [],
                "explanation": "ToolManager unavailable.",
            }

        try:
            available_tools = self.context.get_available_tool_names()
            tool_signatures = self.context.get_tool_signatures()
            if not available_tools:
                self.agent_logger.info("No tools available in ToolManager.")
                return {
                    "feasible": True,
                    "missing_tools": [],
                    "explanation": "No tools available.",
                }

            # Use centralized prompts
            system_prompt = MISSING_TOOLS_PROMPT
            prompt = TOOL_FEASIBILITY_CONTEXT_PROMPT.format(
                task=task, available_tools=json.dumps(tool_signatures, indent=2)
            )

            response = await self.model_provider.get_completion(
                system=system_prompt,
                prompt=prompt,
                format=(
                    self.ToolAnalysisFormat.model_json_schema()
                    if self.model_provider.name == "ollama"
                    else "json"
                ),
            )

            if not response or not response.get("response"):
                self.agent_logger.warning(
                    "Tool feasibility check failed: No response from model."
                )
                return {
                    "feasible": True,
                    "missing_tools": [],
                    "explanation": "Could not analyze tool requirements via LLM.",
                }

            try:
                analysis_data = response["response"]
                parsed_analysis = (
                    json.loads(analysis_data)
                    if isinstance(analysis_data, str)
                    else analysis_data
                )
                validated_analysis = self.ToolAnalysisFormat(**parsed_analysis)

                required_tools_set = set(validated_analysis.required_tools)
                missing_tools = list(required_tools_set - available_tools)

                is_feasible = not bool(missing_tools)
                self.agent_logger.info(
                    f"Tool feasibility check result: Feasible={is_feasible}, Missing={missing_tools}"
                )

                return {
                    "feasible": is_feasible,
                    "missing_tools": missing_tools,
                    "required_tools": validated_analysis.required_tools,
                    "optional_tools": validated_analysis.optional_tools,
                    "explanation": validated_analysis.explanation,
                    "available_tools": list(available_tools),
                }

            except (json.JSONDecodeError, TypeError) as e:
                self.agent_logger.error(
                    f"Error parsing tool feasibility analysis JSON: {e}\nResponse: {response.get('response')}"
                )
                return {
                    "feasible": True,
                    "missing_tools": [],
                    "explanation": f"Error parsing analysis: {e}",
                }
            except Exception as e:
                self.agent_logger.error(
                    f"Error validating tool feasibility analysis: {e}\nData: {parsed_analysis}"
                )
                return {
                    "feasible": True,
                    "missing_tools": [],
                    "explanation": f"Error validating analysis: {e}",
                }

        except Exception as e:
            tb_str = traceback.format_exc()
            self.agent_logger.error(
                f"Error during tool feasibility check: {e}\n{tb_str}"
            )
            return {
                "feasible": True,
                "missing_tools": [],
                "explanation": f"Error during feasibility check: {e}",
            }

    async def generate_summary(self) -> str:
        """Generates a summary of the agent's actions using ToolManager history."""
        self.agent_logger.debug("📜 Generating execution summary...")
        if not self.context.tool_manager:
            return "Summary unavailable: ToolManager missing."

        tool_history = self.context.tool_manager.tool_history
        if not tool_history:
            return "No tool actions were taken during this run."

        try:
            tools_used_names = list(
                set(t.get("name", "unknown") for t in tool_history if t.get("name"))
            )
            history_for_prompt = tool_history[-10:]

            summary_context = {
                "task": self.context.initial_task,
                "tools_used_count": len(tools_used_names),
                "tools_used_names": tools_used_names,
                "total_actions": len(tool_history),
                "recent_actions": history_for_prompt,
                "final_status": str(self.context.task_status),
                "final_result_preview": str(self.context.final_answer or "N/A")[:200]
                + "...",
            }

            # Use centralized prompts
            summary_prompt = SUMMARY_CONTEXT_PROMPT.format(
                summary_context_json=json.dumps(summary_context, indent=2, default=str)
            )
            response = await self.model_provider.get_completion(
                system=SUMMARY_SYSTEM_PROMPT, prompt=summary_prompt
            )

            if response and response.get("response"):
                return response["response"]
            else:
                self.agent_logger.warning(
                    "Summary generation failed: No response from model."
                )
                return f"Agent completed task '{self.context.initial_task[:50]}...' with status {self.context.task_status} after {self.context.iterations} iterations, using {len(tools_used_names)} tools."

        except Exception as e:
            self.agent_logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {e}"

    async def rescope_goal(
        self, original_task: str, error_context: str
    ) -> Dict[str, Any]:
        """Attempts to simplify or rescope the task via LLM."""
        self.agent_logger.info(
            f"Attempting to rescope task due to error: {error_context[:100]}..."
        )
        if not self.context.tool_manager:
            self.agent_logger.error(
                "Cannot rescope goal: ToolManager unavailable for tool list."
            )
            return {
                "rescoped_task": None,
                "explanation": "ToolManager unavailable.",
                "original_task": original_task,
            }

        try:
            available_tools = self.context.get_available_tool_names()

            # Use centralized prompts
            rescope_prompt = RESCOPE_CONTEXT_PROMPT.format(
                task=original_task,
                error_context=error_context,
                available_tools=available_tools,
            )
            response = await self.model_provider.get_completion(
                system=RESCOPE_SYSTEM_PROMPT,
                prompt=rescope_prompt,
                format=(
                    self.RescopeFormat.model_json_schema()
                    if self.model_provider.name == "ollama"
                    else "json"
                ),
            )

            if not response or not response.get("response"):
                self.agent_logger.warning(
                    "Goal rescoping failed: No response from model."
                )
                return {
                    "rescoped_task": None,
                    "explanation": "LLM failed to provide rescope analysis.",
                    "original_task": original_task,
                }

            try:
                rescope_data = response["response"]
                parsed_rescope = (
                    json.loads(rescope_data)
                    if isinstance(rescope_data, str)
                    else rescope_data
                )
                validated_rescope = self.RescopeFormat(**parsed_rescope)

                self.agent_logger.info(
                    f"Rescoping result: New task = {validated_rescope.rescoped_task}"
                )
                return validated_rescope.dict() | {"original_task": original_task}

            except (json.JSONDecodeError, TypeError) as e:
                self.agent_logger.error(
                    f"Error parsing rescope analysis JSON: {e}\nResponse: {response.get('response')}"
                )
                return {
                    "rescoped_task": None,
                    "explanation": f"Error parsing rescope response: {e}",
                    "original_task": original_task,
                }
            except Exception as e:
                self.agent_logger.error(
                    f"Error validating rescope analysis: {e}\nData: {parsed_rescope}"
                )
                return {
                    "rescoped_task": None,
                    "explanation": f"Error validating rescope response: {e}",
                    "original_task": original_task,
                }

        except Exception as e:
            tb_str = traceback.format_exc()
            self.agent_logger.error(f"Error during goal rescoping: {e}\n{tb_str}")
            return {
                "rescoped_task": None,
                "explanation": f"Error during rescoping attempt: {e}",
                "original_task": original_task,
            }

    async def generate_goal_result_evaluation(self) -> Dict[str, Any]:
        """Evaluates how well the final result matches the initial task goal."""
        self.agent_logger.info("🔎 Generating goal VS result evaluation...")
        default_eval = {
            "adherence_score": 0.0,
            "matches_intent": False,
            "explanation": "Evaluation could not be performed.",
            "strengths": [],
            "weaknesses": ["Evaluation failed."],
        }

        if not self.context.tool_manager:
            self.agent_logger.warning("Cannot evaluate goal: ToolManager missing.")
            return default_eval

        tool_history = self.context.tool_manager.tool_history
        if not tool_history and not self.context.final_answer:
            self.agent_logger.info("Evaluation: No actions taken, score 0.0.")
            return {
                "adherence_score": 0.0,
                "matches_intent": False,
                "explanation": "No actions were taken and no final answer provided.",
                "strengths": [],
                "weaknesses": ["No progress made."],
            }

        try:
            # ... (build eval_context dict) ...
            tool_names_used = list(
                set(t.get("name", "unknown") for t in tool_history if t.get("name"))
            )
            final_result_str = (
                self.context.final_answer
                or "No explicit final textual answer provided by agent."
            )
            eval_context = {
                "original_goal": self.context.initial_task,
                "final_result": (
                    final_result_str[:3000] + "..."
                    if len(final_result_str) > 3000
                    else final_result_str
                ),
                "final_status": str(self.context.task_status),
                "tools_used": tool_names_used,
                "action_summary": self.context.session.summary,  # Reuse summary
                "reasoning_log": self.context.reasoning_log[-5:],
            }

            # Use centralized prompts
            eval_prompt = EVALUATION_CONTEXT_PROMPT.format(
                eval_context_json=json.dumps(eval_context, indent=2, default=str)
            )
            response = await self.model_provider.get_completion(
                system=EVALUATION_SYSTEM_PROMPT,
                prompt=eval_prompt,
                format=(
                    self.EvaluationFormat.model_json_schema()
                    if self.model_provider.name == "ollama"
                    else "json"
                ),
            )

            if not response or not response.get("response"):
                self.agent_logger.warning(
                    "Goal evaluation failed: No response from model."
                )
                score = (
                    0.7
                    if self.context.task_status
                    in [TaskStatus.COMPLETE, TaskStatus.RESCOPED_COMPLETE]
                    else 0.3
                )
                return {
                    "adherence_score": score,
                    "matches_intent": score > 0.5,
                    "explanation": "Basic evaluation based on final status.",
                    "strengths": ["Task reached final status."],
                    "weaknesses": ["LLM evaluation failed."],
                }

            try:
                eval_data = response["response"]
                parsed_eval = (
                    json.loads(eval_data) if isinstance(eval_data, str) else eval_data
                )
                validated_eval = self.EvaluationFormat(**parsed_eval)

                self.agent_logger.info(
                    f"📄 Goal evaluation result: Score={validated_eval.adherence_score:.2f}, Matches Intent={validated_eval.matches_intent}"
                )
                self.context.reasoning_log.append(
                    f"Goal Adherence Evaluation: {validated_eval.explanation}"
                )
                return validated_eval.dict()

            except (json.JSONDecodeError, TypeError) as e:
                self.agent_logger.error(
                    f"Error parsing evaluation JSON: {e}\nResponse: {response.get('response')}"
                )
                return default_eval | {"explanation": f"Error parsing evaluation: {e}"}
            except Exception as e:
                self.agent_logger.error(
                    f"Error validating evaluation: {e}\nData: {parsed_eval}"
                )
                return default_eval | {
                    "explanation": f"Error validating evaluation: {e}"
                }

        except Exception as e:
            tb_str = traceback.format_exc()
            self.agent_logger.error(f"Error during goal evaluation: {e}\n{tb_str}")
            return default_eval

    async def _plan(self) -> Optional[Dict[str, Any]]:
        """Generates the next action plan based on the current context."""
        if not self.context.reflect_enabled:
            self.agent_logger.debug("Skipping plan step as reflection is disabled.")
            return {
                "next_step": "Continue task execution.",
                "rationale": "Direct execution mode.",
                "suggested_tools": [],
            }

        self.agent_logger.info("🤔 Planning next action...")

        try:
            # ... (build plan_context dict) ...
            plan_context = {
                "main_task": self.context.initial_task,
                "available_tools": [
                    tool.name for tool in self.context.get_available_tools()
                ],
                "previous_steps_summary": self.context.task_progress[-1000:],
                "last_reflection": (
                    self.context.reflection_manager.get_last_reflection()
                    if self.context.reflection_manager
                    else None
                ),
                "current_iteration": self.context.iterations,
                "max_iterations": self.context.max_iterations,
            }

            # Use centralized prompts
            plan_prompt = PLANNING_CONTEXT_PROMPT.format(
                plan_context_json=json.dumps(plan_context, indent=2, default=str)
            )
            response = await self.model_provider.get_completion(
                system=AGENT_ACTION_PLAN_PROMPT,  # Keep existing system prompt
                prompt=plan_prompt,
                format=(
                    self.PlanFormat.model_json_schema()
                    if self.model_provider.name == "ollama"
                    else "json"
                ),
            )

            if not response or not response.get("response"):
                self.agent_logger.warning("Planning failed: No response from model.")
                return {
                    "next_step": "Continue task execution (planning failed).",
                    "rationale": "Planning model failed.",
                    "suggested_tools": [],
                }

            try:
                plan_data = response["response"]
                parsed_plan = (
                    json.loads(plan_data) if isinstance(plan_data, str) else plan_data
                )
                validated_plan = self.PlanFormat(**parsed_plan)

                self.agent_logger.info(
                    f"Plan generated: Next Step = {validated_plan.next_step[:100]}..."
                )
                self.agent_logger.debug(f"Plan details: {validated_plan.dict()}")
                self.context.reasoning_log.append(
                    f"Plan: {validated_plan.rationale} -> {validated_plan.next_step}"
                )

                return validated_plan.dict()

            except (json.JSONDecodeError, TypeError) as e:
                self.agent_logger.error(
                    f"Error parsing planning JSON: {e}\nResponse: {response.get('response')}"
                )
                return None
            except Exception as e:
                self.agent_logger.error(
                    f"Error validating plan: {e}\nData: {parsed_plan}"
                )
                return None

        except Exception as e:
            tb_str = traceback.format_exc()
            self.agent_logger.error(f"Error during planning: {e}\n{tb_str}")
            return None

    async def _run_task_iteration(
        self, task: str
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Executes one iteration of the ReAct loop: Think/Act -> Reflect -> Plan.
        Overrides the base Agent method.
        Returns a tuple: (last_content, plan)
        """

        # --- 1. Think/Act Step ---
        self.agent_logger.info("🧠 Thinking/Acting...")
        think_act_result_dict = await super()._run_task(task=task)
        last_content = None
        if (
            think_act_result_dict
            and isinstance(think_act_result_dict, dict)
            and think_act_result_dict.get("message")
        ):
            msg = think_act_result_dict["message"]
            msg_content = msg.get("content")
            if msg_content:
                last_content = msg_content
                self.context.reasoning_log.append(
                    f"Assistant Thought/Response: {last_content[:200]}..."
                )
            elif msg.get("tool_calls"):
                self.context.reasoning_log.append(f"Assistant Action: Called tools.")
        elif isinstance(think_act_result_dict, str):
            last_content = think_act_result_dict
            self.context.reasoning_log.append(
                f"Step Result (non-dict): {last_content[:200]}..."
            )

        if self.context.final_answer is not None:
            self.agent_logger.info("Final answer set during Think/Act step.")
            # Ensure plan is None if final answer is set here
            return self.context.final_answer, None

        if think_act_result_dict is None:
            self.agent_logger.warning("Think/Act step failed to produce a result.")
            # Ensure plan is None if think/act failed
            return None, None

        # --- 2. Reflect Step ---
        last_reflection = None
        if self.context.reflect_enabled:
            if self.context.reflection_manager:
                reflection = await self.context.reflection_manager.generate_reflection(
                    think_act_result_dict
                )
                if reflection:
                    last_reflection = reflection
                    self.context.reasoning_log.append(
                        f"Reflection:  Reason={reflection['reason']}"
                    )
                else:
                    self.agent_logger.warning("Reflection step failed.")
            else:
                self.agent_logger.warning(
                    "Reflection enabled but ReflectionManager missing."
                )

        # --- 3. Plan Step (Use Reflection's Next Step) ---
        plan = None
        if (
            last_reflection
            and last_reflection.get("next_step")
            and last_reflection["next_step"].lower() != "none"
        ):
            next_step_from_reflection = last_reflection["next_step"]
            self.agent_logger.info(
                f"Using next step from reflection: {next_step_from_reflection}"
            )
            # Construct a plan dictionary compatible with the rest of the logic
            plan = {
                "next_step": next_step_from_reflection,
                "rationale": f"Based on reflection: {last_reflection.get('reason', 'Proceed as per reflection.')}",
                "suggested_tools": last_reflection.get("required_tools", []),
            }
            # Log it in reasoning log as well
            self.context.reasoning_log.append(
                f"Plan (from Reflection): {plan['rationale']} -> {plan['next_step']}"
            )
        elif not self.context.reflect_enabled:
            # Handle case where reflection is disabled
            self.agent_logger.debug("Reflection disabled, using default plan.")
            plan = {  # Default plan if reflection disabled
                "next_step": "Continue direct task execution.",  # Changed default step
                "rationale": "Direct execution mode, reflection disabled.",
                "suggested_tools": [],
            }
            self.context.reasoning_log.append(
                f"Plan (Default): {plan['rationale']} -> {plan['next_step']}"
            )
        else:
            self.agent_logger.info(
                "Reflection suggested 'None' or failed, no further plan generated."
            )
            # Plan remains None

        # Return both the last content generated during think/act and the derived plan
        return self.context.final_answer or last_content, plan
