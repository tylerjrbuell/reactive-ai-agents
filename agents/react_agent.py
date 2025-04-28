from __future__ import annotations
import json
import traceback
import asyncio
import os
from datetime import datetime
from enum import Enum, auto
from typing import List, Any, Optional, Dict, Set, Union, Tuple, Callable
import time
import random

from pydantic import BaseModel, Field
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

# Import shared types from the new location
from common.types import TaskStatus, AgentMemory


class ReactAgent(Agent):
    """
    A ReAct-style agent that uses reflection and planning within an AgentContext.
    """

    # --- Pydantic models moved inside the class --

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

    def __init__(self, context: AgentContext):
        """
        Initializes the ReactAgent with a pre-configured AgentContext.
        React-specific configurations should be set within the context instance before passing it here.

        Args:
            context: The AgentContext instance holding all configuration, state, and managers.
        """
        super().__init__(context)
        self.agent_logger.info(f"ReactAgent '{self.context.agent_name}' initialized.")

    async def run(
        self, initial_task: str, cancellation_event: Optional[asyncio.Event] = None
    ) -> Dict[str, Any]:
        """
        Run the ReAct agent loop for the given task.

        Args:
            initial_task: The task description.
            cancellation_event: An optional asyncio.Event to signal cancellation.

        Returns:
            A dictionary containing the final status, result, and other execution details.
        """
        # 1. --- Initialization and Context Setup ---
        self.context.start_time = time.time()
        self.context.initial_task = initial_task
        self.context.current_task = initial_task
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

        self.agent_logger.info(f"ReactAgent run starting for task: {initial_task}")

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
                feasibility = await self.check_tool_feasibility(initial_task)
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

                current_task_for_iteration = rescoped_task or self.context.current_task

                try:
                    # --- Run the full React iteration (Think/Act -> Reflect -> Plan) ---
                    # This method now handles internal logic and returns final content for the iter, or None
                    iteration_content = await self._run_task_iteration(
                        task=current_task_for_iteration
                    )

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
            last_reflection = self.context.reflection_manager.get_last_reflection()
            if last_reflection:
                score = last_reflection.get("completion_score", 0.0)
                # Use initial required tools from context if available
                min_req_tools = self.context.min_required_tools
                # Get completed tools from the current reflection
                completed_tools = set(last_reflection.get("completed_tools", []))

                # Check score and tool completion
                score_met = score >= self.context.min_completion_score
                # Check tool completion only if min_required_tools were set
                if min_req_tools is not None:
                    tools_completed = min_req_tools.issubset(completed_tools)
                    self.agent_logger.info(
                        f"required tools: {min_req_tools}, completed tools: {completed_tools}"
                    )
                    self.agent_logger.info(
                        f"score_met: {score_met}, tools_completed: {tools_completed}"
                    )
                    if score_met and tools_completed:
                        self.agent_logger.info(
                            f"Stopping loop: Reflection score {score:.2f} meets threshold ({self.context.min_completion_score:.2f}) AND initial required tools ({min_req_tools}) completed."
                        )
                        self.context.task_status = TaskStatus.COMPLETE
                        return False
                elif score_met:
                    # If initial tools weren't set (e.g., feasibility check off/failed),
                    # fall back to stopping based on score alone.
                    self.agent_logger.info(
                        f"Stopping loop: Reflection score {score:.2f} meets threshold ({self.context.min_completion_score:.2f}). (Initial required tools not available for check)."
                    )
                    self.context.task_status = TaskStatus.COMPLETE
                    return False

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
                task=task, available_tools=list(available_tools)
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
                "task": self.context.current_task,
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

    async def _run_task_iteration(self, task: str) -> Optional[str]:
        """
        Executes one iteration of the ReAct loop: Think/Act -> Reflect -> Plan.
        Overrides the base Agent method.
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
            return self.context.final_answer

        if think_act_result_dict is None:
            self.agent_logger.warning("Think/Act step failed to produce a result.")
            return None

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
                        f"Reflection: Score={reflection['completion_score']:.2f}, Reason={reflection['reason']}"
                    )
                else:
                    self.agent_logger.warning("Reflection step failed.")
            else:
                self.agent_logger.warning(
                    "Reflection enabled but ReflectionManager missing."
                )

        # --- 3. Plan Step ---
        plan = None
        if self.context.reflect_enabled:
            self.agent_logger.info(" Planning...")
            plan = await self._plan()
            if plan and plan.get("next_step") and plan.get("rationale"):
                plan_guidance = f"{plan['rationale']}, Therefore: {plan['next_step']}"
                self.context.current_task = plan_guidance
                self.agent_logger.debug(
                    f"Added plan guidance to messages: {plan_guidance}"
                )

        return self.context.final_answer or last_content
