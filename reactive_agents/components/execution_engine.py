from __future__ import annotations
import asyncio
import time
import traceback
from typing import (
    Dict,
    Any,
    Optional,
    Set,
    TYPE_CHECKING,
    Protocol,
    runtime_checkable,
    List,
    Tuple,
)
import json
from pydantic import BaseModel

from reactive_agents.common.types.status_types import TaskStatus
from reactive_agents.common.types.session_types import AgentSession
from reactive_agents.common.types.event_types import AgentStateEvent
from reactive_agents.common.types.agent_types import (
    EvaluationFormat,
)
from reactive_agents.prompts.agent_prompts import (
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_CONTEXT_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    EVALUATION_CONTEXT_PROMPT,
)

if TYPE_CHECKING:
    from reactive_agents.agents.base import Agent


@runtime_checkable
class ToolFeasibilityChecker(Protocol):
    async def check_tool_feasibility(self, task: str) -> Dict[str, Any]: ...


class AgentExecutionEngine:
    """
    Handles the execution of agent tasks, managing the main execution loop and
    coordinating between different components of the agent system.
    """

    def __init__(self, agent: "Agent"):
        """
        Initialize the execution engine with an agent instance.

        Args:
            agent: The Agent instance containing all necessary components and state for the agent execution.
        """
        # Import here to avoid circular imports
        from reactive_agents.components.task_executor import TaskExecutor

        self.agent = agent
        self.context = agent.context
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Initially not paused
        self._terminate_requested = False
        self._stop_requested = False
        self._current_async_task: Optional[asyncio.Task] = None
        # Initialize TaskExecutor
        self.task_executor = TaskExecutor(agent)
        # Track pending tool calls
        self._pending_tool_calls: List[Dict[str, Any]] = []
        self._tool_call_lock = asyncio.Lock()

    async def execute(
        self,
        initial_task: str,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Execute the agent's task, managing the main execution loop.

        Args:
            initial_task: The task description to execute.
            cancellation_event: Optional event to signal cancellation.

        Returns:
            A dictionary containing the final status, result, and other execution details.
        """
        # Initialize session
        self.context.session = AgentSession(
            initial_task=initial_task,
            current_task=initial_task,
            start_time=time.time(),
            task_status=TaskStatus.INITIALIZED,
            reasoning_log=[],
            task_progress=[],
            task_nudges=[],
            successful_tools=[],
            metrics={},
            completion_score=0.0,
            tool_usage_score=0.0,
            progress_score=0.0,
            answer_quality_score=0.0,
            llm_evaluation_score=0.0,
            instruction_adherence_score=0.0,
        )

        # Reset managers
        if self.context.metrics_manager:
            self.context.metrics_manager.reset()
        if self.context.reflection_manager:
            self.context.reflection_manager.reset()

        # Emit session start event
        self.context.emit_event(
            AgentStateEvent.SESSION_STARTED,
            {
                "initial_task": self.context.session.initial_task,
                "session_id": self.context.session.session_id,
            },
        )

        # Local state for run management
        last_error: Optional[str] = None
        rescoped_task: Optional[str] = None
        final_result_content: Optional[str] = None
        max_failures = self.context.max_task_retries
        failure_count = 0
        active_tasks: Set[asyncio.Task] = set()
        last_plan = None

        try:
            # Pre-run checks
            if not await self._check_dependencies():
                return await self._prepare_final_result()

            if self.context.check_tool_feasibility:
                check_method = getattr(self.agent, "check_tool_feasibility", None)
                if check_method:
                    feasibility = await check_method(initial_task)
                    if not feasibility["feasible"]:
                        self.context.session.task_status = TaskStatus.MISSING_TOOLS
                        return await self._prepare_final_result()

                    # Set min_required_tools in the session if available
                    if "required_tools" in feasibility:
                        self.context.session.min_required_tools = set(
                            feasibility["required_tools"]
                        )
                        self.agent.agent_logger.info(
                            f"Set min_required_tools in session: {self.context.session.min_required_tools}"
                        )
                else:
                    self.agent.agent_logger.warning(
                        "Tool feasibility check not available for this agent type."
                    )

            # Set status to running
            self.context.session.task_status = TaskStatus.RUNNING
            self.context.emit_event(
                AgentStateEvent.TASK_STATUS_CHANGED,
                {"previous_status": "initialized", "new_status": "running"},
            )

            # Main execution loop
            while await self._should_continue():
                # Check for cancellation event
                if cancellation_event and cancellation_event.is_set():
                    self.context.emit_event(
                        AgentStateEvent.CANCELLED,
                        {"message": "Agent cancelled by external event."},
                    )
                    self.context.session.task_status = TaskStatus.CANCELLED
                    return await self._prepare_final_result()

                if self._handle_control_signals():
                    break

                self.context.session.iterations += 1
                self.context.emit_event(
                    AgentStateEvent.ITERATION_STARTED,
                    {
                        "iteration": self.context.session.iterations,
                        "max_iterations": self.context.max_iterations,
                    },
                )

                # Execute iteration
                try:
                    self.agent.agent_logger.debug(
                        f"🔄 Starting Iteration - {self.context.session.iterations}/{self.context.max_iterations}"
                    )
                    iteration_result = await self._execute_iteration(
                        rescoped_task or self.context.session.current_task,
                        last_plan,
                    )
                    last_content, new_plan = iteration_result
                    last_plan = new_plan
                    if last_content:
                        final_result_content = last_content
                    if self.context.session.final_answer:
                        self.context.session.task_status = TaskStatus.COMPLETE
                        # Emit ITERATION_COMPLETED event before breaking
                        self.context.emit_event(
                            AgentStateEvent.ITERATION_COMPLETED,
                            {
                                "iteration": self.context.session.iterations,
                                "has_result": bool(last_content),
                                "has_plan": bool(new_plan),
                            },
                        )
                        break
                    # Emit ITERATION_COMPLETED event at end of iteration
                    self.context.emit_event(
                        AgentStateEvent.ITERATION_COMPLETED,
                        {
                            "iteration": self.context.session.iterations,
                            "has_result": bool(last_content),
                            "has_plan": bool(new_plan),
                        },
                    )
                except Exception as iter_error:
                    failure_count += 1
                    last_error = str(iter_error)
                    await self._handle_iteration_error(
                        iter_error, failure_count, max_failures
                    )

            # Post-execution processing
            self.context.session.summary = await self._generate_summary()
            self.context.session.evaluation = (
                await self._generate_goal_result_evaluation()
            )

        except Exception as run_error:
            return await self._handle_execution_error(run_error)

        finally:
            return await self._finalize_execution(
                rescoped_task,
                final_result_content,
                last_error,
                active_tasks,
            )

    async def _execute_iteration(
        self, task: str, last_plan: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Execute a single iteration of the agent's task.

        Args:
            task: The current task to execute.
            last_plan: The plan from the previous iteration, if any.

        Returns:
            Tuple of (content, plan) from the iteration.
        """
        # Log the current context state
        if self.context.agent_logger:
            self.context.agent_logger.debug("\n=== MODEL INPUT CONTEXT ===")
            self.context.agent_logger.debug(f"Current Task: {task}")
            self.context.agent_logger.debug(
                f"Last Plan: {json.dumps(last_plan, indent=2) if last_plan else 'None'}"
            )
            self.context.agent_logger.debug(
                f"System Prompt: {self.context.session.messages[0]['content'] if self.context.session.messages else 'None'}"
            )
            self.context.agent_logger.debug(
                f"Recent Messages: {json.dumps(self.context.session.messages[-3:], indent=2) if self.context.session.messages else 'None'}"
            )
            self.context.agent_logger.debug(
                f"Recent Reasoning Log: {json.dumps(self.context.session.reasoning_log[-3:], indent=2) if self.context.session.reasoning_log else 'None'}"
            )
            self.context.agent_logger.debug("========================\n")

        # Execute iteration using TaskExecutor
        result = await self._await_with_control(
            self.task_executor.execute_iteration(task)
        )

        if not result:
            return None, None

        # Extract content and plan from result
        think_act = result.get("think_act", {})
        reflection = result.get("reflection", {})
        plan = result.get("plan")

        # Get content from think_act result
        content = None
        if think_act and isinstance(think_act, dict):
            message = think_act.get("message", {})
            content = message.get("content")
        # Generate plan if not provided by TaskExecutor
        if not plan and reflection:
            plan = await self._generate_plan(reflection, last_plan)

        return content, plan

    async def _handle_iteration_error(
        self, error: Exception, failure_count: int, max_failures: int
    ) -> None:
        """
        Handle an error that occurred during an iteration.

        Args:
            error: The exception that occurred
            failure_count: Current number of failures
            max_failures: Maximum number of failures allowed
        """
        tb_str = traceback.format_exc()
        if hasattr(self.context, "agent_logger") and self.context.agent_logger:
            self.context.agent_logger.error(
                f"❌ Iteration {self.context.session.iterations} Error: {str(error)}\n{tb_str}"
            )

        # Add error to errors list with context if not already added
        # Check if this is a model provider error that was already handled
        is_model_provider_error = any(
            e.get("component") == "model_provider" and e.get("error") == str(error)
            for e in self.context.session.errors
        )

        if not is_model_provider_error:
            error_data = {
                "iteration": self.context.session.iterations,
                "error": str(error),
                "traceback": tb_str,
                "timestamp": time.time(),
                "failure_count": failure_count,
                "error_type": (
                    "critical" if failure_count >= max_failures else "non_critical"
                ),
            }
            self.context.session.errors.append(error_data)

        # Add error message to task progress
        error_message = (
            f"ERROR in iteration {self.context.session.iterations}: {str(error)}"
        )
        self.context.session.reasoning_log.append(error_message)

        # Set error in session if this is a critical error
        if failure_count >= max_failures:
            self.context.session.error = str(error)
            self.context.session.task_status = TaskStatus.ERROR

        self.context.emit_event(
            AgentStateEvent.ERROR_OCCURRED,
            {
                "error": "Iteration error",
                "details": str(error),
                "iteration": self.context.session.iterations,
                "failure_count": failure_count,
                "is_critical": failure_count >= max_failures,
            },
        )

    async def _handle_execution_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle an error that occurred during execution.

        Args:
            error: The exception that occurred

        Returns:
            Final result package with error information
        """
        tb_str = traceback.format_exc()
        if hasattr(self.context, "agent_logger") and self.context.agent_logger:
            self.context.agent_logger.error(
                f"Unhandled error during agent run: {error}\n{tb_str}"
            )

        self.context.emit_event(
            AgentStateEvent.ERROR_OCCURRED,
            {"error": "Unhandled error", "details": str(error)},
        )

        if hasattr(self.context, "session") and self.context.session:
            self.context.session.task_status = TaskStatus.ERROR
            self.context.session.evaluation = {
                "adherence_score": 0.0,
                "matches_intent": False,
                "explanation": f"Critical error: {error}",
                "strengths": [],
                "weaknesses": ["Agent run failed critically."],
            }

            self.context.emit_event(
                AgentStateEvent.TASK_STATUS_CHANGED,
                {
                    "previous_status": str(self.context.session.task_status),
                    "new_status": "error",
                },
            )

        return await self._prepare_final_result()

    async def _finalize_execution(
        self,
        rescoped_task: Optional[str],
        final_result_content: Optional[str],
        last_error: Optional[str],
        active_tasks: Set[asyncio.Task],
    ) -> Dict[str, Any]:
        """
        Finalize the execution and prepare the final result.

        Args:
            rescoped_task: The rescoped task if any
            final_result_content: The final content generated
            last_error: The last error if any
            active_tasks: Set of active tasks to clean up

        Returns:
            Final result package
        """
        # Cleanup tasks
        if active_tasks:
            if hasattr(self.context, "agent_logger") and self.context.agent_logger:
                self.context.agent_logger.debug(
                    f"Cleaning up {len(active_tasks)} background tasks..."
                )
            for task in active_tasks:
                if not task.done():
                    task.cancel()
            active_tasks.clear()

        # Get session data safely
        current_session_status = TaskStatus.ERROR
        session_final_answer = None
        session_evaluation = {}

        if hasattr(self.context, "session") and self.context.session:
            current_session_status = self.context.session.task_status
            session_final_answer = self.context.session.final_answer
            session_evaluation = self.context.session.evaluation or {}
            self.context.session.end_time = time.time()
            self.context.session.error = (
                last_error if current_session_status == TaskStatus.ERROR else None
            )

            # Set final answer in session if we have one
            if final_result_content and not session_final_answer:
                self.context.session.final_answer = final_result_content

        # Determine final result
        result_to_use = (
            final_result_content
            or session_final_answer
            or f"Agent failed critically: {last_error}"
        )

        # Update workflow context
        if self.context.workflow_manager:
            self.context.workflow_manager.update_context(
                current_session_status,
                result=result_to_use,
                adherence_score=session_evaluation.get("adherence_score"),
                matches_intent=session_evaluation.get("matches_intent"),
                rescoped=(rescoped_task is not None),
                error=(
                    last_error if current_session_status == TaskStatus.ERROR else None
                ),
            )

        # Finalize metrics
        if self.context.metrics_manager:
            self.context.metrics_manager.finalize_run_metrics()
            metrics_data = self.context.metrics_manager.get_metrics()
            if hasattr(self.context, "session") and self.context.session:
                self.context.session.metrics = metrics_data

        # Save memory
        if self.context.memory_manager:
            if hasattr(self.context, "session") and self.context.session:
                self.context.memory_manager.update_session_history(self.context.session)
                self.context.memory_manager.save_memory()

        # Emit session end event
        elapsed_time = 0
        if hasattr(self.context, "session") and self.context.session:
            elapsed_time = time.time() - self.context.session.start_time

        self.context.emit_event(
            AgentStateEvent.SESSION_ENDED,
            {
                "session_id": getattr(self.context.session, "session_id", None),
                "final_status": str(current_session_status),
                "elapsed_time": elapsed_time,
                "iterations": getattr(self.context.session, "iterations", 0),
            },
        )

        return await self._prepare_final_result()

    def _handle_control_signals(self) -> bool:
        """
        Handle pause, resume, and terminate signals.

        Returns:
            True if execution should stop, False otherwise.
        """
        if self._terminate_requested:
            self.context.emit_event(
                AgentStateEvent.TERMINATED,
                {"message": "Agent terminated by user request."},
            )
            self.context.session.task_status = TaskStatus.CANCELLED
            return True

        if self._paused:
            self.context.emit_event(
                AgentStateEvent.PAUSED, {"message": "Agent is now paused."}
            )
            asyncio.create_task(self._pause_event.wait())
            self.context.emit_event(
                AgentStateEvent.RESUMED, {"message": "Agent resumed by user."}
            )

        if self._stop_requested:
            self.context.emit_event(
                AgentStateEvent.STOPPED,
                {"message": "Agent stopped gracefully by user request."},
            )
            self.context.session.task_status = TaskStatus.CANCELLED
            return True

        return False

    async def _await_with_control(self, coro):
        """Await a coroutine, handling pause/terminate signals."""
        self._current_async_task = asyncio.create_task(coro)
        try:
            result = await self._current_async_task
        except asyncio.CancelledError:
            raise
        finally:
            self._current_async_task = None

        if self._terminate_requested:
            raise Exception("Terminated")
        if self._paused:
            await self._pause_event.wait()

        return result

    async def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a tool call with pause/stop/terminate handling."""
        async with self._tool_call_lock:
            if self._terminate_requested:
                raise Exception("Cannot execute tool call: Agent terminated")
            if self._stop_requested:
                raise Exception("Cannot execute tool call: Agent stopped")
            if self._paused:
                self._pending_tool_calls.append(tool_call)
                await self._pause_event.wait()
                # Remove from pending after resuming, if it's still there
                if tool_call in self._pending_tool_calls:
                    self._pending_tool_calls.remove(tool_call)

            # Execute the tool call through the task executor
            return await self._await_with_control(
                self.task_executor.execute_tool_call(tool_call)
            )

    async def _should_continue(self) -> bool:
        """Determine if the execution loop should continue."""
        return self.task_executor.should_continue()

    async def _check_dependencies(self) -> bool:
        """Check if all dependencies are met."""
        return self.task_executor._check_dependencies()

    async def _generate_reflection(
        self, think_act_result: Any
    ) -> Optional[Dict[str, Any]]:
        """Generate reflection on the current state."""
        if not self.context.reflection_manager:
            return None

        reflection_input = think_act_result
        if isinstance(reflection_input, str):
            reflection_input = {"message": {"content": reflection_input}}

        return await self._await_with_control(
            self.context.reflection_manager.generate_reflection(reflection_input)
        )

    async def _generate_plan(
        self,
        reflection: Optional[Dict[str, Any]],
        last_plan: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Generate a plan based on reflection and last plan."""
        if reflection and reflection.get("next_step"):
            return {
                "next_step": reflection["next_step"],
                "rationale": f"Based on reflection: {reflection.get('reason', 'Proceed as per reflection.')}",
                "suggested_tools": reflection.get("required_tools", []),
            }
        elif not self.context.reflect_enabled:
            return {
                "next_step": "Continue direct task execution.",
                "rationale": "Direct execution mode, reflection disabled.",
                "suggested_tools": [],
            }
        return None

    async def _generate_summary(self) -> str:
        """Generates a summary of the agent's actions using ToolManager history."""
        if self.context.agent_logger:
            self.context.agent_logger.debug("📜 Generating execution summary...")
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
                "task": self.context.session.initial_task,
                "tools_used_count": len(tools_used_names),
                "tools_used_names": tools_used_names,
                "total_actions": len(tool_history),
                "recent_actions": history_for_prompt,
                "final_status": str(self.context.session.task_status),
                "final_result_preview": str(self.context.session.final_answer or "N/A")[
                    :200
                ]
                + "...",
            }

            # Use centralized prompts
            summary_prompt = SUMMARY_CONTEXT_PROMPT.format(
                summary_context_json=json.dumps(summary_context, indent=2, default=str)
            )
            response = await self.agent.model_provider.get_completion(
                system=SUMMARY_SYSTEM_PROMPT, prompt=summary_prompt
            )

            if response and response.get("response"):
                # Check if thinking is enabled and extract thinking content
                thinking_enabled = (
                    self.context.model_provider_options.get("think", False)
                    if self.context.model_provider_options
                    else False
                )
                response_text = response["response"].strip()

                if response_text.startswith("<think>"):
                    # Extract thinking from summary response
                    think_start = response_text.find("<think>")
                    think_end = response_text.find("</think>")

                    if (
                        think_start != -1
                        and think_end != -1
                        and think_end > think_start
                    ):
                        thinking_content = response_text[
                            think_start + 7 : think_end
                        ].strip()
                        # Store thinking with summary context
                        if hasattr(self.context, "session") and thinking_content:
                            thinking_entry = {
                                "timestamp": time.time(),
                                "call_context": "summary_generation",
                                "thinking": thinking_content,
                            }
                            self.context.session.thinking_log.append(thinking_entry)
                            if self.context.agent_logger:
                                self.context.agent_logger.debug(
                                    f"Stored summary thinking: {thinking_content[:100]}..."
                                )

                        # Remove thinking tags from response
                        response_text = (
                            response_text[:think_start] + response_text[think_end + 8 :]
                        )
                        response_text = response_text.strip()

                return response_text
            else:
                if self.context.agent_logger:
                    self.context.agent_logger.warning(
                        "Summary generation failed: No response from model."
                    )
                return f"Agent completed task '{self.context.session.initial_task[:50]}...' with status {self.context.session.task_status} after {self.context.session.iterations} iterations, using {len(tools_used_names)} tools."

        except Exception as e:
            if self.context.agent_logger:
                self.context.agent_logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {e}"

    def calculate_completion_score(self) -> float:
        """Calculate a more comprehensive completion score."""
        session = self.context.session
        if not session:
            return 0.0

        score_components = []

        # 1. Tool Usage Score
        if session.min_required_tools:
            successful_tools_set = set(session.successful_tools)
            tool_score = len(
                session.min_required_tools.intersection(successful_tools_set)
            ) / len(session.min_required_tools)
        else:
            tool_score = 1.0
        session.tool_usage_score = tool_score
        score_components.append(("tool_usage", tool_score, session.tool_usage_weight))

        # 2. Task Progress Score
        # Based on reasoning log and task progress
        progress_score = min(
            1.0, len(session.reasoning_log) / 5
        )  # Consider first 5 steps
        session.progress_score = progress_score
        score_components.append(("progress", progress_score, session.progress_weight))

        # 3. Final Answer Quality
        if session.final_answer:
            # Basic quality checks
            answer = session.final_answer.strip()
            if not answer:
                answer_score = 0.0
            elif answer.lower().startswith(("error", "failed", "could not")):
                answer_score = 0.3
            else:
                # Check against success criteria if available
                if (
                    session.success_criteria
                    and session.success_criteria.required_answer_content
                ):
                    required_content = set(
                        session.success_criteria.required_answer_content
                    )
                    answer_words = set(answer.lower().split())
                    content_score = len(
                        required_content.intersection(answer_words)
                    ) / len(required_content)
                    answer_score = max(
                        0.5, content_score
                    )  # Minimum 0.5 if answer exists
                else:
                    answer_score = 0.8  # Default score for valid answer
        else:
            answer_score = 0.0
        session.answer_quality_score = answer_score
        score_components.append(("answer", answer_score, session.answer_quality_weight))

        # Calculate weighted average
        final_score = sum(score * weight for _, score, weight in score_components)
        return final_score

    async def _generate_goal_result_evaluation(self) -> Dict[str, Any]:
        """Evaluates how well the final result matches the initial task goal."""
        if self.context.agent_logger:
            self.context.agent_logger.info("🔎 Generating goal VS result evaluation...")
        default_eval = {
            "adherence_score": 0.0,
            "matches_intent": False,
            "explanation": "Evaluation could not be performed.",
            "strengths": [],
            "weaknesses": ["Evaluation failed."],
            "instruction_adherence": {
                "score": 0.0,
                "adhered_instructions": [],
                "missed_instructions": [],
                "improvement_suggestions": [],
            },
        }

        if not self.context.tool_manager:
            if self.context.agent_logger:
                self.context.agent_logger.warning(
                    "Cannot evaluate goal: ToolManager missing."
                )
            return default_eval

        tool_history = self.context.tool_manager.tool_history
        if not tool_history and not self.context.session.final_answer:
            if self.context.agent_logger:
                self.context.agent_logger.info(
                    "Evaluation: No actions taken, score 0.0."
                )
            return {
                "adherence_score": 0.0,
                "matches_intent": False,
                "explanation": "No actions were taken and no final answer provided.",
                "strengths": [],
                "weaknesses": ["No progress made."],
                "instruction_adherence": {
                    "score": 0.0,
                    "adhered_instructions": [],
                    "missed_instructions": ["No actions taken to follow instructions"],
                    "improvement_suggestions": [
                        "Take actions to follow the provided instructions"
                    ],
                },
            }

        try:
            tool_names_used = list(
                set(t.get("name", "unknown") for t in tool_history if t.get("name"))
            )
            final_result_str = (
                self.context.session.final_answer
                or "No explicit final textual answer provided by agent."
            )
            eval_context = {
                "original_goal": self.context.session.initial_task,
                "instructions": self.context.instructions,
                "final_result": (
                    final_result_str[:3000] + "..."
                    if len(final_result_str) > 3000
                    else final_result_str
                ),
                "final_status": str(self.context.session.task_status),
                "tools_used": tool_names_used,
                "action_summary": self.context.session.summary,
                "reasoning_log": self.context.session.reasoning_log[-5:],
                "success_criteria": (
                    self.context.session.success_criteria.dict()
                    if self.context.session.success_criteria
                    else None
                ),
            }

            # Get the deterministic completion score
            completion_score = self.calculate_completion_score()
            self.context.session.completion_score = completion_score

            if self.context.agent_logger:
                self.context.agent_logger.info(
                    f"Using deterministic completion score: {completion_score:.2f}"
                )

            # Get LLM evaluation
            eval_prompt = EVALUATION_CONTEXT_PROMPT.format(
                eval_context_json=json.dumps(eval_context, indent=2, default=str)
            )
            response = await self.agent.model_provider.get_completion(
                system=EVALUATION_SYSTEM_PROMPT,
                prompt=eval_prompt,
                format=(
                    EvaluationFormat.model_json_schema()
                    if self.agent.model_provider.name == "ollama"
                    else "json"
                ),
            )

            if not response or not response.get("response"):
                if self.context.agent_logger:
                    self.context.agent_logger.warning(
                        "Goal evaluation failed: No response from model."
                    )
                return {
                    "adherence_score": completion_score,
                    "matches_intent": completion_score > 0.5,
                    "explanation": "Basic evaluation based on completion score; LLM evaluation failed.",
                    "strengths": ["Used required tools proportionally to score."],
                    "weaknesses": ["LLM evaluation call failed."],
                    "instruction_adherence": {
                        "score": completion_score,
                        "adhered_instructions": [],
                        "missed_instructions": [
                            "Could not evaluate instruction adherence"
                        ],
                        "improvement_suggestions": [
                            "Retry evaluation to assess instruction adherence"
                        ],
                    },
                }

            try:
                eval_data = response["response"]
                parsed_eval = (
                    json.loads(eval_data) if isinstance(eval_data, str) else eval_data
                )
                validated_eval_dict = EvaluationFormat(**parsed_eval).dict()

                # Store LLM evaluation score
                llm_score = validated_eval_dict.get("adherence_score", 0.0)
                self.context.session.llm_evaluation_score = llm_score

                # Get instruction adherence score
                instruction_score = validated_eval_dict.get(
                    "instruction_adherence", {}
                ).get("score", 0.0)
                self.context.session.instruction_adherence_score = instruction_score

                # Calculate final adherence score as weighted average
                final_adherence_score = (
                    llm_score * self.context.session.llm_evaluation_weight
                    + completion_score * self.context.session.completion_score_weight
                    + instruction_score * 0.2  # Add instruction adherence weight
                )

                # Update evaluation with combined scores
                validated_eval_dict["adherence_score"] = final_adherence_score
                validated_eval_dict["completion_score"] = completion_score
                validated_eval_dict["llm_score"] = llm_score
                validated_eval_dict["instruction_score"] = instruction_score
                validated_eval_dict["tool_usage_score"] = (
                    self.context.session.tool_usage_score
                )
                validated_eval_dict["progress_score"] = (
                    self.context.session.progress_score
                )
                validated_eval_dict["answer_quality_score"] = (
                    self.context.session.answer_quality_score
                )

                if self.context.agent_logger:
                    self.context.agent_logger.info(
                        f"📄 Goal evaluation result: Adherence Score={final_adherence_score:.2f} "
                        f"(LLM: {llm_score:.2f}, Completion: {completion_score:.2f}, "
                        f"Instructions: {instruction_score:.2f}), "
                        f"Matches Intent={validated_eval_dict.get('matches_intent')}"
                    )
                self.context.session.reasoning_log.append(
                    f"Goal Adherence Evaluation (Score: {final_adherence_score:.2f}): {validated_eval_dict.get('explanation')}"
                )
                return validated_eval_dict

            except (json.JSONDecodeError, TypeError) as e:
                if self.context.agent_logger:
                    self.context.agent_logger.error(
                        f"Error parsing evaluation JSON: {e}\nResponse: {response.get('response')}"
                    )
                return default_eval | {"explanation": f"Error parsing evaluation: {e}"}
            except Exception as e:
                if self.context.agent_logger:
                    self.context.agent_logger.error(
                        f"Error validating evaluation: {e}\nData: {parsed_eval}"
                    )
                return default_eval | {
                    "explanation": f"Error validating evaluation: {e}"
                }

        except Exception as e:
            if self.context.agent_logger:
                self.context.agent_logger.error(f"Error during goal evaluation: {e}")
            return default_eval | {"explanation": f"Error during evaluation: {e}"}

    async def _rescope_goal(
        self, original_task: str, error_context: str
    ) -> Dict[str, Any]:
        """Attempt to rescope the goal based on error context."""
        # This will be implemented in the TaskRescoper service
        return {
            "rescoped_task": None,
            "explanation": "Task rescoping not implemented",
            "expected_tools": [],
        }

    async def _prepare_final_result(self) -> Dict[str, Any]:
        """Prepares the final result of the agent's execution using data from the session."""
        if self.context.agent_logger:
            self.context.agent_logger.info("📝 Preparing final result...")

        # Get data from session
        session = self.context.session
        if not session:
            return {
                "status": "error",
                "result": "No session data available",
                "error": "Session data missing",
            }

        # Calculate elapsed time
        elapsed_time = time.time() - session.start_time

        # Get critical errors from errors list
        critical_errors = [
            error for error in session.errors if error.get("error_type") == "critical"
        ]
        critical_error = critical_errors[0] if critical_errors else None

        # Prepare the final result
        final_result = {
            # Core execution results
            "status": session.task_status.value,
            "result": session.final_answer or "No final answer provided",
            "summary": session.summary,
            "error": critical_error.get("error") if critical_error else session.error,
            # Performance metrics
            "elapsed_time": elapsed_time,
            "iterations": session.iterations,
            "metrics": {
                "tool_calls": session.metrics.get("tool_calls", 0),
                "tool_errors": session.metrics.get("tool_errors", 0),
                "tokens": session.metrics.get("tokens", {}),
                "model_calls": session.metrics.get("model_calls", 0),
                "cache": session.metrics.get("cache", {}),
                "latency": session.metrics.get("latency", {}),
            },
            # Evaluation and scoring
            "evaluation": {
                "adherence_score": (session.evaluation or {}).get(
                    "adherence_score", 0.0
                ),
                "completion_score": session.completion_score,
                "tool_usage_score": session.tool_usage_score,
                "progress_score": session.progress_score,
                "answer_quality_score": session.answer_quality_score,
                "llm_evaluation_score": session.llm_evaluation_score,
                "strengths": (session.evaluation or {}).get("strengths", []),
                "weaknesses": (session.evaluation or {}).get("weaknesses", []),
                "explanation": (session.evaluation or {}).get("explanation", ""),
                "matches_intent": (session.evaluation or {}).get(
                    "matches_intent", False
                ),
            },
            # Task information
            "task": {
                "initial": session.initial_task,
                "current": session.current_task,
                "progress": session.task_progress,
                "nudges": session.task_nudges,
                "required_tools": (
                    list(session.min_required_tools)
                    if session.min_required_tools
                    else []
                ),
                "successful_tools": (
                    list(set(session.successful_tools))
                    if session.successful_tools
                    else []
                ),
                "success_criteria": (
                    session.success_criteria.model_dump()
                    if session.success_criteria
                    else None
                ),
            },
            # Session Execution context
            "session": {
                "id": session.session_id,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "completion_threshold": self.context.min_completion_score,
                "max_iterations": self.context.max_iterations,
                "reflect_enabled": self.context.reflect_enabled,
                "tool_use_enabled": self.context.tool_use_enabled,
                "use_memory_enabled": self.context.use_memory_enabled,
                "collect_metrics_enabled": self.context.collect_metrics_enabled,
            },
            # Logs and reasoning
            "logs": {
                "reasoning": session.reasoning_log,
                "thinking": session.thinking_log,  # Include thinking log in final result
                "tool_history": (
                    self.context.tool_manager.tool_history
                    if self.context.tool_manager
                    else []
                ),
                "errors": {
                    "critical": (
                        critical_error.get("error") if critical_error else session.error
                    ),
                    "errors": session.errors,
                    "error_count": len(session.errors),
                },
            },
        }

        # Add rescoped task info if available
        if session.current_task != session.initial_task:
            final_result["task"]["rescoped"] = True
            final_result["task"]["original"] = session.initial_task

        return final_result

    # Control methods
    async def pause(self):
        """Request the agent to pause at the next safe point."""
        if not self._paused:
            self._paused = True
            self._pause_event.clear()
            self.context.emit_event(
                AgentStateEvent.PAUSE_REQUESTED, {"message": "Pause requested by user."}
            )
            if self.context.agent_logger:
                self.context.agent_logger.info("Agent paused.")

    async def resume(self):
        """Resume the agent if it is paused."""
        if self._paused:
            self._paused = False
            self._pause_event.set()
            self.context.emit_event(
                AgentStateEvent.RESUME_REQUESTED,
                {"message": "Resume requested by user."},
            )
            # Execute any pending tool calls
            pending_calls = self._pending_tool_calls.copy()
            self._pending_tool_calls.clear()
            for tool_call in pending_calls:
                try:
                    await self._execute_tool_call(tool_call)
                except Exception as e:
                    if self.context.agent_logger:
                        self.context.agent_logger.error(
                            f"Error executing pending tool call: {e}"
                        )
            if self.context.agent_logger:
                self.context.agent_logger.info("Agent resumed.")

    async def terminate(self):
        """Request the agent to terminate as soon as possible."""
        if not self._terminate_requested:
            self._terminate_requested = True
            self.context.emit_event(
                AgentStateEvent.TERMINATE_REQUESTED,
                {"message": "Terminate requested by user."},
            )
            if self._current_async_task:
                self._current_async_task.cancel()
            self._pause_event.set()
            # Clear pending tool calls
            self._pending_tool_calls.clear()

            self.context.session.task_status = TaskStatus.CANCELLED

            if self.context.agent_logger:
                self.context.agent_logger.info("Agent terminated.")

    async def stop(self):
        """Request the agent to gracefully stop after the current step."""
        if not self._stop_requested:
            self._stop_requested = True
            self.context.emit_event(
                AgentStateEvent.STOP_REQUESTED,
                {"message": "Stop requested by user (graceful)."},
            )
            if self._current_async_task:
                self._current_async_task.cancel()

            self.context.session.task_status = TaskStatus.CANCELLED
            self._pause_event.set()
            # Clear pending tool calls
            self._pending_tool_calls.clear()

            if self.context.agent_logger:
                self.context.agent_logger.info("Agent stopped gracefully.")
