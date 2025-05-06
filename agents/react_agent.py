from __future__ import annotations
import json
import traceback
import asyncio
from typing import List, Any, Optional, Dict, Set, Tuple
import time

from pydantic import BaseModel, Field, model_validator
from prompts.agent_prompts import (
    MISSING_TOOLS_PROMPT,
    TOOL_FEASIBILITY_CONTEXT_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_CONTEXT_PROMPT,
    RESCOPE_SYSTEM_PROMPT,
    RESCOPE_CONTEXT_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    EVALUATION_CONTEXT_PROMPT,
)
from agents.base import Agent
from context.agent_context import AgentContext, AgentSession
from agent_mcp.client import MCPClient
from context.agent_observer import AgentStateEvent
from context.agent_events import AgentEventManager

# Import TaskStatus directly from the original module to avoid conflicts
from common.types import TaskStatus

# Import confirmation types
from common.types import (
    ConfirmationCallbackProtocol,
)


# --- Agent Configuration Model ---
class ReactAgentConfig(BaseModel):
    # Required parameters
    agent_name: str = Field(description="Name of the agent.")
    provider_model_name: str = Field(description="Name of the LLM provider and model.")

    # Optional parameters
    role: Optional[str] = Field(
        default="Task Executor", description="Role of the agent."
    )
    mcp_client: Optional[MCPClient] = Field(
        default=None, description="An initialized MCPClient instance."
    )
    min_completion_score: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Minimum score for task completion evaluation.",
    )
    instructions: Optional[str] = Field(
        default="Solve the given task.",
        description="High-level instructions for the agent.",
    )
    max_iterations: Optional[int] = Field(
        default=10, description="Maximum number of iterations allowed."
    )
    reflect_enabled: Optional[bool] = Field(
        default=True, description="Whether reflection mechanism is enabled."
    )
    log_level: Optional[str] = Field(
        default="info",
        description="Logging level ('debug', 'info', 'warning', 'error').",
    )
    initial_task: Optional[str] = Field(
        default=None,
        description="The initial task description (can also be passed to run).",
    )
    tool_use_enabled: Optional[bool] = Field(
        default=True, description="Whether the agent can use tools."
    )
    custom_tools: Optional[List[Any]] = Field(
        default_factory=list,
        description="List of custom tool instances to use with the agent.",
    )
    use_memory_enabled: Optional[bool] = Field(
        default=True, description="Whether the agent uses long-term memory."
    )
    collect_metrics_enabled: Optional[bool] = Field(
        default=True, description="Whether to collect performance metrics."
    )
    check_tool_feasibility: Optional[bool] = Field(
        default=True, description="Whether to check tool feasibility before starting."
    )
    enable_caching: Optional[bool] = Field(
        default=True, description="Whether to enable LLM response caching."
    )
    confirmation_callback: Optional[ConfirmationCallbackProtocol] = Field(
        default=None,
        description="Callback for confirming tool use. Can return bool or (bool, feedback).",
    )
    confirmation_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for tool confirmation behavior. If None, defaults will be used.",
    )
    # Store extra kwargs passed, e.g. for specific context managers
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to AgentContext.",
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
        # Process custom tools to ensure they're properly wrapped
        processed_tools = self._process_custom_tools(config.custom_tools)

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
            tools=processed_tools,  # Pass processed tools to context
            use_memory_enabled=config.use_memory_enabled,
            collect_metrics_enabled=config.collect_metrics_enabled,
            check_tool_feasibility=config.check_tool_feasibility,
            enable_caching=config.enable_caching,
            confirmation_callback=config.confirmation_callback,
            confirmation_config=config.confirmation_config,
            **config.kwargs,
        )
        super().__init__(context)
        # Store the config for potential reference, though context holds the state
        self.config = config

        # Initialize event manager for subscription interface
        self._event_manager = AgentEventManager(self.context.state_observer)

    def _process_custom_tools(self, tools):
        """
        Process custom tools to ensure they match the ToolProtocol interface.

        Args:
            tools: List of tools, which could be functions decorated with @tool

        Returns:
            List of tools that all comply with the ToolProtocol interface
        """
        from tools.base import Tool
        from tools.abstractions import ToolResult

        processed_tools = []

        for tool in tools:
            # Skip None values
            if tool is None:
                continue

            # If it's already a proper Tool class instance, use it as is
            if isinstance(tool, Tool):
                processed_tools.append(tool)
            # If it has a tool_definition attribute (likely a decorated function)
            elif hasattr(tool, "tool_definition"):
                # Create a wrapper class that implements the Tool interface
                class DecoratedFunctionWrapper(Tool):
                    # Use the function's attributes
                    name = tool.__name__
                    tool_definition = tool.tool_definition

                    def __init__(self, func):
                        self.func = func

                    async def use(self, params):
                        # Call the original function
                        result = await self.func(**params)
                        return ToolResult(result)

                # Create a wrapper instance and add it
                processed_tools.append(DecoratedFunctionWrapper(tool))
            # Otherwise it's not a compatible tool
            else:
                raise ValueError(
                    f"Custom tool {tool} is not compatible with ToolProtocol"
                )

        return processed_tools

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
        # --- Use initial_task from run() if provided, otherwise context cannot provide it anymore ---
        # Config might hold an initial task if provided at agent init, but run() arg takes precedence.
        current_initial_task = initial_task
        if not current_initial_task:
            # Maybe fetch from config if stored there? For now, raise error.
            # config_initial_task = getattr(self.config, 'initial_task', None)
            # current_initial_task = config_initial_task
            # if not current_initial_task:
            raise ValueError(
                # "An initial task must be provided either during agent config or when calling run()."
                "An initial task must be provided when calling run()."
            )

        # 1. --- Initialization and Context Setup ---
        # Create a new session for this run
        self.context.session = AgentSession(
            initial_task=current_initial_task,
            current_task=current_initial_task,
            start_time=time.time(),  # Override default start time
        )
        # Reset metrics manager if it exists
        if self.context.metrics_manager:
            self.context.metrics_manager.reset()
        # Reset reflection manager if it exists
        if self.context.reflection_manager:
            # Reset internal state if needed, reflections might be loaded from memory later
            self.context.reflection_manager.reset()

        # Emit session start event
        self.context.emit_event(
            AgentStateEvent.SESSION_STARTED,
            {
                "initial_task": self.context.session.initial_task,
                "session_id": self.context.session.session_id,
            },
        )

        # Log the correct initial task from the session
        self.agent_logger.info(
            f"ReactAgent run starting for task: {self.context.session.initial_task}"
        )

        # Local state for run management (some might move to session later if needed)
        last_error: Optional[str] = None
        rescoped_task: Optional[str] = None  # Keep local? Or move to session?
        final_result_content: Optional[str] = None  # Keep local to build final result
        max_failures = self.context.max_task_retries
        failure_count = 0
        active_tasks: Set[asyncio.Task] = set()
        last_plan = None  # Keep local for loop guidance injection

        try:
            # 2. --- Pre-run Checks (Dependencies, Tool Feasibility) ---
            if not self._check_dependencies():
                # Emit error event
                self.context.emit_event(
                    AgentStateEvent.ERROR_OCCURRED,
                    {
                        "error": "Dependencies not met",
                        "details": "Agent dependencies check failed",
                    },
                )
                # Prepare result using session status (rescoped_task is None here)
                return self._prepare_final_result(rescoped_task=None)

            # Use session's initial_task for feasibility check
            if self.context.check_tool_feasibility:
                feasibility = await self.check_tool_feasibility(
                    self.context.session.initial_task
                )
                if feasibility and feasibility.get("required_tools") is not None:
                    self.context.session.min_required_tools = set(
                        feasibility["required_tools"]
                    )
                    self.agent_logger.debug(
                        f"Stored initial required tools in session: {self.context.session.min_required_tools}"
                    )

                if not feasibility["feasible"]:
                    self.agent_logger.warning(
                        f"Missing required tools: {feasibility['missing_tools']}"
                    )
                    self.context.session.task_status = TaskStatus.MISSING_TOOLS
                    self.context.session.reasoning_log.append(
                        f"Cannot complete task: Missing Tools - {feasibility.get('explanation', '')}"
                    )

                    # Emit missing tools event
                    self.context.emit_event(
                        AgentStateEvent.ERROR_OCCURRED,
                        {
                            "error": "Missing required tools",
                            "missing_tools": feasibility["missing_tools"],
                            "explanation": feasibility.get("explanation", ""),
                        },
                    )

                    if self.context.workflow_manager:
                        self.context.workflow_manager.update_context(
                            TaskStatus.MISSING_TOOLS,
                            missing_tools=feasibility["missing_tools"],
                            explanation=feasibility["explanation"],
                        )
                    # Prepare result using session status (rescoped_task is None)
                    return self._prepare_final_result(
                        rescoped_task=None, feasibility=feasibility
                    )

            # 3. --- Set Status to Running ---
            self.context.session.task_status = TaskStatus.RUNNING

            # Emit status changed event
            self.context.emit_event(
                AgentStateEvent.TASK_STATUS_CHANGED,
                {"previous_status": "initialized", "new_status": "running"},
            )

            if self.context.workflow_manager:
                self.context.workflow_manager.update_context(TaskStatus.RUNNING)

            # 4. --- Execution Loop ---
            while self._should_continue():
                if cancellation_event and cancellation_event.is_set():
                    self.agent_logger.info("Task execution cancelled by user.")
                    self.context.session.task_status = TaskStatus.CANCELLED

                    # Emit cancellation event
                    self.context.emit_event(
                        AgentStateEvent.TASK_STATUS_CHANGED,
                        {"previous_status": "running", "new_status": "cancelled"},
                    )

                    break

                self.context.session.iterations += 1
                self.agent_logger.info(
                    f"🔄 ITERATION {self.context.session.iterations}/{self.context.max_iterations or 'unlimited'}"
                )

                # Emit iteration started event
                self.context.emit_event(
                    AgentStateEvent.ITERATION_STARTED,
                    {
                        "iteration": self.context.session.iterations,
                        "max_iterations": self.context.max_iterations,
                    },
                )

                if self.context.workflow_manager:
                    self.context.workflow_manager.update_context(
                        self.context.session.task_status
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
                    self.context.session.messages.append(guidance_message)
                    self.agent_logger.debug(
                        f"Injecting plan guidance: {guidance_message['content']}"
                    )

                current_task_for_iteration = (
                    rescoped_task or self.context.session.current_task
                )

                # --- Inner try/except for iteration-specific errors ---
                try:
                    self.agent_logger.info(
                        f"🔄 Iteration {self.context.session.iterations} current task: {current_task_for_iteration}"
                    )
                    iteration_content, new_plan = await self._run_task_iteration(
                        task=current_task_for_iteration
                    )
                    self.agent_logger.info(f"🔄 Content Preview: {iteration_content}")
                    last_plan = new_plan
                    if iteration_content:
                        final_result_content = iteration_content

                    # Emit iteration completed event
                    self.context.emit_event(
                        AgentStateEvent.ITERATION_COMPLETED,
                        {
                            "iteration": self.context.session.iterations,
                            "has_result": iteration_content is not None,
                            "has_plan": new_plan is not None,
                        },
                    )

                    # Add metrics update after each completed iteration
                    if self.context.metrics_manager:
                        metrics = self.context.metrics_manager.get_metrics()
                        self.context.emit_event(
                            AgentStateEvent.METRICS_UPDATED, {"metrics": metrics}
                        )

                    if self.context.session.final_answer:
                        self.agent_logger.info(
                            "✅ Final answer received during task execution."
                        )
                        self.context.session.task_status = TaskStatus.COMPLETE

                        # Emit final answer event
                        self.context.emit_event(
                            AgentStateEvent.FINAL_ANSWER_SET,
                            {
                                "answer": self.context.session.final_answer,
                                "iteration": self.context.session.iterations,
                            },
                        )

                        # Emit status changed event
                        self.context.emit_event(
                            AgentStateEvent.TASK_STATUS_CHANGED,
                            {"previous_status": "running", "new_status": "complete"},
                        )

                        break

                    self.context.update_system_prompt()
                    failure_count = 0

                except Exception as iter_error:
                    failure_count += 1
                    last_error = str(iter_error)
                    tb_str = traceback.format_exc()
                    self.agent_logger.error(
                        f"❌ Iteration {self.context.session.iterations} Error: {last_error}\n{tb_str}"
                    )
                    self.context.session.reasoning_log.append(
                        f"ERROR in iteration {self.context.session.iterations}: {last_error}"
                    )

                    # Emit error event
                    self.context.emit_event(
                        AgentStateEvent.ERROR_OCCURRED,
                        {
                            "error": "Iteration error",
                            "details": last_error,
                            "iteration": self.context.session.iterations,
                            "failure_count": failure_count,
                        },
                    )

                    if failure_count >= max_failures:
                        self.agent_logger.warning(
                            f"Reached max failures ({max_failures}). Attempting to rescope task."
                        )
                        error_context = f"Multiple ({failure_count}) failures during execution. Last error: {last_error}"
                        rescope_result = await self.rescope_goal(
                            self.context.session.initial_task, error_context
                        )

                        if rescope_result["rescoped_task"]:
                            potential_rescope = rescope_result["rescoped_task"]
                            if isinstance(potential_rescope, str):
                                rescoped_task = potential_rescope  # Update local var
                                self.context.session.current_task = rescoped_task
                                self.agent_logger.info(
                                    f"Task rescoped to: {rescoped_task}"
                                )
                                self.context.session.reasoning_log.append(
                                    f"Task rescoped: {rescope_result['explanation']}"
                                )

                                # Emit task rescope event
                                self.context.emit_event(
                                    AgentStateEvent.TASK_STATUS_CHANGED,
                                    {
                                        "previous_status": str(
                                            self.context.session.task_status
                                        ),
                                        "new_status": "rescoped",
                                        "rescoped_task": rescoped_task,
                                        "original_task": self.context.session.initial_task,
                                        "explanation": rescope_result["explanation"],
                                    },
                                )

                                failure_count = 0
                                if rescope_result.get("expected_tools") is not None:
                                    self.context.session.min_required_tools = set(
                                        rescope_result["expected_tools"]
                                    )
                                    self.agent_logger.debug(
                                        f"Updated required tools for rescoped task in session: {self.context.session.min_required_tools}"
                                    )
                                else:
                                    self.context.session.min_required_tools = None
                                    self.agent_logger.debug(
                                        "Reset required tools in session as rescope did not specify expected tools."
                                    )
                                if self.context.workflow_manager:
                                    self.context.workflow_manager.update_context(
                                        self.context.session.task_status,
                                        rescoped=True,
                                        original_task=self.context.session.initial_task,
                                        rescoped_task=rescoped_task,
                                    )
                                continue  # Continue loop with rescoped task
                            else:
                                self.agent_logger.error(
                                    "Rescoping failed unexpectedly: rescoped_task was not a string."
                                )
                                self.context.session.task_status = TaskStatus.ERROR

                                # Emit error event
                                self.context.emit_event(
                                    AgentStateEvent.ERROR_OCCURRED,
                                    {
                                        "error": "Rescoping failed",
                                        "details": "Rescoped task was not a string",
                                    },
                                )

                                # Emit status changed event
                                self.context.emit_event(
                                    AgentStateEvent.TASK_STATUS_CHANGED,
                                    {
                                        "previous_status": "running",
                                        "new_status": "error",
                                    },
                                )

                                break  # Exit loop
                        # Rescoping failed or not possible
                        self.agent_logger.error(
                            "Could not rescope task after multiple failures."
                        )
                        self.context.session.task_status = TaskStatus.ERROR
                        self.context.session.reasoning_log.append(
                            "Failed to rescope task after multiple errors."
                        )

                        # Emit error event
                        self.context.emit_event(
                            AgentStateEvent.ERROR_OCCURRED,
                            {
                                "error": "Rescoping failed",
                                "details": "Could not rescope task after multiple failures",
                            },
                        )

                        # Emit status changed event
                        self.context.emit_event(
                            AgentStateEvent.TASK_STATUS_CHANGED,
                            {"previous_status": "running", "new_status": "error"},
                        )

                        break  # Exit loop
                # --- End Inner try/except ---
            # --- End While Loop ---

            # 5. --- Determine Final Status After Loop ---
            # Status should be set if loop ended via break (COMPLETE, CANCELLED, ERROR)
            # If loop ended because _should_continue returned false, check conditions:
            if (
                self.context.session.task_status == TaskStatus.RUNNING
            ):  # Only update if still running
                if self.context.session.iterations >= (
                    self.context.max_iterations or float("inf")
                ):
                    self.context.session.task_status = TaskStatus.MAX_ITERATIONS

                    # Emit status changed event
                    self.context.emit_event(
                        AgentStateEvent.TASK_STATUS_CHANGED,
                        {
                            "previous_status": "running",
                            "new_status": "max_iterations_reached",
                        },
                    )

                elif not self.context.session.final_answer:
                    # If loop ended normally but no final answer, likely MAX_ITERATIONS or logic error
                    self.agent_logger.warning(
                        "Loop ended via _should_continue but final_answer is not set. Setting status to MAX_ITERATIONS or ERROR."
                    )
                    self.context.session.task_status = (
                        TaskStatus.MAX_ITERATIONS
                        if self.context.session.iterations
                        >= (self.context.max_iterations or float("inf"))
                        else TaskStatus.ERROR
                    )

                    # Emit status changed event
                    previous_status = "running"
                    new_status = str(self.context.session.task_status)
                    self.context.emit_event(
                        AgentStateEvent.TASK_STATUS_CHANGED,
                        {"previous_status": previous_status, "new_status": new_status},
                    )

                    if (
                        self.context.session.task_status == TaskStatus.ERROR
                        and not last_error
                    ):
                        last_error = "Loop ended unexpectedly without final answer."

                        # Emit error event
                        self.context.emit_event(
                            AgentStateEvent.ERROR_OCCURRED,
                            {"error": "Execution error", "details": last_error},
                        )
                else:
                    # Should not happen if _should_continue logic is correct
                    self.agent_logger.warning(
                        "Loop ended unexpectedly, assuming completion."
                    )
                    self.context.session.task_status = TaskStatus.COMPLETE

                    # Emit status changed event
                    self.context.emit_event(
                        AgentStateEvent.TASK_STATUS_CHANGED,
                        {"previous_status": "running", "new_status": "complete"},
                    )

            self.agent_logger.info(
                f"🏁 Determined final status: {self.context.session.task_status}"
            )

            # Set error message if status is ERROR and last_error has content
            if self.context.session.task_status == TaskStatus.ERROR and last_error:
                final_result_content = f"Error: {last_error}"
            elif self.context.session.task_status == TaskStatus.CANCELLED:
                final_result_content = "Task cancelled by user."

            # 6. --- Post-run Processing (Summary, Evaluation) ---
            self.context.session.summary = await self.generate_summary()
            self.context.session.evaluation = (
                await self.generate_goal_result_evaluation()
            )

            # Emit metrics updated event if metrics available
            if self.context.metrics_manager:
                metrics = self.context.metrics_manager.get_metrics()
                self.context.emit_event(
                    AgentStateEvent.METRICS_UPDATED, {"metrics": metrics}
                )

        # --- Outer Exception Handler for broader errors ---
        except Exception as run_error:
            tb_str = traceback.format_exc()
            self.agent_logger.error(
                f"Unhandled error during agent run: {run_error}\n{tb_str}"
            )

            # Emit error event
            self.context.emit_event(
                AgentStateEvent.ERROR_OCCURRED,
                {"error": "Unhandled error", "details": str(run_error)},
            )

            # Ensure session exists before modifying
            if hasattr(self.context, "session") and self.context.session:
                self.context.session.task_status = TaskStatus.ERROR
                self.context.session.evaluation = {
                    "adherence_score": 0.0,
                    "matches_intent": False,
                    "explanation": f"Critical error: {run_error}",
                    "strengths": [],
                    "weaknesses": ["Agent run failed critically."],
                }

                # Emit status changed event
                self.context.emit_event(
                    AgentStateEvent.TASK_STATUS_CHANGED,
                    {
                        "previous_status": str(self.context.session.task_status),
                        "new_status": "error",
                    },
                )

            # Set local vars for finally block
            last_error = str(run_error)
            final_result_content = f"Critical error during agent run: {run_error}"

        # --- Finally block executes regardless of exceptions ---
        finally:
            # 7. --- Cleanup and Result Preparation ---
            current_session_status = TaskStatus.ERROR  # Default if session missing
            session_final_answer = None
            session_evaluation = {}

            # Safely access session attributes
            if hasattr(self.context, "session") and self.context.session:
                current_session_status = self.context.session.task_status
                session_final_answer = self.context.session.final_answer
                session_evaluation = self.context.session.evaluation
                self.context.session.end_time = time.time()
                self.context.session.error = (
                    last_error if current_session_status == TaskStatus.ERROR else None
                )
                if not self.context.session.final_answer:
                    # Determine final result string carefully
                    result_to_use_for_package = (
                        final_result_content  # Content determined before/during finally
                        or "Task completed without explicit result."
                    )
                    self.context.session.final_answer = result_to_use_for_package
                else:
                    # If final_answer was set by tool, use that for the package
                    result_to_use_for_package = self.context.session.final_answer
            else:
                # Session missing, use defaults and captured error info
                result_to_use_for_package = (
                    final_result_content or f"Agent failed critically: {last_error}"
                )
                session_evaluation = {
                    "adherence_score": 0.0,
                    "matches_intent": False,
                    "explanation": f"Critical error occurred before session completion: {last_error}",
                    "strengths": [],
                    "weaknesses": ["Agent run failed critically."],
                }

            # Cleanup tasks
            if active_tasks:
                self.agent_logger.debug(
                    f"Cleaning up {len(active_tasks)} background tasks..."
                )
                for task in active_tasks:
                    if not task.done():
                        task.cancel()
                active_tasks.clear()

            # Prepare the final result package
            final_result_package = self._prepare_final_result(
                rescoped_task=rescoped_task,  # Pass local rescoped_task
                result_content=result_to_use_for_package,  # Use derived result string
                summary=getattr(
                    self.context.session, "summary", "Summary not generated."
                ),  # Safely get summary
                evaluation=session_evaluation,  # Use derived evaluation
            )

            # Update workflow context safely
            if self.context.workflow_manager:
                self.context.workflow_manager.update_context(
                    current_session_status,
                    result=result_to_use_for_package,
                    adherence_score=session_evaluation.get("adherence_score"),
                    matches_intent=session_evaluation.get("matches_intent"),
                    rescoped=(rescoped_task is not None),
                    error=(
                        last_error
                        if current_session_status == TaskStatus.ERROR
                        else None
                    ),
                )

            # Finalize metrics and add to package/session
            if self.context.metrics_manager:
                self.context.metrics_manager.finalize_run_metrics()
                metrics_data = self.context.metrics_manager.get_metrics()
                final_result_package["metrics"] = metrics_data
                if hasattr(self.context, "session") and self.context.session:
                    self.context.session.metrics = metrics_data

            # Save memory (using session safely)
            if self.context.memory_manager:
                if hasattr(self.context, "session") and self.context.session:
                    self.context.memory_manager.update_session_history(
                        self.context.session
                    )
                    self.context.memory_manager.save_memory()
                else:
                    self.agent_logger.warning(
                        "Cannot save memory, session object missing."
                    )

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

            self.agent_logger.info(
                f"ReactAgent run finished with status: {current_session_status}"
            )
        # --- End Finally Block ---

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
        # Fetch data from the session
        min_req_tools_list = (
            list(self.context.session.min_required_tools)
            if self.context.session.min_required_tools is not None
            else None
        )
        return {
            "status": str(self.context.session.task_status),  # Use session status
            "result": result_content  # Use explicitly passed result content
            or self.context.session.final_answer  # Fallback to session final_answer
            or "No textual result produced.",
            "iterations": self.context.session.iterations,  # Use session iterations
            "summary": summary or "Summary not generated.",  # Passed in
            "reasoning_log": self.context.session.reasoning_log,  # Use session log
            "evaluation": evaluation  # Passed in
            or {
                "adherence_score": self.context.session.completion_score,  # Use session score as fallback default
                "matches_intent": False,
                "explanation": "Evaluation not performed.",
                "strengths": [],
                "weaknesses": [],
            },
            "rescoped": rescoped_task is not None,  # Passed in
            "original_task": self.context.session.initial_task,  # Use session initial_task
            "rescoped_task": rescoped_task,  # Passed in
            "min_required_tools": min_req_tools_list,  # Use locally derived list
            "metrics": self.context.session.metrics,  # Use session metrics
            **(feasibility if feasibility else {}),  # Passed in
        }

    def _check_dependencies(self) -> bool:
        """Delegates dependency check to the WorkflowManager."""
        if self.context.workflow_manager:
            return self.context.workflow_manager.check_dependencies()
        self.agent_logger.debug("No workflow manager, skipping dependency check.")
        return True

    def _should_continue(self) -> bool:
        """Determines if the ReAct loop should continue."""
        # Use session status
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

        # Use session iterations
        if (
            self.context.max_iterations is not None
            and self.context.session.iterations >= self.context.max_iterations
        ):
            self.agent_logger.info(
                f"Stopping loop: Max iterations ({self.context.max_iterations}) reached."
            )
            self.context.session.task_status = (
                TaskStatus.MAX_ITERATIONS
            )  # Update session status
            return False

        # Use session final_answer
        if self.context.session.final_answer is not None:
            self.agent_logger.info("Stopping loop: Final answer set.")
            # Let the main check handle final_answer + tools
            pass

        if not self._check_dependencies():
            self.agent_logger.info("Stopping loop: Dependencies not met.")
            return False

        if self.context.reflect_enabled and self.context.reflection_manager:
            pass  # Reflection no longer determines completion score directly

        # Check required tools completion against actual successful tools context
        tools_completed = False
        deterministic_score = 0.0
        # Use session min_required_tools
        if (
            self.context.session.min_required_tools is not None
            and len(self.context.session.min_required_tools) > 0
        ):
            # Use session successful_tools
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
                # Use session task_nudges
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
        # Use session final_answer
        if (
            score_met
            and tools_completed
            and self.context.session.final_answer is not None
        ):
            self.agent_logger.info(
                f"Stopping loop: Score threshold met ({deterministic_score:.2f} >= {self.context.min_completion_score}), "
                f"all required tools used, and final answer provided."
            )
            self.context.session.task_status = (
                TaskStatus.COMPLETE
            )  # Update session status
            return False
        # Use session final_answer and task_nudges
        elif tools_completed and self.context.session.final_answer is None:
            nudge = "**All required tools used, but requires final_answer(<answer>) tool call.**"
            if nudge not in self.context.session.task_nudges:
                self.context.session.task_nudges.append(nudge)
        # Use session task_nudges
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
            response = await self.model_provider.get_completion(
                system=SUMMARY_SYSTEM_PROMPT, prompt=summary_prompt
            )

            if response and response.get("response"):
                return response["response"]
            else:
                self.agent_logger.warning(
                    "Summary generation failed: No response from model."
                )
                return f"Agent completed task '{self.context.session.initial_task[:50]}...' with status {self.context.session.task_status} after {self.context.session.iterations} iterations, using {len(tools_used_names)} tools."

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
        if not tool_history and not self.context.session.final_answer:
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
                self.context.session.final_answer  # Use session
                or "No explicit final textual answer provided by agent."
            )
            eval_context = {
                "original_goal": self.context.session.initial_task,  # Use session
                "final_result": (
                    final_result_str[:3000] + "..."
                    if len(final_result_str) > 3000
                    else final_result_str
                ),
                "final_status": str(self.context.session.task_status),  # Use session
                "tools_used": tool_names_used,
                "action_summary": self.context.session.summary,  # Use session summary
                "reasoning_log": self.context.session.reasoning_log[
                    -5:
                ],  # Use session log
            }

            # Get the deterministic score calculated during the run
            deterministic_score = (
                self.context.session.completion_score
            )  # Use session score
            self.agent_logger.info(
                f"Using deterministic score for final evaluation: {deterministic_score:.2f}"
            )

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
                # Fallback using deterministic score
                return {
                    "adherence_score": deterministic_score,  # Use deterministic score
                    "matches_intent": deterministic_score > 0.5,  # Simple heuristic
                    "explanation": "Basic evaluation based on tool completion score; LLM evaluation failed.",
                    "strengths": ["Used required tools proportionally to score."],
                    "weaknesses": ["LLM evaluation call failed."],
                }

            try:
                eval_data = response["response"]
                parsed_eval = (
                    json.loads(eval_data) if isinstance(eval_data, str) else eval_data
                )
                validated_eval_dict = self.EvaluationFormat(**parsed_eval).dict()

                # <<< OVERRIDE LLM Score >>>
                original_llm_score = validated_eval_dict.get("adherence_score")
                validated_eval_dict["adherence_score"] = deterministic_score
                # <<< End Override >>>

                self.agent_logger.info(
                    f"📄 Goal evaluation result: Adherence Score={deterministic_score:.2f} (overridden from LLM score: {original_llm_score:.2f}), "
                    f"Matches Intent={validated_eval_dict.get('matches_intent')}"
                )
                # Use session reasoning log
                self.context.session.reasoning_log.append(
                    f"Goal Adherence Evaluation (Score: {deterministic_score:.2f}): {validated_eval_dict.get('explanation')}"
                )
                return validated_eval_dict

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
                # Use session reasoning log
                self.context.session.reasoning_log.append(
                    f"Assistant Thought/Response: {last_content[:200]}..."
                )
            elif msg.get("tool_calls"):
                # Use session reasoning log
                self.context.session.reasoning_log.append(
                    f"Assistant Action: Called tools."
                )

                # Emit tool called event
                tool_calls = msg.get("tool_calls", [])
                for tool_call in tool_calls:
                    self.context.emit_event(
                        AgentStateEvent.TOOL_CALLED,
                        {
                            "tool_name": tool_call.get("name", "unknown"),
                            "tool_id": tool_call.get("id", "unknown"),
                            "parameters": tool_call.get("parameters", {}),
                        },
                    )
        elif isinstance(think_act_result_dict, str):
            last_content = think_act_result_dict
            # Use session reasoning log
            self.context.session.reasoning_log.append(
                f"Step Result (non-dict): {last_content[:200]}..."
            )

        # Use session final_answer
        if self.context.session.final_answer is not None:
            self.agent_logger.info("Final answer set during Think/Act step.")

            # Emit final answer event
            self.context.emit_event(
                AgentStateEvent.FINAL_ANSWER_SET,
                {"answer": self.context.session.final_answer},
            )

            # Ensure plan is None if final answer is set here
            return self.context.session.final_answer, None

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
                    # Use session reasoning log
                    self.context.session.reasoning_log.append(
                        f"Reflection:  Reason={reflection['reason']}"
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
            # Use session reasoning log
            self.context.session.reasoning_log.append(
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
            # Use session reasoning log
            self.context.session.reasoning_log.append(
                f"Plan (Default): {plan['rationale']} -> {plan['next_step']}"
            )
        else:
            self.agent_logger.info(
                "Reflection suggested 'None' or failed, no further plan generated."
            )
            # Plan remains None

        # Return both the last content generated during think/act and the derived plan
        # Use session final_answer
        return self.context.session.final_answer or last_content, plan

    # === Event Subscription Interface ===

    @property
    def events(self):
        """
        Access the event subscription interface for this agent.

        This property provides a fluent API for subscribing to agent events.
        It allows subscribing to events in a type-safe manner without having
        to directly access the underlying observer.

        The events property is both a property accessor and callable:

        Returns:
            AgentEventManager: An interface for subscribing to events

        Example:
            ```python
            # Subscribe to specific events using helper methods
            agent.events.on_tool_called().subscribe(
                lambda event: print(f"Tool called: {event['tool_name']}")
            )

            # Subscribe to any event directly using the callable interface
            agent.events(AgentStateEvent.ERROR_OCCURRED).subscribe(
                lambda event: print(f"Error: {event['error']}")
            )

            # Subscribe to multiple events with the same callback
            def log_event(event):
                print(f"Event: {event['event_type']}")

            agent.events.on_tool_called().subscribe(log_event)
            agent.events.on_tool_completed().subscribe(log_event)
            ```
        """
        return self._event_manager

    # Shorthand methods for common event subscriptions

    def on_session_started(self, callback):
        """
        Subscribe to session started events.

        Args:
            callback: Function to call when a session starts
                      Receives SessionStartedEventData

        Returns:
            The event subscription for method chaining
        """
        return self.events.on_session_started().subscribe(callback)

    def on_session_ended(self, callback):
        """
        Subscribe to session ended events.

        Args:
            callback: Function to call when a session ends
                      Receives SessionEndedEventData

        Returns:
            The event subscription for method chaining
        """
        return self.events.on_session_ended().subscribe(callback)

    def on_task_status_changed(self, callback):
        """
        Subscribe to task status changed events.

        Args:
            callback: Function to call when task status changes
                      Receives TaskStatusChangedEventData

        Returns:
            The event subscription for method chaining
        """
        return self.events.on_task_status_changed().subscribe(callback)

    def on_iteration_started(self, callback):
        """
        Subscribe to iteration started events.

        Args:
            callback: Function to call when an iteration starts
                      Receives IterationStartedEventData

        Returns:
            The event subscription for method chaining
        """
        return self.events.on_iteration_started().subscribe(callback)

    def on_iteration_completed(self, callback):
        """
        Subscribe to iteration completed events.

        Args:
            callback: Function to call when an iteration completes
                      Receives IterationCompletedEventData

        Returns:
            The event subscription for method chaining
        """
        return self.events.on_iteration_completed().subscribe(callback)

    def on_tool_called(self, callback):
        """
        Subscribe to tool called events.

        Args:
            callback: Function to call when a tool is called
                      Receives ToolCalledEventData

        Returns:
            The event subscription for method chaining
        """
        return self.events.on_tool_called().subscribe(callback)

    def on_tool_completed(self, callback):
        """
        Subscribe to tool completed events.

        Args:
            callback: Function to call when a tool completes
                      Receives ToolCompletedEventData

        Returns:
            The event subscription for method chaining
        """
        return self.events.on_tool_completed().subscribe(callback)

    def on_tool_failed(self, callback):
        """
        Subscribe to tool failed events.

        Args:
            callback: Function to call when a tool fails
                      Receives ToolFailedEventData

        Returns:
            The event subscription for method chaining
        """
        return self.events.on_tool_failed().subscribe(callback)

    def on_reflection_generated(self, callback):
        """
        Subscribe to reflection generated events.

        Args:
            callback: Function to call when a reflection is generated
                      Receives ReflectionGeneratedEventData

        Returns:
            The event subscription for method chaining
        """
        return self.events.on_reflection_generated().subscribe(callback)

    def on_final_answer_set(self, callback):
        """
        Subscribe to final answer set events.

        Args:
            callback: Function to call when a final answer is set
                      Receives FinalAnswerSetEventData

        Returns:
            The event subscription for method chaining
        """
        return self.events.on_final_answer_set().subscribe(callback)

    def on_metrics_updated(self, callback):
        """
        Subscribe to metrics updated events.

        Args:
            callback: Function to call when metrics are updated
                      Receives MetricsUpdatedEventData

        Returns:
            The event subscription for method chaining
        """
        return self.events.on_metrics_updated().subscribe(callback)

    def on_error_occurred(self, callback):
        """
        Subscribe to error occurred events.

        Args:
            callback: Function to call when an error occurs
                      Receives ErrorOccurredEventData

        Returns:
            The event subscription for method chaining
        """
        return self.events.on_error_occurred().subscribe(callback)
