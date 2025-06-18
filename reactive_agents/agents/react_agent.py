from __future__ import annotations
import json
import traceback
import asyncio
from typing import (
    List,
    Any,
    Literal,
    Optional,
    Dict,
    TYPE_CHECKING,
)

from pydantic import BaseModel, Field, model_validator
from reactive_agents.config.mcp_config import MCPConfig
from reactive_agents.prompts.agent_prompts import (
    MISSING_TOOLS_PROMPT,
    TOOL_FEASIBILITY_CONTEXT_PROMPT,
    RESCOPE_SYSTEM_PROMPT,
    RESCOPE_CONTEXT_PROMPT,
)
from reactive_agents.agents.base import Agent
from reactive_agents.context.agent_context import AgentContext
from reactive_agents.agent_mcp.client import MCPClient
from reactive_agents.context.agent_events import (
    EventCallback,
    SessionStartedEventData,
    SessionEndedEventData,
    TaskStatusChangedEventData,
    IterationStartedEventData,
    IterationCompletedEventData,
    ToolCalledEventData,
    ToolCompletedEventData,
    ToolFailedEventData,
    ReflectionGeneratedEventData,
    FinalAnswerSetEventData,
    MetricsUpdatedEventData,
    ErrorOccurredEventData,
    PauseRequestedEventData,
    PausedEventData,
    ResumeRequestedEventData,
    ResumedEventData,
    TerminateRequestedEventData,
    TerminatedEventData,
    StopRequestedEventData,
    StoppedEventData,
)
from reactive_agents.common.types.status_types import TaskStatus
from reactive_agents.common.types.confirmation_types import (
    ConfirmationCallbackProtocol,
)
from reactive_agents.common.types.agent_types import (
    ToolAnalysisFormat,
    RescopeFormat,
)
from reactive_agents.components.task_executor import TaskExecutor
from reactive_agents.components.tool_processor import ToolProcessor
from reactive_agents.components.event_manager import EventManager
from reactive_agents.agents.validators.config_validator import ConfigValidator

if TYPE_CHECKING:
    from reactive_agents.components.execution_engine import AgentExecutionEngine


# --- Agent Configuration Model ---
class ReactAgentConfig(BaseModel):
    # Required parameters
    agent_name: str = Field(description="Name of the agent.")
    provider_model_name: str = Field(description="Name of the LLM provider and model.")

    # Optional parameters
    model_provider_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Options for the LLM provider."
    )
    role: Optional[str] = Field(
        default="Task Executor", description="Role of the agent."
    )
    mcp_client: Optional[MCPClient] = Field(
        default=None, description="An initialized MCPClient instance."
    )
    mcp_config: Optional[MCPConfig] = Field(
        default=None, description="MCP config dict or file path to use for MCPClient."
    )
    mcp_server_filter: Optional[List[str]] = Field(
        default_factory=list,
        description="Filter List of MCP servers for the agent to use in the MCPClient.",
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
    log_level: Optional[Literal["debug", "info", "warning", "error", "critical"]] = (
        Field(
            default="info",
            description="Logging level ('debug', 'info', 'warning', 'error' or 'critical').",
        )
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
    # Add workflow_context_shared field to ReactAgentConfig
    workflow_context_shared: Optional[Dict[str, Any]] = Field(
        default=None, description="Shared workflow context data."
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
        # Get all known fields from the model
        known_fields = set(cls.model_fields.keys())

        # Fields that should be processed normally (not captured as extra kwargs)
        normal_fields = known_fields - {"kwargs"}

        extra_kwargs = {}
        processed_values = {}

        for key, value in values.items():
            if key in normal_fields:
                processed_values[key] = value
            else:
                extra_kwargs[key] = value

        # Add the extra kwargs to the processed values
        processed_values["kwargs"] = extra_kwargs
        return processed_values


# --- End Agent Configuration Model ---


class ReactAgent(Agent):
    """
    A ReAct-style agent that uses reflection and planning within an AgentContext.
    """

    execution_engine: "AgentExecutionEngine"
    event_manager: EventManager
    task_executor: TaskExecutor
    tool_processor: ToolProcessor

    def __init__(
        self,
        config: ReactAgentConfig,
    ):
        """
        Initializes the ReactAgent using a configuration object.

        Args:
            config: The ReactAgentConfig object containing all settings.
        """
        # Import here to avoid circular imports
        from reactive_agents.components.execution_engine import AgentExecutionEngine

        # Initialize config validator with string log level
        self.config_validator = ConfigValidator(str(config.log_level or "info"))

        # Validate configuration
        validated_config = self.config_validator.validate_agent_config(
            agent_name=config.agent_name,
            provider_model_name=config.provider_model_name,
            model_provider_options=config.model_provider_options,
            role=config.role,
            mcp_client=config.mcp_client,
            mcp_config=config.mcp_config,
            mcp_server_filter=config.mcp_server_filter,
            min_completion_score=config.min_completion_score,
            instructions=config.instructions,
            max_iterations=config.max_iterations,
            reflect_enabled=config.reflect_enabled,
            log_level=str(config.log_level or "info"),  # Convert to string
            initial_task=config.initial_task,
            tool_use_enabled=config.tool_use_enabled,
            custom_tools=config.custom_tools,
            use_memory_enabled=config.use_memory_enabled,
            collect_metrics_enabled=config.collect_metrics_enabled,
            check_tool_feasibility=config.check_tool_feasibility,
            enable_caching=config.enable_caching,
            confirmation_callback=config.confirmation_callback,
            confirmation_config=config.confirmation_config,
            workflow_context_shared=config.workflow_context_shared,
        )

        # Create the context
        context = AgentContext(**validated_config)

        # Initialize the base Agent class
        super().__init__(context)

        # Initialize the tool processor first
        self.tool_processor = ToolProcessor(self)

        # Process custom tools to ensure they're properly wrapped
        processed_tools = self.tool_processor.process_custom_tools(
            validated_config["custom_tools"]
        )

        # Update context with processed tools
        self.context.tools = processed_tools

        # Initialize the execution engine
        self.execution_engine = AgentExecutionEngine(self)

        # Initialize the event manager
        self.event_manager = EventManager(self)

        # Initialize the task executor last, after execution engine is ready
        self.task_executor = TaskExecutor(self)

        # Store the config for potential reference
        self.config = config
        self._closed = False

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        """Closes the agent's context and associated resources."""
        await self.context.close()
        if self.context.mcp_client is not None:
            self.agent_logger.info(
                f"Closing MCPClient for {self.context.agent_name}..."
            )
            await self.context.mcp_client.close()
            self.agent_logger.info(
                f"{self.context.agent_name} MCPClient closed successfully."
            )
        self._closed = True
        self.agent_logger.info(f"{self.context.agent_name} closed successfully.")

    async def initialize(self) -> ReactAgent:
        """Initialize the agent's context and associated resources."""
        try:
            if self.config.mcp_config or self.config.mcp_server_filter:
                self.context.mcp_config = (
                    MCPConfig.model_validate(self.config.mcp_config, strict=False)
                    if self.config.mcp_config
                    else None
                )
                self.context.mcp_client = await MCPClient(
                    server_config=self.context.mcp_config,
                    server_filter=self.config.mcp_server_filter,
                ).initialize()
            if self.context.tool_manager:
                await self.context.tool_manager._initialize_tools()
            return self
        except Exception as e:
            print("Error initializing MCPClient:", e)
            raise e

    async def run(
        self,
        initial_task: str,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Run the ReAct agent loop for the given task using the AgentExecutionEngine.

        Args:
            initial_task: The task description (required).
            cancellation_event: An optional asyncio.Event to signal cancellation.

        Returns:
            A dictionary containing the final status, result, and other execution details.
        """
        return await self.execution_engine.execute(
            initial_task=initial_task,
            cancellation_event=cancellation_event,
            # self_improve=True,
            # max_improvement_attempts=3,
        )

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
            nudge = "**All required tools used, but requires 'final_answer(<answer>)' tool call.**"
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
                    ToolAnalysisFormat.model_json_schema()
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
                validated_analysis = ToolAnalysisFormat(**parsed_analysis)

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
        """Delegates summary generation to the execution engine."""
        return await self.execution_engine._generate_summary()

    async def generate_goal_result_evaluation(self) -> Dict[str, Any]:
        """Delegates goal result evaluation to the execution engine."""
        return await self.execution_engine._generate_goal_result_evaluation()

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
                    RescopeFormat.model_json_schema()
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
                validated_rescope = RescopeFormat(**parsed_rescope)

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

    async def _run_task_iteration(self, task: str) -> Optional[str]:
        """
        Runs a single iteration of the task execution loop.
        Delegates to TaskExecutor for the actual execution.

        Args:
            task: The current task to execute

        Returns:
            Optional[str]: The result of the iteration
        """
        self.context.session.iterations += 1
        self.agent_logger.info(
            f"Running Iteration: {self.context.session.iterations}/{self.context.max_iterations}"
        )

        # Check if we already have a final answer
        if self.context.session.final_answer:
            self.agent_logger.info("Final answer already set, returning it.")
            return self.context.session.final_answer

        # Execute the iteration using TaskExecutor
        result = await self.task_executor.execute_iteration(task)

        if not result:
            self.agent_logger.warning(
                f"Iteration {self.context.session.iterations} failed or produced no result."
            )
            return None

        # Check if we have a final answer in the result
        if result.get("final_answer"):
            self.context.session.final_answer = result["final_answer"]
            self.agent_logger.info("Final answer set from iteration result.")
            return self.context.session.final_answer

        # Check if we should continue
        if not self.task_executor.should_continue():
            # If we're not continuing and have a think_act result, use that as the final answer
            think_act_result = (
                result.get("think_act", {}).get("message", {}).get("content")
            )
            if think_act_result:
                self.context.session.final_answer = think_act_result
                self.agent_logger.info("Setting final answer from think_act result.")
                return think_act_result

        return None

    # === Event Subscription Interface ===

    @property
    def events(self):
        """
        Access the event subscription interface for this agent.

        Returns:
            EventManager: An interface for subscribing to events
        """
        return self.event_manager

    # --- Agent Control Methods ---
    async def pause(self):
        """Request the agent to pause at the next safe point."""
        await self.execution_engine.pause()

    async def resume(self):
        """Resume the agent if it is paused."""
        await self.execution_engine.resume()

    async def terminate(self):
        """Request the agent to terminate as soon as possible."""
        await self.execution_engine.terminate()

    async def stop(self):
        """Request the agent to gracefully stop after the current step."""
        await self.execution_engine.stop()

    # --- Agent Control Event Subscriptions ---
    def on_pause_requested(
        self, callback: EventCallback[PauseRequestedEventData]
    ) -> None:
        """Register callback for pause requested event."""
        self.event_manager.on_pause_requested(callback)

    def on_paused(self, callback: EventCallback[PausedEventData]) -> None:
        """Register callback for paused event."""
        self.event_manager.on_paused(callback)

    def on_resume_requested(
        self, callback: EventCallback[ResumeRequestedEventData]
    ) -> None:
        """Register callback for resume requested event."""
        self.event_manager.on_resume_requested(callback)

    def on_resumed(self, callback: EventCallback[ResumedEventData]) -> None:
        """Register callback for resumed event."""
        self.event_manager.on_resumed(callback)

    def on_terminate_requested(
        self, callback: EventCallback[TerminateRequestedEventData]
    ) -> None:
        """Register callback for terminate requested event."""
        self.event_manager.on_terminate_requested(callback)

    def on_terminated(self, callback: EventCallback[TerminatedEventData]) -> None:
        """Register callback for terminated event."""
        self.event_manager.on_terminated(callback)

    def on_stop_requested(
        self, callback: EventCallback[StopRequestedEventData]
    ) -> None:
        """Register callback for stop requested event."""
        self.event_manager.on_stop_requested(callback)

    def on_stopped(self, callback: EventCallback[StoppedEventData]) -> None:
        """Register callback for stopped event."""
        self.event_manager.on_stopped(callback)

    def on_session_started(
        self, callback: EventCallback[SessionStartedEventData]
    ) -> None:
        """Register callback for session started event."""
        self.event_manager.on_session_started(callback)

    def on_session_ended(self, callback: EventCallback[SessionEndedEventData]) -> None:
        """Register callback for session ended event."""
        self.event_manager.on_session_ended(callback)

    def on_iteration_started(
        self, callback: EventCallback[IterationStartedEventData]
    ) -> None:
        """Register callback for iteration started event."""
        self.event_manager.on_iteration_started(callback)

    def on_iteration_completed(
        self, callback: EventCallback[IterationCompletedEventData]
    ) -> None:
        """Register callback for iteration completed event."""
        self.event_manager.on_iteration_completed(callback)

    def on_task_status_changed(
        self, callback: EventCallback[TaskStatusChangedEventData]
    ) -> None:
        """Register callback for task status changed event."""
        self.event_manager.on_task_status_changed(callback)

    def on_tool_called(self, callback: EventCallback[ToolCalledEventData]) -> None:
        """Register callback for tool called event."""
        self.event_manager.on_tool_called(callback)

    def on_tool_completed(
        self, callback: EventCallback[ToolCompletedEventData]
    ) -> None:
        """Register callback for tool completed event."""
        self.event_manager.on_tool_completed(callback)

    def on_tool_failed(self, callback: EventCallback[ToolFailedEventData]) -> None:
        """Register callback for tool failed event."""
        self.event_manager.on_tool_failed(callback)

    def on_error_occurred(
        self, callback: EventCallback[ErrorOccurredEventData]
    ) -> None:
        """Register callback for error occurred event."""
        self.event_manager.on_error_occurred(callback)

    def on_reflection_generated(
        self, callback: EventCallback[ReflectionGeneratedEventData]
    ) -> None:
        """Register callback for reflection generated event."""
        self.event_manager.on_reflection_generated(callback)

    def on_final_answer_set(
        self, callback: EventCallback[FinalAnswerSetEventData]
    ) -> None:
        """Register callback for final answer set event."""
        self.event_manager.on_final_answer_set(callback)

    def on_metrics_updated(
        self, callback: EventCallback[MetricsUpdatedEventData]
    ) -> None:
        """Register callback for metrics updated event."""
        self.event_manager.on_metrics_updated(callback)
