from __future__ import annotations
from typing import (
    List,
    Dict,
    Any,
    Literal,
    Optional,
    Callable,
    Sequence,
    Awaitable,
    Union,
    Tuple,
)

from pydantic import BaseModel, Field
import time
from reactive_agents.config.mcp_config import MCPConfig
from reactive_agents.loggers.base import Logger
from reactive_agents.model_providers.base import BaseModelProvider
from reactive_agents.agent_mcp.client import MCPClient
from reactive_agents.prompts.agent_prompts import (
    REACT_AGENT_SYSTEM_PROMPT,
    DYNAMIC_SYSTEM_PROMPT_TEMPLATE,
)
from reactive_agents.tools.abstractions import ToolProtocol
from reactive_agents.common.types.status_types import TaskStatus

# --- Import Manager Classes ---
from reactive_agents.components.metrics_manager import MetricsManager
from reactive_agents.components.memory_manager import MemoryManager
from reactive_agents.components.reflection_manager import ReflectionManager
from reactive_agents.components.workflow_manager import WorkflowManager
from reactive_agents.components.tool_manager import ToolManager

# --- Import AgentSession from its new location ---
from reactive_agents.common.types.session_types import AgentSession

# --- Import AgentStateObserver ---
from .agent_observer import AgentStateObserver
from reactive_agents.common.types.event_types import AgentStateEvent


# Now define AgentContext
class AgentContext(BaseModel):
    """Centralized context holding configuration and components for an agent."""

    # Core Agent Configuration
    agent_name: str
    provider_model_name: str
    instructions: str = ""
    role: str = ""
    role_instructions: Dict[str, Any] = {}

    # --- Workflow Context and Dependencies ---
    workflow_context_shared: Optional[Dict[str, Any]] = None
    workflow_dependencies: List[str] = []

    # Configuration Flags & Settings (Remain in Context)
    tool_use_enabled: bool = True
    reflect_enabled: bool = False
    use_memory_enabled: bool = True
    collect_metrics_enabled: bool = True
    min_completion_score: float = 1.0
    max_iterations: Optional[int] = None
    max_task_retries: int = 3
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    enable_caching: bool = True
    cache_ttl: int = 3600
    offline_mode: bool = False
    retry_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 10.0,
            "retry_network_errors": True,
        }
    )
    check_tool_feasibility: bool = True
    confirmation_callback: Optional[
        Callable[
            [str, Dict[str, Any]], Awaitable[Union[bool, Tuple[bool, Optional[str]]]]
        ]
    ] = None
    confirmation_config: Optional[Dict[str, Any]] = None

    # Core Components (Remain in Context)
    model_provider: Optional[BaseModelProvider] = None
    model_provider_options: Optional[Dict[str, Any]] = None
    mcp_client: Optional[MCPClient] = None
    mcp_config: Optional[MCPConfig] = None
    tools: List[Any] = Field(default_factory=list)

    # Loggers (Remain in Context)
    agent_logger: Optional[Logger] = None
    tool_logger: Optional[Logger] = None
    result_logger: Optional[Logger] = None

    # Component Managers (Remain in Context)
    metrics_manager: Optional["MetricsManager"] = None
    memory_manager: Optional["MemoryManager"] = None
    reflection_manager: Optional["ReflectionManager"] = None
    workflow_manager: Optional["WorkflowManager"] = None
    tool_manager: Optional["ToolManager"] = None

    # --- Add State Observer ---
    state_observer: Optional[AgentStateObserver] = None
    enable_state_observation: bool = True

    @staticmethod
    def _create_default_session() -> AgentSession:
        return AgentSession(
            initial_task="",
            current_task="",
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

    # Session State Holder (Reference to the current run's state)
    session: AgentSession = Field(default_factory=_create_default_session)

    # !! REMOVED FIELDS previously here (initial_task, final_answer, messages, etc.) !!

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        # Call Pydantic's __init__ first to set up fields correctly
        super().__init__(**data)

        # Now, initialize components and managers *after* super().__init__
        self._initialize_loggers()
        assert self.agent_logger is not None
        self._initialize_model_provider()

        # --- Initialize Managers ---
        # Pydantic already initialized managers to None based on Optional type hint
        if self.collect_metrics_enabled:
            self.metrics_manager = MetricsManager(context=self)

        self.tool_manager = ToolManager(context=self)

        if self.use_memory_enabled:
            self.memory_manager = MemoryManager(context=self)

        if self.reflect_enabled:
            self.reflection_manager = ReflectionManager(context=self)

        self.workflow_manager = WorkflowManager(
            context=self,
            workflow_context=self.workflow_context_shared,
            workflow_dependencies=self.workflow_dependencies,
        )

        # Initialize state observer if enabled
        if self.enable_state_observation:
            self.state_observer = AgentStateObserver()
            self.agent_logger.info("State observer initialized.")

        # --- End Initialize Managers ---

        # Set initial system message
        if not self.session.messages:
            self.session.messages = [
                {"role": "system", "content": self._get_initial_system_prompt()}
            ]

        # Initialize current task
        if not self.session.current_task and self.session.initial_task:
            self.session.current_task = self.session.initial_task

        self.agent_logger.info(
            f"AgentContext for '{self.agent_name}' initialized with managers."
        )

    def _initialize_loggers(self):
        if not self.agent_logger:
            self.agent_logger = Logger(
                name=self.agent_name, type="agent", level=self.log_level
            )
        if not self.tool_logger:
            self.tool_logger = Logger(
                name=f"{self.agent_name} Tool", type="tool", level=self.log_level
            )
        if not self.result_logger:
            self.result_logger = Logger(
                name=f"{self.agent_name} Result",
                type="agent_response",
                level=self.log_level,
            )

    def _initialize_model_provider(self):
        assert self.agent_logger is not None
        if not self.model_provider:
            from reactive_agents.model_providers.factory import ModelProviderFactory

            self.model_provider = ModelProviderFactory.get_model_provider(
                self.provider_model_name,
                options=self.model_provider_options,
                context=self,
            )
            self.agent_logger.info(
                f"Model Provider Initialized: {self.model_provider.name}:{self.model_provider.model}"
            )

    def _get_initial_system_prompt(self) -> str:
        """Constructs the initial system prompt based on role and instructions."""
        prompt = REACT_AGENT_SYSTEM_PROMPT.format(
            role=self.role,
            instructions=self.instructions,
            role_specific_instructions=self.role_instructions,
            task=self.session.current_task,
            task_progress="\n".join(self.session.task_progress),
        )
        return prompt

    # --- Observer methods ---
    def emit_event(self, event_type: AgentStateEvent, data: Dict[str, Any]) -> None:
        """
        Emit an event to all registered callbacks.

        Args:
            event_type: The type of event being emitted
            data: The data associated with the event
        """
        if self.state_observer and self.enable_state_observation:
            # Include basic agent/session context with all events
            context_data = {
                "agent_name": self.agent_name,
                "session_id": getattr(self.session, "session_id", None),
                "task": getattr(self.session, "current_task", None),
                "task_status": str(getattr(self.session, "task_status", "unknown")),
                "iterations": getattr(self.session, "iterations", 0),
            }
            # Merge with event-specific data (event data takes precedence)
            event_data = {**context_data, **data}
            self.state_observer.emit(event_type, event_data)

    async def emit_event_async(
        self, event_type: AgentStateEvent, data: Dict[str, Any]
    ) -> None:
        """
        Emit an event to all registered async callbacks.

        Args:
            event_type: The type of event being emitted
            data: The data associated with the event
        """
        if self.state_observer and self.enable_state_observation:
            # Include basic agent/session context with all events
            context_data = {
                "agent_name": self.agent_name,
                "session_id": getattr(self.session, "session_id", None),
                "task": getattr(self.session, "current_task", None),
                "task_status": str(getattr(self.session, "task_status", "unknown")),
                "iterations": getattr(self.session, "iterations", 0),
            }
            # Merge with event-specific data (event data takes precedence)
            event_data = {**context_data, **data}
            await self.state_observer.emit_async(event_type, event_data)

    # Methods to interact with components will be added later
    # e.g., get_tools(), update_metrics(), save_memory(), get_reflection() etc.

    async def close(self):
        """Safely close resources like the MCP client."""
        assert self.agent_logger is not None
        self.agent_logger.info(f"Closing context for {self.agent_name}...")
        # TODO: Add any other closing responsibilities for context
        self.agent_logger.info(f"{self.agent_name} context closed successfully.")

    # Convenience accessors (optional, direct access context.manager is also fine)
    def get_tool_signatures(self) -> List[Dict[str, Any]]:
        return self.tool_manager.tool_signatures if self.tool_manager else []

    def get_available_tools(self) -> Sequence[ToolProtocol]:
        return self.tool_manager.get_available_tools() if self.tool_manager else []

    def get_available_tool_names(self) -> set[str]:
        return (
            self.tool_manager.get_available_tool_names() if self.tool_manager else set()
        )

    def get_tool_history(self) -> List[Dict[str, Any]]:
        return self.tool_manager.tool_history if self.tool_manager else []

    def get_reflections(self) -> List[Dict[str, Any]]:
        return self.reflection_manager.reflections if self.reflection_manager else []

    def get_metrics(self) -> Dict[str, Any]:
        if self.metrics_manager:
            return self.metrics_manager.get_metrics()
        return {}  # Return empty if metrics disabled

    def update_system_prompt(self, next_step: Optional[str] = None):
        """Update system prompt with dynamic context including next step."""
        if self.session.messages and self.session.messages[0]["role"] == "system":
            # Store next step in session if provided
            if next_step:
                self.session.current_next_step = next_step
                self.session.next_step_timestamp = time.time()
            # Generate dynamic prompt
            dynamic_prompt = self._generate_dynamic_system_prompt()
            self.session.messages[0]["content"] = dynamic_prompt

    def _generate_dynamic_system_prompt(self) -> str:
        """Generate a dynamic system prompt with current context."""
        session = self.session
        return DYNAMIC_SYSTEM_PROMPT_TEMPLATE.format(
            role=self.role,
            instructions=self.instructions,
            role_specific_instructions=self.role_instructions,
            task=session.current_task,
            iteration=session.iterations,
            max_iterations=self.max_iterations or "∞",
            task_status=str(session.task_status),
            context_sections=self._build_context_sections(),
            next_step_section=self._build_next_step_section(),
            progress_summary=self._build_progress_summary(),
        )

    def _build_context_sections(self) -> str:
        session = self.session
        sections = []
        if session.min_required_tools:
            sections.append(f"Required Tools: {', '.join(session.min_required_tools)}")
        if session.successful_tools:
            sections.append(f"Completed Tools: {', '.join(session.successful_tools)}")
        if session.reasoning_log:
            recent_reasoning = session.reasoning_log[-3:]
            sections.append(f"Recent Reasoning:\n" + "\n".join(recent_reasoning))
        if session.task_nudges:
            sections.append(f"Task Reminders:\n" + "\n".join(session.task_nudges))
        return "\n\n".join(sections) if sections else "No additional context available."

    def _build_next_step_section(self) -> str:
        if (
            not self.session.current_next_step
            or self.session.current_next_step == "None"
        ):
            return """
CRITICAL NEXT STEP TO EXECUTE:
Use the final_answer tool with parameters: {'answer': '<your_complete_answer_to_the_task>'}

INSTRUCTIONS:
- This is the final step of your task
- Provide a complete, detailed answer that addresses the original question
- Include all relevant information you have gathered
- Be specific and thorough in your response
- Use the exact format: final_answer({'answer': 'your answer here'})
"""

        source_info = (
            f" (generated by {self.session.next_step_source})"
            if self.session.next_step_source
            else ""
        )

        return f"""
CRITICAL NEXT STEP TO EXECUTE{source_info}:

{self.session.current_next_step}

EXECUTION INSTRUCTIONS:
1. Read the next step above carefully - it is a DIRECT INSTRUCTION to execute
2. The next step tells you exactly what tool to use and with what parameters
3. Identify the exact tool name mentioned (after "Use the" or "Execute the")
4. Extract all parameters and their values from the instruction
5. Execute the tool call with the specified parameters exactly as written
6. Do not modify, simplify, or change the parameters
7. Follow the step exactly as written - it is your instruction to follow
8. If the step mentions multiple actions, perform them in order

IMPORTANT:
- You MUST execute this exact step now - it is your instruction
- Do not deviate from this instruction
- Do not add extra steps unless explicitly mentioned
- Do not skip any parameters mentioned in the step
- If you are unsure about any part, execute it exactly as written
- This is a direct command for you to follow

AVAILABLE TOOLS:
{', '.join(self.get_available_tool_names()) if self.tool_manager else 'No tools available'}

Remember: The next step above is your DIRECT INSTRUCTION. Execute it precisely as written.
"""

    def _build_progress_summary(self) -> str:
        session = self.session
        summary_parts = []
        summary_parts.append(f"Completion Score: {session.completion_score:.2f}")
        if session.successful_tools:
            summary_parts.append(f"Tools Used: {len(session.successful_tools)}")
        if session.task_progress:
            recent_progress = session.task_progress[-2:]
            summary_parts.append(f"Recent Progress: {'; '.join(recent_progress)}")
        return (
            "\n".join(summary_parts) if summary_parts else "No progress data available."
        )

    def prune_context_if_needed(self, max_messages: int = 20):
        messages = self.session.messages
        if len(messages) <= max_messages:
            return
        system_message = messages[0] if messages[0]["role"] == "system" else None
        recent_messages = (
            messages[-max_messages + 1 :]
            if system_message
            else messages[-max_messages:]
        )
        if system_message:
            self.session.messages = [system_message] + recent_messages
        else:
            self.session.messages = recent_messages
        if self.agent_logger:
            self.agent_logger.info(
                f"Pruned context from {len(messages)} to {len(self.session.messages)} messages"
            )


# --- Rebuild Models to Resolve Forward References ---
# Call model_rebuild() on dependent models after AgentContext is defined
# This allows them to correctly resolve the 'AgentContext' forward reference.
MetricsManager.model_rebuild(force=True)
MemoryManager.model_rebuild(force=True)
ReflectionManager.model_rebuild(force=True)
WorkflowManager.model_rebuild(force=True)
ToolManager.model_rebuild(force=True)
# --- End Rebuild Models ---
