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
    Set,
)

from pydantic import BaseModel, Field, ConfigDict
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from reactive_agents.core.reasoning.engine import ReasoningEngine
    from reactive_agents.app.agents.reactive_agent import ReactiveAgent
    from reactive_agents.app.agents.base import Agent

from reactive_agents.config.mcp_config import MCPConfig
from reactive_agents.utils.logging import Logger
from reactive_agents.providers.llm.base import BaseModelProvider
from reactive_agents.providers.external.client import MCPClient
from reactive_agents.core.reasoning.prompts.agent_prompts import (
    REACT_AGENT_SYSTEM_PROMPT,
    CONTEXT_SUMMARIZATION_PROMPT,
)
from reactive_agents.core.types.status_types import TaskStatus

# --- Import Manager Classes ---
from reactive_agents.core.metrics.metrics_manager import MetricsManager
from reactive_agents.core.memory.memory_manager import MemoryManager
from reactive_agents.core.memory.vector_memory import VectorMemoryManager

# ReflectionManager is no longer used with simplified infrastructure
from reactive_agents.core.workflows.workflow_manager import WorkflowManager
from reactive_agents.core.tools.tool_manager import ToolManager

# --- Import AgentSession from its new location ---
from reactive_agents.core.types.session_types import AgentSession

# --- Import EventBus ---
from reactive_agents.core.events.event_bus import EventBus
from reactive_agents.core.types.event_types import AgentStateEvent

import tiktoken

# Add imports for new components
from reactive_agents.core.reasoning.task_classifier import TaskClassifier
from reactive_agents.config.settings import get_settings


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

    # Vector Memory Configuration
    vector_memory_enabled: bool = False
    vector_memory_collection: Optional[str] = None
    min_completion_score: float = 1.0
    max_iterations: Optional[int] = None
    max_task_retries: int = 3
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    enable_caching: bool = True
    cache_ttl: int = 3600
    offline_mode: bool = False

    # Context Management Configuration
    max_context_messages: int = 20
    max_context_tokens: Optional[int] = None
    enable_context_pruning: bool = True
    enable_context_summarization: bool = True
    context_pruning_strategy: Literal["conservative", "balanced", "aggressive"] = (
        "balanced"
    )
    # --- New Configurable Context Management Options ---
    context_token_budget: int = 4000
    context_pruning_aggressiveness: Literal[
        "conservative", "balanced", "aggressive"
    ] = "balanced"
    context_summarization_frequency: int = 3  # N iterations between summarizations

    # Response Format Configuration
    response_format: Optional[str] = None

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
    memory_manager: Optional[Union["MemoryManager", "VectorMemoryManager"]] = None
    workflow_manager: Optional["WorkflowManager"] = None
    tool_manager: Optional["ToolManager"] = None

    # --- Add Event Bus ---
    event_bus: Optional[EventBus] = None
    enable_state_observation: bool = True

    # Observability
    observability: Optional[Any] = Field(default=None)  # ContextObservabilityManager

    # Tool use policy: controls when tools are allowed in the agent loop
    tool_use_policy: Literal["always", "required_only", "adaptive", "never"] = (
        "adaptive"
    )

    # Maximum consecutive tool calls before forcing reflection/summary (used in adaptive tool use policy)
    tool_use_max_consecutive_calls: int = 3

    # New reasoning and classification components
    task_classifier: Optional["TaskClassifier"] = None
    reasoning_strategy: str = "reactive"
    enable_reactive_execution: bool = True
    enable_dynamic_strategy_switching: bool = True

    # Private attributes
    _agent: Optional["ReactiveAgent"] = (
        None  # Reference to the agent instance for strategies
    )
    _reasoning_engine: Optional[Any] = None  # Lazy-loaded reasoning engine

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
            successful_tools=set(),
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        # Call Pydantic's __init__ first to set up fields correctly
        super().__init__(**data)

        # Initialize observability manager
        # Remove initialization of observability field

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
            if self.vector_memory_enabled:
                self.agent_logger.info(
                    f"Initializing memory manager for {self.agent_name} with vector memory enabled"
                )
                # Import here to avoid circular imports
                from reactive_agents.core.memory.vector_memory import (
                    VectorMemoryManager,
                    VectorMemoryConfig,
                )
                from reactive_agents.config.settings import get_settings

                # Get settings for vector memory persist directory
                settings = get_settings()
                vector_persist_dir = str(settings.get_vector_memory_path())

                # Create vector memory configuration
                vector_config = VectorMemoryConfig(
                    collection_name=self.vector_memory_collection
                    or self.agent_name.replace(" ", "_").lower(),
                    persist_directory=vector_persist_dir,
                )

                self.memory_manager = VectorMemoryManager(
                    context=self, config=vector_config
                )
                self.agent_logger.info(
                    f"Initialized vector memory with collection: {vector_config.collection_name}"
                )
            else:
                self.agent_logger.info(
                    f"Initializing memory manager for {self.agent_name} with json memory enabled"
                )
                self.memory_manager = MemoryManager(context=self)
        else:
            self.agent_logger.info(f"Memory manager disabled for {self.agent_name}")
            self.memory_manager = None

        # Reflection is now handled by the simplified infrastructure

        self.workflow_manager = WorkflowManager(
            context=self,
            workflow_context=self.workflow_context_shared,
            workflow_dependencies=self.workflow_dependencies,
        )

        # Initialize new components
        if self.enable_reactive_execution:
            self.task_classifier = TaskClassifier(context=self)

        # Initialize event bus if enabled
        if self.enable_state_observation:
            self.event_bus = EventBus(self.agent_name)
            self.agent_logger.info("Event bus initialized.")

        # --- End Initialize Managers ---

        # Set agent name
        self.session.agent_name = self.agent_name

        # Initialize current task
        if not self.session.current_task and self.session.initial_task:
            self.session.current_task = self.session.initial_task

        self.agent_logger.info(
            f"AgentContext for '{self.agent_name}' initialized with managers."
        )

    # === Event System ===
    def emit_event(self, event_type: AgentStateEvent, data: Dict[str, Any]) -> None:
        """
        Emit an event to all registered callbacks.

        Args:
            event_type: The type of event being emitted
            data: The data associated with the event
        """
        if self.event_bus and self.enable_state_observation:
            # Include basic agent/session context with all events
            event_data = {
                "agent_name": self.agent_name,
                "session_id": getattr(self.session, "session_id", None),
                "task": getattr(self.session, "current_task", None),
                "task_status": str(getattr(self.session, "task_status", "unknown")),
                "iterations": getattr(self.session, "iterations", 0),
            }
            # Merge with event-specific data (event data takes precedence)
            event_data = {**event_data, **data}
            self.event_bus.emit(event_type, event_data)

    async def emit_event_async(
        self, event_type: AgentStateEvent, data: Dict[str, Any]
    ) -> None:
        """
        Emit an event to all registered async callbacks.

        Args:
            event_type: The type of event being emitted
            data: The data associated with the event
        """
        if self.event_bus and self.enable_state_observation:
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
            await self.event_bus.emit_async(event_type, event_data)

    # Methods to interact with components will be added later
    # e.g., get_tools(), update_metrics(), save_memory(), get_reflection() etc.

    async def close(self):
        """Safely close resources like the MCP client."""
        assert self.agent_logger is not None
        self.agent_logger.info(f"Closing context for {self.agent_name}...")
        # TODO: Add any other closing responsibilities for context
        self.agent_logger.info(f"{self.agent_name} context closed successfully.")

    # Convenience accessors (optional, direct access context.manager is also fine)
    def get_logger(self):
        if not self.agent_logger:
            raise RuntimeError("Logger is not initialized in this context.")
        return self.agent_logger

    def get_model_provider(self):
        if not self.model_provider:
            raise RuntimeError("ModelProvider is not initialized in this context.")
        return self.model_provider

    def get_tool_manager(self):
        if not self.tool_manager:
            raise RuntimeError("ToolManager is not initialized in this context.")
        return self.tool_manager

    def get_memory_manager(self):
        if not self.memory_manager:
            raise RuntimeError("MemoryManager is not initialized in this context.")
        return self.memory_manager

    def get_reflection_manager(self):
        # Reflection is now handled by the simplified infrastructure
        return None

    def get_workflow_manager(self):
        if not self.workflow_manager:
            raise RuntimeError("WorkflowManager is not initialized in this context.")
        return self.workflow_manager

    @property
    def reasoning_engine(self) -> ReasoningEngine:
        """Get the reasoning engine with lazy initialization."""
        if self._reasoning_engine is None:
            from reactive_agents.core.reasoning.engine import get_reasoning_engine

            self._reasoning_engine = get_reasoning_engine(self)
        return self._reasoning_engine

    def get_reasoning_engine(self):
        """Get the reasoning engine (convenience method)."""
        return self.reasoning_engine

    def get_tools(self):
        return self.tool_manager.get_available_tools() if self.tool_manager else []

    def get_tool_names(self):
        return self.tool_manager.get_available_tool_names() if self.tool_manager else []

    def get_tool_signatures(self):
        return self.tool_manager.tool_signatures if self.tool_manager else []

    def get_tool_by_name(self, name: str):
        if not self.tool_manager:
            return None
        for tool in self.tool_manager.tools:
            if getattr(tool, "name", None) == name:
                return tool
        return None

    def get_reflections(self):
        # Reflection is now handled by the simplified infrastructure
        return []

    def get_session_history(self):
        if self.memory_manager and hasattr(self.memory_manager, "get_session_history"):
            return self.memory_manager.get_session_history()
        return []

    def get_workflow_context(self):
        if self.workflow_manager and hasattr(self.workflow_manager, "get_full_context"):
            return self.workflow_manager.get_full_context()
        return None

    def get_metrics(self) -> Dict[str, Any]:
        if self.metrics_manager:
            return self.metrics_manager.get_metrics()
        return {}  # Return empty if metrics disabled

    def has_completed_required_tools(self) -> tuple[bool, set[str]]:
        """
        Check if all required tools (min_required_tools) have been completed (i.e., are in successful_tools).
        Returns a tuple (tools_completed: bool, missing_tools: set[str])
        """
        min_required_tools = self.session.min_required_tools or set()
        successful_tools = self.session.successful_tools
        if not min_required_tools:
            return True, set()
        missing_tools = min_required_tools - successful_tools
        tools_completed = len(missing_tools) == 0
        return tools_completed, missing_tools

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
        """Initialize the model provider."""
        try:
            from reactive_agents.providers.llm.factory import ModelProviderFactory

            # Create model provider with options
            self.model_provider = ModelProviderFactory.get_model_provider(
                self.provider_model_name,
                options=self.model_provider_options or {},
                context=self,
            )
            if self.agent_logger:
                self.agent_logger.info(
                    f"Initialized model provider: {self.provider_model_name}"
                )

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Failed to initialize model provider: {e}")
            raise RuntimeError(f"Model provider initialization failed: {e}")


# --- Rebuild Models to Resolve Forward References ---
# Call model_rebuild() on dependent models after AgentContext is defined
# This allows them to correctly resolve the 'AgentContext' forward reference.
MetricsManager.model_rebuild(force=True)
MemoryManager.model_rebuild(force=True)
WorkflowManager.model_rebuild(force=True)
ToolManager.model_rebuild(force=True)
# --- End Rebuild Models ---
