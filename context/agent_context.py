from __future__ import annotations

import time
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Callable,
    Sequence,
    Set,
    Awaitable,
    Union,
    Tuple,
)
import asyncio
import uuid

from pydantic import BaseModel, Field

# Placeholder imports - will be replaced with actual component classes
# from components.metrics_manager import MetricsManager
# from components.memory_manager import MemoryManager
# from components.reflection_manager import ReflectionManager
# from components.workflow_manager import WorkflowManager
# from components.tool_manager import ToolManager
from loggers.base import Logger
from model_providers.base import BaseModelProvider
from agent_mcp.client import MCPClient
from prompts.agent_prompts import REACT_AGENT_SYSTEM_PROMPT
from tools.abstractions import ToolProtocol
from common.types import TaskStatus

# Forward references for type hinting (No longer strictly needed with direct imports, but can keep for clarity)
# MetricsManager = Any # Now imported
# MemoryManager = Any # Now imported
# ReflectionManager = Any # Now imported
# WorkflowManager = Any # Now imported
# ToolManager = Any # Now imported

# --- Import Manager Classes ---
from components.metrics_manager import MetricsManager
from components.memory_manager import MemoryManager
from components.reflection_manager import ReflectionManager
from components.workflow_manager import WorkflowManager
from components.tool_manager import ToolManager

# --- Import AgentSession from its new location ---
from .session import AgentSession

# --- Import AgentStateObserver ---
from .agent_observer import AgentStateObserver, AgentStateEvent


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
    log_level: str = "info"
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
    mcp_client: Optional[MCPClient] = None
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

    # Session State Holder (Reference to the current run's state)
    session: AgentSession = Field(default_factory=AgentSession)

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
            from model_providers.factory import ModelProviderFactory

            self.model_provider = ModelProviderFactory.get_model_provider(
                self.provider_model_name
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
            task_progress=self.session.task_progress,
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
        self.agent_logger.info(f"Closing AgentContext for '{self.agent_name}'...")
        if self.mcp_client:
            try:
                # Use the safe close method from the original Agent class logic
                loop = asyncio.get_running_loop()
                close_task = loop.create_task(self.mcp_client.close())
                await asyncio.wait_for(close_task, timeout=5.0)
                self.agent_logger.info("MCP client closed successfully.")
            except asyncio.TimeoutError:
                self.agent_logger.warning("MCP client close timed out.")
            except asyncio.CancelledError:
                self.agent_logger.warning("MCP client closure cancelled.")
                # Consider detaching if necessary, similar to original logic
            except Exception as e:
                self.agent_logger.error(f"Error closing MCP client: {e}")
            finally:
                self.mcp_client = None
        self.agent_logger.info(f"AgentContext for '{self.agent_name}' closed.")

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

    def update_system_prompt(self):
        if self.session.messages and self.session.messages[0]["role"] == "system":
            self.session.messages[0]["content"] = self._get_initial_system_prompt()

    def save_memory_if_enabled(self):
        if self.memory_manager:
            self.memory_manager.save_memory()


# --- Rebuild Models to Resolve Forward References ---
# Call model_rebuild() on dependent models after AgentContext is defined
# This allows them to correctly resolve the 'AgentContext' forward reference.
MetricsManager.model_rebuild(force=True)
MemoryManager.model_rebuild(force=True)
ReflectionManager.model_rebuild(force=True)
WorkflowManager.model_rebuild(force=True)
ToolManager.model_rebuild(force=True)
# --- End Rebuild Models ---
