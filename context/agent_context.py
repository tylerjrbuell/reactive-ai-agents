from __future__ import annotations

import time
from typing import List, Dict, Any, Optional, Callable, Sequence, Set
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
from common.types import TaskStatus, AgentMemory

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

# --- End Import Manager Classes ---


class AgentSession(BaseModel):
    """Represents a single session of an agent run."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    final_result: str = ""
    total_tokens: int = 0
    total_cost: float = 0.0
    summary: str = ""
    evaluation: Dict[str, Any] = {}
    feasibility: str = ""
    result: str = ""
    error: Optional[str] = None
    status: str = ""


class AgentContext(BaseModel):
    """Centralized context holding state and components for an agent run."""

    # Core Agent Configuration
    agent_name: str
    provider_model_name: str  # e.g., "ollama/llama3"
    instructions: str = ""
    role: str = ""
    role_instructions: Dict[str, Any] = {}  # Role specific instructions

    # State Tracking
    initial_task: str = ""
    final_answer: Optional[str] = None
    messages: List[Dict[str, Any]] = []
    task_progress: str = "No progression yet"  # Summary of steps performed
    reasoning_log: List[str] = []  # For thought surfacing
    iterations: int = 0
    task_status: "TaskStatus" = TaskStatus.INITIALIZED
    current_task: str = ""  # Can be the initial task or a rescoped one
    min_required_tools: Optional[Set[str]] = None
    session: AgentSession = AgentSession()
    # --- Workflow Context and Dependencies ---
    # These are passed in or configured externally but used by WorkflowManager
    workflow_context_shared: Optional[Dict[str, Any]] = None  # The shared dict itself
    workflow_dependencies: List[str] = []  # Dependencies for *this* agent instance

    # Configuration Flags & Settings
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
        Callable[[str, Dict[str, Any]], bool | asyncio.Future[bool]]
    ] = None

    # Core Components (lazily initialized or passed in)
    model_provider: Optional[BaseModelProvider] = None
    mcp_client: Optional[MCPClient] = None
    tools: List[Any] = Field(default_factory=list)

    # Loggers
    agent_logger: Optional[Logger] = None
    tool_logger: Optional[Logger] = None
    result_logger: Optional[Logger] = None

    # --- Component Managers ---
    # Use forward references (strings) for type hints
    metrics_manager: Optional["MetricsManager"] = None
    memory_manager: Optional["MemoryManager"] = None
    reflection_manager: Optional["ReflectionManager"] = None
    workflow_manager: Optional["WorkflowManager"] = None
    tool_manager: Optional["ToolManager"] = None
    # --- End Component Managers ---

    # Other potential attributes derived from ReactAgent/Agent
    start_time: float = Field(default_factory=time.time)
    # workflow_dependencies: List[str] = [] # Managed by WorkflowManager now
    # reflections: List[Dict[str, Any]] = [] # Managed by ReflectionManager now
    # tool_history: List[Dict[str, Any]] = [] # Managed by ToolManager now
    # tool_signatures: List[Dict[str, Any]] = [] # Managed by ToolManager now

    class Config:
        arbitrary_types_allowed = (
            True  # Allow complex types like Logger, MCPClient etc.
        )

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
        # --- End Initialize Managers ---

        # Set initial system message
        if not self.messages:
            self.messages = [
                {"role": "system", "content": self._get_initial_system_prompt()}
            ]

        # Initialize current task
        if not self.current_task and self.initial_task:
            self.current_task = self.initial_task

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
            task=self.current_task,
            task_progress=self.task_progress,
        )
        return prompt

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
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = self._get_initial_system_prompt()

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
