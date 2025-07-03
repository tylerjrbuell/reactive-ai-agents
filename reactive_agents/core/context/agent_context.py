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

from pydantic import BaseModel, Field
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reactive_agents.app.agents.reactive_agent import ReactiveAgent

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
from reactive_agents.core.reasoning.reflection_manager import ReflectionManager
from reactive_agents.core.workflows.workflow_manager import WorkflowManager
from reactive_agents.core.tools.tool_manager import ToolManager

# --- Import AgentSession from its new location ---
from reactive_agents.core.types.session_types import AgentSession

# --- Import AgentStateObserver ---
from reactive_agents.core.events.agent_observer import AgentStateObserver
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
    reflection_manager: Optional["ReflectionManager"] = None
    workflow_manager: Optional["WorkflowManager"] = None
    tool_manager: Optional["ToolManager"] = None

    # --- Add State Observer ---
    state_observer: Optional[AgentStateObserver] = None
    enable_state_observation: bool = True

    # Tool use policy: controls when tools are allowed in the agent loop
    tool_use_policy: Literal["always", "required_only", "adaptive", "never"] = (
        "adaptive"
    )

    # Maximum consecutive tool calls before forcing reflection/summary (used in adaptive tool use policy)
    tool_use_max_consecutive_calls: int = 3

    # New reasoning and classification components
    task_classifier: Optional["TaskClassifier"] = None
    reasoning_strategy: str = "reflect_decide_act"  # Default strategy
    enable_reactive_execution: bool = True
    enable_dynamic_strategy_switching: bool = True

    # Private attributes
    _agent: Optional["ReactiveAgent"] = (
        None  # Reference to the agent instance for strategies
    )

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

        if self.reflect_enabled:
            self.reflection_manager = ReflectionManager(context=self)

        self.workflow_manager = WorkflowManager(
            context=self,
            workflow_context=self.workflow_context_shared,
            workflow_dependencies=self.workflow_dependencies,
        )

        # Initialize new components
        if self.enable_reactive_execution:
            self.task_classifier = TaskClassifier(context=self)

        # Initialize state observer if enabled
        if self.enable_state_observation:
            self.state_observer = AgentStateObserver()
            self.agent_logger.info("State observer initialized.")

        # --- End Initialize Managers ---

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
            from reactive_agents.providers.llm.factory import ModelProviderFactory

            self.model_provider = ModelProviderFactory.get_model_provider(
                self.provider_model_name,
                options=self.model_provider_options,
                context=self,
            )
            self.agent_logger.info(
                f"Model Provider Initialized: {self.model_provider.name}:{self.model_provider.model} with options: {self.model_provider.options}"
            )

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
        if not self.reflection_manager:
            raise RuntimeError("ReflectionManager is not initialized in this context.")
        return self.reflection_manager

    def get_workflow_manager(self):
        if not self.workflow_manager:
            raise RuntimeError("WorkflowManager is not initialized in this context.")
        return self.workflow_manager

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
        if self.reflection_manager:
            return self.reflection_manager.reflections
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

    async def summarize_memory(
        self, memory_content: str, memory_type: str = "session"
    ) -> str:
        """
        Summarize memory content using the LLM to create concise, valuable summaries.

        Args:
            memory_content: The raw memory content to summarize
            memory_type: Type of memory (session, reflection, tool_result, etc.)

        Returns:
            A concise summary of the memory content
        """
        if not self.model_provider:
            return (
                memory_content[:200] + "..."
                if len(memory_content) > 200
                else memory_content
            )

        try:
            # Create a memory-specific summarization prompt
            if memory_type == "session":
                summary_prompt = f"""
Summarize this agent session memory in 1-2 sentences. Focus on:
1. What task was accomplished
2. Key result or outcome
3. Any important tools or methods used

Memory content:
{memory_content}

Provide a concise summary that captures the essential information:
"""
            elif memory_type == "reflection":
                summary_prompt = f"""
Summarize this reflection memory in 1 sentence. Focus on:
1. Key insight or learning
2. What was learned or improved

Memory content:
{memory_content}

Provide a concise summary:
"""
            elif memory_type == "tool_result":
                summary_prompt = f"""
Summarize this tool result memory in 1 sentence. Focus on:
1. What tool was used
2. Key data or result obtained

Memory content:
{memory_content}

Provide a concise summary:
"""
            else:
                summary_prompt = f"""
Summarize this {memory_type} memory in 1-2 sentences. Focus on the most important information:

Memory content:
{memory_content}

Provide a concise summary:
"""

            response = await self.model_provider.get_completion(
                system="You are a memory summarization assistant. Create concise, valuable summaries that preserve key information while being easy to read and understand.",
                prompt=summary_prompt,
                options=self.model_provider_options,
            )

            if response and response.message.content:
                summary = response.message.content.strip()
                # Ensure summary is not too long
                if len(summary) > 300:
                    summary = summary[:297] + "..."
                return summary
            else:
                # Fallback to simple truncation
                return (
                    memory_content[:200] + "..."
                    if len(memory_content) > 200
                    else memory_content
                )

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.debug(f"Memory summarization failed: {e}")
            # Fallback to simple truncation
            return (
                memory_content[:200] + "..."
                if len(memory_content) > 200
                else memory_content
            )

    async def inject_memory_message(self, relevant_memories=None):
        """
        Insert or update a [MEMORY] assistant message after the system prompt.
        If relevant_memories is None, attempts to retrieve them for the current task.
        Uses LLM summarization to create concise, valuable memory summaries.
        """
        import asyncio

        if (
            relevant_memories is None
            and hasattr(self, "memory_manager")
            and self.memory_manager
        ):
            try:
                memory_manager = self.memory_manager
                if hasattr(memory_manager, "get_context_memories"):
                    try:
                        asyncio.get_running_loop()
                        # In event loop, skip sync retrieval
                        relevant_memories = []
                    except RuntimeError:
                        relevant_memories = asyncio.run(
                            memory_manager.get_context_memories(  # type: ignore
                                self.session.current_task, max_items=5
                            )
                        )
                else:
                    relevant_memories = []
            except Exception as e:
                if self.agent_logger:
                    self.agent_logger.debug(
                        f"Failed to retrieve memories for [MEMORY] message: {e}"
                    )
                relevant_memories = []
        elif relevant_memories is None:
            relevant_memories = []

            # Format the memory message with LLM summarization
        if relevant_memories:
            memory_lines = ["[MEMORY] Relevant past experiences:"]
            for i, mem in enumerate(
                relevant_memories[:3], 1
            ):  # Limit to 3 most relevant
                # Extract key information
                metadata = mem.get("metadata", {})
                memory_type = metadata.get("memory_type", "unknown")
                relevance = mem.get("relevance_score", 0)
                content = mem.get("content", "")

                # Use LLM summarization for all memory types
                try:
                    summary = await self.summarize_memory(content, memory_type)
                    memory_lines.append(
                        f"{i}. [{memory_type.upper()}] {summary} (relevance: {relevance:.1f})"
                    )
                except Exception as e:
                    if self.agent_logger:
                        self.agent_logger.debug(
                            f"Memory summarization failed for item {i}: {e}"
                        )
                    # Fallback to simple preview
                    content_preview = content[:100].replace("\n", " ")
                    memory_lines.append(
                        f"{i}. [{memory_type.upper()}] {content_preview}... (relevance: {relevance:.1f})"
                    )
            memory_message = "\n".join(memory_lines)
            if self.agent_logger:
                self.agent_logger.debug(f"[MEMORY] Message content: {memory_message}")
        else:
            memory_message = "[MEMORY] No relevant past experiences found."

        messages = self.session.messages
        # Ensure system message is first
        if not messages or messages[0].get("role") != "system":
            self.update_system_prompt()
        # Insert or update the [MEMORY] message at index 1
        if (
            len(messages) > 1
            and messages[1].get("role") == "assistant"
            and messages[1].get("content", "").startswith("[MEMORY]")
        ):
            messages[1]["content"] = memory_message
            if self.agent_logger:
                self.agent_logger.info("Updated [MEMORY] message at index 1.")
        else:
            messages.insert(1, {"role": "assistant", "content": memory_message})
            if self.agent_logger:
                self.agent_logger.info("Inserted new [MEMORY] message at index 1.")

    def update_system_prompt(self):
        """Regenerate and set the system prompt as the first message in the session."""
        messages = self.session.messages
        from reactive_agents.core.reasoning.prompts.base import SystemPrompt

        system_prompt = SystemPrompt(self)
        system_content = system_prompt.generate(
            task=self.session.current_task,
        )
        system_message = {
            "role": "system",
            "content": system_content,
        }
        # Remove any existing system message
        messages[:] = [m for m in messages if m.get("role") != "system"]
        # Insert the regenerated system message at the beginning
        messages.insert(0, system_message)
        if self.agent_logger:
            self.agent_logger.info(
                "Set the system message as the first message (minimal/static)"
            )

    def _generate_system_prompt(self) -> str:
        """Generate a dynamic system prompt with current context."""
        current_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        current_day_of_week = datetime.now().strftime("%A")
        current_timezone = datetime.now(timezone.utc).strftime("%Z")
        model_info = f"{self.provider_model_name}"
        if self.model_provider:
            model_info = f"{self.model_provider.name}:{self.model_provider.model}"
        return REACT_AGENT_SYSTEM_PROMPT.format(
            role=self.role,
            instructions=self.instructions,
            model_info=model_info,
            current_datetime=current_datetime,
            current_day_of_week=current_day_of_week,
            current_timezone=current_timezone,
            response_format=self.response_format
            or "Provide a clear and concise answer.",
        )

    def get_token_encoding(self):
        """
        Select the correct tiktoken encoding for the current model/provider.
        """
        model_name = (
            self.provider_model_name.lower() if self.provider_model_name else ""
        )
        # OpenAI models
        if "gpt-4" in model_name or "gpt-3.5" in model_name:
            return tiktoken.encoding_for_model(model_name)
        # Anthropic (use cl100k_base as a fallback)
        if "claude" in model_name or "anthropic" in model_name:
            return tiktoken.get_encoding("cl100k_base")
        # Groq (Llama3, Mixtral, etc. - fallback to cl100k_base)
        if "groq" in model_name or "llama" in model_name or "mixtral" in model_name:
            return tiktoken.get_encoding("cl100k_base")
        # Ollama (local models, fallback to cl100k_base)
        if "ollama" in model_name:
            return tiktoken.get_encoding("cl100k_base")
        # Default fallback
        return tiktoken.get_encoding("cl100k_base")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a given text using tiktoken.
        """
        encoding = self.get_token_encoding()
        return len(encoding.encode(text or ""))

    def estimate_context_tokens(self) -> int:
        """
        Estimate total token count for current context using tiktoken.
        """
        encoding = self.get_token_encoding()
        total_tokens = 0
        for message in self.session.messages:
            # For OpenAI-style chat models, tiktoken expects a list of dicts with 'role' and 'content'
            if isinstance(message, dict) and "content" in message:
                total_tokens += len(encoding.encode(message["content"] or ""))
        return total_tokens

    def should_prune_context(
        self, max_messages: Optional[int] = None, max_tokens: Optional[int] = None
    ) -> bool:
        """
        Check if context pruning is needed based on configured strategy.

        Args:
            max_messages: Override for max_context_messages
            max_tokens: Override for max_context_tokens

        Returns:
            True if pruning is needed, False otherwise
        """
        if not self.enable_context_pruning:
            return False

        max_messages = max_messages or self.max_context_messages
        max_tokens = max_tokens or self.max_context_tokens

        # Check message count
        if len(self.session.messages) > max_messages:
            return True

        # Check token count if configured
        if max_tokens and self.estimate_context_tokens() > max_tokens:
            return True

        return False

    def get_optimal_pruning_config(self) -> Dict[str, Any]:
        """
        Get optimal pruning configuration based on provider capabilities and new config options.
        """
        if not self.model_provider:
            return {
                "max_messages": self.max_context_messages,
                "max_tokens": self.max_context_tokens or self.context_token_budget,
                "strategy": self.context_pruning_strategy,
            }
        provider_name = self.model_provider.name.lower()
        config = {
            "max_messages": 20,
            "max_tokens": self.context_token_budget,
            "strategy": self.context_pruning_strategy,
        }
        # Provider-specific optimizations
        if "openai" in provider_name:
            config.update(
                {"max_messages": 30, "max_tokens": 8000, "strategy": "adaptive"}
            )
        elif "anthropic" in provider_name:
            config.update(
                {"max_messages": 35, "max_tokens": 100000, "strategy": "token_based"}
            )
        elif "ollama" in provider_name:
            config.update(
                {"max_messages": 15, "max_tokens": 4000, "strategy": "message_count"}
            )
        elif "groq" in provider_name:
            config.update(
                {"max_messages": 20, "max_tokens": 8000, "strategy": "adaptive"}
            )
        # Aggressiveness adjustment
        if self.context_pruning_strategy == "aggressive":
            config["max_messages"] = max(5, config["max_messages"] // 2)
            config["max_tokens"] = max(1000, config["max_tokens"] // 2)
        elif self.context_pruning_strategy == "conservative":
            config["max_messages"] = int(config["max_messages"] * 1.5)
            config["max_tokens"] = int(config["max_tokens"] * 1.5)
        # Override with user settings if provided
        if self.max_context_messages != 20:
            config["max_messages"] = self.max_context_messages
        if self.max_context_tokens:
            config["max_tokens"] = self.max_context_tokens
        if self.context_pruning_strategy != "message_count":
            config["strategy"] = self.context_pruning_strategy
        return config

    def get_context_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current context.

        Returns:
            Dictionary with context statistics
        """
        messages = self.session.messages
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        non_system_messages = [msg for msg in messages if msg["role"] != "system"]

        return {
            "total_messages": len(messages),
            "system_messages": len(system_messages),
            "conversation_messages": len(non_system_messages),
            "has_system_message": len(system_messages) > 0,
            "system_message_first": (
                len(messages) > 0 and messages[0]["role"] == "system"
                if messages
                else False
            ),
        }

    async def manage_context(self):
        """
        Streamlined context management: summarizes and/or prunes context as needed.
        Ensures system message is first, keeps most recent user message, last summary, and current assistant turn.
        Summarizes tool/assistant messages into a single summary message when needed, and prunes detailed messages that have been summarized.
        """
        config = self.get_optimal_pruning_config()
        should_summarize, should_prune = self.summarize_and_prune_context(
            self.session.iterations
        )

        # Always ensure system message is up to date and first
        self.update_system_prompt()
        self.inject_context_message()
        messages = self.session.messages
        system_message = messages[0]
        non_system_messages = messages[1:]

        # Identify the most recent user message
        user_messages = [m for m in non_system_messages if m.get("role") == "user"]
        last_user_message = user_messages[-1] if user_messages else None

        # Identify the most recent summary message (assistant, content starts with [SUMMARY])
        summary_messages = [
            i
            for i, m in enumerate(self.session.messages)
            if m.get("role") == "assistant"
            and m.get("content", "").strip().startswith("[SUMMARY]")
        ]
        last_summary_index = summary_messages[-1] if summary_messages else 0
        last_summary_message = (
            self.session.messages[last_summary_index] if summary_messages else None
        )
        # Store last summary index in session for future incremental summarization
        self.session.last_summary_index = last_summary_index

        # Identify the current assistant turn (last assistant message not a summary)
        assistant_messages = [
            m
            for m in non_system_messages
            if m.get("role") == "assistant"
            and not m.get("content", "").strip().startswith("[SUMMARY]")
        ]
        current_assistant_turn = assistant_messages[-1] if assistant_messages else None

        # If summarization is needed, perform incremental/rolling summary
        if should_summarize:
            # Only summarize messages after the last summary (or after system if no summary yet)
            start_idx = last_summary_index + 1 if summary_messages else 1
            # Find messages to summarize: tool or assistant messages (including tool summaries)
            to_summarize = [
                m
                for m in self.session.messages[start_idx:]
                if m.get("role") in ("assistant", "tool")
                and m is not current_assistant_turn
            ]
            summary_text = "\n".join(
                f"[{m['role']}] {m['content']}"
                for m in to_summarize
                if m.get("content")
            )
            summary_content = ""
            if summary_text.strip() and self.model_provider:
                summary_prompt = CONTEXT_SUMMARIZATION_PROMPT + "\n\n" + summary_text
                try:
                    response = await self.model_provider.get_completion(
                        system="You are a summarization assistant for an AI agent.",
                        prompt=summary_prompt,
                        options=self.model_provider_options,
                    )
                    summary_content = (
                        response.message.content.strip()
                        if response
                        else "Summary unavailable."
                    )
                except Exception as e:
                    if self.agent_logger:
                        self.agent_logger.error(f"Summarization failed: {e}")
                    summary_content = "Summary unavailable due to error."
            summary_message = {
                "role": "assistant",
                "content": f"[SUMMARY OF EARLIER CONTEXT]\n{summary_content}",
            }
            # Build new message list: keep all messages up to last summary, insert new summary, then keep unsummarized messages
            new_messages = self.session.messages[:start_idx]
            new_messages.append(summary_message)
            # Keep any messages after the summarized block that are not tool/assistant (e.g., user, current assistant turn)
            for m in self.session.messages[start_idx:]:
                if m not in to_summarize:
                    new_messages.append(m)
            self.session.messages = new_messages
            # Update last_summary_index
            self.session.last_summary_index = len(new_messages) - 1
            if self.agent_logger:
                self.agent_logger.info(
                    f"Incrementally summarized context to {len(new_messages)} messages (~{self.estimate_context_tokens()} tokens). Last summary index: {self.session.last_summary_index}"
                )
            return  # If summarized, no further pruning needed in this pass

        # If only pruning is needed (not summarizing), keep system, last user, last summary, and current assistant turn
        if should_prune:
            new_messages = [system_message]
            if last_user_message:
                new_messages.append(last_user_message)
            if last_summary_message:
                new_messages.append(last_summary_message)
            if current_assistant_turn:
                new_messages.append(current_assistant_turn)
            self.session.messages = new_messages
            if self.agent_logger:
                self.agent_logger.info(
                    f"Pruned context to {len(new_messages)} messages (~{self.estimate_context_tokens()} tokens)."
                )

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

    def summarize_and_prune_context(self, current_iteration: int = 0):
        """
        Check if summarization/pruning should occur based on frequency, token budget, and aggressiveness.
        """
        config = self.get_optimal_pruning_config()
        should_summarize = self.enable_context_summarization and (
            current_iteration % self.context_summarization_frequency == 0
        )
        should_prune = self.enable_context_pruning and (
            self.estimate_context_tokens() > config["max_tokens"]
            or len(self.session.messages) > config["max_messages"]
        )
        if self.agent_logger:
            self.agent_logger.debug(
                f"summarize_and_prune_context: iteration={current_iteration}, "
                f"should_summarize={should_summarize}, should_prune={should_prune}, "
                f"tokens={self.estimate_context_tokens()}, max_tokens={config['max_tokens']}, "
                f"messages={len(self.session.messages)}, max_messages={config['max_messages']}"
            )
        return should_summarize, should_prune

    def inject_context_message(self):
        """
        Insert or update a minimal [CONTEXT] assistant message after the system prompt.
        Only includes non-empty sections: required tools, completed tools, task nudges, recent reasoning.
        """
        session = self.session
        sections = []
        if session.min_required_tools:
            sections.append(f"Required Tools: {', '.join(session.min_required_tools)}")
        if session.successful_tools:
            sections.append(f"Completed Tools: {', '.join(session.successful_tools)}")
        if session.task_nudges:
            sections.append(f"Task Reminders: {'; '.join(set(session.task_nudges))}")
        if session.reasoning_log:
            recent_reasoning = session.reasoning_log[-2:]
            sections.append(f"Recent Reasoning: {'; '.join(recent_reasoning)}")
        context_content = (
            "[CONTEXT]\n" + "\n".join(sections)
            if sections
            else "[CONTEXT]\nNo additional context."
        )
        messages = self.session.messages
        # Check if a [CONTEXT] message already exists at index 1
        if (
            len(messages) > 1
            and messages[1].get("role") == "assistant"
            and messages[1].get("content", "").startswith("[CONTEXT]")
        ):
            messages[1]["content"] = context_content
            if self.agent_logger:
                self.agent_logger.info("Updated [CONTEXT] message at index 1.")
        else:
            messages.insert(1, {"role": "assistant", "content": context_content})
            if self.agent_logger:
                self.agent_logger.info("Inserted new [CONTEXT] message at index 1.")


# --- Rebuild Models to Resolve Forward References ---
# Call model_rebuild() on dependent models after AgentContext is defined
# This allows them to correctly resolve the 'AgentContext' forward reference.
MetricsManager.model_rebuild(force=True)
MemoryManager.model_rebuild(force=True)
ReflectionManager.model_rebuild(force=True)
WorkflowManager.model_rebuild(force=True)
ToolManager.model_rebuild(force=True)
# --- End Rebuild Models ---
