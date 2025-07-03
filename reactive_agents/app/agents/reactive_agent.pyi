# Type stub file for ReactiveAgent
from typing import Any, Dict, List, Optional, Callable

# Import context and related types
from reactive_agents.core.context.agent_context import AgentContext
from reactive_agents.core.types.agent_types import ReactAgentConfig
from reactive_agents.utils.logging import Logger
from reactive_agents.providers.llm.base import BaseModelProvider

class ReactiveAgent:
    """
    Reactive Agent with dynamic event handling capabilities.

    This agent provides dynamic event subscription methods that are created at runtime.
    All event methods follow the pattern: on_<event_name>(callback)

    The dynamic event methods are created automatically based on the AgentStateEvent enum,
    so no manual updates to this file are needed when adding new events.
    """

    # Base agent attributes
    context: AgentContext
    execution_engine: Optional[Any] = None
    _closed: bool = False
    _initialized: bool = False
    config: Optional[Any] = None

    def __init__(self, config: ReactAgentConfig) -> None:
        """Initialize the ReactiveAgent with enhanced reactive capabilities."""
        ...
    # --- Base agent properties ---
    @property
    def agent_logger(self) -> Logger:
        """Get the agent logger."""
        ...

    @property
    def tool_logger(self) -> Logger:
        """Get the tool logger."""
        ...

    @property
    def result_logger(self) -> Logger:
        """Get the result logger."""
        ...

    @property
    def model_provider(self) -> BaseModelProvider:
        """Get the model provider."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Check if the agent is initialized."""
        ...

    @property
    def is_closed(self) -> bool:
        """Check if the agent is closed."""
        ...
    # --- Base agent abstract methods ---
    async def _execute_task(self, task: str, **kwargs) -> Dict[str, Any]:
        """Execute a task. This is the core method that subclasses must implement."""
        ...
    # --- Base agent core methods ---
    async def _think(self, **kwargs) -> dict | None:
        """Directly calls the model provider for a simple completion."""
        ...

    def _should_use_tools(self, tool_calls: List[Dict[str, Any]]) -> bool:
        """Decide whether to allow tool use in the next think_chain call."""
        ...

    async def _think_chain(
        self,
        remember_messages: bool = True,
        use_tools: bool = True,
        **kwargs,
    ) -> dict | None:
        """Performs a chat completion with tool support and message management."""
        ...

    async def _process_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> list | None:
        """Process tool calls and execute them."""
        ...
    # --- Lifecycle methods (inherited from base) ---
    async def initialize(self) -> "ReactiveAgent":
        """Initialize the ReactiveAgent and wait for vector memory to be ready."""
        ...

    async def close(self) -> None:
        """Close the agent and clean up resources."""
        ...
    # --- Control methods (inherited from base) ---
    async def pause(self) -> None:
        """Pause the agent execution with event emission."""
        ...

    async def resume(self) -> None:
        """Resume the agent execution with event emission."""
        ...

    async def stop(self) -> None:
        """Stop the agent execution with event emission."""
        ...

    async def terminate(self) -> None:
        """Terminate the agent execution with event emission."""
        ...
    # --- Core run method (inherited from base) ---
    async def run(self, initial_task: str, **kwargs) -> Dict[str, Any]:
        """Run the agent with a task."""
        ...
    # --- ReactiveAgent specific methods ---
    async def run_with_strategy(
        self,
        initial_task: str,
        strategy: str = "reflect_decide_act",
        cancellation_event: Any = None,
    ) -> Dict[str, Any]:
        """Run the agent with a specific reasoning strategy."""
        ...

    def get_available_strategies(self) -> List[str]:
        """Get list of available reasoning strategies."""
        ...

    def get_current_strategy(self) -> str:
        """Get the currently active reasoning strategy."""
        ...

    def get_reasoning_context(self) -> Dict[str, Any]:
        """Get the current reasoning context information."""
        ...

    async def search_memory(
        self, query: str, n_results: int = 5, memory_types: List[str] | None = None
    ) -> List[Dict[str, Any]]:
        """Search through agent memories using semantic similarity."""
        ...

    async def get_context_memories(
        self, task: str, max_items: int = 10
    ) -> List[Dict[str, Any]]:
        """Get memories relevant to the current task context."""
        ...

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        ...

    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the agent."""
        ...

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        ...

    def get_available_events(self) -> List[str]:
        """Get list of all available event types for subscription."""
        ...

    def subscribe_to_all_events(self, callback: Any) -> Dict[str, Any]:
        """Subscribe to all available events with a single callback."""
        ...
    # --- Context manager support (inherited from base) ---
    async def __aenter__(self) -> "ReactiveAgent":
        """Async context manager entry - initialize the agent."""
        ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with cleanup."""
        ...
    # --- Dynamic event handler registration ---
    def __getattr__(self, name: str) -> Any:
        """
        Dynamic event handler registration.

        This enables dynamic event subscription using the pattern:
        agent.on_<event_name>(callback)

        The actual event methods are created dynamically at runtime based on
        the AgentStateEvent enum, so no manual updates to this file are needed
        when adding new events.
        """
        ...
