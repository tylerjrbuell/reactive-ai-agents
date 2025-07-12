"""
Builder module for creating agent instances with a fluent interface.

This module provides builder classes and utility functions to simplify
the creation and configuration of agents.
"""

from typing import (
    Any,
    Dict,
    Awaitable,
    Callable,
    Optional,
    List,
    Tuple,
    Union,
    TypeVar,
    Set,
    Literal,
    TYPE_CHECKING,
)
import asyncio
from typing_extensions import Literal
from pydantic import BaseModel, Field

from reactive_agents.providers.external.client import MCPClient
from reactive_agents.core.types.confirmation_types import ConfirmationCallbackProtocol
from reactive_agents.config.logging import LogLevel, formatter
from reactive_agents.utils.logging import Logger
from reactive_agents.config.mcp_config import MCPConfig
from reactive_agents.app.agents.reactive_agent import ReactiveAgent
from reactive_agents.core.tools.base import Tool
from reactive_agents.core.types.event_types import AgentStateEvent
from reactive_agents.core.events.agent_events import (
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
    EventCallback,
    AsyncEventCallback,
)
from reactive_agents.core.types.agent_types import ReactAgentConfig
from reactive_agents.core.types.reasoning_types import ReasoningStrategies
from reactive_agents.config.natural_language_config import create_agent_from_nl


# Define type variables for better type hinting
T = TypeVar("T")
ReactiveAgentBuilderT = TypeVar("ReactiveAgentBuilderT", bound="ReactiveAgentBuilder")

# Import the reusable ReasoningStrategies type
from reactive_agents.core.types.reasoning_types import ReasoningStrategies

ReasoningStrategyType = ReasoningStrategies


class ToolConfig(BaseModel):
    """Configuration for a tool"""

    name: str
    is_custom: bool = False
    description: Optional[str] = None
    source: str = "unknown"


class ConfirmationConfig(BaseModel):
    """Configuration for the confirmation system"""

    enabled: bool = True
    strategy: str = "always"
    excluded_tools: List[str] = Field(default_factory=list)
    included_tools: Optional[List[str]] = None
    allowed_silent_tools: List[str] = Field(default_factory=list)
    timeout: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True


# Convenience function for the absolute simplest agent creation
async def quick_create_agent(
    task: str,
    model: str = "ollama:cogito:14b",
    tools: List[str] = ["brave-search", "time"],
    interactive: bool = False,
) -> Dict[str, Any]:
    """
    Create and run a ReactiveAgent with minimal configuration

    This is the simplest possible way to create and run an agent.

    Args:
        task: The task for the agent to perform
        model: The model to use
        tools: List of tool names to include
        interactive: Whether to require confirmation for tool usage
        use_reactive_v2: Whether to use ReactiveAgentV2 (default: True)

    Returns:
        The result dictionary from the agent run
    """
    # Create simple confirmation callback if interactive is True
    confirmation_callback = None
    if interactive:

        async def simple_callback(
            action_description: str, details: Dict[str, Any]
        ) -> bool:
            print(f"\nTool: {details.get('tool', 'unknown')}")
            user_input = input("Proceed? (y/n) [y]: ").lower().strip()
            return user_input == "y" or user_input == ""

        confirmation_callback = simple_callback

    # Create and run the agent (default to ReactiveAgent)
    agent = await ReactiveAgentBuilder().with_model(model).with_mcp_tools(tools).build()

    if confirmation_callback:
        agent.context.confirmation_callback = confirmation_callback

    try:
        return await agent.run(initial_task=task)
    finally:
        await agent.close()


class ReactiveAgentBuilder:
    """
    Unified builder class for creating ReactiveAgent instances with full framework integration.

    This class provides comprehensive support for:
    - Dynamic reasoning strategies (reflect_decide_act, plan_execute_reflect, reactive, adaptive)
    - Task classification and adaptive strategy switching
    - Natural language configuration
    - Vector memory integration
    - Enhanced event system with dynamic event handlers
    - Advanced tool management (MCP + custom tools)
    - Real-time control operations (pause, resume, stop, terminate)
    - Comprehensive context management

    Examples:
        Basic reactive agent:
        ```python
        agent = await (ReactiveAgentBuilder()
                .with_name("Reactive Agent")
                .with_model("ollama:qwen3:4b")
                .with_reasoning_strategy("reflect_decide_act")
                .with_mcp_tools(["brave-search", "sqlite"])
                .build())
        ```

        Adaptive agent with strategy switching:
        ```python
        agent = await (ReactiveAgentBuilder()
                .with_name("Adaptive Agent")
                .with_reasoning_strategy("adaptive")
                .with_dynamic_strategy_switching(True)
                .with_mcp_tools(["brave-search", "time"])
                .build())
        ```

        Natural language configuration:
        ```python
        agent = await (ReactiveAgentBuilder()
                .with_natural_language_config(
                    "Create an agent that can research topics and analyze data"
                )
                .build())
        ```

        With vector memory:
        ```python
        agent = await (ReactiveAgentBuilder()
                .with_name("Memory Agent")
                .with_vector_memory("research_memories")
                .with_reasoning_strategy("plan_execute_reflect")
                .build())
        ```

        Event-driven agent:
        ```python
        agent = await (ReactiveAgentBuilder()
                .with_name("Event Agent")
                .with_reasoning_strategy("reflect_decide_act")
                .on_tool_called(lambda event: print(f"Tool: {event['tool_name']}"))
                .on_session_started(lambda event: print("Session started"))
                .build())
        ```
    """

    def __init__(self):
        # Initialize with comprehensive defaults
        self._config = {
            "agent_name": "ReactiveAgent",
            "role": "Enhanced Task Executor",
            "provider_model_name": "ollama:cogito:14b",
            "model_provider_options": {},
            "mcp_client": None,
            "min_completion_score": 1.0,
            "instructions": "Solve tasks efficiently using dynamic reasoning strategies.",
            "max_iterations": 10,
            "reflect_enabled": True,
            "log_level": "info",
            "initial_task": None,
            "tool_use_enabled": True,
            "use_memory_enabled": True,
            "collect_metrics_enabled": True,
            "check_tool_feasibility": True,
            "enable_caching": True,
            "confirmation_callback": None,
            "confirmation_config": {},
            "max_context_messages": 20,
            "max_context_tokens": None,
            "enable_context_pruning": True,
            "enable_context_summarization": True,
            "context_pruning_strategy": "balanced",
            "context_token_budget": 4000,
            "context_pruning_aggressiveness": "balanced",
            "context_summarization_frequency": 3,
            "tool_use_policy": "adaptive",
            "tool_use_max_consecutive_calls": 3,
            # Advanced configuration
            "reasoning_strategy": "reactive",  # Changed from reflect_decide_act
            "enable_reactive_execution": True,
            "enable_dynamic_strategy_switching": True,
            "kwargs": {},
        }
        self._mcp_client: Optional[MCPClient] = None
        self._mcp_config: Optional[MCPConfig] = None
        self._mcp_server_filter: Optional[List[str]] = None
        self._custom_tools: List[Any] = []
        self._registered_tools: Set[str] = set()
        self._logger = Logger(
            "ReactiveAgentBuilder", "builder", self._config.get("log_level", "info")
        )
        self._logger.formatter = formatter

        # Advanced features
        self._vector_memory_enabled: bool = False
        self._vector_memory_collection: Optional[str] = None
        self._natural_language_config: Optional[str] = None

    # Basic configuration methods
    def with_name(self, name: str) -> "ReactiveAgentBuilder":
        """Set the agent's name"""
        self._config["agent_name"] = name
        return self

    def with_role(self, role: str) -> "ReactiveAgentBuilder":
        """Set the agent's role"""
        self._config["role"] = role
        return self

    def with_model(self, model_name: str) -> "ReactiveAgentBuilder":
        """Set the model to use for the agent"""
        self._config["provider_model_name"] = model_name
        return self

    def with_model_provider_options(
        self, options: Dict[str, Any]
    ) -> "ReactiveAgentBuilder":
        """Set the model provider options for the agent"""
        self._config["model_provider_options"] = options
        return self

    def with_instructions(self, instructions: str) -> "ReactiveAgentBuilder":
        """Set the agent's instructions"""
        self._config["instructions"] = instructions
        return self

    def with_max_iterations(self, max_iterations: int) -> "ReactiveAgentBuilder":
        """Set the maximum number of iterations for the agent"""
        self._config["max_iterations"] = max_iterations
        return self

    def with_reflection(self, enabled: bool = True) -> "ReactiveAgentBuilder":
        """Enable or disable reflection"""
        self._config["reflect_enabled"] = enabled
        return self

    def with_log_level(self, level: Union[LogLevel, str]) -> "ReactiveAgentBuilder":
        """Set the log level (debug, info, warning, error, critical)"""
        if isinstance(level, LogLevel):
            level = level.value
        self._config["log_level"] = level
        return self

    # Advanced reasoning strategy methods
    def with_reasoning_strategy(
        self,
        strategy: ReasoningStrategies = ReasoningStrategies.ADAPTIVE,
    ) -> "ReactiveAgentBuilder":
        """
        Set the initial reasoning strategy for the agent.

        Available strategies are dynamically discovered from the ReasoningStrategies enum.
        Default: "adaptive"

        Common strategies:
        - "reflect_decide_act" - Reflect, decide, then act (most robust)
        - "plan_execute_reflect" - Plan first, execute, then reflect
        - "reactive" - Quick reactive responses (fastest)
        - "adaptive" - Switch strategies based on task complexity
        """
        # Type safety is handled by the Literal type, just store the strategy
        self._config["reasoning_strategy"] = strategy
        return self

    @staticmethod
    def get_available_strategies() -> List[str]:
        """
        Get a list of all available reasoning strategies.

        Returns:
            List[str]: List of available strategy names

        Example:
            ```python
            strategies = ReactiveAgentBuilder.get_available_strategies()
            print(f"Available strategies: {strategies}")
            ```
        """
        from reactive_agents.core.types.reasoning_types import ReasoningStrategies

        return [strategy.value for strategy in ReasoningStrategies]

    @staticmethod
    def get_strategy_descriptions() -> Dict[str, str]:
        """
        Get descriptions of all available reasoning strategies.

        Returns:
            Dict[str, str]: Dictionary mapping strategy names to descriptions

        Example:
            ```python
            descriptions = ReactiveAgentBuilder.get_strategy_descriptions()
            for strategy, desc in descriptions.items():
                print(f"{strategy}: {desc}")
            ```
        """
        return {
            "reflect_decide_act": "Reflect, decide, then act (most robust, good for complex tasks)",
            "plan_execute_reflect": "Plan first, execute, then reflect (good for structured tasks)",
            "reactive": "Quick reactive responses (fastest, good for simple queries)",
            "adaptive": "Switch strategies based on task complexity (most flexible)",
            "memory_enhanced": "Enhanced with memory capabilities (good for context-heavy tasks)",
        }

    class Strategies:
        """Static class providing autocomplete for available reasoning strategies."""

        @staticmethod
        def get_all() -> List[str]:
            """Get all available strategy names."""
            from reactive_agents.core.types.reasoning_types import ReasoningStrategies

            return [strategy.value for strategy in ReasoningStrategies]

        # Static attributes for autocomplete
        REFLECT_DECIDE_ACT = "reflect_decide_act"
        PLAN_EXECUTE_REFLECT = "plan_execute_reflect"
        REACTIVE = "reactive"
        ADAPTIVE = "adaptive"
        MEMORY_ENHANCED = "memory_enhanced"
        SELF_ASK = "self_ask"
        GOAL_ACTION_FEEDBACK = "goal_action_feedback"

    def with_dynamic_strategy_switching(
        self, enabled: bool = True
    ) -> "ReactiveAgentBuilder":
        """Enable or disable dynamic reasoning strategy switching during execution"""
        self._config["enable_dynamic_strategy_switching"] = enabled
        return self

    def with_reactive_execution(self, enabled: bool = True) -> "ReactiveAgentBuilder":
        """Enable or disable reactive execution engine"""
        self._config["enable_reactive_execution"] = enabled
        return self

    # Natural language and vector memory configuration
    def with_natural_language_config(self, description: str) -> "ReactiveAgentBuilder":
        """
        Configure the agent using natural language description.

        This will use the natural language configuration parser to automatically
        set up the agent based on the description.

        Args:
            description: Natural language description of the desired agent
        """
        self._natural_language_config = description
        return self

    def with_vector_memory(
        self, collection_name: Optional[str] = None
    ) -> "ReactiveAgentBuilder":
        """
        Enable ChromaDB vector memory for semantic memory search.

        Args:
            collection_name: Name of the ChromaDB collection (defaults to agent_name)
        """
        self._vector_memory_enabled = True
        self._vector_memory_collection = collection_name
        return self

    # Tool configuration methods
    def with_mcp_tools(self, server_filter: List[str]) -> "ReactiveAgentBuilder":
        """
        Configure the MCP client with specific server-side tools

        Args:
            server_filter: List of MCP tool names to include
        """
        self._mcp_server_filter = server_filter
        self._config["mcp_server_filter"] = server_filter
        # warn of servers not found
        if self._mcp_config:
            for server_name in server_filter:
                if server_name not in self._mcp_config.mcpServers.keys():
                    self._logger.warning(
                        f"Server {server_name} not found in MCP config skipping..."
                    )

        # Track MCP tools for debugging
        for tool_name in server_filter:
            self._registered_tools.add(f"mcp:{tool_name}")
        return self

    def with_custom_tools(self, tools: List[Any]) -> "ReactiveAgentBuilder":
        """
        Add custom tools to the agent

        These tools should be decorated with the @tool() decorator from tools.decorators

        Args:
            tools: List of custom tool functions or objects
        """
        for tool in tools:
            # If it's already a Tool instance, add it directly
            if hasattr(tool, "name") and hasattr(tool, "tool_definition"):
                self._custom_tools.append(tool)
                self._registered_tools.add(
                    f"custom:{getattr(tool, 'name', str(id(tool)))}"
                )
            # If it's a function decorated with @tool(), wrap it in a Tool class
            elif hasattr(tool, "tool_definition"):
                wrapped_tool = Tool(tool)
                # Ensure the name is preserved from the function
                if not hasattr(wrapped_tool, "name") and hasattr(tool, "__name__"):
                    wrapped_tool.name = tool.__name__
                self._custom_tools.append(wrapped_tool)
                self._registered_tools.add(f"custom:{wrapped_tool.name}")
            else:
                raise ValueError(
                    f"Custom tool {tool.__name__ if hasattr(tool, '__name__') else tool} "
                    f"is not properly decorated with @tool()"
                )

        # Track tool registrations for debugging
        if not hasattr(self, "_debug_registered_tools"):
            self._debug_registered_tools = []
        for tool in self._custom_tools:
            tool_name = getattr(tool, "name", getattr(tool, "__name__", str(id(tool))))
            self._debug_registered_tools.append(f"Custom: {tool_name}")

        return self

    def with_tool_use(self, tool_use: bool = True) -> "ReactiveAgentBuilder":
        self._config["tool_use_enabled"] = tool_use
        return self

    def with_tools(
        self,
        mcp_tools: Optional[List[str]] = None,
        custom_tools: Optional[List[Any]] = None,
    ) -> "ReactiveAgentBuilder":
        """
        Configure both MCP and custom tools at once

        This is a convenience method that combines with_mcp_tools and with_custom_tools

        Args:
            mcp_tools: List of MCP tool names to include
            custom_tools: List of custom tool functions or objects
        """
        # Handle MCP tools
        if mcp_tools:
            self.with_mcp_tools(mcp_tools)

        # Handle custom tools
        if custom_tools:
            self.with_custom_tools(custom_tools)

        # Add metadata to help track tool sources
        self._config["hybrid_tools_config"] = {
            "mcp_tools": mcp_tools or [],
            "custom_tools_count": len(custom_tools) if custom_tools else 0,
        }

        return self

    def with_tool_caching(self, enabled: bool = True) -> "ReactiveAgentBuilder":
        """Enable or disable tool caching"""
        self._config["enable_caching"] = enabled
        return self

    def with_mcp_client(self, mcp_client: MCPClient) -> "ReactiveAgentBuilder":
        """
        Use a pre-configured MCP client

        This allows using an MCP client that has already been initialized
        with specific configurations.

        Args:
            mcp_client: An initialized MCPClient instance
        """
        self._mcp_client = mcp_client
        self._config["mcp_client"] = mcp_client
        return self

    def with_mcp_config(self, mcp_config: MCPConfig) -> "ReactiveAgentBuilder":
        """
        Use an MCP server configuration

        This allows using an MCP client that has already been initialized
        with specific configurations.

        Args:
            mcp_config: An initialized MCPConfig instance
        """
        self._mcp_config = mcp_config
        self._config["mcp_config"] = mcp_config
        return self

    def with_confirmation(
        self,
        callback: ConfirmationCallbackProtocol,
        config: Optional[Union[Dict[str, Any], ConfirmationConfig]] = None,
    ) -> "ReactiveAgentBuilder":
        """
        Configure the confirmation system

        Args:
            callback: The confirmation callback function
            config: Optional configuration for the confirmation system
        """
        self._config["confirmation_callback"] = callback

        if config:
            # Convert Pydantic model to dict if needed
            if isinstance(config, ConfirmationConfig):
                config = config.dict()
            self._config["confirmation_config"] = config

        return self

    def with_advanced_config(self, **kwargs) -> "ReactiveAgentBuilder":
        """
        Set any configuration options directly

        This allows setting any configuration options that don't have specific methods
        """
        self._config.update(kwargs)
        return self

    def with_workflow_context(self, context: Dict[str, Any]) -> "ReactiveAgentBuilder":
        """Set shared workflow context data"""
        self._config["workflow_context_shared"] = context
        return self

    def with_response_format(self, format_spec: str) -> "ReactiveAgentBuilder":
        """Set the response format specification for the agent's final answer"""
        self._config["response_format"] = format_spec
        return self

    def with_max_context_messages(self, value: int) -> "ReactiveAgentBuilder":
        """Set the maximum number of context messages to retain."""
        self._config["max_context_messages"] = value
        return self

    def with_max_context_tokens(self, value: int) -> "ReactiveAgentBuilder":
        """Set the maximum number of context tokens to retain."""
        self._config["max_context_tokens"] = value
        return self

    def with_context_pruning(self, value: bool = True) -> "ReactiveAgentBuilder":
        """Enable or disable context pruning."""
        self._config["enable_context_pruning"] = value
        return self

    def with_context_summarization(self, value: bool = True) -> "ReactiveAgentBuilder":
        """Enable or disable context summarization."""
        self._config["enable_context_summarization"] = value
        return self

    def with_context_pruning_strategy(self, value: str) -> "ReactiveAgentBuilder":
        """Set the context pruning strategy ('conservative', 'balanced', 'aggressive')."""
        self._config["context_pruning_strategy"] = value
        return self

    def with_context_token_budget(self, value: int) -> "ReactiveAgentBuilder":
        """Set the token budget for context management."""
        self._config["context_token_budget"] = value
        return self

    def with_context_pruning_aggressiveness(self, value: str) -> "ReactiveAgentBuilder":
        """Set the aggressiveness of context pruning ('conservative', 'balanced', 'aggressive')."""
        self._config["context_pruning_aggressiveness"] = value
        return self

    def with_context_summarization_frequency(
        self, value: int
    ) -> "ReactiveAgentBuilder":
        """Set the number of iterations between context summarizations."""
        self._config["context_summarization_frequency"] = value
        return self

    def with_tool_use_policy(self, value: str) -> "ReactiveAgentBuilder":
        """Set the tool use policy ('always', 'required_only', 'adaptive', 'never')."""
        self._config["tool_use_policy"] = value
        return self

    def with_tool_use_max_consecutive_calls(self, value: int) -> "ReactiveAgentBuilder":
        """Set the maximum consecutive tool calls before forcing reflection/summarization."""
        self._config["tool_use_max_consecutive_calls"] = value
        return self

    # Factory methods for common agent types
    @classmethod
    async def research_agent(
        cls,
        model: Optional[str] = None,
    ) -> ReactiveAgent:
        """
        Create a pre-configured research agent optimized for information gathering

        Args:
            model: Optional model name to use (default: ollama:cogito:14b)
        """
        builder = cls()
        if model:
            builder.with_model(model)

        return await (
            builder.with_name("Research Agent")
            .with_role("Research Assistant")
            .with_instructions(
                "Research information thoroughly and provide accurate results."
            )
            .with_mcp_tools(["brave-search", "time"])
            .with_reflection(True)
            .with_max_iterations(15)
            .build()
        )

    @classmethod
    async def database_agent(cls, model: Optional[str] = None) -> ReactiveAgent:
        """
        Create a pre-configured database agent optimized for database operations

        Args:
            model: Optional model name to use (default: ollama:cogito:14b)
        """
        builder = cls()
        if model:
            builder.with_model(model)

        return await (
            builder.with_name("Database Agent")
            .with_role("Database Assistant")
            .with_instructions(
                "Perform database operations accurately and efficiently."
            )
            .with_mcp_tools(["sqlite"])
            .with_reflection(True)
            .build()
        )

    @classmethod
    async def crypto_research_agent(
        cls,
        model: Optional[str] = None,
        confirmation_callback: Optional[Callable] = None,
        cryptocurrencies: Optional[List[str]] = None,
    ) -> ReactiveAgent:
        """
        Create a specialized agent for cryptocurrency research and data collection

        Args:
            model: Optional model name to use (default: ollama:cogito:14b)
            confirmation_callback: Optional callback for confirming sensitive operations
            cryptocurrencies: List of cryptocurrencies to track (default: ["Bitcoin", "Ethereum"])
        """
        builder = cls()
        if model:
            builder.with_model(model)

        # Use default list if none provided
        cryptocurrencies = cryptocurrencies or ["Bitcoin", "Ethereum"]

        # Build specialized instructions for crypto research
        crypto_instructions = (
            f"Research current prices for cryptocurrencies and maintain accurate records in a database. "
            f"Focus on these cryptocurrencies: {', '.join(cryptocurrencies)}. "
            f"When researching prices, ensure you get the most current data and properly format the date. "
            f"Always verify data before adding it to the database. Create any necessary tables if they don't exist."
        )

        # Configure the builder
        builder = (
            builder.with_name("Crypto Research Agent")
            .with_role("Financial Data Analyst")
            .with_instructions(crypto_instructions)
            .with_mcp_tools(["brave-search", "sqlite", "time"])
            .with_reflection(True)
            .with_max_iterations(15)
        )

        # Add confirmation callback if provided
        if confirmation_callback:
            builder.with_confirmation(confirmation_callback)

        return await builder.build()

    @classmethod
    async def reactive_research_agent(
        cls, model: Optional[str] = None
    ) -> ReactiveAgent:
        """Create a pre-configured reactive research agent optimized for information gathering"""
        builder = cls()
        if model:
            builder.with_model(model)

        return await (
            builder.with_name("Reactive Research Agent")
            .with_role("Advanced Research Assistant")
            .with_reasoning_strategy(ReasoningStrategies.REFLECT_DECIDE_ACT)
            .with_instructions(
                "Research information thoroughly using dynamic reasoning strategies."
            )
            .with_mcp_tools(["brave-search", "time"])
            .with_reflection(True)
            .with_max_iterations(15)
            .with_dynamic_strategy_switching(True)
            .build()
        )

    @classmethod
    async def adaptive_agent(cls, model: Optional[str] = None) -> ReactiveAgent:
        """Create a pre-configured adaptive agent that switches strategies based on task complexity"""
        builder = cls()
        if model:
            builder.with_model(model)

        return await (
            builder.with_name("Adaptive Agent")
            .with_role("Adaptive Task Executor")
            .with_reasoning_strategy(ReasoningStrategies.ADAPTIVE)
            .with_instructions(
                "Adapt reasoning strategy based on task complexity and requirements."
            )
            .with_mcp_tools(["brave-search", "time", "sqlite"])
            .with_reflection(True)
            .with_max_iterations(20)
            .with_dynamic_strategy_switching(True)
            .build()
        )

    @classmethod
    async def add_custom_tools_to_agent(
        cls, agent: ReactiveAgent, custom_tools: List[Any]
    ) -> ReactiveAgent:
        """
        Add custom tools to an existing agent instance

        This utility method provides a clean way to add custom tools to an agent
        that has already been created, such as one from a factory method.

        Args:
            agent: An existing ReactiveAgent instance
            custom_tools: List of custom tool functions or objects

        Returns:
            The updated agent with the new tools added

        Example:
            ```python
            agent = await ReactiveAgentBuilder.research_agent()
            updated_agent = await ReactiveAgentBuilder.add_custom_tools_to_agent(
                agent, [my_custom_tool]
            )
            ```
        """
        if not agent or not hasattr(agent, "context"):
            raise ValueError("Invalid agent provided")

        # Process and wrap the custom tools if needed
        processed_tools = []
        for tool in custom_tools:
            if hasattr(tool, "name") and hasattr(tool, "tool_definition"):
                processed_tools.append(tool)
            elif hasattr(tool, "tool_definition"):
                processed_tools.append(Tool(tool))
            else:
                raise ValueError(
                    f"Custom tool {tool.__name__ if hasattr(tool, '__name__') else tool} "
                    f"is not properly decorated with @tool()"
                )

        # Check for the tools attribute and add tools
        if hasattr(agent.context, "tools"):
            for tool in processed_tools:
                agent.context.tools.append(tool)

        # Update the tool manager if it exists
        tool_manager = getattr(agent.context, "tool_manager", None)
        if tool_manager is not None:
            # Give a small delay to ensure setup is complete
            await asyncio.sleep(0.1)

            # Add tools to the manager
            for tool in processed_tools:
                tool_manager.tools.append(tool)

            # Update tool signatures if possible
            generate_signatures = getattr(
                tool_manager, "_generate_tool_signatures", None
            )
            if callable(generate_signatures):
                generate_signatures()

        return agent

    # Diagnostic methods
    def debug_tools(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the tools configured for this agent

        This method helps with debugging tool registration issues by providing
        information about which tools are registered and their sources.

        Returns:
            Dict[str, Any]: Diagnostic information about registered tools

        Example:
            ```python
            builder = ReactiveAgentBuilder()
                .with_mcp_tools(["brave-search", "time"])
                .with_custom_tools([my_custom_tool])

            # Debug before building
            tool_info = builder.debug_tools()
            print(f"MCP Tools: {tool_info['mcp_tools']}")
            print(f"Custom Tools: {tool_info['custom_tools']}")
            ```
        """
        mcp_tools = []
        custom_tools = []

        # Extract tool info from registered tools
        for tool_id in self._registered_tools:
            if tool_id.startswith("mcp:"):
                mcp_tools.append(tool_id.split(":", 1)[1])
            elif tool_id.startswith("custom:"):
                custom_tools.append(tool_id.split(":", 1)[1])

        # Get tool info from custom tools list
        custom_tool_details = []
        for tool in self._custom_tools:
            tool_name = getattr(tool, "name", getattr(tool, "__name__", str(id(tool))))
            tool_details = {
                "name": tool_name,
                "has_name_attr": hasattr(tool, "name"),
                "has_tool_definition": hasattr(tool, "tool_definition"),
                "type": type(tool).__name__,
            }
            custom_tool_details.append(tool_details)

        return {
            "mcp_tools": mcp_tools,
            "custom_tools": custom_tools,
            "custom_tool_details": custom_tool_details,
            "mcp_client_initialized": self._mcp_client is not None,
            "server_filter": self._mcp_server_filter,
            "total_tools": len(self._registered_tools),
        }

    @staticmethod
    async def diagnose_agent_tools(agent: ReactiveAgent) -> Dict[str, Any]:
        """
        Diagnose tool registration issues in an existing agent

        This static method examines an agent that has already been created
        to check for tool registration issues and provides detailed diagnostics.

        Args:
            agent: The ReactiveAgent instance to diagnose

        Returns:
            Dict[str, Any]: Diagnostic information about the agent's tools

        Example:
            ```python
            agent = await ReactiveAgentBuilder().with_mcp_tools(["brave-search"]).build()

            # Diagnose after building
            diagnosis = await ReactiveAgentBuilder.diagnose_agent_tools(agent)
            if diagnosis["has_tool_mismatch"]:
                print("Warning: Tool registration mismatch detected!")
                print(f"Tools in context: {diagnosis['context_tools']}")
                print(f"Tools in manager: {diagnosis['manager_tools']}")
            ```
        """
        if not agent or not hasattr(agent, "context"):
            return {"error": "Invalid agent or no context attribute"}

        context = agent.context
        tool_manager = getattr(context, "tool_manager", None)
        context_tools = getattr(context, "tools", [])

        # Get tools from context
        context_tool_names = []
        for tool in context_tools:
            tool_name = getattr(tool, "name", str(id(tool)))
            context_tool_names.append(tool_name)

        # Get tools from tool manager
        manager_tool_names = []
        if tool_manager:
            for tool in getattr(tool_manager, "tools", []):
                tool_name = getattr(tool, "name", str(id(tool)))
                manager_tool_names.append(tool_name)

        # Check for mismatches
        context_set = set(context_tool_names)
        manager_set = set(manager_tool_names)

        missing_in_context = manager_set - context_set
        missing_in_manager = context_set - manager_set

        has_mismatch = len(missing_in_context) > 0 or len(missing_in_manager) > 0

        return {
            "context_tools": context_tool_names,
            "manager_tools": manager_tool_names,
            "has_tool_mismatch": has_mismatch,
            "missing_in_context": list(missing_in_context),
            "missing_in_manager": list(missing_in_manager),
            "has_mcp_client": hasattr(context, "mcp_client")
            and context.mcp_client is not None,
            "has_custom_tools": hasattr(agent, "_has_custom_tools"),
        }

    # Event subscription methods
    def with_subscription(
        self, event_type: AgentStateEvent, callback: EventCallback[Any]
    ) -> "ReactiveAgentBuilder":
        """
        Register a callback function for any event type using a more generic interface.

        This provides a more dynamic way to subscribe to events without using specific helper methods.

        Args:
            event_type: The type of event to observe (from AgentStateEvent enum)
            callback: The callback function to invoke when the event occurs

        Returns:
            self for method chaining

        Example:
            ```python
            builder = (ReactiveAgentBuilder()
                .with_subscription(
                    AgentStateEvent.TOOL_CALLED,
                    lambda event: print(f"Tool called: {event['tool_name']}")
                )
                .build())
            ```
        """
        return self.with_event_callback(event_type, callback)

    def with_async_subscription(
        self, event_type: AgentStateEvent, callback: AsyncEventCallback[Any]
    ) -> "ReactiveAgentBuilder":
        """
        Register an async callback function for any event type using a more generic interface.

        This provides a more dynamic way to subscribe to async events without using specific helper methods.

        Args:
            event_type: The type of event to observe (from AgentStateEvent enum)
            callback: The async callback function to invoke when the event occurs

        Returns:
            self for method chaining

        Example:
            ```python
            async def log_tool_call(event):
                await db.log_event(event['tool_name'], event['parameters'])

            builder = (ReactiveAgentBuilder()
                .with_async_subscription(
                    AgentStateEvent.TOOL_CALLED,
                    log_tool_call
                )
                .build())
            ```
        """
        return self.with_async_event_callback(event_type, callback)

    def with_event_callback(
        self, event_type: AgentStateEvent, callback: EventCallback
    ) -> "ReactiveAgentBuilder":
        """
        Register a callback function for a specific event type.

        This allows setting up event observers before the agent is built.

        Args:
            event_type: The type of event to observe
            callback: The callback function to invoke when the event occurs

        Returns:
            self for method chaining

        Example:
            ```python
            builder = (ReactiveAgentBuilder()
                .with_event_callback(
                    AgentStateEvent.TOOL_CALLED,
                    lambda event: print(f"Tool called: {event['tool_name']}")
                )
                .build())
            ```
        """
        if not hasattr(self, "_event_callbacks"):
            self._event_callbacks = {}

        if event_type not in self._event_callbacks:
            self._event_callbacks[event_type] = []

        self._event_callbacks[event_type].append(callback)
        return self

    def with_async_event_callback(
        self, event_type: AgentStateEvent, callback: AsyncEventCallback
    ) -> "ReactiveAgentBuilder":
        """
        Register an async callback function for a specific event type.

        This allows setting up async event observers before the agent is built.

        Args:
            event_type: The type of event to observe
            callback: The async callback function to invoke when the event occurs

        Returns:
            self for method chaining

        Example:
            ```python
            async def log_tool_call(event):
                await db.log_event(event['tool_name'], event['parameters'])

            builder = (ReactiveAgentBuilder()
                .with_async_event_callback(
                    AgentStateEvent.TOOL_CALLED,
                    log_tool_call
                )
                .build())
            ```
        """
        if not hasattr(self, "_async_event_callbacks"):
            self._async_event_callbacks = {}

        if event_type not in self._async_event_callbacks:
            self._async_event_callbacks[event_type] = []

        self._async_event_callbacks[event_type].append(callback)
        return self

    # Convenience methods for specific event types
    def on_session_started(
        self, callback: EventCallback[SessionStartedEventData]
    ) -> "ReactiveAgentBuilder":
        """Register a callback for session started events"""
        return self.with_event_callback(AgentStateEvent.SESSION_STARTED, callback)

    def on_session_ended(
        self, callback: EventCallback[SessionEndedEventData]
    ) -> "ReactiveAgentBuilder":
        """Register a callback for session ended events"""
        return self.with_event_callback(AgentStateEvent.SESSION_ENDED, callback)

    def on_task_status_changed(
        self, callback: EventCallback[TaskStatusChangedEventData]
    ) -> "ReactiveAgentBuilder":
        """Register a callback for task status changed events"""
        return self.with_event_callback(AgentStateEvent.TASK_STATUS_CHANGED, callback)

    def on_iteration_started(
        self, callback: EventCallback[IterationStartedEventData]
    ) -> "ReactiveAgentBuilder":
        """Register a callback for iteration started events"""
        return self.with_event_callback(AgentStateEvent.ITERATION_STARTED, callback)

    def on_iteration_completed(
        self, callback: EventCallback[IterationCompletedEventData]
    ) -> "ReactiveAgentBuilder":
        """Register a callback for iteration completed events"""
        return self.with_event_callback(AgentStateEvent.ITERATION_COMPLETED, callback)

    def on_tool_called(
        self, callback: EventCallback[ToolCalledEventData]
    ) -> "ReactiveAgentBuilder":
        """Register a callback for tool called events"""
        return self.with_event_callback(AgentStateEvent.TOOL_CALLED, callback)

    def on_tool_completed(
        self, callback: EventCallback[ToolCompletedEventData]
    ) -> "ReactiveAgentBuilder":
        """Register a callback for tool completed events"""
        return self.with_event_callback(AgentStateEvent.TOOL_COMPLETED, callback)

    def on_tool_failed(
        self, callback: EventCallback[ToolFailedEventData]
    ) -> "ReactiveAgentBuilder":
        """Register a callback for tool failed events"""
        return self.with_event_callback(AgentStateEvent.TOOL_FAILED, callback)

    def on_reflection_generated(
        self, callback: EventCallback[ReflectionGeneratedEventData]
    ) -> "ReactiveAgentBuilder":
        """Register a callback for reflection generated events"""
        return self.with_event_callback(AgentStateEvent.REFLECTION_GENERATED, callback)

    def on_final_answer_set(
        self, callback: EventCallback[FinalAnswerSetEventData]
    ) -> "ReactiveAgentBuilder":
        """Register a callback for final answer set events"""
        return self.with_event_callback(AgentStateEvent.FINAL_ANSWER_SET, callback)

    def on_metrics_updated(
        self, callback: EventCallback[MetricsUpdatedEventData]
    ) -> "ReactiveAgentBuilder":
        """Register a callback for metrics updated events"""
        return self.with_event_callback(AgentStateEvent.METRICS_UPDATED, callback)

    def on_error_occurred(
        self, callback: EventCallback[ErrorOccurredEventData]
    ) -> "ReactiveAgentBuilder":
        """Register a callback for error occurred events"""
        return self.with_event_callback(AgentStateEvent.ERROR_OCCURRED, callback)

    def on_session_started_async(
        self, callback: AsyncEventCallback[SessionStartedEventData]
    ) -> "ReactiveAgentBuilder":
        """Register an async callback for session started events"""
        return self.with_async_event_callback(AgentStateEvent.SESSION_STARTED, callback)

    def on_session_ended_async(
        self, callback: AsyncEventCallback[SessionEndedEventData]
    ) -> "ReactiveAgentBuilder":
        """Register an async callback for session ended events"""
        return self.with_async_event_callback(AgentStateEvent.SESSION_ENDED, callback)

    # Build method
    async def build(self) -> ReactiveAgent:
        """
        Build and return a configured ReactiveAgent instance

        Returns:
            ReactiveAgent: A fully configured agent ready to use
        """
        # Process natural language configuration if provided
        if self._natural_language_config:
            try:
                # Skip natural language config processing for now to avoid complexity
                # TODO: Implement proper natural language config processing with model provider
                self._logger.info(
                    f"Natural language config provided: {self._natural_language_config[:100]}..."
                )
                self._logger.warning(
                    "Natural language config processing not yet implemented"
                )
                # TODO: Implement natural language config processing
                # Merge natural language configuration with current config
                # for key, value in nl_config.dict().items():
                #     if value is not None and key not in [
                #         "agent_name",
                #         "role",
                #     ]:  # Preserve explicit settings
                #         self._config[key] = value0
            except Exception as e:
                self._logger.warning(f"Failed to process natural language config: {e}")

        # Configure vector memory if enabled
        if self._vector_memory_enabled:
            collection_name = (
                self._vector_memory_collection or self._config["agent_name"]
            )
            # Configure vector memory settings
            self._config["vector_memory_enabled"] = True
            self._config["vector_memory_collection"] = collection_name
            self._config["use_memory_enabled"] = True
            self._logger.info(
                f"Vector memory enabled with collection: {collection_name}"
            )

        # Add custom tools to the configuration
        if self._custom_tools:
            self._config["tools"] = self._custom_tools

        try:
            # Create ReactAgentConfig and ReactiveAgent
            agent_config = ReactAgentConfig(**self._config)
            agent = ReactiveAgent(config=agent_config)
            await agent.initialize()

            # Set up event callbacks if any were registered
            if hasattr(self, "_event_callbacks"):
                for event_type, callbacks in self._event_callbacks.items():
                    for callback in callbacks:
                        # Use the dynamic event system
                        handler_name = f"on_{event_type.value}"
                        if hasattr(agent, handler_name):
                            handler = getattr(agent, handler_name)
                            handler(callback)

            if hasattr(self, "_async_event_callbacks"):
                for event_type, callbacks in self._async_event_callbacks.items():
                    for callback in callbacks:
                        # Use the dynamic event system
                        handler_name = f"on_{event_type.value}"
                        if hasattr(agent, handler_name):
                            handler = getattr(agent, handler_name)
                            handler(callback)

            return agent

        except Exception as e:
            # Clean up MCP client if agent creation fails
            if self._mcp_client is not None:
                try:
                    await self._mcp_client.close()
                except Exception as cleanup_error:
                    print(
                        f"Error closing MCP client during error handling: {cleanup_error}"
                    )

            raise RuntimeError(f"Failed to create ReactiveAgent: {e}") from e
