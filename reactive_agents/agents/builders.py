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
)
import asyncio
from pydantic import BaseModel, Field

from reactive_agents.agent_mcp.client import MCPClient
from reactive_agents.common.types.confirmation_types import ConfirmationCallbackProtocol
from reactive_agents.config.logging import LogLevel, formatter
from reactive_agents.loggers.base import Logger
from reactive_agents.config.mcp_config import MCPConfig
from .react_agent import ReactAgent, ReactAgentConfig
from reactive_agents.tools.base import Tool
from reactive_agents.context.agent_observer import AgentStateEvent
from reactive_agents.context.agent_events import (
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


# Define type variables for better type hinting
T = TypeVar("T")
ReactAgentBuilderT = TypeVar("ReactAgentBuilderT", bound="ReactAgentBuilder")


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
    model: str = "ollama:qwen2:7b",
    tools: List[str] = ["brave-search", "time"],
    interactive: bool = False,
) -> Dict[str, Any]:
    """
    Create and run a ReactAgent with minimal configuration

    This is the simplest possible way to create and run an agent.

    Args:
        task: The task for the agent to perform
        model: The model to use
        tools: List of tool names to include
        interactive: Whether to require confirmation for tool usage

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

    # Create and run the agent
    agent = await ReactAgentBuilder().with_model(model).with_mcp_tools(tools).build()
    if confirmation_callback:
        agent.context.confirmation_callback = confirmation_callback

    try:
        return await agent.run(initial_task=task)
    finally:
        await agent.close()


class ReactAgentBuilder:
    """
    A builder class for creating ReactAgent instances with a fluent interface.

    This class simplifies the process of creating and configuring ReactAgents by:
    - Providing sensible defaults
    - Offering a fluent interface for configuration
    - Including preset configurations for common use cases

    Examples:
        Basic usage:
        ```python
        agent = await (ReactAgentBuilder()
                .with_name("My Agent")
                .with_model("ollama:qwen3:4b")
                .with_mcp_tools(["brave-search", "sqlite"])
                .build())
        ```

        Using custom tools:
        ```python
        @tool()
        async def my_custom_tool(param1: str) -> str:
            return f"Processed {param1}"

        agent = await (ReactAgentBuilder()
                .with_name("Custom Tool Agent")
                .with_model("ollama:qwen3:4b")
                .with_custom_tools([my_custom_tool])
                .build())
        ```

        Using both MCP and custom tools:
        ```python
        agent = await (ReactAgentBuilder()
                .with_name("Hybrid Agent")
                .with_model("ollama:qwen3:4b")
                .with_mcp_tools(["brave-search"])
                .with_custom_tools([my_custom_tool])
                .build())
        ```

        Using presets:
        ```python
        agent = await ReactAgentBuilder.research_agent(model="ollama:qwen3:4b")
        ```

        Ultra-simple usage:
        ```python
        result = await quick_create_agent("Research the current price of Bitcoin")
        ```
    """

    def __init__(self):
        # Initialize with default values
        self._config = {
            "agent_name": "ReactAgent",
            "role": "Task Executor",
            "provider_model_name": "ollama:qwen2:7b",
            "model_provider_options": {},
            "mcp_client": None,
            "min_completion_score": 1.0,
            "instructions": "Solve the given task as efficiently as possible.",
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
            "kwargs": {},
        }
        self._mcp_client: Optional[MCPClient] = None
        self._mcp_config: Optional[MCPConfig] = None
        self._mcp_server_filter: Optional[List[str]] = None
        self._custom_tools: List[Any] = []
        self._registered_tools: Set[str] = set()
        self._logger = Logger("ReactAgentBuilder", "builder", self._config["log_level"])
        self._logger.formatter = formatter

    # Basic configuration methods

    def with_name(self, name: str) -> "ReactAgentBuilder":
        """Set the agent's name"""
        self._config["agent_name"] = name
        return self

    def with_role(self, role: str) -> "ReactAgentBuilder":
        """Set the agent's role"""
        self._config["role"] = role
        return self

    def with_model(self, model_name: str) -> "ReactAgentBuilder":
        """Set the model to use for the agent"""
        self._config["provider_model_name"] = model_name
        return self

    def with_model_provider_options(
        self, options: Dict[str, Any]
    ) -> "ReactAgentBuilder":
        """Set the model provider options for the agent"""
        self._config["model_provider_options"] = options
        return self

    def with_instructions(self, instructions: str) -> "ReactAgentBuilder":
        """Set the agent's instructions"""
        self._config["instructions"] = instructions
        return self

    def with_max_iterations(self, max_iterations: int) -> "ReactAgentBuilder":
        """Set the maximum number of iterations for the agent"""
        self._config["max_iterations"] = max_iterations
        return self

    def with_reflection(self, enabled: bool = True) -> "ReactAgentBuilder":
        """Enable or disable reflection"""
        self._config["reflect_enabled"] = enabled
        return self

    def with_log_level(self, level: Union[LogLevel, str]) -> "ReactAgentBuilder":
        """Set the log level (debug, info, warning, error, critical)"""
        if isinstance(level, LogLevel):
            level = level.value
        self._config["log_level"] = level
        return self

    # Updated tool methods

    def with_mcp_tools(self, server_filter: List[str]) -> "ReactAgentBuilder":
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

    def with_custom_tools(self, tools: List[Any]) -> "ReactAgentBuilder":
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

    def with_tool_use(self, tool_use: bool = True) -> "ReactAgentBuilder":
        self._config["tool_use_enabled"] = tool_use
        return self

    def with_tools(
        self,
        mcp_tools: Optional[List[str]] = None,
        custom_tools: Optional[List[Any]] = None,
    ) -> "ReactAgentBuilder":
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

    def with_tool_caching(self, enabled: bool = True) -> "ReactAgentBuilder":
        """Enable or disable tool caching"""
        self._config["enable_caching"] = enabled
        return self

    def with_mcp_client(self, mcp_client: MCPClient) -> "ReactAgentBuilder":
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

    def with_mcp_config(self, mcp_config: MCPConfig) -> "ReactAgentBuilder":
        """
        Use an MCP server configuration

        This allows using an MCP client that has already been initialized
        with specific configurations.

        Args:
            mcp_client: An initialized MCPClient instance
        """
        self._mcp_config = mcp_config
        self._config["mcp_config"] = mcp_config
        return self

    def with_confirmation(
        self,
        callback: ConfirmationCallbackProtocol,
        config: Optional[Union[Dict[str, Any], ConfirmationConfig]] = None,
    ) -> "ReactAgentBuilder":
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

    def with_advanced_config(self, **kwargs) -> "ReactAgentBuilder":
        """
        Set any configuration options directly

        This allows setting any configuration options that don't have specific methods
        """
        self._config.update(kwargs)
        return self

    def with_workflow_context(self, context: Dict[str, Any]) -> "ReactAgentBuilder":
        """Set shared workflow context data"""
        self._config["workflow_context_shared"] = context
        return self

    def with_response_format(self, format_spec: str) -> "ReactAgentBuilder":
        """Set the response format specification for the agent's final answer"""
        self._config["response_format"] = format_spec
        return self

    def with_max_context_messages(self, value: int) -> "ReactAgentBuilder":
        """Set the maximum number of context messages to retain."""
        self._config["max_context_messages"] = value
        return self

    def with_max_context_tokens(self, value: int) -> "ReactAgentBuilder":
        """Set the maximum number of context tokens to retain."""
        self._config["max_context_tokens"] = value
        return self

    def with_context_pruning(self, value: bool = True) -> "ReactAgentBuilder":
        """Enable or disable context pruning."""
        self._config["enable_context_pruning"] = value
        return self

    def with_context_summarization(self, value: bool = True) -> "ReactAgentBuilder":
        """Enable or disable context summarization."""
        self._config["enable_context_summarization"] = value
        return self

    def with_context_pruning_strategy(self, value: str) -> "ReactAgentBuilder":
        """Set the context pruning strategy ('conservative', 'balanced', 'aggressive')."""
        self._config["context_pruning_strategy"] = value
        return self

    def with_context_token_budget(self, value: int) -> "ReactAgentBuilder":
        """Set the token budget for context management."""
        self._config["context_token_budget"] = value
        return self

    def with_context_pruning_aggressiveness(self, value: str) -> "ReactAgentBuilder":
        """Set the aggressiveness of context pruning ('conservative', 'balanced', 'aggressive')."""
        self._config["context_pruning_aggressiveness"] = value
        return self

    def with_context_summarization_frequency(self, value: int) -> "ReactAgentBuilder":
        """Set the number of iterations between context summarizations."""
        self._config["context_summarization_frequency"] = value
        return self

    def with_tool_use_policy(self, value: str) -> "ReactAgentBuilder":
        """Set the tool use policy ('always', 'required_only', 'adaptive', 'never')."""
        self._config["tool_use_policy"] = value
        return self

    def with_tool_use_max_consecutive_calls(self, value: int) -> "ReactAgentBuilder":
        """Set the maximum consecutive tool calls before forcing reflection/summarization."""
        self._config["tool_use_max_consecutive_calls"] = value
        return self

    # Factory methods for common agent types

    @classmethod
    async def research_agent(
        cls,
        model: Optional[str] = None,
    ) -> ReactAgent:
        """
        Create a pre-configured research agent optimized for information gathering

        Args:
            model: Optional model name to use (default: ollama:qwen2:7b)
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
    async def database_agent(cls, model: Optional[str] = None) -> ReactAgent:
        """
        Create a pre-configured database agent optimized for database operations

        Args:
            model: Optional model name to use (default: ollama:qwen2:7b)
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
    ) -> ReactAgent:
        """
        Create a specialized agent for cryptocurrency research and data collection

        Args:
            model: Optional model name to use (default: ollama:qwen2:7b)
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
    async def add_custom_tools_to_agent(
        cls, agent: ReactAgent, custom_tools: List[Any]
    ) -> ReactAgent:
        """
        Add custom tools to an existing agent instance

        This utility method provides a clean way to add custom tools to an agent
        that has already been created, such as one from a factory method.

        Args:
            agent: An existing ReactAgent instance
            custom_tools: List of custom tool functions or objects

        Returns:
            The updated agent with the new tools added

        Example:
            ```python
            agent = await ReactAgentBuilder.research_agent()
            updated_agent = await ReactAgentBuilder.add_custom_tools_to_agent(
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

    # Tool registration helper methods

    def _unify_tool_registration(self, agent: ReactAgent) -> None:
        """
        Unify tool registration to ensure all tools are accessible in both
        agent.context.tools and agent.context.tool_manager.tools

        Args:
            agent: The ReactAgent instance to modify
        """
        if not hasattr(agent, "context"):
            return

        context = agent.context
        tool_manager = getattr(context, "tool_manager", None)
        tools_list = getattr(context, "tools", [])

        if not tool_manager:
            return

        # First, collect all tools from both sources
        all_tools = {}

        # Add tools from the tool_manager
        for tool in getattr(tool_manager, "tools", []):
            tool_name = getattr(tool, "name", str(id(tool)))
            all_tools[tool_name] = tool

        # Add tools from the context.tools list
        for tool in tools_list:
            tool_name = getattr(tool, "name", str(id(tool)))
            all_tools[tool_name] = tool

        # Now ensure both collections have all tools

        # 1. Update context.tools
        if tools_list:
            # Clear the current list
            while tools_list:
                tools_list.pop()

            # Add all unified tools
            for tool_name, tool in all_tools.items():
                tools_list.append(tool)

        # 2. Update tool_manager.tools
        tool_manager_tools = getattr(tool_manager, "tools", [])
        if tool_manager_tools:
            # Clear the current list carefully (tools might be used by references elsewhere)
            tool_manager.tools = []

            # Add all unified tools
            for tool_name, tool in all_tools.items():
                tool_manager.tools.append(tool)

        # 3. Regenerate tool signatures if the method exists
        generate_signatures = getattr(tool_manager, "_generate_tool_signatures", None)
        if callable(generate_signatures):
            generate_signatures()

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
            builder = ReactAgentBuilder()
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
    async def diagnose_agent_tools(agent: ReactAgent) -> Dict[str, Any]:
        """
        Diagnose tool registration issues in an existing agent

        This static method examines an agent that has already been created
        to check for tool registration issues and provides detailed diagnostics.

        Args:
            agent: The ReactAgent instance to diagnose

        Returns:
            Dict[str, Any]: Diagnostic information about the agent's tools

        Example:
            ```python
            agent = await ReactAgentBuilder().with_mcp_tools(["brave-search"]).build()

            # Diagnose after building
            diagnosis = await ReactAgentBuilder.diagnose_agent_tools(agent)
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
    ) -> "ReactAgentBuilder":
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
            builder = (ReactAgentBuilder()
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
    ) -> "ReactAgentBuilder":
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

            builder = (ReactAgentBuilder()
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
    ) -> "ReactAgentBuilder":
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
            builder = (ReactAgentBuilder()
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
    ) -> "ReactAgentBuilder":
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

            builder = (ReactAgentBuilder()
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
    ) -> "ReactAgentBuilder":
        """Register a callback for session started events"""
        return self.with_event_callback(AgentStateEvent.SESSION_STARTED, callback)

    def on_session_ended(
        self, callback: EventCallback[SessionEndedEventData]
    ) -> "ReactAgentBuilder":
        """Register a callback for session ended events"""
        return self.with_event_callback(AgentStateEvent.SESSION_ENDED, callback)

    def on_task_status_changed(
        self, callback: EventCallback[TaskStatusChangedEventData]
    ) -> "ReactAgentBuilder":
        """Register a callback for task status changed events"""
        return self.with_event_callback(AgentStateEvent.TASK_STATUS_CHANGED, callback)

    def on_iteration_started(
        self, callback: EventCallback[IterationStartedEventData]
    ) -> "ReactAgentBuilder":
        """Register a callback for iteration started events"""
        return self.with_event_callback(AgentStateEvent.ITERATION_STARTED, callback)

    def on_iteration_completed(
        self, callback: EventCallback[IterationCompletedEventData]
    ) -> "ReactAgentBuilder":
        """Register a callback for iteration completed events"""
        return self.with_event_callback(AgentStateEvent.ITERATION_COMPLETED, callback)

    def on_tool_called(
        self, callback: EventCallback[ToolCalledEventData]
    ) -> "ReactAgentBuilder":
        """Register a callback for tool called events"""
        return self.with_event_callback(AgentStateEvent.TOOL_CALLED, callback)

    def on_tool_completed(
        self, callback: EventCallback[ToolCompletedEventData]
    ) -> "ReactAgentBuilder":
        """Register a callback for tool completed events"""
        return self.with_event_callback(AgentStateEvent.TOOL_COMPLETED, callback)

    def on_tool_failed(
        self, callback: EventCallback[ToolFailedEventData]
    ) -> "ReactAgentBuilder":
        """Register a callback for tool failed events"""
        return self.with_event_callback(AgentStateEvent.TOOL_FAILED, callback)

    def on_reflection_generated(
        self, callback: EventCallback[ReflectionGeneratedEventData]
    ) -> "ReactAgentBuilder":
        """Register a callback for reflection generated events"""
        return self.with_event_callback(AgentStateEvent.REFLECTION_GENERATED, callback)

    def on_final_answer_set(
        self, callback: EventCallback[FinalAnswerSetEventData]
    ) -> "ReactAgentBuilder":
        """Register a callback for final answer set events"""
        return self.with_event_callback(AgentStateEvent.FINAL_ANSWER_SET, callback)

    def on_metrics_updated(
        self, callback: EventCallback[MetricsUpdatedEventData]
    ) -> "ReactAgentBuilder":
        """Register a callback for metrics updated events"""
        return self.with_event_callback(AgentStateEvent.METRICS_UPDATED, callback)

    def on_error_occurred(
        self, callback: EventCallback[ErrorOccurredEventData]
    ) -> "ReactAgentBuilder":
        """Register a callback for error occurred events"""
        return self.with_event_callback(AgentStateEvent.ERROR_OCCURRED, callback)

    # Async convenience methods

    def on_session_started_async(
        self, callback: AsyncEventCallback[SessionStartedEventData]
    ) -> "ReactAgentBuilder":
        """Register an async callback for session started events"""
        return self.with_async_event_callback(AgentStateEvent.SESSION_STARTED, callback)

    def on_session_ended_async(
        self, callback: AsyncEventCallback[SessionEndedEventData]
    ) -> "ReactAgentBuilder":
        """Register an async callback for session ended events"""
        return self.with_async_event_callback(AgentStateEvent.SESSION_ENDED, callback)

    # Build method

    async def build(self) -> ReactAgent:
        """
        Build and return a configured ReactAgent instance

        Returns:
            ReactAgent: A fully configured agent ready to use
        """
        # Add custom tools to the configuration
        if self._custom_tools:
            # Ensure each custom tool has proper tracking attributes
            for tool in self._custom_tools:
                # Ensure tool has a proper name attribute if missing
                if not hasattr(tool, "name") and hasattr(tool, "tool_definition"):
                    if isinstance(tool.tool_definition, dict):
                        tool.name = tool.tool_definition.get("name", f"tool_{id(tool)}")

                # Add tracking metadata for metrics
                if not hasattr(tool, "_is_custom_tool"):
                    setattr(tool, "_is_custom_tool", True)

            # Set custom_tools separately from tools to avoid duplication
            self._config["custom_tools"] = self._custom_tools
            # Remove tools if it exists to prevent conflict
            if "tools" in self._config:
                del self._config["tools"]

        # Create the config and agent
        try:
            agent_config = ReactAgentConfig(**self._config)
            agent = await ReactAgent(config=agent_config).initialize()

            # Post-initialization hook for custom tools
            if hasattr(agent, "context"):
                # Ensure the context knows about custom tools for metrics
                if self._custom_tools and not hasattr(
                    agent.context, "_custom_tool_names"
                ):
                    tool_names = [
                        getattr(tool, "name", str(i))
                        for i, tool in enumerate(self._custom_tools)
                    ]
                    setattr(agent.context, "_custom_tool_names", tool_names)

                # CRITICAL: Unify tool registration - ensure tool_manager and context.tools are synchronized
                self._unify_tool_registration(agent)

                # Store custom tool data on the agent for future reference
                if self._custom_tools:
                    setattr(agent, "_has_custom_tools", True)
                    setattr(
                        agent,
                        "_custom_tools_data",
                        {
                            "count": len(self._custom_tools),
                            "names": [
                                getattr(tool, "name", f"unknown_{i}")
                                for i, tool in enumerate(self._custom_tools)
                            ],
                        },
                    )

            # Register all event callbacks that were set during builder configuration
            if hasattr(self, "_event_callbacks") and agent.context.state_observer:
                for event_type, callbacks in self._event_callbacks.items():
                    for callback in callbacks:
                        agent.context.state_observer.register_callback(
                            event_type, callback
                        )

            # Register all async event callbacks
            if hasattr(self, "_async_event_callbacks") and agent.context.state_observer:
                for event_type, callbacks in self._async_event_callbacks.items():
                    for callback in callbacks:
                        agent.context.state_observer.register_async_callback(
                            event_type, callback
                        )

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

            raise RuntimeError(f"Failed to create ReactAgent: {e}") from e
