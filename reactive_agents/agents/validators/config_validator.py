"""
Configuration validation module for reactive-ai-agent framework.
Handles validation and processing of agent configurations.
"""

from typing import Dict, Any, Optional, List
from reactive_agents.config.mcp_config import MCPConfig
from reactive_agents.agent_mcp.client import MCPClient
from reactive_agents.common.types.confirmation_types import (
    ConfirmationCallbackProtocol,
    ConfirmationConfig,
)
from reactive_agents.loggers.base import Logger


class ConfigValidator:
    """
    Handles validation and processing of agent configurations.
    """

    def __init__(self, log_level: str = "info"):
        """Initialize the ConfigValidator with a log level."""
        self.agent_logger = Logger("ConfigValidator", "config", log_level)

    def validate_agent_config(
        self,
        agent_name: str,
        provider_model_name: str,
        model_provider_options: Optional[Dict[str, Any]] = None,
        role: Optional[str] = None,
        mcp_client: Optional[MCPClient] = None,
        mcp_config: Optional[MCPConfig] = None,
        mcp_server_filter: Optional[List[str]] = None,
        min_completion_score: Optional[float] = None,
        instructions: Optional[str] = None,
        max_iterations: Optional[int] = None,
        reflect_enabled: Optional[bool] = None,
        log_level: Optional[str] = None,
        initial_task: Optional[str] = None,
        tool_use_enabled: Optional[bool] = None,
        custom_tools: Optional[List[Any]] = None,
        use_memory_enabled: Optional[bool] = None,
        collect_metrics_enabled: Optional[bool] = None,
        check_tool_feasibility: Optional[bool] = None,
        enable_caching: Optional[bool] = None,
        confirmation_callback: Optional[ConfirmationCallbackProtocol] = None,
        confirmation_config: Optional[Dict[str, Any]] = None,
        workflow_context_shared: Optional[Dict[str, Any]] = None,
        response_format: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Validate and process agent configuration parameters.

        Args:
            agent_name: Name of the agent
            provider_model_name: Name of the LLM provider and model
            model_provider_options: Options for the LLM provider
            role: Role of the agent
            mcp_client: An initialized MCPClient instance
            mcp_config: MCP config dict or file path
            mcp_server_filter: Filter list of MCP servers
            min_completion_score: Minimum score for task completion
            instructions: High-level instructions for the agent
            max_iterations: Maximum number of iterations allowed
            reflect_enabled: Whether reflection mechanism is enabled
            log_level: Logging level
            initial_task: The initial task description
            tool_use_enabled: Whether the agent can use tools
            custom_tools: List of custom tool instances
            use_memory_enabled: Whether the agent uses long-term memory
            collect_metrics_enabled: Whether to collect performance metrics
            check_tool_feasibility: Whether to check tool feasibility
            enable_caching: Whether to enable LLM response caching
            confirmation_callback: Callback for confirming tool use
            confirmation_config: Configuration for tool confirmation
            workflow_context_shared: Shared workflow context data
            response_format: Format specification for the agent's final response
            **kwargs: Additional keyword arguments

        Returns:
            Dict[str, Any]: Validated and processed configuration
        """
        # Create base configuration
        config = {
            "agent_name": agent_name,
            "provider_model_name": provider_model_name,
            "model_provider_options": model_provider_options or {},
            "role": role or "Task Executor",
            "mcp_client": mcp_client,
            "mcp_config": mcp_config,
            "mcp_server_filter": mcp_server_filter or [],
            "min_completion_score": min_completion_score or 1.0,
            "instructions": instructions or "Solve the given task.",
            "max_iterations": max_iterations or 10,
            "reflect_enabled": reflect_enabled if reflect_enabled is not None else True,
            "log_level": log_level or "info",
            "initial_task": initial_task,
            "tool_use_enabled": (
                tool_use_enabled if tool_use_enabled is not None else True
            ),
            "custom_tools": custom_tools or [],
            "use_memory_enabled": (
                use_memory_enabled if use_memory_enabled is not None else True
            ),
            "collect_metrics_enabled": (
                collect_metrics_enabled if collect_metrics_enabled is not None else True
            ),
            "check_tool_feasibility": (
                check_tool_feasibility if check_tool_feasibility is not None else True
            ),
            "enable_caching": enable_caching if enable_caching is not None else True,
            "confirmation_callback": confirmation_callback,
            "confirmation_config": confirmation_config,
            "workflow_context_shared": workflow_context_shared,
            "response_format": response_format,
            "kwargs": kwargs,
        }

        # Validate required fields
        if not config["agent_name"]:
            raise ValueError("agent_name is required")
        if not config["provider_model_name"]:
            raise ValueError("provider_model_name is required")

        # Validate numeric fields
        if config["min_completion_score"] < 0 or config["min_completion_score"] > 1:
            raise ValueError("min_completion_score must be between 0 and 1")
        if config["max_iterations"] < 1:
            raise ValueError("max_iterations must be greater than 0")

        # Validate log level
        valid_log_levels = ["debug", "info", "warning", "error", "critical"]
        if config["log_level"] not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")

        # Process and validate MCP configuration
        if config["mcp_config"] and not isinstance(config["mcp_config"], MCPConfig):
            try:
                config["mcp_config"] = MCPConfig.model_validate(
                    config["mcp_config"], strict=False
                )
            except Exception as e:
                raise ValueError(f"Invalid MCP configuration: {e}")

        # Process and validate confirmation configuration
        if config["confirmation_config"] and not isinstance(
            config["confirmation_config"], ConfirmationConfig
        ):
            try:
                config["confirmation_config"] = ConfirmationConfig(
                    config["confirmation_config"]
                )
            except Exception as e:
                raise ValueError(f"Invalid confirmation configuration: {e}")

        return config

    def validate_tool_config(self, tool_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool configuration.

        Args:
            tool_config: Tool configuration to validate

        Returns:
            Dict[str, Any]: Validated tool configuration
        """
        required_fields = ["name", "description"]
        for field in required_fields:
            if field not in tool_config:
                raise ValueError(f"Tool configuration missing required field: {field}")

        if not isinstance(tool_config["name"], str):
            raise ValueError("Tool name must be a string")

        if not isinstance(tool_config["description"], str):
            raise ValueError("Tool description must be a string")

        return tool_config

    def validate_workflow_config(
        self, workflow_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate workflow configuration.

        Args:
            workflow_config: Workflow configuration to validate

        Returns:
            Dict[str, Any]: Validated workflow configuration
        """
        if not isinstance(workflow_config, dict):
            raise ValueError("Workflow configuration must be a dictionary")

        return workflow_config
