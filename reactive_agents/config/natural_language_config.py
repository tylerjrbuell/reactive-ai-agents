from __future__ import annotations
import json
import re
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING, cast
from reactive_agents.core.types.agent_types import ReactiveAgentConfig

if TYPE_CHECKING:
    from reactive_agents.providers.llm.base import BaseModelProvider
    from reactive_agents.app.agents.reactive_agent import ReactiveAgent


class NaturalLanguageConfigParser:
    """
    Parses natural language descriptions into agent configurations.

    Example:
    "Create an agent that can analyze PDFs, summarize research, and collaborate
     with another agent using shared memory."

    Becomes:
    ReactiveAgentConfig(
        custom_tools=["pdf_reader", "summarize"],
        kwargs={"reasoning_strategy": "plan_execute_reflect"}
    )
    """

    def __init__(self, model_provider: "BaseModelProvider"):
        self.model_provider = model_provider

        # Tool mappings from natural language to actual tool names
        self.tool_mappings = {
            "pdf": ["pdf_reader", "pdf_analyzer"],
            "summarize": ["summarize", "text_summarizer"],
            "research": ["web_search", "search_papers"],
            "code": ["code_executor", "python_executor"],
            "file": ["file_reader", "file_writer"],
            "database": ["sql_executor", "db_query"],
            "image": ["image_analyzer", "vision_tool"],
            "email": ["email_sender", "email_reader"],
            "calendar": ["calendar_manager", "schedule_tool"],
            "translate": ["translator", "language_tool"],
            "math": ["calculator", "math_solver"],
            "chart": ["chart_generator", "data_visualizer"],
        }

        # Reasoning strategy mappings
        self.reasoning_mappings = {
            "simple": "reactive",
            "complex": "plan_execute_reflect",
            "planning": "plan_execute_reflect",
            "research": "reflect_decide_act",
            "analysis": "reflect_decide_act",
            "collaboration": "adaptive",
            "multi-step": "plan_execute_reflect",
        }

        self.parsing_prompt = self._create_parsing_prompt()

        # Initialize plugin awareness
        self._initialize_plugin_mappings()

    def _initialize_plugin_mappings(self):
        """Initialize plugin-aware mappings for tools and strategies."""
        try:
            from reactive_agents.plugins.plugin_manager import (
                get_plugin_manager,
                PluginType,
            )

            plugin_manager = get_plugin_manager()

            # Get plugin strategy mappings
            self.plugin_strategies = {}
            strategy_plugins = plugin_manager.get_plugins_by_type(PluginType.STRATEGY)
            for plugin_name, plugin in strategy_plugins.items():
                self.plugin_strategies[plugin_name] = plugin.name
                # Add to reasoning mappings if the plugin has recognizable keywords
                if "custom" in plugin_name.lower():
                    self.reasoning_mappings["custom"] = plugin_name

            # Get plugin tool mappings
            self.plugin_tools = {}
            tool_plugins = plugin_manager.get_plugins_by_type(PluginType.TOOL)
            for plugin_name, plugin in tool_plugins.items():
                self.plugin_tools[plugin_name] = plugin.name
                # Add to tool mappings with plugin name as keyword
                tool_keyword = plugin_name.lower().replace("_", " ").replace("-", " ")
                self.tool_mappings[tool_keyword] = [plugin_name]

        except ImportError:
            # Plugin system not available
            self.plugin_strategies = {}
            self.plugin_tools = {}
        except Exception as e:
            # Error loading plugin mappings
            self.plugin_strategies = {}
            self.plugin_tools = {}

    def _create_parsing_prompt(self) -> str:
        """Create the system prompt for parsing natural language configurations."""
        plugin_info = ""
        try:
            if hasattr(self, "plugin_strategies") and self.plugin_strategies:
                strategy_list = list(self.plugin_strategies.keys())
                plugin_info += (
                    f"\n\nAvailable plugin strategies: {', '.join(strategy_list)}"
                )

            if hasattr(self, "plugin_tools") and self.plugin_tools:
                tool_list = list(self.plugin_tools.keys())
                plugin_info += f"\n\nAvailable plugin tools: {', '.join(tool_list)}"
        except:
            pass

        return f"""You are an expert at parsing natural language descriptions into agent configurations.

Parse the user's description and return a JSON configuration object with these fields:

{{
  "agent_name": "descriptive_name",
  "role": "brief role description", 
  "custom_tools": ["tool1", "tool2"],
  "reasoning_strategy": "reactive|reflect_decide_act|plan_execute_reflect|adaptive|<plugin_name>",
  "max_iterations": 10-50,
  "tool_use_enabled": true/false,
  "use_memory_enabled": true/false,
  "collect_metrics_enabled": true/false,
  "instructions": "specific instructions for the agent",
  "communication_protocol": "a2a" (if collaboration mentioned),
  "workflow_context_shared": true (if shared context mentioned),
  "plugins": ["plugin_name1", "plugin_name2"] (if specific plugins mentioned)
}}

Available tools: pdf_reader, summarize, web_search, code_executor, file_reader, 
file_writer, sql_executor, image_analyzer, email_sender, calculator, chart_generator

Reasoning strategies:
- reactive: Simple prompt-response for basic tasks
- reflect_decide_act: Good for research, analysis (default)
- plan_execute_reflect: For complex, multi-step tasks
- adaptive: For collaboration or varying complexity{plugin_info}

Parse this description into a valid JSON configuration:"""

    async def parse_config(self, description: str) -> ReactiveAgentConfig:
        """
        Parse a natural language description into a ReactiveAgentConfig.

        Args:
            description: Natural language description of the desired agent

        Returns:
            ReactiveAgentConfig object ready for agent creation
        """
        try:
            # Get configuration from LLM
            response = await self.model_provider.get_completion(
                system=self.parsing_prompt,
                prompt=description,
                options={"temperature": 0.1},  # Low temperature for consistency
            )

            if not response or not response.message.content:
                raise ValueError("No response from model provider")

            # Parse JSON response
            config_json = self._extract_json(response.message.content)

            # Enhance with defaults and validation
            enhanced_config = self._enhance_config(config_json, description)

            # Create ReactiveAgentConfig
            return ReactiveAgentConfig(**enhanced_config)

        except Exception as e:
            # Fallback to heuristic parsing if LLM fails
            return self._heuristic_parse(description)

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        # Try to find JSON block
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Look for standalone JSON
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")

        return json.loads(json_str)

    def _enhance_config(
        self, config: Dict[str, Any], description: str
    ) -> Dict[str, Any]:
        """Enhance parsed config with defaults and validation."""
        # Set defaults
        enhanced = {
            "agent_name": config.get("agent_name", "nl_configured_agent"),
            "role": config.get("role", "AI Assistant"),
            "provider_model_name": "ollama:llama3.2",  # Default local model
            "custom_tools": config.get("custom_tools", []),
            "instructions": config.get(
                "instructions", f"You are configured based on: {description}"
            ),
            "max_iterations": config.get("max_iterations", 20),
            "tool_use_enabled": config.get(
                "tool_use_enabled", len(config.get("custom_tools", [])) > 0
            ),
            "use_memory_enabled": config.get("use_memory_enabled", True),
            "collect_metrics_enabled": config.get("collect_metrics_enabled", True),
            "reflect_enabled": True,
            "log_level": "info",
        }

        # Set reasoning strategy in kwargs (since it's not a direct ReactiveAgentConfig parameter)
        reasoning = config.get(
            "reasoning_strategy", "reactive"
        )  # Changed from reflect_decide_act
        enhanced["kwargs"] = {
            "reasoning_strategy": reasoning,
            "enable_reactive_execution": True,
        }

        # Handle collaboration features
        if config.get("communication_protocol") == "a2a":
            enhanced["kwargs"]["enable_a2a_communication"] = True
            enhanced["workflow_context_shared"] = config.get(
                "workflow_context_shared", True
            )

        # Handle plugin configurations
        plugins = config.get("plugins", [])
        if plugins:
            enhanced["kwargs"]["enabled_plugins"] = plugins

        # Handle plugin-based reasoning strategies
        if reasoning in getattr(self, "plugin_strategies", {}):
            enhanced["kwargs"]["plugin_reasoning_strategy"] = reasoning

        # Add plugin tools to custom tools
        plugin_tools = []
        for tool_name in config.get("custom_tools", []):
            if tool_name in getattr(self, "plugin_tools", {}):
                plugin_tools.append(tool_name)

        if plugin_tools:
            enhanced["kwargs"]["plugin_tools"] = plugin_tools

        return enhanced

    def _heuristic_parse(self, description: str) -> ReactiveAgentConfig:
        """Fallback heuristic parsing when LLM fails."""
        description_lower = description.lower()

        # Infer tools
        tools = []
        for keyword, tool_names in self.tool_mappings.items():
            if keyword in description_lower:
                tools.extend(tool_names[:1])  # Take first tool for each category

        # Infer reasoning strategy
        reasoning = "reflect_decide_act"  # Default
        for keyword, strategy in self.reasoning_mappings.items():
            if keyword in description_lower:
                reasoning = strategy
                break

        # Infer complexity/iterations
        max_iterations = 15
        if any(
            word in description_lower for word in ["complex", "detailed", "thorough"]
        ):
            max_iterations = 30
        elif any(word in description_lower for word in ["simple", "quick", "basic"]):
            max_iterations = 10

        return ReactiveAgentConfig(
            agent_name="heuristic_agent",
            role="AI Assistant",
            provider_model_name="ollama:llama3.2",
            custom_tools=tools,
            instructions=f"You are configured based on: {description}",
            max_iterations=max_iterations,
            tool_use_enabled=len(tools) > 0,
            use_memory_enabled=True,
            collect_metrics_enabled=True,
            reflect_enabled=True,
            log_level="info",
            kwargs={
                "reasoning_strategy": reasoning,
                "enable_reactive_execution": True,
            },
        )


class AgentFactory:
    """Factory for creating agents from natural language descriptions."""

    def __init__(self, model_provider: "BaseModelProvider"):
        self.config_parser = NaturalLanguageConfigParser(model_provider)

    async def create_agent_from_description(
        self, description: str, agent_class: str = "ReactiveAgent"
    ) -> ReactiveAgent:
        """
        Create an agent from a natural language description.

        Args:
            description: Natural language description of desired agent
            agent_class: Which agent class to instantiate

        Returns:
            Configured agent instance
        """
        # Parse configuration
        config = await self.config_parser.parse_config(description)

        # Import and create agent
        if agent_class == "ReactiveAgent":
            from reactive_agents.app.agents.reactive_agent import ReactiveAgent

            return ReactiveAgent(config)
        else:
            raise ValueError(f"Invalid agent class: {agent_class}")

    async def preview_config(self, description: str) -> Dict[str, Any]:
        """
        Preview the configuration that would be generated from a description.

        Args:
            description: Natural language description

        Returns:
            Dictionary representation of the configuration
        """
        config = await self.config_parser.parse_config(description)
        return config.model_dump()


# Convenience function for quick agent creation
async def create_agent_from_nl(
    description: str, model_provider: "BaseModelProvider"
) -> "ReactiveAgent":
    """
    Convenience function to create an agent from natural language.

    Example:
        agent = await create_agent_from_nl(
            "Create an agent that can analyze PDFs and summarize research",
            model_provider
        )
    """
    factory = AgentFactory(model_provider)
    # Always use ReactiveAgentV2 for the convenience function
    result = await factory.create_agent_from_description(description, "ReactiveAgent")
    # Type cast since we know it's ReactiveAgent when agent_class is "ReactiveAgent"
    return cast("ReactiveAgent", result)
