"""
Enhanced Plugin Management System

Provides hot-loading and management of framework extensions with comprehensive plugin support.
"""

from __future__ import annotations
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Callable
from abc import ABC, abstractmethod
from enum import Enum
import logging
import asyncio
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins supported by the framework."""

    STRATEGY = "strategy"
    TOOL = "tool"
    PROVIDER = "provider"
    MEMORY = "memory"
    PROMPT = "prompt"
    AGENT = "agent"
    WORKFLOW = "workflow"


class PluginMetadata(BaseModel):
    """Metadata for plugin registration."""

    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = Field(default_factory=list)
    framework_version: str = ">=2.0.0"


class PluginInterface(ABC):
    """Base interface for all plugins"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description"""
        pass

    @property
    @abstractmethod
    def plugin_type(self) -> PluginType:
        """Plugin type"""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the plugin"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass

    def is_compatible(self, framework_version: str) -> bool:
        """Check if plugin is compatible with framework version"""
        # Default implementation - plugins can override
        return True


class ReasoningStrategyPlugin(PluginInterface):
    """Interface for reasoning strategy plugins"""

    @property
    def plugin_type(self) -> PluginType:
        return PluginType.STRATEGY

    @abstractmethod
    def get_strategies(self) -> Dict[str, Type]:
        """Return strategy classes provided by this plugin"""
        pass


class ToolPlugin(PluginInterface):
    """Interface for tool plugins"""

    @property
    def plugin_type(self) -> PluginType:
        return PluginType.TOOL

    @abstractmethod
    def get_tools(self) -> Dict[str, Callable]:
        """Return tool functions provided by this plugin"""
        pass


class ProviderPlugin(PluginInterface):
    """Interface for provider plugins (LLM, storage, etc.)"""

    @property
    def plugin_type(self) -> PluginType:
        return PluginType.PROVIDER

    @abstractmethod
    def get_providers(self) -> Dict[str, Type]:
        """Return provider classes"""
        pass


class MemoryPlugin(PluginInterface):
    """Interface for memory system plugins"""

    @property
    def plugin_type(self) -> PluginType:
        return PluginType.MEMORY

    @abstractmethod
    def get_memory_systems(self) -> Dict[str, Type]:
        """Return memory system classes"""
        pass


class PromptPlugin(PluginInterface):
    """Interface for prompt plugins"""

    @property
    def plugin_type(self) -> PluginType:
        return PluginType.PROMPT

    @abstractmethod
    def get_prompts(self) -> Dict[str, str]:
        """Return prompt templates"""
        pass


class AgentPlugin(PluginInterface):
    """Interface for agent plugins"""

    @property
    def plugin_type(self) -> PluginType:
        return PluginType.AGENT

    @abstractmethod
    def get_agents(self) -> Dict[str, Type]:
        """Return agent classes"""
        pass


class WorkflowPlugin(PluginInterface):
    """Interface for workflow plugins"""

    @property
    def plugin_type(self) -> PluginType:
        return PluginType.WORKFLOW

    @abstractmethod
    def get_workflows(self) -> Dict[str, Type]:
        """Return workflow classes"""
        pass


class PluginManager:
    """Enhanced plugin manager with comprehensive plugin support"""

    def __init__(self, framework_version: str = "2.0.0"):
        self.framework_version = framework_version
        self.loaded_plugins: Dict[str, PluginInterface] = {}
        self.plugin_registry: Dict[PluginType, Dict[str, PluginInterface]] = {
            plugin_type: {} for plugin_type in PluginType
        }
        self.plugin_paths: List[Path] = []
        self.auto_load_enabled = True
        self._setup_default_paths()

    def _setup_default_paths(self):
        """Setup default plugin discovery paths."""
        self.plugin_paths.append(Path.cwd() / "plugins")
        self.plugin_paths.append(Path.home() / ".reactive_agents" / "plugins")

    def add_plugin_path(self, path: Path) -> None:
        """Add a directory to search for plugins"""
        if not path.exists():
            logger.warning(f"Plugin path does not exist: {path}")
            return

        self.plugin_paths.append(path)
        logger.info(f"Added plugin path: {path}")

    def discover_plugins(self) -> List[PluginMetadata]:
        """Discover available plugins with metadata"""
        discovered = []

        for path in self.plugin_paths:
            if not path.is_dir():
                continue

            for plugin_dir in path.iterdir():
                if not plugin_dir.is_dir():
                    continue

                # Check for plugin.json metadata file
                metadata_file = plugin_dir / "plugin.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata_data = json.load(f)

                        metadata = PluginMetadata(**metadata_data)
                        discovered.append(metadata)
                        continue
                    except Exception as e:
                        logger.warning(
                            f"Failed to load plugin metadata from {metadata_file}: {e}"
                        )

                # Fallback to directory discovery
                if (plugin_dir / "__init__.py").exists():
                    discovered.append(
                        PluginMetadata(
                            name=plugin_dir.name,
                            version="unknown",
                            description="Auto-discovered plugin",
                            author="unknown",
                            plugin_type=PluginType.TOOL,  # Default type
                        )
                    )

        return discovered

    async def load_plugin(
        self, plugin_name: str, plugin_path: Optional[Path] = None
    ) -> PluginInterface:
        """Load and initialize a plugin"""

        if plugin_name in self.loaded_plugins:
            logger.warning(f"Plugin {plugin_name} is already loaded")
            return self.loaded_plugins[plugin_name]

        try:
            # Find plugin path if not provided
            if plugin_path is None:
                plugin_path = self._find_plugin_path(plugin_name)
                if plugin_path is None:
                    raise ValueError(f"Plugin {plugin_name} not found in plugin paths")

            # Import plugin module
            spec = importlib.util.spec_from_file_location(
                plugin_name, plugin_path / "__init__.py"
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load plugin spec for {plugin_name}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin class (should implement PluginInterface)
            plugin_class = None
            for item_name in dir(module):
                item = getattr(module, item_name)
                if (
                    inspect.isclass(item)
                    and issubclass(item, PluginInterface)
                    and item != PluginInterface
                ):
                    plugin_class = item
                    break

            if plugin_class is None:
                raise ValueError(
                    f"No PluginInterface implementation found in {plugin_name}"
                )

            # Instantiate and initialize plugin
            plugin_instance = plugin_class()

            # Check compatibility
            if not plugin_instance.is_compatible(self.framework_version):
                raise ValueError(
                    f"Plugin {plugin_name} is not compatible with framework version {self.framework_version}"
                )

            # Initialize plugin
            await plugin_instance.initialize()

            # Store loaded plugin
            self.loaded_plugins[plugin_name] = plugin_instance

            # Register in type-specific registry
            plugin_type = plugin_instance.plugin_type
            self.plugin_registry[plugin_type][plugin_name] = plugin_instance

            logger.info(
                f"Successfully loaded plugin: {plugin_instance.name} v{plugin_instance.version} ({plugin_type.value})"
            )
            return plugin_instance

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            raise

    def _find_plugin_path(self, plugin_name: str) -> Optional[Path]:
        """Find the path to a plugin"""
        for path in self.plugin_paths:
            plugin_path = path / plugin_name
            if plugin_path.is_dir() and (plugin_path / "__init__.py").exists():
                return plugin_path
        return None

    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get a loaded plugin by name"""
        return self.loaded_plugins.get(name)

    def get_plugins_by_type(
        self, plugin_type: PluginType
    ) -> Dict[str, PluginInterface]:
        """Get all loaded plugins of a specific type"""
        return self.plugin_registry.get(plugin_type, {})

    async def unload_plugin(self, name: str) -> None:
        """Unload a plugin"""
        if name in self.loaded_plugins:
            try:
                plugin = self.loaded_plugins[name]
                await plugin.cleanup()

                # Remove from main registry
                del self.loaded_plugins[name]

                # Remove from type-specific registry
                plugin_type = plugin.plugin_type
                if name in self.plugin_registry[plugin_type]:
                    del self.plugin_registry[plugin_type][name]

                logger.info(f"Unloaded plugin: {name}")
            except Exception as e:
                logger.error(f"Error unloading plugin {name}: {e}")
                raise

    def list_loaded_plugins(self) -> List[Dict[str, str]]:
        """List all loaded plugins with their info"""
        return [
            {
                "name": plugin.name,
                "version": plugin.version,
                "description": plugin.description,
                "type": plugin.plugin_type.value,
            }
            for plugin in self.loaded_plugins.values()
        ]

    async def load_all_discovered(self) -> Dict[str, bool]:
        """Load all discovered plugins"""
        results = {}
        discovered = self.discover_plugins()

        for plugin_metadata in discovered:
            try:
                await self.load_plugin(plugin_metadata.name)
                results[plugin_metadata.name] = True
            except Exception as e:
                logger.error(f"Failed to auto-load plugin {plugin_metadata.name}: {e}")
                results[plugin_metadata.name] = False

        return results

    async def reload_plugin(self, name: str) -> PluginInterface:
        """Reload a plugin (unload then load)"""
        if name in self.loaded_plugins:
            await self.unload_plugin(name)
        return await self.load_plugin(name)

    async def cleanup_all(self) -> None:
        """Cleanup all loaded plugins"""
        for name in list(self.loaded_plugins.keys()):
            await self.unload_plugin(name)


class PluginGenerator:
    """Generator for creating plugin scaffolding (Laravel-style)."""

    def __init__(self, plugin_manager: Optional[PluginManager] = None):
        self.plugin_manager = plugin_manager or get_plugin_manager()
        self.templates_path = Path(__file__).parent / "templates"

    def generate_strategy_plugin(
        self, name: str, description: str = "", author: str = ""
    ) -> Path:
        """Generate a strategy plugin scaffold."""
        plugin_path = Path.cwd() / "plugins" / name
        plugin_path.mkdir(parents=True, exist_ok=True)

        # Create plugin.json
        metadata = {
            "name": name,
            "version": "1.0.0",
            "description": description or f"Custom strategy plugin: {name}",
            "author": author,
            "plugin_type": "strategy",
            "dependencies": [],
            "framework_version": ">=2.0.0",
        }

        with open(plugin_path / "plugin.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Create __init__.py
        class_name = name.title().replace("_", "").replace("-", "")
        init_content = f'''"""
{name} Strategy Plugin

{description}
"""

from reactive_agents.core.reasoning.strategies.base import BaseReasoningStrategy
from reactive_agents.core.types.reasoning_types import ReasoningStrategies
from reactive_agents.plugins.plugin_manager import PluginInterface, PluginType


class {class_name}Strategy(BaseReasoningStrategy):
    """Custom strategy implementation."""
    
    async def execute_iteration(self, task: str, reasoning_context) -> dict:
        """Execute one iteration of this strategy."""
        # TODO: Implement your strategy logic here
        return {{
            "action_taken": "custom_strategy_action",
            "result": "Strategy executed successfully",
            "should_continue": False,
            "strategy_used": "{name}"
        }}
    
    def get_strategy_name(self) -> ReasoningStrategies:
        """Get the strategy enum identifier."""
        # TODO: Add your strategy to ReasoningStrategies enum
        return ReasoningStrategies.ADAPTIVE  # Placeholder


class {class_name}Plugin(PluginInterface):
    """Plugin implementation for {name} strategy."""
    
    @property
    def name(self) -> str:
        return "{name}"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "{description}"
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.STRATEGY
    
    async def initialize(self) -> None:
        """Initialize the plugin."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass
    
    def get_strategy_class(self) -> type:
        """Return the strategy class."""
        return {class_name}Strategy
'''

        with open(plugin_path / "__init__.py", "w") as f:
            f.write(init_content)

        # Create README
        readme_content = f"""# {name.title()} Strategy Plugin

{description}

## Installation

1. Copy this plugin to your `plugins/` directory
2. The plugin will be automatically discovered and loaded

## Configuration

Configure the plugin in your agent configuration:

```python
# Enable the strategy
agent_config.kwargs["reasoning_strategy"] = "{name}"
```

## Development

1. Implement your strategy logic in the `execute_iteration` method
2. Add your strategy to the `ReasoningStrategies` enum
3. Test your strategy with different task types

## API

### Strategy Methods

- `execute_iteration(task, reasoning_context)`: Main strategy execution
- `should_switch_strategy(reasoning_context)`: Strategy switching logic
- `get_strategy_name()`: Return strategy identifier

### Plugin Methods

- `initialize()`: Plugin initialization
- `cleanup()`: Plugin cleanup
"""

        with open(plugin_path / "README.md", "w") as f:
            f.write(readme_content)

        return plugin_path

    def generate_tool_plugin(
        self, name: str, description: str = "", author: str = ""
    ) -> Path:
        """Generate a tool plugin scaffold."""
        plugin_path = Path.cwd() / "plugins" / name
        plugin_path.mkdir(parents=True, exist_ok=True)

        # Create plugin.json
        metadata = {
            "name": name,
            "version": "1.0.0",
            "description": description or f"Custom tool plugin: {name}",
            "author": author,
            "plugin_type": "tool",
            "dependencies": [],
            "framework_version": ">=2.0.0",
        }

        with open(plugin_path / "plugin.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Create __init__.py
        class_name = name.title().replace("_", "").replace("-", "")
        init_content = f'''"""
{name} Tool Plugin

{description}
"""

from reactive_agents.core.tools.base import Tool
from reactive_agents.plugins.plugin_manager import PluginInterface, PluginType


class {class_name}Tool(Tool):
    """Custom tool implementation."""
    
    name = "{name}"
    description = "{description}"
    
    def __init__(self):
        super().__init__()
    
    async def execute(self, **kwargs) -> dict:
        """Execute the tool."""
        # TODO: Implement your tool logic here
        return {{
            "result": "Tool executed successfully",
            "data": kwargs
        }}


class {class_name}Plugin(PluginInterface):
    """Plugin implementation for {name} tool."""
    
    @property
    def name(self) -> str:
        return "{name}"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "{description}"
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.TOOL
    
    async def initialize(self) -> None:
        """Initialize the plugin."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass
    
    def get_tools(self) -> list:
        """Return tool instances."""
        return [{class_name}Tool()]
'''

        with open(plugin_path / "__init__.py", "w") as f:
            f.write(init_content)

        # Create README
        readme_content = f"""# {name.title()} Tool Plugin

{description}

## Installation

1. Copy this plugin to your `plugins/` directory
2. The plugin will be automatically discovered and loaded

## Usage

The tool will be available to agents once loaded.

## Development

1. Implement your tool logic in the `execute` method
2. Update the tool description and parameters
3. Test your tool with different inputs

## API

### Tool Methods

- `execute(**kwargs)`: Main tool execution
- `validate_input(input_data)`: Input validation
- `get_schema()`: Tool schema for LLM
"""

        with open(plugin_path / "README.md", "w") as f:
            f.write(readme_content)

        return plugin_path

    def generate_provider_plugin(
        self, name: str, description: str = "", author: str = ""
    ) -> Path:
        """Generate a provider plugin scaffold."""
        plugin_path = Path.cwd() / "plugins" / name
        plugin_path.mkdir(parents=True, exist_ok=True)

        # Create plugin.json
        metadata = {
            "name": name,
            "version": "1.0.0",
            "description": description or f"Custom provider plugin: {name}",
            "author": author,
            "plugin_type": "provider",
            "dependencies": [],
            "framework_version": ">=2.0.0",
        }

        with open(plugin_path / "plugin.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Create __init__.py
        class_name = name.title().replace("_", "").replace("-", "")
        init_content = f'''"""
{name} Provider Plugin

{description}
"""

from reactive_agents.providers.llm.base import BaseModelProvider
from reactive_agents.plugins.plugin_manager import PluginInterface, PluginType


class {class_name}Provider(BaseModelProvider):
    """Custom provider implementation."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using the provider."""
        # TODO: Implement your provider logic here
        return f"Response from {class_name}Provider: {{prompt}}"
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return True


class {class_name}Plugin(PluginInterface):
    """Plugin implementation for {name} provider."""
    
    @property
    def name(self) -> str:
        return "{name}"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "{description}"
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.PROVIDER
    
    async def initialize(self) -> None:
        """Initialize the plugin."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass
    
    def get_providers(self) -> dict:
        """Return provider classes."""
        return {{"{name}": {class_name}Provider}}
'''

        with open(plugin_path / "__init__.py", "w") as f:
            f.write(init_content)

        # Create README
        readme_content = f"""# {name.title()} Provider Plugin

{description}

## Installation

1. Copy this plugin to your `plugins/` directory
2. The plugin will be automatically discovered and loaded

## Configuration

Configure the provider in your agent configuration:

```python
# Configure the provider
agent_config.kwargs["model_provider"] = "{name}"
```

## Development

1. Implement your provider logic in the `generate_response` method
2. Handle authentication and API calls
3. Test your provider with different prompts

## API

### Provider Methods

- `generate_response(prompt, **kwargs)`: Generate response
- `is_available()`: Check availability
- `get_supported_models()`: List supported models
"""

        with open(plugin_path / "README.md", "w") as f:
            f.write(readme_content)

        return plugin_path

    def generate_agent_plugin(
        self, name: str, description: str = "", author: str = ""
    ) -> Path:
        """Generate an agent plugin scaffold."""
        plugin_path = Path.cwd() / "plugins" / name
        plugin_path.mkdir(parents=True, exist_ok=True)

        # Create plugin.json
        metadata = {
            "name": name,
            "version": "1.0.0",
            "description": description or f"Custom agent plugin: {name}",
            "author": author,
            "plugin_type": "agent",
            "dependencies": [],
            "framework_version": ">=2.0.0",
        }

        with open(plugin_path / "plugin.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Create __init__.py
        class_name = name.title().replace("_", "").replace("-", "")
        init_content = f'''"""
{name} Agent Plugin

{description}
"""

from reactive_agents.app.agents.base import BaseAgent
from reactive_agents.plugins.plugin_manager import PluginInterface, PluginType


class {class_name}Agent(BaseAgent):
    """Custom agent implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_type = "{name}"
    
    async def process_task(self, task: str) -> dict:
        """Process a task using this agent."""
        # TODO: Implement your agent logic here
        return {{
            "result": "Task processed successfully",
            "agent_type": self.agent_type,
            "task": task
        }}


class {class_name}Plugin(PluginInterface):
    """Plugin implementation for {name} agent."""
    
    @property
    def name(self) -> str:
        return "{name}"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "{description}"
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.AGENT
    
    async def initialize(self) -> None:
        """Initialize the plugin."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass
    
    def get_agents(self) -> dict:
        """Return agent classes."""
        return {{"{name}": {class_name}Agent}}
'''

        with open(plugin_path / "__init__.py", "w") as f:
            f.write(init_content)

        # Create README
        readme_content = f"""# {name.title()} Agent Plugin

{description}

## Installation

1. Copy this plugin to your `plugins/` directory
2. The plugin will be automatically discovered and loaded

## Usage

Create an instance of the agent:

```python
from plugins.{name} import {class_name}Agent

agent = {class_name}Agent()
result = await agent.process_task("your task")
```

## Development

1. Implement your agent logic in the `process_task` method
2. Add specialized capabilities for your domain
3. Test your agent with different task types

## API

### Agent Methods

- `process_task(task)`: Process a task
- `initialize()`: Initialize the agent
- `cleanup()`: Cleanup resources
"""

        with open(plugin_path / "README.md", "w") as f:
            f.write(readme_content)

        return plugin_path


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def get_plugin_generator() -> PluginGenerator:
    """Get a plugin generator instance"""
    return PluginGenerator()


def initialize_plugin_system(plugin_paths: Optional[List[str]] = None) -> PluginManager:
    """Initialize the plugin system with default paths"""
    manager = get_plugin_manager()

    # Add default plugin paths
    if plugin_paths is None:
        plugin_paths = ["plugins", "reactive_agents/plugins/examples"]

    for path_str in plugin_paths:
        path = Path(path_str)
        if path.exists():
            manager.add_plugin_path(path)

    return manager
