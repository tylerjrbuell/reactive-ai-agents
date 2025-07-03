"""
Framework Core

Internal framework components and utilities.
"""

# Engine components
from .engine import (
    ExecutionEngine,
    ReactiveExecutionEngine,
    TaskExecutor,
    TaskClassifier,
    MetricsManager,
    AgentContext,
)

# Event system
from .events import (
    EventSubscription,
    AgentObserver,
    EventManager,
    EventBus,
    AgentEventBus,
)

# Memory management
from .memory import (
    MemoryManager,
    VectorMemoryManager,
)

# Tool system
from .tools import (
    Tool,
    ToolProtocol,
    MCPToolWrapper,
    ToolResult,
    ToolManager,
    ToolProcessor,
    DataExtractor,
    SearchDataManager,
)

# Reasoning system
from .reasoning import (
    ReflectionManager,
    PlanManager,
)

# Workflow system
from .workflows import (
    WorkflowManager,
)


# Plugin system integration
def initialize_core_framework(plugin_paths=None, load_plugins=True):
    """
    Initialize the core framework with plugin system integration.

    Args:
        plugin_paths: List of paths to search for plugins
        load_plugins: Whether to automatically load discovered plugins

    Returns:
        PluginManager instance if plugin system is available
    """
    try:
        from reactive_agents.plugins.plugin_manager import initialize_plugin_system

        if load_plugins:
            plugin_manager = initialize_plugin_system(plugin_paths)

            # Auto-load all discovered plugins
            import asyncio

            async def auto_load_plugins():
                results = await plugin_manager.load_all_discovered()
                loaded_count = sum(results.values())
                total_count = len(results)

                if loaded_count > 0:
                    print(f"🔌 Loaded {loaded_count}/{total_count} plugins")

            # Run plugin loading
            try:
                asyncio.run(auto_load_plugins())
            except RuntimeError:
                # Already in async context
                pass

            return plugin_manager

    except ImportError:
        # Plugin system not available
        pass
    except Exception as e:
        print(f"Warning: Failed to initialize plugin system: {e}")

    return None


__all__ = [
    # Engine
    "ExecutionEngine",
    "ReactiveExecutionEngine",
    "TaskExecutor",
    "TaskClassifier",
    "MetricsManager",
    "AgentContext",
    # Events
    "EventSubscription",
    "AgentObserver",
    "EventManager",
    "EventBus",
    "AgentEventBus",
    # Memory
    "MemoryManager",
    "VectorMemoryManager",
    # Tools
    "Tool",
    "ToolProtocol",
    "MCPToolWrapper",
    "ToolResult",
    "ToolManager",
    "ToolProcessor",
    "DataExtractor",
    "SearchDataManager",
    # Reasoning
    "ReflectionManager",
    "PlanManager",
    # Workflows
    "WorkflowManager",
    # Framework initialization
    "initialize_core_framework",
]
