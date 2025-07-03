"""
Execution Engine

Core execution loop and task management.
"""

# Core engine components
from .execution_engine import AgentExecutionEngine as ExecutionEngine
from .reactive_execution_engine import ReactiveExecutionEngine
from .task_executor import TaskExecutor
from reactive_agents.core.reasoning.task_classifier import TaskClassifier
from reactive_agents.core.metrics.metrics_manager import MetricsManager

# Context
from ..context.agent_context import AgentContext

# Event management
from reactive_agents.core.events.agent_events import EventSubscription
from reactive_agents.core.events.agent_observer import (
    AgentStateObserver as AgentObserver,
)
from reactive_agents.core.events.event_manager import EventManager
from reactive_agents.core.events.event_bus import EventBus

# Memory management
from reactive_agents.core.memory.memory_manager import MemoryManager
from reactive_agents.core.memory.vector_memory import VectorMemoryManager

# Tool management
from reactive_agents.core.tools.tool_manager import ToolManager
from reactive_agents.core.tools.tool_processor import ToolProcessor
from reactive_agents.core.tools.data_extractor import DataExtractor

# Reasoning components
from reactive_agents.core.reasoning.reflection_manager import ReflectionManager
from reactive_agents.core.reasoning.plan_manager import PlanManager

# Workflow management
from reactive_agents.core.workflows.workflow_manager import WorkflowManager

__all__ = [
    # Core engine
    "ExecutionEngine",
    "ReactiveExecutionEngine",
    "TaskExecutor",
    "TaskClassifier",
    "MetricsManager",
    # Context
    "AgentContext",
    # Events
    "EventSubscription",
    "AgentObserver",
    "EventManager",
    "EventBus",
    # Memory
    "MemoryManager",
    "VectorMemoryManager",
    # Tools
    "ToolManager",
    "ToolProcessor",
    "DataExtractor",
    # Reasoning
    "ReflectionManager",
    "PlanManager",
    # Workflows
    "WorkflowManager",
]
