"""
Core Engine Module

Provides the execution engine for reactive agents.
"""

# Core engine components
from .execution_engine import ExecutionEngine
from reactive_agents.core.reasoning.task_classifier import TaskClassifier
from reactive_agents.core.metrics.metrics_manager import MetricsManager

# Context
from ..context.agent_context import AgentContext

# Event management
from reactive_agents.core.events.agent_events import EventSubscription
from reactive_agents.core.events.event_bus import EventBus

# Memory management
from reactive_agents.core.memory.memory_manager import MemoryManager
from reactive_agents.core.memory.vector_memory import VectorMemoryManager

# Tool management
from reactive_agents.core.tools.tool_manager import ToolManager

from reactive_agents.core.tools.data_extractor import DataExtractor

# Reasoning components
from reactive_agents.core.reasoning.strategy_manager import StrategyManager
from reactive_agents.core.reasoning.engine import ReasoningEngine

# Workflow management
from reactive_agents.core.workflows.workflow_manager import WorkflowManager

__all__ = [
    # Core engine
    "ExecutionEngine",
    "TaskClassifier",
    "MetricsManager",
    # Context
    "AgentContext",
    # Events
    "EventSubscription",
    "EventBus",
    # Memory
    "MemoryManager",
    "VectorMemoryManager",
    # Tools
    "ToolManager",

    "DataExtractor",
    # Reasoning
    "StrategyManager",
    # Workflows
    "WorkflowManager",
]
