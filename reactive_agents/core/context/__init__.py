"""
Context System

Core context management components for agent execution and configuration.
"""

from reactive_agents.core.context.agent_context import AgentContext
from reactive_agents.core.context.context_manager import (
    ContextManager,
    MessageRole,
    ContextWindow,
)

__all__ = ["AgentContext", "ContextManager", "MessageRole", "ContextWindow"]
