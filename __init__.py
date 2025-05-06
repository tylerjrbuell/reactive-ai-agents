"""
reactive-ai-agent - A custom reactive AI Agent framework for LLM-driven task execution

This framework provides a flexible system for creating AI agents that can:
- Use different LLM providers (Ollama, Groq)
- Reflect on their actions and improve iteratively
- Use tools through a simple decorator-based interface
"""

__version__ = "0.1.0a2"

from agents.base import Agent
from agents.react_agent import ReactAgent
from model_providers.factory import ModelProviderFactory
from model_providers.base import BaseModelProvider
from tools.decorators import tool
from tools.base import Tool
from tools.abstractions import ToolProtocol
from config.workflow import AgentConfig, WorkflowConfig, Workflow
from loggers.base import Logger

__all__ = [
    "Agent",
    "ReactAgent",
    "ModelProviderFactory",
    "BaseModelProvider",
    "Tool",
    "tool",
    "ToolProtocol",
    "AgentConfig",
    "WorkflowConfig",
    "Workflow",
    "Logger",
]
