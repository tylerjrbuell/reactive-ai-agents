"""
reactive-ai-agent - A custom reactive AI Agent framework for LLM-driven task execution

This framework provides a flexible system for creating AI agents that can:
- Use different LLM providers (Ollama, Groq)
- Reflect on their actions and improve iteratively
- Use tools through a simple decorator-based interface
"""

__version__ = "0.1.0"

from .agents import Agent, ReactAgent
from .model_providers import ModelProviderFactory, BaseModelProvider
from .tools import Tool, tool, ToolProtocol
from .config import AgentConfig, WorkflowConfig, Workflow
from .loggers import Logger

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
