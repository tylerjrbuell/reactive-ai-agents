"""
reactive-ai-agent - A custom reactive AI Agent framework for LLM-driven task execution

This framework provides a flexible system for creating AI agents that can:
- Use different LLM providers (Ollama, Groq)
- Reflect on their actions and improve iteratively
- Use tools through a simple decorator-based interface
"""

__version__ = "0.1.0a5"

import reactive_agents

__all__ = [
    "reactive_agents",
]
