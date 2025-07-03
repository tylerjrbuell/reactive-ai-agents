"""
Application Layer

User-facing components for building agents and workflows.
"""

# Agent system
from .agents.base import Agent
from .agents.reactive_agent import ReactiveAgent
from .builders.agent import ReactiveAgentBuilder

# Workflow system
from reactive_agents.core.workflows.workflow_manager import WorkflowManager
from reactive_agents.app.workflows.orchestrator import WorkflowOrchestrator

# Communication system
from .communication.a2a_protocol import A2ACommunicationProtocol

__all__ = [
    # Agents
    "Agent",
    "ReactiveAgent",
    "ReactiveAgentBuilder",
    # Workflows
    "WorkflowManager",
    "WorkflowOrchestrator",
    # Communication
    "A2ACommunicationProtocol",
]
