"""
Model Context Protocol (MCP) module for reactive-ai-agent framework.
Provides client implementation and helper functions for MCP.
"""

from .client import MCPClient
from .helpers.general import *

__all__ = [
    "MCPClient",
]
