"""
Context Adapters Module

Provides strategy-specific context management adapters.
Each strategy can have its own context management behavior.
"""

from .base_context import BaseContextAdapter
from .default_context import DefaultContextAdapter
from .plan_execute_reflect_context import PlanExecuteReflectContextAdapter

__all__ = [
    "BaseContextAdapter",
    "DefaultContextAdapter",
    "PlanExecuteReflectContextAdapter",
]
