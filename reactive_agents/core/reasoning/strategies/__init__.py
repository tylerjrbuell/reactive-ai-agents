"""
Reasoning strategies module for reactive agents.
Provides pluggable reasoning patterns that agents can use and switch between dynamically.
"""

from .base import BaseReasoningStrategy
from .strategy_manager import StrategyManager

__all__ = ["BaseReasoningStrategy", "StrategyManager"]


# Lazy imports to avoid circular dependencies
def get_strategy_manager():
    from .strategy_manager import StrategyManager

    return StrategyManager


def get_base_strategy():
    from .base import BaseReasoningStrategy

    return BaseReasoningStrategy
