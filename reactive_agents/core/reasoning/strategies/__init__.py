"""
Reasoning Strategies

All available reasoning strategies for the reactive-agents framework.
Each strategy implements a different approach to problem-solving and task execution.
"""

from .base import BaseReasoningStrategy
from .reactive import ReactiveStrategy
from .plan_execute_reflect import PlanExecuteReflectStrategy
from .reflect_decide_act import ReflectDecideActStrategy

__all__ = [
    "BaseReasoningStrategy",
    "ReactiveStrategy",
    "PlanExecuteReflectStrategy",
    "ReflectDecideActStrategy",
]


# Lazy imports to avoid circular dependencies
def get_strategy_manager():
    from ..strategy_manager import StrategyManager

    return StrategyManager


def get_base_strategy():
    from .base import BaseReasoningStrategy

    return BaseReasoningStrategy
