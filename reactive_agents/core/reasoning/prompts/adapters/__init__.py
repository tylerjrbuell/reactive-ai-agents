"""
Prompt Adapters for Reasoning Strategies

This module contains specialized prompt adapters that bridge between AgentContext
and strategy-specific prompt generation. Each adapter focuses on minimal,
targeted prompts for specific reasoning phases.
"""

from .plan_execute_reflect import PlanExecuteReflectStrategyAdapter

__all__ = ["PlanExecuteReflectStrategyAdapter"]
