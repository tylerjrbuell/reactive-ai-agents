"""
Prompts module for reactive-ai-agent framework.
Contains system prompts and templates for agents.
"""

from .agent_prompts import (
    TASK_PLANNING_SYSTEM_PROMPT,
    TOOL_ACTION_SUMMARY_PROMPT,
    AGENT_ACTION_PLAN_PROMPT,
    REACT_AGENT_SYSTEM_PROMPT,
    PERCENTAGE_COMPLETE_TASK_REFLECTION_PROMPT,
)

from .base import (
    BasePrompt,
    PromptContext,
    SystemPrompt,
    SingleStepPlanningPrompt,
    ReflectionPrompt,
    ToolSelectionPrompt,
    FinalAnswerPrompt,
    StrategyTransitionPrompt,
    ErrorRecoveryPrompt,
    PlanGenerationPrompt,
    TaskCompletionValidationPrompt,
    PlanProgressReflectionPrompt,
    PlanExtensionPrompt,
)

__all__ = [
    "TASK_PLANNING_SYSTEM_PROMPT",
    "TOOL_ACTION_SUMMARY_PROMPT",
    "AGENT_ACTION_PLAN_PROMPT",
    "REACT_AGENT_SYSTEM_PROMPT",
    "PERCENTAGE_COMPLETE_TASK_REFLECTION_PROMPT",
    "BasePrompt",
    "PromptContext",
    "SystemPrompt",
    "SingleStepPlanningPrompt",
    "ReflectionPrompt",
    "ToolSelectionPrompt",
    "FinalAnswerPrompt",
    "StrategyTransitionPrompt",
    "ErrorRecoveryPrompt",
    "PlanGenerationPrompt",
    "TaskCompletionValidationPrompt",
    "PlanProgressReflectionPrompt",
    "PlanExtensionPrompt",
]
