"""
Simplified Reasoning engine

Essential shared components for reasoning strategies without over-engineering.
Provides basic prompt access, tool execution, and context preservation.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import json

from reactive_agents.core.types.agent_types import (
    AgentThinkChainResult,
    AgentThinkResult,
)

# Import the class-based prompt system
from .prompts.base import (
    SystemPrompt,
    TaskPlanningPrompt,
    ReflectionPrompt,
    PlanGenerationPrompt,
    StepExecutionPrompt,
    TaskCompletionValidationPrompt,
    PlanProgressReflectionPrompt,
    ErrorRecoveryPrompt,
    FinalAnswerPrompt,
    ToolSelectionPrompt,
    StrategyTransitionPrompt,
    PlanExtensionPrompt,
    TaskGoalEvaluationPrompt,
)

from reactive_agents.core.context.context_manager import ContextManager

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class ReasoningEngine:
    """
    Simple, clean infrastructure for reasoning strategies.

    Provides essential shared components:
    - Class-based prompt access
    - Tool execution
    - Context preservation
    - Basic completion checking
    """

    def __init__(self, context: "AgentContext"):
        self.context = context
        self.agent_logger = context.agent_logger

        # Simple data stores
        self._preserved_context: Dict[str, Any] = {}
        self._completed_actions: List[str] = []

        # Initialize prompt classes
        self._prompts = {
            "system": SystemPrompt(context),
            "planning": TaskPlanningPrompt(context),
            "reflection": ReflectionPrompt(context),
            "plan_generation": PlanGenerationPrompt(context),
            "step_execution": StepExecutionPrompt(context),
            "completion_validation": TaskCompletionValidationPrompt(context),
            "plan_progress_reflection": PlanProgressReflectionPrompt(context),
            "error_recovery": ErrorRecoveryPrompt(context),
            "final_answer": FinalAnswerPrompt(context),
            "tool_selection": ToolSelectionPrompt(context),
            "strategy_transition": StrategyTransitionPrompt(context),
            "plan_extension": PlanExtensionPrompt(context),
            "task_goal_evaluation": TaskGoalEvaluationPrompt(context),
        }

    # === Prompt Management ===
    async def think_chain(
        self, use_tools: bool = False
    ) -> Optional[AgentThinkChainResult]:
        """Execute a thinking step with optional tool use."""
        if not hasattr(self.context, "_agent") or not self.context._agent:
            return None

        try:
            result = await self.context._agent._think_chain(use_tools=use_tools)
            return result
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Think operation failed: {e}")
            return None

    async def think(
        self, prompt: str, response_format: Optional[str] = None, **kwargs
    ) -> Optional[AgentThinkResult]:
        """Execute a thinking step with optional tool use."""
        if not hasattr(self.context, "_agent") or not self.context._agent:
            return None
        try:
            return await self.context._agent._think(
                prompt=prompt, format=response_format, **kwargs
            )
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Think operation failed: {e}")
            return None

    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """Get a sophisticated prompt using the class-based system."""

        try:
            return self._prompts[prompt_type].generate(**kwargs)
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Failed to generate {prompt_type} prompt: {e}")
            return self._get_fallback_prompt(prompt_type, **kwargs)

    def _get_fallback_prompt(self, prompt_type: str, **kwargs) -> str:
        """Fallback to basic prompts if class-based generation fails."""
        prompts = {
            "reflection": """
Please reflect on the following task and current progress:

Task: {task}
Current Progress: {progress}
Last Action Result: {last_result}

Provide a brief reflection on:
1. What has been accomplished so far
2. What still needs to be done
3. Any adjustments needed to the approach

Reflection:""",
            "planning": """
Please create a step-by-step plan for the following task:

Task: {task}
Context: {context}

Provide a clear, numbered plan with specific actions:""",
            "execution": """
Please execute the next step in solving this task:

Task: {task}
Current Step: {step}
Context: {context}

Execute the step and provide the result:""",
            "completion": """
Please evaluate if the following task has been completed successfully:

Task: {task}
Actions Taken: {actions}
Results: {results}

Is the task complete? Provide a yes/no answer with brief reasoning:""",
        }

        template = prompts.get(prompt_type, "Please help with: {task}")
        return template.format(**kwargs)

    # === Enhanced Prompt Methods ===
    async def get_planning_prompt(self, task: str, **kwargs) -> str:
        """Get a sophisticated planning prompt with full context."""
        return self.get_prompt("planning", task=task, **kwargs)

    async def get_reflection_prompt(
        self, task: str, progress: str = "", last_result: str = "", **kwargs
    ) -> str:
        """Get a sophisticated reflection prompt with full context."""
        return self.get_prompt(
            "reflection",
            task=task,
            progress=progress,
            last_result=last_result,
            **kwargs,
        )

    async def get_step_execution_prompt(self, task: str, step: str, **kwargs) -> str:
        """Get a sophisticated step execution prompt with full context."""
        return self.get_prompt("step_execution", task=task, step=step, **kwargs)

    async def get_plan_generation_prompt(self, task: str, **kwargs) -> str:
        """Get a sophisticated plan generation prompt with full context."""
        return self.get_prompt("plan_generation", task=task, **kwargs)

    async def get_tool_selection_prompt(self, step_description: str, **kwargs) -> str:
        """Get a sophisticated tool selection prompt with full context."""
        return self.get_prompt(
            "tool_selection", step_description=step_description, **kwargs
        )

    async def get_strategy_transition_prompt(
        self, current_strategy: str, **kwargs
    ) -> str:
        """Get a sophisticated strategy transition prompt with full context."""
        return self.get_prompt(
            "strategy_transition", current_strategy=current_strategy, **kwargs
        )

    async def get_error_recovery_prompt(self, error_context: str, **kwargs) -> str:
        """Get a sophisticated error recovery prompt with full context."""
        return self.get_prompt("error_recovery", error_context=error_context, **kwargs)

    async def get_final_answer_prompt(self, **kwargs) -> str:
        """Get a sophisticated final answer prompt with full context."""
        return self.get_prompt("final_answer", **kwargs)

    # === Tool Execution ===
    async def execute_tools(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        if not self.context.tool_manager:
            return []

        results = []
        for tool_call in tool_calls:
            try:
                result = await self.context.tool_manager.use_tool(tool_call)
                tool_name = tool_call.get("name") or tool_call.get("function", {}).get(
                    "name"
                )

                # Track completed actions
                if tool_name and tool_name not in self._completed_actions:
                    self._completed_actions.append(tool_name)

                results.append(
                    {
                        "tool_name": tool_name,
                        "result": result,
                        "success": result is not None
                        and "error" not in str(result).lower(),
                    }
                )

            except Exception as e:
                if self.agent_logger:
                    self.agent_logger.error(f"Tool execution failed: {e}")
                results.append(
                    {
                        "tool_name": tool_call.get("name", "unknown"),
                        "result": str(e),
                        "success": False,
                    }
                )

        return results

    def get_completed_actions(self) -> List[str]:
        """Get list of completed actions."""
        return self._completed_actions.copy()

    # === Context Preservation ===
    def preserve_context(self, key: str, value: Any) -> None:
        """Store important context data."""
        self._preserved_context[key] = value

    def get_preserved_context(self, key: Optional[str] = None) -> Any:
        """Get preserved context data."""
        if key:
            return self._preserved_context.get(key)
        return self._preserved_context.copy()

    def get_context_summary(self) -> str:
        """Get a summary of preserved context."""
        if not self._preserved_context:
            return "No preserved context data."

        summary_parts = []
        for key, value in self._preserved_context.items():
            if isinstance(value, (str, int, float, bool)):
                summary_parts.append(f"- {key}: {value}")
            elif isinstance(value, (list, dict)):
                summary_parts.append(
                    f"- {key}: {type(value).__name__} with {len(value)} items"
                )
            else:
                summary_parts.append(f"- {key}: {type(value).__name__}")

        return "Preserved Context:\n" + "\n".join(summary_parts)

    # === Completion Validation ===
    async def check_completion(
        self, task: str, required_actions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Check if task appears to be completed using sophisticated validation."""
        try:
            # Use the class-based completion validation prompt
            completion_prompt = self.get_prompt(
                "completion_validation",
                task=task,
                actions=self._completed_actions,
                results=self.get_context_summary(),
            )

            result = await self.think(completion_prompt)

            # Parse the result for completion information
            if result and result.content:
                content = result.content
                # Try to extract completion information from the response
                is_complete = any(
                    phrase in content.lower()
                    for phrase in ["complete", "finished", "done", "accomplished"]
                )
                confidence = 0.8 if is_complete else 0.5
            else:
                is_complete = False
                confidence = 0.3

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Completion validation failed: {e}")
            # Fallback to basic completion check
            is_complete = bool(getattr(self.context.session, "final_answer", None))
            confidence = 0.5

        # Basic completion check
        has_final_answer = bool(getattr(self.context.session, "final_answer", None))

        # Check required actions if provided
        actions_completed = True
        if required_actions:
            missing_actions = [
                action
                for action in required_actions
                if action not in self._completed_actions
            ]
            actions_completed = len(missing_actions) == 0

        # Combine checks
        final_is_complete = is_complete and has_final_answer and actions_completed
        final_confidence = min(confidence, 1.0 if final_is_complete else 0.6)

        return {
            "is_complete": final_is_complete,
            "confidence": final_confidence,
            "has_final_answer": has_final_answer,
            "actions_completed": actions_completed,
            "completed_actions": self._completed_actions.copy(),
        }

    # === Reflection ===
    async def reflect(
        self, task: str, progress: str = "", last_result: str = ""
    ) -> Optional[str]:
        """Generate a sophisticated reflection on current progress."""
        reflection_prompt = await self.get_reflection_prompt(
            task=task, progress=progress, last_result=last_result
        )

        result = await self.think(reflection_prompt)
        if result and result.content:
            return result.content

        return None

    # === Reset ===
    def reset(self):
        """Reset engine for new task."""
        self._preserved_context.clear()
        self._completed_actions.clear()

    # === Centralized Task Completion ===
    async def should_complete_task(
        self, task: str, progress_indicators: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Determine if the task should be completed now.

        Args:
            task: The current task
            progress_indicators: List of completed actions/steps
            **kwargs: Additional context (execution_history, steps_completed, etc.)

        Returns:
            Dict with completion info: {is_complete, confidence, reason}
        """
        try:
            # Prepare execution context for improved completion validation
            execution_context = {
                "task": task,
                "actions": progress_indicators or self._completed_actions,
                "results": self.get_context_summary(),
                "execution_summary": kwargs.get("execution_summary", ""),
                "steps_completed": kwargs.get(
                    "steps_completed", len(self._completed_actions)
                ),
                "steps_total": kwargs.get("steps_total", 0),
                "execution_history": kwargs.get("execution_history", []),
            }

            # Remove duplicated parameters from kwargs to avoid conflicts
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k not in execution_context
            }

            # Use the improved completion validation prompt
            completion_prompt = self.get_prompt(
                "completion_validation",
                **execution_context,
                **filtered_kwargs,
            )

            result = await self.think(completion_prompt)

            if result and result.content:
                content = result.content
                # Try to parse structured response
                try:
                    import json
                    import re

                    json_match = re.search(r'\{[^}]*"is_complete"[^}]*\}', content)
                    if json_match:
                        completion_data = json.loads(json_match.group())
                        is_complete = completion_data.get("is_complete", False)
                        confidence = completion_data.get("confidence", 0.5)
                        reason = completion_data.get(
                            "reason", "Based on completion analysis"
                        )
                    else:
                        # Fallback to keyword detection
                        is_complete = any(
                            phrase in content.lower()
                            for phrase in [
                                "complete",
                                "finished",
                                "done",
                                "accomplished",
                            ]
                        )
                        confidence = 0.7 if is_complete else 0.4
                        reason = "Based on keyword analysis"
                except:
                    # Fallback to keyword detection
                    is_complete = any(
                        phrase in content.lower()
                        for phrase in ["complete", "finished", "done", "accomplished"]
                    )
                    confidence = 0.7 if is_complete else 0.4
                    reason = "Based on keyword analysis"
            else:
                is_complete = False
                confidence = 0.3
                reason = "No completion analysis available"

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Task completion check failed: {e}")
            is_complete = False
            confidence = 0.2
            reason = f"Completion check error: {e}"

        # Additional checks
        has_final_answer = bool(getattr(self.context.session, "final_answer", None))

        return {
            "is_complete": is_complete,
            "confidence": confidence,
            "reason": reason,
            "has_final_answer": has_final_answer,
            "completed_actions": self._completed_actions.copy(),
        }

    async def generate_final_answer(
        self, task: str, execution_summary: str = "", **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a final answer for the completed task.

        Args:
            task: The original task
            execution_summary: Summary of what was accomplished
            **kwargs: Additional context (execution_history, reflection, etc.)

        Returns:
            Final answer string or None if generation failed
        """
        try:
            final_prompt = await self.get_final_answer_prompt(
                task=task,
                execution_summary=execution_summary,
                **kwargs,
            )

            self.context.session.messages.append(
                {
                    "role": "assistant",
                    "content": f"I will now generate a final answer for the task: {task}",
                }
            )
            self.context.session.messages.append(
                {"role": "user", "content": final_prompt}
            )

            # Use direct thinking instead of tool-based thinking for final answer
            result = await self.think_chain(use_tools=True)
            if self.agent_logger:
                self.agent_logger.info(f"Final answer result: {result}")
            if result and result.content:
                # Try to parse as JSON first
                try:
                    import json

                    final_answer_data = json.loads(result.content)
                    return final_answer_data
                except json.JSONDecodeError:
                    # If not valid JSON, wrap in a final_answer structure
                    return {"final_answer": result.content}

            return None
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Final answer generation failed: {e}")
            return None

    def get_context_manager(self) -> ContextManager:
        """
        Get or create a context manager for the agent.

        Returns:
            ContextManager instance
        """
        if not hasattr(self, "_context_manager"):
            self._context_manager = ContextManager(self.context)
        return self._context_manager


# Global engine cache
_engine_cache: Dict[str, ReasoningEngine] = {}


def get_reasoning_engine(context: "AgentContext") -> ReasoningEngine:
    """Get or create engine instance for the given context."""
    # Use agent name as cache key
    cache_key = getattr(context, "agent_name", "default")

    if cache_key not in _engine_cache:
        _engine_cache[cache_key] = ReasoningEngine(context)

    return _engine_cache[cache_key]
