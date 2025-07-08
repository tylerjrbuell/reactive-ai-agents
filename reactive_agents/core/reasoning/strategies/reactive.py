from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING
from reactive_agents.core.types.reasoning_types import (
    ReasoningStrategies,
    ReasoningContext,
)
from .base import BaseReasoningStrategy, StrategyResult, StrategyCapabilities
from ..infrastructure import Infrastructure

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class ReactiveStrategy(BaseReasoningStrategy):
    """
    Simple reactive reasoning strategy - direct prompt-response.
    Best for simple tasks that don't require multi-step reasoning.
    Uses the simplified infrastructure for tool execution and context preservation.
    """

    @property
    def name(self) -> str:
        return "reactive"

    @property
    def capabilities(self) -> list[StrategyCapabilities]:
        return [StrategyCapabilities.TOOL_EXECUTION]

    def __init__(self, infrastructure: "Infrastructure"):
        super().__init__(infrastructure)

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """Execute one reactive iteration - simple and direct."""
        try:
            if self.agent_logger:
                self.agent_logger.debug("🔀 Executing reactive iteration")

            # Use infrastructure's enhanced prompt system for reactive execution
            # For reactive strategy, we can use the planning prompt in a simplified way
            prompt = await self.infrastructure.get_planning_prompt(
                task=task,
                context="Execute this task directly using available tools as needed",
            )

            if self.context.tool_use_enabled:
                prompt += "\n\nUse the available tools when appropriate. When you have the information needed to answer the user's question, use the final_answer tool."

            # Execute thinking with tool use
            result = await self._think_chain(use_tools=True)

            if not result:
                return StrategyResult(
                    action_taken="model_call",
                    result={"error": "No response from model"},
                    should_continue=False,
                    strategy_used="reactive",
                )

            content = result.get("content", "")
            tool_calls = result.get("tool_calls", [])

            # Handle tool execution results
            if tool_calls:
                # Preserve context for tool execution
                self._preserve_context(
                    f"tool_execution_{len(tool_calls)}_calls", tool_calls
                )

                # Check if this was a final_answer tool call
                if any(
                    tc.get("function", {}).get("name") == "final_answer"
                    for tc in tool_calls
                ):
                    # Use centralized completion logic
                    return await self.complete_task_if_ready(
                        task=task,
                        execution_summary=f"Reactive execution with final_answer tool. Tools used: {len(tool_calls)}",
                        tool_calls=tool_calls,
                        result_content=content,
                    )

                return StrategyResult(
                    action_taken="tool_execution",
                    result={"tools_executed": len(tool_calls)},
                    should_continue=True,
                    strategy_used="reactive",
                )

            # Check if content looks like a final answer
            if any(
                phrase in content.lower()
                for phrase in ["final answer", "conclusion", "result:", "answer:"]
            ):
                # Use centralized completion logic
                return await self.complete_task_if_ready(
                    task=task,
                    execution_summary=f"Reactive execution with text-based final answer: {content[:100]}...",
                    result_content=content,
                )

            # Continue for clarification or more work
            return StrategyResult(
                action_taken="completion",
                result={"response": content},
                should_continue=True,
                strategy_used="reactive",
            )

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Reactive iteration failed: {e}")
            return self._format_error_result(e, "reactive_iteration")

    def should_switch_strategy(
        self, reasoning_context: ReasoningContext
    ) -> Optional[str]:
        """Switch if task becomes complex or errors occur."""
        if reasoning_context.error_count >= 2:
            return "reflect_decide_act"

        if (
            reasoning_context.iteration_count >= 3
            and not reasoning_context.last_action_result
        ):
            return "plan_execute_reflect"

        return None

    def get_strategy_name(self) -> ReasoningStrategies:
        """Return the strategy identifier."""
        return ReasoningStrategies.REACTIVE
