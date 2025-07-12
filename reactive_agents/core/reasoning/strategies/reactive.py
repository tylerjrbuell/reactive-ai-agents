from __future__ import annotations
from typing import Dict, Any, List, Optional

from reactive_agents.core.types.reasoning_types import ReasoningContext
from reactive_agents.core.reasoning.engine import ReasoningEngine
from reactive_agents.core.reasoning.strategies.base import (
    BaseReasoningStrategy,
    StrategyResult,
    StrategyCapabilities,
)


class ReactiveStrategy(BaseReasoningStrategy):
    """
    Simplified reactive reasoning strategy.
    Directly executes tools for the user task, with optional reflection and completion check.
    Best for simple tasks that don't require multi-step planning.
    """

    @property
    def name(self) -> str:
        return "reactive"

    @property
    def capabilities(self) -> List[StrategyCapabilities]:
        return [
            StrategyCapabilities.TOOL_EXECUTION,
            StrategyCapabilities.REFLECTION,
            StrategyCapabilities.ADAPTATION,
        ]

    @property
    def description(self) -> str:
        return "Direct prompt-response reasoning with tool execution and optional reflection."

    async def initialize(self, task: str, reasoning_context: ReasoningContext) -> None:
        """Initialize the reactive strategy."""
        # Set the strategy in context manager
        self.context_manager.set_active_strategy(self.name)

        # Add initial task message
        self.context_manager.add_message(
            role="user", content=f"Task: {task}\nPlease work on this task step by step."
        )

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """
        Execute one iteration of the reactive strategy:
        1. Execute the tool(s) for the task
        2. Reflect on the result
        3. Check for completion
        """
        try:
            # Step 1: Execute the tool(s) for the task
            # Use native tool calling by default, fall back to manual prompting
            use_native_tools = getattr(
                self.context, "supports_native_tool_calling", True
            )

            execution_result = await self.execute_with_tools(
                task, f"Complete the task: {task}", use_native_tools=use_native_tools
            )

            # Step 2: Reflect on the result
            reflection = await self.reflect_on_progress(
                task, execution_result, reasoning_context
            )

            # Step 3: Check if task is complete
            if (
                reflection.get("goal_achieved", False)
                or reflection.get("next_action") == "complete"
            ):
                if self.agent_logger:
                    self.agent_logger.info("Goal achieved, completing task")

                # Generate final answer
                execution_summary = f"Completed task using {execution_result.get('method', 'unknown')} method"
                final_answer = await self.generate_final_answer(task, execution_summary)

                return StrategyResult(
                    action_taken="task_completed",
                    should_continue=False,
                    final_answer=final_answer.get("final_answer"),
                    status="completed",
                    evaluation=reflection,
                    result={
                        "execution_result": execution_result,
                        "reflection": reflection,
                        "final_answer": final_answer,
                    },
                )

            # Continue if not complete
            return StrategyResult(
                action_taken="tool_executed",
                should_continue=True,
                status="in_progress",
                evaluation=reflection,
                result={
                    "execution_result": execution_result,
                    "reflection": reflection,
                },
            )

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Error in reactive strategy: {e}")

            return StrategyResult(
                action_taken="error_occurred",
                should_continue=True,
                status="error",
                result={
                    "error": str(e),
                    "recovery_action": "retry_with_different_approach",
                },
            )
