from __future__ import annotations
from typing import Dict, Any, List, Optional

from reactive_agents.core.types.reasoning_types import ReasoningContext
from reactive_agents.core.reasoning.strategies.base import (
    StrategyResult,
    StrategyCapabilities,
)
from reactive_agents.core.reasoning.strategy_components import (
    ComponentBasedStrategy,
)


class PlanExecuteReflectStrategy(ComponentBasedStrategy):
    """
    Simplified Plan-Execute-Reflect strategy using centralized components.
    1. Plan: Break down goals into actionable steps using PlanningComponent
    2. Execute: Execute specific steps using ToolExecutionComponent
    3. Reflect: Evaluate progress using ReflectionComponent
    """

    @property
    def name(self) -> str:
        return "plan_execute_reflect"

    @property
    def capabilities(self) -> List[StrategyCapabilities]:
        return [
            StrategyCapabilities.PLANNING,
            StrategyCapabilities.TOOL_EXECUTION,
            StrategyCapabilities.REFLECTION,
        ]

    @property
    def description(self) -> str:
        return "Plan-execute-reflect reasoning with planning, execution, and reflection using centralized components."

    async def initialize(self, task: str, reasoning_context: ReasoningContext) -> None:
        """Initialize the plan-execute-reflect strategy using centralized components."""
        # Call parent initialization
        await super().initialize(task, reasoning_context)

        # Get relevant memories using MemoryIntegrationComponent
        relevant_memories = await self.get_memories(task, max_items=5)
        if relevant_memories:
            if self.agent_logger:
                self.agent_logger.info(
                    f"Found {len(relevant_memories)} relevant memories for task"
                )

        # Generate initial plan using PlanningComponent
        plan_result = await self._generate_plan(task, reasoning_context)
        plan_steps = plan_result.get("plan_steps", [])

        # Store plan in preserved context
        self.engine.preserve_context("plan", plan_steps)
        self.engine.preserve_context("current_step_index", 0)

        # Add initial context
        self.context_manager.add_message(
            role="assistant",
            content=f"I need to attempt to complete this task: {task}\n Using the following {len(plan_steps)} step plan: {plan_steps}",
        )

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """
        Execute one iteration of the plan-execute-reflect strategy:
        1. Get the current step from the plan
        2. Execute the step using tools
        3. Reflect on the results
        4. Move to the next step or complete the task
        """
        try:
            # Get current plan state
            plan = self.engine.get_preserved_context("plan")
            step_index = self.engine.get_preserved_context("current_step_index")

            if not plan or step_index is None:
                # Re-initialize if needed
                await self.initialize(task, reasoning_context)
                plan = self.engine.get_preserved_context("plan")
                step_index = (
                    self.engine.get_preserved_context("current_step_index") or 0
                )

            # Check if we've completed all steps
            if not plan or step_index >= len(plan):
                return await self._handle_plan_completion(task, reasoning_context)

            # Get the current step
            current_step = plan[step_index]
            step_description = (
                current_step.get("description", "")
                if isinstance(current_step, dict)
                else str(current_step)
            )

            if self.agent_logger:
                self.agent_logger.info(
                    f"📋 Executing step {step_index + 1}/{len(plan)}: {step_description}"
                )

            # Execute the current step using ToolExecutionComponent
            execution_result = await self.execute_tool(task, step_description)

            # Reflect on the step execution using ReflectionComponent
            reflection = await self.reflect(task, execution_result, reasoning_context)

            # Check if we have found the information needed to complete the task
            should_complete = self._should_complete_task(
                task, execution_result, reflection
            )
            if self.agent_logger:
                self.agent_logger.info(
                    f"🔍 _should_complete_task returned: {should_complete}"
                )
                if should_complete:
                    self.agent_logger.info(
                        "🚀 Task should complete, calling _handle_task_completion"
                    )

            if should_complete:
                return await self._handle_task_completion(
                    task, execution_result, reflection
                )

            # Determine next action based on reflection
            next_action = reflection.get("next_action", "continue")
            if self.agent_logger:
                self.agent_logger.info(f"🔄 Next action from reflection: {next_action}")

            if next_action == "complete":
                if self.agent_logger:
                    self.agent_logger.info(
                        "🚀 Next action is 'complete', calling _handle_task_completion"
                    )
                return await self._handle_task_completion(
                    task, execution_result, reflection
                )
            elif next_action == "retry":
                # Stay on current step
                return self._create_step_result(
                    "step_retried", step_description, execution_result, reflection
                )
            else:  # continue
                # Move to next step
                next_step_index = step_index + 1
                self.engine.preserve_context("current_step_index", next_step_index)

                if self.agent_logger:
                    self.agent_logger.info(
                        f"✅ Step {step_index + 1} completed, moving to step {next_step_index + 1}/{len(plan)}"
                    )

                return self._create_step_result(
                    "step_completed", step_description, execution_result, reflection
                )

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Error in plan-execute-reflect strategy: {e}")

            # Use ErrorHandlingComponent to handle the error
            error_result = await self.handle_error(
                task=task,
                error_context="plan_execute_reflect_strategy",
                error_count=reasoning_context.error_count,
                last_error=str(e),
            )

            return StrategyResult(
                action_taken="error_occurred",
                should_continue=True,
                status="error",
                result={
                    "error": str(e),
                    "recovery_action": error_result.get(
                        "recovery_action", "retry_current_step"
                    ),
                    "error_handling": error_result,
                },
            )

    async def _generate_plan(
        self, task: str, reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """Generate a plan for the task using the PlanningComponent."""
        return await self.plan(task, reasoning_context)

    async def _handle_plan_completion(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """Handle when all plan steps are completed using TaskEvaluationComponent."""
        # Evaluate overall task completion using TaskEvaluationComponent
        evaluation = await self.evaluate(
            task, progress_summary="All plan steps completed"
        )

        if evaluation.get("is_complete", False):
            final_answer = await self.complete_task(
                task, execution_summary="All plan steps completed successfully"
            )
            return StrategyResult(
                action_taken="task_completed",
                should_continue=False,
                final_answer=final_answer.get("final_answer"),
                status="completed",
                evaluation=evaluation,
            )
        else:
            return StrategyResult(
                action_taken="plan_exhausted",
                should_continue=True,
                status="needs_replanning",
                evaluation=evaluation,
            )

    async def _handle_task_completion(
        self, task: str, execution_result: Dict[str, Any], reflection: Dict[str, Any]
    ) -> StrategyResult:
        """
        Handle task completion using the CompletionComponent.

        Args:
            task: The original task
            execution_result: Results from tool execution
            reflection: Reflection on the progress

        Returns:
            StrategyResult indicating task completion
        """
        # Compose execution summary for the final answer prompt
        execution_summary = ""
        if reflection and isinstance(reflection, dict):
            execution_summary = (
                reflection.get("progress_assessment")
                or reflection.get("reasoning")
                or ""
            )
        if not execution_summary:
            execution_summary = str(execution_result)

        # Check if we already have a final answer in the session (set by final_answer tool)
        session_final_answer = None
        if self.context and self.context.session:
            session_final_answer = self.context.session.final_answer

        if session_final_answer:
            actual_answer = {"final_answer": session_final_answer}
        else:
            if self.agent_logger:
                self.agent_logger.info(
                    "🔄 No final answer in session, generating one..."
                )
            # Use the CompletionComponent to generate final answer
            actual_answer = await self.complete_task(
                task,
                execution_summary=execution_summary,
                execution_result=execution_result,
                reflection=reflection,
            )

            # Set the session's final answer for consistency (only if we don't already have one)
            if (
                self.context
                and self.context.session
                and not self.context.session.final_answer
            ):
                self.context.session.final_answer = actual_answer.get("final_answer")

        # Get task metrics from context
        task_metrics = {}
        if self.context.metrics_manager:
            self.context.metrics_manager.finalize_run_metrics()
            task_metrics = self.context.metrics_manager.get_metrics()

        # Get session data for additional context
        session_data = {}
        if self.context.session:
            session_data = {
                "session_id": getattr(self.context.session, "session_id", "unknown"),
                "iterations": self.context.session.iterations,
                "successful_tools": (
                    list(self.context.session.successful_tools)
                    if self.context.session.successful_tools
                    else []
                ),
                "task_status": (
                    str(self.context.session.task_status)
                    if self.context.session.task_status
                    else "unknown"
                ),
                "final_answer": self.context.session.final_answer,
                "completion_score": self.context.session.completion_score,
            }

        # Create proper evaluation with is_complete flag
        evaluation = {
            "is_complete": True,
            "confidence": reflection.get("confidence", 1.0),
            "reasoning": reflection.get("reasoning", "Task completed successfully"),
            "goal_achieved": reflection.get("goal_achieved", True),
            "completion_score": reflection.get("completion_score", 1.0),
            "reflection": reflection,
        }

        final_answer_value = actual_answer.get("final_answer")

        return StrategyResult(
            action_taken="task_completed",
            should_continue=False,
            final_answer=final_answer_value,
            status="completed",
            evaluation=evaluation,
            result={
                "execution_result": execution_result,
                "reflection": reflection,
                "final_answer": actual_answer,
                "task_metrics": task_metrics,
                "session_data": session_data,
            },
        )

    def _should_complete_task(
        self, task: str, execution_result: Dict[str, Any], reflection: Dict[str, Any]
    ) -> bool:
        """
        Determine if the task should be completed based on execution results.
        This method is plan-aware and only completes when all planned steps are done.

        Args:
            task: The original task
            execution_result: Results from tool execution
            reflection: Reflection on the progress from ReflectionComponent

        Returns:
            True if the task should be completed, False otherwise
        """
        # Check if we have a final answer from a tool
        if "final_answer" in execution_result:
            if self.agent_logger:
                self.agent_logger.info(
                    "✅ Completion triggered: final_answer in execution_result"
                )
            return True

        # Check if reflection indicates goal is achieved
        if reflection.get("goal_achieved", False):
            if self.agent_logger:
                self.agent_logger.info("✅ Completion triggered: goal_achieved = True")
            return True

        # Check if reflection explicitly indicates completion
        if reflection.get("next_action") == "complete":
            if self.agent_logger:
                self.agent_logger.info(
                    "✅ Completion triggered: next_action = 'complete'"
                )
            return True

        # PLAN-AWARE COMPLETION: Check if we've completed all planned steps
        plan = self.engine.get_preserved_context("plan")
        current_step_index = (
            self.engine.get_preserved_context("current_step_index") or 0
        )

        if plan and current_step_index is not None:
            total_steps = len(plan)
            completed_steps = current_step_index

            if self.agent_logger:
                self.agent_logger.debug(
                    f"📋 Plan progress: {completed_steps}/{total_steps} steps completed"
                )

            # Only complete if we've finished all planned steps
            if completed_steps >= total_steps:
                if self.agent_logger:
                    self.agent_logger.info(
                        f"✅ Completion triggered: All {total_steps} planned steps completed"
                    )
                return True
            else:
                if self.agent_logger:
                    self.agent_logger.debug(
                        f"⏳ Not completing: {total_steps - completed_steps} steps remaining"
                    )
                return False

        # Fallback for non-plan tasks: Check completion score (but be more conservative)
        if (
            reflection.get("completion_score", 0) >= 0.9
        ):  # Higher threshold for plan-based
            if self.agent_logger:
                self.agent_logger.info(
                    f"✅ Completion triggered: completion_score >= 0.9 (actual: {reflection.get('completion_score', 0)})"
                )
            return True

        # For price queries, check if we have specific price information
        if any(
            keyword in task.lower() for keyword in ["price", "cost", "value", "worth"]
        ):
            # Check tool calls for price information
            tool_calls = execution_result.get("tool_calls", [])
            for call in tool_calls:
                if isinstance(call, dict) and "result" in call:
                    result_text = str(call["result"])
                    # Look for price patterns (e.g., $123.45, $123,456.78)
                    import re

                    price_patterns = [
                        r"\$[\d,]+\.?\d*",  # $123.45 or $123,456
                        r"[\d,]+\.?\d*\s*(?:USD|usd|dollars?)",  # 123.45 USD
                        r"price.*?[\d,]+\.?\d*",  # price 123.45
                        r"worth.*?[\d,]+\.?\d*",  # worth 123.45
                    ]
                    for pattern in price_patterns:
                        if re.search(pattern, result_text, re.IGNORECASE):
                            return True

            # Also check execution_results if present
            tool_results = execution_result.get("execution_results", [])
            for result in tool_results:
                if isinstance(result, dict) and "result" in result:
                    result_text = str(result["result"])
                    # Look for price patterns
                    import re

                    price_patterns = [
                        r"\$[\d,]+\.?\d*",  # $123.45 or $123,456
                        r"[\d,]+\.?\d*\s*(?:USD|usd|dollars?)",  # 123.45 USD
                        r"price.*?[\d,]+\.?\d*",  # price 123.45
                        r"worth.*?[\d,]+\.?\d*",  # worth 123.45
                    ]
                    for pattern in price_patterns:
                        if re.search(pattern, result_text, re.IGNORECASE):
                            return True

        # REMOVED: The aggressive safety mechanism that was causing premature completion
        # This was designed for reactive strategies, not plan-based ones

        return False

    def _create_step_result(
        self,
        action: str,
        step_description: str,
        execution_result: Dict[str, Any],
        reflection: Dict[str, Any],
    ) -> StrategyResult:
        """Create a standard step result."""
        # Create proper evaluation format for step results
        evaluation = {
            "is_complete": False,
            "confidence": reflection.get("confidence", 0.5),
            "reasoning": reflection.get("reasoning", "Step in progress"),
            "goal_achieved": reflection.get("goal_achieved", False),
            "completion_score": reflection.get("completion_score", 0.0),
            "reflection": reflection,
        }

        return StrategyResult(
            action_taken=action,
            should_continue=True,
            status="in_progress",
            evaluation=evaluation,
            result={
                "step": step_description,
                "execution_result": execution_result,
                "reflection": reflection,
            },
        )
