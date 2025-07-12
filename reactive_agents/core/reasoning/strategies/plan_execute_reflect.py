from __future__ import annotations
from typing import Dict, Any, List, Optional

from reactive_agents.core.types.reasoning_types import ReasoningContext
from reactive_agents.core.reasoning.engine import ReasoningEngine
from reactive_agents.core.reasoning.strategies.base import (
    BaseReasoningStrategy,
    StrategyResult,
    StrategyCapabilities,
)


class PlanExecuteReflectStrategy(BaseReasoningStrategy):
    """
    Simplified Plan-Execute-Reflect strategy.
    1. Plan: Break down goals into actionable steps
    2. Execute: Execute specific steps with proper tool mapping
    3. Reflect: Evaluate progress and determine next actions
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
        return (
            "Plan-execute-reflect reasoning with planning, execution, and reflection."
        )

    async def initialize(self, task: str, reasoning_context: ReasoningContext) -> None:
        """Initialize the plan-execute-reflect strategy."""
        # Set the strategy in context manager
        self.context_manager.set_active_strategy(self.name)

        # Generate initial plan
        plan_result = await self._generate_plan(task, reasoning_context)
        plan_steps = plan_result.get("plan_steps", [])

        # Store plan in preserved context
        self.engine.preserve_context("plan", plan_steps)
        self.engine.preserve_context("current_step_index", 0)

        # Add initial context
        self.context_manager.add_message(
            role="user",
            content=f"Task: {task}\nPlan created with {len(plan_steps)} steps.",
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

            # Execute the current step
            use_native_tools = getattr(
                self.context, "supports_native_tool_calling", True
            )
            execution_result = await self.execute_with_tools(
                task, step_description, use_native_tools=use_native_tools
            )

            # Reflect on the step execution
            reflection = await self.reflect_on_progress(
                task, execution_result, reasoning_context
            )

            # Check if we have found the information needed to complete the task
            if self._should_complete_task(task, execution_result, reflection):
                return await self._handle_task_completion(
                    task, execution_result, reflection
                )

            # Determine next action based on reflection
            next_action = reflection.get("next_action", "continue")

            if next_action == "complete":
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
                self.engine.preserve_context("current_step_index", step_index + 1)
                return self._create_step_result(
                    "step_completed", step_description, execution_result, reflection
                )

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Error in plan-execute-reflect strategy: {e}")

            return StrategyResult(
                action_taken="error_occurred",
                should_continue=True,
                status="error",
                result={"error": str(e), "recovery_action": "retry_current_step"},
            )

    async def _generate_plan(
        self, task: str, reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """Generate a plan for the task."""
        plan_prompt = f"""Task: {task}

Please create a detailed plan to complete this task. Break it down into specific, actionable steps.

Respond with a JSON object in this format:
{{
    "plan_steps": [
        {{
            "step_number": 1,
            "description": "Clear description of what to do",
            "purpose": "Why this step is needed"
        }},
        {{
            "step_number": 2,
            "description": "Next step description",
            "purpose": "Why this step is needed"
        }}
    ],
    "reasoning": "Your reasoning for this plan"
}}

Only respond with valid JSON, no additional text."""

        self.context_manager.add_message(role="user", content=plan_prompt)
        result = await self._think_chain(use_tools=False)

        if result and result.result_json:
            return result.result_json

        # Fallback plan
        return {
            "plan_steps": [
                {
                    "step_number": 1,
                    "description": f"Work on the task: {task}",
                    "purpose": "Complete the given task",
                }
            ],
            "reasoning": "Fallback plan due to planning failure",
        }

    async def _handle_plan_completion(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """Handle when all plan steps are completed."""
        # Evaluate overall task completion
        evaluation = await self.evaluate_task_completion(
            task, "All plan steps completed"
        )

        if evaluation.get("is_complete", False):
            final_answer = await self.generate_final_answer(
                task, "All plan steps completed successfully"
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
        Handle task completion and generate final answer using the model's final answer prompt.

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

        # Use the engine's final answer prompt to get the answer from the model
        final_answer_json = await self.engine.generate_final_answer(
            task,
            execution_summary=execution_summary,
            execution_result=execution_result,
            reflection=reflection,
        )
        actual_answer = (
            final_answer_json if final_answer_json else {"final_answer": None}
        )

        # Set the session's final answer for consistency
        if self.context and self.context.session:
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

        return StrategyResult(
            action_taken="task_completed",
            should_continue=False,
            final_answer=actual_answer.get("final_answer"),
            status="completed",
            evaluation=reflection,
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

        Args:
            task: The original task
            execution_result: Results from tool execution
            reflection: Reflection on the progress

        Returns:
            True if the task should be completed, False otherwise
        """
        # Check if we have a final answer from a tool
        if "final_answer" in execution_result:
            return True

        # Check if reflection indicates goal is achieved
        if reflection.get("goal_achieved", False):
            return True

        # Check if completion score is high enough
        if reflection.get("completion_score", 0) >= 0.7:
            return True

        # Check if reflection indicates completion
        if reflection.get("next_action") == "complete":
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

        # Safety mechanism: If we have multiple successful tool calls and the task seems to be about gathering information
        if any(
            keyword in task.lower()
            for keyword in ["what", "find", "get", "search", "look"]
        ):
            tool_calls = execution_result.get("tool_calls", [])
            if (
                len(tool_calls) >= 2
            ):  # If we've made multiple tool calls, we likely have enough info
                return True

        # Additional safety: If we have any successful tool calls and the reflection shows progress
        tool_calls = execution_result.get("tool_calls", [])
        if tool_calls and reflection.get("progress_assessment"):
            # If we have tool results and the reflection indicates progress, we likely have enough info
            return True

        return False

    def _create_step_result(
        self,
        action: str,
        step_description: str,
        execution_result: Dict[str, Any],
        reflection: Dict[str, Any],
    ) -> StrategyResult:
        """Create a standard step result."""
        return StrategyResult(
            action_taken=action,
            should_continue=True,
            status="in_progress",
            evaluation=reflection,
            result={
                "step": step_description,
                "execution_result": execution_result,
                "reflection": reflection,
            },
        )
