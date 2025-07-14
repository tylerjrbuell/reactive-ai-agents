from __future__ import annotations
from typing import Dict, Any, Optional, List
from reactive_agents.core.types.reasoning_types import ReasoningContext
from reactive_agents.core.reasoning.engine import ReasoningEngine
from reactive_agents.core.reasoning.strategies.base import (
    BaseReasoningStrategy,
    StrategyResult,
    StrategyCapabilities,
)


class ReflectDecideActStrategy(BaseReasoningStrategy):
    """
    Simplified Reflect-Decide-Act strategy.
    1. Reflect: Evaluate current progress and blockers
    2. Decide: Choose the next action or plan
    3. Act: Execute the chosen action (tool or answer)
    """

    @property
    def name(self) -> str:
        return "reflect_decide_act"

    @property
    def capabilities(self) -> List[StrategyCapabilities]:
        return [
            StrategyCapabilities.REFLECTION,
            StrategyCapabilities.PLANNING,
            StrategyCapabilities.TOOL_EXECUTION,
        ]

    @property
    def description(self) -> str:
        return "Reflect-decide-act reasoning with reflection, decision, and action."

    async def initialize(self, task: str, reasoning_context: ReasoningContext) -> None:
        """Initialize the reflect-decide-act strategy."""
        # Set the strategy in context manager
        self.context_manager.set_active_strategy(self.name)

        # Initialize cycle tracking
        self.engine.preserve_context("cycle_count", 0)
        self.engine.preserve_context("last_action_result", None)

        # Add initial context
        self.context_manager.add_message(
            role="user",
            content=f"Task: {task}\nI will approach this using reflect-decide-act cycles.",
        )

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """
        Execute one iteration of the reflect-decide-act strategy:
        1. Reflect on current progress
        2. Decide next action
        3. Act on the decision
        4. Evaluate if should continue or complete
        """
        try:
            # Get cycle state
            cycle_count = self.engine.get_preserved_context("cycle_count") or 0
            last_action_result = self.engine.get_preserved_context("last_action_result")

            # Phase 1: Reflection
            reflection = await self.reflect_on_progress(
                task, last_action_result or {}, reasoning_context
            )

            # Phase 2: Decision - determine next action
            decision = await self._make_decision(task, reflection, reasoning_context)

            # Phase 3: Act - execute the decision
            action_result = await self._execute_decision(
                task, decision, reasoning_context
            )

            # Update preserved context
            self.engine.preserve_context("cycle_count", cycle_count + 1)
            self.engine.preserve_context("last_action_result", action_result)

            # Phase 4: Evaluate if task is complete
            goal_achieved = reflection.get("goal_achieved", False)
            next_action = reflection.get("next_action", "continue")

            if self.agent_logger:
                self.agent_logger.info(
                    f"🔍 Completion check: goal_achieved={goal_achieved}, next_action={next_action}"
                )

            if goal_achieved or next_action == "complete":
                if self.agent_logger:
                    self.agent_logger.info(
                        "✅ Goal achieved or next_action=complete, calling _handle_task_completion"
                    )
                return await self._handle_task_completion(
                    task, action_result, reflection, cycle_count + 1
                )

            # Continue with next cycle
            # Create proper evaluation format for cycle results
            evaluation = {
                "is_complete": False,
                "confidence": reflection.get("confidence", 0.5),
                "reasoning": reflection.get("reasoning", "Cycle in progress"),
                "goal_achieved": reflection.get("goal_achieved", False),
                "completion_score": reflection.get("completion_score", 0.0),
                "reflection": reflection,
            }

            return StrategyResult(
                action_taken="reflect_decide_act_cycle",
                should_continue=True,
                status="in_progress",
                evaluation=evaluation,
                result={
                    "cycle_count": cycle_count + 1,
                    "reflection": reflection,
                    "decision": decision,
                    "action_result": action_result,
                },
            )

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Error in reflect-decide-act strategy: {e}")

            return StrategyResult(
                action_taken="error_occurred",
                should_continue=True,
                status="error",
                result={"error": str(e), "recovery_action": "retry_current_cycle"},
            )

    async def _make_decision(
        self, task: str, reflection: Dict[str, Any], reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """Make a decision about the next action based on reflection."""
        decision_prompt = f"""Task: {task}

Current reflection:
{reflection}

Based on the reflection, what should I do next? Please provide a decision in this JSON format:
{{
    "next_action": "use_tool|provide_answer|gather_info",
    "action_description": "Detailed description of what to do",
    "reasoning": "Why this action is the best choice",
    "expected_outcome": "What I expect to achieve"
}}

Only respond with valid JSON, no additional text."""

        self.context_manager.add_message(role="user", content=decision_prompt)
        result = await self._think_chain(use_tools=False)

        if result and result.result_json:
            return result.result_json

        # Fallback decision
        return {
            "next_action": "use_tool",
            "action_description": f"Continue working on the task: {task}",
            "reasoning": "Fallback decision due to decision generation failure",
            "expected_outcome": "Make progress on the task",
        }

    async def _execute_decision(
        self, task: str, decision: Dict[str, Any], reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """Execute the decision made in the decide phase."""
        next_action = decision.get("next_action", "use_tool")
        action_description = decision.get(
            "action_description", "Continue working on the task"
        )

        if next_action == "use_tool":
            # Execute tools to complete the action
            use_native_tools = getattr(
                self.context, "supports_native_tool_calling", True
            )
            return await self.execute_with_tools(
                task, action_description, use_native_tools=use_native_tools
            )
        elif next_action == "provide_answer":
            # Generate final answer
            return await self.generate_final_answer(task, action_description)
        elif next_action == "gather_info":
            # Gather more information through thinking
            info_prompt = f"Task: {task}\n\n{action_description}\n\nPlease provide the requested information."
            self.context_manager.add_message(role="user", content=info_prompt)
            result = await self._think_chain(use_tools=False)

            return {
                "action_type": "gather_info",
                "information": (
                    result.content if result else "Unable to gather information"
                ),
                "method": "thinking",
            }
        else:
            # Default to tool execution
            use_native_tools = getattr(
                self.context, "supports_native_tool_calling", True
            )
            return await self.execute_with_tools(
                task, action_description, use_native_tools=use_native_tools
            )

    async def _handle_task_completion(
        self,
        task: str,
        action_result: Dict[str, Any],
        reflection: Dict[str, Any],
        cycle_count: int,
    ) -> StrategyResult:
        """Handle task completion."""
        # Check if we already have a final answer in the session (set by final_answer tool)
        session_final_answer = None
        if self.context and self.context.session:
            session_final_answer = self.context.session.final_answer

        if session_final_answer:
            final_answer = {"final_answer": session_final_answer}
        else:
            if self.agent_logger:
                self.agent_logger.info(
                    "🔄 No final answer in session, generating one..."
                )
            # Fallback: Generate final answer
            execution_summary = f"Completed task using reflect-decide-act strategy after {cycle_count} cycles"
            final_answer = await self.generate_final_answer(task, execution_summary)

        # Create proper evaluation with is_complete flag
        evaluation = {
            "is_complete": True,
            "confidence": reflection.get("confidence", 1.0),
            "reasoning": reflection.get("reasoning", "Task completed successfully"),
            "goal_achieved": reflection.get("goal_achieved", True),
            "completion_score": reflection.get("completion_score", 1.0),
            "reflection": reflection,
        }

        final_answer_value = final_answer.get("final_answer")

        return StrategyResult(
            action_taken="task_completed",
            should_continue=False,
            final_answer=final_answer_value,
            status="completed",
            evaluation=evaluation,
            result={
                "cycle_count": cycle_count,
                "action_result": action_result,
                "reflection": reflection,
                "final_answer": final_answer,
            },
        )
