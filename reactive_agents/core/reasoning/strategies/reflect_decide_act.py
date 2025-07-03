from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING
import json
from reactive_agents.core.types.reasoning_types import (
    ReasoningStrategies,
    ReasoningContext,
)
from reactive_agents.core.reasoning.prompts.base import (
    ReflectionPrompt,
    TaskPlanningPrompt,
    SystemPrompt,
    FinalAnswerPrompt,
)
from .base import BaseReasoningStrategy

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class ReflectDecideActStrategy(BaseReasoningStrategy):
    """
    Reflect-Decide-Act strategy: Reactive planning every iteration.
    Each iteration: reflect on current state, decide next best action, execute it.
    This is the truly reactive strategy we want as the new default.
    """

    def __init__(self, context: "AgentContext"):
        super().__init__(context)
        self.reflection_prompt = ReflectionPrompt(context)
        self.planning_prompt = TaskPlanningPrompt(context)
        self.system_prompt = SystemPrompt(context)
        self.final_answer_prompt = FinalAnswerPrompt(context)

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """Execute reflect-decide-act iteration."""

        if self.agent_logger:
            self.agent_logger.info("🔄 Reflect-Decide-Act Strategy: Iteration start")

        # Phase 1: Reflect on current state
        reflection_result = await self._reflect_on_progress(task, reasoning_context)

        # Phase 2: Decide on next action based on reflection
        decision_result = await self._decide_next_action(
            task, reasoning_context, reflection_result
        )

        # Phase 3: Act on the decision
        action_result = await self._execute_action(decision_result)

        # Determine if we should continue
        should_continue = self._should_continue_iterating(
            reflection_result, decision_result, action_result
        )

        return {
            "action_taken": "reflect_decide_act",
            "result": {
                "reflection": reflection_result,
                "decision": decision_result,
                "action": action_result,
            },
            "should_continue": should_continue,
            "strategy_used": "reflect_decide_act",
        }

    async def _reflect_on_progress(
        self, task: str, reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """Reflect on current progress and state."""

        if self.agent_logger:
            self.agent_logger.debug("🤔 Phase 1: Reflection")

        # Get available tools for better reflection
        available_tools = []
        if hasattr(self.context, "get_tool_signatures"):
            tool_signatures = self.context.get_tool_signatures()
            available_tools = [
                sig.get("function", {}).get("name") for sig in tool_signatures
            ]

        reflection_prompt = self.reflection_prompt.generate(
            task=task,
            reasoning_strategy="reflect_decide_act",
            task_classification=reasoning_context.task_classification,
            last_result=reasoning_context.last_action_result,
            available_tools=available_tools,
        )

        # Use the unified structured response method
        reflection = await self._generate_structured_response(
            system_prompt=reflection_prompt,
            user_prompt="Reflect on the current progress and state. If sufficient data has been collected, consider providing a final analysis.",
            use_tools=False,
        )

        if not reflection:
            # Fallback reflection
            return {
                "progress_assessment": "Unable to generate reflection",
                "completion_score": 0.5,
                "next_action_type": "continue",
                "reasoning": "Reflection generation failed, continuing with default approach",
            }

        return reflection

    async def _decide_next_action(
        self, task: str, reasoning_context: ReasoningContext, reflection: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decide on the optimal next action based on reflection."""

        if self.agent_logger:
            self.agent_logger.debug("🎯 Phase 2: Decision")

        # Check if reflection suggests finalizing
        if reflection.get("next_action_type") == "finalize":
            # Generate a dynamic final answer using the centralized prompt
            final_answer_prompt = self.final_answer_prompt.generate(
                task=task,
                reflection=reflection,
                reasoning_strategy="reflect_decide_act",
                task_classification=reasoning_context.task_classification,
            )

            # Generate dynamic final answer using the LLM
            final_answer_response = await self._generate_structured_response(
                system_prompt=final_answer_prompt,
                user_prompt="Generate a comprehensive final answer based on the task context and progress.",
                use_tools=False,
            )

            # Extract the final answer or use fallback
            dynamic_answer = "Task completed successfully"
            if final_answer_response and isinstance(final_answer_response, dict):
                # The new FinalAnswerPrompt returns a structured response with "final_answer" field
                dynamic_answer = (
                    final_answer_response.get("final_answer")
                    or final_answer_response.get("answer")
                    or final_answer_response.get("response")
                    or final_answer_response.get("content")
                    or str(final_answer_response)
                )
            elif final_answer_response and isinstance(final_answer_response, str):
                dynamic_answer = final_answer_response

            return {
                "next_step": "Provide final answer",
                "rationale": "Task appears complete based on reflection",
                "tool_needed": "final_answer",
                "parameters": {"answer": dynamic_answer},
                "confidence": reflection.get("completion_score", 0.8),
            }

        # Check if reflection suggests strategy switch - validate the strategy exists and setting allows it
        if reflection.get("next_action_type") == "switch_strategy":
            # First check if dynamic strategy switching is enabled
            if not self.context.enable_dynamic_strategy_switching:
                if self.agent_logger:
                    self.agent_logger.info(
                        "Strategy switching disabled by configuration, continuing with current strategy"
                    )
                # Continue with current approach instead of switching
                return {
                    "next_step": "Continue with current strategy",
                    "rationale": "Strategy switching is disabled in configuration",
                    "tool_needed": None,
                    "parameters": {},
                    "confidence": 0.6,
                }

            suggested_strategy = reflection.get("suggested_strategy")
            # Use actual ReasoningStrategies enum values dynamically
            # Get all strategies except the current one (no point switching to the same strategy)
            valid_strategies = [
                strategy.value
                for strategy in ReasoningStrategies
                if strategy != ReasoningStrategies.REFLECT_DECIDE_ACT
            ]

            if suggested_strategy in valid_strategies:
                return {
                    "next_step": "Switch reasoning strategy",
                    "rationale": f"Switching to {suggested_strategy} based on reflection",
                    "tool_needed": None,
                    "parameters": {},
                    "confidence": 0.6,
                    "strategy_switch": suggested_strategy,
                }
            else:
                if self.agent_logger:
                    self.agent_logger.warning(
                        f"Invalid strategy switch requested: {suggested_strategy}. Valid strategies: {valid_strategies}"
                    )

        # Get available tools and their signatures for better planning
        available_tools = []
        tool_signatures = []
        if hasattr(self.context, "get_tool_signatures"):
            tool_signatures = self.context.get_tool_signatures()
            available_tools = [
                sig.get("function", {}).get("name") for sig in tool_signatures
            ]

        # Generate next step using planning prompt
        planning_prompt = self.planning_prompt.generate(
            task=task,
            reasoning_strategy="reflect_decide_act",
            task_classification=reasoning_context.task_classification,
            error_context="; ".join(reflection.get("blockers", [])),
            available_tools=available_tools,
            tool_signatures=tool_signatures,
        )

        # Use the unified structured response method
        decision = await self._generate_structured_response(
            system_prompt=planning_prompt,
            user_prompt=f"Based on reflection: {json.dumps(reflection)}, decide the next optimal action.",
            use_tools=False,
        )

        if not decision:
            # Fallback decision
            return {
                "next_step": "Continue with current approach",
                "rationale": "Unable to generate specific decision, proceeding",
                "tool_needed": None,
                "parameters": {},
                "confidence": 0.3,
            }

        return decision

    async def _execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the decided action."""

        if self.agent_logger:
            self.agent_logger.debug("⚡ Phase 3: Action")

        # Check for strategy switch
        if "strategy_switch" in decision:
            return {
                "type": "strategy_switch",
                "new_strategy": decision["strategy_switch"],
                "success": True,
            }

        # Execute tool if needed using the unified method
        tool_needed = decision.get("tool_needed")
        if tool_needed:
            tool_result = await self._execute_tool_decision(decision)
            if tool_result:
                return {
                    "type": "tool_execution",
                    "tool": tool_needed,
                    "result": tool_result,
                    "success": tool_result.get("success", False),
                }

        # If no tool needed, add the step as reasoning/thinking using unified method
        step_description = decision.get("next_step", "Continue reasoning")
        reasoning_content = (
            f"Reasoning: {step_description}. {decision.get('rationale', '')}"
        )

        await self._add_reasoning_message(reasoning_content)

        return {
            "type": "reasoning_step",
            "step": step_description,
            "rationale": decision.get("rationale", ""),
            "success": True,
        }

    def _should_continue_iterating(
        self,
        reflection: Dict[str, Any],
        decision: Dict[str, Any],
        action: Dict[str, Any],
    ) -> bool:
        """Determine if we should continue iterating."""

        # Stop if final answer was given
        if (
            decision.get("tool_needed") == "final_answer"
            or action.get("tool") == "final_answer"
        ):
            return False

        # Stop if strategy switch requested
        if action.get("type") == "strategy_switch":
            return False

        # Stop if completion score is very high
        completion_score = reflection.get("completion_score", 0.0)
        if completion_score >= 0.95:
            return False

        # Stop if we have collected sufficient data and should provide final analysis
        if completion_score >= 0.6 and reflection.get("success_indicators"):
            # Check if we have enough data to provide a meaningful analysis
            success_indicators = reflection.get("success_indicators", [])
            data_collection_indicators = [
                indicator
                for indicator in success_indicators
                if "retrieved" in indicator.lower() or "collected" in indicator.lower()
            ]
            if len(data_collection_indicators) >= 2:
                # We have sufficient data, should provide final analysis
                return True

        # Continue otherwise
        return True

    def should_switch_strategy(
        self, reasoning_context: ReasoningContext
    ) -> Optional[Dict[str, Any]]:
        """Determine if strategy should be switched."""

        # Switch to planning if task requires long-term coordination
        if (
            reasoning_context.iteration_count >= 4
            and reasoning_context.task_classification
            and reasoning_context.task_classification.get("complexity_score", 0) > 0.7
        ):
            return {
                "to_strategy": ReasoningStrategies.PLAN_EXECUTE_REFLECT,
                "reason": "Task complexity requires structured planning approach",
                "confidence": 0.7,
                "trigger": "high_complexity",
            }

        # Switch to reactive if task is very simple
        if (
            reasoning_context.task_classification
            and reasoning_context.task_classification.get("task_type")
            == "simple_lookup"
            and reasoning_context.iteration_count >= 2
        ):
            return {
                "to_strategy": ReasoningStrategies.REACTIVE,
                "reason": "Simple task doesn't need structured reasoning",
                "confidence": 0.6,
                "trigger": "task_simplicity",
            }

        return None

    def get_strategy_name(self) -> ReasoningStrategies:
        """Return the strategy identifier."""
        return ReasoningStrategies.REFLECT_DECIDE_ACT
