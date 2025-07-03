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
)
from .base import BaseReasoningStrategy

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class MemoryEnhancedStrategy(BaseReasoningStrategy):
    """
    Memory-Enhanced reasoning strategy that leverages vector memory for better decision making.

    This strategy:
    1. Retrieves relevant memories for the current task
    2. Uses memory insights to inform reflection and planning
    3. Learns from past successes and failures
    4. Adapts strategies based on historical patterns
    5. Avoids repeating unsuccessful patterns
    """

    def __init__(self, context: "AgentContext"):
        super().__init__(context)
        self.reflection_prompt = ReflectionPrompt(context)
        self.planning_prompt = TaskPlanningPrompt(context)
        self.system_prompt = SystemPrompt(context)

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """Execute one memory-enhanced iteration."""

        if self.agent_logger:
            self.agent_logger.debug("🧠 Phase 1: Memory-Enhanced Reflection")

        # Step 1: Enhanced reflection with memory insights
        reflection = await self._enhanced_reflection(task, reasoning_context)

        if self.agent_logger:
            self.agent_logger.debug("🎯 Phase 2: Memory-Informed Decision")

        # Step 2: Decision making informed by memory
        decision = await self._memory_informed_decision(
            task, reasoning_context, reflection
        )

        if self.agent_logger:
            self.agent_logger.debug("⚡ Phase 3: Memory-Guided Action")

        # Step 3: Execute action with memory context
        action = await self._execute_memory_guided_action(decision)

        # Step 4: Store insights for future learning
        await self._store_learning_insights(task, reflection, decision, action)

        # Determine if we should continue
        should_continue = self._should_continue_iterating(reflection, decision, action)

        return {
            "action_taken": "memory_enhanced_iteration",
            "result": {
                "reflection": reflection,
                "decision": decision,
                "action": action,
                "memory_insights": reflection.get("learning_insights", []),
            },
            "should_continue": should_continue,
            "strategy_used": "memory_enhanced",
        }

    async def _enhanced_reflection(
        self, task: str, reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """Enhanced reflection that incorporates memory insights."""

        # Get relevant memories for this task
        relevant_memories = []
        if hasattr(self.context, "memory_manager") and self.context.memory_manager:
            try:
                memory_manager = self.context.memory_manager
                if hasattr(memory_manager, "get_context_memories"):
                    # Get memories related to this task
                    relevant_memories = await memory_manager.get_context_memories(  # type: ignore
                        task, max_items=5
                    )
            except Exception as e:
                if self.agent_logger:
                    self.agent_logger.debug(
                        f"Failed to retrieve memories for reflection: {e}"
                    )

        # Inject or update the [MEMORY] message in the context
        await self.context.inject_memory_message(relevant_memories)

        # Generate reflection prompt with memory context (no longer passing memories to system prompt)
        reflection_prompt = self.reflection_prompt.generate(
            task=task,
            reasoning_strategy="memory_enhanced",
            task_classification=reasoning_context.task_classification,
            last_result=reasoning_context.last_action_result,
        )

        reflection = await self._think_and_reflect(reflection_prompt)

        if not reflection:
            # Fallback reflection
            return {
                "progress_assessment": "Unable to assess progress",
                "completion_score": 0.5,
                "blockers": ["Unable to generate reflection"],
                "success_indicators": [],
                "next_action_type": "continue",
                "learning_insights": [],
                "recommendations": ["Continue with current approach"],
            }

        return reflection

    async def _memory_informed_decision(
        self, task: str, reasoning_context: ReasoningContext, reflection: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make decisions informed by memory insights."""

        # Get tool-specific memories if tools are involved
        tool_memories = []
        if hasattr(self.context, "memory_manager") and self.context.memory_manager:
            try:
                memory_manager = self.context.memory_manager
                if hasattr(memory_manager, "search_memory"):
                    # Search for tool-related memories
                    tool_memories = await memory_manager.search_memory(  # type: ignore
                        "tool usage patterns", n_results=3, memory_types=["tool_result"]
                    )
            except Exception as e:
                if self.agent_logger:
                    self.agent_logger.debug(f"Failed to retrieve tool memories: {e}")

        # Inject or update the [MEMORY] message in the context
        await self.context.inject_memory_message(tool_memories)

        # Generate decision prompt with memory context (no longer passing memories to planning prompt)
        decision_prompt = self.planning_prompt.generate(
            task=task,
            reasoning_strategy="memory_enhanced",
            task_classification=reasoning_context.task_classification,
            error_context="; ".join(reflection.get("blockers", [])),
        )

        decision = await self._think_and_decide(
            decision_prompt,
            user_prompt=f"Based on reflection and memory insights: {json.dumps(reflection)}, decide the next optimal action.",
        )

        if not decision:
            # Fallback decision
            return {
                "next_step": "Continue with current approach",
                "rationale": "Unable to generate specific decision, proceeding",
                "tool_needed": None,
                "parameters": {},
                "confidence": 0.3,
                "memory_influence": "No specific memory influence",
                "avoid_patterns": [],
            }

        return decision

    async def _execute_memory_guided_action(
        self, decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute action with memory-guided context."""

        # Check for strategy switch
        if "strategy_switch" in decision:
            return {
                "type": "strategy_switch",
                "new_strategy": decision["strategy_switch"],
                "success": True,
            }

        # Execute tool if needed
        tool_needed = decision.get("tool_needed")
        if tool_needed:
            tool_result = await self._execute_tool_if_needed(decision)
            if tool_result:
                return {
                    "type": "tool_execution",
                    "tool": tool_needed,
                    "result": tool_result,
                    "success": "error" not in tool_result,
                    "memory_influence": decision.get("memory_influence", ""),
                }

        # If no tool needed, add the step as reasoning/thinking
        step_description = decision.get("next_step", "Continue reasoning")
        memory_influence = decision.get("memory_influence", "")

        reasoning_content = (
            f"Reasoning: {step_description}. {decision.get('rationale', '')}"
        )
        if memory_influence:
            reasoning_content += f" Memory influence: {memory_influence}"

        self.context.session.messages.append(
            {
                "role": "assistant",
                "content": reasoning_content,
            }
        )

        return {
            "type": "reasoning_step",
            "step": step_description,
            "rationale": decision.get("rationale", ""),
            "success": True,
            "memory_influence": memory_influence,
        }

    async def _store_learning_insights(
        self,
        task: str,
        reflection: Dict[str, Any],
        decision: Dict[str, Any],
        action: Dict[str, Any],
    ):
        """Store insights for future learning."""

        if hasattr(self.context, "memory_manager") and self.context.memory_manager:
            try:
                memory_manager = self.context.memory_manager
                if hasattr(memory_manager, "store_reflection_memory"):
                    # Store reflection with learning insights
                    reflection_content = f"""
                    Task: {task}
                    Reflection: {reflection.get('progress_assessment', '')}
                    Learning Insights: {', '.join(reflection.get('learning_insights', []))}
                    Recommendations: {', '.join(reflection.get('recommendations', []))}
                    Decision: {decision.get('next_step', '')}
                    Memory Influence: {decision.get('memory_influence', '')}
                    """

                    await memory_manager.store_reflection_memory(  # type: ignore
                        reflection_content.strip(),
                        {
                            "task": task,
                            "completion_score": reflection.get("completion_score", 0),
                            "success": action.get("success", False),
                            "strategy": "memory_enhanced",
                        },
                    )
            except Exception as e:
                if self.agent_logger:
                    self.agent_logger.debug(f"Failed to store learning insights: {e}")

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

        # Stop if memory suggests this pattern has failed before
        avoid_patterns = decision.get("avoid_patterns", [])
        if avoid_patterns and len(avoid_patterns) > 2:
            # If we're avoiding many patterns, consider switching strategies
            return False

        # Continue otherwise
        return True

    def should_switch_strategy(
        self, reasoning_context: ReasoningContext
    ) -> Optional[Dict[str, Any]]:
        """Determine if strategy should be switched based on memory patterns."""

        # Switch to reactive if memory shows simple tasks work better with direct approach
        if (
            reasoning_context.iteration_count >= 2
            and reasoning_context.task_classification
            and reasoning_context.task_classification.get("task_type")
            == "simple_lookup"
        ):
            return {
                "to_strategy": ReasoningStrategies.REACTIVE,
                "reason": "Memory suggests simple tasks work better with direct approach",
                "confidence": 0.7,
                "trigger": "memory_pattern_simple_task",
            }

        # Switch to reflect-decide-act if memory shows complex tasks need more structure
        if (
            reasoning_context.iteration_count >= 3
            and reasoning_context.task_classification
            and reasoning_context.task_classification.get("complexity_score", 0) > 0.6
        ):
            return {
                "to_strategy": ReasoningStrategies.REFLECT_DECIDE_ACT,
                "reason": "Memory suggests complex tasks need more structured reasoning",
                "confidence": 0.8,
                "trigger": "memory_pattern_complex_task",
            }

        return None

    async def _think_and_reflect(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Helper method to get model reflection and parse JSON response."""
        try:
            if not self.model_provider:
                return None

            response = await self.model_provider.get_completion(
                system=prompt,
                prompt="Please provide your reflection on the current situation.",
                options=self.context.model_provider_options,
            )

            if response and response.message.content:
                try:
                    return json.loads(response.message.content)
                except json.JSONDecodeError:
                    # If not JSON, return as text response
                    return {"response": response.message.content}

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.warning(f"Think and reflect failed: {e}")

        return None

    def get_strategy_name(self) -> ReasoningStrategies:
        """Return the strategy identifier."""
        return (
            ReasoningStrategies.ADAPTIVE
        )  # Use ADAPTIVE for now, could add MEMORY_ENHANCED to enum
