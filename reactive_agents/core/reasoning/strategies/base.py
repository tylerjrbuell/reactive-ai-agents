from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import json
from reactive_agents.core.types.reasoning_types import (
    ReasoningStrategies,
    ReasoningContext,
    StrategySwitch,
)

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext
    from reactive_agents.app.agents.reactive_agent import ReactiveAgent


class BaseReasoningStrategy(ABC):
    """
    Base class for all reasoning strategies.

    This class provides a unified interface that leverages the existing
    agent infrastructure for thinking, tool execution, and message management.
    """

    def __init__(self, context: "AgentContext"):
        self.context = context
        self.agent_logger = context.agent_logger
        self.model_provider = context.model_provider
        # Get reference to the agent for using its methods
        self.agent: Optional["ReactiveAgent"] = getattr(context, "_agent", None)

    @abstractmethod
    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """
        Execute one iteration of this reasoning strategy.

        Args:
            task: The current task description
            reasoning_context: Context about current reasoning state

        Returns:
            Dictionary with iteration results including:
            - action_taken: What action was performed
            - result: The result of the action
            - should_continue: Whether to continue iterating
            - next_strategy: Optional strategy to switch to
        """
        pass

    def should_switch_strategy(
        self, reasoning_context: ReasoningContext
    ) -> Optional[StrategySwitch]:
        """
        Determine if strategy should be switched based on current context.

        Args:
            reasoning_context: Current reasoning state

        Returns:
            StrategySwitch object if switch recommended, None otherwise
        """
        # Default implementation - no switching
        return None

    @abstractmethod
    def get_strategy_name(self) -> ReasoningStrategies:
        """Get the strategy enum identifier."""
        pass

    # --- Unified thinking methods using base agent infrastructure ---

    async def _think(
        self, messages: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Use the agent's direct thinking method for simple completions.

        Args:
            messages: Optional message list, defaults to context messages
            **kwargs: Additional arguments passed to the agent's _think method

        Returns:
            The completion result or None if failed
        """
        if not self.agent:
            return await self._fallback_think(messages, **kwargs)

        # Use the agent's established thinking method
        kwargs.setdefault("messages", messages or self.context.session.messages)
        return await self.agent._think(**kwargs)

    async def _think_chain(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        use_tools: bool = True,
        remember_messages: bool = True,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Use the agent's thinking chain method for completions with tool support.

        Args:
            messages: Optional message list, defaults to context messages
            use_tools: Whether to enable tool usage
            remember_messages: Whether to remember messages in context
            **kwargs: Additional arguments passed to the agent's _think_chain method

        Returns:
            The completion result with tool calls or None if failed
        """
        if not self.agent:
            return await self._fallback_think_chain(
                messages, use_tools, remember_messages, **kwargs
            )

        # Use the agent's established thinking chain method
        kwargs.setdefault("messages", messages or self.context.session.messages)
        return await self.agent._think_chain(
            use_tools=use_tools, remember_messages=remember_messages, **kwargs
        )

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Use the agent's tool execution method for consistent tool handling.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tool execution results or None if failed
        """
        if not self.agent or not tool_calls:
            return None

        # Use the agent's established tool execution method
        return await self.agent._process_tool_calls(tool_calls)

    # --- Unified helper methods ---

    async def _generate_structured_response(
        self, system_prompt: str, user_prompt: str, use_tools: bool = False, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a structured response using the agent's thinking methods.

        Args:
            system_prompt: System prompt for the completion
            user_prompt: User prompt for the completion
            use_tools: Whether to enable tool usage
            **kwargs: Additional arguments

        Returns:
            Parsed JSON response or None if failed
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = await self._think_chain(
            messages=messages, use_tools=use_tools, remember_messages=False, **kwargs
        )

        if not result or not result.get("content"):
            return None

        try:
            return json.loads(result["content"])
        except json.JSONDecodeError:
            # If not JSON, return as text response
            return {"response": result["content"]}

    async def _execute_tool_decision(
        self, decision: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a tool based on a decision, using the agent's tool infrastructure.

        Args:
            decision: Decision dict containing tool_needed and parameters

        Returns:
            Tool execution result or None if no tool needed
        """
        tool_needed = decision.get("tool_needed")
        if not tool_needed:
            return None

        parameters = decision.get("parameters", {})

        # Create tool call structure compatible with agent's _process_tool_calls
        tool_calls = [
            {"name": tool_needed, "arguments": parameters, "id": f"call_{tool_needed}"}
        ]

        # Use the agent's tool execution method
        results = await self._execute_tool_calls(tool_calls)

        if results and len(results) > 0:
            return {
                "tool_name": tool_needed,
                "result": results[0].get("result"),
                "success": results[0].get("success", False),
            }

        return None

    async def _add_reasoning_message(
        self, content: str, role: str = "assistant"
    ) -> None:
        """
        Add a reasoning message to the context using established patterns.

        Args:
            content: Message content
            role: Message role (default: assistant)
        """
        self.context.session.messages.append({"role": role, "content": content})

    # --- Fallback methods for cases where agent reference is not available ---

    async def _fallback_think(
        self, messages: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Fallback thinking method when agent reference is not available."""
        try:
            if not self.model_provider:
                return None

            kwargs.setdefault("messages", messages or self.context.session.messages)
            result = await self.model_provider.get_completion(**kwargs)

            if result and result.message:
                return {"content": result.message.content}
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.warning(f"Fallback think failed: {e}")
        return None

    async def _fallback_think_chain(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        use_tools: bool = True,
        remember_messages: bool = True,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Fallback thinking chain method when agent reference is not available."""
        try:
            if not self.model_provider:
                return None

            kwargs.setdefault("messages", messages or self.context.session.messages)

            if use_tools:
                tool_signatures = self.context.get_tool_signatures()
                result = await self.model_provider.get_chat_completion(
                    tools=tool_signatures,
                    tool_use_required=bool(tool_signatures),
                    **kwargs,
                )
            else:
                result = await self.model_provider.get_completion(**kwargs)

            if result and result.message:
                return {
                    "content": result.message.content,
                    "tool_calls": getattr(result.message, "tool_calls", []),
                }
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.warning(f"Fallback think chain failed: {e}")
        return None

    # --- Legacy compatibility methods (deprecated) ---

    async def _think_and_decide(
        self, prompt: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Legacy method for backward compatibility.

        DEPRECATED: Use _generate_structured_response instead.
        """
        if self.agent_logger:
            self.agent_logger.warning(
                "_think_and_decide is deprecated, use _generate_structured_response instead"
            )

        return await self._generate_structured_response(
            system_prompt=prompt,
            user_prompt=kwargs.get("user_prompt", "Proceed with the analysis."),
            use_tools=False,
            **kwargs,
        )

    async def _execute_tool_if_needed(
        self, step_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Legacy method for backward compatibility.

        DEPRECATED: Use _execute_tool_decision instead.
        """
        if self.agent_logger:
            self.agent_logger.warning(
                "_execute_tool_if_needed is deprecated, use _execute_tool_decision instead"
            )

        return await self._execute_tool_decision(step_info)

    # --- Utility methods ---

    def _detect_stagnation(self, reasoning_context: ReasoningContext) -> bool:
        """Detect if the agent is stuck or making no progress."""
        return (
            reasoning_context.stagnation_count > 3
            or reasoning_context.error_count > 2
            or (
                reasoning_context.iteration_count > 5
                and not reasoning_context.success_indicators
            )
        )

    def _update_reasoning_context(
        self,
        reasoning_context: ReasoningContext,
        action_result: Optional[Dict[str, Any]] = None,
        success: bool = True,
    ) -> ReasoningContext:
        """Update reasoning context with latest iteration results."""

        # Update counters
        reasoning_context.iteration_count += 1

        if success and action_result:
            reasoning_context.stagnation_count = 0
            if "tool" in str(action_result):
                tool_name = action_result.get("tool_name", "unknown")
                reasoning_context.tool_usage_history.append(tool_name)
        else:
            reasoning_context.stagnation_count += 1
            reasoning_context.error_count += 1

        reasoning_context.last_action_result = action_result

        return reasoning_context
