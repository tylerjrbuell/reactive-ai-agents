from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING
import json
from reactive_agents.core.types.reasoning_types import (
    ReasoningStrategies,
    ReasoningContext,
)
from reactive_agents.core.reasoning.prompts.base import SystemPrompt
from .base import BaseReasoningStrategy
import logging

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class ReactiveStrategy(BaseReasoningStrategy):
    """
    Reactive reasoning strategy - no planning, pure prompt-response.
    Best for simple tasks that don't require multi-step reasoning.
    """

    def __init__(self, context: "AgentContext"):
        super().__init__(context)
        self.system_prompt = SystemPrompt(context)

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """Execute one reactive iteration - simple prompt and response."""
        try:
            if self.agent_logger:
                self.agent_logger.debug("🔀 Executing reactive iteration")

            # Use context messages for continuation
            messages = self.context.session.messages.copy()

            # Prepend a system message with tool instructions and available tools
            tool_signatures = self.context.get_tool_signatures()
            tool_names = [tool["function"]["name"] for tool in tool_signatures]

            system_content = (
                "You are a helpful assistant. Use the available tools when appropriate. "
                f"Available tools: {', '.join(tool_names)}. "
                "When you have the information needed to answer the user's question, use the final_answer tool. "
                "Do not repeat tool calls unnecessarily. "
                "If a tool can answer the user's question, call it directly. "
                "Do not ask for clarification if a tool can be used."
            )

            # Add detailed tool signatures for better understanding
            if tool_signatures:
                system_content += (
                    f"\n\nTool Signatures:\n{json.dumps(tool_signatures, indent=2)}"
                )

            # For the first iteration, use only system and user message
            if len(messages) == 1 and messages[0]["role"] == "user":
                messages = [{"role": "system", "content": system_content}, messages[0]]
            elif not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": system_content}] + messages
            else:
                # Replace the first system message to ensure clarity
                messages[0]["content"] = system_content

            # Debug logging (only in debug mode)
            if self.agent_logger:
                self.agent_logger.debug(f"Messages sent to model: {len(messages)}")
                self.agent_logger.debug(
                    f"Tool signatures: {len(self.context.get_tool_signatures())} tools"
                )

            # Simple chat completion without tools initially
            if not self.model_provider:
                return {
                    "action_taken": "model_call",
                    "result": {"error": "No model provider available"},
                    "should_continue": False,
                }

            result = await self.model_provider.get_chat_completion(
                messages=messages,
                tools=self.context.get_tool_signatures(),
                tool_use_required=self.context.tool_use_enabled,
                options=self.context.model_provider_options,
            )

            # Debug logging for model response
            if self.agent_logger:
                content = getattr(result.message, "content", "")
                tool_calls = getattr(result.message, "tool_calls", [])
                tool_calls_count = len(tool_calls) if tool_calls else 0
                self.agent_logger.debug(
                    f"Model response: content='{content[:50]}...', tool_calls={tool_calls_count}"
                )

            if not result or not result.message:
                return {
                    "action_taken": "model_call",
                    "result": {"error": "No response from model"},
                    "should_continue": False,
                }

            message = result.message
            content = message.content or ""
            tool_calls = message.tool_calls or []

            # Add assistant response to messages
            if content.strip():
                self.context.session.messages.append(
                    {"role": "assistant", "content": content}
                )

            # Handle tool calls if present
            if tool_calls and self.context.tool_use_enabled:
                # Execute tools through tool manager
                if self.context.tool_manager:
                    for tool_call in tool_calls:
                        await self.context.tool_manager.use_tool(tool_call)

                # Check if this was a final_answer tool call
                if any(
                    tool_call.function.name == "final_answer"
                    for tool_call in tool_calls
                ):
                    return {
                        "action_taken": "final_answer",
                        "result": {"tools_executed": len(tool_calls)},
                        "should_continue": False,
                    }

                return {
                    "action_taken": "tool_execution",
                    "result": {"tools_executed": len(tool_calls)},
                    "should_continue": True,
                }

            # Check if this looks like a final answer
            if any(
                phrase in content.lower()
                for phrase in [
                    "final answer",
                    "conclusion",
                    "result:",
                    "answer:",
                    "the current time is",
                    "the time is",
                ]
            ):
                self.context.session.final_answer = content
                return {
                    "action_taken": "final_answer",
                    "result": {"final_answer": content},
                    "should_continue": False,
                }

            # If we have tools available and the model didn't use them,
            # we should continue to allow the model to use tools
            if self.context.tool_use_enabled and self.context.get_tool_signatures():
                # Check if the content suggests the model needs clarification
                clarification_phrases = [
                    "could you please specify",
                    "which",
                    "if no",
                    "please specify",
                    "what do you mean",
                    "clarify",
                    "specify",
                ]

                if any(phrase in content.lower() for phrase in clarification_phrases):
                    # This is a clarification request, continue the conversation
                    return {
                        "action_taken": "clarification",
                        "result": {"response": content},
                        "should_continue": True,
                    }
                else:
                    # The model should use tools for this task, continue
                    return {
                        "action_taken": "completion",
                        "result": {"response": content},
                        "should_continue": True,
                    }
            else:
                # No tools available, treat as completion
                return {
                    "action_taken": "completion",
                    "result": {"response": content},
                    "should_continue": len(content.strip()) > 0,
                }

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Reactive iteration failed: {e}")

            return {
                "action_taken": "error",
                "result": {"error": str(e)},
                "should_continue": False,
            }

    def should_switch_strategy(
        self, reasoning_context: ReasoningContext
    ) -> Optional[Dict[str, Any]]:
        """Reactive strategy may switch if task becomes complex or errors occur."""

        # Switch to reflect-decide-act if getting errors or no progress
        if (
            reasoning_context.error_count >= 2
            or reasoning_context.stagnation_count >= 3
        ):
            return {
                "to_strategy": ReasoningStrategies.REFLECT_DECIDE_ACT,
                "reason": "Reactive approach hitting errors, need more structured reasoning",
                "confidence": 0.8,
                "trigger": "error_threshold",
            }

        # Switch if task seems to require planning
        if (
            reasoning_context.iteration_count >= 2
            and reasoning_context.task_classification
            and reasoning_context.task_classification.get("task_type")
            in ["multi_step", "planning"]
        ):
            return {
                "to_strategy": ReasoningStrategies.PLAN_EXECUTE_REFLECT,
                "reason": "Task requires structured multi-step approach",
                "confidence": 0.7,
                "trigger": "task_complexity",
            }

        return None

    def get_strategy_name(self) -> ReasoningStrategies:
        """Return the strategy identifier."""
        return ReasoningStrategies.REACTIVE
