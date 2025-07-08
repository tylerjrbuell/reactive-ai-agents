"""
Plan-Execute-Reflect Context Adapter

Implements context management optimized for plan-execute-reflect strategy.
Preserves step execution history, tool results, and validation data.
"""

from typing import Dict, Any, List
from .base_context import BaseContextAdapter


class PlanExecuteReflectContextAdapter(BaseContextAdapter):
    """Context management optimized for plan-execute-reflect strategy."""

    async def manage_context(
        self,
        should_summarize: bool,
        should_prune: bool,
        config: Dict[str, Any],
        messages: List[Dict[str, Any]],
        system_message: Dict[str, Any],
    ) -> None:
        """
        Context management optimized for plan-execute-reflect strategy.
        Preserves step execution history, tool results, and validation data.
        """
        # Identify key messages to preserve
        user_messages = [m for m in messages if m.get("role") == "user"]
        last_user_message = user_messages[-1] if user_messages else None

        # Find execution-related messages (plan-execute-reflect specific)
        execution_messages = []
        step_messages = []
        tool_messages = []

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            # Preserve execution prompts and results
            if role == "user" and any(
                keyword in content.lower()
                for keyword in [
                    "execute step",
                    "step execution",
                    "current step",
                    "required tools",
                ]
            ):
                execution_messages.append(msg)

            # Preserve step validation and results
            elif role == "assistant" and any(
                keyword in content.lower()
                for keyword in [
                    "step",
                    "validation",
                    "success",
                    "failure",
                    "tool_calls",
                ]
            ):
                step_messages.append(msg)

            # Preserve tool results
            elif role == "tool":
                tool_messages.append(msg)

        # For plan-execute-reflect, be more conservative with summarization
        if should_summarize and len(messages) > config["max_messages"] * 2:
            # Only summarize if we're way over the limit
            await self._summarize_plan_execute_reflect_context(
                messages,
                system_message,
                last_user_message,
                execution_messages,
                step_messages,
                tool_messages,
            )
        elif should_prune and len(messages) > config["max_messages"] * 1.5:
            # Conservative pruning for plan-execute-reflect
            await self._prune_plan_execute_reflect_context(
                messages,
                system_message,
                last_user_message,
                execution_messages,
                step_messages,
                tool_messages,
            )

    async def _summarize_plan_execute_reflect_context(
        self,
        messages: List[Dict[str, Any]],
        system_message: Dict[str, Any],
        last_user_message: Dict[str, Any] | None,
        execution_messages: List[Dict[str, Any]],
        step_messages: List[Dict[str, Any]],
        tool_messages: List[Dict[str, Any]],
    ):
        """Summarize context while preserving plan-execute-reflect specific data."""
        # Extract structured data from all messages
        extracted_data = self._extract_structured_data_from_messages(messages)

        # Create summary of non-critical messages
        messages_to_summarize = [
            m
            for m in messages
            if m not in execution_messages
            and m not in step_messages
            and m not in tool_messages
            and m != system_message
            and m != last_user_message
        ]

        summary_text = "\n".join(
            f"[{m['role']}] {m['content']}"
            for m in messages_to_summarize
            if m.get("content")
        )

        summary_content = ""
        if summary_text.strip() and self.context.model_provider:
            enhanced_prompt = self._create_enhanced_summarization_prompt(
                summary_text, extracted_data
            )

            try:
                response = await self.context.model_provider.get_completion(
                    system="You are a summarization assistant for an AI agent using plan-execute-reflect strategy. Preserve all step execution data and tool results.",
                    prompt=enhanced_prompt,
                    options=self.context.model_provider_options,
                )
                summary_content = (
                    response.message.content.strip()
                    if response
                    else "Summary unavailable."
                )
            except Exception as e:
                if self.agent_logger:
                    self.agent_logger.error(
                        f"Plan-execute-reflect summarization failed: {e}"
                    )
                summary_content = "Summary unavailable due to error."

        # Append extracted data to summary
        if extracted_data:
            summary_content += self._format_extracted_data_for_summary(extracted_data)

        summary_message = {
            "role": "assistant",
            "content": f"[SUMMARY OF EARLIER CONTEXT]\n{summary_content}",
        }

        # Build new message list preserving critical plan-execute-reflect data
        new_messages = [system_message]
        if last_user_message:
            new_messages.append(last_user_message)
        new_messages.append(summary_message)

        # Preserve execution-related messages
        new_messages.extend(execution_messages[-3:])  # Keep last 3 execution messages
        new_messages.extend(step_messages[-5:])  # Keep last 5 step messages
        new_messages.extend(tool_messages[-10:])  # Keep last 10 tool messages

        self.context.session.messages = new_messages

        if self.agent_logger:
            self.agent_logger.info(
                f"Plan-execute-reflect summarized context to {len(new_messages)} messages (~{self.context.estimate_context_tokens()} tokens)."
            )

    async def _prune_plan_execute_reflect_context(
        self,
        messages: List[Dict[str, Any]],
        system_message: Dict[str, Any],
        last_user_message: Dict[str, Any] | None,
        execution_messages: List[Dict[str, Any]],
        step_messages: List[Dict[str, Any]],
        tool_messages: List[Dict[str, Any]],
    ):
        """Conservative pruning for plan-execute-reflect strategy."""
        # Find the most recent summary message
        summary_messages = [
            i
            for i, m in enumerate(self.context.session.messages)
            if m.get("role") == "assistant"
            and m.get("content", "").strip().startswith("[SUMMARY]")
        ]
        last_summary_message = (
            self.context.session.messages[summary_messages[-1]]
            if summary_messages
            else None
        )

        # Build new message list with conservative pruning
        new_messages = [system_message]
        if last_user_message:
            new_messages.append(last_user_message)
        if last_summary_message:
            new_messages.append(last_summary_message)

        # Preserve more execution-related messages than standard pruning
        new_messages.extend(execution_messages[-2:])  # Keep last 2 execution messages
        new_messages.extend(step_messages[-3:])  # Keep last 3 step messages
        new_messages.extend(tool_messages[-5:])  # Keep last 5 tool messages

        self.context.session.messages = new_messages

        if self.agent_logger:
            self.agent_logger.info(
                f"Plan-execute-reflect pruned context to {len(new_messages)} messages (~{self.context.estimate_context_tokens()} tokens)."
            )

    def _extract_structured_data_from_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract only the minimum, relevant data needed for the next step(s).
        Scans upcoming steps and required tool calls, and only includes fields needed as input for those tools.
        """
        # Find the next user message that contains a step execution prompt
        next_step_required_fields = set()
        next_step_tools = set()
        for msg in messages:
            if msg.get("role") == "user" and any(
                kw in msg.get("content", "").lower()
                for kw in ["execute step", "step:", "required tools:"]
            ):
                # Try to extract required tool(s) from the message
                import re

                tool_match = re.search(
                    r"required tools:\s*([\w, _-]+)", msg["content"], re.IGNORECASE
                )
                if tool_match:
                    tools = [t.strip() for t in tool_match.group(1).split(",")]
                    next_step_tools.update(tools)
                break
        # If we have tool names, get their input fields from tool signatures
        if next_step_tools and hasattr(self.context, "get_tool_signatures"):
            for tool_sig in self.context.get_tool_signatures():
                tool_name = tool_sig["function"]["name"]
                if tool_name in next_step_tools:
                    params = tool_sig["function"].get("parameters", {})
                    next_step_required_fields.update(params.keys())
        # Now, scan previous tool results for only these fields
        relevant_data = {}
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                try:
                    result = msg.get("content", "")
                    import json

                    if isinstance(result, str):
                        result = json.loads(result)
                    for field in next_step_required_fields:
                        if field in result and field not in relevant_data:
                            relevant_data[field] = result[field]
                except Exception:
                    continue
            # Stop if we've found all required fields
            if len(relevant_data) == len(next_step_required_fields):
                break
        # Label the data for clarity
        labeled_data = {}
        for field, value in relevant_data.items():
            label = f"{field} (for {', '.join(next_step_tools)})"
            labeled_data[label] = value
        return labeled_data
