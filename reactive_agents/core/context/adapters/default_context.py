"""
Default Context Adapter

Implements standard context management for most strategies.
Uses the original summarization and pruning logic.
"""

from typing import Dict, Any, List
from .base_context import BaseContextAdapter


class DefaultContextAdapter(BaseContextAdapter):
    """Default context management for standard strategies."""

    async def manage_context(
        self,
        should_summarize: bool,
        should_prune: bool,
        config: Dict[str, Any],
        messages: List[Dict[str, Any]],
        system_message: Dict[str, Any],
    ) -> None:
        """
        Standard context management for other strategies.
        Uses the original summarization and pruning logic.
        """
        non_system_messages = messages[1:]

        # Identify the most recent user message
        user_messages = [m for m in non_system_messages if m.get("role") == "user"]
        last_user_message = user_messages[-1] if user_messages else None

        # Identify the most recent summary message (assistant, content starts with [SUMMARY])
        summary_messages = [
            i
            for i, m in enumerate(self.context.session.messages)
            if m.get("role") == "assistant"
            and m.get("content", "").strip().startswith("[SUMMARY]")
        ]
        last_summary_index = summary_messages[-1] if summary_messages else 0
        last_summary_message = (
            self.context.session.messages[last_summary_index]
            if summary_messages
            else None
        )
        # Store last summary index in session for future incremental summarization
        self.context.session.last_summary_index = last_summary_index

        # Identify the current assistant turn (last assistant message not a summary)
        assistant_messages = [
            m
            for m in non_system_messages
            if m.get("role") == "assistant"
            and not m.get("content", "").strip().startswith("[SUMMARY]")
        ]
        current_assistant_turn = assistant_messages[-1] if assistant_messages else None

        # If summarization is needed, perform incremental/rolling summary with data extraction
        if should_summarize:
            # Only summarize messages after the last summary (or after system if no summary yet)
            start_idx = last_summary_index + 1 if summary_messages else 1
            # Find messages to summarize: tool or assistant messages (including tool summaries)
            to_summarize = [
                m
                for m in self.context.session.messages[start_idx:]
                if m.get("role") in ("assistant", "tool")
                and m is not current_assistant_turn
            ]

            # Extract structured data from messages before summarizing
            extracted_data = self._extract_structured_data_from_messages(to_summarize)

            summary_text = "\n".join(
                f"[{m['role']}] {m['content']}"
                for m in to_summarize
                if m.get("content")
            )

            summary_content = ""
            if summary_text.strip() and self.context.model_provider:
                # Enhanced prompt with extracted data preservation
                enhanced_prompt = self._create_enhanced_summarization_prompt(
                    summary_text, extracted_data
                )

                try:
                    response = await self.context.model_provider.get_completion(
                        system="You are a summarization assistant for an AI agent. You must preserve all structured data that might be needed later.",
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
                        self.agent_logger.error(f"Summarization failed: {e}")
                    summary_content = "Summary unavailable due to error."

            # Append extracted data to summary if available
            if extracted_data:
                summary_content += self._format_extracted_data_for_summary(
                    extracted_data
                )

            summary_message = {
                "role": "assistant",
                "content": f"[SUMMARY OF EARLIER CONTEXT]\n{summary_content}",
            }
            # Build new message list: keep all messages up to last summary, insert new summary, then keep unsummarized messages
            new_messages = self.context.session.messages[:start_idx]
            new_messages.append(summary_message)
            # Keep any messages after the summarized block that are not tool/assistant (e.g., user, current assistant turn)
            for m in self.context.session.messages[start_idx:]:
                if m not in to_summarize:
                    new_messages.append(m)
            self.context.session.messages = new_messages
            # Update last_summary_index
            self.context.session.last_summary_index = len(new_messages) - 1
            if self.agent_logger:
                self.agent_logger.info(
                    f"Incrementally summarized context to {len(new_messages)} messages (~{self.context.estimate_context_tokens()} tokens). Last summary index: {self.context.session.last_summary_index}"
                )
            return  # If summarized, no further pruning needed in this pass

        # If only pruning is needed (not summarizing), keep system, last user, last summary, and current assistant turn
        if should_prune:
            new_messages = [system_message]
            if last_user_message:
                new_messages.append(last_user_message)
            if last_summary_message:
                new_messages.append(last_summary_message)
            if current_assistant_turn:
                new_messages.append(current_assistant_turn)
            self.context.session.messages = new_messages
            if self.agent_logger:
                self.agent_logger.info(
                    f"Pruned context to {len(new_messages)} messages (~{self.context.estimate_context_tokens()} tokens)."
                )
