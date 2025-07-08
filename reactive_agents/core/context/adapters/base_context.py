"""
Base Context Adapter

Defines the interface for strategy-specific context management adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class BaseContextAdapter(ABC):
    """Base class for strategy-specific context management adapters."""

    def __init__(self, context: "AgentContext"):
        self.context = context
        self.agent_logger = context.agent_logger

    @abstractmethod
    async def manage_context(
        self,
        should_summarize: bool,
        should_prune: bool,
        config: Dict[str, Any],
        messages: List[Dict[str, Any]],
        system_message: Dict[str, Any],
    ) -> None:
        """
        Manage context for the specific strategy.

        Args:
            should_summarize: Whether summarization is needed
            should_prune: Whether pruning is needed
            config: Context management configuration
            messages: Current session messages
            system_message: System message
        """
        pass

    def _extract_structured_data_from_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract structured data from messages using the context's data extractor."""
        if not hasattr(self.context, "tool_manager") or not self.context.tool_manager:
            return {}

        # Get all search data from the tool manager's SearchDataManager
        all_search_data = (
            self.context.tool_manager.search_data_manager.get_all_search_data()
        )

        # Also extract data from message content using DataExtractor
        combined_text = "\n".join([msg.get("content", "") for msg in messages])
        extracted_data = self.context.tool_manager.data_extractor.extract_all(
            combined_text
        )

        # Combine search data with extracted data
        combined_data = {
            "search_tool_data": all_search_data,
            "extracted_content": extracted_data.dict(),
        }

        return combined_data

    def _create_enhanced_summarization_prompt(
        self, summary_text: str, extracted_data: Dict[str, Any]
    ) -> str:
        """Create an enhanced summarization prompt with extracted data context."""
        from reactive_agents.core.reasoning.prompts.agent_prompts import (
            CONTEXT_SUMMARIZATION_PROMPT,
        )

        prompt = CONTEXT_SUMMARIZATION_PROMPT + "\n\n"

        # Add extracted data context if available
        if extracted_data:
            prompt += "\nIMPORTANT STRUCTURED DATA TO PRESERVE:\n"

            # Include search tool data (comprehensive data preservation)
            search_data = extracted_data.get("search_tool_data", {})
            for tool_name, tool_data in search_data.items():
                prompt += f"\n{tool_name.upper()} RESULTS:\n"
                tool_extracted = tool_data.get("extracted_data", {})

                # Preserve all data types found by the DataExtractor
                for data_type, values in tool_extracted.items():
                    if values:
                        if isinstance(values, list) and values:
                            # Limit output but preserve more items for critical data
                            limit = (
                                10
                                if data_type in ["emails", "urls", "phone_numbers"]
                                else 5
                            )
                            prompt += f"  - {data_type}: {values[:limit]}\n"
                        elif values:
                            prompt += f"  - {data_type}: {str(values)[:200]}\n"  # Truncate very long values

                # Include raw result snippets for pattern matching
                raw_result = str(tool_data.get("result", ""))
                if raw_result and len(raw_result) > 100:
                    # Extract key patterns that might contain IDs or important data
                    import re

                    # Generic ID patterns (hexadecimal, numeric, alphanumeric with specific lengths)
                    id_patterns = [
                        r"\b[a-f0-9]{16,}\b",  # Long hex strings (like email IDs)
                        r"\b[A-Za-z0-9]{20,}\b",  # Long alphanumeric strings (API keys, tokens)
                        r"\b\d{10,}\b",  # Long numeric IDs
                        r"\b[A-Z]{2,}[0-9]{6,}\b",  # Codes like ABC123456
                    ]

                    found_ids = []
                    for pattern in id_patterns:
                        matches = re.findall(pattern, raw_result)
                        found_ids.extend(matches[:5])  # Limit to prevent overwhelm

                    if found_ids:
                        # Remove duplicates
                        unique_ids = list(dict.fromkeys(found_ids))
                        prompt += f"  - potential_ids: {unique_ids[:10]}\n"

            # Include content-extracted data
            content_data = extracted_data.get("extracted_content", {})
            if content_data:
                prompt += f"\nCONTENT ANALYSIS:\n"
                for data_type, values in content_data.items():
                    if values and data_type not in [
                        "structured_data"
                    ]:  # Skip complex nested data
                        if isinstance(values, list) and values:
                            prompt += f"  - {data_type}: {values[:5]}\n"
                        elif isinstance(values, dict) and values:
                            # Show first few key-value pairs from structured data
                            items = list(values.items())[:3]
                            prompt += f"  - {data_type}: {dict(items)}\n"

        prompt += f"\nCONTENT TO SUMMARIZE:\n{summary_text}"

        return prompt

    def _format_extracted_data_for_summary(self, extracted_data: Dict[str, Any]) -> str:
        """Format extracted data for inclusion in summary - generic for all data types."""
        if not extracted_data:
            return ""

        formatted_lines = ["\n\n[EXTRACTED DATA FOR FUTURE REFERENCE]"]

        # Format search tool data
        search_data = extracted_data.get("search_tool_data", {})
        for tool_name, tool_data in search_data.items():
            tool_extracted = tool_data.get("extracted_data", {})

            # Process all types of extracted data generically
            if tool_extracted:
                formatted_lines.append(f"\n{tool_name.upper()} FINDINGS:")

                # Extract and preserve IDs from raw results using comprehensive patterns
                result_str = str(tool_data.get("result", ""))
                if result_str:
                    all_ids = self._extract_all_ids_from_text(result_str)
                    if all_ids:
                        formatted_lines.append(f"  IDs found: {all_ids}")

                # Include all extracted data types
                for data_type, values in tool_extracted.items():
                    if values:
                        if isinstance(values, list) and values:
                            # Format list data clearly
                            display_values = values[:8]  # Show more for better context
                            if len(values) > 8:
                                display_values.append(f"... and {len(values) - 8} more")
                            formatted_lines.append(f"  {data_type}: {display_values}")
                        elif isinstance(values, dict) and values:
                            # Show key-value pairs for structured data
                            items = list(values.items())[:5]
                            formatted_lines.append(f"  {data_type}: {dict(items)}")
                        else:
                            # Single values
                            formatted_lines.append(
                                f"  {data_type}: {str(values)[:100]}"
                            )

        # Include content analysis data
        content_data = extracted_data.get("extracted_content", {})
        if content_data and any(content_data.values()):
            formatted_lines.append(f"\nCONTENT ANALYSIS:")
            for data_type, values in content_data.items():
                if values and data_type != "structured_data":
                    if isinstance(values, list) and values:
                        formatted_lines.append(f"  {data_type}: {values[:5]}")

        return "\n".join(formatted_lines) if len(formatted_lines) > 1 else ""

    def _extract_all_ids_from_text(self, text: str) -> List[str]:
        """Extract all potential IDs from text using comprehensive patterns."""
        import re

        all_ids = []

        # Comprehensive ID patterns for different systems
        id_patterns = [
            # Email/Message IDs (Gmail, Outlook, etc.)
            r"\b[a-f0-9]{16,24}\b",  # Gmail message IDs
            r"\b[A-Za-z0-9+/]{20,}={0,2}\b",  # Base64-like IDs
            # Database/API IDs
            r'\bid["\']?\s*:\s*["\']?([A-Za-z0-9_-]+)["\']?',  # JSON id fields
            r'\b(?:user_id|session_id|token|key|ref)\s*[=:]\s*["\']?([A-Za-z0-9_-]+)["\']?',
            # File/Path IDs
            r"\/([A-Za-z0-9_-]{10,})\/",  # IDs in file paths
            # Transaction/Confirmation codes
            r"\b[A-Z]{2,}[0-9]{6,}\b",  # Codes like ABC123456
            r"\b[0-9]{8,15}\b",  # Long numeric IDs
            # UUIDs and similar
            r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b",
            r"\b[A-Za-z0-9]{32}\b",  # 32-character IDs
        ]

        for pattern in id_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if isinstance(matches[0], tuple) if matches else False:
                # Handle grouped patterns
                matches = [
                    match[0] if isinstance(match, tuple) else match for match in matches
                ]
            all_ids.extend(matches)

        # Remove duplicates while preserving order, filter out common false positives
        unique_ids = []
        false_positives = {
            "function",
            "return",
            "string",
            "number",
            "boolean",
            "object",
            "array",
        }

        for id_val in all_ids:
            if (
                id_val not in unique_ids
                and len(id_val) >= 8  # Minimum reasonable ID length
                and id_val.lower() not in false_positives
                and not id_val.isdigit()
                or len(id_val) >= 10
            ):  # Avoid short numbers
                unique_ids.append(id_val)

        return unique_ids[:15]  # Limit to prevent overwhelming the summary
