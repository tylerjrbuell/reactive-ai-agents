"""Tool result validation and data extraction."""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from reactive_agents.core.tools.data_extractor import DataExtractor, SearchDataManager
from reactive_agents.utils.logging import Logger

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class ToolValidator:
    """Validates tool results and provides usage suggestions."""

    def __init__(self, context: "AgentContext"):
        self.context = context
        self.data_extractor = DataExtractor()
        self.search_data_manager = SearchDataManager()

    def validate_tool_result_usage(
        self, tool_name: str, params: Dict[str, Any], result: Any
    ) -> Dict[str, Any]:
        """Validate that tool results are being used correctly and not ignored."""
        validation = {
            "valid": True,
            "warnings": [],
            "suggestions": [],
            "extracted_data": {},
        }

        try:
            # Get extracted data from search results using the new manager
            extracted_data = self.search_data_manager.get_extracted_data(tool_name)
            if extracted_data:
                validation["extracted_data"] = extracted_data

                # Check for various data types
                for data_type, values in extracted_data.items():
                    if values:
                        # Ensure values is a proper list and handle any unhashable types
                        try:
                            if isinstance(values, list):
                                # Filter out any unhashable types and take first 3 items
                                safe_values = []
                                for value in values:
                                    try:
                                        # Test if the value is hashable by trying to use it as a dict key
                                        _ = {value: None}
                                        safe_values.append(str(value))
                                        if len(safe_values) >= 3:
                                            break
                                    except (TypeError, ValueError):
                                        # Skip unhashable values
                                        continue

                                if safe_values:
                                    validation["suggestions"].append(
                                        f"Found {data_type}: {safe_values}..."
                                    )
                            else:
                                # Handle non-list values
                                validation["suggestions"].append(
                                    f"Found {data_type}: {str(values)}"
                                )
                        except Exception as e:
                            # If we can't process the values, just log it and continue
                            if (
                                hasattr(self.context, "tool_logger")
                                and self.context.tool_logger
                            ):
                                self.context.tool_logger.debug(
                                    f"Could not process {data_type} values: {e}"
                                )

            # Tool-specific validation
            self._validate_search_tools(tool_name, result, validation)
            self._validate_file_tools(tool_name, params, result, validation)
            self._validate_general_tools(tool_name, result, validation)

            if validation["warnings"]:
                validation["valid"] = False

        except Exception as e:
            if hasattr(self.context, "tool_logger") and self.context.tool_logger:
                self.context.tool_logger.debug(f"Error in result validation: {e}")

        return validation

    def _validate_search_tools(
        self, tool_name: str, result: Any, validation: Dict[str, Any]
    ) -> None:
        """Validate search tool results."""
        if tool_name not in ["brave_web_search", "brave_local_search"]:
            return

        result_str = str(result)

        # Check if search returned meaningful results
        if any(
            phrase in result_str.lower()
            for phrase in ["not available", "no results", "not found", "error"]
        ):
            validation["warnings"].append("Search returned no meaningful results")

        # Check for placeholder indicators
        placeholder_indicators = [
            "placeholder",
            "example",
            "sample",
            "test data",
            "dummy",
        ]
        if any(indicator in result_str.lower() for indicator in placeholder_indicators):
            validation["warnings"].append(
                "Search results contain placeholder indicators"
            )

    def _validate_file_tools(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Any,
        validation: Dict[str, Any],
    ) -> None:
        """Validate file operation tools."""
        if tool_name == "write_file":
            content = params.get("content", "")

            # Check for placeholder content
            placeholder_indicators = [
                "placeholder",
                "example",
                "sample",
                "test",
                "dummy",
                "TODO",
                "FIXME",
            ]
            if any(
                indicator in content.lower() for indicator in placeholder_indicators
            ):
                validation["warnings"].append(
                    "File content appears to contain placeholder data"
                )

            # Check if content matches extracted data from searches
            all_search_data = self.search_data_manager.get_all_search_data()
            for search_tool, search_info in all_search_data.items():
                extracted_data = search_info.get("extracted_data", {})

                # Use the data extractor's validation method
                data_validation = self.data_extractor.validate_data_usage(
                    content, self.data_extractor.extract_all("")
                )
                if not data_validation["valid"]:
                    validation["warnings"].extend(data_validation["warnings"])
                    validation["suggestions"].extend(data_validation["suggestions"])

        elif tool_name in ["read_file", "read_multiple_files"]:
            # Validate file reading operations
            result_str = str(result)
            if "error" in result_str.lower() or "not found" in result_str.lower():
                validation["warnings"].append("File read operation may have failed")

        elif tool_name in ["create_directory", "move_file", "edit_file"]:
            # Validate file operations
            result_str = str(result)
            if "error" in result_str.lower() or "failed" in result_str.lower():
                validation["warnings"].append("File operation may have failed")
            elif "success" in result_str.lower():
                validation["suggestions"].append(
                    "File operation completed successfully"
                )

    def _validate_general_tools(
        self, tool_name: str, result: Any, validation: Dict[str, Any]
    ) -> None:
        """Validate general tool execution patterns."""
        result_str = str(result)

        # Check for error indicators
        error_indicators = [
            "error",
            "failed",
            "exception",
            "timeout",
            "not found",
            "unauthorized",
        ]
        if any(indicator in result_str.lower() for indicator in error_indicators):
            validation["warnings"].append(
                "Tool execution may have encountered an error"
            )

        # Check for success indicators
        success_indicators = ["success", "completed", "done", "ok", "successful"]
        if any(indicator in result_str.lower() for indicator in success_indicators):
            validation["suggestions"].append("Tool execution completed successfully")

        # Check for empty or minimal results
        if len(result_str.strip()) < 10 and tool_name not in ["final_answer"]:
            validation["warnings"].append("Tool returned minimal or empty result")

    def store_search_data(
        self, tool_name: str, params: Dict[str, Any], result: Any
    ) -> None:
        """Store structured data for search tools."""
        if "search" in tool_name and result:
            self.search_data_manager.store_search_data(
                tool_name, params, result, self.data_extractor
            )

    def get_extracted_data(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get extracted data, optionally filtered by tool name."""
        if tool_name:
            return self.search_data_manager.get_extracted_data(tool_name) or {}
        return self.search_data_manager.get_all_search_data() or {}

    def clear_extracted_data(self) -> None:
        """Clear all extracted search data."""
        self.search_data_manager = SearchDataManager()  # Reset the manager
