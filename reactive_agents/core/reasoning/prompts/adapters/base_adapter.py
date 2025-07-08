"""
Base Strategy Adapter

Provides common functionality for all strategy adapters including:
- Failure analysis and tracking
- Pattern recognition
- Learning from recoveries
- Cross-adapter improvements
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set, TYPE_CHECKING
import json
from datetime import datetime
import re

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class BaseStrategyAdapter:
    """Base class for all strategy adapters with shared analysis capabilities."""

    def __init__(self, context: "AgentContext"):
        """Initialize the base adapter with tracking capabilities."""
        self.context = context
        self.agent_logger = context.agent_logger

        # Initialize tracking storage
        self._tool_stats = (
            {}
        )  # {tool_name: {success: int, failure: int, error_types: {type: count}}}
        self._step_stats = {}  # {step_number: {success: bool, failure_type: str}}
        self._recovery_patterns = (
            {}
        )  # {pattern_key: [{failure: dict, recovery: dict, timestamp: str}]}
        self._failure_patterns = (
            {}
        )  # {pattern_key: {count: int, first_seen: str, examples: list}}

    def analyze_failure(
        self, step_info: Dict[str, Any], execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a step failure to determine root cause and suggest recovery.

        Args:
            step_info: The step that failed
            execution_result: The execution result

        Returns:
            Dict with failure analysis
        """
        # Extract key information
        tool_calls = execution_result.get("tool_calls", [])
        content = execution_result.get("content", "")
        error_messages = self._extract_error_messages(content, tool_calls)

        # Identify failure type
        failure_type = self._identify_failure_type(error_messages, tool_calls)

        # Find root cause
        root_cause = self._determine_root_cause(
            failure_type, error_messages, tool_calls, step_info
        )

        # Update failure patterns
        pattern_key = f"{failure_type}:{root_cause}"
        if pattern_key not in self._failure_patterns:
            self._failure_patterns[pattern_key] = {
                "count": 0,
                "first_seen": datetime.now().isoformat(),
                "examples": [],
            }

        self._failure_patterns[pattern_key]["count"] += 1
        self._failure_patterns[pattern_key]["examples"].append(
            {
                "step": step_info,
                "result": execution_result,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Look for previous successful recovery
        previous_recovery = self._find_recovery_pattern(failure_type, root_cause)

        return {
            "failure_type": failure_type,
            "root_cause": root_cause,
            "error_messages": error_messages,
            "previous_recovery": previous_recovery,
            "pattern_key": pattern_key,
        }

    def record_recovery(
        self, failure_analysis: Dict[str, Any], recovery_result: Dict[str, Any]
    ) -> None:
        """
        Record a successful recovery pattern.

        Args:
            failure_analysis: The original failure analysis
            recovery_result: The successful recovery result
        """
        pattern_key = failure_analysis["pattern_key"]

        if pattern_key not in self._recovery_patterns:
            self._recovery_patterns[pattern_key] = []

        self._recovery_patterns[pattern_key].append(
            {
                "failure": failure_analysis,
                "recovery": recovery_result,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def update_tool_stats(
        self, tool_name: str, success: bool, error_type: Optional[str] = None
    ) -> None:
        """Update tool usage statistics."""
        if tool_name not in self._tool_stats:
            self._tool_stats[tool_name] = {
                "success": 0,
                "failure": 0,
                "error_types": {},
            }

        if success:
            self._tool_stats[tool_name]["success"] += 1
        else:
            self._tool_stats[tool_name]["failure"] += 1
            if error_type:
                self._tool_stats[tool_name]["error_types"][error_type] = (
                    self._tool_stats[tool_name]["error_types"].get(error_type, 0) + 1
                )

    def update_step_stats(
        self, step_number: int, success: bool, failure_type: Optional[str] = None
    ) -> None:
        """Update step execution statistics."""
        self._step_stats[step_number] = {
            "success": success,
            "failure_type": failure_type if not success else None,
        }

    def add_recovery_pattern(self, pattern: str, success: bool) -> None:
        """Add a recovery pattern attempt."""
        if pattern not in self._recovery_patterns:
            self._recovery_patterns[pattern] = []

        self._recovery_patterns[pattern].append(
            {
                "pattern": pattern,
                "success": success,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_failure_insights(self) -> Dict[str, Any]:
        """
        Get insights about failures and tool reliability.

        Returns:
            Dict with failure insights and tool reliability metrics
        """
        insights = {
            "tool_reliability": {"problematic_tools": [], "reliable_tools": []},
            "common_failures": [],
            "recovery_patterns": [],
        }

        # Analyze tool stats
        for tool_name, stats in self._tool_stats.items():
            total_calls = stats["success"] + stats["failure"]
            if total_calls > 0:
                success_rate = stats["success"] / total_calls
                if success_rate < 0.7:  # Less than 70% success rate
                    insights["tool_reliability"]["problematic_tools"].append(
                        {
                            "name": tool_name,
                            "success_rate": success_rate,
                            "total_calls": total_calls,
                            "common_errors": stats.get("error_types", {}),
                        }
                    )
                elif success_rate > 0.9:  # More than 90% success rate
                    insights["tool_reliability"]["reliable_tools"].append(
                        {
                            "name": tool_name,
                            "success_rate": success_rate,
                            "total_calls": total_calls,
                        }
                    )

        # Analyze step failures
        failure_counts = {}
        for step_num, stats in self._step_stats.items():
            if not stats["success"]:
                failure_type = stats.get("failure_type", "unknown")
                failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1

        # Sort failures by frequency
        sorted_failures = sorted(
            failure_counts.items(), key=lambda x: x[1], reverse=True
        )

        # Add common failures
        for failure_type, count in sorted_failures:
            insights["common_failures"].append({"type": failure_type, "count": count})

        # Add recovery patterns that worked
        for pattern_key, recoveries in self._recovery_patterns.items():
            if recoveries:
                successful_recoveries = [
                    r for r in recoveries if r.get("success", False)
                ]
                total_recoveries = len(recoveries)
                if successful_recoveries:
                    success_rate = len(successful_recoveries) / total_recoveries
                    insights["recovery_patterns"].append(
                        {
                            "pattern": pattern_key,
                            "success_rate": success_rate,
                            "total_attempts": total_recoveries,
                        }
                    )

        return insights

    def _extract_error_messages(
        self, content: str, tool_calls: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract error messages from execution result.

        Args:
            content: Execution content
            tool_calls: List of tool calls

        Returns:
            List of error messages
        """
        error_messages = []

        # Look for error patterns in content
        error_patterns = [
            r"error:?\s*(.+?)(?:\n|$)",
            r"exception:?\s*(.+?)(?:\n|$)",
            r"failed:?\s*(.+?)(?:\n|$)",
            r"invalid:?\s*(.+?)(?:\n|$)",
        ]

        for pattern in error_patterns:
            matches = re.finditer(pattern, content.lower())
            error_messages.extend(match.group(1).strip() for match in matches)

        # Extract errors from tool calls
        for call in tool_calls:
            if isinstance(call, dict):
                result = call.get("result", "")
                if isinstance(result, str):
                    for pattern in error_patterns:
                        matches = re.finditer(pattern, result.lower())
                        error_messages.extend(
                            match.group(1).strip() for match in matches
                        )

        return error_messages

    def _identify_failure_type(
        self, error_messages: List[str], tool_calls: List[Dict[str, Any]]
    ) -> str:
        """
        Identify the type of failure.

        Args:
            error_messages: List of error messages
            tool_calls: List of tool calls

        Returns:
            Failure type string
        """
        # Check for authentication failures
        auth_patterns = {
            "unauthorized",
            "unauthenticated",
            "auth",
            "login",
            "credential",
        }
        if any(any(p in msg for p in auth_patterns) for msg in error_messages):
            return "auth_failure"

        # Check for permission failures
        perm_patterns = {"permission", "access denied", "forbidden"}
        if any(any(p in msg for p in perm_patterns) for msg in error_messages):
            return "permission_failure"

        # Check for data/validation failures
        data_patterns = {"invalid", "missing", "required", "not found"}
        if any(any(p in msg for p in data_patterns) for msg in error_messages):
            return "data_validation_failure"

        # Check for tool execution failures
        if any(call.get("error") for call in tool_calls if isinstance(call, dict)):
            return "tool_execution_failure"

        # Default to generic failure
        return "unknown_failure"

    def _determine_root_cause(
        self,
        failure_type: str,
        error_messages: List[str],
        tool_calls: List[Dict[str, Any]],
        step_info: Dict[str, Any],
    ) -> str:
        """
        Determine the root cause of a failure.

        Args:
            failure_type: Type of failure
            error_messages: List of error messages
            tool_calls: List of tool calls
            step_info: Information about the failed step

        Returns:
            Root cause string
        """
        if failure_type == "auth_failure":
            return "missing_or_invalid_credentials"

        if failure_type == "permission_failure":
            return "insufficient_permissions"

        if failure_type == "data_validation_failure":
            # Look for specific data issues
            if any("required" in msg for msg in error_messages):
                return "missing_required_data"
            if any("invalid" in msg for msg in error_messages):
                return "invalid_data_format"
            return "data_validation_error"

        if failure_type == "tool_execution_failure":
            # Check tool-specific errors
            for call in tool_calls:
                if isinstance(call, dict) and call.get("error"):
                    error = call["error"]
                    if "timeout" in str(error).lower():
                        return "tool_timeout"
                    if "rate limit" in str(error).lower():
                        return "rate_limit_exceeded"
            return "tool_execution_error"

        return "unknown_cause"

    def _find_recovery_pattern(
        self, failure_type: str, root_cause: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find a matching recovery pattern.

        Args:
            failure_type: Type of failure
            root_cause: Root cause of failure

        Returns:
            Recovery pattern if found
        """
        pattern_key = f"{failure_type}:{root_cause}"

        if pattern_key in self._recovery_patterns:
            # Return most recent successful recovery
            recoveries = self._recovery_patterns[pattern_key]
            if recoveries:
                return recoveries[-1]

        return None
