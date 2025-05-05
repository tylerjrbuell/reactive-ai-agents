from __future__ import annotations
import json
from datetime import datetime
from enum import Enum
from typing import (
    List,
    Any,
    Optional,
    Dict,
    Union,
    Tuple,
    Protocol,
    runtime_checkable,
    Awaitable,
)

from pydantic import BaseModel, Field


class TaskStatus(Enum):
    """Standardized task status values"""

    INITIALIZED = "initialized"
    WAITING_DEPENDENCIES = "waiting_for_dependencies"
    RUNNING = "running"
    MISSING_TOOLS = "missing_tools"
    COMPLETE = "complete"
    RESCOPED_COMPLETE = "rescoped_complete"
    MAX_ITERATIONS = "max_iterations_reached"
    ERROR = "error"
    CANCELLED = "cancelled"

    def __str__(self):
        return self.value


class AgentMemory(BaseModel):
    """Model for agent memory storage"""

    agent_name: str
    session_history: List[Dict[str, Any]] = []
    tool_preferences: Dict[str, Any] = {}
    user_preferences: Dict[str, Any] = {}
    reflections: List[Dict[str, Any]] = []
    last_updated: datetime = Field(default_factory=datetime.now)


# ---- Confirmation System Types ----

# Type alias for the confirmation callback return value
ConfirmationResult = Union[bool, Tuple[bool, Optional[str]]]

# Type alias for the asynchronous confirmation callback return value
ConfirmationResultAwaitable = Awaitable[ConfirmationResult]


@runtime_checkable
class ConfirmationCallbackProtocol(Protocol):
    """Protocol defining the signature for confirmation callbacks."""

    def __call__(
        self, action_description: str, details: Dict[str, Any]
    ) -> Union[ConfirmationResult, ConfirmationResultAwaitable]:
        """
        A callback to prompt for confirmation before executing a tool.

        Args:
            action_description: Human-readable description of the action being confirmed
            details: Dictionary containing metadata about the tool operation,
                     including at minimum 'tool' (name) and 'params' (parameters)

        Returns:
            Either:
            - bool: True to allow the action, False to cancel
            - Tuple[bool, str]: (allow_action, feedback) where feedback is optional user guidance
            - An awaitable that resolves to one of the above
        """
        ...


# Type definition for the confirmation configuration
class ConfirmationConfig(Dict[str, Any]):
    """
    Type for tool confirmation configuration settings.

    This dictionary can contain the following keys:

    - always_confirm: List[str] - Tool names that always require confirmation
    - never_confirm: List[str] - Tool names that never require confirmation
    - patterns: Dict[str, str] - Substring patterns mapped to actions ('confirm'/'proceed')
    - default_action: str - Default action for tools not matching other rules ('confirm'/'proceed')

    Example:
    {
        "always_confirm": ["delete_file", "send_email"],
        "never_confirm": ["final_answer", "read_file"],
        "patterns": {
            "write": "confirm",
            "delete": "confirm",
            "create": "proceed",
        },
        "default_action": "proceed"
    }
    """

    @classmethod
    def create_default(cls) -> "ConfirmationConfig":
        """
        Creates a default confirmation configuration.

        Returns:
            A ConfirmationConfig with sensible defaults
        """
        return cls(
            {
                "always_confirm": [],
                "never_confirm": ["final_answer"],
                "patterns": {
                    "write": "confirm",
                    "delete": "confirm",
                    "remove": "confirm",
                    "update": "confirm",
                    "create": "confirm",
                    "send": "confirm",
                    "email": "confirm",
                    "subscribe": "confirm",
                    "unsubscribe": "confirm",
                    "payment": "confirm",
                    "post": "confirm",
                    "put": "confirm",
                },
                "default_action": "proceed",
            }
        )

    def requires_confirmation(self, tool_name: str, description: str = "") -> bool:
        """
        Determine if a tool requires confirmation based on this configuration.

        Args:
            tool_name: The name of the tool to check
            description: Optional tool description to check against patterns

        Returns:
            True if the tool requires confirmation, False otherwise
        """
        # Check if tool is in the always_confirm list
        if tool_name in self.get("always_confirm", []):
            return True

        # Check if tool is in the never_confirm list
        if tool_name in self.get("never_confirm", []):
            return False

        # Check patterns
        patterns = self.get("patterns", {})
        for pattern, action in patterns.items():
            if pattern in tool_name.lower() or (
                description and pattern in description.lower()
            ):
                return action.lower() == "confirm"

        # Use default action
        return self.get("default_action", "proceed").lower() == "confirm"
