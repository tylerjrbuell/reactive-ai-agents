"""Tool confirmation and user interaction system."""

import asyncio
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from reactive_agents.core.types.confirmation_types import ConfirmationCallbackProtocol
from reactive_agents.utils.logging import Logger

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class ToolConfirmation:
    """Handles tool confirmation logic and user interaction."""

    def __init__(
        self, 
        context: "AgentContext",
        confirmation_callback: Optional[ConfirmationCallbackProtocol] = None,
        confirmation_config: Optional[Dict[str, Any]] = None
    ):
        self.context = context
        self.confirmation_callback = confirmation_callback
        self.confirmation_config = confirmation_config or self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default confirmation configuration."""
        return {
            "always_confirm": [],  # List of tool names that always require confirmation
            "never_confirm": [
                "final_answer"
            ],  # Tools that never require confirmation
            "patterns": {  # Patterns to match in tool names or descriptions
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
            "default_action": "proceed",  # Default action if no pattern matches: "proceed" or "confirm"
        }

    def tool_requires_confirmation(self, tool_name: str, description: str = "") -> bool:
        """Determine if a tool requires confirmation based on configuration."""
        config = self.confirmation_config

        # Check if tool is in the always_confirm list
        if tool_name in config.get("always_confirm", []):
            return True

        # Check if tool is in the never_confirm list
        if tool_name in config.get("never_confirm", []):
            return False

        # Check patterns
        patterns = config.get("patterns", {})
        for pattern, action in patterns.items():
            if pattern in tool_name.lower() or (
                description and pattern in description.lower()
            ):
                return action.lower() == "confirm"

        # Use default action
        return config.get("default_action", "proceed").lower() == "confirm"

    async def request_confirmation(
        self, 
        tool_name: str, 
        params: Dict[str, Any],
        description: str = ""
    ) -> Tuple[bool, Optional[str]]:
        """Request confirmation for a tool execution."""
        if not self.confirmation_callback:
            return True, None  # Default to allow if no callback

        action_description = f"Use tool '{tool_name}' with parameters: {params}"
        details = {"tool": tool_name, "params": params, "description": description}

        try:
            result = None
            if asyncio.iscoroutinefunction(self.confirmation_callback):
                result = await self.confirmation_callback(action_description, details)
            else:
                # For synchronous callbacks
                result = self.confirmation_callback(action_description, details)

            # Check if result is awaitable and await it if needed
            if inspect.isawaitable(result):
                result = await result

            # Handle different return types from the callback
            if isinstance(result, tuple) and len(result) == 2:
                confirmed, feedback = result
                return bool(confirmed), feedback
            else:
                return bool(result), None

        except Exception as e:
            # Log error and default to deny for safety
            if hasattr(self.context, "tool_logger") and self.context.tool_logger:
                self.context.tool_logger.error(
                    f"Error during confirmation callback for {tool_name}: {e}"
                )
            return False, f"Confirmation failed due to error: {e}"

    def inject_user_feedback(
        self, 
        tool_name: str, 
        params: Dict[str, Any], 
        feedback: str
    ) -> None:
        """Inject user feedback into the agent's context as a message."""
        if hasattr(self.context, "tool_logger") and self.context.tool_logger:
            self.context.tool_logger.info(
                f"Injecting user feedback for tool '{tool_name}': {feedback}"
            )

        # Create a user message with the feedback
        feedback_message = {
            "role": "user",
            "content": f"Feedback for your '{tool_name}' tool call: {feedback}\nPlease adjust your approach based on this feedback.",
        }

        # Add to the session messages
        self.context.session.messages.append(feedback_message)

        # Also add to reasoning log for clarity
        self.context.session.reasoning_log.append(
            f"User feedback for tool '{tool_name}': {feedback}"
        )

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update confirmation configuration."""
        self.confirmation_config.update(new_config)

    def add_always_confirm(self, tool_name: str) -> None:
        """Add a tool to always confirm list."""
        self.confirmation_config.setdefault("always_confirm", []).append(tool_name)

    def add_never_confirm(self, tool_name: str) -> None:
        """Add a tool to never confirm list."""
        self.confirmation_config.setdefault("never_confirm", []).append(tool_name)

    def add_pattern(self, pattern: str, action: str) -> None:
        """Add a confirmation pattern."""
        self.confirmation_config.setdefault("patterns", {})[pattern] = action

    def remove_always_confirm(self, tool_name: str) -> None:
        """Remove a tool from always confirm list."""
        always_confirm = self.confirmation_config.get("always_confirm", [])
        if tool_name in always_confirm:
            always_confirm.remove(tool_name)

    def remove_never_confirm(self, tool_name: str) -> None:
        """Remove a tool from never confirm list."""
        never_confirm = self.confirmation_config.get("never_confirm", [])
        if tool_name in never_confirm:
            never_confirm.remove(tool_name)

    def remove_pattern(self, pattern: str) -> None:
        """Remove a confirmation pattern."""
        patterns = self.confirmation_config.get("patterns", {})
        patterns.pop(pattern, None)

    def get_config(self) -> Dict[str, Any]:
        """Get current confirmation configuration."""
        return self.confirmation_config.copy()