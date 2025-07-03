from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Union,
    Awaitable,
    Protocol,
    runtime_checkable,
)

# Type alias for the confirmation callback return value
ConfirmationResult = Union[bool, Tuple[bool, Optional[str]]]

# Type alias for the asynchronous confirmation callback return value
ConfirmationResultAwaitable = Awaitable[ConfirmationResult]


@runtime_checkable
class ConfirmationCallbackProtocol(Protocol):
    """Protocol defining the signature for confirmation callbacks."""

    def __call__(
        self, action_description: str, details: Dict[str, Any]
    ) -> Union[ConfirmationResult, ConfirmationResultAwaitable]: ...


class ConfirmationConfig(Dict[str, Any]):
    """
    Type for tool confirmation configuration settings.
    See original docstring for details.
    """

    @classmethod
    def create_default(cls) -> "ConfirmationConfig":
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
        if tool_name in self.get("always_confirm", []):
            return True
        if tool_name in self.get("never_confirm", []):
            return False
        patterns = self.get("patterns", {})
        for pattern, action in patterns.items():
            if pattern in tool_name.lower() or (
                description and pattern in description.lower()
            ):
                return action.lower() == "confirm"
        return self.get("default_action", "proceed").lower() == "confirm"
