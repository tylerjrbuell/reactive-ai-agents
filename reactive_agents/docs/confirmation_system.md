# Tool Confirmation System

The ReactAgent framework includes a robust tool confirmation system that allows users to control which tools require confirmation before execution and provide feedback to the agent during confirmation prompts.

## Features

- **Configurable Confirmation Rules**: Configure which tools require confirmation using explicit lists or pattern matching
- **User Feedback**: Provide feedback to the agent during the confirmation process to guide its behavior
- **Type-Safe Interface**: Strong typing for the confirmation callback and configuration
- **Centralized Types**: All types are defined in a single location for easy imports

## Type Definitions

The system is built around these core types (found in `common/types.py`):

- **ConfirmationResult**: The return type of confirmation callbacks (`bool | Tuple[bool, Optional[str]]`)
- **ConfirmationCallbackProtocol**: Protocol defining the signature of confirmation callbacks
- **ConfirmationConfig**: Type-safe configuration object that determines which tools require confirmation

## Configuration Options

The confirmation system can be configured through the `confirmation_config` option in the `ReactAgentConfig`:

```python
# Import the types from the centralized location
from common.types import ConfirmationConfig

# Create configuration
confirmation_config = {
    # Tools that always require confirmation regardless of name/description
    "always_confirm": ["search_web", "send_email"],

    # Tools that never require confirmation regardless of name/description
    "never_confirm": ["final_answer", "read_file"],

    # Patterns to match in tool names or descriptions
    # Values can be either "confirm" or "proceed"
    "patterns": {
        "write": "confirm",
        "delete": "confirm",
        "update": "confirm",
        "create": "confirm",
    },

    # Default action if no other rules match: "confirm" or "proceed"
    "default_action": "proceed"
}
```

## Creating a Confirmation Callback

To create a confirmation callback, implement a function with this signature:

```python
from typing import Dict, Any, Tuple, Optional, Union
from common.types import ConfirmationResult

async def my_confirmation_callback(
    action_description: str,
    details: Dict[str, Any]
) -> ConfirmationResult:
    """
    Args:
        action_description: Description of the action being confirmed
        details: Dict containing at least "tool" (name) and "params" (parameters)

    Returns:
        Either:
        - bool: True to proceed, False to cancel
        - Tuple[bool, str]: (proceed, feedback) where feedback is a message for the agent
    """
    # Example implementation:
    tool_name = details.get("tool", "unknown")
    user_input = input(f"Confirm {tool_name}? (y/n/f for feedback): ").lower()

    if user_input.startswith('f'):
        feedback = input("Your feedback: ")
        proceed = input("Proceed? (y/n): ").lower().startswith('y')
        return (proceed, feedback)
    else:
        return user_input.startswith('y')
```

## Feedback Mechanism

When a callback returns a tuple `(proceed, feedback)`, the feedback string will be injected into the agent's context as a user message. This allows users to provide guidance to the agent before it continues execution.

For example, if the feedback is "Use a different tool", the agent will see this as a user message and can adjust its approach accordingly in the next iteration.

## Usage Example

Here's a basic usage example:

```python
from agents.react_agent import ReactAgent, ReactAgentConfig
from common.types import ConfirmationResult

async def confirmation_callback(description, details) -> ConfirmationResult:
    print(f"Tool: {details['tool']}")
    print(f"Description: {description}")
    response = input("Confirm? (y/n/f): ").lower()

    if response.startswith('f'):
        feedback = input("Feedback: ")
        return (input("Proceed? (y/n): ").lower().startswith('y'), feedback)
    return response.startswith('y')

# Create agent config with confirmation setup
agent_config = ReactAgentConfig(
    agent_name="MyAgent",
    provider_model_name="ollama:qwen2:7b",
    confirmation_callback=confirmation_callback,
    confirmation_config={
        "always_confirm": ["delete_file", "send_email"],
        "never_confirm": ["final_answer", "read_file"],
        "patterns": {
            "write": "confirm",
            "delete": "confirm",
            "update": "proceed",
        },
        "default_action": "proceed"
    },
    # ... other config options ...
)

# Create and run the agent
agent = ReactAgent(config=agent_config)
result = await agent.run("Your task prompt here")
```

## Additional Examples

For more examples, see:

- `examples/tool_confirmation_with_feedback.py` - Complete example with an agent
- `examples/confirmation_types_usage.py` - Standalone examples of the type interface
