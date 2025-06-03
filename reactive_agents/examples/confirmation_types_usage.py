"""
Example demonstrating how to use the confirmation types in a minimal way.
This shows the type interface without requiring a full agent setup.
"""

import asyncio
from typing import Dict, Any
from common.types import (
    ConfirmationConfig,
    ConfirmationResult,
)


# Example 1: A synchronous confirmation callback
def simple_confirmation_callback(
    action_description: str, details: Dict[str, Any]
) -> ConfirmationResult:
    """A simple synchronous confirmation callback that always returns True with feedback."""
    print(f"Would confirm: {action_description}")
    return (True, "This is feedback for the action")


# Example 2: An asynchronous confirmation callback
async def async_confirmation_callback(
    action_description: str, details: Dict[str, Any]
) -> ConfirmationResult:
    """An asynchronous confirmation callback that simulates user input."""
    tool_name = details.get("tool", "unknown")
    print(f"Confirming action for tool: {tool_name}")

    # Simulate some async work
    await asyncio.sleep(0.5)

    # Determine whether to confirm based on tool name
    if "delete" in tool_name:
        return (False, "Don't use delete operations, try update instead")
    elif "write" in tool_name:
        return (True, "Make sure to include a header in the file")
    else:
        return True


# Example 3: Using the ConfirmationConfig class
def demonstrate_confirmation_config():
    # Create a default configuration
    default_config = ConfirmationConfig.create_default()
    print("\n=== Default Configuration ===")
    print(f"Always confirm: {default_config.get('always_confirm', [])}")
    print(f"Never confirm: {default_config.get('never_confirm', [])}")
    print(f"Pattern count: {len(default_config.get('patterns', {}))}")

    # Create a custom configuration
    custom_config = ConfirmationConfig(
        {
            "always_confirm": ["email_send", "file_delete"],
            "never_confirm": ["read_file", "list_dir", "final_answer"],
            "patterns": {
                "write": "confirm",
                "delete": "confirm",
                "update": "proceed",
            },
            "default_action": "proceed",
        }
    )

    print("\n=== Custom Configuration ===")

    # Test some tools with the configuration
    tools_to_test = [
        "read_file",
        "file_delete",
        "email_send",
        "write_to_file",
        "update_record",
        "unknown_tool",
    ]

    for tool in tools_to_test:
        requires = custom_config.requires_confirmation(tool)
        print(f"Tool '{tool}' requires confirmation: {requires}")


async def main():
    # Demonstrate the confirmation callbacks
    print("=== Confirmation Callbacks ===")

    # Test the synchronous callback
    sync_result = simple_confirmation_callback(
        "Create a new file 'example.txt'",
        {"tool": "create_file", "params": {"path": "example.txt"}},
    )
    print(f"Sync callback result: {sync_result}")

    # Test the asynchronous callback
    async_result_1 = await async_confirmation_callback(
        "Delete file 'example.txt'",
        {"tool": "delete_file", "params": {"path": "example.txt"}},
    )
    print(f"Async callback result (delete): {async_result_1}")

    async_result_2 = await async_confirmation_callback(
        "Write to file 'example.txt'",
        {"tool": "write_file", "params": {"path": "example.txt", "content": "Hello"}},
    )
    print(f"Async callback result (write): {async_result_2}")

    # Demonstrate the confirmation config
    demonstrate_confirmation_config()


if __name__ == "__main__":
    asyncio.run(main())
