import asyncio
from typing import Dict, Any
from agents.react_agent import ReactAgent, ReactAgentConfig
from reactive_agents.providers.external.client import MCPClient
import readline  # For better input handling

# Import the new type definitions
from common.types import (
    ConfirmationCallbackProtocol,
    ConfirmationConfig,
    ConfirmationResult,
)


async def example_confirmation_callback(
    action_description: str, details: Dict[str, Any]
) -> ConfirmationResult:
    """
    Example confirmation callback that allows the user to provide feedback.

    Args:
        action_description: Description of the action being confirmed
        details: Dictionary containing tool name and parameters

    Returns:
        Either a boolean (confirmed/rejected) or a tuple of (confirmed, feedback)
    """
    tool_name = details.get("tool", "unknown")
    params = details.get("params", {})

    print("\n" + "=" * 80)
    print(f"🔔 TOOL CONFIRMATION REQUIRED: {tool_name}")
    print(f"Description: {action_description}")
    print("=" * 80)

    while True:
        response = input("Confirm this action? [y]es/[n]o/[f]eedback: ").lower().strip()

        if response.startswith("y"):
            return True
        elif response.startswith("n"):
            return False
        elif response.startswith("f"):
            feedback = input("Enter your feedback for the agent: ")
            confirm_with_feedback = (
                input("Proceed with this action after sending feedback? [y/n]: ")
                .lower()
                .strip()
            )
            return (confirm_with_feedback.startswith("y"), feedback)
        else:
            print("Invalid response. Please enter 'y', 'n', or 'f'.")


async def main():
    # Create a custom confirmation configuration using the ConfirmationConfig type
    confirmation_config = ConfirmationConfig(
        {
            "always_confirm": [
                "search_web",
                "send_email",
            ],  # Tools that always require confirmation
            "never_confirm": [
                "final_answer",
                "read_file",
            ],  # Tools that never require confirmation
            "patterns": {  # Patterns to match in tool names or descriptions
                "write": "confirm",
                "delete": "confirm",
                "create": "confirm",
                "edit": "confirm",
                "update": "confirm",
                "search": "confirm",
            },
            "default_action": "proceed",  # Default for tools that don't match patterns: "proceed" or "confirm"
        }
    )

    # Test the configuration directly
    print(
        "Confirmation required for 'delete_file':",
        confirmation_config.requires_confirmation("delete_file"),
    )
    print(
        "Confirmation required for 'read_file':",
        confirmation_config.requires_confirmation("read_file"),
    )
    print(
        "Confirmation required for 'write_to_file':",
        confirmation_config.requires_confirmation("write_to_file"),
    )

    # Create the agent config with our confirmation callback and custom config
    agent_config = ReactAgentConfig(
        agent_name="ConfirmationExampleAgent",
        provider_model_name="ollama:qwen3:4b",  # Change to your preferred model
        instructions="Help the user with their tasks, but be careful with critical operations.",
        max_iterations=15,
        reflect_enabled=True,
        confirmation_callback=example_confirmation_callback,
        confirmation_config=confirmation_config,
        role="You are a helpful assistant that can help the user with their tasks.",
        mcp_client=await MCPClient(server_filter=["brave-search"]).initialize(),
        min_completion_score=0.9,
        log_level="DEBUG",
        initial_task="Search for information about Python's asyncio library and create a summary file.",
        tool_use_enabled=True,
        use_memory_enabled=True,
        collect_metrics_enabled=True,
        check_tool_feasibility=True,
        enable_caching=True,
        kwargs={},
    )

    # Create the agent
    agent = ReactAgent(config=agent_config)

    # Run the agent with an initial task
    try:
        task = "Search for information about Python's asyncio library and create a summary file."
        print(f"Starting agent with task: {task}")

        result = await agent.run(initial_task=task)

        print("\n" + "=" * 80)
        print("AGENT RESULT:")
        print(f"Status: {result['status']}")
        print(f"Result: {result['result']}")
        print("=" * 80)

    finally:
        # Clean up
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
