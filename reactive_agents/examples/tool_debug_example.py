"""
Tool Registration Diagnostics Example

This example demonstrates how to use the diagnostic features of ReactiveAgentBuilder
to debug tool registration issues. It shows how to:

1. Use the debug_tools method to check tool configuration before building
2. Use the diagnose_agent_tools method to check tool registration after building
3. Debug hybrid tool setups with both MCP and custom tools
"""

import json
import asyncio
import os

from agents import ReactiveAgentBuilder
from tools.decorators import tool


# Define some example custom tools
@tool(description="Get the square of a number")
async def square(number: int) -> int:
    """Calculate the square of a number."""
    return number * number


@tool(description="Get the cube of a number")
async def cube(number: int) -> int:
    """Calculate the cube of a number."""
    return number * number * number


async def main():
    print("=== Tool Registration Diagnostics Example ===\n")

    # Example 1: Debug tools before building
    print("Example 1: Debug tools before building")
    print("--------------------------------------")
    builder = (
        ReactiveAgentBuilder()
        .with_name("Debug Example Agent")
        .with_model("ollama:cogito:14b")
        .with_mcp_tools(["time", "brave-search"])
        .with_custom_tools([square, cube])
    )

    # Get diagnostic info
    diagnostics = builder.debug_tools()
    print(f"Total tools: {diagnostics['total_tools']}")
    print(f"MCP tools: {diagnostics['mcp_tools']}")
    print(f"Custom tools: {diagnostics['custom_tools']}")
    print("\nCustom tool details:")
    for tool in diagnostics["custom_tool_details"]:
        print(
            f"  - {tool['name']} (type: {tool['type']}, has_name: {tool['has_name_attr']})"
        )

    print("\n")

    # Example 2: Diagnose tool registration after building
    print("Example 2: Diagnose after building")
    print("--------------------------------")
    try:
        agent = await builder.build()
        print(f"Agent built successfully: {agent.context.agent_name}")

        # Diagnose the agent's tools
        diagnosis = await ReactiveAgentBuilder.diagnose_agent_tools(agent)
        print(f"Context tools: {diagnosis['context_tools']}")
        print(f"Manager tools: {diagnosis['manager_tools']}")

        if diagnosis["has_tool_mismatch"]:
            print("\n⚠️ WARNING: Tool registration mismatch detected!")
            print(f"Missing in context: {diagnosis['missing_in_context']}")
            print(f"Missing in manager: {diagnosis['missing_in_manager']}")
        else:
            print("\n✅ All tools correctly registered in both locations")

        # Close the agent when done
        await agent.close()

    except Exception as e:
        print(f"Error building or diagnosing agent: {e}")

    print("\n")

    # Example 3: Debugging a tool registration issue (simulated)
    print("Example 3: Debug a tool registration issue (simulated)")
    print("--------------------------------------------------")

    print(
        """
This example simulates what happens if tools are added directly to an agent's context
without using the ReactiveAgentBuilder's proper tool registration methods.
    """
    )

    # Instead of using proper methods, print what would go wrong
    print(
        """
Common tool registration issues:

1. Adding tools directly to agent.context.tools without updating tool_manager
   - Tools will appear in context but not be available to the agent
   - The tool_manager doesn't know about these tools and can't call them

2. Adding tools to tool_manager without updating context.tools
   - Tools will be callable but won't appear in some metadata
   - Can cause inconsistent behavior in tracking and reflection

3. Not regenerating tool signatures after adding tools
   - The agent won't have proper schema information about the tools
   - May lead to incorrect tool parameter handling

Using ReactiveAgentBuilder.add_custom_tools_to_agent() or the builder's unified
registration methods prevents all these issues.
    """
    )


if __name__ == "__main__":
    asyncio.run(main())
