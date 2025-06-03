import json
import asyncio
import dotenv
import warnings
import tracemalloc
import traceback
from typing import Any, Dict, Optional, Callable, Awaitable, List, Tuple
from pydantic import PydanticDeprecatedSince211

from agents import ReactAgent, ReactAgentBuilder
from agents.builders import ConfirmationConfig, LogLevel
from agents.react_agent import ReactAgentConfig
from tools.decorators import tool

warnings.simplefilter("ignore", ResourceWarning)
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince211)
tracemalloc.start()
dotenv.load_dotenv()


# Example custom tool using the @tool decorator
@tool(description="Get the current weather for a specified location")
async def custom_weather_tool(location: str) -> str:
    """
    Get the weather for a given location (simulated).

    Args:
        location: The location to get the weather for. Supported locations are:
                 New York, London, Tokyo, and Sydney.

    Returns:
        A string with the simulated weather information
    """
    # This is a simulated tool - in a real app, this would call a weather API
    weather_data = {
        "New York": {"temp": "72°F", "condition": "Sunny"},
        "London": {"temp": "18°C", "condition": "Rainy"},
        "Tokyo": {"temp": "25°C", "condition": "Cloudy"},
        "Sydney": {"temp": "22°C", "condition": "Partly Cloudy"},
    }

    if location in weather_data:
        data = weather_data[location]
        return f"Weather in {location}: {data['temp']}, {data['condition']}"
    else:
        return f"Weather data for {location} not available. Try New York, London, Tokyo, or Sydney."


@tool(description="Get the current price of a cryptocurrency")
async def crypto_price_simulator(coin: str) -> str:
    """
    Get the current price of a cryptocurrency (simulated).

    Args:
        coin: The name of the cryptocurrency. Supported coins are:
              Bitcoin, Ethereum, Solana, and Cardano.

    Returns:
        A string with the simulated price information
    """
    # This is a simulated tool - in a real app, this would call a crypto API
    prices = {
        "bitcoin": "$45,234.21",
        "ethereum": "$2,890.65",
        "solana": "$98.42",
        "cardano": "$1.21",
    }

    coin = coin.lower()
    if coin in prices:
        return f"Current price of {coin.title()}: {prices[coin]}"
    else:
        return f"Price data for {coin} not available. Try Bitcoin, Ethereum, Solana, or Cardano."


async def run_examples(examples):
    # Run each example sequentially with a timeout
    await asyncio.gather(*(run_example(*example) for example in examples))


async def run_example(
    example_num: int,
    task: str,
    builder_fn: Callable[[], Awaitable[ReactAgent]],
    confirmation_callback: Optional[Callable[..., Awaitable[bool]]] = None,
    timeout: int = 60,
) -> None:
    """
    Run a single example with proper error handling and cleanup.

    Args:
        example_num: Example number for display
        task: The task to run
        builder_fn: Async function that builds and returns the agent
        confirmation_callback: Optional confirmation callback function
        timeout: Timeout in seconds for the example execution
    """
    # Use a separate task for each example to ensure proper resource management
    task_complete = asyncio.Event()
    agent = None

    async def example_worker():
        nonlocal agent
        try:
            print(f"\n=== Example {example_num}: {task} ===")

            # Build the agent
            agent = await builder_fn()
            # Run the task
            result = await agent.run(initial_task=task)

            # Print results
            print(f"\n--- Example {example_num} Result ---")
            print(json.dumps(result, indent=4, default=str))
            print("------------------------")

        except asyncio.CancelledError:
            print(f"\nExample {example_num} was cancelled")
        except Exception as e:
            print(f"\nError in Example {example_num}: {e}")
            traceback.print_exc()
        finally:
            # Mark task as complete before cleanup
            task_complete.set()

            # Clean up resources in the same task
            if agent:
                try:
                    print(f"\nClosing Agent for Example {example_num}...")
                    await agent.close()
                    print(f"Agent closed for Example {example_num}.")
                except Exception as e:
                    print(f"Error closing agent for Example {example_num}: {e}")

    # Create and start the task
    example_task = asyncio.create_task(example_worker())

    try:
        # Wait for the task to complete with timeout
        await asyncio.wait_for(task_complete.wait(), timeout=timeout)
        # Wait for task to finish if completed naturally
        if example_task.done():
            await example_task
        else:
            # Cancel task if not already done
            example_task.cancel()
            try:
                await example_task
            except asyncio.CancelledError:
                pass
    except asyncio.TimeoutError:
        print(f"\nExample {example_num} timed out after {timeout} seconds")
        example_task.cancel()
        try:
            await example_task
        except asyncio.CancelledError:
            pass


async def main():
    """
    Example usage of different ways to create and use ReactAgents with both
    MCP client tools and custom tools.
    """
    try:
        # For confirmation callbacks
        async def confirmation_callback(
            action_description: str, details: Dict[str, Any]
        ) -> bool:
            print(f"\n--- Tool Confirmation Request ---")
            print(f"Tool: {details.get('tool', 'unknown')}")
            print(f"Parameters: {json.dumps(details.get('params', {}), indent=2)}")
            user_input = (
                input("Proceed with this tool execution? (y/n) [y]: ").lower().strip()
            )
            return user_input == "y" or user_input == ""

        # Create a Pydantic confirmation config
        confirmation_config = ConfirmationConfig(
            enabled=True,
            strategy="always",
            allowed_silent_tools=["get_current_time"],
        )

        # Run examples sequentially with proper cleanup between them
        examples: List[
            Tuple[int, str, Callable[[], Awaitable[ReactAgent]], Callable, int]
        ] = []

        # Example 1: Using only MCP tools
        task1 = "What is the current time in New York and Tokyo?"

        async def build_agent1():
            return await (
                ReactAgentBuilder()
                .with_name("MCP Tools Agent")
                .with_model(PROVIDER_MODEL)
                .with_mcp_tools(["time", "brave-search"])
                .with_instructions("Answer questions using MCP tools.")
                .with_confirmation(confirmation_callback, confirmation_config)
                .with_log_level(LogLevel.INFO)
                .build()
            )

        examples.append((1, task1, build_agent1, confirmation_callback, 60))

        # Example 2: Using only custom tools
        task2 = "What's the weather in Tokyo and London? Also, what is the current price of Bitcoin and Ethereum?"

        async def build_agent2():
            # First check the tool registration
            builder = (
                ReactAgentBuilder()
                .with_name("Custom Tools Agent")
                .with_model(PROVIDER_MODEL)
                .with_custom_tools([custom_weather_tool, crypto_price_simulator])
                .with_instructions("Answer questions using custom tools.")
                .with_confirmation(confirmation_callback, confirmation_config)
            )

            # Debug tools before building
            tool_info = builder.debug_tools()
            print("\n--- Pre-build Tool Registration Check ---")
            print(f"Custom tools: {', '.join(tool_info['custom_tools'])}")
            print("----------------------------------------\n")

            return await builder.build()

        examples.append((2, task2, build_agent2, confirmation_callback, 60))

        # Example 3: Using both MCP and custom tools
        task3 = "What time is it in New York? What's the weather in Tokyo? What's the price of Solana?"

        async def build_agent3():
            # Create the agent with explicit tool configuration
            builder = (
                ReactAgentBuilder()
                .with_name("Hybrid Tools Agent")
                .with_model(PROVIDER_MODEL)
                .with_tools(
                    mcp_tools=["time"],
                    custom_tools=[custom_weather_tool, crypto_price_simulator],
                )
                .with_instructions("Answer questions using both MCP and custom tools.")
                .with_confirmation(confirmation_callback)
            )

            # Debug tools before building
            tool_info = builder.debug_tools()
            print("\n--- Hybrid Agent Pre-build Tool Check ---")
            print(f"MCP tools: {', '.join(tool_info['mcp_tools'])}")
            print(f"Custom tools: {', '.join(tool_info['custom_tools'])}")
            print("----------------------------------------\n")

            agent = await builder.build()

            # Verify tool registration after building
            diagnosis = await ReactAgentBuilder.diagnose_agent_tools(agent)
            print("\n--- Hybrid Agent Post-build Tool Check ---")
            print(f"Tools in context: {', '.join(diagnosis['context_tools'])}")
            print(f"Tools in manager: {', '.join(diagnosis['manager_tools'])}")

            if diagnosis["has_tool_mismatch"]:
                print("⚠️ WARNING: Tool registration mismatch detected!")
            else:
                print("✅ All tools properly registered")
            print("----------------------------------------\n")

            # Enhanced instructions to ensure tool usage
            agent.context.instructions += (
                "\nIMPORTANT: For this task, you'll need to:\n"
                + "1. Use get_current_time for New York time\n"
                + "2. Use custom_weather_tool for Tokyo weather\n"
                + "3. Use crypto_price_simulator for Solana price\n"
                + "Make sure to use ALL these tools to complete the task."
            )

            return agent

        examples.append((3, task3, build_agent3, confirmation_callback, 60))

        # Example 4: Using the factory method with custom tools added
        task4 = "Research when Bitcoin was created and what its current price is"

        async def build_agent4():
            # Start with a research agent but add a custom crypto price tool
            agent = await ReactAgentBuilder.research_agent(model=PROVIDER_MODEL)

            # Use the new utility method to add the custom tool
            return await ReactAgentBuilder.add_custom_tools_to_agent(
                agent, [crypto_price_simulator]
            )

        examples.append((4, task4, build_agent4, confirmation_callback, 60))

        await run_examples(examples)
    except Exception as e:
        print(f"\nAn error occurred in main: {e}")
        traceback.print_exc()


async def legacy_main():
    """
    Example usage of the legacy way to create a ReactAgent with custom tools.

    This demonstrates creating a minimal ReactAgent using ReactAgentConfig with custom tools.
    """
    mcp_config = {
        "mcpServers": {
            "duckduckgo": {
                "args": ["run", "-i", "--rm", "mcp/duckduckgo"],
                "command": "docker",
            }
        }
    }
    agent = await ReactAgent(
        config=ReactAgentConfig(
            agent_name="Legacy Agent",
            provider_model_name=PROVIDER_MODEL,
            mcp_config=mcp_config,
        )
    ).initialize()
    result = await agent.run(initial_task="When did Chris Cornell die?")
    print(result)
    await agent.close()


async def context_managed_main():

    mcp_config = {
        "mcpServers": {
            "time": {
                "args": ["run", "-i", "--rm", "mcp/time"],
                "command": "docker",
            }
        }
    }
    #  Create agent with required parameters and custom tools
    async with ReactAgent(
        config=ReactAgentConfig(
            agent_name="Context Managed Agent",
            provider_model_name=PROVIDER_MODEL,
            mcp_config=mcp_config,
        )
    ) as agent:
        # Run the agent with the task specified at runtime
        result = await agent.run(initial_task="What is the current time in New York?")
        print(result)


if __name__ == "__main__":
    PROVIDER_MODEL = "ollama:cogito:14b"
    # 👇 Run the basic minimal agent example
    asyncio.run(legacy_main())
    # 👇 Run the builder pattern examples
    asyncio.run(main())
    # 👇 Run the context managed agent example
    asyncio.run(context_managed_main())
