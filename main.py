import json
import asyncio
import dotenv
import warnings
import tracemalloc
import traceback
from typing import Any, Dict, Optional, Callable, Awaitable
from pydantic import PydanticDeprecatedSince211

from reactive_agents.app.agents import ReactiveAgent
from reactive_agents.app.builders.agent import (
    ReactiveAgentBuilder as Builder,
    ConfirmationConfig,
    LogLevel,
)
from reactive_agents.config.mcp_config import MCPConfig, MCPServerConfig
from reactive_agents.core.tools.decorators import tool
from reactive_agents.core.types.reasoning_types import ReasoningStrategies

warnings.simplefilter("ignore", ResourceWarning)
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince211)
tracemalloc.start()
dotenv.load_dotenv()


@tool()
async def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web using Google Custom Search JSON API.

    Args:
        query (str): The search query string.
        num_results (int): Number of search results to return (default is 5).

    Returns:
        str: A string containing the search results.
    """
    return query + " - " + str(num_results)


# Example custom tool using the @tool decorator
@tool()
async def custom_weather_tool(location: str) -> str:
    """
    Get the weather for a given location (simulated).

    Args:
        location: (string) The location to get the weather for. Supported locations are:
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


@tool()
async def multi_city_weather_tool(cities: str, unit: str = "celsius") -> str:
    """
    Get the weather for multiple cities (simulated).

    Args:
        cities: (string) Comma-separated list of cities to get weather for.
               Supported cities: New York, London, Tokyo, Sydney
        unit: (string) Temperature unit - "celsius" or "fahrenheit" (default: celsius)

    Returns:
        A string with weather information for all requested cities
    """
    # This is a simulated tool - in a real app, this would call a weather API
    weather_data = {
        "New York": {"temp_c": 22, "temp_f": 72, "condition": "Sunny", "humidity": 65},
        "London": {"temp_c": 18, "temp_f": 64, "condition": "Rainy", "humidity": 80},
        "Tokyo": {"temp_c": 25, "temp_f": 77, "condition": "Cloudy", "humidity": 70},
        "Sydney": {
            "temp_c": 22,
            "temp_f": 72,
            "condition": "Partly Cloudy",
            "humidity": 75,
        },
    }

    city_list = [city.strip() for city in cities.split(",")]
    results = []

    for city in city_list:
        if city in weather_data:
            data = weather_data[city]
            temp = data["temp_c"] if unit.lower() == "celsius" else data["temp_f"]
            temp_unit = "°C" if unit.lower() == "celsius" else "°F"
            results.append(
                f"{city}: {temp}{temp_unit}, {data['condition']} (Humidity: {data['humidity']}%)"
            )
        else:
            results.append(f"{city}: Weather data not available")

    return "\n".join(results)


@tool()
async def crypto_price_simulator(coin: str) -> str:
    """
    Get the current price of a cryptocurrency (simulated).

    Args:
        coin: (string) The name of the cryptocurrency. Supported coins are:
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
    builder_fn: Callable[[], Awaitable[ReactiveAgent]],
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

            # Set up event handlers to showcase the dynamic event system
            agent.on_session_started(
                lambda event: print(
                    f"🎬 Session started: {event.get('agent_name', 'Unknown')}"
                )
            )
            agent.on_tool_called(
                lambda event: print(
                    f"🔧 Tool called: {event.get('tool_name', 'Unknown')}"
                )
            )
            agent.on_iteration_completed(
                lambda event: print(
                    f"🔄 Iteration {event.get('iteration', 0)} completed"
                )
            )
            agent.on_final_answer_set(
                lambda event: print(
                    f"✅ Final answer set with score: {event.get('completion_score', 0)}"
                )
            )

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
    Example usage of the new ReactiveAgent framework with enhanced features:
    - Dynamic event system
    - Multiple reasoning strategies
    - Advanced tool management
    - Real-time control operations
    """
    try:
        # For confirmation callbacks
        async def confirm(action_description: str, details: Dict[str, Any]) -> bool:
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

        # Example 1: Basic ReactiveAgent with dynamic reasoning
        async def build_agent1():
            return await (
                Builder()
                .with_name("Dynamic Research Agent")
                .with_model("ollama:cogito:14b")
                .with_role("Advanced Research Assistant")
                .with_instructions(
                    "Research topics thoroughly using dynamic reasoning strategies."
                )
                .with_reasoning_strategy(ReasoningStrategies.REFLECT_DECIDE_ACT)
                .with_mcp_tools(["brave-search", "time"])
                .with_custom_tools([custom_weather_tool])
                .with_confirmation(confirm, confirmation_config)
                .with_max_iterations(15)
                .with_dynamic_strategy_switching(True)
                .build()
            )

        # Example 2: Adaptive agent with strategy switching
        async def build_agent2():
            return await (
                Builder()
                .with_name("Adaptive Agent")
                .with_model("ollama:cogito:14b")
                .with_role("Adaptive Task Executor")
                .with_instructions("Adapt reasoning strategy based on task complexity.")
                .with_reasoning_strategy(ReasoningStrategies.ADAPTIVE)
                .with_mcp_tools(["brave-search", "time"])
                .with_custom_tools([crypto_price_simulator])
                .with_max_iterations(20)
                .with_dynamic_strategy_switching(True)
                .build()
            )

        # Example 3: Plan-execute-reflect strategy
        async def build_agent3():
            return await (
                Builder()
                .with_name("Planner Agent")
                .with_model("ollama:cogito:14b")
                .with_role("Strategic Planner")
                .with_instructions("Plan, execute, and reflect on complex tasks.")
                .with_reasoning_strategy(ReasoningStrategies.PLAN_EXECUTE_REFLECT)
                .with_mcp_tools(["brave-search", "time"])
                .with_custom_tools([web_search, custom_weather_tool])
                .with_max_iterations(12)
                .build()
            )

        # Example 4: Reactive strategy for quick responses
        async def build_agent4():
            return await (
                Builder()
                .with_name("Quick Response Agent")
                .with_model("ollama:cogito:14b")
                .with_role("Quick Response Assistant")
                .with_instructions(
                    "Provide quick, reactive responses to simple queries."
                )
                .with_reasoning_strategy(ReasoningStrategies.REACTIVE)
                .with_mcp_tools(["time"])
                .with_custom_tools([crypto_price_simulator])
                .with_max_iterations(5)
                .build()
            )

        # Example 5: Advanced agent with comprehensive event handling
        async def build_agent5():
            agent = await (
                Builder()
                .with_name("Event-Driven Agent")
                .with_model("ollama:cogito:14b")
                .with_role("Event-Driven Assistant")
                .with_instructions(
                    "Demonstrate comprehensive event handling capabilities."
                )
                .with_reasoning_strategy(ReasoningStrategies.REFLECT_DECIDE_ACT)
                .with_tools(
                    mcp_tools=["brave-search", "time"],
                    custom_tools=[custom_weather_tool, crypto_price_simulator],
                )
                .with_max_iterations(10)
                .build()
            )

            # Set up comprehensive event handlers
            agent.on_session_started(
                lambda event: print(f"🚀 Session started: {event.get('agent_name')}")
            )
            agent.on_session_ended(
                lambda event: print(f"🏁 Session ended: {event.get('agent_name')}")
            )
            agent.on_task_status_changed(
                lambda event: print(f"📊 Task status: {event.get('status')}")
            )
            agent.on_iteration_started(
                lambda event: print(f"🔄 Starting iteration {event.get('iteration')}")
            )
            agent.on_iteration_completed(
                lambda event: print(f"✅ Completed iteration {event.get('iteration')}")
            )
            agent.on_tool_called(
                lambda event: print(f"🔧 Called tool: {event.get('tool_name')}")
            )
            agent.on_tool_completed(
                lambda event: print(f"✅ Tool completed: {event.get('tool_name')}")
            )
            agent.on_tool_failed(
                lambda event: print(f"❌ Tool failed: {event.get('tool_name')}")
            )
            agent.on_reflection_generated(
                lambda event: print(f"🤔 Reflection generated")
            )
            agent.on_final_answer_set(lambda event: print(f"🎯 Final answer set"))
            agent.on_metrics_updated(lambda event: print(f"📈 Metrics updated"))
            agent.on_error_occurred(
                lambda event: print(f"⚠️ Error occurred: {event.get('error')}")
            )

            return agent

        # Example 6: Control operations demo
        async def control_operations_demo():
            print("\n=== Control Operations Demo ===")

            agent = await (
                Builder()
                .with_name("Controllable Agent")
                .with_model("ollama:cogito:14b")
                .with_role("Demo Agent")
                .with_instructions("Demonstrate control operations.")
                .with_reasoning_strategy(ReasoningStrategies.PLAN_EXECUTE_REFLECT)
                .with_mcp_tools(["time"])
                .with_max_iterations(10)
                .build()
            )
            # Set up control event handlers
            agent.on_pause_requested(lambda event: print("⏸️ Pause requested"))
            agent.on_paused(lambda event: print("⏸️ Agent paused"))
            agent.on_resume_requested(lambda event: print("▶️ Resume requested"))
            agent.on_resumed(lambda event: print("▶️ Agent resumed"))
            agent.on_stop_requested(lambda event: print("⏹️ Stop requested"))
            agent.on_stopped(lambda event: print("⏹️ Agent stopped"))
            agent.on_terminate_requested(lambda event: print("🔚 Terminate requested"))
            agent.on_terminated(lambda event: print("🔚 Agent terminated"))

            # Start a task in the background
            task = asyncio.create_task(
                agent.run("Research the current time and weather in New York")
            )

            # Wait a bit then demonstrate control operations
            await asyncio.sleep(2)
            print("\n--- Demonstrating Control Operations ---")

            await agent.pause()
            await asyncio.sleep(10)

            await agent.resume()
            await asyncio.sleep(3)

            await agent.stop()

            try:
                await task
            except Exception as e:
                print(f"Task stopped as expected: {e}")

            await agent.close()
            print("Control operations demo completed.")

        # Define examples
        examples = [
            (
                1,
                "Research the current weather in Tokyo and provide a brief summary",
                build_agent1,
            ),
            (
                2,
                "Get the current price of Bitcoin and explain its significance",
                build_agent2,
            ),
            (
                3,
                "Plan and execute a research task about renewable energy trends",
                build_agent3,
            ),
            (4, "Quickly provide the current time and a random fact", build_agent4),
            (
                5,
                "Research the weather in London and the price of Ethereum",
                build_agent5,
            ),
        ]

        # Run the examples
        print("🚀 Starting ReactiveAgent Framework Examples")
        print("=" * 60)

        # await run_examples(examples)

        # Run the control operations demo
        await control_operations_demo()

        print("\n🎉 All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()


async def classic_main():
    """
    Classic example using the simplified quick_create_agent function.
    """
    print("\n=== Classic Quick Create Example ===")

    try:
        from reactive_agents.app.builders.agent import quick_create_agent

        result = await quick_create_agent(
            task="Research the current weather in Paris and provide a summary",
            model="ollama:cogito:14b",
            tools=["brave-search", "time"],
            interactive=False,
        )

        print("Classic Example Result:")
        print(json.dumps(result, indent=4, default=str))

    except Exception as e:
        print(f"Error in classic example: {e}")
        traceback.print_exc()


async def context_managed_main():
    """
    Example using context manager for automatic resource cleanup.
    """
    print("\n=== Context Manager Example ===")

    try:
        agent = await (
            Builder()
            .with_name("Context Managed Agent")
            .with_model("ollama:cogito:14b")
            .with_role("Context Demo")
            .with_instructions("Demonstrate context manager usage.")
            .with_mcp_tools(["time"])
            .with_custom_tools([custom_weather_tool])
            .build()
        )

        async with agent:
            result = await agent.run("Get the current time and weather in Sydney")
            print("Context Manager Result:")
            print(json.dumps(result, indent=4, default=str))

    except Exception as e:
        print(f"Error in context manager example: {e}")
        traceback.print_exc()


async def test():
    """
    Simple test function for quick validation.
    """
    print("\n=== Simple Test ===")

    try:
        agent = (
            await (
                Builder()
                .with_name("Test Agent")
                .with_model("ollama:cogito:14b")
                # .with_mcp_tools(["brave-search"])
                # .with_custom_tools([custom_weather_tool])
                .with_mcp_config(
                    MCPConfig(
                        mcpServers={
                            "gmail": MCPServerConfig(
                                command="mcp-proxy",
                                args=["http://localhost:5000/mcp"],
                            )
                        }
                    )
                )
                .with_log_level(LogLevel.DEBUG)
                .with_reasoning_strategy(ReasoningStrategies.REACTIVE)
                .with_max_iterations(10)
                .with_tool_caching(False)
                # .with_reflection(True)
                # .with_vector_memory()
                .with_model_provider_options(
                    {"num_gpu": 256, "num_ctx": 4000, "temperature": 0.2}
                )
                .build()
            )
        )
        task = """
        Authenticate to gmail using the browser, Then pull 25 emails from my inbox, if the email subject and content indicates marketing or spam,
        move it to the trash, Finally, send an email to tylerjrbuell@gmail.com with a summary of the emails that were moved to the trash.
        """

        result = await agent.run(task)
        print("Test Result:")

        # Use a custom JSON encoder to handle non-serializable objects
        def json_serializer(obj):
            if hasattr(obj, "value"):
                return obj.value
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return str(obj)

        print(json.dumps(result, indent=2, default=json_serializer))
        await agent.close()

    except Exception as e:
        print(f"Error in test: {e}")
        traceback.print_exc()


async def test_reflect_decide_act():
    """
    Test the reflect_decide_act strategy specifically.
    """
    print("\n=== Testing Reflect-Decide-Act Strategy ===")

    try:
        agent = await (
            Builder()
            .with_name("RDA Test Agent")
            .with_model("ollama:cogito:14b")
            .with_role("Test Assistant")
            .with_instructions(
                "You are a test assistant for the reflect-decide-act strategy."
            )
            .with_reasoning_strategy(ReasoningStrategies.REFLECT_DECIDE_ACT)
            .with_mcp_tools(["brave-search", "filesystem"])
            .with_custom_tools([custom_weather_tool])
            .with_max_iterations(5)
            .with_log_level(LogLevel.DEBUG)
            .with_model_provider_options(
                {"num_gpu": 256, "num_ctx": 4000, "temperature": 0}
            )
            .build()
        )

        # Test with a task that should use reflect-decide-act
        task = "What is the current price of bitcoin, xrp, ethereum, solana? Create a markdown file called ./prices.md with the prices in a nice markdown table"
        print(f"Running task: {task}")
        result = await agent.run(task)

        print("RDA Strategy Test Result:")
        print(result.get("final_answer", "No final answer"))
        # print(json.dumps(result, indent=2, default=str))

        await agent.close()

    except Exception as e:
        print(f"Error in reflect_decide_act test: {e}")
        traceback.print_exc()


async def test_reactive_strategy():
    """
    Test the reactive strategy specifically.
    """
    print("\n=== Testing Reactive Strategy ===")

    try:
        agent = await (
            Builder()
            .with_name("Reactive Test Agent")
            .with_model("ollama:cogito:14b")
            .with_role("Quick Response Assistant")
            .with_instructions(
                "Provide quick, reactive responses to simple queries using the reactive strategy."
            )
            .with_reasoning_strategy(ReasoningStrategies.REACTIVE)
            .with_mcp_tools(["time", "brave-search"])
            .with_custom_tools([crypto_price_simulator, custom_weather_tool])
            .with_max_iterations(10)
            .with_log_level(LogLevel.DEBUG)
            .with_dynamic_strategy_switching(False)  # Force reactive strategy
            .build()
        )

        # Run a simple task
        task = "Get the current price of bitcoin, xrp, ethereum, solana"
        print(f"Running reactive strategy task: {task}")

        result = await agent.run(task)

        print("Reactive Strategy Test Result:")
        print(json.dumps(result, indent=4))

        # Check if the task was completed successfully
        success = (
            result.get("status") == "complete"
            and result.get("final_answer") is not None
        )
        print(f"\n✅ Reactive Strategy Test: {'PASSED' if success else 'FAILED'}")

        if success:
            print(f"✅ Final Answer: {result.get('final_answer', 'No final answer')}")
        else:
            print(f"❌ Status: {result.get('status', 'unknown')}")
            print(f"❌ Completion Score: {result.get('completion_score', 0)}")

        await agent.close()
        return result

    except Exception as e:
        print(f"Error during reactive strategy test: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_plan_execute_reflect():
    """
    Test the plan_execute_reflect strategy specifically.
    """
    print("\n=== Testing Plan-Execute-Reflect Strategy ===")

    try:
        agent = (
            await (
                Builder()
                .with_name("Plan-Execute-Reflect Test Agent")
                .with_model("ollama:gemma3n")
                .with_role("Test Assistant")
                .with_instructions("You are an agent that can plan and execute tasks.")
                .with_reasoning_strategy(ReasoningStrategies.PLAN_EXECUTE_REFLECT)
                # .with_mcp_tools(["brave-search", "filesystem"])
                .with_custom_tools([custom_weather_tool])
                .with_max_iterations(5)
                .with_log_level(LogLevel.DEBUG)
                .with_model_provider_options(
                    {"num_gpu": 256, "num_ctx": 4000, "temperature": 0}
                )
                .build()
            )
        )

        # Test with a task that should use plan_execute_reflect
        task = "Get the weather in Tokyo"
        print(f"Running task: {task}")
        result = await agent.run(task)

        print("Plan-Execute-Reflect Strategy Test Result:")

        def json_serializer(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            elif hasattr(obj, "dict"):
                return obj.dict()
            return str(obj)

        # Print the main result structure
        print(json.dumps(result, indent=4, default=json_serializer))

        await agent.close()

    except Exception as e:
        print(f"Error in plan_execute_reflect test: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Run the main examples (includes reactive strategy test)
    asyncio.run(main())

    # Run additional examples
    # asyncio.run(classic_main())
    # asyncio.run(context_managed_main())
    # asyncio.run(test())

    # Test reflect_decide_act strategy
    # asyncio.run(test_reflect_decide_act())

    # Test reactive strategy specifically
    # asyncio.run(test_reactive_strategy())

    # Test plan_execute_reflect strategy
    # asyncio.run(test_plan_execute_reflect())
