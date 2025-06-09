# Reactive AI Agent Framework

[![CI Build Status](https://github.com/tylerjrbuell/reactive-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/tylerjrbuell/reactive-agents/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/reactive-agents.svg)](https://badge.fury.io/py/reactive-agents)
[![Python](https://img.shields.io/pypi/pyversions/reactive-agents.svg)](https://pypi.org/project/reactive-agents/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A custom reactive AI Agent framework that allows for creating reactive agents to carry out tasks using tools. The framework provides a flexible system for creating AI agents that can use different LLM providers (Ollama, Groq) and reflect on their actions and improve iteratively.

## Quick Start

The simplest way to create a reactive agent using the recommended Builder pattern:

```python
import asyncio
from reactive_agents.agents import ReactAgentBuilder
from reactive_agents.tools.decorators import tool

# Define a custom tool
@tool(description="Get the current weather for a location")
async def weather_tool(location: str) -> str:
    # In a real app, this would call a weather API
    return f"The weather in {location} is sunny and 72°F"

async def main():
    # Build agent with minimal configuration and custom tools
    agent = await (
        ReactAgentBuilder()
        .with_name("QuickStart Agent")
        .with_model("ollama:qwen2:7b")
        .with_custom_tools([weather_tool])
        .build()
    )

    # Initialize the agent (done by .build() when using the builder)

    # Run the agent with a task
    result = await agent.run(
        initial_task="What's the weather in Tokyo?"
    )
    print(result)

    # Always close the agent when done
    await agent.close()

asyncio.run(main())
```

## Context Management Support (New)

`ReactAgentBuilder` and the resulting agent instances support Python's async context management protocol, making resource management even easier. This is the recommended way to use agents when possible:

```python
import asyncio
from reactive_agents.agents import ReactAgentBuilder
from reactive_agents.tools.decorators import tool

@tool(description="Get the current weather for a location")
async def weather_tool(location: str) -> str:
    return f"The weather in {location} is sunny and 72°F"

async def main():
    async with ReactAgentBuilder().with_name("Context Managed Agent").with_model("ollama:qwen2:7b").with_custom_tools([weather_tool]).build() as agent:
        # The agent is automatically initialized and will be closed when the block exits
        result = await agent.run(
            initial_task="What's the weather in Tokyo?"
        )
        print(result)

asyncio.run(main())
```

**Key Points:**

- Use the `ReactAgentBuilder`'s `.build()` method, preferably within an `async with` block, for automatic initialization and cleanup.
- If you need to manage the agent lifecycle manually after building (less recommended), remember to `await agent.close()` when done.

## Overview

The main purpose of this project is to create a custom AI Agent Framework that allows AI Agents driven by Large Language Models (LLMs) to make real-time decisions and take action to solve real-world tasks. Key features include:

- **Model Providers**: Currently Supports `Ollama` for open-source models (local) or `Groq` fast cloud-based models.
- **Agent Reflection**: The agent has the ability to reflect on its previous actions, improve as it iterates, and grade itself until it arrives at a final result.
- **Tool Integration**: Agents can use both MCP client tools (server-side) and custom Python functions decorated with `@tool()`.
- **Builder Pattern**: Easy agent creation with a fluent interface and sensible defaults using the `ReactAgentBuilder` class.
- **Model Context Protocol (MCP)**: Supports distributed tool execution through MCP servers, allowing agents to use tools from multiple sources.
- **Workflow Management**: Supports creating complex agent workflows with dependencies and parallel execution.
- **Strong Type Hinting**: Uses Pydantic models for configuration to ensure type safety and better developer experience.

## Installation

You can install the package directly from PyPI:

```sh
pip install reactive-agents
```

Or using Poetry:

```sh
poetry add reactive-agents
```

For development installation:

## Installation Instructions

To install and set up this project locally, follow these steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/tylerjrbuell/reactive-agents
   cd reactive-agents
   ```

2. Install dependencies using Poetry:

   ```sh
   poetry install
   ```

3. Configure your environment by setting up necessary variables in `.env`:
   ```env
   OLLAMA_HOST=http://localhost:11434
   GROQ_API_KEY=your_groq_api_key  # Required for Groq model provider
   BRAVE_API_KEY=your_brave_api_key # Required for brave-search MCP server
   MCP_CONFIG_PATH=/path/to/custom/mcp_config.json # Optional: Path to custom MCP configuration
   ```

## MCP Configuration

The framework uses Model Context Protocol (MCP) servers to provide standardized tool interfaces. Server configuration is highly customizable through:

1. Environment variables
2. Custom configuration files
3. Docker settings override

### Default MCP Servers

Included MCP servers:

- **local**: Local tool execution server
- **time**: Time-related utilities
- **filesystem**: File system operations (mounted at /projects)
- **sqlite**: SQLite database operations
- **playwright**: Web automation tools
- **brave-search**: Web search using Brave API
- **duckduckgo**: Web search using DuckDuckGo

### Custom Configuration

You can customize MCP servers in a few ways:

1. Environment variable (`MCP_CONFIG_PATH`):

   ```env
   MCP_CONFIG_PATH=/path/to/custom/mcp_config.json
   ```

2. Python

   ```python

   # Minimal example using `ReactAgentBuilder`
   agent = await ReactAgentBuilder().with_mcp_config_path("/path/to/custom/mcp_config.json").build()

   # Minimal Example using `ReactAgent` class directly
   from reactive_agents.agents import ReactAgent, ReactAgentConfig

   agentConf = ReactAgentConfig(mcp_config_path="/path/to/custom/mcp_config.json")
   agent = await ReactAgent(config=agentConf).initialize()

   ```

- JSON configuration file example:

  ```json
  {
    "mcpServers": {
      "custom-server": {
        "command": "python",
        "args": ["./path/to/server.py"],
        "env": {
          "CUSTOM_VAR": "value"
        },
        "working_dir": "/path/to/working/dir",
        "enabled": true
      },
      // For networked Docker servers
      "custom-docker-server": {
        "command": "docker",
        "args": ["run", "--name", "my-server", "-i", "--rm", "my-image"],
        "docker": {
          "network": "my-network",
          "extra_mounts": ["type=bind,src=/host/path,dst=/container/path"],
          "extra_env": {
            "DOCKER_VAR": "value"
          }
        }
      }
    },
    // For non-networked Docker servers
    "default_docker_config": {
      "host": "unix:///var/run/docker.sock",
      "network": "default-network"
    }
  }
  ```

## Usage Details

## Creating Agents

There are several ways to create and use reactive agents:

### 1. Quick Create Function (Simplest Method)

For beginners or quick testing, use the `quick_create_agent` function:

```python
import asyncio
from reactive_agents.agents import quick_create_agent

async def main():
    # Create and run an agent in a single line
    result = await quick_create_agent(
        task="Research the current price of Bitcoin",
        model="ollama:qwen3:4b",
        tools=["brave-search", "time"],
        interactive=True  # Set to True to confirm tool executions
    )
    print(f"Result: {result}")

asyncio.run(main())
```

### 2. Builder Pattern (Recommended)

For most use cases, use the `ReactAgentBuilder` class with its fluent interface:

```python
import asyncio
from reactive_agents.agents import ReactAgentBuilder
from reactive_agents.agents.builders import ConfirmationConfig

async def main():
    # Create a confirmation callback for interactive use
    async def confirmation_callback(action, details):
        print(f"Tool: {details.get('tool')}")
        return input("Approve? (y/n) [y]: ").lower() in ("", "y")

    # Create a configuration for the confirmation system
    config = ConfirmationConfig(
        strategy="always",
        allowed_silent_tools=["get_current_time"]
    )

    # Build an agent with a fluent interface
    agent = await (
        ReactAgentBuilder()
        .with_name("Research Agent")
        .with_model("ollama:qwen3:4b")
        .with_mcp_tools(["brave-search", "time"])
        .with_instructions("Research information thoroughly.")
        .with_max_iterations(10)
        .with_reflection(True)
        .with_confirmation(confirmation_callback, config)
        .build()
    )

    try:
        # Run the agent with a task
        result = await agent.run("What is the current price of Bitcoin?")
        print(f"Result: {result}")
    finally:
        # Always close the agent to clean up resources
        await agent.close()

asyncio.run(main())
```

### 3. Factory Methods (Preset Configurations)

For common use cases, use the factory methods:

```python
import asyncio
from reactive_agents.agents import ReactAgentBuilder

async def main():
    # Create a specialized research agent
    agent = await ReactAgentBuilder.research_agent(model="ollama:qwen3:4b")

    try:
        result = await agent.run("Research the history of Bitcoin")
        print(f"Result: {result}")
    finally:
        await agent.close()

    # Create a database-focused agent
    db_agent = await ReactAgentBuilder.database_agent(model="ollama:llama3:8b")

    try:
        result = await db_agent.run("Create a table of cryptocurrency prices")
        print(f"Result: {result}")
    finally:
        await db_agent.close()

asyncio.run(main())
```

### 4. Using Custom Tools

You can create and use custom tools with the `@tool()` decorator:

```python
import asyncio
from reactive_agents.agents import ReactAgentBuilder
from reactive_agents.tools.decorators import tool

# Define a custom tool
@tool(description="Get the current weather for a location")
async def weather_tool(location: str) -> str:
    """
    Get weather information for a location.

    Args:
        location: The city to get weather for

    Returns:
        A string with weather information
    """
    # In a real app, this would call a weather API
    return f"The weather in {location} is sunny and 72°F"

async def main():
    # Create an agent with a custom tool
    agent = await (
        ReactAgentBuilder()
        .with_name("Weather Agent")
        .with_model("ollama:qwen3:4b")
        .with_custom_tools([weather_tool])
        .build()
    )

    try:
        result = await agent.run("What's the weather in New York?")
        print(f"Result: {result}")
    finally:
        await agent.close()

asyncio.run(main())
```

### 5. Hybrid Tool Usage

You can combine MCP tools and custom tools in a single agent:

```python
import asyncio
from reactive_agents.agents import ReactAgentBuilder
from reactive_agents.tools.decorators import tool

@tool(description="Get cryptocurrency price information")
async def crypto_price(coin: str) -> str:
    # Simulated tool
    prices = {"bitcoin": "$41,234.56", "ethereum": "$2,345.67"}
    return prices.get(coin.lower(), f"No price data for {coin}")

async def main():
    # Create an agent with both MCP and custom tools
    agent = await (
        ReactAgentBuilder()
        .with_name("Research Agent")
        .with_model("ollama:qwen3:4b")
        .with_tools(
            mcp_tools=["brave-search", "time"],
            custom_tools=[crypto_price]
        )
        .build()
    )

    try:
        result = await agent.run(
            "What time is it now? What is Bitcoin's price?"
        )
        print(f"Result: {result}")
    finally:
        await agent.close()

asyncio.run(main())
```

### 6. Adding Custom Tools to Existing Agents

You can add custom tools to an agent that has already been created:

```python
import asyncio
from reactive_agents.agents import ReactAgentBuilder
from reactive_agents.tools.decorators import tool

@tool(description="Get cryptocurrency price information")
async def crypto_price(coin: str) -> str:
    # Simulated tool
    prices = {"bitcoin": "$41,234.56", "ethereum": "$2,345.67"}
    return prices.get(coin.lower(), f"No price data for {coin}")

async def main():
    # Create a research agent from the factory
    agent = await ReactAgentBuilder.research_agent()

    # Add a custom tool to the existing agent
    agent = await ReactAgentBuilder.add_custom_tools_to_agent(
        agent, [crypto_price]
    )

    try:
        result = await agent.run(
            "Research Bitcoin and get its current price"
        )
        print(f"Result: {result}")
    finally:
        await agent.close()

asyncio.run(main())
```

### 7. Traditional ReactAgent Creation (Legacy Method)

While the Builder Pattern is recommended, you can still create `ReactAgent` instances directly for advanced configurations or backward compatibility. However, note that directly instantiating `ReactAgent` **does not automatically handle initialization or cleanup**. You must manually call `await agent.initialize()` after creation and `await agent.close()` when done.

```python
import asyncio
from reactive_agents.agent_mcp.client import MCPClient
from reactive_agents.agents import ReactAgent, ReactAgentConfig

async def main():
    # Initialize MCP client
    mcp_client = await MCPClient(
        server_filter=["local", "brave-search"]
    ).initialize()

    # Create the agent config
    config = ReactAgentConfig(
        agent_name="TaskAgent",
        role="Task Executor",
        provider_model_name="ollama:cogito:14b",
        min_completion_score=1.0,
        instructions="Complete tasks efficiently.",
        mcp_client=mcp_client,
        log_level="info",
        max_iterations=5,
        reflect_enabled=True
    )

    # Create the agent
    agent = ReactAgent(config=config)
    await agent.initialize()  # <-- Required for full setup

    try:
        result = await agent.run("Research the price of Bitcoin")
        print(f"Result: {result}")
    finally:
        await agent.close()
        await mcp_client.close()

asyncio.run(main())
```

### 8. Agent Event Subscription

You can monitor and react to agent lifecycle events in real-time using the event subscription system. The framework provides two main approaches for event subscription:

#### Method 1: Using Specific Event Methods

```python
import asyncio
from reactive_agents.agents import ReactAgentBuilder
from reactive_agents.context.agent_events import ToolCalledEventData, ToolCompletedEventData

async def main():
    # Create counters to track events
    tool_calls = 0
    successful_tools = 0

    # Define callbacks for specific events
    def on_tool_called(event: ToolCalledEventData):
        nonlocal tool_calls
        tool_calls += 1
        print(f"Tool #{tool_calls}: {event['tool_name']} called with parameters: {event['parameters']}")

    def on_tool_completed(event: ToolCompletedEventData):
        nonlocal successful_tools
        successful_tools += 1
        print(f"Tool completed: {event['tool_name']} (execution time: {event['execution_time']:.2f}s)")
        print(f"Result: {event['result']}")

    # Create an agent with event subscriptions
    agent = await (
        ReactAgentBuilder()
        .with_name("Observable Agent")
        .with_model("ollama:qwen3:4b")
        .with_mcp_tools(["brave-search"])
        # Subscribe to events with type-safe callbacks
        .on_session_started(lambda event: print(f"Session started: {event['session_id']}"))
        .on_tool_called(on_tool_called)
        .on_tool_completed(on_tool_completed)
        .on_iteration_started(lambda event: print(f"Iteration {event['iteration']} started"))
        .on_final_answer_set(lambda event: print(f"Final answer: {event['answer']}"))
        .build()
    )

    try:
        result = await agent.run("What is the current price of Bitcoin?")
        print(f"\nSummary:\n- Tool calls: {tool_calls}\n- Successful tools: {successful_tools}")
        print(f"Final result: {result}")
    finally:
        await agent.close()

asyncio.run(main())
```

#### Method 2: Using Generic with_subscription Method

For a more flexible approach, you can use the generic `with_subscription` method:

```python
import asyncio
from reactive_agents.agents import ReactAgentBuilder
from reactive_agents.context.agent_observer import AgentStateEvent
from reactive_agents.context.agent_events import ToolCalledEventData, SessionStartedEventData

async def main():
    # Define callbacks for events
    def log_session_start(event: SessionStartedEventData):
        print(f"New session started: {event['session_id']} with task: {event['initial_task']}")

    def log_tool_usage(event: ToolCalledEventData):
        print(f"Tool called: {event['tool_name']} with parameters: {event['parameters']}")

    # Create an agent with generic event subscriptions
    agent = await (
        ReactAgentBuilder()
        .with_name("Generic Subscription Agent")
        .with_model("ollama:qwen3:4b")
        .with_mcp_tools(["brave-search"])
        # Use the generic subscription method with proper event types
        .with_subscription(AgentStateEvent.SESSION_STARTED, log_session_start)
        .with_subscription(AgentStateEvent.TOOL_CALLED, log_tool_usage)
        # You can mix and match with the specific methods
        .on_tool_completed(lambda event: print(f"Tool {event['tool_name']} completed"))
        .build()
    )

    try:
        result = await agent.run("Research the latest cryptocurrency trends")
        print(f"Final result: {result}")
    finally:
        await agent.close()

asyncio.run(main())
```

You can also use the async version for asynchronous callbacks:

```python
import asyncio
from reactive_agents.agents import ReactAgentBuilder
from reactive_agents.context.agent_observer import AgentStateEvent

async def log_tool_async(event):
    # Simulate async database logging
    await asyncio.sleep(0.1)  # Simulate network delay
    print(f"Async logged: Tool {event['tool_name']} with params {event['parameters']}")

async def main():
    agent = await (
        ReactAgentBuilder()
        .with_name("Async Subscription Agent")
        .with_model("ollama:qwen3:4b")
        .with_mcp_tools(["brave-search"])
        # Register async callbacks using the _async suffix methods
        .on_session_started_async(lambda event: asyncio.create_task(log_agent_session(event)))
        .on_tool_called_async(log_tool_call)
        .build()
    )
    # ... rest of the code
```

For more advanced usage and complete type safety, see the [Agent State Observation documentation](docs/README_AGENT_STATE_OBSERVATION.md).

### Available Agent State Events

The `AgentStateEvent` enum defines all the observable events in the agent's lifecycle. You can subscribe to these events using the methods described above.

Here is a comprehensive list of the available events:

- `SESSION_STARTED`: Emitted when a new agent run session begins.
- `SESSION_ENDED`: Emitted when an agent run session finishes (either successfully or due to an error/stop).
- `TASK_STATUS_CHANGED`: Emitted when the agent's internal task status is updated.
- `ITERATION_STARTED`: Emitted at the beginning of each agent iteration.
- `ITERATION_COMPLETED`: Emitted at the end of each agent iteration.
- `TOOL_CALLED`: Emitted just before a tool is executed.
- `TOOL_COMPLETED`: Emitted after a tool successfully completes.
- `TOOL_FAILED`: Emitted if a tool execution fails.
- `REFLECTION_GENERATED`: Emitted after the agent generates a reflection on its previous steps.
- `FINAL_ANSWER_SET`: Emitted when the agent determines the final answer for the task.
- `METRICS_UPDATED`: Emitted when the agent's internal performance metrics are updated.
- `ERROR_OCCURRED`: Emitted when an unhandled error occurs during the agent's execution.
- `PAUSE_REQUESTED`: Emitted when a pause of the agent's execution is requested.
- `PAUSED`: Emitted when the agent successfully pauses its execution.
- `RESUME_REQUESTED`: Emitted when a resume of the agent's execution is requested.
- `RESUMED`: Emitted when the agent successfully resumes its execution.
- `STOP_REQUESTED`: Emitted when a graceful stop of the agent's execution is requested.
- `STOPPED`: Emitted when the agent successfully stops gracefully.
- `TERMINATE_REQUESTED`: Emitted when a forceful termination of the agent's execution is requested.
- `TERMINATED`: Emitted when the agent is forcefully terminated.

## Tool Diagnostics and Debugging

The framework provides diagnostic tools to help debug tool registration issues:

```python
import asyncio
from reactive_agents.agents import ReactAgentBuilder
from reactive_agents.tools.decorators import tool

@tool(description="Example custom tool")
async def example_tool(param: str) -> str:
    return f"Result: {param}"

async def main():
    # Create a builder with tools
    builder = (
        ReactAgentBuilder()
        .with_name("Diagnostic Agent")
        .with_mcp_tools(["brave-search"])
        .with_custom_tools([example_tool])
    )

    # Debug tools before building
    diagnostics = builder.debug_tools()
    print(f"MCP tools: {diagnostics['mcp_tools']}")
    print(f"Custom tools: {diagnostics['custom_tools']}")

    # Build the agent
    agent = await builder.build()

    # Diagnose tool registration after building
    diagnosis = await ReactAgentBuilder.diagnose_agent_tools(agent)
    if diagnosis["has_tool_mismatch"]:
        print("WARNING: Tool registration mismatch detected!")
    else:
        print("All tools properly registered")

    await agent.close()

asyncio.run(main())
```

## Workflow Configuration (Multi-Agent Setup)

You can create multi-agent workflows:

```python
from config.workflow import AgentConfig, WorkflowConfig, Workflow

def create_workflow() -> Workflow:
    workflow_config = WorkflowConfig()

    # Add a planner agent
    planner = AgentConfig(
        role="planner",
        model="ollama:cogito:14b",
        min_score=0.9,
        instructions="Break down tasks into steps.",
        mcp_servers=["local", "brave-search"]
    )

    # Add an executor agent that depends on the planner
    executor = AgentConfig(
        role="executor",
        model="ollama:cogito:14b",
        min_score=1.0,
        instructions="Execute the planned steps.",
        dependencies=["planner"],
        mcp_servers=["local", "filesystem", "sqlite"]
    )

    workflow_config.add_agent(planner)
    workflow_config.add_agent(executor)

    return Workflow(workflow_config)

async def main():
    workflow = create_workflow()
    result = await workflow.run("Research Bitcoin price and store in database")
    print(f"Workflow result: {result}")

asyncio.run(main())
```

## Custom MCP Servers

Create custom MCP servers:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-agent-mcp")

@mcp.tool()
def my_custom_tool(param: str) -> str:
    """Tool description"""
    return f"Result: {param}"

mcp.run(transport="stdio")
```

## Running Tests

To run the tests:

```sh
poetry run pytest
```

If you have Docker installed and would like to run the `test_real_agent_execution` test, which interacts with a real Ollama instance and Brave Search MCP server, set the `ENABLE_REAL_EXECUTION=1` environment variable before running pytest:

```sh
ENABLE_REAL_EXECUTION=1 poetry run pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
