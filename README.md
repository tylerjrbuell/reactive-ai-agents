# Reactive AI Agent Framework

[![PyPI version](https://badge.fury.io/py/reactive-agents.svg)](https://badge.fury.io/py/reactive-agents)
[![Python](https://img.shields.io/pypi/pyversions/reactive-agents.svg)](https://pypi.org/project/reactive-agents/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A custom reactive AI Agent framework that allow for creating reactive agents to carry out tasks using tools. The framework provides a flexible system for creating AI agents that can use different LLM providers (Ollama, Groq) and reflect on their actions and improve iteratively.

## Overview

The main purpose of this project is to create a custom AI Agent Framework that allows AI Agents driven by Large Language Models (LLMs) to make real-time decisions and take action to solve real-world tasks. Key features include:

- **Model Providers**: Currently Supports `Ollama` for open-source models (local) or `Groq` fast cloud-based models.
- **Agent Reflection**: The agent has the ability to reflect on its previous actions, improve as it iterates, and grade itself until it arrives at a final result.
- **Tool Integration**: Agents can take tools as ordinary Python functions and use a `@tool()` decorator to transform these functions into function definitions that the language model can understand.
- **Model Context Protocol (MCP)**: Supports distributed tool execution through MCP servers, allowing agents to use tools from multiple sources.
- **Workflow Management**: Supports creating complex agent workflows with dependencies and parallel execution.

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
   git clone https://github.com/yourusername/projectname.git
   cd projectname
   ```

2. Install dependencies using Poetry:

   ```sh
   poetry install
   ```

3. Configure your environment by setting up necessary variables in `.env`:
   ```env
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

You can customize MCP servers in two ways:

1. Environment variable (`MCP_CONFIG_PATH`):

   ```env
   MCP_CONFIG_PATH=/path/to/custom/mcp_config.json
   ```

2. JSON configuration file:
   ```json
   {
     "servers": {
       "custom-server": {
         "command": "python",
         "args": ["./path/to/server.py"],
         "env": {
           "CUSTOM_VAR": "value"
         },
         "working_dir": "/path/to/working/dir",
         "enabled": true
       },
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
     "default_docker_config": {
       "host": "unix:///var/run/docker.sock",
       "network": "default-network"
     }
   }
   ```

### Configure MCP Servers in Workflow

Specify which MCP servers to use in your agent configuration:

```python
from config.workflow import AgentConfig, WorkflowConfig, Workflow

# Create an agent with specific MCP servers
agent_config = AgentConfig(
    role="researcher",
    model="ollama:cogito:14b",
    mcp_servers=["local", "brave-search", "sqlite"],  # Specify servers to use
    min_score=0.7,
    instructions="You are a research specialist.",
)
```

### Custom MCP Servers

Create custom MCP servers by:

1. Creating a new Python server file using the MCP SDK
2. Adding the server configuration to your custom config file
3. Using the server in your agent configuration

Example local server in [agent_mcp/servers/server.py](agent_mcp/servers/server.py):

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-agent-mcp")

@mcp.tool()
def my_custom_tool(param: str) -> str:
    """Tool description"""
    return f"Result: {param}"

mcp.run(transport="stdio")
```

## Usage Details

## Creating Agents

There are two main ways to use reactive agents:

1. Compose single granular agents using the `ReactAgent` class:

   ```python
   import asyncio
   from agent_mcp.client import MCPClient

   mcp_client = await MCPClient(server_filter=["local", "brave-search"]).initialize() # Filter servers if needed

   agent = ReactAgent(
       name="TaskAgent",
       role="Task Executor",
       provider_model="ollama:cogito:14b", # Add available provider:model combinations (currently supports ollama and groq)
       min_completion_score=1.0, # Adjust as needed for your use case (0.0-1.0)
       instructions="You are an AI agent. Complete the task as quickly as possible.",
       mcp_client=mcp_client,
       log_level="info",
       max_iterations=5 # Adjust as needed for your use case
       reflect=True # Set to True to enable reflection ability which allows the agent to reflect on its actions and improve as it iterates
   )
   asyncio.run(agent.run("Find the current price of xrp using a web search, then create a table called crypto_prices (currency, price, timestamp), then insert the price of xrp into the table."))
   ```

2. Compose multi-agent workflows using the `Workflow`, `AgentConfig`, and `WorkflowConfig` classes:

   ```python
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
   workflow = Workflow(workflow_config)
   asyncio.run(workflow.run("Find the current price of xrp using a web search, then create a table called crypto_prices (currency, price, timestamp), then insert the price of xrp into the table."))
   ```

3. Run the main application script to test the agents and workflows:

   ```sh
   python main.py
   ```

## Agent Configuration

Agents can be configured with various parameters:

- **role**: The agent's role in the workflow
- **model**: The LLM model to use (format: "provider:model")
- **min_score**: Minimum completion score required (0.0-1.0)
- **instructions**: Role-specific instructions
- **dependencies**: List of other agent roles this agent depends on
- **max_iterations**: Maximum number of task iterations
- **mcp_servers**: List of MCP servers to use
- **reflect**: Enable/disable agent reflection capability (default: False) longer iteration but more accurate results
- **instructions_as_task**: Use instructions as the task input

## Tools and Decorators

Create custom tools without using MCP using the `@tool()` decorator:

```python
from tools.decorators import tool

@tool()
async def my_custom_tool(param1: str, param2: int) -> str:
    """
    Tool description here.

    Args:
        param1 (str): Description of param1
        param2 (int): Description of param2

    Returns:
        str: Description of return value
    """
    # Tool implementation
    return result
```

Custom tools can then be added to agents using the tools parameter in the `ReactAgent` constructor or `AgentConfig`.

## Running Tests

To run the tests:

```sh
poetry run pytest
```

## Modules Description

- **agents**: Contains the base Agent class and ReactAgent implementation for reflective task execution.

- **model_providers**: Supports multiple LLM providers:

  - `OllamaModelProvider`: Local model execution using Ollama
  - `GroqModelProvider`: Cloud-based model execution using Groq

- **tools**: Tool implementation and MCP integration:

  - `@tool()` decorator for creating tool definitions
  - `MCPToolWrapper` for integrating with MCP servers
  - Built-in tools for common operations

- **config**: Configuration management:
  - `WorkflowConfig`: Defines agent workflows and dependencies
  - `mcp_config.py`: MCP server configuration
  - `logging.py`: Logging configuration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
