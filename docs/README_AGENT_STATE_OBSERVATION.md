# Real-time Agent State Observation

This system allows real-time monitoring of ReactAgent state and events as agents are executing tasks.

## Overview

The agent state observation system consists of three main components:

1. **AgentStateObserver**: A class that manages callbacks for different agent state events
2. **AgentStateEvent**: An enumeration of all observable events in an agent's lifecycle
3. **Integration with AgentContext**: Modifications to emit events at key points in the agent lifecycle

## Features

- Real-time observation of agent execution
- Track events like session start/end, iterations, tool usage, status changes, and errors
- Support for both synchronous and asynchronous callbacks
- Ability to register multiple observers for different event types
- Statistics tracking for monitoring system performance
- **NEW: Type-safe event subscription interface with proper TypeScript-like type checking**

## Events

The system tracks the following events:

- **SESSION_STARTED**: When a new agent session begins
- **SESSION_ENDED**: When an agent session completes
- **TASK_STATUS_CHANGED**: When the task status changes (e.g., running to complete)
- **ITERATION_STARTED**: When a new iteration begins
- **ITERATION_COMPLETED**: When an iteration completes
- **TOOL_CALLED**: When the agent calls a tool
- **TOOL_COMPLETED**: When a tool execution completes
- **TOOL_FAILED**: When a tool execution fails
- **REFLECTION_GENERATED**: When the agent generates a reflection
- **FINAL_ANSWER_SET**: When the agent sets a final answer
- **METRICS_UPDATED**: When agent metrics are updated
- **ERROR_OCCURRED**: When an error occurs during execution

## Usage

### Basic Usage

```python
from agents.react_agent import ReactAgent, ReactAgentConfig
from context.agent_observer import AgentStateEvent

# Create an agent with state observation enabled
agent_config = ReactAgentConfig(
    agent_name="ObservedAgent",
    # ...other configuration...
)

agent = ReactAgent(config=agent_config)

# Register a callback for a specific event
def handle_tool_call(event_data):
    print(f"Tool called: {event_data.get('tool_name')}")

agent.context.state_observer.register_callback(
    AgentStateEvent.TOOL_CALLED, handle_tool_call
)

# Run the agent
await agent.run(initial_task="Your task here")
```

### NEW: Type-Safe Subscription Interface

The new typed subscription interface provides a cleaner, more intuitive way to subscribe to events with proper type checking:

```python
from agents.react_agent import ReactAgent, ReactAgentConfig
from context.agent_events import ToolCalledEventData

# Create the agent
agent = ReactAgent(config=agent_config)

# Method 1: Direct subscription with typed callback
def handle_tool_call(event: ToolCalledEventData):
    print(f"Tool called: {event['tool_name']}")
    print(f"Parameters: {event['parameters']}")

agent.on_tool_called(handle_tool_call)

# Method 2: Using the fluent API with method chaining
(agent.events
    .on_session_started()
    .subscribe(lambda event: print(f"Session started: {event['session_id']}")))
```

See the full documentation in [TYPED_EVENT_SUBSCRIPTIONS.md](docs/TYPED_EVENT_SUBSCRIPTIONS.md).

### Adding a Custom Observer

You can create your own observer class to handle events:

```python
class MyAgentMonitor:
    def __init__(self):
        self.tool_calls = 0
        self.events = []

    def handle_event(self, event_data):
        self.events.append(event_data)

        if event_data.get("event_type") == "tool_called":
            self.tool_calls += 1
            print(f"Tool call #{self.tool_calls}: {event_data.get('tool_name')}")

# Register the monitor with all event types
monitor = MyAgentMonitor()
for event_type in AgentStateEvent:
    agent.context.state_observer.register_callback(event_type, monitor.handle_event)
```

## Examples

Three example applications are provided to demonstrate the agent state observation system:

1. **agent_observer_demo.py**: A simple command-line demonstration of tracking agent state
2. **realtime_agent_monitor.py**: A web-based real-time monitoring dashboard using FastAPI and WebSockets
3. **typed_event_subscription_demo.py**: Demo of the new type-safe subscription interface

To run the examples:

```bash
# Simple demo (requires no additional dependencies)
python examples/agent_observer_demo.py

# Web dashboard (requires fastapi, uvicorn, websockets)
pip install fastapi uvicorn websockets
python examples/realtime_agent_monitor.py
# Then open your browser to http://localhost:8888/

# Typed subscription demo
python examples/typed_event_subscription_demo.py
```

## Implementation Details

The implementation consists of four main parts:

1. `context/agent_observer.py`: Defines the `AgentStateObserver` class and `AgentStateEvent` enum
2. `context/agent_events.py`: Defines TypedDict models and a fluent subscription API
3. `context/agent_context.py`: Adds the observer to the agent context and provides methods to emit events
4. `agents/react_agent.py`: Adds event emission at key points in the agent lifecycle and exposes subscription methods

## Integration with External Systems

The agent state observation system can be integrated with external monitoring systems like:

- Prometheus metrics collection
- ElasticSearch for event logging
- Real-time dashboards using WebSockets
- Integration with application monitoring tools

By implementing appropriate observers that forward the events to these systems.

## Extending the System

You can extend the system by:

1. Adding new event types to the `AgentStateEvent` enum
2. Adding corresponding TypedDict models to `context/agent_events.py`
3. Emitting these events in appropriate places in the agent code
4. Creating observers that handle the new event types

This allows for a flexible and extensible monitoring system that can be adapted to various needs.
