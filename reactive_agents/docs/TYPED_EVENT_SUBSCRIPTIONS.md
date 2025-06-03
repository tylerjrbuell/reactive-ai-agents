# Typed Event Subscriptions for ReactAgent

This document explains how to use the typed event subscription interface for ReactAgent, which provides a type-safe way to observe agent events during execution.

## Overview

The typed event subscription interface enables you to:

1. Subscribe to specific agent events with proper TypeScript-like type checking
2. Use a fluent, method-chaining API for clean subscription syntax
3. Process events with confidence in their structure and available fields
4. Filter and transform events based on their content

## Basic Usage

Here's a simple example showing how to subscribe to events:

```python
from agents.react_agent import ReactAgent, ReactAgentConfig
from context.agent_events import SessionStartedEventData, ToolCalledEventData

# Create a ReactAgent
agent = ReactAgent(config=ReactAgentConfig(...))

# Method 1: Direct subscription with typed callback
def handle_session_start(event: SessionStartedEventData):
    print(f"Session started: {event['session_id']}")
    print(f"Task: {event['initial_task']}")

agent.on_session_started(handle_session_start)

# Method 2: Using the events property with method chaining
def handle_tool_call(event: ToolCalledEventData):
    print(f"Tool called: {event['tool_name']}")
    print(f"Parameters: {event['parameters']}")

agent.events.on_tool_called().subscribe(handle_tool_call)

# Run the agent
await agent.run(initial_task="Your task here")
```

## Available Event Types

The following event types are available:

| Event Type | Description | Data Type | Key Fields |
|------------|-------------|-----------|------------|
| `SESSION_STARTED` | When a new agent session begins | `SessionStartedEventData` | `session_id`, `initial_task` |
| `SESSION_ENDED` | When an agent session completes | `SessionEndedEventData` | `final_status`, `elapsed_time` |
| `TASK_STATUS_CHANGED` | When task status changes | `TaskStatusChangedEventData` | `previous_status`, `new_status`, `rescoped_task` |
| `ITERATION_STARTED` | When a new iteration begins | `IterationStartedEventData` | `iteration`, `max_iterations` |
| `ITERATION_COMPLETED` | When an iteration completes | `IterationCompletedEventData` | `iteration`, `has_result`, `has_plan` |
| `TOOL_CALLED` | When the agent calls a tool | `ToolCalledEventData` | `tool_name`, `tool_id`, `parameters` |
| `TOOL_COMPLETED` | When a tool execution completes | `ToolCompletedEventData` | `tool_name`, `tool_id`, `result`, `execution_time` |
| `TOOL_FAILED` | When a tool execution fails | `ToolFailedEventData` | `tool_name`, `tool_id`, `error`, `details` |
| `REFLECTION_GENERATED` | When the agent generates a reflection | `ReflectionGeneratedEventData` | `reason`, `next_step`, `required_tools` |
| `FINAL_ANSWER_SET` | When the agent sets a final answer | `FinalAnswerSetEventData` | `answer` |
| `METRICS_UPDATED` | When agent metrics are updated | `MetricsUpdatedEventData` | `metrics` |
| `ERROR_OCCURRED` | When an error occurs during execution | `ErrorOccurredEventData` | `error`, `details` |

All event types inherit from `BaseEventData`, which provides common fields:

```python
class BaseEventData(TypedDict):
    timestamp: float
    event_type: str
    agent_name: str
    session_id: str
    task: Optional[str]
    task_status: str
    iterations: int
```

## Subscription Methods

### Direct Methods

ReactAgent provides direct subscription methods for each event type:

```python
agent.on_session_started(callback)
agent.on_session_ended(callback)
agent.on_task_status_changed(callback)
agent.on_iteration_started(callback)
agent.on_iteration_completed(callback)
agent.on_tool_called(callback)
agent.on_tool_completed(callback)
agent.on_tool_failed(callback)
agent.on_reflection_generated(callback)
agent.on_final_answer_set(callback)
agent.on_metrics_updated(callback)
agent.on_error_occurred(callback)
```

### Fluent API via `events` Property

You can also use the `events` property for method chaining:

```python
# Subscribe to tool called events
(agent.events
    .on_tool_called()
    .subscribe(handle_tool_call))

# Subscribe to multiple event types
tool_subscription = agent.events.on_tool_called()
tool_subscription.subscribe(handle_tool_call)
tool_subscription.subscribe(log_tool_call)
```

## Filtering Events

You can easily filter events by implementing filter logic in your callback:

```python
def handle_critical_errors(event: ErrorOccurredEventData):
    """Only handle critical errors"""
    if "critical" in event["error"].lower():
        print(f"CRITICAL ERROR: {event['error']}")
        log_critical_error(event)

agent.on_error_occurred(handle_critical_errors)
```

## Async Event Handling

You can use async callbacks for event handling:

```python
async def handle_tool_result(event: ToolCompletedEventData):
    """Process tool results asynchronously"""
    # Do some async processing
    await store_result_in_database(event["result"])
    await notify_external_service(event)

# Register async callback
await agent.events.on_tool_completed().subscribe_async(handle_tool_result)
```

## Unsubscribing

To unsubscribe all callbacks for a specific event type:

```python
# Unsubscribe all callbacks from tool called events
agent.events.on_tool_called().unsubscribe_all()
```

## Integration with Monitoring Systems

You can integrate the event system with external monitoring tools:

```python
class PrometheusMonitor:
    def __init__(self, agent):
        self.metrics = {}
        
        # Register for all event types
        agent.on_tool_called(self.handle_tool_call)
        agent.on_session_ended(self.handle_session_ended)
        # ...
    
    def handle_tool_call(self, event: ToolCalledEventData):
        tool_name = event["tool_name"]
        self.metrics[f"tool_calls_{tool_name}"] = self.metrics.get(f"tool_calls_{tool_name}", 0) + 1
        # Update Prometheus metrics
        # ...
```

## Example: Real-Time Monitoring Dashboard

See the `examples/typed_event_subscription_demo.py` file for a complete example of monitoring agent events with the typed subscription interface. 