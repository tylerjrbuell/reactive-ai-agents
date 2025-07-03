# Dynamic Event Method Typing System

## Overview

The ReactiveAgent framework provides a sophisticated type system for dynamic event handler methods that combines the flexibility of runtime method creation with the safety and developer experience of static typing. This system enables precise type inference for each event method while preserving the dynamic nature of the event system.

## Key Features

### ✅ **Precise Type Inference**

Each dynamic event method (`on_<event_name>`) is precisely typed to ensure callbacks receive the correct event data type.

### ✅ **Full IntelliSense Support**

IDEs provide complete auto-completion, parameter hints, and type checking for all dynamic methods.

### ✅ **Runtime Validation**

Invalid event methods are caught at runtime with helpful error messages.

### ✅ **Preserves Dynamic Nature**

Methods are still created dynamically - no loss of flexibility.

### ✅ **Type Safety**

Compile-time type checking ensures callback functions match expected signatures.

## Architecture

The type system consists of several key components:

### 1. Type Stub File (`.pyi`)

- **File**: `reactive_agents/app/agents/reactive_agent.pyi`
- **Purpose**: Provides type information for the ReactiveAgent class
- **Features**:
  - Overloaded `__getattr__` methods with `Literal` types
  - Precise return type annotations for each event method
  - Comprehensive docstrings with examples

### 2. Protocol Definitions

- **File**: `reactive_agents/core/types/agent_protocols.py`
- **Purpose**: Defines the `EventHandlerProtocol` for type checking
- **Features**:
  - Protocol-based typing for event handler interfaces
  - Flexible typing that works with any implementation
  - Preserves dynamic method creation

### 3. Event Mappings

- **File**: `reactive_agents/core/types/event_mappings.py`
- **Purpose**: Provides type mappings and utilities
- **Features**:
  - Complete mapping of event names to data types
  - Validation utilities
  - Event categorization
  - Runtime type checking helpers

### 4. Event Data Types

- **File**: `reactive_agents/core/events/agent_events.py`
- **Purpose**: Defines all event data structures
- **Features**:
  - `TypedDict` definitions for each event type
  - Generic `EventCallback` protocol
  - Base event data structure

## Usage Examples

### Basic Event Handler Registration

```python
from reactive_agents.app.agents.reactive_agent import ReactiveAgent
from reactive_agents.core.events.agent_events import SessionStartedEventData

# Create agent
agent = ReactiveAgent(config)

# Define typed callback - IDE knows the exact type
def handle_session_start(event: SessionStartedEventData) -> None:
    # Full IntelliSense support for event fields
    print(f"Session started: {event['initial_task']}")
    print(f"Agent: {event['agent_name']}")
    print(f"Session ID: {event['session_id']}")

# Register callback - method is dynamically created with proper typing
agent.on_session_started(handle_session_start)
```

### Tool Event Handling

```python
from reactive_agents.core.events.agent_events import ToolCalledEventData, ToolCompletedEventData

def handle_tool_call(event: ToolCalledEventData) -> None:
    # IDE provides auto-completion for all fields
    print(f"Tool called: {event['tool_name']}")
    print(f"Parameters: {event['parameters']}")
    print(f"Tool ID: {event['tool_id']}")

def handle_tool_completion(event: ToolCompletedEventData) -> None:
    print(f"Tool completed: {event['tool_name']}")
    print(f"Result: {event['result']}")
    print(f"Execution time: {event['execution_time']}s")

# Both methods are precisely typed
agent.on_tool_called(handle_tool_call)
agent.on_tool_completed(handle_tool_completion)
```

### Error Handling

```python
from reactive_agents.core.events.agent_events import ErrorOccurredEventData

def handle_error(event: ErrorOccurredEventData) -> None:
    print(f"Error: {event['error']}")
    if event.get('details'):
        print(f"Details: {event['details']}")

    # Access base event fields
    print(f"Timestamp: {event['timestamp']}")
    print(f"Agent: {event['agent_name']}")

agent.on_error_occurred(handle_error)
```

### Control Events

```python
from reactive_agents.core.events.agent_events import PausedEventData, ResumedEventData

def handle_pause(event: PausedEventData) -> None:
    print(f"Agent paused at {event['timestamp']}")

def handle_resume(event: ResumedEventData) -> None:
    print(f"Agent resumed at {event['timestamp']}")

agent.on_paused(handle_pause)
agent.on_resumed(handle_resume)
```

### Generic Event Handler

```python
from reactive_agents.core.types.agent_protocols import EventData

def handle_any_event(event: EventData) -> None:
    """Handle any event type - use event_type to determine specific handling"""
    event_type = event['event_type']

    if event_type == 'session_started':
        # Type narrowing - IDE knows this is SessionStartedEventData
        print(f"Session started: {event['initial_task']}")
    elif event_type == 'tool_called':
        # Type narrowing - IDE knows this is ToolCalledEventData
        print(f"Tool called: {event['tool_name']}")
    elif event_type == 'error_occurred':
        # Type narrowing - IDE knows this is ErrorOccurredEventData
        print(f"Error: {event['error']}")

# Subscribe to all events
agent.subscribe_to_all_events(handle_any_event)
```

## Available Event Methods

### Session Events

- `on_session_started(callback: EventCallback[SessionStartedEventData])`
- `on_session_ended(callback: EventCallback[SessionEndedEventData])`

### Task Events

- `on_task_status_changed(callback: EventCallback[TaskStatusChangedEventData])`

### Iteration Events

- `on_iteration_started(callback: EventCallback[IterationStartedEventData])`
- `on_iteration_completed(callback: EventCallback[IterationCompletedEventData])`

### Tool Events

- `on_tool_called(callback: EventCallback[ToolCalledEventData])`
- `on_tool_completed(callback: EventCallback[ToolCompletedEventData])`
- `on_tool_failed(callback: EventCallback[ToolFailedEventData])`

### Reflection Events

- `on_reflection_generated(callback: EventCallback[ReflectionGeneratedEventData])`
- `on_final_answer_set(callback: EventCallback[FinalAnswerSetEventData])`

### Metrics Events

- `on_metrics_updated(callback: EventCallback[MetricsUpdatedEventData])`

### Error Events

- `on_error_occurred(callback: EventCallback[ErrorOccurredEventData])`

### Control Events

- `on_pause_requested(callback: EventCallback[PauseRequestedEventData])`
- `on_paused(callback: EventCallback[PausedEventData])`
- `on_resume_requested(callback: EventCallback[ResumeRequestedEventData])`
- `on_resumed(callback: EventCallback[ResumedEventData])`
- `on_stop_requested(callback: EventCallback[StopRequestedEventData])`
- `on_stopped(callback: EventCallback[StoppedEventData])`
- `on_terminate_requested(callback: EventCallback[TerminateRequestedEventData])`
- `on_terminated(callback: EventCallback[TerminatedEventData])`
- `on_cancelled(callback: EventCallback[CancelledEventData])`

## Event Data Structures

All event data structures inherit from `BaseEventData` and include these common fields:

```python
class BaseEventData(TypedDict):
    timestamp: float          # Unix timestamp when event occurred
    event_type: str          # Event type identifier
    agent_name: str          # Name of the agent
    session_id: str          # Session identifier
    task: Optional[str]      # Current task (if applicable)
    task_status: str         # Current task status
    iterations: int          # Current iteration count
```

### Specific Event Data Types

Each event type extends `BaseEventData` with additional fields:

- **SessionStartedEventData**: `initial_task: str`
- **SessionEndedEventData**: `final_status: str`, `elapsed_time: float`
- **ToolCalledEventData**: `tool_name: str`, `tool_id: str`, `parameters: Dict[str, Any]`
- **ToolCompletedEventData**: `tool_name: str`, `tool_id: str`, `result: Any`, `execution_time: float`
- **ErrorOccurredEventData**: `error: str`, `details: Optional[str]`

## Utilities

### Type Validation

```python
from reactive_agents.core.types.event_mappings import (
    is_valid_event_method,
    get_method_data_type,
    get_available_event_methods
)

# Check if method is valid
is_valid = is_valid_event_method("on_session_started")  # True
is_valid = is_valid_event_method("on_invalid_event")    # False

# Get data type for method
data_type = get_method_data_type("on_session_started")  # SessionStartedEventData

# Get all available methods
methods = get_available_event_methods()  # List of all method names
```

### Event Categories

```python
from reactive_agents.core.types.event_mappings import (
    get_event_categories,
    get_events_by_category
)

# Get all categories
categories = get_event_categories()

# Get events in a specific category
tool_events = get_events_by_category("tool")  # ['tool_called', 'tool_completed', 'tool_failed']
```

## IDE Support

### VSCode

- Full IntelliSense support for dynamic methods
- Parameter hints show exact event data types
- Auto-completion includes all available event methods
- Type checking catches callback signature mismatches

### PyCharm

- Complete type inference for dynamic methods
- Code completion with type information
- Inspection warnings for invalid event methods
- Refactoring support for callback functions

### Other IDEs

- Any IDE supporting Python type stubs (`.pyi` files) will provide enhanced support
- Language servers like Pylsp and Pyright provide full type checking

## Best Practices

### 1. Use Specific Event Types

```python
# Good - specific type
def handle_session_start(event: SessionStartedEventData) -> None:
    print(event['initial_task'])

# Avoid - generic type
def handle_session_start(event: Dict[str, Any]) -> None:
    print(event['initial_task'])  # No type safety
```

### 2. Handle Optional Fields

```python
def handle_error(event: ErrorOccurredEventData) -> None:
    print(f"Error: {event['error']}")

    # Use .get() for optional fields
    if event.get('details'):
        print(f"Details: {event['details']}")
```

### 3. Use Type Narrowing for Generic Handlers

```python
def handle_any_event(event: EventData) -> None:
    if event['event_type'] == 'session_started':
        # Type checker knows this is SessionStartedEventData
        assert isinstance(event, dict)  # Runtime check if needed
        print(f"Task: {event['initial_task']}")
```

### 4. Validate Event Methods

```python
from reactive_agents.core.types.event_mappings import is_valid_event_method

method_name = "on_session_started"
if is_valid_event_method(method_name):
    # Safe to use
    handler = getattr(agent, method_name)
    handler(my_callback)
```

## Technical Implementation

### Overload Resolution

The type system uses `@overload` decorators with `Literal` types to provide precise type inference:

```python
@overload
def __getattr__(self, name: Literal["on_session_started"]) -> Callable[[EventCallback[SessionStartedEventData]], Any]: ...

@overload
def __getattr__(self, name: Literal["on_tool_called"]) -> Callable[[EventCallback[ToolCalledEventData]], Any]: ...
```

### Runtime Method Creation

The actual implementation in `ReactiveAgent.__getattr__` creates methods dynamically:

```python
def __getattr__(self, name: str):
    if name.startswith("on_"):
        event_name = name[3:]  # Remove 'on_' prefix

        # Validate event type
        try:
            event_type = AgentStateEvent(event_name)
        except ValueError:
            raise AttributeError(f"Invalid event method: {name}")

        # Return callback registration function
        def register_callback(callback):
            return self.event_manager.register_callback(event_type, callback)

        return register_callback
```

## Migration Guide

### From Generic Callbacks

```python
# Old approach - no type safety
def my_callback(event: dict) -> None:
    print(event.get('tool_name'))  # No IntelliSense

agent.on_tool_called(my_callback)

# New approach - full type safety
def my_callback(event: ToolCalledEventData) -> None:
    print(event['tool_name'])  # Full IntelliSense

agent.on_tool_called(my_callback)
```

### From Manual Event Registration

```python
# Old approach - manual registration
agent.event_manager.register_callback(AgentStateEvent.TOOL_CALLED, my_callback)

# New approach - typed dynamic methods
agent.on_tool_called(my_callback)
```

## Performance Considerations

- **Type Checking**: Happens at development time, no runtime overhead
- **Method Creation**: Dynamic methods are created on first access and cached
- **Memory Usage**: Minimal overhead from type stubs and protocols
- **Runtime Validation**: Only validates method names, not callback signatures

## Troubleshooting

### Common Issues

1. **Method Not Found**: Ensure event name is valid

   ```python
   # Check valid methods
   from reactive_agents.core.types.event_mappings import get_available_event_methods
   print(get_available_event_methods())
   ```

2. **Type Errors**: Ensure callback signature matches event data type

   ```python
   # Correct signature
   def callback(event: SessionStartedEventData) -> None: ...

   # Incorrect signature
   def callback(event: str) -> None: ...  # Type error
   ```

3. **IntelliSense Not Working**: Ensure IDE supports Python type stubs
   - Check that `.pyi` files are being recognized
   - Restart language server if needed

### Debug Mode

Enable debug logging to see event method creation:

```python
import logging
logging.getLogger('reactive_agents').setLevel(logging.DEBUG)
```

## Future Enhancements

- **Async Callback Support**: Enhanced typing for async event handlers
- **Event Filtering**: Type-safe event filtering and transformation
- **Custom Event Types**: Support for user-defined event types
- **Performance Monitoring**: Type-aware performance metrics for event handlers

## Benefits

- ✅ **Precise type inference** for each event method
- ✅ **Full IntelliSense support** in IDEs
- ✅ **Compile-time type checking**
- ✅ **Runtime validation** and error messages
- ✅ **Preserves dynamic nature** of event system
- ✅ **Auto-completion** shows available methods
- ✅ **Type safety** without sacrificing flexibility

## Technical Implementation

The type system uses several key components:

1. **Type Stub File** (`.pyi`) - Provides overloaded `__getattr__` methods with precise typing
2. **Protocol Definitions** - Defines the `EventHandlerProtocol` interface
3. **Event Mappings** - Maps event names to data types with validation utilities
4. **Event Data Types** - `TypedDict` definitions for each event type

This approach provides the best of both worlds: the flexibility of dynamic method creation with the safety and developer experience of static typing.
