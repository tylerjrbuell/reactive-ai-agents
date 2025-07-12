#!/usr/bin/env python3
"""
Type Inference Demo for ReactiveAgent Dynamic Event Methods

This demo shows how the enhanced type system provides precise type inference
for dynamic event handler methods, ensuring callbacks receive the correct
event data types with full IntelliSense support.
"""

from typing import TYPE_CHECKING
from reactive_agents.core.types.agent_protocol_types import EventHandlerProtocol
from reactive_agents.core.types.event_mapping_types import (
    get_available_event_methods,
    get_event_categories,
    get_events_by_category,
    is_valid_event_method,
    get_method_data_type,
)

if TYPE_CHECKING:
    # Import for type checking only
    from reactive_agents.app.agents.reactive_agent import ReactiveAgent
    from reactive_agents.core.events.agent_events import (
        SessionStartedEventData,
        ToolCalledEventData,
        ErrorOccurredEventData,
        PausedEventData,
    )


def demonstrate_type_inference():
    """
    Demonstrate how the type system provides precise type inference
    for dynamic event handler methods.
    """
    print("=== ReactiveAgent Dynamic Event Method Type Inference Demo ===\n")

    # This would be a real agent instance in practice
    # agent: ReactiveAgent = create_agent()

    print("1. Available Event Methods:")
    methods = get_available_event_methods()
    for method in sorted(methods):
        data_type = get_method_data_type(method)
        print(f"   {method} -> {data_type.__name__ if data_type else 'Unknown'}")

    print(f"\n   Total: {len(methods)} event methods available\n")

    print("2. Event Categories:")
    categories = get_event_categories()
    for cat_name, cat_info in categories.items():
        print(f"   {cat_info['name']}: {cat_info['description']}")
        events = get_events_by_category(cat_name)
        for event in events:
            print(f"      - on_{event}")

    print("\n3. Type-Safe Event Handler Examples:")
    print("   (These would provide full IntelliSense support in IDEs)\n")

    # Example 1: Session started event
    print("   # Session Started Event Handler")
    print("   def handle_session_start(event: SessionStartedEventData) -> None:")
    print("       # IDE knows 'event' has these fields:")
    print("       #   - initial_task: str")
    print("       #   - timestamp: float")
    print("       #   - agent_name: str")
    print("       #   - session_id: str")
    print("       print(f'Session started with task: {event[\"initial_task\"]}')")
    print("   ")
    print("   # agent.on_session_started(handle_session_start)")
    print()

    # Example 2: Tool called event
    print("   # Tool Called Event Handler")
    print("   def handle_tool_call(event: ToolCalledEventData) -> None:")
    print("       # IDE knows 'event' has these fields:")
    print("       #   - tool_name: str")
    print("       #   - tool_id: str")
    print("       #   - parameters: Dict[str, Any]")
    print("       #   - timestamp: float")
    print(
        '       print(f\'Tool called: {event["tool_name"]} with params: {event["parameters"]}\')'
    )
    print("   ")
    print("   # agent.on_tool_called(handle_tool_call)")
    print()

    # Example 3: Error occurred event
    print("   # Error Occurred Event Handler")
    print("   def handle_error(event: ErrorOccurredEventData) -> None:")
    print("       # IDE knows 'event' has these fields:")
    print("       #   - error: str")
    print("       #   - details: Optional[str]")
    print("       #   - timestamp: float")
    print("       print(f'Error occurred: {event[\"error\"]}')")
    print("       if event.get('details'):")
    print("           print(f'Details: {event[\"details\"]}')")
    print("   ")
    print("   # agent.on_error_occurred(handle_error)")
    print()

    # Example 4: Control event
    print("   # Paused Event Handler")
    print("   def handle_pause(event: PausedEventData) -> None:")
    print("       # IDE knows 'event' has these fields:")
    print("       #   - timestamp: float")
    print("       #   - agent_name: str")
    print("       #   - session_id: str")
    print(
        '       print(f\'Agent {event["agent_name"]} paused at {event["timestamp"]}\')'
    )
    print("   ")
    print("   # agent.on_paused(handle_pause)")
    print()

    print("4. Method Validation:")
    test_methods = [
        "on_session_started",
        "on_tool_called",
        "on_invalid_event",
        "not_an_event_method",
    ]

    for method in test_methods:
        is_valid = is_valid_event_method(method)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"   {method}: {status}")

    print("\n5. Benefits of This Type System:")
    print("   ✓ Precise type inference for each event method")
    print("   ✓ Full IntelliSense support in IDEs")
    print("   ✓ Compile-time type checking")
    print("   ✓ Runtime validation and error messages")
    print("   ✓ Preserves dynamic nature of event system")
    print("   ✓ Auto-completion shows available methods")
    print("   ✓ Type safety without sacrificing flexibility")

    print("\n=== Demo Complete ===")


def demonstrate_protocol_usage():
    """
    Demonstrate how the EventHandlerProtocol can be used for typing.
    """
    print("\n=== Protocol Usage Demo ===\n")

    def work_with_event_handler(handler: EventHandlerProtocol) -> None:
        """
        Function that accepts any object implementing the event handler protocol.

        Args:
            handler: Object that implements EventHandlerProtocol
        """
        print("Working with event handler...")

        # The protocol ensures these methods exist and are properly typed
        available_events = handler.get_available_events()
        print(f"Handler supports {len(available_events)} event types")

        # Example callback that would be properly typed
        def sample_callback(event) -> None:
            print(f"Received event: {event.get('event_type')}")

        # This would work with proper type inference
        # handler.on_session_started(sample_callback)

    print("   Protocol ensures type safety for event handler objects")
    print("   while preserving dynamic method creation.")


if __name__ == "__main__":
    demonstrate_type_inference()
    demonstrate_protocol_usage()
