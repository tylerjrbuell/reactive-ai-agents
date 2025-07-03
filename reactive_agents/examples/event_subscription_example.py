#!/usr/bin/env python3
"""
Event Subscription Example

This example demonstrates two ways to subscribe to agent events:
1. Using the ReactiveAgentBuilder with dynamic subscription methods
2. Using the agent.events interface for existing agent instances

Both approaches provide a clean and extensible way to work with events:

For the builder pattern:
    builder.with_subscription(AgentStateEvent.TOOL_CALLED, callback)
    builder.on_tool_called(callback)  # Type-specific helper

For existing agent instances:
    agent.events(AgentStateEvent.TOOL_CALLED).subscribe(callback)
    agent.events.on_tool_called().subscribe(callback)  # Type-specific helper
"""

import asyncio
from typing import Dict, Any

from context.agent_observer import AgentStateEvent
from agents.builders import ReactiveAgentBuilder
from context.agent_events import (
    ToolCalledEventData,
    ToolCompletedEventData,
    SessionStartedEventData,
)


# Define some callback functions for our events
def log_tool_called(event: ToolCalledEventData) -> None:
    """Log when a tool is called"""
    print(f"🔧 Tool Called: {event['tool_name']}")
    print(f"   Parameters: {event['parameters']}")


def log_tool_completed(event: ToolCompletedEventData) -> None:
    """Log when a tool completes"""
    print(f"✅ Tool Completed: {event['tool_name']}")
    print(f"   Result: {event['result']}")
    print(f"   Time: {event['execution_time']:.2f}s")


def log_session_started(event: SessionStartedEventData) -> None:
    """Log when a session starts"""
    print(f"🚀 Session Started: {event['session_id']}")
    print(f"   Task: {event['initial_task']}")


# Generic event logger for any event type
def log_any_event(event: Dict[str, Any]) -> None:
    """Generic event logger that can be used for any event type"""
    event_type = event.get("event_type", "unknown")
    print(f"📝 Event: {event_type}")
    # Print a few key fields if they exist
    for key in ["agent_name", "task", "iteration", "tool_name", "error"]:
        if key in event:
            print(f"   {key}: {event[key]}")


async def example_builder_subscription():
    """Example using the builder pattern with dynamic subscriptions"""
    print("\n=== Example 1: Builder Pattern with Dynamic Subscriptions ===")

    # Create an agent using the builder pattern with dynamic subscriptions
    agent = (
        await (
            ReactiveAgentBuilder()
            .with_name("Subscription Demo Agent")
            .with_model("ollama:qwen2:7b")
            .with_mcp_tools(["brave-search", "time"])
            # Use the new dynamic subscription methods
            .with_subscription(AgentStateEvent.TOOL_CALLED, log_tool_called)
            .with_subscription(AgentStateEvent.TOOL_COMPLETED, log_tool_completed)
            # Subscribe to session events with the specific helper methods
            .on_session_started(log_session_started)
            # Use the generic log function for multiple events
            .with_subscription(AgentStateEvent.ERROR_OCCURRED, log_any_event)
            .with_subscription(AgentStateEvent.METRICS_UPDATED, log_any_event)
            .build()
        )
    )

    # Run the agent
    try:
        result = await agent.run(initial_task="What time is it in New York and Tokyo?")
        print("\nAgent completed with result:")
        print(f"Status: {result['status']}")
        print(f"Result: {result['result']}")
    finally:
        await agent.close()


async def example_existing_agent_subscription():
    """Example using the events interface on an existing agent instance"""
    print("\n=== Example 2: Using events Interface on Existing Agent ===")

    # Create a basic agent without subscriptions
    agent = (
        await ReactiveAgentBuilder().with_mcp_tools(["brave-search", "time"]).build()
    )

    # Add event subscriptions using the events interface
    agent.events.on_tool_called().subscribe(log_tool_called)
    agent.events.on_tool_completed().subscribe(log_tool_completed)
    agent.events.on_session_started().subscribe(log_session_started)

    # Use the cleaner callable interface for dynamic subscription
    agent.events(AgentStateEvent.ERROR_OCCURRED).subscribe(log_any_event)
    agent.events(AgentStateEvent.METRICS_UPDATED).subscribe(log_any_event)

    # Run the agent
    try:
        result = await agent.run(initial_task="What is the current Bitcoin price?")
        print("\nAgent completed with result:")
        print(f"Status: {result['status']}")
        print(f"Result: {result['result']}")
    finally:
        await agent.close()


async def main():
    """Run the examples"""
    await example_builder_subscription()
    await example_existing_agent_subscription()


if __name__ == "__main__":
    asyncio.run(main())
