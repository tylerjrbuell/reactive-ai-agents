#!/usr/bin/env python
"""
Example demonstrating the typed event subscription interface for ReactAgent.

This example shows how to:
1. Subscribe to specific event types with type-safe callbacks
2. Use both direct subscription methods and the events property
3. Filter events based on their content

Usage:
    python examples/typed_event_subscription_demo.py
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Set

from agents.react_agent import ReactAgent, ReactAgentConfig
from reactive_agents.providers.external.client import MCPClient
from context.agent_observer import AgentStateEvent
from context.agent_events import (
    MetricsUpdatedEventData,
    ToolCalledEventData,
    SessionStartedEventData,
    FinalAnswerSetEventData,
    ErrorOccurredEventData,
    TaskStatusChangedEventData,
)


class TypedEventMonitor:
    """
    A demonstration of typed event handling.

    This class shows how to use the typed event subscription interface
    to handle agent events with proper typing.
    """

    def __init__(self):
        """Initialize the monitor with tracking state"""
        self.start_time = time.time()
        self.session_id = None
        self.tool_usage: Dict[str, int] = {}
        self.status_changes: List[str] = []
        self.errors: List[Dict[str, Any]] = []

    def handle_session_start(self, event: SessionStartedEventData) -> None:
        """
        Handle session started events with proper typing.

        Args:
            event: The typed session started event data
        """
        self.session_id = event["session_id"]
        print(f"\n[{self._timestamp()}] 🚀 Session started: {self.session_id}")
        print(f"  Task: {event['initial_task']}")

    def handle_tool_call(self, event: ToolCalledEventData) -> None:
        """
        Handle tool called events with proper typing.

        Args:
            event: The typed tool called event data
        """
        tool_name = event["tool_name"]
        self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1

        # Access typed fields with confidence
        params_preview = (
            str(event["parameters"])[:50] + "..."
            if len(str(event["parameters"])) > 50
            else str(event["parameters"])
        )
        print(
            f"[{self._timestamp()}] 🔧 Tool call: {tool_name} (ID: {event['tool_id']})"
        )
        print(f"  Parameters: {params_preview}")

    def handle_status_change(self, event: TaskStatusChangedEventData) -> None:
        """
        Handle status change events with proper typing.

        Args:
            event: The typed status change event data
        """
        previous = event["previous_status"]
        new = event["new_status"]
        change = f"{previous} → {new}"
        self.status_changes.append(change)

        # Handle rescoping specially - TypedDict gives us confidence that these fields exist
        if new == "rescoped" and event.get("rescoped_task"):
            print(f"[{self._timestamp()}] 🔄 Task rescoped: {event['rescoped_task']}")
            if event.get("explanation"):
                print(f"  Reason: {event['explanation']}")
        else:
            print(f"[{self._timestamp()}] 📊 Status: {change}")

    def handle_error(self, event: ErrorOccurredEventData) -> None:
        """
        Handle error events with proper typing.

        Args:
            event: The typed error event data
        """
        error_info = {
            "error": event["error"],
            "details": event.get("details", "No details"),
            "timestamp": event["timestamp"],
        }
        self.errors.append(error_info)
        print(f"[{self._timestamp()}] ❌ Error: {event['error']}")
        print(f"  Details: {event.get('details', 'No details')}")

    def handle_final_answer(self, event: FinalAnswerSetEventData) -> None:
        """
        Handle final answer events with proper typing.

        Args:
            event: The typed final answer event data
        """
        answer = event["answer"]
        answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
        print(f"[{self._timestamp()}] ✅ Final answer: {answer_preview}")

    def handle_metrics_updated(self, event: MetricsUpdatedEventData) -> None:
        """
        Handle metrics updated events with proper typing.

        Args:
            event: The typed metrics updated event data
        """
        print(
            f"[{self._timestamp()}] 📊 Metrics updated: {json.dumps(event, indent=2)}"
        )

    def _timestamp(self) -> str:
        """Get a formatted timestamp for console output"""
        return datetime.now().strftime("%H:%M:%S")

    def print_summary(self) -> None:
        """Print a summary of the events received"""
        runtime = time.time() - self.start_time

        print("\n" + "=" * 60)
        print(f"EVENT MONITOR SUMMARY (Runtime: {runtime:.2f}s)")
        print("=" * 60)

        if self.tool_usage:
            print("\nTool Usage:")
            for tool, count in sorted(
                self.tool_usage.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  - {tool}: {count} calls")

        if self.status_changes:
            print("\nStatus Transitions:")
            for i, change in enumerate(self.status_changes, 1):
                print(f"  {i}. {change}")

        if self.errors:
            print("\nErrors Encountered:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error['error']}")
                print(f"     Details: {error['details']}")

        print("=" * 60)


async def main():
    """Main function demonstrating typed event subscriptions"""
    monitor = TypedEventMonitor()

    # Create an agent
    mcp_client = MCPClient(
        server_filter=["time", "sqlite", "brave-search"]
    )  # Not used in this example
    await mcp_client.initialize()
    agent_config = ReactAgentConfig(
        agent_name="TaskAgent",
        role="Task Execution Agent",
        provider_model_name="ollama:cogito:14b",  # Adjust based on your installation
        instructions="Solve the task as best as you can. Take verification steps whenever possible.",
        mcp_client=mcp_client,
        max_iterations=10,
        reflect_enabled=True,
        min_completion_score=1.0,
        log_level="info",
        tool_use_enabled=True,
        use_memory_enabled=True,
        collect_metrics_enabled=True,
        check_tool_feasibility=True,
        enable_caching=True,
        initial_task=None,  # Will be set in the run method
        confirmation_callback=None,
        kwargs={},
    )

    # Create the agent
    agent = ReactAgent(config=agent_config)

    print("Setting up typed event subscriptions...")

    # Method 1: Direct subscription methods with typed callbacks
    agent.on_session_started(monitor.handle_session_start)
    agent.on_tool_called(monitor.handle_tool_call)
    agent.on_final_answer_set(monitor.handle_final_answer)
    agent.on_metrics_updated(monitor.handle_metrics_updated)

    # Method 2: Using the events property with method chaining
    (agent.events.on_task_status_changed().subscribe(monitor.handle_status_change))

    # Method 3: Subscribe to events with a filter
    def handle_critical_error(event: ErrorOccurredEventData) -> None:
        """Handle only critical errors"""
        error_type = event["error"]
        if "critical" in error_type.lower():
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 CRITICAL ERROR: {error_type}"
            )

    # Register the filtered callback
    agent.events.on_error_occurred().subscribe(monitor.handle_error)
    agent.events.on_error_occurred().subscribe(handle_critical_error)

    # Execute a task
    print("\nStarting agent execution...")
    result = await agent.run(
        initial_task="Create a table called crypto_prices with the following columns: symbol, price, timestamp. Then lookup the current price of BTC and ETH and insert the results into the table."
    )

    # Wait a moment to ensure all events are processed
    await asyncio.sleep(1)

    # Print event summary
    monitor.print_summary()

    # Print final result
    print("\nFINAL RESULT:")
    print(f"Status: {result.get('status')}")
    print(f"Result: {result.get('result')}")

    # Cleanup
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
