"""
Context Observability Demo

This demo shows how to use the integrated context observability system
with the enhanced AgentStateObserver for real-time monitoring of agent context management.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from reactive_agents.core.context.agent_context import AgentContext
from reactive_agents.core.events.agent_observer import AgentStateObserver
from reactive_agents.core.types.event_types import AgentStateEvent
from reactive_agents.core.types.session_types import AgentSession
from reactive_agents.core.types.status_types import TaskStatus


class ContextMonitoringObserver:
    """Observer for monitoring context-related events in real-time."""

    def __init__(self):
        self.context_events = []
        self.operation_events = []
        self.token_events = []
        self.snapshot_events = []

    def handle(self, event: Dict[str, Any]) -> None:
        """Handle incoming events and categorize them."""
        event_type = event.get("event_type")

        if event_type == AgentStateEvent.CONTEXT_CHANGED.value:
            self.handle_context_changed(event)
        elif event_type == AgentStateEvent.OPERATION_COMPLETED.value:
            self.handle_operation_completed(event)
        elif event_type == AgentStateEvent.TOKENS_USED.value:
            self.handle_tokens_used(event)
        elif event_type == AgentStateEvent.SNAPSHOT_TAKEN.value:
            self.handle_snapshot_taken(event)

    def handle_context_changed(self, event_data: Dict[str, Any]):
        """Handle context change events."""
        self.context_events.append(event_data)
        print(
            f"🔄 Context changed: {event_data['change_type']} "
            f"({event_data['message_count_before']}→{event_data['message_count_after']} messages, "
            f"{event_data['token_estimate_before']}→{event_data['token_estimate_after']} tokens)"
        )

    def handle_operation_completed(self, event_data: Dict[str, Any]):
        """Handle operation completion events."""
        self.operation_events.append(event_data)
        print(
            f"⚡ Operation completed: {event_data['operation_name']} "
            f"({event_data['duration']:.3f}s)"
        )

    def handle_tokens_used(self, event_data: Dict[str, Any]):
        """Handle token usage events."""
        self.token_events.append(event_data)
        print(
            f"💾 Tokens used: {event_data['tokens_used']} "
            f"for {event_data['operation_type']} "
            f"(efficiency: {event_data['efficiency_ratio']:.2f})"
        )

    def handle_snapshot_taken(self, event_data: Dict[str, Any]):
        """Handle snapshot events."""
        self.snapshot_events.append(event_data)
        print(f"📸 Snapshot taken: {event_data['snapshot_type']}")


async def setup_context_with_observability():
    """Set up context with integrated observability."""

    # Create enhanced observer with context observability
    observer = AgentStateObserver()

    # Create context monitoring observer
    context_observer = ContextMonitoringObserver()

    # Register context monitoring callbacks
    observer.register_callback(
        AgentStateEvent.CONTEXT_CHANGED, context_observer.handle_context_changed
    )
    observer.register_callback(
        AgentStateEvent.OPERATION_COMPLETED, context_observer.handle_operation_completed
    )
    observer.register_callback(
        AgentStateEvent.TOKENS_USED, context_observer.handle_tokens_used
    )
    observer.register_callback(
        AgentStateEvent.SNAPSHOT_TAKEN, context_observer.handle_snapshot_taken
    )

    # Create agent context with observability
    context = AgentContext(
        agent_name="ObservableAgent",
        provider_model_name="openai:gpt-3.5-turbo",
        instructions="You are an observable agent for testing context management.",
        role="Context Test Agent",
        tool_use_enabled=True,
        reflect_enabled=True,
        use_memory_enabled=True,
        collect_metrics_enabled=True,
        enable_state_observation=True,
        reasoning_strategy="plan_execute_reflect",
        enable_reactive_execution=True,
        enable_dynamic_strategy_switching=True,
        max_context_messages=15,
        context_token_budget=3000,
        context_pruning_aggressiveness="balanced",
        context_summarization_frequency=3,
        state_observer=observer,  # Use the enhanced observer
    )

    return context, context_observer


async def run_observability_demo():
    """Run the context observability demonstration."""

    print("🚀 Starting Context Observability Demo")
    print("=" * 50)

    # Set up context with observability
    context, observer = await setup_context_with_observability()

    print(f"\n📋 Context initialized with integrated observability")
    print("-" * 50)

    try:
        # Simulate context management operations
        print("\n🔄 Simulating context management operations...")

        # Add some messages to trigger context management
        context.session.messages.extend(
            [
                {"role": "user", "content": "Test message 1"},
                {"role": "assistant", "content": "Test response 1"},
                {"role": "user", "content": "Test message 2"},
                {"role": "assistant", "content": "Test response 2"},
            ]
        )

        # Track context change manually using the observer
        if context.state_observer:
            context.state_observer.track_context_change(
                change_type="message_added",
                message_count_before=0,
                message_count_after=4,
                token_estimate_before=0,
                token_estimate_after=150,
                operation_duration=0.1,
                additional_data={"source": "demo"},
            )

        # Trigger context management
        await context.manage_context()

        # Add more messages to trigger summarization
        for i in range(10):
            context.session.messages.extend(
                [
                    {"role": "user", "content": f"Message {i+3}"},
                    {"role": "assistant", "content": f"Response {i+3}"},
                ]
            )

            # Track operation timing
            if context.state_observer:
                context.state_observer.track_operation(
                    operation_name="context_management",
                    duration=0.05 + (i * 0.01),
                    success=True,
                    metadata={"iteration": i},
                )

                # Track token usage
                context.state_observer.track_token_usage(
                    tokens_used=50 + (i * 10),
                    operation_type="context_processing",
                    efficiency_ratio=0.9 - (i * 0.02),
                    context_size=len(context.session.messages),
                )

            await context.manage_context()

        print("\n" + "=" * 50)
        print("📊 CONTEXT OBSERVABILITY RESULTS")
        print("=" * 50)

        # Get context insights from the enhanced observer
        insights = (
            context.state_observer.get_context_insights()
            if context.state_observer
            else {}
        )

        print(f"\n🔍 Context Changes:")
        print(f"  - Total changes: {insights['context_changes']['total_changes']}")
        print(
            f"  - Total tokens used: {insights['context_changes']['total_tokens_used']}"
        )
        print(
            f"  - Average message change: {insights['context_changes']['avg_message_change']:.1f}"
        )

        print(f"\n⏱️  Operation Timings:")
        for operation, stats in insights["operation_timings"].items():
            print(
                f"  - {operation}: {stats['count']} operations, "
                f"avg {stats['avg_duration']:.3f}s"
            )

        print(f"\n💾 Token Usage:")
        if insights["token_usage"]:
            print(f"  - Total tokens: {insights['token_usage']['total_tokens_used']}")
            print(
                f"  - Average efficiency: {insights['token_usage']['avg_efficiency_ratio']:.2f}"
            )

        print(f"\n📸 Snapshots taken: {insights['snapshots_count']}")

        # Export debug data
        debug_data = (
            context.state_observer.export_debug_data() if context.state_observer else {}
        )

        print(
            f"\n💾 Debug data exported with {len(debug_data['context_history'])} context changes"
        )

        # Show recent context changes
        print(f"\n🔄 Recent Context Changes:")
        for i, change in enumerate(debug_data["context_history"][-5:], 1):
            print(
                f"  {i}. {change['change_type']}: "
                f"{change['message_count_before']}→{change['message_count_after']} messages, "
                f"{change['token_estimate_before']}→{change['token_estimate_after']} tokens"
            )

        # Show event summary
        print(f"\n📡 Event Summary:")
        print(f"  - Context changes: {len(observer.context_events)}")
        print(f"  - Operations: {len(observer.operation_events)}")
        print(f"  - Token usage: {len(observer.token_events)}")
        print(f"  - Snapshots: {len(observer.snapshot_events)}")

        return insights, debug_data

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return None, None


async def analyze_context_efficiency(debug_data: Dict[str, Any]):
    """Analyze context efficiency from debug data."""

    print("\n" + "=" * 50)
    print("🔬 CONTEXT EFFICIENCY ANALYSIS")
    print("=" * 50)

    if not debug_data or "context_history" not in debug_data:
        print("❌ No debug data available for analysis")
        return

    context_history = debug_data["context_history"]

    # Analyze context change patterns
    change_types = {}
    token_deltas = []
    message_deltas = []

    for change in context_history:
        change_type = change["change_type"]
        change_types[change_type] = change_types.get(change_type, 0) + 1

        token_delta = change["token_estimate_after"] - change["token_estimate_before"]
        token_deltas.append(token_delta)

        message_delta = change["message_count_after"] - change["message_count_before"]
        message_deltas.append(message_delta)

    print(f"\n📊 Change Type Distribution:")
    for change_type, count in change_types.items():
        percentage = (count / len(context_history)) * 100
        print(f"  - {change_type}: {count} ({percentage:.1f}%)")

    if token_deltas:
        avg_token_delta = sum(token_deltas) / len(token_deltas)
        print(f"\n💾 Token Usage Analysis:")
        print(f"  - Average token delta: {avg_token_delta:+.0f}")
        print(f"  - Total token change: {sum(token_deltas):+.0f}")
        print(
            f"  - Token efficiency: {'Good' if avg_token_delta < 0 else 'Needs optimization'}"
        )

    if message_deltas:
        avg_message_delta = sum(message_deltas) / len(message_deltas)
        print(f"\n📝 Message Count Analysis:")
        print(f"  - Average message delta: {avg_message_delta:+.1f}")
        print(f"  - Total message change: {sum(message_deltas):+.0f}")
        print(
            f"  - Context management: {'Efficient' if abs(avg_message_delta) < 5 else 'Aggressive'}"
        )


async def main():
    """Main demo function."""

    print("🎯 Integrated Context Observability System Demo")
    print("This demo shows real-time monitoring of agent context management")
    print("using the enhanced AgentStateObserver with integrated observability.\n")

    # Run the demo
    insights, debug_data = await run_observability_demo()

    if debug_data:
        # Analyze context efficiency
        await analyze_context_efficiency(debug_data)

        # Save debug data to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"context_debug_data_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(debug_data, f, indent=2, default=str)

        print(f"\n💾 Debug data saved to: {filename}")

    print("\n✅ Integrated Context Observability Demo Complete!")


if __name__ == "__main__":
    asyncio.run(main())
