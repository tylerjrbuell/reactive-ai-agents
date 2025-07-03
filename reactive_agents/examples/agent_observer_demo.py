#!/usr/bin/env python
"""
Example demonstrating how to use the AgentStateObserver to track agent state in real-time.

This example shows how to:
1. Create a ReactAgent with state observation enabled
2. Register callbacks to monitor agent state events
3. Process events and collect metrics

Usage:
    python examples/agent_observer_demo.py
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from agents.react_agent import ReactAgent, ReactAgentConfig
from reactive_agents.providers.external.client import MCPClient
from context.agent_observer import AgentStateEvent

# Note: In a real implementation, ensure the calculator tool is available
# For this example, we'll use a placeholder import comment
# from tools.calculator import CalculatorTool


class AgentMonitor:
    """
    Example class that monitors agent state events in real-time.

    This class demonstrates how an external application can hook into
    agent state and track metrics during execution.
    """

    def __init__(self):
        """Initialize the monitor with empty state"""
        self.events: List[Dict[str, Any]] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.tools_used: Dict[str, int] = {}
        self.start_time = time.time()

        # Metrics
        self.metrics = {
            "total_events": 0,
            "sessions_started": 0,
            "sessions_completed": 0,
            "errors": 0,
            "tool_calls": 0,
            "events_by_type": {},
            "runtime_metrics": {},
        }

        # Summary information for each completed session
        self.session_summaries: Dict[str, Dict[str, Any]] = {}

    def handle_event(self, event_data: Dict[str, Any]) -> None:
        """
        Process an agent state event.

        Args:
            event_data: The event data from the agent
        """
        # Store raw event
        self.events.append(event_data)

        # Update basic metrics
        self.metrics["total_events"] += 1
        event_type = event_data.get("event_type", "unknown")
        self.metrics["events_by_type"][event_type] = (
            self.metrics["events_by_type"].get(event_type, 0) + 1
        )

        # Process specific event types
        session_id = event_data.get("session_id")

        # Safely handle None session_id
        if session_id is None:
            session_id = "unknown"

        if event_type == "session_started":
            # New session started
            self.metrics["sessions_started"] += 1
            self.active_sessions[session_id] = {
                "start_time": event_data.get("timestamp", time.time()),
                "task": event_data.get("initial_task", ""),
                "status": "running",
                "iterations": 0,
                "tool_calls": 0,
            }
            print(
                f"\n[{datetime.now().strftime('%H:%M:%S')}] ✨ Session started: {session_id}"
            )
            print(f"Task: {event_data.get('initial_task')}")

        elif event_type == "session_ended":
            # Session completed
            self.metrics["sessions_completed"] += 1
            elapsed = event_data.get("elapsed_time", 0)
            iterations = event_data.get("iterations", 0)
            final_status = event_data.get("final_status", "unknown")

            # Update session summary
            if session_id in self.active_sessions:
                session_info = self.active_sessions[session_id]
                session_info["elapsed_time"] = elapsed
                session_info["iterations"] = iterations
                session_info["status"] = final_status

                # Move to completed sessions
                self.session_summaries[session_id] = self.active_sessions.pop(
                    session_id
                )

            print(
                f"\n[{datetime.now().strftime('%H:%M:%S')}] 🏁 Session ended: {session_id}"
            )
            print(f"Status: {final_status}")
            print(f"Duration: {elapsed:.2f}s, Iterations: {iterations}")

        elif event_type == "iteration_started":
            # Iteration started
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["iterations"] = event_data.get(
                    "iteration", 0
                )
            print(
                f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔄 Iteration {event_data.get('iteration')} started"
            )

        elif event_type == "task_status_changed":
            # Task status changed
            new_status = event_data.get("new_status", "unknown")
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["status"] = new_status

            previous_status = event_data.get("previous_status", "unknown")
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Status changed: {previous_status} → {new_status}"
            )

        elif event_type == "tool_called":
            # Tool called
            self.metrics["tool_calls"] += 1
            tool_name = event_data.get("tool_name", "unknown")
            self.tools_used[tool_name] = self.tools_used.get(tool_name, 0) + 1

            if session_id in self.active_sessions:
                self.active_sessions[session_id]["tool_calls"] = (
                    self.active_sessions[session_id].get("tool_calls", 0) + 1
                )

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Tool called: {tool_name}"
            )

        elif event_type == "error_occurred":
            # Error occurred
            self.metrics["errors"] += 1
            error_type = event_data.get("error", "unknown")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {error_type}")
            print(f"  Details: {event_data.get('details', 'No details provided')}")

        elif event_type == "final_answer_set":
            # Final answer set
            answer = event_data.get("answer", "")
            answer_preview = str(answer)[:50]
            if len(str(answer)) > 50:
                answer_preview += "..."

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Final answer: {answer_preview}"
            )

    def print_summary(self) -> None:
        """Print a summary of the monitoring session"""
        runtime = time.time() - self.start_time

        print("\n" + "=" * 50)
        print(f"📊 AGENT MONITORING SUMMARY (Runtime: {runtime:.2f}s)")
        print("=" * 50)
        print(f"Sessions started: {self.metrics['sessions_started']}")
        print(f"Sessions completed: {self.metrics['sessions_completed']}")
        print(f"Total events processed: {self.metrics['total_events']}")
        print(f"Tool calls: {self.metrics['tool_calls']}")
        print(f"Errors: {self.metrics['errors']}")

        if self.tools_used:
            print("\nTools used:")
            for tool, count in sorted(
                self.tools_used.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  - {tool}: {count} calls")

        if self.session_summaries:
            print("\nCompleted sessions:")
            for session_id, info in self.session_summaries.items():
                print(
                    f"  - {session_id[:8]}: {info.get('status')} ({info.get('elapsed_time', 0):.2f}s, {info.get('iterations', 0)} iterations)"
                )


async def main():
    """Main function demonstrating the AgentStateObserver"""
    # Create a monitor
    monitor = AgentMonitor()

    # Create an agent with state observation enabled
    mcp_client = MCPClient("http://localhost:8000")  # Not used in this example

    # Configure the ReactAgent with state observation enabled
    # Note: Default values will be used for parameters not explicitly set
    agent_config = ReactAgentConfig(
        agent_name="ObservedAgent",
        role="Math Solver",
        provider_model_name="ollama:qwen3:8b",  # Adjust based on your installation
        instructions="Solve math problems using the calculator tool when needed.",
        mcp_client=mcp_client,
        max_iterations=5,
        reflect_enabled=False,
        min_completion_score=1.0,
        log_level="info",
        tool_use_enabled=True,
        use_memory_enabled=True,
        collect_metrics_enabled=True,
        check_tool_feasibility=True,
        enable_caching=True,
        initial_task=None,  # Will be set in the run method
        confirmation_callback=None,
        kwargs={},  # Empty additional kwargs
    )

    # Create the agent
    agent = ReactAgent(config=agent_config)

    # Make sure observer is initialized
    if agent.context.state_observer is not None:
        # Register our event handler with the agent's observer
        # Register for all event types to monitor everything
        for event_type in AgentStateEvent:
            print(f"Registering callback for event type: {event_type}")
            agent.context.state_observer.register_callback(
                event_type, monitor.handle_event
            )
    else:
        print("Warning: Agent state observer is not initialized!")

    # In a real implementation, add required tools
    # Example: calculator_tool = CalculatorTool()
    # agent.context.tools.append(calculator_tool)

    # Execute a simple task
    print("Starting agent execution...")
    result = await agent.run(initial_task="Calculate (17 * 34) + (92 / 4)")

    # Wait a moment to ensure all events are processed
    await asyncio.sleep(1)

    # Print summary
    monitor.print_summary()

    # Print the final result
    print("\n" + "=" * 50)
    print("AGENT EXECUTION RESULT:")
    print("=" * 50)
    print(f"Status: {result.get('status')}")
    print(f"Result: {result.get('result')}")

    # Cleanup
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
