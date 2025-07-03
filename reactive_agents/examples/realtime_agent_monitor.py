#!/usr/bin/env python
"""
Example demonstrating how to create a real-time web-based agent monitoring system.

This example shows how to:
1. Create a real-time web server using FastAPI and websockets
2. Send agent state events to connected clients
3. Visualize agent state in a browser

Usage:
    pip install fastapi uvicorn websockets
    python examples/realtime_agent_monitor.py

Then open a browser to http://localhost:8888/
"""

import asyncio
import json
import time
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Set

# Note: This example requires additional dependencies:
# pip install fastapi uvicorn websockets
try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:
    print("This example requires additional dependencies.")
    print("Please install them with: pip install fastapi uvicorn websockets")
    import sys

    sys.exit(1)

from agents.react_agent import ReactAgent, ReactAgentConfig
from reactive_agents.providers.external.client import MCPClient
from context.agent_observer import AgentStateEvent

# Create FastAPI app
app = FastAPI(title="Real-time Agent Monitor")

# Directory for static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


# Create a WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # Handle any errors that occur when sending to a client
                pass


# Initialize connection manager
manager = ConnectionManager()

# Track agent state
agent_state = {
    "sessions": {},
    "events": [],
    "active_agents": set(),
    "tools_used": {},
    "event_counts": {},
}


# Create a simple HTML page for real-time monitoring
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Agent Monitor</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            width: 90%;
            margin: 20px auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-gap: 20px;
        }
        .card {
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        #events-list {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            background: #fdfdfd;
            font-family: monospace;
        }
        .event-item {
            margin-bottom: 8px;
            padding: 8px;
            border-left: 4px solid #3498db;
            background-color: #f8f9fa;
        }
        .event-time {
            color: #7f8c8d;
            font-size: 0.8em;
        }
        .event-type {
            font-weight: bold;
            margin-right: 8px;
        }
        .tool-call {
            border-left-color: #2ecc71;
        }
        .error-event {
            border-left-color: #e74c3c;
        }
        .status-change {
            border-left-color: #f39c12;
        }
        .final-answer {
            border-left-color: #9b59b6;
        }
        #tools-chart, #events-chart {
            width: 100%;
            height: 250px;
        }
        #session-info {
            grid-column: span 2;
        }
        .full-width {
            grid-column: span 2;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            grid-gap: 15px;
        }
        .stat-box {
            background: #fff;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card full-width">
            <h1>Real-time Agent Monitor</h1>
            <p>Connected to the agent server. Monitoring events in real-time.</p>
        </div>
        
        <div class="card">
            <h2>Event Stats</h2>
            <div class="stats-grid" id="stat-boxes">
                <!-- Will be filled dynamically -->
            </div>
        </div>
        
        <div class="card">
            <h2>Agent Sessions</h2>
            <div id="sessions-list">
                <!-- Will be filled dynamically -->
            </div>
        </div>
        
        <div class="card" id="session-info">
            <h2>Latest Events</h2>
            <div id="events-list">
                <!-- Will be filled dynamically -->
            </div>
        </div>
    </div>

    <script>
        let ws = new WebSocket(`ws://${window.location.host}/ws`);
        const eventsListElement = document.getElementById('events-list');
        const sessionsListElement = document.getElementById('sessions-list');
        const statBoxesElement = document.getElementById('stat-boxes');
        
        let stats = {
            "events_total": 0,
            "sessions_total": 0,
            "tool_calls": 0,
            "errors": 0,
            "active_sessions": 0,
            "completed_sessions": 0
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'event') {
                // Add new event to the list
                const eventData = data.data;
                
                updateStats(eventData);
                addEventToList(eventData);
                updateSessionsList();
            } else if (data.type === 'state_update') {
                // Full state update
                updateFullState(data.data);
            }
        };
        
        function updateStats(eventData) {
            // Update basic stats
            stats.events_total++;
            
            // Update specific stats based on event type
            if (eventData.event_type === 'session_started') {
                stats.sessions_total++;
                stats.active_sessions++;
            } else if (eventData.event_type === 'session_ended') {
                stats.active_sessions--;
                stats.completed_sessions++;
            } else if (eventData.event_type === 'tool_called') {
                stats.tool_calls++;
            } else if (eventData.event_type === 'error_occurred') {
                stats.errors++;
            }
            
            // Render stats
            renderStats();
        }
        
        function renderStats() {
            statBoxesElement.innerHTML = '';
            
            // Create stat boxes
            for (const [key, value] of Object.entries(stats)) {
                const box = document.createElement('div');
                box.className = 'stat-box';
                
                // Format the label
                const label = key.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                
                box.innerHTML = `
                    <div class="stat-value">${value}</div>
                    <div class="stat-label">${label}</div>
                `;
                
                statBoxesElement.appendChild(box);
            }
        }
        
        function addEventToList(eventData) {
            const eventItem = document.createElement('div');
            
            // Add appropriate class based on event type
            eventItem.className = 'event-item';
            if (eventData.event_type === 'tool_called') {
                eventItem.className += ' tool-call';
            } else if (eventData.event_type === 'error_occurred') {
                eventItem.className += ' error-event';
            } else if (eventData.event_type === 'task_status_changed') {
                eventItem.className += ' status-change';
            } else if (eventData.event_type === 'final_answer_set') {
                eventItem.className += ' final-answer';
            }
            
            // Format event time
            const date = new Date(eventData.timestamp * 1000);
            const timeString = date.toLocaleTimeString();
            
            // Prepare event details
            let details = '';
            if (eventData.event_type === 'tool_called') {
                details = `Tool: ${eventData.tool_name}`;
            } else if (eventData.event_type === 'task_status_changed') {
                details = `${eventData.previous_status} → ${eventData.new_status}`;
            } else if (eventData.event_type === 'error_occurred') {
                details = `Error: ${eventData.error}`;
            } else if (eventData.event_type === 'final_answer_set') {
                details = `Answer: ${eventData.answer?.substring(0, 50)}...`;
            } else if (eventData.event_type === 'iteration_started') {
                details = `Iteration: ${eventData.iteration}`;
            } else if (eventData.event_type === 'session_started') {
                details = `Session: ${eventData.session_id}`;
            } else if (eventData.event_type === 'session_ended') {
                details = `Session: ${eventData.session_id}, Status: ${eventData.final_status}`;
            }
            
            // Create event content
            eventItem.innerHTML = `
                <span class="event-time">${timeString}</span>
                <span class="event-type">${eventData.event_type}</span>
                <span class="event-details">${details}</span>
            `;
            
            eventsListElement.appendChild(eventItem);
            eventsListElement.scrollTop = eventsListElement.scrollHeight;
            
            // Limit number of events shown
            if (eventsListElement.children.length > 100) {
                eventsListElement.removeChild(eventsListElement.firstChild);
            }
        }
        
        function updateSessionsList() {
            // This would be updated with actual session data in a full implementation
            sessionsListElement.innerHTML = `<p>Active sessions: ${stats.active_sessions}</p>
                                            <p>Completed sessions: ${stats.completed_sessions}</p>`;
        }
        
        function updateFullState(stateData) {
            // Update with full state if needed
            console.log("Full state update received", stateData);
        }
        
        // Initialize stats
        renderStats();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get():
    """Return the HTML page for the monitoring interface"""
    return HTML


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections"""
    await manager.connect(websocket)
    try:
        # Send initial state if there's any history
        if agent_state["events"]:
            await websocket.send_text(
                json.dumps({"type": "state_update", "data": agent_state})
            )

        # Keep connection open
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


class WebSocketEventHandler:
    """
    Handler for agent events that broadcasts them to WebSocket clients.
    """

    def __init__(self, connection_manager):
        self.manager = connection_manager

    async def handle_event(self, event_data: Dict[str, Any]) -> None:
        """
        Process an agent state event and broadcast it to WebSocket clients.

        Args:
            event_data: The event data from the agent
        """
        # Store event in history
        agent_state["events"].append(event_data)

        # Limit history size
        if len(agent_state["events"]) > 1000:
            agent_state["events"] = agent_state["events"][-1000:]

        # Update event counts
        event_type = event_data.get("event_type", "unknown")
        agent_state["event_counts"][event_type] = (
            agent_state["event_counts"].get(event_type, 0) + 1
        )

        # Process specific event types
        session_id = event_data.get("session_id")
        agent_name = event_data.get("agent_name", "unknown")

        # Add agent to active agents
        if agent_name:
            agent_state["active_agents"].add(agent_name)

        # Safely handle None session_id
        if session_id is None:
            session_id = f"unknown-{time.time()}"

        if event_type == "session_started":
            # New session started
            agent_state["sessions"][session_id] = {
                "start_time": event_data.get("timestamp", time.time()),
                "task": event_data.get("initial_task", ""),
                "status": "running",
                "iterations": 0,
                "tool_calls": 0,
                "agent_name": agent_name,
            }

        elif event_type == "session_ended":
            # Update session information
            if session_id in agent_state["sessions"]:
                agent_state["sessions"][session_id]["status"] = event_data.get(
                    "final_status", "unknown"
                )
                agent_state["sessions"][session_id]["end_time"] = event_data.get(
                    "timestamp", time.time()
                )
                agent_state["sessions"][session_id]["elapsed_time"] = event_data.get(
                    "elapsed_time", 0
                )
                agent_state["sessions"][session_id]["iterations"] = event_data.get(
                    "iterations", 0
                )

        elif event_type == "tool_called":
            # Update tool usage stats
            tool_name = event_data.get("tool_name", "unknown")
            agent_state["tools_used"][tool_name] = (
                agent_state["tools_used"].get(tool_name, 0) + 1
            )

            # Update session tool calls
            if session_id in agent_state["sessions"]:
                agent_state["sessions"][session_id]["tool_calls"] = (
                    agent_state["sessions"][session_id].get("tool_calls", 0) + 1
                )

        # Broadcast event to all connected clients
        await self.manager.broadcast(json.dumps({"type": "event", "data": event_data}))


async def run_agent_task(handler):
    """Run an agent task for demo purposes"""
    # Create an agent with state observation enabled
    mcp_client = MCPClient("http://localhost:8000")  # Not used in this example

    # Configure the ReactAgent with state observation enabled
    agent_config = ReactAgentConfig(
        agent_name="WebMonitoredAgent",
        role="Math Solver",
        provider_model_name="ollama:qwen2:7b",  # Adjust based on your installation
        instructions="Solve math problems using available tools.",
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
            agent.context.state_observer.register_callback(
                event_type, handler.handle_event
            )
    else:
        print("Warning: Agent state observer is not initialized!")

    # Execute a simple task
    print("Starting agent execution...")
    result = await agent.run(initial_task="Calculate (17 * 34) + (92 / 4)")
    print(f"Agent execution completed: {result.get('status')}")

    # Cleanup
    await agent.close()


async def run_periodic_tasks(handler):
    """Run periodic agent tasks for demonstration"""
    while True:
        # Run an agent task
        await run_agent_task(handler)

        # Wait before starting the next one
        await asyncio.sleep(15)  # Run a new task every 15 seconds


@app.on_event("startup")
async def startup_event():
    """Start the periodic task runner when the application starts"""
    handler = WebSocketEventHandler(manager)
    asyncio.create_task(run_periodic_tasks(handler))


def main():
    """Run the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=8888)


if __name__ == "__main__":
    main()
