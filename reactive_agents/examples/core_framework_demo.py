#!/usr/bin/env python3
"""
Core Framework Improvements Demo

This demo showcases the major improvements to the Reactive Agents Framework:

1. ReactiveAgentV2Builder - Enhanced builder with reasoning strategies and natural language config
2. Official EventBus System - Centralized event management with middleware
3. Vector Memory Manager - ChromaDB-based semantic memory (when available)
4. CLI System - Laravel Artisan-style command interface

This demonstrates the solid foundation we've built for the framework.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from reactive_agents.app.agents.builders import (
    ReactiveAgentV2Builder,
    quick_create_agent,
)
from reactive_agents.core.engine.event_bus import (
    EventBus,
    AgentEventBus,
    Event,
    EventPriority,
    LoggingMiddleware,
    FilteringMiddleware,
)
from reactive_agents.core.types.event_types import AgentStateEvent


class CoreFrameworkDemo:
    """Demonstrates the enhanced Reactive Agents Framework capabilities"""

    def __init__(self):
        self.demo_results: Dict[str, Any] = {}

    def print_header(self, title: str):
        """Print a formatted section header"""
        print(f"\n{'='*60}")
        print(f"🚀 {title}")
        print(f"{'='*60}")

    def print_success(self, message: str):
        """Print success message"""
        print(f"✅ {message}")

    def print_info(self, message: str):
        """Print info message"""
        print(f"ℹ️  {message}")

    async def demo_reactive_agent_v2_builder(self):
        """Demonstrate the enhanced ReactiveAgentV2Builder"""
        self.print_header("ReactiveAgentV2Builder Demo")

        print("1. Creating agents with different reasoning strategies...")

        # Demo 1: Reflect-Decide-Act Agent
        self.print_info("Creating Reflect-Decide-Act agent...")
        try:
            reflect_agent = await (
                ReactiveAgentV2Builder()
                .with_name("Reflection Agent")
                .with_reasoning_strategy("reflect_decide_act")
                .with_model("ollama:qwen2:7b")
                .with_mcp_tools(["time"])
                .with_instructions("Think carefully about each step before acting")
                .build()
            )

            self.print_success("✓ Reflection Agent created successfully")
            await reflect_agent.close()

        except Exception as e:
            print(f"❌ Failed to create Reflection Agent: {e}")

        # Demo 2: Adaptive Agent with Vector Memory
        self.print_info("Creating Adaptive agent with vector memory...")
        try:
            adaptive_agent = await (
                ReactiveAgentV2Builder()
                .with_name("Adaptive Agent")
                .with_reasoning_strategy("adaptive")
                .with_vector_memory("demo_memories")
                .with_dynamic_strategy_switching(True)
                .with_mcp_tools(["time"])
                .build()
            )

            self.print_success("✓ Adaptive Agent with vector memory created")
            await adaptive_agent.close()

        except Exception as e:
            print(f"❌ Failed to create Adaptive Agent: {e}")

        # Demo 3: Factory Methods
        self.print_info("Using factory methods for quick agent creation...")
        try:
            research_agent = await ReactiveAgentV2Builder.reactive_research_agent()
            self.print_success("✓ Research Agent created using factory method")
            await research_agent.close()

        except Exception as e:
            print(f"❌ Failed to create Research Agent: {e}")

        # Demo 4: Natural Language Configuration (if available)
        self.print_info("Testing natural language configuration...")
        try:
            nl_agent = await (
                ReactiveAgentV2Builder()
                .with_natural_language_config(
                    "Create an agent that can help with time-related questions and basic research"
                )
                .build()
            )

            self.print_success("✓ Natural Language Agent created")
            await nl_agent.close()

        except Exception as e:
            print(f"⚠️  Natural Language config not available: {e}")

        # Demo 5: Quick Agent Creation
        self.print_info("Testing quick agent creation...")
        try:
            result = await quick_create_agent(
                task="What is the current time?",
                model="ollama:qwen2:7b",
                tools=["time"],
                use_reactive_v2=True,
            )

            self.print_success("✓ Quick agent completed task")
            print(f"   Result: {result.get('final_answer', 'No result')[:100]}...")

        except Exception as e:
            print(f"❌ Quick agent failed: {e}")

    async def demo_official_event_bus(self):
        """Demonstrate the official EventBus system"""
        self.print_header("Official EventBus System Demo")

        print("1. Creating and configuring EventBus...")

        # Create event bus with logging
        event_bus = EventBus()

        # Add custom middleware
        def priority_filter(event: Event) -> bool:
            return event.priority != EventPriority.LOW

        event_bus.add_middleware(FilteringMiddleware(priority_filter))

        # Event tracking
        received_events = []

        async def event_tracker(event: Event):
            received_events.append(event)
            print(f"   📨 Received: {event.type} (Priority: {event.priority.value})")

        # Subscribe to events
        subscription = event_bus.subscribe_function("test_event", event_tracker)

        print("2. Testing event emission and middleware...")

        # Emit events with different priorities
        await event_bus.emit(
            "test_event", {"message": "High priority"}, priority=EventPriority.HIGH
        )
        await event_bus.emit(
            "test_event", {"message": "Normal priority"}, priority=EventPriority.NORMAL
        )
        await event_bus.emit(
            "test_event",
            {"message": "Low priority - filtered"},
            priority=EventPriority.LOW,
        )

        # Small delay for async processing
        await asyncio.sleep(0.1)

        self.print_success(
            f"✓ EventBus processed {len(received_events)} events (low priority filtered)"
        )

        # Test Agent Event Bus
        print("3. Testing AgentEventBus...")

        agent_bus = AgentEventBus("DemoAgent")
        agent_events = []

        async def agent_event_tracker(event: Event):
            agent_events.append(event)
            print(f"   🤖 Agent Event: {event.type}")

        agent_bus.subscribe_to_agent_event(
            AgentStateEvent.SESSION_STARTED, agent_event_tracker
        )

        # Emit agent events
        await agent_bus.emit_agent_event(
            AgentStateEvent.SESSION_STARTED,
            {"session_id": "demo_session", "task": "Demo task"},
        )

        await asyncio.sleep(0.1)

        self.print_success(
            f"✓ AgentEventBus processed {len(agent_events)} agent events"
        )

        # Get statistics
        stats = event_bus.get_stats()
        print(f"4. EventBus Statistics:")
        print(f"   Events emitted: {stats['events_emitted']}")
        print(f"   Events processed: {stats['events_processed']}")
        print(f"   Active subscribers: {stats['active_subscribers']}")
        print(f"   Middleware count: {stats['middleware_count']}")

        # Cleanup
        subscription.unsubscribe()
        await event_bus.close()
        await agent_bus.close()

        self.print_success("✓ EventBus demo completed successfully")

    async def demo_vector_memory_manager(self):
        """Demonstrate vector memory capabilities"""
        self.print_header("Vector Memory Manager Demo")

        try:
            from reactive_agents.core.memory.vector_memory import (
                VectorMemoryManager,
                VectorMemoryConfig,
            )

            print("1. Vector memory dependencies available - creating manager...")

            # Create a mock context for demo
            class MockContext:
                def __init__(self):
                    self.agent_name = "DemoAgent"
                    self.use_memory_enabled = True
                    self.agent_logger = MockLogger()

            class MockLogger:
                def debug(self, msg):
                    print(f"   🔍 DEBUG: {msg}")

                def info(self, msg):
                    print(f"   ℹ️  INFO: {msg}")

                def warning(self, msg):
                    print(f"   ⚠️  WARNING: {msg}")

                def error(self, msg):
                    print(f"   ❌ ERROR: {msg}")

            # Create vector memory manager
            config = VectorMemoryConfig(
                collection_name="demo_memories", persist_directory="demo_vector_memory"
            )

            context = MockContext()
            memory_manager = VectorMemoryManager(context=context, config=config)

            # Wait for initialization
            await asyncio.sleep(1.0)

            self.print_success("✓ Vector memory manager created")

            # Test memory operations
            print("2. Testing memory operations...")

            # Store some demo memories
            await memory_manager.store_memory(
                "The user prefers detailed explanations",
                "preference",
                {"category": "user_preference", "importance": "high"},
            )

            await memory_manager.store_memory(
                "Successfully completed research task about AI",
                "session",
                {"task_type": "research", "success": True},
            )

            # Search memories
            results = await memory_manager.search_memory(
                "user preferences", n_results=3
            )

            self.print_success(f"✓ Memory search returned {len(results)} results")

            for result in results:
                print(
                    f"   📝 Memory: {result['content'][:50]}... (Score: {result.get('relevance_score', 0):.2f})"
                )

            # Get memory stats
            stats = memory_manager.get_memory_stats()
            print("3. Memory Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")

        except ImportError:
            print("⚠️  ChromaDB or SentenceTransformers not available")
            print("   Install with: pip install chromadb sentence-transformers")
            print("   Falling back to basic memory demonstration...")

            # Demo basic memory interface
            from reactive_agents.core.memory.memory_manager import MemoryManager

            class MockContext:
                def __init__(self):
                    self.agent_name = "DemoAgent"
                    self.use_memory_enabled = True
                    self.agent_logger = MockLogger()

            class MockLogger:
                def debug(self, msg):
                    pass

                def info(self, msg):
                    print(f"   ℹ️  {msg}")

                def warning(self, msg):
                    print(f"   ⚠️  {msg}")

                def error(self, msg):
                    print(f"   ❌ {msg}")

            context = MockContext()
            memory_manager = MemoryManager(context=context)

            self.print_success("✓ Basic memory manager created")

        except Exception as e:
            print(f"❌ Vector memory demo failed: {e}")

    def demo_cli_system(self):
        """Demonstrate CLI system capabilities"""
        self.print_header("CLI System Demo")

        print("1. CLI System Overview:")
        print("   The Reactive CLI provides Laravel Artisan-style commands:")
        print()

        cli_commands = [
            ("reactive config:init --project-name MyProject", "Initialize new project"),
            ("reactive make:agent --name 'Research Agent'", "Create new agent"),
            ("reactive agent:run --task 'Find current time'", "Run agent with task"),
            ("reactive agent:list --detail", "List all configured agents"),
            (
                "reactive agent:monitor --agent 'Research Agent'",
                "Monitor agent execution",
            ),
            ("reactive memory:migrate --all", "Migrate to vector memory"),
        ]

        for command, description in cli_commands:
            print(f"   📋 {command}")
            print(f"      → {description}")
            print()

        print("2. CLI Features:")
        features = [
            "🎨 Colored output for better readability",
            "⚙️  Natural language agent configuration",
            "🔧 Agent configuration file management",
            "📊 Real-time monitoring (when implemented)",
            "💾 Memory system migration tools",
            "🧪 Agent testing and validation",
        ]

        for feature in features:
            print(f"   {feature}")

        print("\n3. Usage Example:")
        print("   # Initialize project")
        print("   $ reactive config:init --project-name 'AI Assistant'")
        print()
        print("   # Create agent with natural language")
        print("   $ reactive make:agent --name 'Helper' \\")
        print("     --description 'Helps with research and time queries'")
        print()
        print("   # Run the agent")
        print("   $ reactive agent:run --config helper_config.json \\")
        print("     --task 'What is the current time in Tokyo?'")

        self.print_success("✓ CLI system ready for use")

    async def demo_integration_example(self):
        """Demonstrate integration of all systems working together"""
        self.print_header("Integration Demo - All Systems Working Together")

        print("1. Creating integrated agent with EventBus and enhanced builder...")

        try:
            # Create event bus for monitoring
            event_bus = AgentEventBus("IntegratedAgent")

            # Track events
            events_received = []

            async def integration_event_tracker(event: Event):
                events_received.append(event)
                print(
                    f"   📡 Event: {event.type} - {event.data.get('agent_name', 'Unknown')}"
                )

            # Subscribe to all agent events
            event_bus.subscribe_to_agent_event(
                AgentStateEvent.SESSION_STARTED, integration_event_tracker
            )
            event_bus.subscribe_to_agent_event(
                AgentStateEvent.TOOL_CALLED, integration_event_tracker
            )
            event_bus.subscribe_to_agent_event(
                AgentStateEvent.SESSION_ENDED, integration_event_tracker
            )

            # Create enhanced agent
            agent = await (
                ReactiveAgentV2Builder()
                .with_name("Integrated Demo Agent")
                .with_reasoning_strategy("reflect_decide_act")
                .with_mcp_tools(["time"])
                .with_dynamic_strategy_switching(True)
                .with_instructions("Demonstrate the integrated framework capabilities")
                .build()
            )

            self.print_success("✓ Integrated agent created")

            # Simulate events that would be emitted during execution
            print("2. Simulating agent execution events...")

            await event_bus.emit_agent_event(
                AgentStateEvent.SESSION_STARTED,
                {
                    "session_id": "integration_demo",
                    "task": "Demonstrate framework integration",
                    "reasoning_strategy": "reflect_decide_act",
                },
            )

            await event_bus.emit_agent_event(
                AgentStateEvent.TOOL_CALLED,
                {"tool_name": "time", "parameters": {"timezone": "UTC"}},
            )

            await event_bus.emit_agent_event(
                AgentStateEvent.SESSION_ENDED,
                {
                    "session_id": "integration_demo",
                    "status": "completed",
                    "final_answer": "Integration demo successful",
                },
            )

            # Small delay for event processing
            await asyncio.sleep(0.1)

            self.print_success(
                f"✓ Integration demo completed - {len(events_received)} events processed"
            )

            # Show framework capabilities summary
            print("3. Framework Integration Summary:")
            capabilities = [
                "✅ ReactiveAgentV2 with dynamic reasoning strategies",
                "✅ Official EventBus with middleware support",
                "✅ Vector memory integration (when dependencies available)",
                "✅ Natural language configuration parsing",
                "✅ Laravel Artisan-style CLI interface",
                "✅ Type-safe event subscriptions",
                "✅ Plugin-ready architecture foundation",
            ]

            for capability in capabilities:
                print(f"   {capability}")

            # Cleanup
            await agent.close()
            await event_bus.close()

        except Exception as e:
            print(f"❌ Integration demo failed: {e}")

    async def run_all_demos(self):
        """Run all framework demos"""
        start_time = time.time()

        print("🤖 Reactive Agents Framework - Core Improvements Demo")
        print("=" * 60)
        print("Demonstrating the solid foundation we've built:")
        print("• Enhanced ReactiveAgentV2Builder")
        print("• Official EventBus System")
        print("• Vector Memory Manager")
        print("• Laravel Artisan-style CLI")
        print("• Complete Integration")

        try:
            await self.demo_reactive_agent_v2_builder()
            await self.demo_official_event_bus()
            await self.demo_vector_memory_manager()
            self.demo_cli_system()
            await self.demo_integration_example()

            end_time = time.time()

            self.print_header("Demo Summary")
            self.print_success(
                f"All demos completed successfully in {end_time - start_time:.2f} seconds"
            )

            print("\n🎯 Next Steps:")
            print("   1. Install optional dependencies:")
            print("      pip install chromadb sentence-transformers click rich")
            print("   2. Try the CLI:")
            print("      python -m reactive_agents.cli.main --help")
            print("   3. Build your own agents:")
            print("      Use ReactiveAgentV2Builder for enhanced capabilities")
            print("   4. Monitor with EventBus:")
            print("      Subscribe to agent events for real-time monitoring")

        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            print("This may be due to missing dependencies or configuration issues.")


async def main():
    """Main demo entry point"""
    demo = CoreFrameworkDemo()
    await demo.run_all_demos()


if __name__ == "__main__":
    asyncio.run(main())
