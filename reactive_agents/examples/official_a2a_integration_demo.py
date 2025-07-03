#!/usr/bin/env python3
"""
Official A2A Integration Demo

This demo shows how to integrate the Reactive Agents framework with
the official Google A2A protocol, following patterns from the
official A2A samples repository.

Key Features Demonstrated:
1. Atomic task delegation following A2A principles
2. Agent capability discovery
3. Standard A2A endpoints simulation
4. Official A2A message patterns
5. Bridge to official A2A SDK

References:
- https://github.com/a2aproject/a2a-samples
- https://github.com/a2aproject/A2A
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from reactive_agents.app.agents.reactive_agent import ReactiveAgentV2
from reactive_agents.config.natural_language_config import create_agent_from_nl
from reactive_agents.communication.a2a_official_bridge import (
    A2AOfficialBridge,
    A2AAtomicTask,
    create_a2a_compatible_agent_network,
    demonstrate_official_a2a_pattern,
)
from reactive_agents.providers.llm.factory import ModelProviderFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_a2a_sample_agents():
    """
    Create agents following patterns from official A2A samples.

    This mirrors the multi-agent setups seen in the official
    A2A repository examples.
    """

    print("\n🤖 Creating A2A Compatible Agents...")
    print("=" * 50)

    # Create model provider for agents
    model_provider = ModelProviderFactory.get_model_provider("ollama:llama3.2")

    # Research Agent (similar to google_adk samples)
    research_agent = await create_agent_from_nl(
        "Create a research agent that can analyze documents, search for information, "
        "and provide detailed summaries. Use plan-execute-reflect reasoning for "
        "complex multi-step research tasks.",
        model_provider,
    )
    # Set the agent name after creation
    research_agent.context.agent_name = "research_agent"

    # Analysis Agent (similar to crewai samples)
    analysis_agent = await create_agent_from_nl(
        "Create an analysis agent that specializes in data processing, pattern "
        "recognition, and generating insights. Use adaptive reasoning to handle "
        "various types of analytical tasks.",
        model_provider,
    )
    # Set the agent name after creation
    analysis_agent.context.agent_name = "analysis_agent"

    # Communication Agent (for A2A coordination)
    comm_agent = await create_agent_from_nl(
        "Create a communication agent that coordinates between other agents, "
        "routes tasks, and manages workflow execution. Use reactive reasoning "
        "for quick routing decisions.",
        model_provider,
    )
    # Set the agent name after creation
    comm_agent.context.agent_name = "communication_agent"

    return [research_agent, analysis_agent, comm_agent]


async def demonstrate_atomic_task_delegation():
    """
    Demonstrate atomic task delegation following official A2A principles.

    Per A2A documentation: Tasks should be atomic and processed by a single
    selected agent from start to finish.
    """

    print("\n🎯 Atomic Task Delegation Demo")
    print("=" * 50)

    # Create A2A compatible agents
    agents = await create_a2a_sample_agents()
    a2a_bridge = await create_a2a_compatible_agent_network(agents)

    # Example 1: Research Task (Atomic)
    print("\n📚 Delegating Research Task...")
    research_task = await a2a_bridge.delegate_atomic_task(
        task_description="Research the latest developments in AI agent frameworks and summarize key findings",
        required_capabilities=["reasoning", "complex_planning"],
        input_data={
            "topic": "AI agent frameworks",
            "focus": "recent developments",
            "output_format": "summary",
        },
    )

    print(f"   Task ID: {research_task.task_id}")
    print(f"   Status: {research_task.status}")
    print(f"   Assigned Agent: {research_task.assigned_agent_id}")
    if research_task.result:
        print(f"   Success: {research_task.result.get('success', False)}")

    # Example 2: Analysis Task (Atomic)
    print("\n📊 Delegating Analysis Task...")
    analysis_task = await a2a_bridge.delegate_atomic_task(
        task_description="Analyze the provided data for patterns and generate insights",
        required_capabilities=["reasoning"],
        input_data={
            "data_type": "market_trends",
            "analysis_depth": "comprehensive",
            "output_format": "structured_report",
        },
    )

    print(f"   Task ID: {analysis_task.task_id}")
    print(f"   Status: {analysis_task.status}")
    print(f"   Assigned Agent: {analysis_task.assigned_agent_id}")
    if analysis_task.result:
        print(f"   Success: {analysis_task.result.get('success', False)}")

    return a2a_bridge, [research_task, analysis_task]


async def demonstrate_agent_discovery():
    """
    Demonstrate agent discovery following A2A patterns.

    This shows how agents can discover each other's capabilities,
    similar to the discovery mechanisms in official A2A samples.
    """

    print("\n🔍 Agent Discovery Demo")
    print("=" * 50)

    agents = await create_a2a_sample_agents()
    a2a_bridge = await create_a2a_compatible_agent_network(agents)

    # Get discovery information
    discovery_info = a2a_bridge.get_agent_discovery_info()

    print(f"Protocol Version: {discovery_info['protocol_version']}")
    print(f"Supported Features: {', '.join(discovery_info['supported_features'])}")

    print("\nAvailable Agents:")
    for agent_id, agent_info in discovery_info["agents"].items():
        print(f"\n  🤖 {agent_id}")
        print(f"     Status: {agent_info['status']}")
        print(f"     Capabilities:")
        for cap in agent_info["capabilities"]:
            print(f"       - {cap['name']}: {cap['description']}")

    return discovery_info


async def demonstrate_a2a_endpoints_simulation():
    """
    Simulate the standard A2A endpoints that would be used
    with the official A2A SDK.
    """

    print("\n🔗 A2A Endpoints Simulation")
    print("=" * 50)

    agents = await create_a2a_sample_agents()
    a2a_bridge = await create_a2a_compatible_agent_network(agents)

    # Simulate standard A2A endpoints
    endpoints = a2a_bridge.a2a_endpoints

    print("Standard A2A Endpoints:")
    for endpoint_name, endpoint_path in endpoints.items():
        print(f"  {endpoint_name}: {endpoint_path}")

    # Simulate task delegation via endpoint
    print("\n📤 Simulating /a2a/delegate endpoint...")
    task = await a2a_bridge.delegate_atomic_task(
        "Process this request through the A2A delegation endpoint",
        required_capabilities=["reasoning"],
    )

    # Simulate status check via endpoint
    print("📋 Simulating /a2a/status endpoint...")
    status = await a2a_bridge.get_task_status(task.task_id)
    if status:
        print(f"   Task Status: {status.status}")
        if status.completed_at and status.created_at:
            execution_time = status.completed_at - status.created_at
            print(f"   Execution Time: {execution_time:.2f}s")
        else:
            print("   Execution Time: In progress")

    # Simulate discovery via endpoint
    print("🔍 Simulating /a2a/discover endpoint...")
    discovery = a2a_bridge.get_agent_discovery_info()
    print(f"   Found {len(discovery['agents'])} agents")

    return endpoints


async def demonstrate_official_a2a_patterns():
    """
    Demonstrate patterns that mirror official A2A samples.

    This shows how to structure multi-agent systems following
    the established patterns from the official A2A repository.
    """

    print("\n🏗️  Official A2A Patterns Demo")
    print("=" * 50)

    # Pattern 1: Host-Worker Model (from official samples)
    print("\n1. Host-Worker Pattern:")
    agents = await create_a2a_sample_agents()
    a2a_bridge = await create_a2a_compatible_agent_network(agents)

    # Host agent coordinates workers
    host_agent = agents[2]  # communication_agent acts as host
    worker_agents = agents[:2]  # research and analysis agents as workers

    print(f"   Host Agent: {host_agent.context.agent_name}")
    print(f"   Worker Agents: {[agent.context.agent_name for agent in worker_agents]}")

    # Pattern 2: Atomic Task Processing
    print("\n2. Atomic Task Processing:")
    atomic_tasks = [
        "Summarize recent AI research papers",
        "Analyze market trends in technology sector",
        "Generate insights from data patterns",
    ]

    for task_desc in atomic_tasks:
        task = await a2a_bridge.delegate_atomic_task(task_desc)
        print(f"   ✓ Task: {task_desc[:50]}... -> {task.assigned_agent_id}")

    # Pattern 3: Capability-Based Routing
    print("\n3. Capability-Based Routing:")
    specialized_task = await a2a_bridge.delegate_atomic_task(
        "Perform complex multi-step analysis with detailed planning",
        required_capabilities=["complex_planning", "reasoning"],
    )
    print(f"   ✓ Complex Task -> {specialized_task.assigned_agent_id}")

    return a2a_bridge


async def show_next_steps_for_official_integration():
    """
    Show the next steps needed for full official A2A integration.
    """

    print("\n🚀 Next Steps for Official A2A Integration")
    print("=" * 50)

    next_steps = [
        "1. Install Official A2A SDK",
        "   pip install a2a-sdk  # (when available)",
        "",
        "2. Replace Bridge with Official SDK",
        "   from a2a import Agent, TaskDelegate",
        "   # Replace A2AOfficialBridge with official classes",
        "",
        "3. Implement Standard Authentication",
        "   # Add OAuth2 or other auth mechanisms",
        "   # Follow official A2A authentication patterns",
        "",
        "4. Use Official Endpoints",
        "   # Replace simulated endpoints with real ones",
        "   # Integrate with official A2A network",
        "",
        "5. Follow Official Message Schemas",
        "   # Ensure message format compliance",
        "   # Use official message validation",
        "",
        "6. Test with Official A2A Samples",
        "   # Validate compatibility with other A2A agents",
        "   # Test multi-language interoperability",
    ]

    for step in next_steps:
        print(step)


async def main():
    """
    Main demo function showing official A2A integration.
    """

    print("\n🎯 Official A2A Integration Demo")
    print("🔗 Integrating Reactive Agents with Google A2A Protocol")
    print("=" * 60)

    try:
        # 1. Demonstrate atomic task delegation
        a2a_bridge, tasks = await demonstrate_atomic_task_delegation()

        # 2. Demonstrate agent discovery
        discovery_info = await demonstrate_agent_discovery()

        # 3. Simulate A2A endpoints
        endpoints = await demonstrate_a2a_endpoints_simulation()

        # 4. Show official A2A patterns
        pattern_bridge = await demonstrate_official_a2a_patterns()

        # 5. Show integration path
        await show_next_steps_for_official_integration()

        print("\n✅ Official A2A Integration Demo Complete!")
        print("\n📝 Summary:")
        print(f"   - Created A2A compatible bridge")
        print(f"   - Demonstrated atomic task delegation")
        print(f"   - Showed agent capability discovery")
        print(f"   - Simulated standard A2A endpoints")
        print(f"   - Followed official A2A patterns")
        print(f"   - Ready for official SDK integration")

        # Final demonstration
        result = await demonstrate_official_a2a_pattern()
        print(f"\n🎯 {result['message']}")

        return {
            "success": True,
            "a2a_bridge": a2a_bridge,
            "discovery_info": discovery_info,
            "endpoints": endpoints,
            "integration_ready": True,
        }

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Run the demo
    result = asyncio.run(main())

    if not result["success"]:
        print(f"❌ Demo failed: {result['error']}")
        sys.exit(1)
    else:
        print("\n🎉 Ready for official A2A SDK integration!")
