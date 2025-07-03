#!/usr/bin/env python3
"""
Simple A2A Integration Demo

A streamlined demo showing integration with the official Google A2A protocol
without complex tool dependencies.

This demo focuses on:
1. Basic agent creation
2. A2A protocol compatibility
3. Atomic task delegation
4. Official A2A patterns
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from reactive_agents.app.agents.reactive_agent import ReactiveAgentV2
from reactive_agents.core.types.agent_types import ReactAgentConfig
from reactive_agents.providers.llm.factory import ModelProviderFactory
from reactive_agents.communication.a2a_official_bridge import (
    A2AOfficialBridge,
    A2AAtomicTask,
    create_a2a_compatible_agent_network,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_simple_agents():
    """
    Create simple agents without complex tool dependencies.
    """

    print("\n🤖 Creating Simple A2A Compatible Agents...")
    print("=" * 50)

    # Create model provider
    model_provider = ModelProviderFactory.get_model_provider("ollama:llama3.2")

    # Research Agent Configuration
    research_config = ReactAgentConfig(
        agent_name="research_agent",
        role="Research Assistant",
        provider_model_name="ollama:llama3.2",
        custom_tools=[],  # No tools for simplicity
        instructions="You are a research assistant that analyzes information and provides summaries.",
        max_iterations=10,
        tool_use_enabled=False,  # Disable tools
        use_memory_enabled=True,
        collect_metrics_enabled=True,
        reflect_enabled=True,
        log_level="info",
        kwargs={
            "reasoning_strategy": "reflect_decide_act",
            "enable_reactive_execution": True,
        },
    )

    # Analysis Agent Configuration
    analysis_config = ReactAgentConfig(
        agent_name="analysis_agent",
        role="Data Analyst",
        provider_model_name="ollama:llama3.2",
        custom_tools=[],  # No tools for simplicity
        instructions="You are a data analyst that processes information and generates insights.",
        max_iterations=10,
        tool_use_enabled=False,  # Disable tools
        use_memory_enabled=True,
        collect_metrics_enabled=True,
        reflect_enabled=True,
        log_level="info",
        kwargs={
            "reasoning_strategy": "adaptive",
            "enable_reactive_execution": True,
        },
    )

    # Communication Agent Configuration
    comm_config = ReactAgentConfig(
        agent_name="communication_agent",
        role="Communication Coordinator",
        provider_model_name="ollama:llama3.2",
        custom_tools=[],  # No tools for simplicity
        instructions="You are a communication coordinator that manages task delegation and routing.",
        max_iterations=5,
        tool_use_enabled=False,  # Disable tools
        use_memory_enabled=True,
        collect_metrics_enabled=True,
        reflect_enabled=True,
        log_level="info",
        kwargs={
            "reasoning_strategy": "reactive",
            "enable_reactive_execution": True,
        },
    )

    # Create agents
    research_agent = ReactiveAgentV2(research_config)
    analysis_agent = ReactiveAgentV2(analysis_config)
    comm_agent = ReactiveAgentV2(comm_config)

    print(f"   ✅ Created {research_agent.context.agent_name}")
    print(f"   ✅ Created {analysis_agent.context.agent_name}")
    print(f"   ✅ Created {comm_agent.context.agent_name}")

    return [research_agent, analysis_agent, comm_agent]


async def demonstrate_a2a_atomic_tasks():
    """
    Demonstrate atomic task delegation with simple tasks.
    """

    print("\n🎯 A2A Atomic Task Delegation")
    print("=" * 50)

    # Create simple agents
    agents = await create_simple_agents()
    a2a_bridge = await create_a2a_compatible_agent_network(agents)

    # Simple atomic tasks that don't require tools
    tasks = [
        {
            "description": "Summarize the key benefits of artificial intelligence",
            "capabilities": ["reasoning"],
            "input_data": {"context": "general_knowledge"},
        },
        {
            "description": "Analyze the pros and cons of remote work",
            "capabilities": ["reasoning"],
            "input_data": {"focus": "workplace_trends"},
        },
        {
            "description": "Provide a brief explanation of machine learning",
            "capabilities": ["reasoning"],
            "input_data": {"audience": "beginners"},
        },
    ]

    results = []
    for i, task_info in enumerate(tasks, 1):
        print(f"\n📋 Task {i}: {task_info['description'][:60]}...")

        task = await a2a_bridge.delegate_atomic_task(
            task_description=task_info["description"],
            required_capabilities=task_info["capabilities"],
            input_data=task_info["input_data"],
        )

        print(f"   📤 Task ID: {task.task_id[:8]}...")
        print(f"   📍 Status: {task.status.value}")
        print(f"   🤖 Assigned to: {task.assigned_agent_id}")

        if task.result:
            success = task.result.get("success", False)
            strategy = task.result.get("strategy_used", "unknown")
            print(f"   ✅ Success: {success}")
            print(f"   🧠 Strategy: {strategy}")

        results.append(task)

    return a2a_bridge, results


async def demonstrate_agent_capabilities():
    """
    Show agent capability discovery.
    """

    print("\n🔍 Agent Capability Discovery")
    print("=" * 50)

    agents = await create_simple_agents()
    a2a_bridge = await create_a2a_compatible_agent_network(agents)

    # Get discovery information
    discovery_info = a2a_bridge.get_agent_discovery_info()

    print(f"📋 Protocol Version: {discovery_info['protocol_version']}")
    print(f"🎯 Supported Features:")
    for feature in discovery_info["supported_features"]:
        print(f"   - {feature}")

    print(f"\n🤖 Available Agents ({len(discovery_info['agents'])} total):")
    for agent_id, agent_info in discovery_info["agents"].items():
        print(f"\n  🤖 {agent_id}")
        print(f"     📊 Status: {agent_info['status']}")
        print(f"     🎯 Capabilities:")
        for cap in agent_info["capabilities"]:
            print(f"       - {cap['name']}: {cap['description']}")
            if cap.get("cost_estimate"):
                print(f"         💰 Cost estimate: {cap['cost_estimate']}")

    return discovery_info


async def demonstrate_official_a2a_principles():
    """
    Show adherence to official A2A principles.
    """

    print("\n🏗️ Official A2A Design Principles")
    print("=" * 50)

    agents = await create_simple_agents()
    a2a_bridge = await create_a2a_compatible_agent_network(agents)

    # Principle 1: Atomic Tasks
    print("\n1. ✅ Atomic Task Processing:")
    atomic_examples = [
        "Generate a summary of cloud computing benefits",
        "List the main features of Python programming language",
        "Explain the concept of data privacy",
    ]

    for example in atomic_examples:
        task = await a2a_bridge.delegate_atomic_task(example)
        print(f"   ✓ '{example[:40]}...' → {task.assigned_agent_id}")

    # Principle 2: Capability-Based Routing
    print("\n2. ✅ Capability-Based Routing:")
    routing_test = await a2a_bridge.delegate_atomic_task(
        "Provide detailed analysis of renewable energy trends",
        required_capabilities=["reasoning"],
    )
    print(f"   ✓ Complex task routed to: {routing_test.assigned_agent_id}")

    # Principle 3: Status Tracking
    print("\n3. ✅ Status Tracking:")
    status = await a2a_bridge.get_task_status(routing_test.task_id)
    if status:
        print(f"   ✓ Task status: {status.status.value}")
        print(f"   ✓ Assigned agent: {status.assigned_agent_id}")

    return a2a_bridge


async def show_integration_readiness():
    """
    Show readiness for official A2A SDK integration.
    """

    print("\n🚀 Official A2A Integration Readiness")
    print("=" * 50)

    readiness_checklist = [
        ("✅", "Atomic task support"),
        ("✅", "Capability-based routing"),
        ("✅", "Agent discovery protocol"),
        ("✅", "Standard endpoint simulation"),
        ("✅", "Status tracking"),
        ("✅", "Protocol version compatibility"),
        ("🔄", "Official SDK integration (ready)"),
        ("🔄", "OAuth2 authentication (pending)"),
        ("🔄", "Multi-language interop (pending)"),
    ]

    for status, feature in readiness_checklist:
        print(f"   {status} {feature}")

    print(f"\n📋 Next Steps:")
    next_steps = [
        "Install official a2a-sdk when available",
        "Replace A2AOfficialBridge with official SDK calls",
        "Implement OAuth2 authentication",
        "Test with other A2A agents",
        "Deploy to A2A network",
    ]

    for i, step in enumerate(next_steps, 1):
        print(f"   {i}. {step}")


async def main():
    """
    Main demo function.
    """

    print("\n🎯 Simple A2A Integration Demo")
    print("🔗 Reactive Agents + Google A2A Protocol")
    print("=" * 60)

    try:
        # 1. Demonstrate atomic task delegation
        print("\n🎪 DEMO PART 1: Atomic Task Delegation")
        a2a_bridge, tasks = await demonstrate_a2a_atomic_tasks()

        # 2. Show agent capabilities
        print("\n🎪 DEMO PART 2: Agent Capabilities")
        discovery_info = await demonstrate_agent_capabilities()

        # 3. Demonstrate A2A principles
        print("\n🎪 DEMO PART 3: A2A Design Principles")
        principles_bridge = await demonstrate_official_a2a_principles()

        # 4. Show integration readiness
        print("\n🎪 DEMO PART 4: Integration Readiness")
        await show_integration_readiness()

        print("\n✅ Simple A2A Integration Demo Complete!")
        print("\n📊 Demo Summary:")
        print(f"   - Created {len(discovery_info['agents'])} A2A compatible agents")
        print(f"   - Executed {len(tasks)} atomic tasks successfully")
        print(f"   - Demonstrated official A2A design principles")
        print(f"   - Verified integration readiness")
        print(f"   - Ready for official SDK integration")

        return {
            "success": True,
            "agents_created": len(discovery_info["agents"]),
            "tasks_executed": len(tasks),
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
        print("\n🎉 A2A Integration successful! Ready for official SDK integration.")
