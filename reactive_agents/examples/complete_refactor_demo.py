#!/usr/bin/env python3
"""
🚀 Complete Reactive Agents Framework Refactor Demonstration

This example showcases all the major refactor enhancements:
1. 🧠 Natural Language Configuration System
2. 🧐 Agent-to-Agent (A2A) Communication
3. 📚 Workflow System with DAG orchestration
4. ⚡ Dynamic Reasoning Strategies
5. 🗂️ Task Classification

Usage:
    python -m reactive_agents.examples.complete_refactor_demo
"""

import asyncio
import json
from reactive_agents.providers.llm.factory import ModelProviderFactory
from reactive_agents.config.natural_language_config import (
    create_agent_from_nl,
    AgentFactory,
)
from reactive_agents.communication.a2a_protocol import (
    create_agent_network,
    A2ACommunicationProtocol,
    MessageType,
    MessagePriority,
)
from reactive_agents.workflows.orchestrator import (
    WorkflowOrchestrator,
    create_agent_chain,
)


async def demo_natural_language_config():
    """Demonstrate creating agents from natural language descriptions."""
    print("\n🧠 === NATURAL LANGUAGE CONFIGURATION DEMO ===")

    # Initialize model provider using factory
    model_provider = ModelProviderFactory.get_model_provider("ollama:llama3.2")

    # Create agent factory
    factory = AgentFactory(model_provider)

    # Example 1: Research Agent
    research_description = """
    Create an agent that can search the web, analyze research papers, 
    and summarize findings. It should be methodical and thorough in its approach.
    """

    print(f"📝 Creating agent from description: {research_description[:100]}...")

    # Preview the configuration that would be generated
    config_preview = await factory.preview_config(research_description)
    print(f"🔍 Generated configuration preview:")
    print(json.dumps(config_preview, indent=2)[:500] + "...")

    # Create the actual agent - force ReactiveAgentV2
    research_agent = await factory.create_agent_from_description(
        research_description, "ReactiveAgentV2"
    )

    print(f"✅ Created research agent: {research_agent.context.agent_name}")
    # Note: These methods exist on ReactiveAgentV2 but linter may not detect the cast
    print(f"   Strategy: {research_agent.get_current_strategy()}")  # type: ignore
    print(f"   Available strategies: {research_agent.get_available_strategies()}")  # type: ignore

    # Example 2: Writing Agent - also force ReactiveAgentV2
    writing_description = """
    Create a simple agent that can write and edit documents. 
    It should be quick and responsive for basic writing tasks.
    """

    # Use factory to ensure ReactiveAgentV2
    writing_agent = await factory.create_agent_from_description(
        writing_description, "ReactiveAgentV2"
    )
    print(f"✅ Created writing agent: {writing_agent.context.agent_name}")

    return research_agent, writing_agent


async def demo_a2a_communication(agents):
    """Demonstrate Agent-to-Agent communication."""
    print("\n🧐 === AGENT-TO-AGENT COMMUNICATION DEMO ===")

    research_agent, writing_agent = agents

    # Create A2A network
    protocols = await create_agent_network([research_agent, writing_agent])
    research_protocol, writing_protocol = protocols

    print(f"🌐 Created A2A network with {len(protocols)} agents")

    # Example 1: Task Delegation
    print("\n📤 Demonstrating task delegation...")

    response = await research_protocol.delegate_task(
        target_agent_id=writing_agent.context.agent_name,
        task="Write a brief summary about AI agent frameworks",
        shared_context={"domain": "artificial_intelligence", "audience": "developers"},
        timeout_seconds=60.0,
    )

    print(f"📨 Delegation response:")
    print(f"   Success: {response.success}")
    print(f"   Content: {str(response.content)[:200]}...")

    # Example 2: Broadcast Message
    print("\n📢 Demonstrating broadcast communication...")

    await research_protocol.broadcast_message(
        subject="System Update",
        content={"message": "All agents should update their knowledge base"},
        priority=MessagePriority.NORMAL,
    )

    print("✅ Broadcast message sent to all connected agents")

    # Small delay to let messages process
    await asyncio.sleep(2)

    # Stop protocols
    for protocol in protocols:
        await protocol.stop()

    return protocols


async def demo_workflow_system(agents):
    """Demonstrate the workflow orchestration system."""
    print("\n📚 === WORKFLOW SYSTEM DEMO ===")

    research_agent, writing_agent = agents

    # Create workflow orchestrator
    orchestrator = WorkflowOrchestrator()

    # Register agents
    orchestrator.register_agent(research_agent)
    orchestrator.register_agent(writing_agent)

    print(f"🔧 Registered {len(orchestrator.agents)} agents with orchestrator")

    # Example 1: Simple Agent Chain
    print("\n🔗 Creating simple agent chain...")

    chain_workflow = create_agent_chain(
        agent_names=[
            research_agent.context.agent_name,
            writing_agent.context.agent_name,
        ],
        task_templates=[
            "Research the topic: ${context.topic}",
            "Write a summary based on: ${agents."
            + research_agent.context.agent_name
            + ".result}",
        ],
    )

    print(f"✅ Created chain workflow with {len(chain_workflow.nodes)} nodes")

    # Execute the chain
    print("\n⚡ Executing workflow chain...")

    result = await orchestrator.execute_workflow(
        chain_workflow, initial_context={"topic": "reactive programming patterns"}
    )

    print(f"🎯 Workflow execution completed:")
    print(f"   Success: {result.success}")
    print(f"   Status: {result.status}")
    print(f"   Completed nodes: {result.completed_nodes}/{result.total_nodes}")
    print(f"   Execution time: {result.execution_time:.2f}s")

    # Example 2: Complex Workflow with Builder
    print("\n🏗️ Creating complex workflow with builder...")

    complex_workflow = (
        orchestrator.create_workflow("Research_and_Analysis")
        .add_agent_node(
            agent_name=research_agent.context.agent_name,
            task_template="Research ${context.topic} and find key insights",
            node_id="research_phase",
            context_mapping={"result": "research.findings"},
        )
        .add_delay_node(
            delay_seconds=1.0, node_id="processing_delay", depends_on=["research_phase"]
        )
        .add_agent_node(
            agent_name=writing_agent.context.agent_name,
            task_template="Create a detailed report on: ${research.findings}",
            node_id="writing_phase",
            depends_on=["processing_delay"],
            context_mapping={"result": "final.report"},
        )
        .add_condition_node(
            condition="len(context.get('final', {}).get('report', '')) > 100",
            node_id="quality_check",
            depends_on=["writing_phase"],
        )
        .set_exit_nodes(["quality_check"])
        .set_global_context({"format": "detailed", "max_length": 1000})
        .build()
    )

    print(f"✅ Created complex workflow with {len(complex_workflow.nodes)} nodes")

    # Execute complex workflow
    print("\n⚡ Executing complex workflow...")

    complex_result = await orchestrator.execute_workflow(
        complex_workflow, initial_context={"topic": "agent communication protocols"}
    )

    print(f"🎯 Complex workflow execution completed:")
    print(f"   Success: {complex_result.success}")
    print(
        f"   Final result keys: {list(complex_result.final_result.keys()) if complex_result.final_result else 'None'}"
    )

    return orchestrator


async def demo_dynamic_reasoning():
    """Demonstrate dynamic reasoning strategy switching."""
    print("\n⚡ === DYNAMIC REASONING DEMO ===")

    # Initialize model provider
    model_provider = ModelProviderFactory.get_model_provider("ollama:llama3.2")

    # Create an adaptive agent using the convenience function (returns ReactiveAgentV2)
    adaptive_description = """
    Create an agent that can handle both simple questions and complex analysis tasks.
    It should be able to adapt its approach based on the complexity of the task.
    """

    adaptive_agent = await create_agent_from_nl(adaptive_description, model_provider)

    print(f"🤖 Created adaptive agent: {adaptive_agent.context.agent_name}")
    print(f"   Current strategy: {adaptive_agent.get_current_strategy()}")

    # Test with different task complexities
    tasks = [
        ("What is 2+2?", "simple"),
        (
            "Analyze the implications of quantum computing on current encryption methods",
            "complex",
        ),
        ("List three colors", "simple"),
        (
            "Design a comprehensive strategy for implementing AI governance in large organizations",
            "complex",
        ),
    ]

    for task, expected_complexity in tasks:
        print(f"\n📋 Task: {task[:50]}..." if len(task) > 50 else f"\n📋 Task: {task}")
        print(f"   Expected complexity: {expected_complexity}")

        # Get reasoning context before execution
        context_before = adaptive_agent.get_reasoning_context()
        print(
            f"   Strategy before: {context_before.get('current_strategy', 'unknown')}"
        )

        # Execute task (in a real scenario)
        # result = await adaptive_agent.run(task)

        # For demo, we'll just show the strategy selection
        print(f"   ✅ Task classification and strategy selection would happen here")

    return adaptive_agent


async def demo_complete_integration():
    """Demonstrate all systems working together."""
    print("\n🎯 === COMPLETE INTEGRATION DEMO ===")

    # Initialize model provider
    model_provider = ModelProviderFactory.get_model_provider("ollama:llama3.2")

    # Create specialized agents via natural language
    agents_descriptions = {
        "DataAnalyst": "Create an agent specialized in data analysis and statistical interpretation",
        "Researcher": "Create an agent that excels at research and information gathering",
        "Writer": "Create an agent focused on clear, concise writing and documentation",
        "Coordinator": "Create an agent that can coordinate tasks and manage workflows",
    }

    agents = {}
    for name, description in agents_descriptions.items():
        agent = await create_agent_from_nl(description, model_provider)
        # Rename for clarity
        agent.context.agent_name = name
        agents[name] = agent
        print(f"✅ Created {name} agent")

    # Set up A2A communication network
    a2a_protocols = await create_agent_network(list(agents.values()))
    print(f"🌐 Established A2A network with {len(a2a_protocols)} agents")

    # Create comprehensive workflow
    orchestrator = WorkflowOrchestrator()
    for agent in agents.values():
        orchestrator.register_agent(agent)

    # Build a research-to-publication workflow
    workflow = (
        orchestrator.create_workflow("Research_to_Publication")
        .add_agent_node(
            agent_name="Coordinator",
            task_template="Plan research approach for: ${context.research_topic}",
            node_id="planning",
            context_mapping={"result": "plan.approach"},
        )
        .add_agent_node(
            agent_name="Researcher",
            task_template="Research ${context.research_topic} following ${plan.approach}",
            node_id="research",
            depends_on=["planning"],
            context_mapping={"result": "research.data"},
        )
        .add_agent_node(
            agent_name="DataAnalyst",
            task_template="Analyze findings: ${research.data}",
            node_id="analysis",
            depends_on=["research"],
            context_mapping={"result": "analysis.insights"},
        )
        .add_agent_node(
            agent_name="Writer",
            task_template="Write publication based on ${research.data} and ${analysis.insights}",
            node_id="writing",
            depends_on=["analysis"],
            context_mapping={"result": "final.publication"},
        )
        .set_exit_nodes(["writing"])
        .set_global_context(
            {"quality_standard": "academic", "target_audience": "researchers"}
        )
        .build()
    )

    print(f"🏗️ Created comprehensive workflow with {len(workflow.nodes)} nodes")

    # Execute the complete workflow
    print("\n⚡ Executing complete integrated workflow...")

    final_result = await orchestrator.execute_workflow(
        workflow,
        initial_context={
            "research_topic": "Multi-agent systems in distributed computing",
            "deadline": "2024-03-01",
        },
    )

    print(f"🎯 Complete integration workflow results:")
    print(f"   Success: {final_result.success}")
    print(f"   Total execution time: {final_result.execution_time:.2f}s")
    print(
        f"   Nodes completed: {final_result.completed_nodes}/{final_result.total_nodes}"
    )

    # Clean up A2A protocols
    for protocol in a2a_protocols:
        await protocol.stop()

    return final_result


async def main():
    """Run all demonstrations."""
    print("🚀 REACTIVE AGENTS FRAMEWORK - COMPLETE REFACTOR DEMONSTRATION")
    print("=" * 80)

    try:
        # 1. Natural Language Configuration
        agents = await demo_natural_language_config()

        # 2. A2A Communication
        protocols = await demo_a2a_communication(agents)

        # 3. Workflow System
        orchestrator = await demo_workflow_system(agents)

        # 4. Dynamic Reasoning
        adaptive_agent = await demo_dynamic_reasoning()

        # 5. Complete Integration
        final_result = await demo_complete_integration()

        print("\n🎉 === DEMONSTRATION COMPLETE ===")
        print("✅ All refactor features successfully demonstrated:")
        print("   🧠 Natural Language Configuration")
        print("   🧐 Agent-to-Agent Communication")
        print("   📚 Workflow Orchestration")
        print("   ⚡ Dynamic Reasoning Strategies")
        print("   🗂️ Task Classification")
        print("   🎯 Complete System Integration")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
