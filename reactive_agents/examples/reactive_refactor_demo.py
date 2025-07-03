#!/usr/bin/env python3
"""
Demonstration of the refactored reactive agent framework.

This script showcases the new features:
1. Task classification
2. Dynamic reasoning strategy selection
3. Per-iteration planning
4. Strategy switching
5. Modular prompt system
"""

import asyncio
import sys
import json
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from reactive_agents.app.agents.reactive_agent import ReactiveAgentV2
from reactive_agents.core.types.agent_types import ReactAgentConfig


async def demo_task_classification():
    """Demonstrate task classification capabilities."""
    print("🔍 Task Classification Demo")
    print("=" * 50)

    # Simple config for testing
    config = ReactAgentConfig(
        agent_name="TaskClassifierDemo",
        provider_model_name="ollama:cogito:14b",  # Use a small model for demo
        role="Task Classification Assistant",
        instructions="You are a helpful assistant that demonstrates task classification.",
        max_iterations=10,
        tool_use_enabled=False,  # Disable tools for this demo
        log_level="info",
    )

    test_tasks = [
        "What is the capital of France?",  # simple_lookup
        "Write a creative story about a robot who learns to paint",  # creative_generation
        "Analyze the pros and cons of renewable energy vs fossil fuels",  # analysis
        "Create a 5-step plan to learn Python programming",  # planning
        "Search for recent AI research papers and summarize the key findings",  # multi_step + tool_required
    ]

    async with ReactiveAgentV2(config) as agent:
        for i, task in enumerate(test_tasks, 1):
            print(f"\n{i}. Task: {task}")

            # Classify the task using the agent's classifier
            if hasattr(agent.context, "task_classifier"):
                classification = await agent.context.task_classifier.classify_task(task)
                print(f"   Classification: {classification.task_type.value}")
                print(f"   Confidence: {classification.confidence:.2f}")
                print(f"   Complexity: {classification.complexity_score:.2f}")
                print(f"   Estimated Steps: {classification.estimated_steps}")
            else:
                print("   Task classifier not available")


async def demo_strategy_selection():
    """Demonstrate reasoning strategy selection and switching."""
    print("\n🧠 Reasoning Strategy Demo")
    print("=" * 50)

    config = ReactAgentConfig(
        agent_name="StrategyDemo",
        provider_model_name="ollama:cogito:14b",
        role="Strategy Demonstration Assistant",
        instructions="You are a helpful assistant demonstrating different reasoning strategies.",
        max_iterations=5,
        tool_use_enabled=False,
        log_level="info",
    )

    test_scenarios = [
        {
            "task": "What is 2 + 2?",
            "expected_strategy": "reactive",
            "description": "Simple arithmetic should use reactive strategy",
        },
        {
            "task": "Explain the process of photosynthesis step by step",
            "expected_strategy": "reflect_decide_act",
            "description": "Explanation task should use structured reasoning",
        },
        {
            "task": "Plan a complex software project with multiple components, dependencies, and team coordination",
            "expected_strategy": "plan_execute_reflect",
            "description": "Complex planning should use planning strategy",
        },
    ]

    async with ReactiveAgentV2(config) as agent:
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{i}. Scenario: {scenario['description']}")
            print(f"   Task: {scenario['task']}")
            print(f"   Expected Strategy: {scenario['expected_strategy']}")

            # Get initial strategy selection
            if hasattr(agent.execution_engine, "task_classifier"):
                classification = (
                    await agent.execution_engine.task_classifier.classify_task(
                        scenario["task"]
                    )
                )
                selected_strategy = (
                    agent.execution_engine.strategy_manager.select_initial_strategy(
                        classification
                    )
                )
                print(f"   Selected Strategy: {selected_strategy.value}")
                print(f"   Task Type: {classification.task_type.value}")


async def demo_reactive_execution():
    """Demonstrate the reactive execution with a real task."""
    print("\n⚡ Reactive Execution Demo")
    print("=" * 50)

    config = ReactAgentConfig(
        agent_name="ReactiveDemo",
        provider_model_name="ollama:cogito:14b",
        role="Helpful Assistant",
        instructions="You are a helpful assistant that can analyze problems and provide solutions.",
        max_iterations=10,
        tool_use_enabled=True,  # Enable tools for this demo
        log_level="info",
    )

    task = "Analyze the benefits and drawbacks of remote work and provide a balanced perspective"

    print(f"Task: {task}")
    print("\nStarting reactive execution...")

    async with ReactiveAgentV2(config) as agent:
        result = await agent.run(task)

        print(f"\nExecution Results:")
        print(f"Status: {result.get('status')}")
        print(f"Strategy Used: {result.get('reasoning_strategy')}")
        print(f"Iterations: {result.get('iterations')}")
        print(f"Completion Score: {result.get('completion_score', 0):.2f}")

        if result.get("final_answer"):
            print(f"\nFinal Answer (first 200 chars):")
            print(f"{result['final_answer'][:200]}...")

        # Show reasoning context
        reasoning_context = agent.get_reasoning_context()
        print(f"\nReasoning Context:")
        print(f"Strategy: {reasoning_context.get('current_strategy')}")
        print(f"Iterations: {reasoning_context.get('iteration_count')}")
        print(f"Errors: {reasoning_context.get('error_count')}")


async def demo_strategy_switching():
    """Demonstrate dynamic strategy switching."""
    print("\n🔄 Strategy Switching Demo")
    print("=" * 50)

    config = ReactAgentConfig(
        agent_name="SwitchingDemo",
        provider_model_name="ollama:cogito:14b",
        role="Adaptive Assistant",
        instructions="You are an assistant that adapts your reasoning approach based on the task.",
        max_iterations=8,
        tool_use_enabled=False,
        log_level="info",
    )

    # Test different strategies on the same task
    task = "Explain machine learning concepts"
    strategies = ["reactive", "reflect_decide_act", "adaptive"]

    async with ReactiveAgentV2(config) as agent:
        print(f"Task: {task}")
        print(f"Available Strategies: {agent.get_available_strategies()}")

        for strategy in strategies:
            print(f"\n--- Testing with {strategy} strategy ---")

            result = await agent.run_with_strategy(task, strategy)

            print(f"Final Strategy: {result.get('reasoning_strategy')}")
            print(f"Iterations: {result.get('iterations')}")
            print(f"Status: {result.get('status')}")


async def main():
    """Run all demonstrations."""
    print("🎯 Reactive Agent Framework Refactor Demo")
    print("=" * 60)
    print("This demo showcases the new reactive capabilities:")
    print("1. Task classification")
    print("2. Dynamic reasoning strategy selection")
    print("3. Per-iteration reactive planning")
    print("4. Strategy switching")
    print("5. Modular architecture")
    print("=" * 60)

    try:
        await demo_task_classification()
        await demo_strategy_selection()
        await demo_reactive_execution()
        await demo_strategy_switching()

        print("\n✅ All demonstrations completed successfully!")
        print("\nKey improvements demonstrated:")
        print("- Task classification informs strategy selection")
        print("- Strategies can switch dynamically based on performance")
        print("- Per-iteration planning instead of static plans")
        print("- Modular, extensible architecture")
        print("- Enhanced observability and control")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
