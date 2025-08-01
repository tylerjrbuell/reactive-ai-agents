"""
Example demonstrating the component-based strategy system with the ContextManager.
"""

import asyncio
import logging
from typing import Dict, Any

from reactive_agents.app.agents.reactive_agent import ReactiveAgent
from reactive_agents.app.builders.agent import ReactiveAgentBuilder
from reactive_agents.core.types.reasoning_types import (
    ReasoningContext,
    ReasoningStrategies,
)
from reactive_agents.core.reasoning.engine import ReasoningEngine
from reactive_agents.core.reasoning.strategies.reactive import ReactiveStrategy
from reactive_agents.core.context.context_manager import MessageRole


async def run_component_based_strategy():
    """Run a demo of the component-based strategy system."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting component-based strategy demo")

    # Build a reactive agent
    builder = ReactiveAgentBuilder()
    agent = await builder.with_model("gpt-4-turbo").with_tools([]).build()

    # Create engine and strategy
    engine = ReasoningEngine(agent.context)
    strategy = ReactiveStrategy(engine)

    # Get context manager
    context_manager = engine.get_context_manager()

    # Initialize the strategy with a task
    task = "Show me how context management works with component-based strategies"
    reasoning_context = ReasoningContext(
        current_strategy=ReasoningStrategies.REACTIVE, iteration_count=0, error_count=0
    )

    logger.info(f"Initializing strategy {strategy.name} for task: {task}")
    await strategy.initialize(task, reasoning_context)

    # Manually add some messages to demonstrate context management
    context_manager.add_message(
        MessageRole.USER, "Can you explain how context windows work?"
    )

    # Create and use a context window
    window = context_manager.add_window("explanation", importance=0.9)
    context_manager.add_message(
        MessageRole.ASSISTANT, "Context windows group related messages together."
    )
    context_manager.add_message(
        MessageRole.ASSISTANT,
        "They help with organization and preservation during pruning.",
    )
    context_manager.add_message(
        MessageRole.USER, "How do they interact with strategies?"
    )
    context_manager.add_message(
        MessageRole.ASSISTANT,
        "Different strategies can optimize context management differently.",
        {"type": "explanation"},
    )
    context_manager.close_window(window)

    # Show current context
    logger.info(f"Current context has {len(context_manager.messages)} messages")
    for i, msg in enumerate(context_manager.messages):
        role = msg.get("role", "unknown")
        content_preview = (
            msg.get("content", "")[:50] + "..."
            if len(msg.get("content", "")) > 50
            else msg.get("content", "")
        )
        logger.info(f"Message {i}: [{role}] {content_preview}")

    # Show windows
    logger.info(f"Context windows: {len(context_manager.windows)}")
    for window in context_manager.windows:
        logger.info(
            f"Window '{window.name}': Messages {window.start_idx}-{window.end_idx}, Importance: {window.importance}"
        )

    # Execute a few iterations of the strategy
    for i in range(3):
        reasoning_context.iteration_count = i + 1
        logger.info(f"Executing iteration {reasoning_context.iteration_count}")

        # Add some messages to simulate conversation
        context_manager.add_message(
            MessageRole.USER, f"This is user message for iteration {i+1}"
        )

        # Execute the strategy
        result = await strategy.execute_iteration(task, reasoning_context)

        logger.info(
            f"Iteration {i+1} result: {result.action}, continue: {result.should_continue}"
        )

        # Force pruning on the second iteration
        if i == 1:
            logger.info("Forcing context pruning")
            context_manager.summarize_and_prune(force=True)

            # Show pruned context
            logger.info(f"After pruning: {len(context_manager.messages)} messages")
            for j, msg in enumerate(context_manager.messages):
                role = msg.get("role", "unknown")
                content_preview = (
                    msg.get("content", "")[:50] + "..."
                    if len(msg.get("content", "")) > 50
                    else msg.get("content", "")
                )
                logger.info(f"Message {j}: [{role}] {content_preview}")

    # Finalize the strategy
    await strategy.finalize(task, reasoning_context)

    logger.info("Component-based strategy demo completed")


if __name__ == "__main__":
    asyncio.run(run_component_based_strategy())
