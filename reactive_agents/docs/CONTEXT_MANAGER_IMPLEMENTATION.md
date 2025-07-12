# ContextManager Implementation Summary

## Overview

The `ContextManager` is a central component of the reactive agents framework that provides strategy-aware context management. It works alongside the component-based reasoning system to intelligently manage conversation context, optimize memory usage, and improve agent performance.

## Key Features

1. **Strategy-Aware Context Management**

   - Dynamically adjusts context handling based on active reasoning strategy
   - Optimizes token usage for different strategy patterns
   - Preserves important context while pruning less relevant information

2. **Message Organization**

   - Categorizes messages by role (user, assistant, tool, system)
   - Groups related messages into logical "context windows"
   - Provides utilities for filtering and accessing specific message types

3. **Smart Context Pruning**

   - Implements configurable pruning thresholds based on model capabilities
   - Preserves high-importance messages and windows
   - Summarizes pruned sections to maintain context continuity
   - Respects strategy-specific preservation rules

4. **Integration with Component-Based Reasoning**
   - Direct integration with `ComponentBasedStrategy` classes
   - Enables reasoning strategies to access optimized context
   - Automatically sets up strategy-specific context windows
   - Provides metadata tracking for message importance

## Implementation Details

### Core Classes

1. **ContextManager**

   - Central class for all context management operations
   - Configurable for different strategies and models
   - Provides APIs for adding, accessing, and manipulating context

2. **MessageRole**

   - Enum for standardizing message roles across the system
   - Includes USER, ASSISTANT, TOOL, SYSTEM, and SUMMARY roles

3. **ContextWindow**
   - Data class representing a logical window of related messages
   - Tracks start/end indices, importance, and metadata

### Strategy-Specific Optimization

The ContextManager maintains strategy-specific configurations for:

- Summarization frequency
- Token threshold multipliers
- Message threshold multipliers
- Preserved message roles

For example:

- **Reactive Strategy**: Minimal context with focus on recent messages
- **Plan-Execute-Reflect Strategy**: Preserves planning context and summaries
- **Reflect-Decide-Act Strategy**: Prioritizes reflection messages and system guidance

### Component Integration

The `ComponentBasedStrategy` class provides direct access to the ContextManager, allowing strategies to:

1. Add strategy-specific context windows
2. Track important phases of execution
3. Preserve critical information across iterations
4. Optimize context for different reasoning patterns

### Context Pruning Process

1. **Detection**

   - Monitors token count and message count against thresholds
   - Adjusts thresholds based on active strategy

2. **Summarization**

   - Creates summary messages for prunable sections
   - Preserves semantic content while reducing token usage

3. **Preservation Rules**

   - Applies strategy-specific preservation rules
   - Respects custom rules added by strategies
   - Preserves windows with high importance

4. **Token Management**
   - Maintains optimized context size for different models
   - Dynamically adjusts based on model capabilities

## Usage Example

```python
# Get the context manager from engine
context_manager = engine.get_context_manager()

# Set active strategy
context_manager.set_active_strategy("plan_execute_reflect")

# Create a context window for planning phase
planning_window = context_manager.add_window("planning_phase", importance=0.9)

# Add messages to the window
context_manager.add_message(MessageRole.ASSISTANT, "I'll create a plan with 3 steps.")
context_manager.add_message(
    MessageRole.ASSISTANT,
    "Here's my plan: Step 1...",
    {"type": "planning", "preserve": True}
)

# Close the window when planning is complete
context_manager.close_window(planning_window)

# Get strategy-optimized context
optimized_context = context_manager.get_context_for_strategy()
```

## Benefits

1. **Memory Efficiency**: Optimizes context usage to maximize the effective context window
2. **Strategy-Specific Optimization**: Adapts context management to different reasoning patterns
3. **Improved Agent Performance**: Maintains important context while reducing noise
4. **Modularity**: Separates context management concerns from reasoning logic
5. **Extensibility**: Easily supports new strategies and context management approaches
