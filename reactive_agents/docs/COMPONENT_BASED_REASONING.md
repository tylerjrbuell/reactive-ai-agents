# Component-Based Reasoning System

The component-based reasoning system is a modular approach to creating agent reasoning strategies, providing better organization, reusability, and consistency across different reasoning patterns.

## Architecture

The system is built around the following core concepts:

### 1. Components

Components are reusable building blocks that encapsulate specific functionality:

- **BaseComponent**: Abstract base class for all components
- **ThinkingComponent**: Generates thoughts and reasoning
- **PlanningComponent**: Creates task plans
- **ToolExecutionComponent**: Handles tool selection and execution
- **ReflectionComponent**: Reflects on progress and results
- **TaskEvaluationComponent**: Evaluates task completion status
- **CompletionComponent**: Handles final answer generation
- **ErrorHandlingComponent**: Manages error handling and recovery
- **MemoryIntegrationComponent**: Integrates with agent memory
- **StrategyTransitionComponent**: Manages strategy transitions

### 2. ComponentRegistry

The `ComponentRegistry` manages component instantiation and retrieval:

- Registers standard components
- Allows lookup by name or type
- Supports custom component registration

### 3. ComponentBasedStrategy

Abstract base class for creating strategies from components:

- Extends the standard BaseReasoningStrategy
- Provides convenient access to registered components
- Defines the common structure for component-based strategies
- Handles strategy initialization and finalization

### 4. ContextManager

The `ContextManager` centralizes conversation context management:

- **Strategy-aware message handling**: Optimizes context based on active strategy
- **Message role classification**: Categorizes messages by role (user, assistant, tool, system)
- **Context windows**: Groups related messages logically
- **Preservation rules**: Maintains important context across iterations
- **Dynamic pruning**: Intelligently prunes context based on strategy needs

## Benefits

1. **Consistency**: Standardized approach to common tasks like error handling and reflection
2. **Reusability**: Components can be shared across different strategy implementations
3. **Modularity**: Easier to extend and maintain individual pieces of functionality
4. **Testability**: Components can be tested in isolation
5. **Intelligent context management**: Strategy-specific context handling
6. **Memory integration**: Better management of relevant memories

## Creating a Component-Based Strategy

To create a new strategy:

1. Inherit from `ComponentBasedStrategy`
2. Define the required properties (name, capabilities, description)
3. Implement the `execute_iteration` method using available components
4. Optionally override `initialize` and `finalize` methods

Example:

```python
class EnhancedReactiveStrategy(ComponentBasedStrategy):
    @property
    def name(self) -> str:
        return "enhanced_reactive"

    @property
    def capabilities(self) -> List[StrategyCapabilities]:
        return [
            StrategyCapabilities.TOOL_EXECUTION,
            StrategyCapabilities.REFLECTION,
            StrategyCapabilities.ADAPTATION
        ]

    @property
    def description(self) -> str:
        return "Enhanced reactive reasoning that directly responds to tasks using appropriate tools"

    async def execute_iteration(self, task: str, reasoning_context: ReasoningContext) -> StrategyResult:
        # Use components to implement the strategy logic
        execution_result = await self.execute_tool(task, f"Execute: {task}")
        reflection = await self.reflect(task, execution_result, reasoning_context)

        # Return appropriate StrategyResult
        return StrategyResult(...)
```

## Context Management

The context management system allows you to:

1. **Create context windows** for logical groups of messages:

   ```python
   window = self.context_manager.add_window("planning_phase", importance=0.8)
   # Add messages...
   self.context_manager.close_window(window)
   ```

2. **Add structured messages** with metadata:

   ```python
   self.context_manager.add_message(
       MessageRole.ASSISTANT,
       "I'll search for information about that topic",
       {"type": "planning"}
   )
   ```

3. **Control context preservation**:

   ```python
   self.context_manager.add_preservation_rule(
       lambda msg: msg.get("metadata", {}).get("type") == "critical_info"
   )
   ```

4. **Optimize context** for different strategies:
   ```python
   optimized_messages = self.context_manager.get_context_for_strategy()
   ```

## Registration and Discovery

Component-based strategies are automatically discovered and registered when placed in the `strategies` directory, with no additional registration code required.

## Recommended Practices

1. Use the appropriate component for each function (don't reinvent functionality)
2. Leverage context windows to group related messages
3. Add metadata to messages for better context management
4. Use consistent error handling via the ErrorHandlingComponent
5. Close windows when they're no longer needed
6. Set appropriate window importance to guide pruning decisions
