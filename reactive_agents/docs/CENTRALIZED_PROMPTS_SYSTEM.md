# Centralized Prompts System

The reactive-agents framework now includes a centralized prompt system that provides reusable, dynamic prompts across different reasoning strategies. This system eliminates hardcoded prompts and ensures consistency while enabling powerful context-aware prompt generation.

## Overview

The centralized prompt system consists of:

1. **Base Prompt Classes**: Abstract base classes for creating dynamic prompts
2. **Specialized Prompt Classes**: Ready-to-use prompts for common scenarios
3. **Context Integration**: Automatic inclusion of relevant context information
4. **Memory Integration**: Seamless integration with the agent's memory system

## Key Benefits

- **Reusability**: Prompts can be used across different reasoning strategies
- **Consistency**: Standardized formatting and structure across all prompts
- **Dynamic Context**: Automatically includes relevant context information
- **Memory Integration**: Leverages agent memory for better prompt quality
- **Maintainability**: Easy to update and extend prompts in one place
- **Flexibility**: Supports both structured and free-form prompt generation

## Available Prompt Classes

### Base Classes

- `BasePrompt`: Abstract base class for all prompts
- `PromptContext`: Data model for dynamic prompt context

### Specialized Prompts

- `SystemPrompt`: Basic system prompts for agent initialization
- `TaskPlanningPrompt`: Planning and decision-making prompts
- `ReflectionPrompt`: Reflection and evaluation prompts
- `FinalAnswerPrompt`: Comprehensive final answer generation
- `StrategyTransitionPrompt`: Strategy switching decision prompts
- `ErrorRecoveryPrompt`: Error handling and recovery prompts
- `ToolSelectionPrompt`: Tool selection and configuration prompts

## Usage Examples

### Basic Usage

```python
from reactive_agents.core.reasoning.prompts.base import (
    FinalAnswerPrompt,
    ReflectionPrompt,
    TaskPlanningPrompt,
)

# Create prompt instances
final_answer_prompt = FinalAnswerPrompt(agent_context)
reflection_prompt = ReflectionPrompt(agent_context)
planning_prompt = TaskPlanningPrompt(agent_context)

# Generate prompts with context
final_prompt = final_answer_prompt.generate(
    task="Research AI developments",
    reflection=reflection_data,
    reasoning_strategy="reflect_decide_act"
)

reflection_prompt_text = reflection_prompt.generate(
    task="Research AI developments",
    last_result=last_action_result,
    reasoning_strategy="reflect_decide_act"
)
```

### Advanced Usage with Memory

```python
from reactive_agents.core.reasoning.prompts.base import FinalAnswerPrompt

# Create prompt with memory integration
final_answer_prompt = FinalAnswerPrompt(agent_context)

# The prompt automatically includes relevant memories
final_prompt = final_answer_prompt.generate(
    task="Research AI developments",
    reflection={
        "completion_score": 0.85,
        "success_indicators": ["Found relevant papers", "Identified trends"],
        "progress_assessment": "Task nearly complete"
    },
    # Memory is automatically retrieved and included
    reasoning_strategy="reflect_decide_act"
)
```

### Custom Prompt Creation

```python
from reactive_agents.core.reasoning.prompts.base import BasePrompt

class CustomPrompt(BasePrompt):
    """Custom prompt for specific use cases."""

    def generate(self, **kwargs) -> str:
        """Generate custom prompt based on context."""
        context = self._get_prompt_context(**kwargs)

        # Access context information
        task = context.task
        tools = context.available_tools
        memories = context.relevant_memories

        # Build your custom prompt
        prompt = f"""Custom prompt for: {task}

Available tools: {', '.join(tools)}
Memory insights: {len(memories)} relevant experiences

{your_prompt_content}
        """

        return prompt
```

## Integration with Reasoning Strategies

The centralized prompt system integrates seamlessly with reasoning strategies:

```python
from reactive_agents.core.reasoning.prompts.base import FinalAnswerPrompt

class MyReasoningStrategy(BaseReasoningStrategy):
    def __init__(self, context):
        super().__init__(context)
        self.final_answer_prompt = FinalAnswerPrompt(context)

    async def generate_final_answer(self, task, reflection):
        # Use centralized prompt instead of hardcoded text
        prompt = self.final_answer_prompt.generate(
            task=task,
            reflection=reflection,
            reasoning_strategy="my_strategy"
        )

        # Generate response using the structured prompt
        response = await self._generate_structured_response(
            system_prompt=prompt,
            user_prompt="Generate final answer",
            use_tools=False
        )

        return response
```

## Context Integration

The `PromptContext` class automatically gathers relevant information:

```python
class PromptContext(BaseModel):
    task: str = ""
    role: str = ""
    instructions: str = ""
    current_datetime: str = ""
    available_tools: List[str] = []
    tool_signatures: List[Dict[str, Any]] = []
    recent_messages: List[Dict[str, Any]] = []
    iteration_count: int = 0
    reasoning_strategy: Optional[str] = None
    task_classification: Optional[Dict[str, Any]] = None
    success_indicators: List[str] = []
    error_context: Optional[str] = None
    relevant_memories: List[Dict[str, Any]] = []
    memory_stats: Optional[Dict[str, Any]] = None
    tool_usage_history: List[str] = []
```

## Memory Integration

The prompt system automatically integrates with the agent's memory:

- **Relevant Memories**: Automatically retrieves memories related to the current task
- **Memory Stats**: Includes memory statistics for context
- **Learning Integration**: Uses past experiences to inform prompt generation
- **Pattern Recognition**: Identifies successful and failed patterns from memory

## Best Practices

1. **Use Specialized Prompts**: Prefer specialized prompt classes over generic ones
2. **Pass Context**: Always provide relevant context parameters
3. **Memory Integration**: Let the system handle memory retrieval automatically
4. **Structured Responses**: Use prompts that generate structured JSON responses
5. **Error Handling**: Include error context when available
6. **Strategy Awareness**: Make prompts strategy-aware for better results

## Migration Guide

### From Hardcoded Prompts

**Before:**

```python
final_answer_prompt = f"""Based on task completion, provide final answer for: {task}
Context: {context}
Progress: {progress}
Generate comprehensive final answer."""
```

**After:**

```python
final_answer_prompt = FinalAnswerPrompt(context)
prompt = final_answer_prompt.generate(
    task=task,
    reflection=reflection_data,
    reasoning_strategy="reflect_decide_act"
)
```

### From Static Prompts

**Before:**

```python
reflection_prompt = "Reflect on the current progress and state."
```

**After:**

```python
reflection_prompt = ReflectionPrompt(context)
prompt = reflection_prompt.generate(
    task=task,
    reasoning_strategy="reflect_decide_act",
    last_result=last_action_result
)
```

## Demo

Run the centralized prompts demo to see the system in action:

```bash
python reactive_agents/examples/centralized_prompts_demo.py
```

This demo showcases:

- FinalAnswerPrompt with reflection integration
- StrategyTransitionPrompt for strategy switching
- ErrorRecoveryPrompt for error handling
- ReflectionPrompt for progress evaluation
- TaskPlanningPrompt for decision making
- PromptContext system for automatic context gathering

## Future Enhancements

- **Prompt Templates**: Support for template-based prompt generation
- **Localization**: Multi-language prompt support
- **Prompt Optimization**: A/B testing and prompt performance optimization
- **Custom Prompt Registry**: Plugin system for custom prompt types
- **Prompt Caching**: Caching for frequently used prompts
