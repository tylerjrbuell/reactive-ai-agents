from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pydantic import BaseModel
from datetime import datetime
import json

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class PromptContext(BaseModel):
    """Context data for dynamic prompt generation."""

    task: str = ""
    role: str = ""
    instructions: str = ""
    current_datetime: str = ""
    current_day_of_week: str = ""
    current_timezone: str = ""
    model_info: str = ""
    available_tools: List[str] = []
    tool_signatures: List[Dict[str, Any]] = []
    recent_messages: List[Dict[str, Any]] = []
    iteration_count: int = 0
    reasoning_strategy: Optional[str] = None
    task_classification: Optional[Dict[str, Any]] = None
    success_indicators: List[str] = []
    error_context: Optional[str] = None
    # Memory integration
    relevant_memories: List[Dict[str, Any]] = []
    memory_stats: Optional[Dict[str, Any]] = None
    # Tool usage tracking
    tool_usage_history: List[str] = []


class BasePrompt(ABC):
    """Base class for dynamic prompts."""

    def __init__(self, context: "AgentContext"):
        self.context = context

    @abstractmethod
    def generate(self, **kwargs) -> str:
        """Generate the prompt based on context and additional kwargs."""
        pass

    def _get_prompt_context(self, **kwargs) -> PromptContext:
        """Extract common context information for prompt generation."""
        now = datetime.now()

        # Get relevant memories - prefer passed memories, fall back to sync retrieval
        relevant_memories = kwargs.get("relevant_memories", [])
        memory_stats = None

        # If no memories were passed, try to get them synchronously (for backward compatibility)
        if (
            not relevant_memories
            and hasattr(self.context, "memory_manager")
            and self.context.memory_manager
        ):
            try:
                memory_manager = self.context.memory_manager
                is_ready = False
                try:
                    if hasattr(memory_manager, "is_ready"):
                        is_ready = memory_manager.is_ready()  # type: ignore
                except:
                    is_ready = False

                if hasattr(memory_manager, "get_context_memories") and is_ready:
                    # Only try sync retrieval if we're not in an event loop
                    import asyncio

                    try:
                        # Check if we're in an event loop
                        asyncio.get_running_loop()
                        # We're in an event loop - skip sync retrieval
                        if (
                            hasattr(self.context, "agent_logger")
                            and self.context.agent_logger
                        ):
                            self.context.agent_logger.debug(
                                "In event loop, skipping sync memory retrieval. Pass memories as parameter."
                            )
                    except RuntimeError:
                        # No event loop running, run synchronously
                        relevant_memories = asyncio.run(
                            memory_manager.get_context_memories(  # type: ignore
                                kwargs.get("task", self.context.session.current_task),
                                max_items=5,
                            )
                        )

                # Get memory statistics
                if hasattr(memory_manager, "get_memory_stats"):
                    memory_stats = memory_manager.get_memory_stats()  # type: ignore
            except Exception as e:
                # Log error but continue without memory
                if hasattr(self.context, "agent_logger") and self.context.agent_logger:
                    self.context.agent_logger.debug(
                        f"Failed to retrieve memories for prompt: {e}"
                    )

        # Remove all PromptContext fields from kwargs to avoid duplicate parameter errors
        prompt_context_fields = set(PromptContext.__fields__.keys())
        kwargs_without_duplicates = {
            k: v for k, v in kwargs.items() if k not in prompt_context_fields
        }

        return PromptContext(
            task=kwargs.get("task", self.context.session.current_task),
            role=self.context.role,
            instructions=self.context.instructions,
            current_datetime=now.strftime("%Y-%m-%d %H:%M:%S"),
            current_day_of_week=now.strftime("%A"),
            current_timezone=str(now.astimezone().tzinfo),
            model_info=self.context.provider_model_name,
            available_tools=[
                tool.get("function", {}).get("name", "")
                for tool in self.context.get_tool_signatures()
            ],
            tool_signatures=self.context.get_tool_signatures(),
            recent_messages=(
                self.context.session.messages[-5:]
                if self.context.session.messages
                else []
            ),
            iteration_count=self.context.session.iterations,
            reasoning_strategy=kwargs.get("reasoning_strategy"),
            task_classification=kwargs.get("task_classification"),
            success_indicators=kwargs.get("success_indicators", []),
            error_context=kwargs.get("error_context"),
            relevant_memories=relevant_memories,
            memory_stats=memory_stats,
            **kwargs_without_duplicates,
        )


class SystemPrompt(BasePrompt):
    """Dynamic system prompt generator."""

    def generate(self, **kwargs) -> str:
        """Generate a minimal system prompt based on static context only."""
        context = self._get_prompt_context(**kwargs)

        base_prompt = f"""You are an advanced AI agent. Your role: {context.role}\n\nTask: {context.task}\nInstructions: {context.instructions}\nCurrent Date/Time: {context.current_datetime} ({context.current_day_of_week}, {context.current_timezone})\nModel: {context.model_info}\nIteration: {context.iteration_count}\n"""

        # Add tool info if available
        if context.available_tools:
            base_prompt += f"\nAvailable Tools: {', '.join(context.available_tools)}"

        # Add static guidelines only
        base_prompt += "\n\n# Guidelines: Respond to the user's task using your tools and reasoning abilities. When you have gathered the necessary information, use the final_answer tool to provide your complete response."

        return base_prompt


class TaskPlanningPrompt(BasePrompt):
    """Dynamic prompt for task planning."""

    def generate(self, **kwargs) -> str:
        """Generate a planning prompt based on current context and strategy."""
        context = self._get_prompt_context(**kwargs)

        prompt = f"""You are a task planning specialist. Create a single, optimal next step for this task.

Current Task: {context.task}
Available Tools: {', '.join(context.available_tools)}
Current Iteration: {context.iteration_count}"""

        if context.task_classification:
            prompt += f"\nTask Type: {context.task_classification.get('task_type', 'unknown')}"
            prompt += f"\nEstimated Complexity: {context.task_classification.get('complexity_score', 0.5):.1f}"

        if context.recent_messages:
            prompt += f"\nRecent Context: {json.dumps(context.recent_messages[-2:], indent=2)}"

        if context.error_context:
            prompt += f"\nPrevious Error: {context.error_context}"

        # Add full tool signatures for better tool understanding
        if context.tool_signatures:
            prompt += f"\n\nAvailable Tool Signatures:\n{json.dumps(context.tool_signatures, indent=2)}"

        # Add memory context if available
        if context.relevant_memories:
            prompt += f"\n\n# Relevant Past Experiences"
            for i, memory in enumerate(context.relevant_memories[:2], 1):
                memory_type = memory.get("metadata", {}).get("memory_type", "unknown")
                content = memory.get("content", "")[:150]
                success = memory.get("metadata", {}).get("success", True)
                prompt += f"\n{i}. [{memory_type.upper()}] {'✅' if success else '❌'}: {content}..."

        prompt += """

Output Format: JSON
{
    "next_step": "<specific action to take>",
    "rationale": "<reasoning for this step>", 
    "tool_needed": "<tool name if required, null otherwise>",
    "parameters": {"<param>": "<value>"},
    "confidence": <float 0.0-1.0>,
    "memory_influence": "<how past experiences influenced this decision>",
    "avoid_patterns": ["<patterns from memory to avoid>"]
}

Guidelines:
- Generate ONE optimal next step, not a full plan
- Be specific about tool usage and parameters
- Use the tool signatures to understand required parameters and their types
- Consider the current context and previous attempts
- Adapt based on task type and complexity
- Focus on making immediate progress toward the goal
- Learn from past experiences - what worked and what didn't
- Avoid repeating patterns that led to failure in similar situations
- Ensure parameters match the expected data types from the tool signatures"""

        return prompt


class ReflectionPrompt(BasePrompt):
    """Dynamic prompt for reflection and evaluation."""

    def generate(self, **kwargs) -> str:
        """Generate a reflection prompt based on current state."""
        context = self._get_prompt_context(**kwargs)
        last_result = kwargs.get("last_result", "No previous result")

        prompt = f"""You are a reflection specialist evaluating progress on a task.

Task: {context.task}
Current Iteration: {context.iteration_count}
Last Action Result: {json.dumps(last_result) if isinstance(last_result, dict) else str(last_result)}

Strategy: {context.reasoning_strategy or 'adaptive'}"""

        if context.task_classification:
            prompt += f"\nTask Type: {context.task_classification.get('task_type')}"

        if context.recent_messages:
            prompt += f"\nRecent Progress: {json.dumps(context.recent_messages[-3:], indent=2)}"

        # Add full tool signatures for better reflection on tool usage
        if context.tool_signatures:
            prompt += f"\n\nAvailable Tool Signatures:\n{json.dumps(context.tool_signatures, indent=2)}"

        prompt += """

Output Format: JSON
{
    "progress_assessment": "<evaluation of current progress>",
    "completion_score": <float 0.0-1.0>,
    "blockers": ["<list of current blockers>"],
    "success_indicators": ["<list of positive indicators>"],
    "next_action_type": "<finalize|continue|switch_strategy>",
    "suggested_strategy": "<strategy name if switching>",
    "learning_insights": ["<insights from past experiences>"],
    "recommendations": ["<specific recommendations based on memory>"]
}

Guidelines:
- Be honest about progress and challenges
- Consider what worked and didn't work in similar past situations
- Identify patterns from your memory that could help or hinder progress
- Provide specific, actionable insights
- Consider if a different approach would be more effective based on past experiences
- Use the tool signatures to understand if the right tools were used with correct parameters
- Evaluate tool usage effectiveness based on the available capabilities"""

        return prompt


class ToolSelectionPrompt(BasePrompt):
    """Dynamic prompt for tool selection and configuration."""

    def generate(self, **kwargs) -> str:
        """Generate a tool selection prompt."""
        context = self._get_prompt_context(**kwargs)
        step_description = kwargs.get("step_description", "Execute the next action")

        if not context.tool_signatures:
            return "No tools available for this task."

        prompt = f"""You are a tool selection expert. Choose and configure the optimal tool for this step.

Step: {step_description}
Task Context: {context.task}

Available Tools:
{json.dumps(context.tool_signatures, indent=2)}"""

        if context.recent_messages:
            prompt += f"\nRecent Context: {json.dumps(context.recent_messages[-2:], indent=2)}"

        prompt += """

Output Format: JSON
{
    "tool_calls": [
        {
            "function": {
                "name": "<tool_name>",
                "arguments": {"<param>": "<value>"}
            }
        }
    ],
    "reasoning": "<why this tool and these parameters>"
}

Guidelines:
- Select the most appropriate tool for the specific step
- Provide accurate parameters based on tool signatures
- Consider the context and previous attempts
- Only use tools that are actually available
- Ensure parameters match the expected data types"""

        return prompt


class FinalAnswerPrompt(BasePrompt):
    """Dynamic prompt for generating final answers based on task completion."""

    def generate(self, **kwargs) -> str:
        """Generate a final answer prompt."""
        context = self._get_prompt_context(**kwargs)
        reflection = kwargs.get("reflection", {})

        # Get completion metrics
        completion_score = reflection.get("completion_score", 0.8)
        success_indicators = reflection.get("success_indicators", [])
        progress_assessment = reflection.get(
            "progress_assessment", "Task appears complete"
        )

        prompt = f"""You are a task completion specialist. Generate a comprehensive final answer based on task context and progress.

Task: {context.task}
Role: {context.role}
Instructions: {context.instructions}

Context and Progress:
- Completion Score: {completion_score}
- Success Indicators: {', '.join(success_indicators)}
- Progress Assessment: {progress_assessment}
- Iteration Count: {context.iteration_count}"""

        # Add memory context if available
        if context.relevant_memories:
            prompt += f"\n\nRelevant Past Experiences:"
            for i, memory in enumerate(context.relevant_memories[:3], 1):
                memory_type = memory.get("metadata", {}).get("memory_type", "unknown")
                content = memory.get("content", "")[:100]
                prompt += f"\n{i}. [{memory_type.upper()}]: {content}..."

        # Add recent progress context
        if context.recent_messages:
            prompt += f"\n\nRecent Progress: {json.dumps(context.recent_messages[-3:], indent=2)}"

        # Add tool usage summary if available
        if context.tool_usage_history:
            prompt += f"\n\nTools Used: {', '.join(context.tool_usage_history)}"

        prompt += """

Output Format: JSON
{
    "final_answer": "<comprehensive answer that directly addresses the original task>",
    "summary": "<brief summary of what was accomplished>",
    "key_findings": ["<finding1>", "<finding2>", ...],
    "confidence": <float 0.0-1.0>,
    "methodology": "<how the task was approached>",
    "limitations": ["<any limitations or caveats>"]
}

Guidelines:
- Provide a comprehensive final answer that directly addresses the original task
- Summarize what was accomplished and how
- Include key findings and insights discovered
- Be honest about confidence level and any limitations
- Use past experiences to inform the quality and completeness of the answer
- Ensure the answer is actionable and useful to the user
- Consider the agent's role and instructions in formulating the response"""

        return prompt


class StrategyTransitionPrompt(BasePrompt):
    """Dynamic prompt for strategy transition decisions."""

    def generate(self, **kwargs) -> str:
        """Generate a strategy transition prompt."""
        context = self._get_prompt_context(**kwargs)
        current_strategy = kwargs.get("current_strategy", "unknown")
        available_strategies = kwargs.get("available_strategies", [])
        performance_metrics = kwargs.get("performance_metrics", {})

        prompt = f"""You are a strategy transition specialist. Analyze the current situation and recommend whether to switch reasoning strategies.

Current Strategy: {current_strategy}
Task: {context.task}
Iteration: {context.iteration_count}
Available Strategies: {', '.join(available_strategies)}

Performance Metrics:
- Error Count: {performance_metrics.get('error_count', 0)}
- Stagnation Count: {performance_metrics.get('stagnation_count', 0)}
- Success Rate: {performance_metrics.get('success_rate', 0.0)}
- Completion Score: {performance_metrics.get('completion_score', 0.0)}"""

        if context.task_classification:
            prompt += f"\nTask Classification: {json.dumps(context.task_classification, indent=2)}"

        if context.recent_messages:
            prompt += f"\nRecent Progress: {json.dumps(context.recent_messages[-3:], indent=2)}"

        prompt += """

Output Format: JSON
{
    "should_switch": <boolean>,
    "recommended_strategy": "<strategy_name or null>",
    "reasoning": "<detailed reasoning for the recommendation>",
    "confidence": <float 0.0-1.0>,
    "trigger": "<what triggered this recommendation>",
    "expected_benefits": ["<benefit1>", "<benefit2>", ...],
    "risks": ["<risk1>", "<risk2>", ...]
}

Guidelines:
- Analyze current strategy performance objectively
- Consider task complexity, type, and requirements
- Evaluate if a different strategy would be more effective
- Factor in iteration count and error patterns
- Consider available tools and their alignment with strategies
- Be conservative about switching - only recommend if clear benefits exist
- Provide specific reasoning for the recommendation"""

        return prompt


class ErrorRecoveryPrompt(BasePrompt):
    """Dynamic prompt for error recovery and adaptation."""

    def generate(self, **kwargs) -> str:
        """Generate an error recovery prompt."""
        context = self._get_prompt_context(**kwargs)
        error_context = kwargs.get("error_context", "")
        error_count = kwargs.get("error_count", 0)
        last_error = kwargs.get("last_error", "")

        prompt = f"""You are an error recovery specialist. Analyze the error situation and recommend recovery actions.

Task: {context.task}
Error Count: {error_count}
Last Error: {last_error}
Error Context: {error_context}

Recent Context: {json.dumps(context.recent_messages[-3:], indent=2)}"""

        if context.available_tools:
            prompt += f"\nAvailable Tools: {', '.join(context.available_tools)}"

        prompt += """

Output Format: JSON
{
    "recovery_action": "<specific action to take>",
    "rationale": "<why this action should work>",
    "alternative_approach": "<alternative if primary action fails>",
    "confidence": <float 0.0-1.0>,
    "error_analysis": "<analysis of what went wrong>",
    "prevention_measures": ["<measure1>", "<measure2>", ...],
    "tool_adjustments": {"<tool_name>": "<adjustment_needed>"}
}

Guidelines:
- Analyze the root cause of the error
- Propose specific, actionable recovery steps
- Consider alternative approaches if the primary recovery fails
- Suggest prevention measures for similar errors
- Adjust tool usage patterns if needed
- Be realistic about recovery chances
- Focus on getting back on track toward the original goal"""

        return prompt
