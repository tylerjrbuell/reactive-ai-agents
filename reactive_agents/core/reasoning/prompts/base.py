from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Literal
from pydantic import BaseModel
from datetime import datetime
import json

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext

# All valid prompt keys for registration and lookup
PromptKey = Literal[
    "system",
    "planning",
    "reflection",
    "plan_generation",
    "step_execution",
    "completion_validation",
    "plan_progress_reflection",
    "error_recovery",
    "final_answer",
    "tool_selection",
    "strategy_transition",
    "plan_extension",
    "task_goal_evaluation",
    "tool_call_system",
    "memory_summarization",
    "ollama_manual_tool",
]


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
        import json  # Add import here for json usage

        context = self._get_prompt_context(**kwargs)
        goal_evaluation = kwargs.get("goal_evaluation", None)

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

        # Add goal evaluation feedback if provided
        if goal_evaluation:
            prompt += (
                f"\n\nGOAL EVALUATION FEEDBACK: {json.dumps(goal_evaluation, indent=2)}"
            )

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
        import json  # Add import here for json usage

        context = self._get_prompt_context(**kwargs)
        last_result = kwargs.get("last_result", {})
        goal_evaluation = kwargs.get("goal_evaluation", None)

        # Extract execution summary from last_result
        execution_summary = last_result.get("execution_summary", [])
        total_steps = last_result.get("total_steps", 0)
        successful_steps = last_result.get("successful_steps", 0)
        failed_steps = last_result.get("failed_steps", 0)
        error_count = last_result.get("error_count", 0)
        reflection_count = last_result.get("reflection_count", 0)
        plan_success_rate = last_result.get("plan_success_rate", 0.0)
        step_success_rate = last_result.get("step_success_rate", 0.0)
        recovery_success_rate = last_result.get("recovery_success_rate", 0.0)
        last_reflection = last_result.get("last_reflection", {})

        prompt = f"""You are a reflection specialist evaluating progress on a task.

Task: {context.task}
Current Iteration: {context.iteration_count}

EXECUTION STATE:
Total Steps: {total_steps}
Successful Steps: {successful_steps}
Failed Steps: {failed_steps}
Error Count: {error_count}
Reflection Count: {reflection_count}

METRICS:
Plan Success Rate: {plan_success_rate:.2f}
Step Success Rate: {step_success_rate:.2f}
Recovery Success Rate: {recovery_success_rate:.2f}

EXECUTION SUMMARY:"""

        # Add detailed execution summary
        if execution_summary:
            for step in execution_summary:
                prompt += f"\n{step.status} {step.step}"
                if step.get("tool_results"):
                    for result in step.tool_results:
                        prompt += f"\n  → {result}"

        # Add previous reflection context if available
        if last_reflection:
            prompt += "\n\nPREVIOUS REFLECTION:"
            prompt += f"\nGoal Achieved: {last_reflection.get('goal_achieved', False)}"
            prompt += (
                f"\nCompletion Score: {last_reflection.get('completion_score', 0.0)}"
            )
            prompt += f"\nNext Action: {last_reflection.get('next_action', 'unknown')}"
            if last_reflection.get("blockers"):
                prompt += f"\nBlockers: {', '.join(last_reflection.blockers)}"

        if context.task_classification:
            prompt += f"\n\nTask Type: {context.task_classification.get('task_type')}"

        if context.recent_messages:
            prompt += f"\n\nRecent Progress: {json.dumps(context.recent_messages[-3:], indent=2)}"

        # Add goal evaluation feedback if provided
        if goal_evaluation:
            prompt += (
                f"\n\nGOAL EVALUATION FEEDBACK: {json.dumps(goal_evaluation, indent=2)}"
            )

        prompt += """

CRITICAL: You MUST respond with ONLY valid JSON. Do not include any other text or explanation.
Your response must be parseable by JSON.parse() without any preprocessing.

{
    "progress_assessment": "<summary of current progress>",
    "goal_achieved": <boolean - true ONLY if ALL steps succeeded AND no errors>,
    "completion_score": <float 0.0-1.0 based on successful_steps/total_steps>,
    "next_action": "<continue|retry|complete>",
    "confidence": <float 0.0-1.0>,
    "blockers": ["<list of current blockers>"],
    "success_indicators": ["<list of positive indicators>"],
    "learning_insights": ["<insights from execution>"]
}

Guidelines (remember to ONLY output valid JSON):
1. Task Completion Logic:
   - Set next_action="complete" if:
     a) All steps are done (even if some failed) AND no retry would help, OR
     b) The task cannot proceed further (e.g., no emails found to process)
   - Set goal_achieved=true ONLY if ALL steps succeeded AND error_count=0
   - You can have next_action="complete" even if goal_achieved=false

2. Retry Logic:
   - Set next_action="retry" ONLY if:
     a) There were actual failures AND
     b) Retrying could reasonably fix the issue
   - Don't retry if the issue cannot be fixed (e.g., no emails exist to process)

3. Continue Logic:
   - Set next_action="continue" ONLY if:
     a) There are remaining steps AND
     b) No blocking errors exist

4. Scoring:
   - Set completion_score based on step_success_rate
   - Set confidence based on success rates and recovery rates
   - List ONLY ACTUAL blockers from execution
   - Include specific success indicators from completed steps
5. The 'progress_assessment' field MUST be a concise summary of the agent's current progress toward the task, referencing any completed steps, partial results, or blockers. This field is REQUIRED for correct operation of the agent."""

        return prompt


class ToolSelectionPrompt(BasePrompt):
    """Dynamic prompt for tool selection and configuration."""

    def generate(self, **kwargs) -> str:
        """Generate a tool selection prompt."""
        import json  # Add import here for json usage

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
        import json  # Add import here for json usage

        context = self._get_prompt_context(**kwargs)
        reflection = kwargs.get("reflection", {})

        # Patch: handle both dict and str for reflection
        if isinstance(reflection, str):
            # Try to parse as JSON, else fallback to empty dict
            try:
                reflection_dict = json.loads(reflection)
                if isinstance(reflection_dict, dict):
                    reflection = reflection_dict
                else:
                    reflection = {}
            except Exception:
                reflection = {}

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
        import json  # Add import here for json usage

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


class PlanGenerationPrompt(BasePrompt):
    """Dynamic prompt for generating granular, tool-focused task plans."""

    def generate(self, **kwargs) -> str:
        """Generate a plan generation prompt focused on granular, tool-based steps."""
        context = self._get_prompt_context(**kwargs)

        prompt = f"""Break down this task into specific, granular tool actions:

TASK: {context.task}

AVAILABLE TOOLS:"""

        # Add concise tool list
        for sig in context.tool_signatures:
            tool_name = sig.get("function", {}).get("name", "unknown")
            tool_desc = sig.get("function", {}).get("description", "")[:60] + "..."
            prompt += f"\n• {tool_name}: {tool_desc}"

        prompt += f"""

Create a plan with small, specific steps. Each step should:
- Use ONE tool to accomplish ONE thing
- Be specific about what data to use
- Build on results from previous steps

GUIDELINES:
- Break complex actions into multiple small steps
- Be specific about tool parameters and data sources
- Ensure each step produces actionable output for the next step
- Aim for 3-7 granular steps total
- Focus on tool usage, not reasoning steps

Be this specific and granular for your task."""

        return prompt


class StepExecutionPrompt(BasePrompt):
    """Simplified prompt for executing individual plan steps with tool focus."""

    def generate(self, **kwargs) -> str:
        """Generate a focused step execution prompt for tool invocation."""
        context = self._get_prompt_context(**kwargs)
        step = kwargs.get("step", "")
        required_tools = kwargs.get("required_tools", [])

        prompt = f"""Execute this step:

STEP: {step}
REQUIRED TOOLS: {', '.join(required_tools) if required_tools else 'Any appropriate tool'}

AVAILABLE TOOLS: {', '.join(context.available_tools)}"""

        # Add minimal context from previous results if provided
        step_context = kwargs.get("context", "")
        if step_context and "Execute step" in step_context:
            # Extract any useful data like email IDs, search results, etc.
            parts = step_context.split(".")
            for part in parts:
                if any(
                    keyword in part.lower()
                    for keyword in ["ids", "results", "found", "email", "data"]
                ):
                    prompt += f"\nPREVIOUS DATA: {part.strip()}"

        prompt += f"""

INSTRUCTIONS:
1. Use the appropriate tool(s) to complete this step
2. If you need specific data from previous steps, look for it in the context
3. Call the tool function directly with proper parameters
4. If this step completes the entire task, use final_answer with your result

Execute the step now."""

        return prompt


class TaskCompletionValidationPrompt(BasePrompt):
    """Dynamic prompt for validating task completion based on execution results."""

    def generate(self, **kwargs) -> str:
        """Generate a task completion validation prompt focused on execution results."""
        import json  # Add import here for json usage

        context = self._get_prompt_context(**kwargs)

        # Get execution-specific context
        execution_summary = kwargs.get("execution_summary", "")
        actions_taken = kwargs.get("actions", [])
        tool_results = kwargs.get("results", "")
        steps_completed = kwargs.get("steps_completed", 0)
        steps_total = kwargs.get("steps_total", 0)
        execution_history = kwargs.get("execution_history", [])

        prompt = f"""TASK COMPLETION VALIDATOR

ORIGINAL TASK: {context.task}

EXECUTION RESULTS ANALYSIS:
Steps Completed: {steps_completed}/{steps_total}

ACTUAL EXECUTION HISTORY:"""

        # Add execution history if available
        if execution_history:
            prompt += "\n"
            for i, step in enumerate(execution_history, 1):
                step_desc = step.get("description", "Unknown step")
                step_success = (
                    "✅ SUCCESS" if step.get("success", False) else "❌ FAILED"
                )
                tool_calls = step.get("tool_calls", [])

                prompt += f"\nStep {i}: {step_desc}"
                prompt += f"\n  Status: {step_success}"

                if tool_calls:
                    for tc in tool_calls:
                        tool_name = tc.get("name", "unknown")
                        tool_result = tc.get("result")
                        if isinstance(tool_result, list) and tool_result:
                            result_summary = str(tool_result[0])[:100]
                        elif isinstance(tool_result, str):
                            result_summary = tool_result[:100]
                        elif tool_result is not None:
                            result_summary = str(tool_result)[:100]
                        else:
                            result_summary = "No result"
                        prompt += f"\n  Tool: {tool_name} -> {result_summary}"

        prompt += f"""

COMPLETION ANALYSIS:
Based ONLY on the execution history above, determine if the original task is complete.

VALIDATION RULES:
1. If all planned steps were executed successfully, the task is COMPLETE
2. If tools returned success status (like "status": "sent"), count as SUCCESS
3. If steps show ✅ SUCCESS status, they are completed
4. Focus ONLY on whether the original task requirements were met

OUTPUT ONLY JSON:
{{
    "is_complete": <true if ALL required components are DONE>,
    "completion_score": <1.0 if complete, 0.0 if not>,
    "reason": "<specific reason based on execution results>",
    "confidence": <1.0 if certain, lower if uncertain>
}}

Remember: Base your decision ONLY on the execution history above. If all steps show SUCCESS status, mark as complete."""

        return prompt


class PlanProgressReflectionPrompt(BasePrompt):
    """Dynamic prompt for reflecting on plan progress."""

    def generate(self, **kwargs) -> str:
        """Generate a plan progress reflection prompt."""
        import json  # Add import here for json usage

        context = self._get_prompt_context(**kwargs)
        current_step_index = kwargs.get("current_step_index", 0)
        plan_steps = kwargs.get("plan_steps", [])
        last_result = kwargs.get("last_result", {})

        prompt = f"""You are a plan progress evaluator. Analyze the current progress and determine next steps.

TASK: {context.task}
CURRENT STEP: {current_step_index + 1}/{len(plan_steps)}
LAST RESULT: {json.dumps(last_result, indent=2)}

PLAN OVERVIEW:"""

        for i, step in enumerate(plan_steps):
            status = (
                "✅"
                if i < current_step_index
                else "⏳" if i == current_step_index else "⏸️"
            )
            # Handle both dict and PlanStep object
            if hasattr(step, "description"):
                step_description = step.description
            else:
                step_description = step.get("description", "No description")
            prompt += f"\n{status} Step {i+1}: {step_description}"

        prompt += (
            f"\nRECENT PROGRESS: {json.dumps(context.recent_messages[-3:], indent=2)}"
        )

        if context.relevant_memories:
            prompt += f"\n\nRELEVANT EXPERIENCES:"
            for i, memory in enumerate(context.relevant_memories[:2], 1):
                content = memory.get("content", "")[:100]
                success = memory.get("metadata", {}).get("success", True)
                prompt += f"\n{i}. {'✅' if success else '❌'}: {content}..."

        prompt += """

Output Format: JSON
{
    "progress_assessment": "<detailed evaluation of current progress>",
    "current_step_status": "<pending|in_progress|completed|failed>",
    "overall_completion_score": <float 0.0-1.0>,
    "blockers": ["<blocker1>", "<blocker2>", ...],
    "next_action": "<continue|extend_plan|complete_task|retry>",
    "confidence": <float 0.0-1.0>,
    "learning_insights": ["<insight1>", "<insight2>", ...],
    "recommendations": ["<recommendation1>", "<recommendation2>", ...]
}

Guidelines:
- Evaluate progress against the original task
- Consider the success of recent steps
- Identify any blockers or issues
- Recommend appropriate next actions
- Learn from similar past experiences
- Be honest about current progress and challenges
- Focus on making progress toward the original goal"""

        return prompt


class PlanExtensionPrompt(BasePrompt):
    """Dynamic prompt for extending plans when needed."""

    def generate(self, **kwargs) -> str:
        """Generate a plan extension prompt."""
        context = self._get_prompt_context(**kwargs)
        current_plan = kwargs.get("current_plan", [])
        completion_gaps = kwargs.get("completion_gaps", [])

        prompt = f"""You are a plan extension specialist. The current plan is complete but the task is not finished.

ORIGINAL TASK: {context.task}
COMPLETION GAPS: {', '.join(completion_gaps)}

CURRENT PLAN:"""

        for i, step in enumerate(current_plan):
            # Handle both string descriptions and dict/object steps
            if isinstance(step, str):
                step_description = step
            elif hasattr(step, "description"):
                step_description = step.description
            else:
                step_description = step.get("description", "No description")
            prompt += f"\n{i+1}. {step_description}"

        prompt += (
            f"\nRECENT PROGRESS: {json.dumps(context.recent_messages[-3:], indent=2)}"
        )
        prompt += f"\nAVAILABLE TOOLS: {', '.join(context.available_tools)}"

        prompt += """

Generate additional steps to complete the remaining task components.

Output Format: JSON
{
    "additional_steps": [
        {
            "step_number": <int>,
            "description": "<specific action to take>",
            "purpose": "<what this step accomplishes>",
            "is_action": <boolean>,
            "required_tools": ["<tool_names>"],
            "success_criteria": "<how to know this step is complete>",
            "addresses_gap": "<which completion gap this addresses>"
        }
    ],
    "rationale": "<why these steps are needed>",
    "confidence": <float 0.0-1.0>
}

Guidelines:
- Address the specific completion gaps identified
- Create focused, actionable steps
- Use available tools appropriately
- Ensure each step has clear success criteria
- Don't duplicate existing completed work
- Focus on what's missing to complete the task"""

        return prompt


class TaskGoalEvaluationPrompt(BasePrompt):
    """
    Reusable prompt for LLM-powered task completion evaluation.
    Generates a prompt for the LLM to assess whether the task is complete, confidence, reasoning, and missing requirements.
    """

    def generate(self, **kwargs) -> str:
        context = self._get_prompt_context(**kwargs)
        progress_summary = kwargs.get("progress_summary", "")
        latest_output = kwargs.get("latest_output", "")
        execution_log = kwargs.get("execution_log", "")
        meta = kwargs.get("meta", {})
        success_criteria = kwargs.get(
            "success_criteria", meta.get("success_criteria", "None provided")
        )

        prompt = f"""You are a task evaluator. Determine whether the task has been completed using the task description and context provided.

Task Description:
{context.task}

Progress Summary:
{progress_summary}

Latest Output:
{latest_output}

Execution Log:
{execution_log}

Success Criteria (if available):
{success_criteria}

Evaluate the following:
1. Is the task completed successfully?
2. What is your confidence score between 0 and 1?
3. If the task is not complete, what is missing?
4. Provide a short reasoning explanation.

Return your answer in JSON format with:
- completion: true/false
- completion_score: float (0.0 to 1.0)
- reasoning: string
- missing_requirements: list
"""
        return prompt


class ToolCallSystemPrompt(BasePrompt):
    """Dynamic system prompt for tool calling generation."""

    def generate(self, **kwargs) -> str:
        """Generate a system prompt for tool calling with dynamic tool signatures."""
        context = self._get_prompt_context(**kwargs)
        tool_signatures = kwargs.get("tool_signatures", context.tool_signatures)
        max_calls = kwargs.get("max_calls", 1)
        task = kwargs.get("task", context.task)

        prompt = f"""Role: Tool Selection and configuration Expert
Objective: Create {max_calls} tool call(s) for the given task using the available tool signatures

Guidelines:
- The tool call must adhere to the specific task
- Use the tool signatures to effectively create tool calls that align with the task
- Use the context provided in conjunction with the tool signatures to create tool calls that align with the task
- Only use valid parameters and valid parameter data types and avoid using tool signatures that are not available
- Check all data types are correct based on the tool signatures provided in available tools to avoid issues when the tool is used
- Pay close attention to the signatures and parameters provided
- Do not try to consolidate multiple tool calls into one call
- Do not try to use tools that are not available

Task: {task}

Available Tool signatures: {tool_signatures}

Output Format: JSON with the following structure:
{{
    "tool_calls": [
        {{
            "function": {{
                "name": "<tool_name>",
                "arguments": <tool_parameters>
            }}
        }}
    ]
}}"""

        return prompt


class MemorySummarizationPrompt(BasePrompt):
    """Dynamic prompt for memory summarization."""

    def generate(self, **kwargs) -> str:
        """Generate a memory summarization prompt based on memory type and content."""
        context = self._get_prompt_context(**kwargs)
        memory_content = kwargs.get("memory_content", "")
        memory_type = kwargs.get("memory_type", "session")

        # Create type-specific summarization prompts
        if memory_type == "session":
            prompt = f"""Summarize this agent session memory in 1-2 sentences. Focus on:
1. What task was accomplished
2. Key result or outcome
3. Any important tools or methods used

Memory content:
{memory_content}

Provide a concise summary that captures the essential information:"""
        elif memory_type == "reflection":
            prompt = f"""Summarize this reflection memory in 1 sentence. Focus on:
1. Key insight or learning
2. What was learned or improved

Memory content:
{memory_content}

Provide a concise summary:"""
        elif memory_type == "tool_result":
            prompt = f"""Summarize this tool result memory in 1 sentence. Focus on:
1. What tool was used
2. Key data or result obtained

Memory content:
{memory_content}

Provide a concise summary:"""
        else:
            prompt = f"""Summarize this {memory_type} memory in 1-2 sentences. Focus on the most important information:

Memory content:
{memory_content}

Provide a concise summary:"""

        return prompt


class OllamaManualToolPrompt(BasePrompt):
    """Dynamic prompt for Ollama manual tool calling generation."""

    def generate(self, **kwargs) -> str:
        """Generate a system prompt for manual tool calling with Ollama models."""
        context = self._get_prompt_context(**kwargs)
        tool_signatures = kwargs.get("tool_signatures", context.tool_signatures)
        max_calls = kwargs.get("max_calls", 1)
        task = kwargs.get("task", context.task)

        prompt = f"""Role: Tool Selection and configuration Expert
Objective: Create {max_calls} tool call(s) for the given task using the available tool signatures

Guidelines:
- The tool call must adhere to the specific task
- Use the tool signatures to effectively create tool calls that align with the task
- Use the context provided in conjunction with the tool signatures to create tool calls that align with the task
- Only use valid parameters and valid parameter data types and avoid using tool signatures that are not available
- Check all data types are correct based on the tool signatures provided in available tools to avoid issues when the tool is used
- Pay close attention to the signatures and parameters provided
- Do not try to consolidate multiple tool calls into one call
- Do not try to use tools that are not available

Task: {task}

Available Tool signatures: {tool_signatures}

Output Format: JSON with the following structure:
{{
    "tool_calls": [
        {{
            "function": {{
                "name": "<tool_name>",
                "arguments": <tool_parameters>
            }}
        }}
    ]
}}"""

        return prompt
