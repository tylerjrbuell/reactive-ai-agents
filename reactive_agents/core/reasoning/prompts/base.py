from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Literal, Type
from pydantic import BaseModel
from datetime import datetime
import json


from reactive_agents.core.types.prompt_types import (
    ErrorRecoveryOutput,
    FinalAnswerOutput,
    PlanExtensionOutput,
    PlanProgressReflectionOutput,
    SingleStepPlanningOutput,
    StrategyTransitionOutput,
    TaskCompletionValidationOutput,
    TaskGoalEvaluationOutput,
    ToolCallSystemOutput,
    ToolSelectionOutput,
)
from reactive_agents.core.types.reasoning_component_types import Plan, ReflectionResult

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext
    from reactive_agents.core.types.agent_types import AgentThinkResult
    from reactive_agents.core.types.execution_types import ExecutionResult

# All valid prompt keys for registration and lookup
PromptKey = Literal[
    "system",
    "single_step_planning",
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
    "execution_result_summary",
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

    @property
    @abstractmethod
    def output_model(self) -> Optional[Type[BaseModel]]:
        """The Pydantic model describing the expected output format, if any."""
        return None

    @property
    def system_prompt(self) -> Optional[str]:
        """The system prompt to be used for the prompt."""
        return None

    @abstractmethod
    def generate(self, **kwargs) -> str:
        """Generate the prompt based on context and additional kwargs."""
        pass

    async def get_completion(self, **kwargs) -> Optional[AgentThinkResult]:
        """
        Generates the prompt and executes it to get a completion in one step.

        This convenience method uses the prompt's own `output_model` to
        request a structured response from the LLM.

        Args:
            **kwargs: Arguments to be passed to the `generate` method.

        Returns:
            The result from the `think` call, which may include a parsed Pydantic model.
        """
        # 1. Generate the prompt string
        prompt_string = self.generate(**kwargs)
        # 2. Call the engine's think method with the prompt and the output model
        return await self.context.reasoning_engine.think(
            prompt=prompt_string,
            format=self.output_model.model_json_schema() if self.output_model else None,
            system=self.system_prompt,
        )

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
        prompt_context_fields = set(PromptContext.model_fields.keys())
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

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return None

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


class SingleStepPlanningPrompt(BasePrompt):
    """Dynamic prompt for single step planning."""

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return SingleStepPlanningOutput

    def generate(self, **kwargs) -> str:
        """Generate a planning prompt based on current context and strategy."""
        context = self._get_prompt_context(**kwargs)
        goal_evaluation = kwargs.get("goal_evaluation", None)

        prompt = f"""You are a task planning specialist. Create a single, optimal next step for this task.\n\nCurrent Task: {context.task}\nAvailable Tools: {', '.join(context.available_tools)}\nCurrent Iteration: {context.iteration_count}"""

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

        prompt += """\n\nGuidelines:\n- Generate ONE optimal next step, not a full plan\n- Be specific about tool usage and parameters\n- Use the tool signatures to understand required parameters and their types\n- Consider the current context and previous attempts\n- Adapt based on task type and complexity\n- Focus on making immediate progress toward the goal\n- Learn from past experiences - what worked and what didn't\n- Avoid repeating patterns that led to failure in similar situations\n- Ensure parameters match the expected data types from the tool signatures"""

        return prompt


class ReflectionPrompt(BasePrompt):
    """Dynamic prompt for reflection and evaluation."""

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return ReflectionResult

    def generate(self, **kwargs) -> str:
        """Generate a reflection prompt based on current state."""
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

        prompt = f"""You are a reflection specialist evaluating progress on a task.\n\nTask: {context.task}\nCurrent Iteration: {context.iteration_count}\n\nEXECUTION STATE:\nTotal Steps: {total_steps}\nSuccessful Steps: {successful_steps}\nFailed Steps: {failed_steps}\nError Count: {error_count}\nReflection Count: {reflection_count}\n\nMETRICS:\nPlan Success Rate: {plan_success_rate:.2f}\nStep Success Rate: {step_success_rate:.2f}\nRecovery Success Rate: {recovery_success_rate:.2f}\n\nEXECUTION SUMMARY:"""

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

        prompt += """\n\nGuidelines (remember to ONLY output valid JSON):\n1. Task Completion Logic:\n   - Set next_action="complete" if:\n     a) All steps are done (even if some failed) AND no retry would help, OR\n     b) The task cannot proceed further (e.g., no emails found to process)\n   - Set goal_achieved=true ONLY if ALL steps succeeded AND error_count=0\n   - You can have next_action="complete" even if goal_achieved=false\n\n2. Retry Logic:\n   - Set next_action="retry" ONLY if:\n     a) There were actual failures AND\n     b) Retrying could reasonably fix the issue\n   - Don't retry if the issue cannot be fixed (e.g., no emails exist to process)\n\n3. Continue Logic:\n   - Set next_action="continue" ONLY if:\n     a) There are remaining steps AND\n     b) No blocking errors exist\n\n4. Scoring:\n   - Set completion_score based on step_success_rate\n   - Set confidence based on success rates and recovery rates\n   - List ONLY ACTUAL blockers from execution\n   - Include specific success indicators from completed steps\n5. The 'progress_assessment' field MUST be a concise summary of the agent's current progress toward the task, referencing any completed steps, partial results, or blockers. This field is REQUIRED for correct operation of the agent."""

        return prompt


class ToolSelectionPrompt(BasePrompt):
    """Dynamic prompt for tool selection and configuration."""

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return ToolSelectionOutput

    def generate(self, **kwargs) -> str:
        """Generate a tool selection prompt."""
        context = self._get_prompt_context(**kwargs)
        step_description = kwargs.get("step_description", "Execute the next action")

        if not context.tool_signatures:
            return "No tools available for this task."

        prompt = f"""You are a tool selection expert. Choose and configure the optimal tool for this step.\n\nStep: {step_description}\nTask Context: {context.task}\n\nAvailable Tools:\n{json.dumps(context.tool_signatures, indent=2)}"""

        if context.recent_messages:
            prompt += f"\nRecent Context: {json.dumps(context.recent_messages[-2:], indent=2)}"

        prompt += """\n\nGuidelines:\n- Select the most appropriate tool for the specific step\n- Provide accurate parameters based on tool signatures\n- Consider the context and previous attempts\n- Only use tools that are actually available\n- Ensure parameters match the expected data types"""

        return prompt


class FinalAnswerPrompt(BasePrompt):
    """Dynamic prompt for generating final answers based on task completion."""

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return FinalAnswerOutput

    def generate(self, **kwargs) -> str:
        """Generate a final answer prompt."""
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

        # Add messages summary
        messages_summary = "\n".join(
            [
                f"{message.get('role')}: {message.get('content', '')[:100]}..."
                for message in context.recent_messages
                if message.get("content") is not None
            ]
        )

        # Get completion metrics
        completion_score = reflection.get("completion_score", 0.8)
        success_indicators = reflection.get("success_indicators", [])
        progress_assessment = reflection.get(
            "progress_assessment", "Task appears complete"
        )

        prompt = f"""You are a task completion specialist. Generate a comprehensive final answer based on task context and progress.\n\n        Task: {context.task}\n        Role: {context.role}\n        Instructions: {context.instructions}\n\n        Context and Progress:\n        - Completion Score: {completion_score}\n        - Success Indicators: {', '.join(success_indicators)}\n        - Progress Assessment: {progress_assessment}\n        - Iteration Count: {context.iteration_count}"""

        prompt += f"\n\nRecent Messages:\n{messages_summary}\n"

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

        prompt += """\n\nGuidelines:\n- Provide a comprehensive final answer that directly addresses the original task\n- Summarize what was accomplished and how\n- Include key findings and insights discovered\n- Be honest about confidence level and any limitations\n- Use past experiences to inform the quality and completeness of the answer\n- Ensure the answer is actionable and useful to the user\n- Consider the agent's role and instructions in formulating the response"""

        return prompt


class StrategyTransitionPrompt(BasePrompt):
    """Dynamic prompt for strategy transition decisions."""

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return StrategyTransitionOutput

    def generate(self, **kwargs) -> str:
        """Generate a strategy transition prompt."""
        context = self._get_prompt_context(**kwargs)
        current_strategy = kwargs.get("current_strategy", "unknown")
        available_strategies = kwargs.get("available_strategies", [])
        performance_metrics = kwargs.get("performance_metrics", {})

        prompt = f"""You are a strategy transition specialist. Analyze the current situation and recommend whether to switch reasoning strategies.\n\nCurrent Strategy: {current_strategy}\nTask: {context.task}\nIteration: {context.iteration_count}\nAvailable Strategies: {', '.join(available_strategies)}\n\nPerformance Metrics:\n- Error Count: {performance_metrics.get('error_count', 0)}\n- Stagnation Count: {performance_metrics.get('stagnation_count', 0)}\n- Success Rate: {performance_metrics.get('success_rate', 0.0)}\n- Completion Score: {performance_metrics.get('completion_score', 0.0)}"""

        if context.task_classification:
            prompt += f"\nTask Classification: {json.dumps(context.task_classification, indent=2)}"

        if context.recent_messages:
            prompt += f"\nRecent Progress: {json.dumps(context.recent_messages[-3:], indent=2)}"

        prompt += """\n\nGuidelines:\n- Analyze current strategy performance objectively\n- Consider task complexity, type, and requirements\n- Evaluate if a different strategy would be more effective\n- Factor in iteration count and error patterns\n- Consider available tools and their alignment with strategies\n- Be conservative about switching - only recommend if clear benefits exist\n- Provide specific reasoning for the recommendation"""

        return prompt


class ErrorRecoveryPrompt(BasePrompt):
    """Dynamic prompt for error recovery and adaptation."""

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return ErrorRecoveryOutput

    def generate(self, **kwargs) -> str:
        """Generate an error recovery prompt."""
        context = self._get_prompt_context(**kwargs)
        error_context = kwargs.get("error_context", "")
        error_count = kwargs.get("error_count", 0)
        last_error = kwargs.get("last_error", "")

        prompt = f"""You are an error recovery specialist. Analyze the error situation and recommend recovery actions.\n\nTask: {context.task}\nError Count: {error_count}\nLast Error: {last_error}\nError Context: {error_context}\n\nRecent Context: {json.dumps(context.recent_messages[-3:], indent=2)}"""

        if context.available_tools:
            prompt += f"\nAvailable Tools: {', '.join(context.available_tools)}"

        prompt += """\n\nGuidelines:\n- Analyze the root cause of the error\n- Propose specific, actionable recovery steps\n- Consider alternative approaches if the primary recovery fails\n- Suggest prevention measures for similar errors\n- Adjust tool usage patterns if needed\n- Be realistic about recovery chances\n- Focus on getting back on track toward the original goal"""

        return prompt


class PlanGenerationPrompt(BasePrompt):
    """Dynamic prompt for generating granular, tool-focused task plans."""

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return Plan

    @property
    def system_prompt(self) -> Optional[str]:
        return """
    Role: You are a task planner. 
    Objective: You are given a task and a list of tools that can be used to complete the task.
    You need to create a plan for the task using the following guidelines:
    GUIDELINES:
        - Break complex actions into multiple small steps
        - Be specific about tool parameters and data sources
        - Ensure each step produces actionable output for the next step
        - A step should only be considered an action if it includes required tools. Otherwise it should not be an action.
        - Not all steps require tools. Some steps may be purely reasoning steps to invoke thought processes.
        - Choose tools required for each step very carefully and accurately and only if the step is an action.
        - Each step status should be set to pending since the plan is not yet executed.
        - Aim for 3-7 steps total (if possible) but more importantly enough steps to accomplish the task.
        - **IMPORTANT** ALWAYS include verification steps in your plan to ensure each action accomplished what it was intended to do.
        
    """

    def generate(self, **kwargs) -> str:
        """Generate a plan generation prompt focused on granular, tool-based steps."""
        context = self._get_prompt_context(**kwargs)

        prompt = f"""Break down the given task into specific, granular tool actions based on the following context:
        
        TASK: {context.task}
        Role: {context.role}
        Instructions: {context.instructions}
        AVAILABLE TOOLS:
        {json.dumps(context.tool_signatures, indent=2)}"""

        prompt += f"""
        Create a plan with intentional steps that are specific to the task and the tools available.
        """

        return prompt


class TaskCompletionValidationPrompt(BasePrompt):
    """Dynamic prompt for validating task completion based on execution results."""

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return TaskCompletionValidationOutput

    def generate(self, **kwargs) -> str:
        """Generate a task completion validation prompt focused on execution results."""
        context = self._get_prompt_context(**kwargs)

        # Get execution-specific context
        execution_history = kwargs.get("execution_history", [])
        steps_completed = sum(1 for step in execution_history if step.get("success"))
        steps_total = len(execution_history)

        prompt = f"""TASK COMPLETION VALIDATOR\n\nORIGINAL TASK: {context.task}\n\nEXECUTION RESULTS ANALYSIS:\nSteps Completed: {steps_completed}/{steps_total}\n\nACTUAL EXECUTION HISTORY:"""

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

        prompt += f"""\n\nCOMPLETION ANALYSIS:\nBased ONLY on the execution history above, determine if the original task is complete.\n\nVALIDATION RULES:\n1. If all planned steps were executed successfully, the task is COMPLETE\n2. If tools returned success status (like "status": "sent"), count as SUCCESS\n3. If steps show ✅ SUCCESS status, they are completed\n4. Focus ONLY on whether the original task requirements were met\n\nRemember: Base your decision ONLY on the execution history above. If all steps show SUCCESS status, mark as complete."""

        return prompt


class PlanProgressReflectionPrompt(BasePrompt):
    """Dynamic prompt for reflecting on plan progress."""

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return PlanProgressReflectionOutput

    def generate(self, **kwargs) -> str:
        """Generate a plan progress reflection prompt."""
        context = self._get_prompt_context(**kwargs)
        current_step_index = kwargs.get("current_step_index", 0)
        plan_steps = kwargs.get("plan_steps", [])
        last_result = kwargs.get("last_result", {})

        prompt = f"""You are a plan progress evaluator. Analyze the current progress and determine next steps.\n\nTASK: {context.task}\nCURRENT STEP: {current_step_index + 1}/{len(plan_steps)}\nLAST RESULT: {json.dumps(last_result, indent=2)}\n\nPLAN OVERVIEW:"""

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

        prompt += """\n\nGuidelines:\n- Evaluate progress against the original task\n- Consider the success of recent steps\n- Identify any blockers or issues\n- Recommend appropriate next actions\n- Learn from similar past experiences\n- Be honest about current progress and challenges\n- Focus on making progress toward the original goal"""

        return prompt


class PlanExtensionPrompt(BasePrompt):
    """Dynamic prompt for extending plans when needed."""

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return PlanExtensionOutput

    def generate(self, **kwargs) -> str:
        """Generate a plan extension prompt."""
        context = self._get_prompt_context(**kwargs)
        current_plan = kwargs.get("current_plan", [])
        completion_gaps = kwargs.get("completion_gaps", [])

        prompt = f"""You are a plan extension specialist. The current plan is complete but the task is not finished.\n\nORIGINAL TASK: {context.task}\nCOMPLETION GAPS: {', '.join(completion_gaps)}\n\nCURRENT PLAN:"""

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

        prompt += """\n\nGenerate additional steps to complete the remaining task components.\n\nGuidelines:\n- Address the specific completion gaps identified\n- Create focused, actionable steps\n- Use available tools appropriately\n- Ensure each step has clear success criteria\n- Don't duplicate existing completed work\n- Focus on what's missing to complete the task"""

        return prompt


class TaskGoalEvaluationPrompt(BasePrompt):
    """
    Reusable prompt for LLM-powered task completion evaluation.
    Generates a prompt for the LLM to assess whether the task is complete, confidence, reasoning, and missing requirements.
    """

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return TaskGoalEvaluationOutput

    def generate(self, **kwargs) -> str:
        context = self._get_prompt_context(**kwargs)
        progress_summary = kwargs.get("progress_summary", "")
        latest_output = kwargs.get("latest_output", "")
        execution_log = kwargs.get("execution_log", "")
        meta = kwargs.get("meta", {})
        success_criteria = kwargs.get(
            "success_criteria", meta.get("success_criteria", "None provided")
        )

        prompt = f"""You are a task evaluator. Determine whether the task has been completed using the task description and context provided.\n\nTask Description:\n{context.task}\n\nProgress Summary:\n{progress_summary}\n\nLatest Output:\n{latest_output}\n\nExecution Log:\n{execution_log}\n\nSuccess Criteria (if available):\n{success_criteria}\n\nEvaluate the following:\n1. Is the task completed successfully?\n2. What is your confidence score between 0 and 1?\n3. If the task is not complete, what is missing?\n4. Provide a short reasoning explanation."""
        return prompt


class ToolCallSystemPrompt(BasePrompt):
    """Dynamic system prompt for tool calling generation."""

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return ToolCallSystemOutput

    def generate(self, **kwargs) -> str:
        """Generate a system prompt for tool calling with dynamic tool signatures."""
        context = self._get_prompt_context(**kwargs)
        tool_signatures = kwargs.get("tool_signatures", context.tool_signatures)
        max_calls = kwargs.get("max_calls", 1)
        task = kwargs.get("task", context.task)

        prompt = f"""Role: Tool Selection and configuration Expert\nObjective: Create {max_calls} tool call(s) for the given task using the available tool signatures\n\nGuidelines:\n- The tool call must adhere to the specific task\n- Use the tool signatures to effectively create tool calls that align with the task\n- Use the context provided in conjunction with the tool signatures to create tool calls that align with the task\n- Only use valid parameters and valid parameter data types and avoid using tool signatures that are not available\n- Check all data types are correct based on the tool signatures provided in available tools to avoid issues when the tool is used\n- Pay close attention to the signatures and parameters provided\n- Do not try to consolidate multiple tool calls into one call\n- Do not try to use tools that are not available\n\nTask: {task}\n\nAvailable Tool signatures: {tool_signatures}"""

        return prompt


class MemorySummarizationPrompt(BasePrompt):
    """Dynamic prompt for memory summarization."""

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return None

    def generate(self, **kwargs) -> str:
        """Generate a memory summarization prompt based on memory type and content."""
        context = self._get_prompt_context(**kwargs)
        memory_content = kwargs.get("memory_content", "")
        memory_type = kwargs.get("memory_type", "session")

        # Create type-specific summarization prompts
        if memory_type == "session":
            prompt = f"""Summarize this agent session memory in 1-2 sentences. Focus on:\n1. What task was accomplished\n2. Key result or outcome\n3. Any important tools or methods used\n\nMemory content:\n{memory_content}\n\nProvide a concise summary that captures the essential information:"""
        elif memory_type == "reflection":
            prompt = f"""Summarize this reflection memory in 1 sentence. Focus on:\n1. Key insight or learning\n2. What was learned or improved\n\nMemory content:\n{memory_content}\n\nProvide a concise summary:"""
        elif memory_type == "tool_result":
            prompt = f"""Summarize this tool result memory in 1 sentence. Focus on:\n1. What tool was used\n2. Key data or result obtained\n\nMemory content:\n{memory_content}\n\nProvide a concise summary:"""
        else:
            prompt = f"""Summarize this {memory_type} memory in 1-2 sentences. Focus on the most important information:\n\nMemory content:\n{memory_content}\n\nProvide a concise summary:"""

        return prompt


class OllamaManualToolPrompt(BasePrompt):
    """Dynamic prompt for Ollama manual tool calling generation."""

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return ToolCallSystemOutput

    def generate(self, **kwargs) -> str:
        """Generate a system prompt for manual tool calling with Ollama models."""
        context = self._get_prompt_context(**kwargs)
        tool_signatures = kwargs.get("tool_signatures", context.tool_signatures)
        max_calls = kwargs.get("max_calls", 1)
        task = kwargs.get("task", context.task)

        prompt = f"""Role: Tool Selection and configuration Expert
        Objective: Create {max_calls} tool call(s) for the task: {task} using the available tool signatures
        Guidelines:
        - The tool call must adhere to the specific task
        - Use the tool signatures to effectively create tool calls that align with the task
        - Use the context provided in conjunction with the tool signatures to create tool calls that align with the task
        - Only use valid parameters and valid parameter data types and avoid using tool signatures that are not available
        - Check all data types are correct based on the tool signatures provided in available tools to avoid issues when the tool is used
        - Pay close attention to the signatures and parameters provided
        - Do not try to consolidate multiple tool calls into one call
        - Do not try to use tools that are not available
        
        Available Tool signatures: {json.dumps(tool_signatures, indent=2)}"""

        return prompt


class ExecutionResultSummaryPrompt(BasePrompt):
    """Dynamic prompt for generating a summary of an ExecutionResult."""

    @property
    def output_model(self) -> Optional[Type[BaseModel]]:
        return None

    def generate(self, **kwargs) -> str:
        """Generate an execution result summary prompt."""
        execution_result: Optional["ExecutionResult"] = kwargs.get("execution_result")
        if not execution_result:
            return "Cannot generate summary: ExecutionResult not provided."

        agent_name = getattr(self.context, "agent_name", "the agent")

        prompt = f"""
        {execution_result.to_prompt_string()}

        Please generate the summary now, speaking from my perspective as '{agent_name}'.
        The summary should answer the questions: "What did I do?" and "How did I perform?".
        Focus on the key outcomes, decisions, and any notable events like errors or strategy choices.
        """
        import textwrap

        return textwrap.dedent(prompt).strip()
