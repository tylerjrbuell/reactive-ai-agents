from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Dict,
    Any,
    Optional,
    List,
    TYPE_CHECKING,
    Callable,
    Awaitable,
    Union,
    Tuple,
)
import json
import asyncio

from reactive_agents.core.types.agent_types import (
    AgentThinkChainResult,
    AgentThinkResult,
)
from reactive_agents.core.types.reasoning_types import (
    ReasoningContext,
    TaskGoalEvaluationContext,
)
from reactive_agents.core.types.reasoning_component_types import (
    ComponentType,
    ComponentMetadata,
    StepResult,
    Plan,
    PlanStep,
    ReflectionResult,
    ToolExecutionResult,
    CompletionResult,
    ErrorRecoveryResult,
    StrategyTransitionResult,
)
from reactive_agents.core.reasoning.engine import ReasoningEngine
from reactive_agents.core.reasoning.goal_evaluator import TaskGoalEvaluator
from reactive_agents.core.reasoning.strategies.base import (
    StrategyCapabilities,
    StrategyResult,
    BaseReasoningStrategy,
)
from reactive_agents.core.reasoning.prompts.base import BasePrompt, PromptKey

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class BaseComponent:
    """Base class for all strategy components."""

    def __init__(
        self,
        engine: ReasoningEngine,
        component_type: ComponentType,
        name: str,
        description: str = "",
    ):
        """
        Initialize a strategy component.

        Args:
            engine: Reasoning engine
            component_type: Type of component
            name: Component name
            description: Component description
        """
        self.engine = engine
        self.component_type = component_type
        self.name = name
        self.description = description or f"{name} component"
        self.context = engine.context
        self.agent_logger = engine.context.agent_logger

    def __str__(self) -> str:
        """String representation."""
        return f"{self.name} ({self.component_type.value})"

    def log_usage(self, message: str) -> None:
        """Log component usage."""
        if self.agent_logger:
            self.agent_logger.debug(f"{self.name}: {message}")

    def _get_prompt(self, prompt_name: PromptKey, **kwargs) -> str:
        """
        Get a prompt by name.

        Args:
            prompt_name: Name of the prompt

            **kwargs: Context to pass to the prompt (for fallback compatibility)

        Returns:
            Prompt instance
        """
        prompt = self.engine.get_prompt(prompt_name, **kwargs)
        if not prompt:
            raise ValueError(f"Prompt {prompt_name} not found")
        return prompt  # type: ignore


class ThinkingComponent(BaseComponent):
    """Component for generating thoughts and reasoning."""

    def __init__(
        self,
        engine: ReasoningEngine,
        name: str = "Thinking",
        description: str = "Generates thoughts and reasoning",
    ):
        super().__init__(engine, ComponentType.THINKING, name, description)

    async def think(self, prompt: str, **kwargs) -> Optional[AgentThinkResult]:
        """
        Generate thoughts using the engine's thinking capabilities.

        Args:
            prompt: Prompt for thinking

        Returns:
            Structured thinking result
        """
        self.log_usage(f"Thinking with prompt: {prompt[:50]}...")
        return await self.engine.think(prompt, **kwargs)

    async def think_chain(
        self, use_tools: bool = False
    ) -> Optional[AgentThinkChainResult]:
        """
        Execute a chain of thoughts.

        Args:
            use_tools: Whether to allow tool usage during thinking

        Returns:
            Result of the thinking chain
        """
        self.log_usage(f"Running thinking chain with tools: {use_tools}")
        return await self.engine.think_chain(use_tools)


class PlanningComponent(BaseComponent):
    """Component for task planning."""

    def __init__(
        self,
        engine: ReasoningEngine,
        name: str = "Planning",
        description: str = "Generates task plans",
    ):
        super().__init__(engine, ComponentType.PLANNING, name, description)

    async def generate_plan(
        self, task: str, reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """
        Generate a plan for a task by having a conversation with the model.
        """
        self.log_usage(f"Generating plan for task: {task[:50]}...")

        # 1. Generate the user message that asks for a plan.
        plan_prompt_str = self._get_prompt("plan_generation", task=task)

        # 2. Add the user message to the session to guide the planning.
        # The context manager should handle this to maintain a clean history.
        context_manager = self.engine.get_context_manager()

        # 3. Call think_chain, which uses the conversational history.
        thinking_result = await self.engine.think(
            plan_prompt_str, response_format="json"
        )
        if thinking_result:
            context_manager.add_message(
                role="assistant",
                content=f"Plan: {thinking_result.result_json}",
            )

        self.log_usage(f"Raw thinking_result from plan generation: {thinking_result}")

        if not thinking_result or not thinking_result.result_json:
            if self.agent_logger:
                self.agent_logger.warning("Failed to generate valid plan")
            return {"plan_steps": [], "error": "Failed to generate valid plan"}

        # The assistant's response (the plan) is automatically added to the history
        # by the think_chain call. We just return the structured JSON.
        return thinking_result.result_json


class ToolExecutionComponent(BaseComponent):
    """Component for tool execution."""

    def __init__(
        self,
        engine: ReasoningEngine,
        name: str = "ToolExecution",
        description: str = "Executes tools and processes results",
    ):
        super().__init__(engine, ComponentType.TOOL_EXECUTION, name, description)

    async def execute_tools(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls using the engine.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            Results of tool execution
        """
        self.log_usage(f"Executing {len(tool_calls)} tool calls")
        return await self.engine.execute_tools(tool_calls)

    async def select_and_execute_tool(
        self, task: str, step_description: str
    ) -> Dict[str, Any]:
        """
        Select and execute the most appropriate tool for a task conversationally.
        """
        self.log_usage(f"Selecting tool for: {step_description[:50]}...")

        # 1. Generate the user message that asks the model to perform the step.
        tool_prompt_str = self._get_prompt(
            "tool_selection", task=task, step_description=step_description
        )

        # 2. Add the message to context.
        context_manager = self.engine.get_context_manager()
        context_manager.add_message(role="user", content=tool_prompt_str)

        # 3. Call think_chain, which will now respond with a tool call.
        # The agent's `_think_chain` should handle the tool execution automatically.
        thinking_result = await self.engine.think_chain(use_tools=True)

        if not thinking_result or not thinking_result.tool_calls:
            if self.agent_logger:
                self.agent_logger.warning(
                    "Failed to select a valid tool conversationally"
                )
            return {"error": "Failed to select a valid tool"}

        # The engine and agent handle the execution and adding the tool_result message.
        # We can return information about the call.
        return {
            "tool_calls": thinking_result.tool_calls,
            "reasoning": thinking_result.content,
        }


class ReflectionComponent(BaseComponent):
    """Component for reflection and evaluation."""

    def __init__(
        self,
        engine: ReasoningEngine,
        name: str = "Reflection",
        description: str = "Reflects on progress and results",
    ):
        super().__init__(engine, ComponentType.REFLECTION, name, description)

    async def reflect_on_progress(
        self,
        task: str,
        last_result: Dict[str, Any],
        reasoning_context: ReasoningContext,
    ) -> Dict[str, Any]:
        """
        Reflect on current progress conversationally.
        """
        self.log_usage("Reflecting on progress")

        # 1. Generate the user message that asks the model to reflect.
        reflection_prompt_str = self._get_prompt(
            "reflection", task=task, last_result=last_result
        )

        # 2. Add the message to context.
        context_manager = self.engine.get_context_manager()

        # 3. Call think_chain to get the reflection.
        reflection_result = await self.engine.think(
            reflection_prompt_str, response_format="json"
        )

        if reflection_result:
            context_manager.add_message(
                role="assistant",
                content=f"Reflection: {reflection_result.result_json}",
            )

        if not reflection_result or not reflection_result.result_json:
            if self.agent_logger:
                self.agent_logger.warning("Failed to generate reflection")
            return {
                "goal_achieved": False,
                "completion_score": 0.0,
                "next_action": "continue",
                "error": "Failed to generate reflection",
            }

        return reflection_result.result_json

    async def reflect_on_plan_progress(
        self,
        task: str,
        current_step_index: int,
        plan_steps: List[Dict[str, Any]],
        last_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Reflect on plan progress.

        Args:
            task: The current task
            current_step_index: Index of the current step
            plan_steps: List of plan steps
            last_result: Result from last step

        Returns:
            Plan progress reflection
        """
        self.log_usage(
            f"Reflecting on plan progress (step {current_step_index+1}/{len(plan_steps)})"
        )

        # Use the engine's prompt system to generate a plan reflection prompt
        plan_reflection_prompt_str = self._get_prompt(
            "plan_progress_reflection",
            task=task,
            current_step_index=current_step_index,
            plan_steps=plan_steps,
            last_result=last_result,
        )

        reflection_result = await self.engine.think(
            plan_reflection_prompt_str, response_format="json"
        )

        if not reflection_result:
            if self.agent_logger:
                self.agent_logger.warning("Failed to generate plan reflection")
            return {
                "current_step_status": "unknown",
                "overall_completion_score": 0.0,
                "next_action": "continue",
                "error": "Failed to generate plan reflection",
            }

        return reflection_result.result_json


class TaskEvaluationComponent(BaseComponent):
    """Component for task completion evaluation."""

    def __init__(
        self,
        engine: ReasoningEngine,
        name: str = "TaskEvaluation",
        description: str = "Evaluates task completion status",
    ):
        super().__init__(engine, ComponentType.EVALUATION, name, description)

    async def evaluate_task_completion(
        self,
        task: str,
        progress_summary: str = "",
        latest_output: str = "",
        execution_log: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate if a task is complete.

        Args:
            task: The task to evaluate
            progress_summary: Summary of progress
            latest_output: Latest output from the agent
            execution_log: Log of execution
            meta: Additional metadata

        Returns:
            Evaluation result
        """
        self.log_usage(f"Evaluating task completion for: {task[:50]}...")

        eval_context = TaskGoalEvaluationContext(
            task_description=task,
            progress_summary=progress_summary,
            latest_output=latest_output,
            execution_log=execution_log,
            meta=meta or {},
        )
        print(f"eval_context: {eval_context}")

        # Use the TaskGoalEvaluator directly
        evaluator = TaskGoalEvaluator(
            model_provider=self.engine.context.model_provider,
            agent_context=self.context,
            eval_context=eval_context,
        )

        try:
            evaluation = await evaluator.get_goal_evaluation()

            if self.agent_logger:
                self.agent_logger.debug(f"Raw evaluation result: {evaluation}")

            result = {
                "is_complete": evaluation.completion,
                "completion_score": evaluation.completion_score,
                "reasoning": evaluation.reasoning,
                "missing_requirements": evaluation.missing_requirements,
            }

            if self.agent_logger:
                self.agent_logger.debug(f"Processed evaluation result: {result}")

            return result

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Task evaluation failed: {e}")
            return {
                "is_complete": False,
                "completion_score": 0.0,
                "reasoning": f"Evaluation failed: {str(e)}",
                "missing_requirements": ["Task evaluation encountered an error"],
            }


class CompletionComponent(BaseComponent):
    """Component for task completion and final answer generation."""

    def __init__(
        self,
        engine: ReasoningEngine,
        name: str = "Completion",
        description: str = "Handles task completion and final answer generation",
    ):
        super().__init__(engine, ComponentType.COMPLETION, name, description)

    async def generate_final_answer(
        self,
        task: str,
        execution_summary: str = "",
        reflection: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a final answer conversationally.
        """
        self.log_usage("Attempting to generate final answer")

        # 1. Generate the user message that asks for a final answer.
        final_answer_prompt_str = self._get_prompt(
            "final_answer",
            task=task,
            execution_summary=execution_summary,
            reflection=reflection or {},
            **kwargs,
        )

        # 2. Add the message to context.
        context_manager = self.engine.get_context_manager()
        context_manager.add_message(role="user", content=final_answer_prompt_str)

        # 3. Call think_chain to get the final answer.
        final_answer_result = await self.engine.think_chain(use_tools=False)

        if not final_answer_result or not final_answer_result.result_json:
            if self.agent_logger:
                self.agent_logger.warning("Failed to generate valid final answer")
            return {
                "final_answer": f"I attempted to complete the task: {task}, but encountered some issues.",
                "confidence": 0.2,
                "error": "Failed to generate valid final answer",
            }

        if self.agent_logger:
            self.agent_logger.debug(
                f"Generated final answer: {final_answer_result.result_json}"
            )
        return final_answer_result.result_json


class ErrorHandlingComponent(BaseComponent):
    """Component for error handling and recovery."""

    def __init__(
        self,
        engine: ReasoningEngine,
        name: str = "ErrorHandling",
        description: str = "Handles errors and recovery",
    ):
        super().__init__(engine, ComponentType.ERROR_HANDLING, name, description)

    async def handle_error(
        self,
        task: str,
        error_context: str,
        error_count: int,
        last_error: str,
    ) -> Dict[str, Any]:
        """
        Handle an error and generate recovery steps.

        Args:
            task: The current task
            error_context: Context in which the error occurred
            error_count: Count of errors so far
            last_error: The last error message

        Returns:
            Error handling result
        """
        self.log_usage(f"Handling error: {last_error[:50]}...")

        # Use the engine's prompt system to generate an error recovery prompt
        prompt_obj = self._get_prompt(
            "error_recovery",
            task=task,
            error_context=error_context,
            error_count=error_count,
            last_error=last_error,
        )
        error_prompt_str = self._get_prompt(
            "error_recovery",
            task=task,
            error_context=error_context,
            error_count=error_count,
            last_error=last_error,
        )
        recovery_result = await self.engine.think(
            error_prompt_str, response_format="json"
        )

        if not recovery_result or not recovery_result.result_json.get(
            "recovery_action"
        ):
            if self.agent_logger:
                self.agent_logger.warning("Failed to generate valid error recovery")
            return {
                "recovery_action": "retry",
                "error": "Failed to generate valid error recovery",
            }

        return recovery_result.result_json


class MemoryIntegrationComponent(BaseComponent):
    """Component for integrating with agent memory."""

    def __init__(
        self,
        engine: ReasoningEngine,
        name: str = "MemoryIntegration",
        description: str = "Integrates with agent memory",
    ):
        super().__init__(engine, ComponentType.MEMORY_INTEGRATION, name, description)

    async def get_relevant_memories(
        self, task: str, max_items: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get memories relevant to the current task.

        Args:
            task: The current task
            max_items: Maximum number of memories to retrieve

        Returns:
            List of relevant memories
        """
        self.log_usage(f"Retrieving relevant memories for: {task[:50]}...")

        if (
            not hasattr(self.context, "memory_manager")
            or not self.context.memory_manager
        ):
            return []

        try:
            memory_manager = self.context.memory_manager

            # Check if memory manager is ready
            is_ready = False
            try:
                if hasattr(memory_manager, "is_ready"):
                    is_ready = memory_manager.is_ready()  # type: ignore
            except:
                is_ready = False

            if hasattr(memory_manager, "get_context_memories") and is_ready:
                relevant_memories = await memory_manager.get_context_memories(  # type: ignore
                    task, max_items=max_items
                )
                return relevant_memories
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.warning(f"Failed to retrieve memories: {e}")

        return []

    def preserve_context(self, key: str, value: Any) -> None:
        """
        Preserve important context.

        Args:
            key: Context key
            value: Context value
        """
        self.engine.preserve_context(key, value)

    def get_preserved_context(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get preserved context.

        Args:
            key: Optional specific key to retrieve

        Returns:
            Preserved context
        """
        return self.engine.get_preserved_context(key)


class StrategyTransitionComponent(BaseComponent):
    """Component for strategy transition decisions."""

    def __init__(
        self,
        engine: ReasoningEngine,
        name: str = "StrategyTransition",
        description: str = "Handles strategy transitions",
    ):
        super().__init__(engine, ComponentType.STRATEGY_TRANSITION, name, description)

    async def should_switch_strategy(
        self,
        current_strategy: str,
        available_strategies: List[str],
        reasoning_context: ReasoningContext,
        performance_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Determine if the strategy should be switched.

        Args:
            current_strategy: Current strategy name
            available_strategies: Available strategies
            reasoning_context: Current reasoning context
            performance_metrics: Performance metrics

        Returns:
            Strategy transition decision
        """
        self.log_usage(f"Evaluating strategy transition from {current_strategy}")

        # Use the engine's prompt system to generate a strategy transition prompt
        prompt_obj = self._get_prompt(
            "strategy_transition",
            current_strategy=current_strategy,
            available_strategies=available_strategies,
            performance_metrics=performance_metrics,
        )
        transition_prompt_str = self._get_prompt(
            "strategy_transition",
            current_strategy=current_strategy,
            available_strategies=available_strategies,
            performance_metrics=performance_metrics,
        )
        transition_result = await self.engine.think(
            transition_prompt_str, response_format="json"
        )

        if not transition_result or not transition_result.result_json.get(
            "should_switch"
        ):
            if self.agent_logger:
                self.agent_logger.warning(
                    "Failed to generate valid strategy transition decision"
                )
            return {"should_switch": False, "recommended_strategy": None}

        return transition_result.result_json


class ComponentRegistry:
    """Registry of strategy components."""

    def __init__(self, engine: ReasoningEngine):
        """
        Initialize the registry.

        Args:
            engine: Reasoning engine
        """
        self.engine = engine
        self._components: Dict[str, BaseComponent] = {}

        # Register standard components
        self.register(ThinkingComponent(engine))
        self.register(PlanningComponent(engine))
        self.register(ToolExecutionComponent(engine))
        self.register(ReflectionComponent(engine))
        self.register(TaskEvaluationComponent(engine))
        self.register(CompletionComponent(engine))
        self.register(ErrorHandlingComponent(engine))
        self.register(MemoryIntegrationComponent(engine))
        self.register(StrategyTransitionComponent(engine))

    def register(self, component: BaseComponent) -> None:
        """
        Register a component.

        Args:
            component: The component to register
        """
        self._components[component.name] = component

    def get(self, name: str) -> Optional[BaseComponent]:
        """
        Get a component by name.

        Args:
            name: Component name

        Returns:
            The component or None if not found
        """
        return self._components.get(name)

    def get_by_type(self, component_type: ComponentType) -> List[BaseComponent]:
        """
        Get components by type.

        Args:
            component_type: Component type

        Returns:
            List of components of the specified type
        """
        return [
            component
            for component in self._components.values()
            if component.component_type == component_type
        ]

    def get_all(self) -> List[BaseComponent]:
        """
        Get all registered components.

        Returns:
            List of all components
        """
        return list(self._components.values())


class ComponentBasedStrategy(BaseReasoningStrategy):
    """
    Base class for reasoning strategies built from components.

    This provides a foundation for building custom strategies
    by composing reusable components.
    """

    def __init__(self, engine: ReasoningEngine):
        """
        Initialize the strategy.

        Args:
            engine: Reasoning engine
        """
        super().__init__(engine)
        self.registry = ComponentRegistry(engine)

        # Cache components for quick access
        self._thinking = self.registry.get("Thinking")
        self._planning = self.registry.get("Planning")
        self._tools = self.registry.get("ToolExecution")
        self._reflection = self.registry.get("Reflection")
        self._evaluation = self.registry.get("TaskEvaluation")
        self._completion = self.registry.get("Completion")
        self._error_handling = self.registry.get("ErrorHandling")
        self._memory = self.registry.get("MemoryIntegration")
        self._transition = self.registry.get("StrategyTransition")

    @property
    def name(self) -> str:
        """Return the name of this strategy."""
        return "ComponentBasedStrategy"

    @property
    def capabilities(self) -> List[StrategyCapabilities]:
        """Return the capabilities this strategy supports."""
        return [
            StrategyCapabilities.TOOL_EXECUTION,
            StrategyCapabilities.REFLECTION,
            StrategyCapabilities.ADAPTATION,
        ]

    @abstractmethod
    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """Execute one iteration of this reasoning strategy."""
        pass

    # Convenience methods to access components
    async def think(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Generate thoughts."""
        if not self._thinking or not isinstance(self._thinking, ThinkingComponent):
            raise ValueError("Thinking component not found")
        result = await self._thinking.think(prompt)
        return result.result_json if result else {}

    async def plan(
        self, task: str, reasoning_context: ReasoningContext
    ) -> Dict[str, Any]:
        """Generate a plan."""
        if not self._planning or not isinstance(self._planning, PlanningComponent):
            raise ValueError("Planning component not found")
        return await self._planning.generate_plan(task, reasoning_context)

    async def execute_tool(self, task: str, step_description: str) -> Dict[str, Any]:
        """Select and execute a tool."""
        if not self._tools or not isinstance(self._tools, ToolExecutionComponent):
            raise ValueError("ToolExecution component not found")
        return await self._tools.select_and_execute_tool(task, step_description)

    async def reflect(
        self,
        task: str,
        last_result: Dict[str, Any],
        reasoning_context: ReasoningContext,
    ) -> Dict[str, Any]:
        """Reflect on progress."""
        if not self._reflection or not isinstance(
            self._reflection, ReflectionComponent
        ):
            raise ValueError("Reflection component not found")
        return await self._reflection.reflect_on_progress(
            task, last_result, reasoning_context
        )

    async def evaluate(
        self, task: str, progress_summary: str = "", **kwargs
    ) -> Dict[str, Any]:
        """Evaluate task completion."""
        if not self._evaluation or not isinstance(
            self._evaluation, TaskEvaluationComponent
        ):
            raise ValueError("TaskEvaluation component not found")

        if self.agent_logger:
            self.agent_logger.debug("Starting task evaluation")

        result = await self._evaluation.evaluate_task_completion(
            task, progress_summary=progress_summary, **kwargs
        )

        if self.agent_logger:
            self.agent_logger.debug(f"Task evaluation result: {result}")

        return result

    async def handle_error(
        self, task: str, error_context: str, error_count: int, last_error: str
    ) -> Dict[str, Any]:
        """Handle an error."""
        if not self._error_handling or not isinstance(
            self._error_handling, ErrorHandlingComponent
        ):
            raise ValueError("ErrorHandling component not found")
        return await self._error_handling.handle_error(
            task, error_context, error_count, last_error
        )

    async def get_memories(self, task: str, max_items: int = 5) -> List[Dict[str, Any]]:
        """Get relevant memories."""
        if not self._memory or not isinstance(self._memory, MemoryIntegrationComponent):
            raise ValueError("MemoryIntegration component not found")
        return await self._memory.get_relevant_memories(task, max_items)

    async def complete_task(
        self, task: str, execution_summary: str = "", **kwargs
    ) -> dict:
        """
        Use the CompletionComponent to generate a final answer and completion result.
        All strategies should call this method to ensure consistent completion logic.
        """
        if not self._completion or not isinstance(
            self._completion, CompletionComponent
        ):
            raise ValueError("Completion component not found")

        if self.agent_logger:
            self.agent_logger.debug("Generating final answer for completed task")

        # Generate final answer
        completion_result = await self._completion.generate_final_answer(
            task, execution_summary=execution_summary, **kwargs
        )

        if self.agent_logger:
            self.agent_logger.debug(f"Completion result: {completion_result}")

        # Ensure we have a valid final answer
        if not completion_result.get("final_answer"):
            if self.agent_logger:
                self.agent_logger.warning("No final answer in completion result")
            return {
                "status": "incomplete",
                "reason": "Failed to generate final answer",
                "completion_result": completion_result,
            }

        return completion_result

    @property
    def context_manager(self):
        """Get the context manager."""
        return self.engine.get_context_manager()

    async def initialize(self, task: str, reasoning_context: ReasoningContext) -> None:
        """
        Initialize the strategy for a new task.

        Args:
            task: The task to be executed
            reasoning_context: Initial reasoning context
        """
        # Set active strategy in the context manager
        self.context_manager.set_active_strategy(self.name)

        # Create a task window in the context
        self.task_window = self.context_manager.add_window(
            name=f"{self.name}_task",
            importance=0.8,
        )

        # Add initialization message
        self.context_manager.add_message(
            "system", f"Beginning task using {self.name} strategy: {task}"
        )
