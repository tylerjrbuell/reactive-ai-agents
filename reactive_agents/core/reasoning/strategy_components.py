from __future__ import annotations
from abc import abstractmethod
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Set,
    TYPE_CHECKING,
)
import time

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
    ReflectionResult,
    ToolExecutionResult,
    CompletionResult,
    ErrorRecoveryResult,
    StrategyTransitionResult,
    Plan,
    PlanStep,
    StepResult,
)
from reactive_agents.core.reasoning.engine import ReasoningEngine
from reactive_agents.core.reasoning.goal_evaluator import TaskGoalEvaluator
from reactive_agents.core.reasoning.strategies.base import (
    StrategyCapabilities,
    StrategyResult,
    BaseReasoningStrategy,
)
from reactive_agents.core.reasoning.prompts.base import BasePrompt, PromptKey
from reactive_agents.core.memory.vector_memory import MemoryItem
from reactive_agents.core.reasoning.steps.base import BaseReasoningStep
from reactive_agents.core.types.reasoning_types import (
    ContinueThinkingPayload,
    StrategyAction,
)
from reactive_agents.core.reasoning.protocols import (
    ComponentResult,
    ComponentContext,
    ReasoningComponent,
    RetryStrategy,
    ComponentCapability,
    ComponentStatus,
    ComponentHealthCheck,
)


if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class BaseComponent:
    """Base class for all strategy components that implements the ReasoningComponent protocol."""

    def __init__(
        self,
        engine: ReasoningEngine,
        component_type: ComponentType,
        name: str,
        description: str = "",
    ) -> None:
        """
        Initialize a strategy component.

        Args:
            engine (ReasoningEngine): Reasoning engine
            component_type (ComponentType): Type of component
            name (str): Component name
            description (str): Component description
        """
        self.engine = engine
        self.component_type = component_type
        self.name = name
        self.description = description or f"{name} component"
        self.context = engine.context
        self.agent_logger = engine.context.agent_logger
        self.status = ComponentStatus.READY
        self.error_count = 0
        self.last_success_time: Optional[float] = None

    def __str__(self) -> str:
        """String representation."""
        return f"{self.name} ({self.component_type.value})"

    def log_usage(self, message: str) -> None:
        """
        Log component usage.

        Args:
            message (str): Message to log
        """
        if self.agent_logger:
            self.agent_logger.debug(f"{self.name}: {message}")

    def _get_prompt(self, prompt_name: PromptKey, **kwargs: Any) -> BasePrompt:
        """
        Get a prompt instance by name.

        Args:
            prompt_name (PromptKey): Name of the prompt
            **kwargs: Context to pass to the prompt (for fallback compatibility)

        Returns:
            BasePrompt: Prompt instance
        """
        prompt = self.engine.get_prompt(prompt_name, **kwargs)
        if not prompt:
            raise ValueError(f"Prompt {prompt_name} not found")
        return prompt

    # Protocol implementation methods
    async def execute(self, context: ComponentContext) -> ComponentResult[Any]:
        """
        Default execute method - should be overridden by subclasses.

        Args:
            context: The execution context

        Returns:
            ComponentResult indicating operation not implemented
        """
        return ComponentResult.error_result(
            error=f"execute() not implemented for {self.name}",
            retry_strategy=RetryStrategy.NONE,
        )

    def can_retry(self, error: Exception, context: ComponentContext) -> bool:
        """
        Default retry logic - can be overridden by subclasses.

        Args:
            error: The exception that occurred
            context: The execution context

        Returns:
            True if the operation should be retried
        """
        # Default: retry on transient errors, but not on validation errors
        transient_errors = (ConnectionError, TimeoutError)
        return isinstance(error, transient_errors) and self.error_count < 3

    def get_capabilities(self) -> Set[str]:
        """
        Return the capabilities this component provides.

        Returns:
            Set of capability names based on component type
        """
        capability_map = {
            ComponentType.THINKING: {ComponentCapability.THINKING.value},
            ComponentType.PLANNING: {ComponentCapability.PLANNING.value},
            ComponentType.TOOL_EXECUTION: {ComponentCapability.TOOL_EXECUTION.value},
            ComponentType.REFLECTION: {ComponentCapability.REFLECTION.value},
            ComponentType.EVALUATION: {ComponentCapability.EVALUATION.value},
            ComponentType.COMPLETION: {ComponentCapability.COMPLETION.value},
            ComponentType.ERROR_HANDLING: {ComponentCapability.ERROR_HANDLING.value},
            ComponentType.MEMORY_INTEGRATION: {
                ComponentCapability.MEMORY_INTEGRATION.value
            },
            ComponentType.STRATEGY_TRANSITION: {
                ComponentCapability.STRATEGY_TRANSITION.value
            },
        }
        return capability_map.get(self.component_type, set())

    def get_retry_strategy(
        self, error: Exception, context: ComponentContext
    ) -> RetryStrategy:
        """
        Determine the appropriate retry strategy for a given error.

        Args:
            error: The exception that occurred
            context: The execution context

        Returns:
            The recommended retry strategy
        """
        if isinstance(error, (ConnectionError, TimeoutError)):
            return RetryStrategy.EXPONENTIAL_BACKOFF
        elif isinstance(error, ValueError):
            return RetryStrategy.NONE  # Don't retry validation errors
        else:
            return RetryStrategy.LINEAR_BACKOFF

    def get_health_check(self) -> ComponentHealthCheck:
        """Get the current health status of this component."""
        performance_score = max(0.0, 1.0 - (self.error_count * 0.1))

        return ComponentHealthCheck(
            status=self.status,
            last_success=self.last_success_time,
            error_count=self.error_count,
            performance_score=performance_score,
            dependencies_healthy=True,  # Assume healthy unless overridden
            message=f"{self.name} component operational",
        )

    def _record_success(self) -> None:
        """Record a successful operation."""
        self.last_success_time = time.time()
        self.status = ComponentStatus.READY
        # Reset error count on success
        if self.error_count > 0:
            self.error_count = max(0, self.error_count - 1)

    def _record_error(self, error: Exception) -> None:
        """Record a failed operation."""
        self.error_count += 1
        self.status = (
            ComponentStatus.ERROR if self.error_count > 3 else ComponentStatus.READY
        )
        if self.agent_logger:
            self.agent_logger.warning(
                f"{self.name} component error #{self.error_count}: {error}"
            )

    async def _execute_with_monitoring(
        self, operation: str, func, *args, **kwargs
    ) -> ComponentResult[Any]:
        """
        Execute an operation with monitoring and error handling.

        Args:
            operation: Name of the operation being performed
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            ComponentResult with success/error information
        """
        start_time = time.time()
        self.status = ComponentStatus.BUSY

        try:
            result = await func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # Convert to ms

            self._record_success()

            return ComponentResult.success_result(
                data=result,
                confidence=1.0,
                execution_time_ms=execution_time,
                operation=operation,
                component=self.name,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._record_error(e)

            context = ComponentContext(
                task="unknown", session_id="unknown"  # Will be filled by caller
            )

            return ComponentResult.error_result(
                error=str(e),
                retry_strategy=self.get_retry_strategy(e, context),
                execution_time_ms=execution_time,
                operation=operation,
                component=self.name,
            )


class ThinkingComponent(BaseComponent):
    """Component for generating thoughts and reasoning."""

    def __init__(
        self,
        engine: ReasoningEngine,
        name: str = "Thinking",
        description: str = "Generates thoughts and reasoning",
    ):
        super().__init__(engine, ComponentType.THINKING, name, description)

    async def think(self, prompt: str, **kwargs: Any) -> Optional[AgentThinkResult]:
        """
        Generate thoughts using the engine's thinking capabilities.

        Args:
            prompt (str): Prompt for thinking
            **kwargs: Additional context for the engine (see ReasoningEngine.think)

        Returns:
            Optional[AgentThinkResult]: Structured thinking result, or None if failed
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
    ) -> Plan:
        """
        Generate a plan for a task by having a conversation with the model.
        Returns a Plan Pydantic model with all steps as PlanStep and results as StepResult.
        """
        self.log_usage(f"Generating plan for task")

        context_manager = self.engine.get_context_manager()
        plan_prompt = self.engine.get_prompt("plan_generation", task=task)
        thinking_result = await plan_prompt.get_completion()
        print(f"thinking_result: {thinking_result}")
        if thinking_result:
            context_manager.add_message(
                role="assistant",
                content=f"Plan: {thinking_result.result_json}",
            )

        # self.log_usage(f"Raw thinking_result from plan generation: {thinking_result}")

        if not thinking_result or not thinking_result.result_json:
            if self.agent_logger:
                self.agent_logger.warning("Failed to generate valid plan")
            return Plan(
                plan_steps=[], metadata={"error": "Failed to generate valid plan"}
            )
        try:
            return Plan(**thinking_result.result_json)
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.warning(f"Plan parse error: {e}")
            return Plan(plan_steps=[], metadata={"error": str(e)})


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
    ) -> ToolExecutionResult:
        """
        Select and execute the most appropriate tool for a task conversationally.
        Returns a ToolExecutionResult Pydantic model.
        """
        self.log_usage(f"Selecting tool for: {step_description[:50]}...")

        tool_prompt_str = self._get_prompt(
            "tool_selection", task=task, step_description=step_description
        )
        context_manager = self.engine.get_context_manager()
        context_manager.add_message(role="user", content=step_description)
        thinking_result = await self.engine.think_chain(use_tools=True)

        if not thinking_result or not thinking_result.tool_calls:
            if self.agent_logger:
                self.agent_logger.warning(
                    "Failed to select a valid tool conversationally"
                )
            return ToolExecutionResult(
                tool_calls=[],
                results=[],
                reasoning="",
                error="Failed to select a valid tool",
            )
        try:
            tool_calls_dicts = [
                tc.dict() if hasattr(tc, "dict") else dict(tc)
                for tc in thinking_result.tool_calls
            ]
            return ToolExecutionResult(
                tool_calls=tool_calls_dicts,
                results=[],  # Actual tool execution results can be filled in if available
                reasoning=thinking_result.content,
                error=None,
            )
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.warning(f"ToolExecutionResult parse error: {e}")
            return ToolExecutionResult(
                tool_calls=[], results=[], reasoning="", error=str(e)
            )


class ReflectionComponent(BaseComponent):
    """Component for reflection and evaluation."""

    def __init__(
        self,
        engine: ReasoningEngine,
        name: str = "Reflection",
        description: str = "Reflects on progress and results",
    ):
        super().__init__(engine, ComponentType.REFLECTION, name, description)

    async def execute(
        self, context: ComponentContext
    ) -> ComponentResult[ReflectionResult]:
        """
        Execute reflection using the standardized interface.

        Args:
            context: The execution context containing task and state information

        Returns:
            ComponentResult containing the reflection result
        """
        return await self._execute_with_monitoring(
            operation="reflect_on_progress",
            func=self._internal_reflect_on_progress,
            context=context,
        )

    async def _internal_reflect_on_progress(
        self, context: ComponentContext
    ) -> ReflectionResult:
        """Internal reflection logic that follows the original implementation."""
        # Extract parameters from context
        task = context.task
        last_result = context.previous_results.get("last_result", {})

        self.log_usage("Reflecting on progress")

        reflection_prompt = self._get_prompt(
            "reflection", task=task, last_result=last_result
        )

        reflection_result = await reflection_prompt.get_completion()

        if not reflection_result or not reflection_result.result_json:
            raise RuntimeError("Failed to generate reflection")

        try:
            return ReflectionResult(**reflection_result.result_json)
        except Exception as e:
            raise ValueError(f"ReflectionResult parse error: {e}")

    async def reflect_on_progress(
        self,
        task: str,
        last_result: Dict[str, Any],
        reasoning_context: ReasoningContext,
    ) -> ReflectionResult:
        """
        Reflect on current progress conversationally.
        Returns a ReflectionResult Pydantic model.
        """
        self.log_usage("Reflecting on progress")

        reflection_prompt = self._get_prompt(
            "reflection", task=task, last_result=last_result
        )

        reflection_result = await reflection_prompt.get_completion()

        if not reflection_result or not reflection_result.result_json:
            if self.agent_logger:
                self.agent_logger.warning("Failed to generate reflection")
            return ReflectionResult(
                progress_assessment="Failed to generate reflection",
                goal_achieved=False,
                completion_score=0.0,
                next_action="continue",
                confidence=0.2,
                blockers=["Reflection generation failed"],
                error="Failed to generate reflection",
            )
        try:
            return ReflectionResult(**reflection_result.result_json)
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.warning(f"ReflectionResult parse error: {e}")
            return ReflectionResult(
                progress_assessment="Reflection parse error",
                goal_achieved=False,
                completion_score=0.0,
                next_action="continue",
                confidence=0.2,
                blockers=["Reflection parse error"],
                error=str(e),
            )

    async def reflect_on_plan_progress(
        self,
        task: str,
        current_step_index: int,
        plan_steps: List[Dict[str, Any]],
        last_result: Dict[str, Any],
    ) -> ReflectionResult:
        """
        Reflect on plan progress.
        Returns a ReflectionResult Pydantic model.
        """
        self.log_usage(
            f"Reflecting on plan progress (step {current_step_index+1}/{len(plan_steps)})"
        )

        plan_reflection_prompt = self._get_prompt(
            "plan_progress_reflection",
            task=task,
            current_step_index=current_step_index,
            plan_steps=plan_steps,
            last_result=last_result,
        )

        reflection_result = await plan_reflection_prompt.get_completion()

        if not reflection_result or not reflection_result.result_json:
            if self.agent_logger:
                self.agent_logger.warning("Failed to generate plan reflection")
            return ReflectionResult(
                progress_assessment="Failed to generate plan reflection",
                goal_achieved=False,
                completion_score=0.0,
                next_action="continue",
                confidence=0.2,
                blockers=["Plan reflection generation failed"],
                error="Failed to generate plan reflection",
            )
        try:
            return ReflectionResult(**reflection_result.result_json)
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.warning(f"Plan ReflectionResult parse error: {e}")
            return ReflectionResult(
                progress_assessment="Plan reflection parse error",
                goal_achieved=False,
                completion_score=0.0,
                next_action="continue",
                confidence=0.2,
                blockers=["Plan reflection parse error"],
                error=str(e),
            )


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
    ) -> CompletionResult:
        """
        Evaluate if a task is complete.
        Returns a CompletionResult Pydantic model.
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

        evaluator = TaskGoalEvaluator(
            model_provider=self.engine.context.model_provider,
            agent_context=self.context,
            eval_context=eval_context,
        )

        try:
            evaluation = await evaluator.get_goal_evaluation()

            if self.agent_logger:
                self.agent_logger.debug(f"Raw evaluation result: {evaluation}")

            return CompletionResult(
                is_complete=evaluation.completion,
                completion_score=evaluation.completion_score,
                reasoning=evaluation.reasoning,
                missing_requirements=evaluation.missing_requirements,
                confidence=1.0 if evaluation.completion else 0.5,
            )

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Task evaluation failed: {e}")
            return CompletionResult(
                is_complete=False,
                completion_score=0.0,
                reasoning=f"Evaluation failed: {str(e)}",
                missing_requirements=["Task evaluation encountered an error"],
                confidence=0.0,
            )


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
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Generate a final answer conversationally.
        Returns a CompletionResult Pydantic model.
        """
        self.log_usage("Attempting to generate final answer")

        final_answer_prompt = self._get_prompt(
            "final_answer",
            task=task,
            execution_summary=execution_summary,
            reflection=reflection or {},
            **kwargs,
        )

        context_manager = self.engine.get_context_manager()
        context_manager.add_message(
            role="user", content=final_answer_prompt.generate(**kwargs)
        )
        final_answer_result = await final_answer_prompt.get_completion()

        if not final_answer_result or not final_answer_result.result_json:
            if self.agent_logger:
                self.agent_logger.warning("Failed to generate valid final answer")
            return CompletionResult(
                is_complete=False,
                should_complete=False,
                completion_score=0.0,
                final_answer=f"I attempted to complete the task: {task}, but encountered some issues.",
                reasoning="Failed to generate valid final answer",
                missing_requirements=["Final answer generation failed"],
                confidence=0.2,
            )
        try:
            # Accept both direct final_answer and full CompletionResult dicts
            result_json = final_answer_result.result_json
            if "final_answer" in result_json and len(result_json) == 1:
                return CompletionResult(
                    final_answer=result_json["final_answer"],
                    is_complete=True,
                    should_complete=True,
                    completion_score=1.0,
                    reasoning="",
                    missing_requirements=[],
                    confidence=1.0,
                )
            return CompletionResult(**result_json)
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.warning(f"CompletionResult parse error: {e}")
            return CompletionResult(
                is_complete=False,
                should_complete=False,
                completion_score=0.0,
                final_answer=None,
                reasoning=str(e),
                missing_requirements=["Final answer parse error"],
                confidence=0.0,
            )


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
    ) -> ErrorRecoveryResult:
        """
        Handle an error and generate recovery steps.
        Returns an ErrorRecoveryResult Pydantic model.
        """
        self.log_usage(f"Handling error: {last_error[:50]}...")

        error_prompt = self._get_prompt(
            "error_recovery",
            task=task,
            error_context=error_context,
            error_count=error_count,
            last_error=last_error,
        )
        recovery_result = await error_prompt.get_completion()

        if (
            not recovery_result
            or not recovery_result.result_json
            or not recovery_result.result_json.get("recovery_action")
        ):
            if self.agent_logger:
                self.agent_logger.warning("Failed to generate valid error recovery")
            return ErrorRecoveryResult(
                recovery_action="retry",
                rationale="Failed to generate valid error recovery",
                error_analysis="Failed to generate valid error recovery",
            )
        try:
            return ErrorRecoveryResult(**recovery_result.result_json)
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.warning(f"ErrorRecoveryResult parse error: {e}")
            return ErrorRecoveryResult(
                recovery_action="retry",
                rationale="ErrorRecoveryResult parse error",
                error_analysis=str(e),
            )


class MemoryIntegrationComponent(BaseComponent):
    """Component for integrating with agent memory."""

    def __init__(
        self,
        engine: ReasoningEngine,
        name: str = "MemoryIntegration",
        description: str = "Integrates with agent memory",
    ) -> None:
        super().__init__(engine, ComponentType.MEMORY_INTEGRATION, name, description)

    async def get_relevant_memories(
        self, task: str, max_items: int = 5
    ) -> List[MemoryItem]:
        """
        Get memories relevant to the current task.

        Args:
            task (str): The current task
            max_items (int): Maximum number of memories to retrieve

        Returns:
            List[MemoryItem]: List of relevant memories
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
                # Convert dicts to MemoryItem if needed
                memory_items: List[MemoryItem] = []
                for mem in relevant_memories:
                    if isinstance(mem, MemoryItem):
                        memory_items.append(mem)
                    else:
                        try:
                            memory_items.append(MemoryItem(**mem))
                        except Exception:
                            continue
                return memory_items
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.warning(f"Failed to retrieve memories: {e}")

        return []


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
    ) -> StrategyTransitionResult:
        """
        Determine if the strategy should be switched.
        Returns a StrategyTransitionResult Pydantic model.
        """
        self.log_usage(f"Evaluating strategy transition from {current_strategy}")

        transition_prompt = self._get_prompt(
            "strategy_transition",
            current_strategy=current_strategy,
            available_strategies=available_strategies,
            performance_metrics=performance_metrics,
        )
        transition_result = await transition_prompt.get_completion()

        if (
            not transition_result
            or not transition_result.result_json
            or "should_switch" not in transition_result.result_json
        ):
            if self.agent_logger:
                self.agent_logger.warning(
                    "Failed to generate valid strategy transition decision"
                )
            return StrategyTransitionResult(
                should_switch=False,
                recommended_strategy=None,
                reasoning="Failed to generate valid strategy transition decision",
            )
        try:
            return StrategyTransitionResult(**transition_result.result_json)
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.warning(f"StrategyTransitionResult parse error: {e}")
            return StrategyTransitionResult(
                should_switch=False,
                recommended_strategy=None,
                reasoning=str(e),
            )


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
    by composing reusable, declarative reasoning steps.
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

    @property
    @abstractmethod
    def steps(self) -> List[BaseReasoningStep]:
        """
        Define the pipeline of reasoning steps for this strategy.
        This must be implemented by all concrete strategies.
        """
        pass

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """
        Execute one iteration of this reasoning strategy by running its
        declarative pipeline of reasoning steps.
        """
        state = self.get_state()

        for step in self.steps:
            step.strategy = self  # Inject self into the step
            result = await step.execute(state, task, reasoning_context)
            if result is not None:
                # The step terminated the iteration, so we return its result.
                return result

        # If the loop completes without any step terminating, it implies
        # the pipeline finished its course for this iteration. We return a
        # default result to continue to the next iteration.
        if self.agent_logger:
            self.agent_logger.debug(
                "Step pipeline completed. Continuing to next iteration."
            )

        return StrategyResult.create(
            payload=ContinueThinkingPayload(
                action=StrategyAction.CONTINUE_THINKING,
                reasoning="Completed a full reasoning cycle, continuing to next iteration.",
            ),
            should_continue=True,
        )

    # Convenience methods to access components
    async def think(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Generate thoughts."""
        if not self._thinking or not isinstance(self._thinking, ThinkingComponent):
            raise ValueError("Thinking component not found")
        result = await self._thinking.think(prompt)
        return result.result_json if result else {}

    async def plan(self, task: str, reasoning_context: ReasoningContext) -> Plan:
        """
        Generate a plan for a task.

        Args:
            task (str): The current task
            reasoning_context (ReasoningContext): Reasoning context for the plan

        Returns:
            Plan: The generated plan (all steps are PlanStep, results are StepResult)
        """
        if not self._planning or not isinstance(self._planning, PlanningComponent):
            raise ValueError("Planning component not found")
        return await self._planning.generate_plan(task, reasoning_context)

    async def execute_tool(
        self, task: str, step_description: str
    ) -> ToolExecutionResult:
        """Select and execute a tool."""
        if not self._tools or not isinstance(self._tools, ToolExecutionComponent):
            raise ValueError("ToolExecution component not found")
        return await self._tools.select_and_execute_tool(task, step_description)

    async def reflect(
        self,
        task: str,
        last_result: Dict[str, Any],  # TODO: Use a more specific type if possible
        reasoning_context: ReasoningContext,
    ) -> ReflectionResult:
        """
        Reflect on progress.

        Args:
            task (str): The current task
            last_result (Dict[str, Any]): Result from the last step (TODO: use a model)
            reasoning_context (ReasoningContext): Reasoning context

        Returns:
            ReflectionResult: The reflection result
        """
        if not self._reflection or not isinstance(
            self._reflection, ReflectionComponent
        ):
            raise ValueError("Reflection component not found")
        return await self._reflection.reflect_on_progress(
            task, last_result, reasoning_context
        )

    async def reflect_with_context(
        self,
        task: str,
        last_result: Dict[str, Any],
        reasoning_context: ReasoningContext,
    ) -> ComponentResult[ReflectionResult]:
        """
        Reflection using the standardized component interface.

        Args:
            task: The current task
            last_result: Result from the last step
            reasoning_context: Reasoning context

        Returns:
            ComponentResult containing the reflection result
        """
        if not self._reflection or not isinstance(
            self._reflection, ReflectionComponent
        ):
            return ComponentResult.error_result(
                error="Reflection component not found",
                retry_strategy=RetryStrategy.NONE,
            )

        # Create component context
        context = self.create_component_context(task, "reflect")
        context.previous_results["last_result"] = last_result

        # Use the standard execute method
        return await self._reflection.execute(context)

    async def evaluate_with_context(
        self, task: str, progress_summary: str = "", **kwargs: Any
    ) -> ComponentResult[CompletionResult]:
        """
        Evaluation using the standardized component interface.

        Args:
            task: The current task
            progress_summary: Summary of progress
            **kwargs: Additional context

        Returns:
            ComponentResult containing the evaluation result
        """
        if not self._evaluation or not isinstance(
            self._evaluation, TaskEvaluationComponent
        ):
            return ComponentResult.error_result(
                error="TaskEvaluation component not found",
                retry_strategy=RetryStrategy.NONE,
            )

        # Create component context
        context = self.create_component_context(task, "evaluate")
        context.previous_results.update(kwargs)
        context.previous_results["progress_summary"] = progress_summary

        # For evaluation, we need to call the original method since it's not yet using ComponentResult
        # This is a bridge until we fully migrate all components
        try:
            result = await self._evaluation.evaluate_task_completion(
                task, progress_summary=progress_summary, **kwargs
            )
            return ComponentResult.success_result(
                data=result,
                confidence=result.confidence if hasattr(result, "confidence") else 0.8,
            )
        except Exception as e:
            return ComponentResult.error_result(
                error=str(e), retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF
            )

    def get_component_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all components.

        Returns:
            Dictionary mapping component names to health status
        """
        health_status = {}

        component_map = {
            "thinking": self._thinking,
            "planning": self._planning,
            "tools": self._tools,
            "reflection": self._reflection,
            "evaluation": self._evaluation,
            "completion": self._completion,
            "error_handling": self._error_handling,
            "memory": self._memory,
            "transition": self._transition,
        }

        for name, component in component_map.items():
            if component and hasattr(component, "get_health_check"):
                health_status[name] = component.get_health_check().model_dump()
            else:
                health_status[name] = {
                    "status": "unknown",
                    "message": f"{name} component not available or incompatible",
                }

        return health_status

    def get_component_capabilities(self) -> Set[str]:
        """
        Get combined capabilities of all available components.

        Returns:
            Set of all capabilities provided by components
        """
        all_capabilities = set()

        component_map = {
            "thinking": self._thinking,
            "planning": self._planning,
            "tools": self._tools,
            "reflection": self._reflection,
            "evaluation": self._evaluation,
            "completion": self._completion,
            "error_handling": self._error_handling,
            "memory": self._memory,
            "transition": self._transition,
        }

        for component in component_map.values():
            if component and hasattr(component, "get_capabilities"):
                all_capabilities.update(component.get_capabilities())

        return all_capabilities

    async def evaluate(
        self, task: str, progress_summary: str = "", **kwargs: Any
    ) -> CompletionResult:
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
    ) -> ErrorRecoveryResult:
        """Handle an error."""
        if not self._error_handling or not isinstance(
            self._error_handling, ErrorHandlingComponent
        ):
            raise ValueError("ErrorHandling component not found")
        return await self._error_handling.handle_error(
            task, error_context, error_count, last_error
        )

    async def get_memories(self, task: str, max_items: int = 5) -> List[MemoryItem]:
        """
        Get relevant memories.

        Args:
            task (str): The current task
            max_items (int): Maximum number of memories to retrieve

        Returns:
            List[MemoryItem]: List of relevant memories
        """
        if not self._memory or not isinstance(self._memory, MemoryIntegrationComponent):
            raise ValueError("MemoryIntegration component not found")
        return await self._memory.get_relevant_memories(task, max_items)

    async def complete_task(
        self, task: str, execution_summary: str = "", **kwargs: Any
    ) -> CompletionResult:
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
        if not completion_result.final_answer:
            if self.agent_logger:
                self.agent_logger.warning("No final answer in completion result")
            return CompletionResult(
                is_complete=False,
                should_complete=False,
                completion_score=0.0,
                final_answer=None,
                reasoning="Failed to generate final answer",
                missing_requirements=["No final answer in completion result"],
                confidence=0.0,
            )

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
