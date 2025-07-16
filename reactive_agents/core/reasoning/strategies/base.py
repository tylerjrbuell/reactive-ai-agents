from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from enum import Enum
import json
from reactive_agents.core.types.agent_types import (
    AgentThinkChainResult,
    AgentThinkResult,
)
from reactive_agents.core.types.event_types import AgentStateEvent
from reactive_agents.core.types.prompt_types import (
    FinalAnswerOutput,
    ReflectionOutput,
    ToolSelectionOutput,
)
from reactive_agents.core.types.reasoning_types import (
    ActionPayload,
    EvaluationPayload,
    ReasoningContext,
    StrategyAction,
    TaskGoalEvaluationResult,
)
from reactive_agents.core.reasoning.engine import ReasoningEngine

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext
    from reactive_agents.core.types.session_types import BaseStrategyState


class StrategyCapabilities(Enum):
    """Capabilities that a strategy can declare."""

    TOOL_EXECUTION = "tool_execution"
    PLANNING = "planning"
    REFLECTION = "reflection"
    MEMORY_USAGE = "memory_usage"
    ADAPTATION = "adaptation"
    COLLABORATION = "collaboration"


from pydantic import BaseModel, Field


class StrategyResult(BaseModel):
    """
    A strongly-typed result from a strategy iteration.
    It contains a specific action and a corresponding payload with the required data.
    """

    action: StrategyAction
    payload: ActionPayload = Field(..., discriminator="action")
    should_continue: bool = True

    # This allows creating the model with a simplified syntax
    @classmethod
    def create(
        cls, payload: ActionPayload, should_continue: bool = True
    ) -> "StrategyResult":
        return cls(
            action=payload.action, payload=payload, should_continue=should_continue
        )


class BaseReasoningStrategy(ABC):
    """
    Base class for all reasoning strategies.

    This interface provides a clean, standardized way to implement
    reasoning strategies that can be plugged into the framework.
    """

    def __init__(self, engine: "ReasoningEngine"):
        """
        Initialize the strategy with shared engine.

        Args:
            engine: The reasoning engine providing shared services
        """
        self.engine = engine
        self.context = engine.context
        self.agent_logger = engine.context.agent_logger

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this strategy."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[StrategyCapabilities]:
        """Return the capabilities this strategy supports."""
        pass

    @property
    def description(self) -> str:
        """Return a description of this strategy (optional override)."""
        return f"Base reasoning strategy: {self.name}"

    @abstractmethod
    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """
        Execute one iteration of this reasoning strategy.

        Args:
            task: The current task description
            reasoning_context: Context about current reasoning state

        Returns:
            StrategyResult with execution results
        """
        pass

    # === Standardized Strategy Contract ===

    def get_state(self) -> "BaseStrategyState":
        """
        Get the strategy state from session, initializing if needed.

        This method dynamically determines the correct state type based on the strategy name
        and the STRATEGY_REGISTRY. It provides type safety and eliminates the need to
        define _get_state() in each subclass.

        Returns:
            The appropriate strategy state instance

        Raises:
            TypeError: If the state type doesn't match what's expected
            ValueError: If the strategy is not registered
        """
        from reactive_agents.core.types.session_types import (
            STRATEGY_REGISTRY,
        )

        strategy_name = self.name

        # Initialize state if not present
        if strategy_name not in self.context.session.strategy_state:
            self.context.session.initialize_strategy_state(strategy_name)

        # Get state from session
        state = self.context.session.get_strategy_state(strategy_name)
        if state is None:
            raise ValueError(f"No state found for strategy: {strategy_name}")

        # Get expected state class from registry
        registry_entry = STRATEGY_REGISTRY.get(strategy_name)
        if registry_entry and "state_cls" in registry_entry:
            expected_state_cls = registry_entry["state_cls"]

            # Type check for safety
            if not isinstance(state, expected_state_cls):
                raise TypeError(
                    f"Expected {expected_state_cls.__name__} for strategy '{strategy_name}', "
                    f"got {type(state).__name__}"
                )

        return state

    async def reflect_on_progress(
        self,
        task: str,
        execution_results: Dict[str, Any],
        reasoning_context: ReasoningContext,
    ) -> Optional[ReflectionOutput]:
        """
        Standard method for reflecting on progress in any strategy.

        Args:
            task: The main task
            execution_results: Results from tool execution
            reasoning_context: Current reasoning context

        Returns:
            Reflection results with consistent format
        """
        # Use centralized reflection prompt with dynamic context
        reflection_prompt = self.engine.get_prompt(
            "reflection",
            task=task,
            last_result=execution_results,
            iteration_count=reasoning_context.iteration_count,
            error_count=reasoning_context.error_count,
            tool_usage_history=reasoning_context.tool_usage_history,
        )

        # Add to context and get response
        result = await reflection_prompt.get_completion()

        if result and result.result_json:
            self.context.emit_event(
                AgentStateEvent.REFLECTION_GENERATED,
                {
                    "reflection": result.result_json,
                },
            )

            return ReflectionOutput.model_validate(result.result_json)

        return None

    async def evaluate_task_completion(
        self, task: str, execution_summary: str = "", **kwargs
    ) -> Optional[EvaluationPayload]:
        """
        Standard method for evaluating task completion.

        Args:
            task: The main task
            execution_summary: Summary of what was accomplished
            **kwargs: Additional context

        Returns:
            Evaluation results with consistent format
        """
        # Use the engine's completion checking
        prompt = self.engine.get_prompt("task_goal_evaluation")
        result = await prompt.get_completion(
            task=task, execution_summary=execution_summary, **kwargs
        )

        if result and result.result_json:
            eval_output = TaskGoalEvaluationResult.model_validate(result.result_json)
            return EvaluationPayload(
                action=StrategyAction.EVALUATE_COMPLETION,
                is_complete=eval_output.completion,
                reasoning=eval_output.reasoning,
                confidence=eval_output.completion_score,
            )
        return None

    async def generate_final_answer(
        self, task: str, execution_summary: str = "", **kwargs
    ) -> Optional[FinalAnswerOutput]:
        """
        Standard method for generating final answers.

        Args:
            task: The main task
            execution_summary: Summary of what was accomplished
            **kwargs: Additional context

        Returns:
            Final answer with consistent format
        """
        # Use the engine's final answer generation
        prompt = self.engine.get_prompt("final_answer")
        result = await prompt.get_completion(
            task=task, execution_summary=execution_summary, **kwargs
        )

        if result and result.result_json:
            return FinalAnswerOutput.model_validate(result.result_json)

        return None

    async def initialize(self, task: str, reasoning_context: ReasoningContext) -> None:
        """
        Initialize the strategy for a new task (optional override).

        Args:
            task: The task to be executed
            reasoning_context: Initial reasoning context
        """
        pass

    async def finalize(self, task: str, reasoning_context: ReasoningContext) -> None:
        """
        Finalize the strategy after task completion (optional override).

        Args:
            task: The completed task
            reasoning_context: Final reasoning context
        """
        pass

    def should_switch_strategy(
        self, reasoning_context: ReasoningContext
    ) -> Optional[str]:
        """
        Determine if strategy should switch to another (optional override).

        Args:
            reasoning_context: Current reasoning context

        Returns:
            Strategy name to switch to, or None to continue
        """
        return None

    # Utility methods for strategies
    async def _think(
        self, prompt: str, format: Optional[str] = None
    ) -> Optional[AgentThinkResult]:
        """Execute a thinking step using the engine."""
        return await self.engine.think(prompt, format=format)

    async def _think_chain(
        self, use_tools: bool = False
    ) -> Optional[AgentThinkChainResult]:
        """Execute a thinking step using the engine."""
        return await self.engine.think_chain(use_tools)

    async def _execute_tools(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls using the engine's canonical path."""
        return await self.engine.execute_tools(tool_calls)

    async def _manual_tool_prompting(
        self, task: str, step_description: str
    ) -> Optional[ToolSelectionOutput]:
        """
        Manual tool prompting for models that don't support native tool calling.
        This method uses the centralized prompt system for consistency.

        Args:
            task: The main task
            step_description: Description of the current step

        Returns:
            Dictionary with tool execution results
        """
        # Use centralized tool selection prompt with dynamic context
        prompt = self.engine.get_prompt(
            "tool_selection",
            step_description=step_description,
            task=task,
        )
        # Get model response
        result = await prompt.get_completion()
        if result and result.result_json:
            return ToolSelectionOutput.model_validate(result.result_json)

        return None

    def _format_error_result(
        self, error: Exception, action: str = "unknown"
    ) -> StrategyResult:
        """Format an error as a StrategyResult."""
        from reactive_agents.core.types.reasoning_types import ErrorPayload

        payload = ErrorPayload(
            action=StrategyAction.ERROR,
            error_message=str(error),
            details={"failed_action": action},
        )
        return StrategyResult.create(payload, should_continue=False)


class StrategyPlugin:
    """
    Plugin wrapper for strategies to provide metadata and registration info.
    """

    def __init__(
        self,
        strategy_class: type[BaseReasoningStrategy],
        name: str,
        description: str,
        version: str = "1.0.0",
        author: str = "unknown",
        capabilities: Optional[List[StrategyCapabilities]] = None,
        config_schema: Optional[Dict[str, Any]] = None,
    ):
        self.strategy_class = strategy_class
        self.name = name
        self.description = description
        self.version = version
        self.author = author
        self.capabilities = capabilities or []
        self.config_schema = config_schema or {}

    def create_instance(self, engine: "ReasoningEngine") -> BaseReasoningStrategy:
        """Create an instance of the strategy."""
        return self.strategy_class(engine)

    def to_dict(self) -> Dict[str, Any]:
        """Convert plugin info to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "capabilities": [cap.value for cap in self.capabilities],
            "config_schema": self.config_schema,
        }
