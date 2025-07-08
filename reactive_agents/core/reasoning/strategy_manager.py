from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING
from reactive_agents.core.types.reasoning_types import (
    ReasoningContext,
    ReasoningStrategies,
)
from reactive_agents.core.types.task_types import TaskClassification

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext
    from reactive_agents.core.reasoning.infrastructure import Infrastructure


class StrategyManager:
    """
    Clean, simplified strategy manager.

    Manages strategy selection and execution without plugin overhead.
    Focuses on core functionality and direct strategy access.
    """

    def __init__(self, infrastructure: "Infrastructure", context: "AgentContext"):
        self.infrastructure = infrastructure
        self.context = context
        self.agent_logger = context.agent_logger

        # Strategy registry (direct imports, no plugin system)
        self.strategies: Dict[str, Any] = {}
        self.current_strategy: Optional[Any] = None
        self.current_strategy_name: Optional[str] = None

        # Initialize reasoning context
        self.reasoning_context = ReasoningContext(
            current_strategy=ReasoningStrategies.REFLECT_DECIDE_ACT
        )

        # Load available strategies
        self._load_strategies()

        if self.agent_logger:
            self.agent_logger.info(
                f"✅ Strategy manager initialized with {len(self.strategies)} strategies"
            )

    def _load_strategies(self):
        """Load and register available strategies."""
        try:
            # Import strategies directly
            from reactive_agents.core.reasoning.strategies.reactive import (
                ReactiveStrategy,
            )
            from reactive_agents.core.reasoning.strategies.reflect_decide_act import (
                ReflectDecideActStrategy,
            )
            from reactive_agents.core.reasoning.strategies.plan_execute_reflect import (
                PlanExecuteReflectStrategy,
            )

            # Register strategies (simple dict, no plugin overhead)
            self.strategies = {
                "reactive": ReactiveStrategy,
                "reflect_decide_act": ReflectDecideActStrategy,
                "plan_execute_reflect": PlanExecuteReflectStrategy,
            }

        except ImportError as e:
            if self.agent_logger:
                self.agent_logger.warning(f"Could not load some strategies: {e}")
            # Fallback to basic set
            self.strategies = {}

    def get_available_strategies(self) -> list[str]:
        """Get list of available strategy names."""
        return list(self.strategies.keys())

    def set_strategy(self, strategy_name: str) -> bool:
        """Set the current active strategy."""
        if strategy_name not in self.strategies:
            if self.agent_logger:
                self.agent_logger.error(f"Strategy '{strategy_name}' not found")
            return False

        # Create strategy instance
        try:
            strategy_class = self.strategies[strategy_name]
            self.current_strategy = strategy_class(self.infrastructure)
            self.current_strategy_name = strategy_name

            if self.agent_logger:
                self.agent_logger.debug(f"Set current strategy to: {strategy_name}")
            return True

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(
                    f"Failed to create strategy '{strategy_name}': {e}"
                )
            return False

    def get_current_strategy_name(self) -> str:
        """Get the name of the current strategy."""
        return self.current_strategy_name or "none"

    def select_optimal_strategy(self, task_classification: TaskClassification) -> str:
        """Select optimal strategy based on task classification."""
        task_type = task_classification.task_type.value
        complexity = task_classification.complexity_score

        # Simple strategy selection logic
        if task_type == "simple_lookup" or complexity < 0.3:
            selected = "reactive"
        elif task_type in ["multi_step", "planning"] or complexity > 0.7:
            selected = "plan_execute_reflect"
        else:
            selected = "reflect_decide_act"  # Default for most tasks

        # Set the selected strategy
        if self.set_strategy(selected):
            if self.agent_logger:
                self.agent_logger.info(
                    f"🎯 Selected strategy: {selected} "
                    f"(type: {task_type}, complexity: {complexity:.2f})"
                )
            return selected
        else:
            # Fallback to first available
            fallback = (
                self.get_available_strategies()[0]
                if self.get_available_strategies()
                else "reactive"
            )
            self.set_strategy(fallback)
            return fallback

    async def execute_iteration(self, task: str, reasoning_context: ReasoningContext):
        """Execute one iteration using the current strategy."""
        if not self.current_strategy:
            # Set default strategy if none is set
            self.set_strategy("reflect_decide_act")

        if not self.current_strategy:
            # Return error if still no strategy
            from reactive_agents.core.reasoning.strategies.base import StrategyResult

            return StrategyResult(
                action_taken="error",
                result={"error": "No strategy available"},
                should_continue=False,
                strategy_used="none",
            )

        try:
            # Execute strategy
            result = await self.current_strategy.execute_iteration(
                task, reasoning_context
            )

            # Update reasoning context
            reasoning_context.iteration_count += 1
            if hasattr(result, "strategy_used"):
                reasoning_context.last_action_result = result.result

            return result

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Strategy execution failed: {e}")

            from reactive_agents.core.reasoning.strategies.base import StrategyResult

            return StrategyResult(
                action_taken="error",
                result={"error": str(e)},
                should_continue=False,
                strategy_used=self.current_strategy_name or "unknown",
            )

    def reset(self):
        """Reset the strategy manager for a new task."""
        self.reasoning_context = ReasoningContext(
            current_strategy=ReasoningStrategies.REFLECT_DECIDE_ACT
        )
        # Note: Keep current strategy instance for reuse

        if self.agent_logger:
            self.agent_logger.debug("Strategy manager reset for new task")

    def get_reasoning_context(self) -> ReasoningContext:
        """Get the current reasoning context."""
        return self.reasoning_context

    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """Get basic information about a strategy."""
        if strategy_name not in self.strategies:
            return {"error": "Strategy not found"}

        return {
            "name": strategy_name,
            "class": self.strategies[strategy_name].__name__,
            "available": True,
        }
