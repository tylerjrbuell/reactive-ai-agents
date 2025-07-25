from __future__ import annotations
from typing import Dict, Any, List, Optional, Type, TYPE_CHECKING
import logging
import importlib
import inspect
import os
import json
import pkgutil
from pathlib import Path

from reactive_agents.core.reasoning.strategies.base import (
    BaseReasoningStrategy,
    StrategyResult,
)
from reactive_agents.core.reasoning.strategy_components import ComponentBasedStrategy
from reactive_agents.core.reasoning.engine import ReasoningEngine
from reactive_agents.core.reasoning.strategies.reactive import ReactiveStrategy
from reactive_agents.core.reasoning.strategies.plan_execute_reflect import (
    PlanExecuteReflectStrategy,
)
from reactive_agents.core.reasoning.strategies.reflect_decide_act import (
    ReflectDecideActStrategy,
)
from reactive_agents.core.types.reasoning_types import (
    ReasoningContext,
    ReasoningStrategies,
)

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class StrategyManager:
    """
    Manages reasoning strategies that can be used by agents.

    Provides functionality to:
    1. Register strategies
    2. Select an appropriate strategy based on task
    3. Switch between strategies during execution
    """

    def __init__(self, agent_context: "AgentContext"):
        """
        Initialize the strategy manager.

        Args:
            agent_context: The agent context
        """
        self.agent_context = agent_context
        self.logger = agent_context.agent_logger or logging.getLogger(__name__)
        self.strategies: Dict[str, Type[BaseReasoningStrategy]] = {}
        self.active_strategy: Optional[BaseReasoningStrategy] = None
        self.engine = agent_context.reasoning_engine

        # Auto-register built-in strategies
        self._register_built_in_strategies()

    def _register_built_in_strategies(self):
        """Auto-register all built-in strategies."""
        strategy_path = os.path.join(os.path.dirname(__file__), "strategies")

        component_strategies_count = 0
        regular_strategies_count = 0

        try:
            # Import the strategies package
            strategies_pkg = importlib.import_module(
                "reactive_agents.core.reasoning.strategies"
            )

            # Find all modules in the strategies directory
            for _, name, is_pkg in pkgutil.iter_modules([strategy_path]):
                # Skip __init__ and base
                if name == "__init__" or name == "base":
                    continue

                try:
                    # Import the module
                    module = importlib.import_module(
                        f"reactive_agents.core.reasoning.strategies.{name}"
                    )

                    # Find strategy classes in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)

                        # Check if it's a strategy class (excluding base classes)
                        if (
                            inspect.isclass(attr)
                            and issubclass(attr, BaseReasoningStrategy)
                            and attr != BaseReasoningStrategy
                            and attr != ComponentBasedStrategy
                        ):
                            # Register the strategy
                            self.register_strategy(attr)

                            # Track component vs regular strategies
                            if issubclass(attr, ComponentBasedStrategy):
                                component_strategies_count += 1
                                self.logger.debug(
                                    f"Registered component-based strategy: {attr.__name__}"
                                )
                            else:
                                regular_strategies_count += 1

                except Exception as e:
                    self.logger.warning(f"Failed to load strategy module {name}: {e}")

            # Log summary of loaded strategies
            self.logger.info(
                f"Registered {component_strategies_count + regular_strategies_count} strategies "
                f"({component_strategies_count} component-based, {regular_strategies_count} regular)"
            )

        except Exception as e:
            self.logger.warning(f"Failed to load built-in strategies: {e}")

    def register_strategy(self, strategy_class: Type[BaseReasoningStrategy]) -> None:
        """
        Register a strategy with the manager.

        Args:
            strategy_class: The strategy class to register
        """
        # Create a temporary instance to get name and capabilities
        temp_instance = strategy_class(self.engine)
        strategy_name = temp_instance.name

        # Check for duplicate names
        if strategy_name in self.strategies:
            self.logger.warning(
                f"Strategy {strategy_name} already registered. Overwriting."
            )

        # Register the strategy class
        self.strategies[strategy_name] = strategy_class
        self.logger.debug(f"Registered strategy: {strategy_name}")

    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available strategies.

        Returns:
            Dict mapping strategy names to their capabilities
        """
        result = {}

        for name, strategy_class in self.strategies.items():
            # Create a temporary instance to get capabilities
            temp_instance = strategy_class(self.engine)

            # Build the info dict
            info = {
                "name": name,
                "description": temp_instance.description,
                "capabilities": [cap.value for cap in temp_instance.capabilities],
                "is_component_based": issubclass(
                    strategy_class, ComponentBasedStrategy
                ),
            }

            result[name] = info

        return result

    def get_strategy_by_name(self, name: str) -> Optional[BaseReasoningStrategy]:
        """
        Get a strategy instance by name.

        Args:
            name: Strategy name

        Returns:
            Strategy instance or None if not found
        """
        if name not in self.strategies:
            self.logger.warning(f"Strategy {name} not found")
            return None

        strategy_class = self.strategies[name]
        return strategy_class(self.engine)

    def get_strategy_info(self, strategy_name: str) -> dict:
        """
        Get information about a specific strategy.
        """
        if strategy_name not in self.strategies:
            return {"error": f"Strategy '{strategy_name}' not found"}
        strategy_class = self.strategies[strategy_name]
        temp_instance = strategy_class(self.engine)
        return {
            "name": temp_instance.name,
            "description": getattr(temp_instance, "description", ""),
            "capabilities": [
                cap.value for cap in getattr(temp_instance, "capabilities", [])
            ],
            "is_component_based": issubclass(strategy_class, ComponentBasedStrategy),
        }

    def select_strategy_for_task(
        self, task: str, reasoning_context: Optional[ReasoningContext] = None
    ) -> BaseReasoningStrategy:
        """
        Select an appropriate strategy for a task.

        Args:
            task: Task description
            reasoning_context: Optional reasoning context

        Returns:
            Selected strategy instance
        """
        # Default to reactive strategy if available
        default_strategy_name = "reactive"

        # If we have a configured default strategy, try to access it safely
        try:
            if hasattr(self.agent_context, "config"):
                config = getattr(self.agent_context, "config", None)
                if config and hasattr(config, "reasoning"):
                    reasoning_config = getattr(config, "reasoning", None)
                    if reasoning_config and hasattr(
                        reasoning_config, "default_strategy"
                    ):
                        configured_default = reasoning_config.default_strategy
                        if configured_default and configured_default in self.strategies:
                            default_strategy_name = configured_default
        except AttributeError:
            self.logger.debug("Could not access config for default strategy")

        # Check if we have the default strategy
        if default_strategy_name not in self.strategies:
            # Fall back to first available strategy
            if not self.strategies:
                raise ValueError("No strategies registered")
            default_strategy_name = next(iter(self.strategies.keys()))

        # Get default strategy class
        strategy_class = self.strategies[default_strategy_name]

        # Create and return the strategy instance
        self.active_strategy = strategy_class(self.engine)
        return self.active_strategy

    async def switch_strategy(
        self, new_strategy_name: str, task: str, reasoning_context: ReasoningContext
    ) -> Optional[BaseReasoningStrategy]:
        """
        Switch to a different strategy.

        Args:
            new_strategy_name: Name of the strategy to switch to
            task: Current task
            reasoning_context: Current reasoning context

        Returns:
            New strategy instance or None if switch failed
        """
        if new_strategy_name not in self.strategies:
            self.logger.warning(
                f"Cannot switch to unknown strategy: {new_strategy_name}"
            )
            return None

        # Clean up current strategy if present
        if self.active_strategy:
            await self.active_strategy.finalize(task, reasoning_context)

        self.set_strategy(new_strategy_name)

        # Initialize the new strategy
        await self.initialize_active_strategy(task, reasoning_context)

        return self.active_strategy

    # Add these methods to the StrategyManager class

    def reset(self) -> None:
        """Reset the strategy manager state."""
        self.active_strategy = None
        self.logger.debug("Strategy manager reset")

    def set_strategy(self, strategy_name: str) -> None:
        """
        Set the active strategy by name.

        Args:
            strategy_name: Name of the strategy to set
        """
        if strategy_name not in self.strategies:
            self.logger.warning(f"Strategy '{strategy_name}' not found, using default")
            strategy_name = next(iter(self.strategies.keys()))

        strategy_class = self.strategies[strategy_name]
        self.active_strategy = strategy_class(self.engine)
        self.logger.debug(f"Set active strategy: {strategy_name}")

    async def initialize_active_strategy(
        self, task: str, reasoning_context: ReasoningContext
    ) -> None:
        """
        Initialize the active strategy for a new task.

        This should be called when starting execution with a strategy.

        Args:
            task: The task to initialize for
            reasoning_context: The reasoning context
        """
        if not self.active_strategy:
            raise ValueError("No active strategy set")

        if hasattr(self.active_strategy, "initialize"):
            await self.active_strategy.initialize(task, reasoning_context)
            self.logger.debug(f"Initialized strategy: {self.active_strategy.name}")
        else:
            self.logger.debug(
                f"Strategy {self.active_strategy.name} has no initialize method"
            )

    def get_current_strategy_name(self) -> str:
        """
        Get the name of the currently active strategy.

        Returns:
            Name of the active strategy or 'none' if no strategy is active
        """
        if not self.active_strategy:
            return "none"
        return self.active_strategy.name

    def get_current_strategy_enum(self) -> ReasoningStrategies:
        """
        Get the enum value of the currently active strategy.
        """
        strategy_name = self.get_current_strategy_name()
        if strategy_name == "none":
            # Return a default strategy when none is active
            return ReasoningStrategies.ADAPTIVE
        return ReasoningStrategies(strategy_name)

    def select_optimal_strategy(self, classification: Any) -> str:
        """
        Select optimal strategy based on task classification.

        Args:
            classification: Task classification result

        Returns:
            Name of the selected strategy
        """
        # Simple mapping of task types to strategies
        task_type = getattr(classification, "task_type", None)
        if task_type:
            strategy_map = {
                "information_retrieval": "reactive",
                "data_analysis": "plan_execute_reflect",
                "creative": "reflect_decide_act",
                "planning": "plan_execute_reflect",
                "reasoning": "reflect_decide_act",
                "coding": "plan_execute_reflect",
            }

            strategy_name = strategy_map.get(task_type.value, "reactive")
            self.set_strategy(strategy_name)
            return strategy_name

        # Default to reactive
        self.set_strategy("reactive")
        return "reactive"

    async def select_and_initialize_strategy(
        self, classification: Any, task: str, reasoning_context: ReasoningContext
    ) -> str:
        """
        Select optimal strategy and initialize it for the task.

        Args:
            classification: Task classification result
            task: The task to initialize for
            reasoning_context: The reasoning context

        Returns:
            Name of the selected strategy
        """
        strategy_name = self.select_optimal_strategy(classification)
        await self.initialize_active_strategy(task, reasoning_context)
        return strategy_name

    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """
        Execute one iteration using the active strategy.

        Args:
            task: Current task
            reasoning_context: Reasoning context

        Returns:
            Result of the strategy execution
        """
        if not self.active_strategy:
            raise ValueError("No active strategy set")

        return await self.active_strategy.execute_iteration(task, reasoning_context)
