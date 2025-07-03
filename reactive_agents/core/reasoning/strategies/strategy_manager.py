from __future__ import annotations
import time
from typing import Dict, Any, Optional, TYPE_CHECKING, List
from reactive_agents.core.types.reasoning_types import (
    ReasoningStrategies,
    ReasoningContext,
)
from reactive_agents.core.types.task_types import TaskClassification

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext
    from reactive_agents.core.reasoning.strategies.base import BaseReasoningStrategy


class StrategyManager:
    """
    Manages reasoning strategy selection, switching, and plugin registration.
    Provides the core intelligence for adaptive reasoning.
    """

    def __init__(self, context: "AgentContext"):
        self.context = context
        self.agent_logger = context.agent_logger
        self.strategies: Dict[ReasoningStrategies, "BaseReasoningStrategy"] = {}
        self.current_strategy: Optional[ReasoningStrategies] = None
        self.reasoning_context = ReasoningContext(
            current_strategy=ReasoningStrategies.REFLECT_DECIDE_ACT  # Default
        )
        self._register_default_strategies()
        self._register_plugin_strategies()

    def _register_default_strategies(self):
        """Dynamically discover and register all available reasoning strategies."""
        # Strategy mapping - maps enum values to strategy classes
        strategy_mapping = {
            "reactive": ".reactive.ReactiveStrategy",
            "reflect_decide_act": ".reflect_decide_act.ReflectDecideActStrategy",
            "plan_execute_reflect": ".plan_execute_reflect.PlanExecuteReflectStrategy",
            "adaptive": ".adaptive.AdaptiveStrategy",
            "memory_enhanced": ".memory_enhanced.MemoryEnhancedStrategy",
        }

        # Dynamically import and register strategies
        for strategy_name, import_path in strategy_mapping.items():
            try:
                # Dynamic import
                module_path, class_name = import_path.rsplit(".", 1)
                module = __import__(
                    f"reactive_agents.core.reasoning.strategies{module_path}",
                    fromlist=[class_name],
                )
                strategy_class = getattr(module, class_name)

                # Create and register strategy instance
                strategy_instance = strategy_class(self.context)
                self.register_strategy(strategy_instance)

                if self.agent_logger:
                    self.agent_logger.debug(
                        f"✅ Successfully registered strategy: {strategy_name}"
                    )

            except (ImportError, AttributeError) as e:
                if self.agent_logger:
                    self.agent_logger.debug(
                        f"⚠️ Strategy {strategy_name} not available: {e}"
                    )
            except Exception as e:
                if self.agent_logger:
                    self.agent_logger.warning(
                        f"❌ Failed to register strategy {strategy_name}: {e}"
                    )

        # Set default strategy
        self.current_strategy = ReasoningStrategies.REFLECT_DECIDE_ACT

        if self.agent_logger:
            available_strategies = [s.value for s in self.strategies.keys()]
            self.agent_logger.info(f"🎯 Available strategies: {available_strategies}")

    def _register_plugin_strategies(self):
        """Register strategies from loaded plugins."""
        try:
            from reactive_agents.plugins.plugin_manager import (
                get_plugin_manager,
                PluginType,
                ReasoningStrategyPlugin,
            )

            plugin_manager = get_plugin_manager()
            strategy_plugins = plugin_manager.get_plugins_by_type(PluginType.STRATEGY)

            for plugin_name, plugin in strategy_plugins.items():
                try:
                    # Check if it's a ReasoningStrategyPlugin
                    if isinstance(plugin, ReasoningStrategyPlugin):
                        # Get strategies from the plugin
                        strategies = plugin.get_strategies()
                        for strategy_name, strategy_class in strategies.items():
                            # Create strategy instance
                            strategy_instance = strategy_class(self.context)
                            self.register_strategy(strategy_instance)

                            if self.agent_logger:
                                self.agent_logger.info(
                                    f"🔌 Registered plugin strategy: {strategy_name} from {plugin_name}"
                                )

                except Exception as e:
                    if self.agent_logger:
                        self.agent_logger.warning(
                            f"❌ Failed to register plugin strategy {plugin_name}: {e}"
                        )

        except ImportError:
            # Plugin system not available
            if self.agent_logger:
                self.agent_logger.debug(
                    "Plugin system not available for strategy registration"
                )
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.warning(f"Error loading strategy plugins: {e}")

    def register_strategy(self, strategy: "BaseReasoningStrategy"):
        """Register a new reasoning strategy."""
        strategy_name = strategy.get_strategy_name()
        self.strategies[strategy_name] = strategy

        if self.agent_logger:
            self.agent_logger.debug(
                f"Registered reasoning strategy: {strategy_name.value}"
            )

    def register_plugin_strategy(self, strategy_name: str, strategy_class: type):
        """Register a strategy from a plugin."""
        try:
            # Create strategy instance
            strategy_instance = strategy_class(self.context)
            self.register_strategy(strategy_instance)

            if self.agent_logger:
                self.agent_logger.info(
                    f"🔌 Registered plugin strategy: {strategy_name}"
                )
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(
                    f"❌ Failed to register plugin strategy {strategy_name}: {e}"
                )

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return [strategy.value for strategy in self.strategies.keys()]

    def get_plugin_strategies(self) -> Dict[str, str]:
        """Get mapping of plugin strategy names to their descriptions."""
        plugin_strategies = {}
        try:
            from reactive_agents.plugins.plugin_manager import (
                get_plugin_manager,
                PluginType,
            )

            plugin_manager = get_plugin_manager()
            strategy_plugins = plugin_manager.get_plugins_by_type(PluginType.STRATEGY)

            for plugin_name, plugin in strategy_plugins.items():
                plugin_strategies[plugin_name] = plugin.description

        except ImportError:
            pass
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.debug(f"Error getting plugin strategies: {e}")

        return plugin_strategies

    def get_current_strategy(self) -> Optional["BaseReasoningStrategy"]:
        """Get the currently active strategy instance."""
        if self.current_strategy and self.current_strategy in self.strategies:
            return self.strategies[self.current_strategy]
        return None

    async def execute_iteration(self, task: str) -> Dict[str, Any]:
        """
        Execute one iteration using the current strategy.
        Handles strategy switching if recommended.
        """
        current_strategy_impl = self.get_current_strategy()
        if not current_strategy_impl:
            return {"error": "No active reasoning strategy", "should_continue": False}

        if self.agent_logger:
            strategy_name = (
                self.current_strategy.value if self.current_strategy else "unknown"
            )
            self.agent_logger.info(f"🧠 Using strategy: {strategy_name}")

        # Execute the iteration
        try:
            result = await current_strategy_impl.execute_iteration(
                task, self.reasoning_context
            )

            # Update reasoning context
            self._update_reasoning_context(result)

            # Check for strategy switching recommendation
            await self._check_strategy_switch(current_strategy_impl)

            return result

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"Strategy execution failed: {e}")

            # Increment error count and try to switch to a more robust strategy
            self.reasoning_context.error_count += 1
            await self._handle_strategy_failure()

            return {
                "error": str(e),
                "should_continue": True,  # Try to continue with different strategy
                "strategy_used": (
                    self.current_strategy.value if self.current_strategy else "unknown"
                ),
            }

    async def _check_strategy_switch(self, current_strategy: "BaseReasoningStrategy"):
        """Check if the current strategy recommends switching."""
        switch_recommendation = current_strategy.should_switch_strategy(
            self.reasoning_context
        )

        if (
            switch_recommendation
            and self.current_strategy
            and switch_recommendation.to_strategy
        ):
            if self.agent_logger:
                self.agent_logger.info(
                    f"🔄 Strategy switch recommended: {self.current_strategy.value} → "
                    f"{switch_recommendation.to_strategy.value}"
                )

            await self._switch_strategy(
                switch_recommendation.to_strategy,
                switch_recommendation.reason,
            )

    async def _switch_strategy(self, new_strategy: ReasoningStrategies, reason: str):
        """Switch to a new reasoning strategy."""
        if new_strategy not in self.strategies:
            if self.agent_logger:
                self.agent_logger.warning(
                    f"Strategy {new_strategy.value} not available"
                )
            return

        old_strategy = self.current_strategy
        self.current_strategy = new_strategy
        self.reasoning_context.current_strategy = new_strategy

        # Track strategy switches
        if not hasattr(self.reasoning_context, "strategy_switches"):
            self.reasoning_context.strategy_switches = []

        self.reasoning_context.strategy_switches.append(
            {
                "from_strategy": old_strategy.value if old_strategy else "unknown",
                "to_strategy": new_strategy.value,
                "reason": reason,
                "iteration": self.reasoning_context.iteration_count,
                "timestamp": time.time(),
            }
        )

        if self.agent_logger:
            self.agent_logger.info(
                f"✅ Switched strategy: {old_strategy.value if old_strategy else 'None'} → "
                f"{new_strategy.value}. Reason: {reason}"
            )

    async def _handle_strategy_failure(self):
        """Handle failures by switching to a more robust strategy."""

        # Define strategy robustness order (most robust last)
        robustness_order = [
            ReasoningStrategies.REACTIVE,
            ReasoningStrategies.REFLECT_DECIDE_ACT,
            ReasoningStrategies.PLAN_EXECUTE_REFLECT,
            ReasoningStrategies.ADAPTIVE,
        ]

        if not self.current_strategy:
            return

        try:
            current_index = robustness_order.index(self.current_strategy)
        except ValueError:
            # Current strategy not in robustness order, try to switch to reflect_decide_act
            if ReasoningStrategies.REFLECT_DECIDE_ACT in self.strategies:
                await self._switch_strategy(
                    ReasoningStrategies.REFLECT_DECIDE_ACT,
                    "Switching to fallback strategy due to errors",
                )
            return

        # Try to switch to next more robust strategy
        for next_strategy in robustness_order[current_index + 1 :]:
            if next_strategy in self.strategies:
                await self._switch_strategy(
                    next_strategy,
                    f"Escalating to more robust strategy due to errors",
                )
                return

        # If no more robust strategies available, log the issue
        if self.agent_logger:
            self.agent_logger.warning(
                f"No more robust strategies available. Current: {self.current_strategy.value}"
            )

    def _update_reasoning_context(self, iteration_result: Dict[str, Any]):
        """Update reasoning context based on iteration results."""
        self.reasoning_context.iteration_count += 1

        # Extract action result
        action_result = iteration_result.get("result")
        self.reasoning_context.last_action_result = action_result

        # Update success/failure counters
        if iteration_result.get("error"):
            self.reasoning_context.error_count += 1
            self.reasoning_context.stagnation_count += 1
        else:
            self.reasoning_context.stagnation_count = 0

        # Track tool usage
        if action_result and isinstance(action_result, dict):
            # Check for different tool usage patterns
            if "tool" in action_result:
                tool_name = action_result.get("tool", "unknown_tool")
                if tool_name not in self.reasoning_context.tool_usage_history:
                    self.reasoning_context.tool_usage_history.append(tool_name)
            elif "tool_calls" in action_result:
                tool_calls = action_result.get("tool_calls", [])
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "unknown_tool")
                    if tool_name not in self.reasoning_context.tool_usage_history:
                        self.reasoning_context.tool_usage_history.append(tool_name)
            elif "tools_executed" in action_result:
                # For reactive strategy that returns tools_executed count
                # We need to get the actual tool names from the tool manager
                if self.context.tool_manager and self.context.tool_manager.tool_history:
                    # Get the most recent tool calls
                    recent_tools = []
                    for entry in self.context.tool_manager.tool_history[
                        -action_result["tools_executed"] :
                    ]:
                        tool_name = entry.get("name", "unknown_tool")
                if tool_name not in self.reasoning_context.tool_usage_history:
                    self.reasoning_context.tool_usage_history.append(tool_name)

    def select_initial_strategy(
        self, task_classification: TaskClassification
    ) -> ReasoningStrategies:
        """
        Select the optimal initial strategy based on task classification.

        Args:
            task_classification: The classified task information

        Returns:
            The recommended initial reasoning strategy
        """
        task_type = task_classification.task_type.value
        complexity = task_classification.complexity_score

        # Store task classification in reasoning context
        self.reasoning_context.task_classification = task_classification.model_dump()

        # Strategy selection logic
        if task_type == "simple_lookup" or complexity < 0.3:
            selected = ReasoningStrategies.REACTIVE
        elif task_type in ["multi_step", "planning"] or complexity > 0.7:
            # Try to use plan_execute_reflect if available, fallback to reflect_decide_act
            if ReasoningStrategies.PLAN_EXECUTE_REFLECT in self.strategies:
                selected = ReasoningStrategies.PLAN_EXECUTE_REFLECT
            else:
                selected = ReasoningStrategies.REFLECT_DECIDE_ACT
        else:
            # Default to reflect_decide_act for most tasks
            selected = ReasoningStrategies.REFLECT_DECIDE_ACT

        self.current_strategy = selected
        self.reasoning_context.current_strategy = selected

        if self.agent_logger:
            self.agent_logger.info(
                f"🎯 Selected initial strategy: {selected.value} "
                f"(task_type: {task_type}, complexity: {complexity:.2f})"
            )

        return selected

    def get_reasoning_context(self) -> ReasoningContext:
        """Get the current reasoning context."""
        return self.reasoning_context

    def reset(self):
        """Reset the strategy manager for a new task."""
        self.reasoning_context = ReasoningContext(
            current_strategy=self.current_strategy
            or ReasoningStrategies.REFLECT_DECIDE_ACT
        )

        if self.agent_logger:
            self.agent_logger.debug("Strategy manager reset for new task")
