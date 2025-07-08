from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from enum import Enum
import json
from reactive_agents.core.types.reasoning_types import ReasoningContext
from reactive_agents.core.reasoning.infrastructure import Infrastructure

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class StrategyCapabilities(Enum):
    """Capabilities that a strategy can declare."""

    TOOL_EXECUTION = "tool_execution"
    PLANNING = "planning"
    REFLECTION = "reflection"
    MEMORY_USAGE = "memory_usage"
    ADAPTATION = "adaptation"
    COLLABORATION = "collaboration"


class StrategyResult:
    """Standardized result from strategy execution."""

    def __init__(
        self,
        action_taken: str,
        result: Dict[str, Any],
        should_continue: bool = True,
        confidence: float = 0.5,
        strategy_used: str = "unknown",
        next_strategy: Optional[str] = None,
        final_answer: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.action_taken = action_taken
        self.result = result
        self.should_continue = should_continue
        self.confidence = confidence
        self.strategy_used = strategy_used
        self.next_strategy = next_strategy
        self.final_answer = final_answer
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action_taken": self.action_taken,
            "result": self.result,
            "should_continue": self.should_continue,
            "confidence": self.confidence,
            "strategy_used": self.strategy_used,
            "next_strategy": self.next_strategy,
            "final_answer": self.final_answer,
            "metadata": self.metadata,
        }


class BaseReasoningStrategy(ABC):
    """
    Base class for all reasoning strategies.

    This interface provides a clean, standardized way to implement
    reasoning strategies that can be plugged into the framework.
    """

    def __init__(self, infrastructure: "Infrastructure"):
        """
        Initialize the strategy with shared infrastructure.

        Args:
            infrastructure: The reasoning infrastructure providing shared services
        """
        self.infrastructure = infrastructure
        self.context = infrastructure.context
        self.agent_logger = infrastructure.context.agent_logger

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
    async def _think(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Execute a thinking step using the infrastructure."""
        return await self.infrastructure.think(prompt)

    async def _think_chain(self, use_tools: bool = False) -> Optional[Dict[str, Any]]:
        """Execute a thinking step using the infrastructure."""
        return await self.infrastructure.think_chain(use_tools)

    async def _execute_tools(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls using the infrastructure."""
        return await self.infrastructure.execute_tools(tool_calls)

    def _preserve_context(self, key: str, value: Any) -> None:
        """Preserve important context using the infrastructure."""
        self.infrastructure.preserve_context(key, value)

    def _get_preserved_context(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Get preserved context using the infrastructure."""
        return self.infrastructure.get_preserved_context(key)

    # === Centralized Task Completion ===
    async def check_task_completion(
        self, task: str, progress_summary: str = "", **kwargs
    ) -> Dict[str, Any]:
        """
        Check if the task should be completed using centralized logic.

        Args:
            task: The original task
            progress_summary: Summary of progress made
            **kwargs: Additional context for completion checking

        Returns:
            Dict with completion information
        """
        return await self.infrastructure.should_complete_task(
            task, progress_summary=progress_summary, **kwargs
        )

    async def generate_final_answer(
        self, task: str, execution_summary: str = "", **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a final answer using centralized logic.

        Args:
            task: The original task
            execution_summary: Summary of what was accomplished
            **kwargs: Additional context for answer generation

        Returns:
            Final answer string or None
        """
        return await self.infrastructure.generate_final_answer(
            task, execution_summary, **kwargs
        )

    async def complete_task_if_ready(
        self, task: str, execution_summary: str = "", **kwargs
    ) -> StrategyResult:
        """
        Check completion and generate final answer if ready using centralized logic.

        Args:
            task: The original task
            execution_summary: Summary of what was accomplished
            **kwargs: Additional context

        Returns:
            StrategyResult indicating whether task was completed
        """
        completion_data = await self.infrastructure.complete_task_if_ready(
            task, execution_summary, **kwargs
        )
        print(completion_data)

        should_complete = completion_data.get("should_complete", False)
        final_answer = completion_data.get("final_answer")
        completion_info = completion_data.get("completion_info", {})

        action_taken = "task_completed" if should_complete else "completion_checked"

        return StrategyResult(
            action_taken=action_taken,
            result={
                "completion_check": completion_info,
                "execution_summary": execution_summary,
                "is_complete": should_complete,
            },
            should_continue=not should_complete,
            confidence=completion_info.get("confidence", 0.5),
            strategy_used=self.name,
            final_answer=final_answer,
        )

    def _extract_required_actions(self, task: str) -> List[str]:
        """Extract required actions from task description."""
        # Basic action extraction - strategies can override
        action_keywords = [
            "search",
            "find",
            "get",
            "retrieve",
            "fetch",
            "lookup",
            "check",
            "send",
            "create",
            "write",
            "generate",
            "make",
            "build",
            "compose",
            "delete",
            "remove",
            "clean",
            "clear",
            "purge",
            "unsubscribe",
            "update",
            "modify",
            "change",
            "edit",
            "alter",
            "set",
            "configure",
            "analyze",
            "review",
            "evaluate",
            "assess",
            "examine",
            "inspect",
            "list",
            "show",
            "display",
            "view",
            "read",
            "browse",
            "scan",
        ]

        task_lower = task.lower()
        actions = []

        for keyword in action_keywords:
            if keyword in task_lower:
                actions.append(keyword)

        return actions if actions else ["execute_task"]

    def _format_error_result(
        self, error: Exception, action: str = "unknown"
    ) -> StrategyResult:
        """Format an error as a StrategyResult."""
        return StrategyResult(
            action_taken=f"{action}_error",
            result={"error": str(error)},
            should_continue=False,
            confidence=0.0,
            strategy_used=self.name,
        )


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

    def create_instance(
        self, infrastructure: "Infrastructure"
    ) -> BaseReasoningStrategy:
        """Create an instance of the strategy."""
        return self.strategy_class(infrastructure)

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
