from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from enum import Enum
import json
from reactive_agents.core.types.agent_types import (
    AgentThinkChainResult,
    AgentThinkResult,
)
from reactive_agents.core.types.reasoning_types import ReasoningContext
from reactive_agents.core.reasoning.engine import ReasoningEngine

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
    """Result from a strategy iteration."""

    def __init__(
        self,
        action_taken: str,
        should_continue: bool = True,
        final_answer: Optional[str] = None,
        status: str = "unknown",
        evaluation: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize a strategy result.

        Args:
            action_taken: The action that was taken
            should_continue: Whether to continue executing
            final_answer: Optional final answer if task is complete
            status: Status of the action
            evaluation: Task evaluation result if available
            **kwargs: Additional result data
        """
        self.action_taken = action_taken
        self.should_continue = should_continue
        self.final_answer = final_answer
        self.status = status
        self.evaluation = evaluation or {}
        self.additional_data = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""

        # Create a temporary instance to use _make_serializable
        # We'll create a simple helper function here
        def make_serializable(obj: Any) -> Any:
            """Convert objects to JSON-serializable format."""
            try:
                # Handle Pydantic models
                if hasattr(obj, "model_dump"):
                    return obj.model_dump()
                elif hasattr(obj, "dict"):
                    return obj.dict()

                # Handle dictionaries
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}

                # Handle lists and tuples
                if isinstance(obj, (list, tuple)):
                    return [make_serializable(item) for item in obj]

                # Handle basic types
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj

                # Handle sets
                if isinstance(obj, set):
                    return list(obj)

                # For other objects, try to convert to string
                return str(obj)

            except Exception:
                # If all else fails, return string representation
                return str(obj)

        result_dict = {
            "action_taken": self.action_taken,
            "should_continue": self.should_continue,
            "final_answer": self.final_answer,
            "status": self.status,
            "evaluation": make_serializable(self.evaluation),
        }

        # Add additional data with serialization
        for key, value in self.additional_data.items():
            result_dict[key] = make_serializable(value)

        return result_dict


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

    async def execute_with_tools(
        self, task: str, step_description: str, use_native_tools: bool = True
    ) -> Dict[str, Any]:
        """
        Standard method for executing tools in any strategy.

        Args:
            task: The main task
            step_description: What to accomplish in this step
            use_native_tools: Whether to use native tool calling or manual prompting

        Returns:
            Execution results with consistent format
        """
        return await self._execute_tool_for_task(
            task, step_description, use_native_tools
        )

    async def reflect_on_progress(
        self,
        task: str,
        execution_results: Dict[str, Any],
        reasoning_context: ReasoningContext,
    ) -> Dict[str, Any]:
        """
        Standard method for reflecting on progress in any strategy.

        Args:
            task: The main task
            execution_results: Results from tool execution
            reasoning_context: Current reasoning context

        Returns:
            Reflection results with consistent format
        """
        # Convert execution results to JSON-serializable format
        serializable_results = self._make_serializable(execution_results)

        # Create reflection prompt
        reflection_prompt = f"""Task: {task}
        
Recent execution results:
{json.dumps(serializable_results, indent=2)}

Current context:
- Iteration: {reasoning_context.iteration_count}
- Error count: {reasoning_context.error_count}
- Tools used: {reasoning_context.tool_usage_history}

Please reflect on the progress and provide guidance in this JSON format:
{{
    "progress_assessment": "Brief summary of what has been accomplished",
    "goal_achieved": true/false,
    "completion_score": 0.0-1.0,
    "next_action": "continue|retry|complete",
    "confidence": 0.0-1.0,
    "blockers": ["list of current blockers"],
    "success_indicators": ["list of positive indicators"],
    "reasoning": "Your reasoning about the current state"
}}

Only respond with valid JSON, no additional text."""

        # Add to context and get response
        self.context_manager.add_message(role="user", content=reflection_prompt)
        result = await self._think_chain(use_tools=False)

        if result and result.result_json:
            return result.result_json

        # Fallback reflection
        return {
            "progress_assessment": "Unable to generate proper reflection",
            "goal_achieved": False,
            "completion_score": 0.0,
            "next_action": "continue",
            "confidence": 0.2,
            "blockers": ["Reflection generation failed"],
            "success_indicators": [],
            "reasoning": "Failed to generate reflection",
        }

    async def evaluate_task_completion(
        self, task: str, execution_summary: str = "", **kwargs
    ) -> Dict[str, Any]:
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
        return await self.engine.should_complete_task(
            task, execution_summary=execution_summary, **kwargs
        )

    async def generate_final_answer(
        self, task: str, execution_summary: str = "", **kwargs
    ) -> Dict[str, Any]:
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
        result = await self.engine.generate_final_answer(
            task, execution_summary=execution_summary, **kwargs
        )

        if result:
            return result

        # Fallback final answer
        return {
            "final_answer": f"I worked on the task: {task}. {execution_summary}",
            "confidence": 0.5,
            "method": "fallback",
        }

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
    async def _think(self, prompt: str) -> Optional[AgentThinkResult]:
        """Execute a thinking step using the engine."""
        return await self.engine.think(prompt)

    async def _think_chain(
        self, use_tools: bool = False
    ) -> Optional[AgentThinkChainResult]:
        """Execute a thinking step using the engine."""
        return await self.engine.think_chain(use_tools)

    async def _execute_tools(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls using the engine."""
        return await self.engine.execute_tools(tool_calls)

    async def _execute_tool_for_task(
        self, task: str, step_description: str, use_tools: bool = True
    ) -> Dict[str, Any]:
        """
        Unified tool execution method that works for both native and non-native models.

        Args:
            task: The main task
            step_description: Description of the current step
            use_tools: Whether to use native tool calling (True) or manual prompting (False)

        Returns:
            Dictionary with tool execution results
        """
        if use_tools:
            # Try native tool calling first
            try:
                # Add context for tool selection
                self.context_manager.add_message(
                    role="user",
                    content=f"Task: {task}\nStep: {step_description}\nPlease use the appropriate tool to complete this step.",
                )

                result = await self._think_chain(use_tools=True)
                if result and result.tool_calls:
                    return {
                        "tool_calls": result.tool_calls,
                        "reasoning": result.content,
                        "method": "native_tool_calling",
                    }
                else:
                    # Fall back to manual prompting
                    return await self._manual_tool_prompting(task, step_description)
            except Exception as e:
                if self.agent_logger:
                    self.agent_logger.warning(
                        f"Native tool calling failed: {e}, falling back to manual prompting"
                    )
                return await self._manual_tool_prompting(task, step_description)
        else:
            # Use manual tool prompting for non-native models
            return await self._manual_tool_prompting(task, step_description)

    async def _manual_tool_prompting(
        self, task: str, step_description: str
    ) -> Dict[str, Any]:
        """
        Manual tool prompting for models that don't support native tool calling.

        Args:
            task: The main task
            step_description: Description of the current step

        Returns:
            Dictionary with tool execution results
        """
        # Get available tools for prompting
        tool_signatures = []
        if self.context.tool_manager:
            tool_signatures = [
                {
                    "name": tool.name,
                    "description": tool.tool_definition.get("function", {}).get(
                        "description", ""
                    ),
                    "parameters": tool.tool_definition.get("function", {}).get(
                        "parameters", {}
                    ),
                }
                for tool in self.context.tool_manager.tools
            ]

        # Create manual tool selection prompt
        prompt = f"""Task: {task}
Step: {step_description}

Available tools:
{json.dumps(tool_signatures, indent=2)}

Please select the most appropriate tool and provide the parameters in this exact JSON format:
{{
    "tool_calls": [
        {{
            "function": {{
                "name": "<tool_name>",
                "arguments": {{"param": "value"}}
            }}
        }}
    ],
    "reasoning": "Why I chose this tool and these parameters"
}}

Only respond with valid JSON, no additional text."""

        # Add the prompt to context
        self.context_manager.add_message(role="user", content=prompt)

        # Get model response
        result = await self._think_chain(use_tools=False)
        if result and result.result_json:
            tool_calls = result.result_json.get("tool_calls", [])
            if tool_calls:
                # Execute the selected tools
                execution_results = await self._execute_tools(tool_calls)
                return {
                    "tool_calls": tool_calls,
                    "execution_results": execution_results,
                    "reasoning": result.result_json.get("reasoning", ""),
                    "method": "manual_tool_prompting",
                }

        return {
            "error": "Failed to select or execute tools",
            "method": "manual_tool_prompting",
        }

    @property
    def context_manager(self):
        """Get the context manager from the engine."""
        return self.engine.get_context_manager()

    def _preserve_context(self, key: str, value: Any) -> None:
        """Preserve important context using the engine."""
        self.engine.preserve_context(key, value)

    def _get_preserved_context(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Get preserved context using the engine."""
        return self.engine.get_preserved_context(key)

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

    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert objects to JSON-serializable format.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable version of the object
        """
        try:
            # Handle Pydantic models
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            elif hasattr(obj, "dict"):
                return obj.dict()

            # Handle dictionaries
            if isinstance(obj, dict):
                return {k: self._make_serializable(v) for k, v in obj.items()}

            # Handle lists and tuples
            if isinstance(obj, (list, tuple)):
                return [self._make_serializable(item) for item in obj]

            # Handle basic types
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj

            # Handle sets
            if isinstance(obj, set):
                return list(obj)

            # For other objects, try to convert to string
            return str(obj)

        except Exception:
            # If all else fails, return string representation
            return str(obj)

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
