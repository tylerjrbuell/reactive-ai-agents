from __future__ import annotations
from typing import List, Dict, Any, Optional, Union, Callable, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class MessageRole(Enum):
    """Standard message roles for context management."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass
class ContextWindow:
    """
    Represents a logical window of related messages in the context.
    Used to group messages that should be managed together.
    """

    name: str
    start_idx: int
    end_idx: int
    importance: float = 1.0  # Higher = less likely to be pruned
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """
    Manages context for reasoning strategies, providing strategy-aware
    context manipulation, preservation, and pruning.

    This class centralizes all operations related to managing the agent's
    conversation context, including adding messages, pruning, summarizing,
    and preserving important information.
    """

    def __init__(self, agent_context: "AgentContext"):
        """
        Initialize the context manager.

        Args:
            agent_context: The agent context to manage
        """
        self.agent_context = agent_context
        self.windows: List[ContextWindow] = []
        self.preservation_rules: List[Callable[[Dict[str, Any]], bool]] = []
        self._current_strategy: str | None = None

        # Strategy-specific pruning configurations
        self.strategy_configs = {
            "reactive": {
                "summarization_frequency": 4,
                "token_threshold_multiplier": 1.0,
                "message_threshold_multiplier": 1.0,
                "preserved_roles": [MessageRole.USER],
            },
            "plan_execute_reflect": {
                "summarization_frequency": 8,
                "token_threshold_multiplier": 1.5,
                "message_threshold_multiplier": 1.5,
                "preserved_roles": [MessageRole.USER, MessageRole.ASSISTANT],
            },
            "reflect_decide_act": {
                "summarization_frequency": 6,
                "token_threshold_multiplier": 1.2,
                "message_threshold_multiplier": 1.3,
                "preserved_roles": [MessageRole.USER, MessageRole.SYSTEM],
            },
            # Default configuration for any strategy
            "default": {
                "summarization_frequency": 4,
                "token_threshold_multiplier": 1.0,
                "message_threshold_multiplier": 1.0,
                "preserved_roles": [MessageRole.USER],
            },
        }

    def set_active_strategy(self, strategy_name: str | None) -> None:
        """
        Set the current active strategy to adjust context management behavior.

        Args:
            strategy_name: Name of the active strategy
        """
        self._current_strategy = strategy_name
        if self.agent_context.agent_logger:
            self.agent_context.agent_logger.debug(
                f"Context manager: Set active strategy to {strategy_name}"
            )

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages in the context.

        Returns:
            List of messages
        """
        return self.agent_context.session.messages

    def add_message(
        self,
        role: Union[str, MessageRole],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add a message to the context.

        Args:
            role: Role of the message sender
            content: Content of the message
            metadata: Optional metadata for the message

        Returns:
            Index of the added message
        """
        role_value = role.value if isinstance(role, MessageRole) else role
        message: Dict[str, Any] = {"role": role_value, "content": content}
        if metadata:
            message["metadata"] = metadata  # type: ignore

        self.agent_context.session.messages.append(message)

        # Log the addition if logging is enabled
        if self.agent_context.agent_logger:
            self.agent_context.agent_logger.debug(
                f"Context manager: Added {role_value} message, index={len(self.messages) - 1}"
            )

        return len(self.messages) - 1

    def add_window(
        self, name: str, start_idx: Optional[int] = None, importance: float = 1.0
    ) -> ContextWindow:
        """
        Create a new context window starting at the given index or latest message.

        Args:
            name: Name of the window
            start_idx: Starting message index
            importance: Importance value (0.0-1.0)

        Returns:
            The created window
        """
        if start_idx is None:
            start_idx = len(self.messages) - 1

        window = ContextWindow(
            name=name, start_idx=start_idx, end_idx=start_idx, importance=importance
        )
        self.windows.append(window)

        # Log the window creation if logging is enabled
        if self.agent_context.agent_logger:
            self.agent_context.agent_logger.debug(
                f"Context manager: Created window '{name}', start={start_idx}, importance={importance}"
            )

        return window

    def close_window(self, window: Union[str, ContextWindow]) -> None:
        """
        Close a context window at the current message.

        Args:
            window: Window or window name to close
        """
        if isinstance(window, str):
            # Find by name
            for w in self.windows:
                if w.name == window:
                    window = w
                    break

        if isinstance(window, ContextWindow):
            window.end_idx = len(self.messages) - 1

            # Log the window closure if logging is enabled
            if self.agent_context.agent_logger:
                self.agent_context.agent_logger.debug(
                    f"Context manager: Closed window '{window.name}', span={window.start_idx}-{window.end_idx}"
                )

    def get_messages_by_role(
        self, role: Union[str, MessageRole]
    ) -> List[Dict[str, Any]]:
        """
        Get all messages with a specific role.

        Args:
            role: Role to filter by

        Returns:
            List of messages with the specified role
        """
        role_value = role.value if isinstance(role, MessageRole) else role
        return [m for m in self.messages if m.get("role") == role_value]

    def get_messages_in_window(
        self, window: Union[str, ContextWindow]
    ) -> List[Dict[str, Any]]:
        """
        Get all messages in a specific window.

        Args:
            window: Window or window name to get messages from

        Returns:
            List of messages in the window
        """
        if isinstance(window, str):
            # Find by name
            for w in self.windows:
                if w.name == window:
                    window = w
                    break

        if isinstance(window, ContextWindow):
            return self.messages[window.start_idx : window.end_idx + 1]
        return []

    def add_preservation_rule(self, rule: Callable[[Dict[str, Any]], bool]) -> None:
        """
        Add a rule for preserving messages during pruning.

        Args:
            rule: Function that takes a message and returns True if it should be preserved
        """
        self.preservation_rules.append(rule)

    def should_preserve_message(self, message: Dict[str, Any]) -> bool:
        """
        Check if a message should be preserved during pruning.

        Args:
            message: The message to check

        Returns:
            True if the message should be preserved
        """
        # Check custom rules
        for rule in self.preservation_rules:
            if rule(message):
                return True

        # Check strategy-specific preserved roles
        if self._current_strategy and self._current_strategy in self.strategy_configs:
            config = self.strategy_configs[self._current_strategy]
            preserved_role_values = [r.value for r in config.get("preserved_roles", [])]
            if message.get("role") in preserved_role_values:
                return True

        # Check metadata for preservation flag
        metadata = message.get("metadata", {})
        if metadata.get("preserve") is True:
            return True

        return False

    def summarize_and_prune(self, force: bool = False) -> bool:
        """
        Check if summarization/pruning should occur and perform it if needed.

        Args:
            force: Force summarization/pruning regardless of thresholds

        Returns:
            True if pruning occurred
        """
        # Get strategy-specific configuration
        config = self.strategy_configs.get(
            self._current_strategy if self._current_strategy else "default"
        )

        base_config = self._get_pruning_config()

        # Calculate thresholds based on strategy
        summarization_frequency = max(
            4,
            getattr(self.agent_context, "context_summarization_frequency", 5)
            * (config.get("summarization_frequency", 1) if config else 1),
        )

        max_tokens_threshold = base_config["max_tokens"] * (
            config.get("token_threshold_multiplier", 1.0) if config else 1.0
        )
        max_messages_threshold = base_config["max_messages"] * (
            config.get("message_threshold_multiplier", 1.0) if config else 1.0
        )

        current_iteration = self.agent_context.session.iterations

        # Determine if we should summarize/prune
        should_summarize = force or (
            getattr(self.agent_context, "enable_context_summarization", True)
            and (current_iteration % summarization_frequency == 0)
        )

        should_prune = force or (
            getattr(self.agent_context, "enable_context_pruning", True)
            and (
                self.agent_context.estimate_context_tokens() > max_tokens_threshold
                or len(self.messages) > max_messages_threshold
            )
        )

        # Log decision
        if self.agent_context.agent_logger:
            self.agent_context.agent_logger.debug(
                f"Context manager ({self._current_strategy or 'default'}): "
                f"iteration={current_iteration}, "
                f"should_summarize={should_summarize}, should_prune={should_prune}, "
                f"tokens={self.agent_context.estimate_context_tokens()}, max_tokens={max_tokens_threshold}, "
                f"messages={len(self.messages)}, max_messages={max_messages_threshold}"
            )

        if should_summarize or should_prune:
            # Perform actual summarization and pruning
            self._perform_summarize_and_prune(should_summarize)
            return True

        return False

    def _get_pruning_config(self) -> Dict[str, Any]:
        """
        Get the pruning configuration.

        Returns:
            Pruning configuration
        """
        # Try to use the agent's method if available
        if hasattr(self.agent_context, "get_optimal_pruning_config"):
            return self.agent_context.get_optimal_pruning_config()

        # Fallback to reasonable defaults
        model_name = getattr(self.agent_context, "provider_model_name", "")
        if "gpt-4" in model_name:
            return {"max_tokens": 120000, "max_messages": 60}
        elif "gpt-3.5" in model_name:
            return {"max_tokens": 12000, "max_messages": 40}
        elif "claude-3" in model_name:
            return {"max_tokens": 180000, "max_messages": 80}
        else:
            return {"max_tokens": 8000, "max_messages": 30}

    def _perform_summarize_and_prune(self, should_summarize: bool = True) -> None:
        """
        Actually perform the summarization and pruning operations.

        Args:
            should_summarize: Whether summarization should be performed
        """
        if len(self.messages) < 3:
            # Not enough messages to summarize/prune
            return

        preserved_indices = set()
        prunable_indices = []

        # First pass: identify messages to preserve
        for i, message in enumerate(self.messages):
            # Always keep the first system message and the last few messages
            if i == 0 and message.get("role") == "system":
                preserved_indices.add(i)
                continue

            # Keep the most recent messages
            if i >= len(self.messages) - 3:
                preserved_indices.add(i)
                continue

            if self.should_preserve_message(message):
                preserved_indices.add(i)
            else:
                # Check if message is in an important window
                in_important_window = False
                for window in self.windows:
                    if (
                        window.start_idx <= i <= window.end_idx
                        and window.importance > 0.7
                        and (len(self.messages) - i)
                        > 5  # Not one of the most recent messages
                    ):
                        in_important_window = True
                        break

                prunable_indices.append((i, message, in_important_window))

        # If summarization is requested, create summary chunks for prunable segments
        if should_summarize:
            # Find contiguous chunks of prunable messages
            chunks = self._identify_prunable_chunks(prunable_indices, preserved_indices)

            # Summarize each chunk
            for chunk in chunks:
                if len(chunk) >= 3:  # Only summarize if enough messages
                    start_idx, end_idx = chunk[0][0], chunk[-1][0]
                    chunk_messages = [m[1] for m in chunk]
                    summary = self._generate_summary(chunk_messages, start_idx, end_idx)

                    # Add summary message where the chunk starts
                    self.messages[start_idx] = {
                        "role": MessageRole.ASSISTANT.value,
                        "content": summary,
                        "metadata": {
                            "is_summary": True,
                            "summarized_range": [start_idx, end_idx],
                            "preserve": True,
                        },
                    }
                    # Mark all other messages in the chunk for removal
                    indices_to_remove = [
                        i
                        for i in range(start_idx + 1, end_idx + 1)
                        if i not in preserved_indices
                    ]
                    for i in sorted(indices_to_remove, reverse=True):
                        del self.messages[i]

        # Sync the session messages with the updated messages list
        self.agent_context.session.messages = self.messages.copy()

        # Log the results if logging is enabled
        if self.agent_context.agent_logger:
            self.agent_context.agent_logger.info(
                f"Context manager: Pruned context from {len(self.messages)} to "
                f"{len(self.agent_context.session.messages)} messages"
            )

    def _identify_prunable_chunks(
        self, prunable_indices: List[tuple], preserved_indices: set
    ) -> List[List[tuple]]:
        """
        Identify contiguous chunks of prunable messages.

        Args:
            prunable_indices: List of (index, message, in_important_window) tuples
            preserved_indices: Set of indices that must be preserved

        Returns:
            List of chunks, where each chunk is a list of prunable message tuples
        """
        chunks = []
        current_chunk = []

        for item in sorted(prunable_indices):
            idx = item[0]
            if not current_chunk or idx == current_chunk[-1][0] + 1:
                # Contiguous message
                current_chunk.append(item)
            else:
                # Start of a new chunk
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [item]

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _generate_summary(
        self, messages: List[Dict[str, Any]], start_idx: int, end_idx: int
    ) -> str:
        """
        Generate a summary of the messages.

        Args:
            messages: Messages to summarize
            start_idx: Starting index
            end_idx: Ending index

        Returns:
            Summary text
        """
        # Implement your summarization logic here, for now we'll use a simple approach
        roles = [m.get("role", "unknown") for m in messages]
        role_counts = {role: roles.count(role) for role in set(roles)}

        # Create a simple summary
        summary = (
            f"[Summary of {len(messages)} messages from indices {start_idx}-{end_idx}: "
        )
        for role, count in role_counts.items():
            summary += f"{count} {role} messages, "
        summary = summary.rstrip(", ") + "]"

        # TODO: Implement more sophisticated summarization using LLM
        # This would typically use the agent's LLM to create a better summary

        return summary

    # Additional utility methods
    def get_latest_n_messages(self, n: int) -> List[Dict[str, Any]]:
        """
        Get the latest N messages from context.

        Args:
            n: Number of messages to retrieve

        Returns:
            List of messages
        """
        return self.messages[-n:] if len(self.messages) >= n else self.messages

    def get_context_for_strategy(self) -> List[Dict[str, Any]]:
        """
        Get optimized context based on current strategy.

        Returns:
            List of messages optimized for the current strategy
        """
        # Strategy-specific logic for constructing optimal context
        if not self._current_strategy:
            return self.messages

        if self._current_strategy == "reactive":
            # Reactive strategies work best with most recent context
            return self.get_latest_n_messages(20)

        elif self._current_strategy == "plan_execute_reflect":
            # Plan-execute-reflect needs planning context
            return self._get_plan_execute_reflect_context()

        elif self._current_strategy == "reflect_decide_act":
            # Reflect-decide-act needs reflection context
            return self._get_reflect_decide_act_context()

        # Default: return all messages
        return self.messages

    def _get_plan_execute_reflect_context(self) -> List[Dict[str, Any]]:
        """
        Get optimized context for plan-execute-reflect strategy.

        Returns:
            List of messages
        """
        # Find plan windows
        plan_window = None
        for window in self.windows:
            if "plan" in window.name.lower():
                plan_window = window
                break

        if plan_window:
            # Include the plan window and recent messages
            plan_messages = self.get_messages_in_window(plan_window)
            recent_messages = self.get_latest_n_messages(10)

            # Combine and deduplicate
            context = []
            seen_indices = set()

            # Add system messages first
            for i, message in enumerate(self.messages):
                if message.get("role") == "system" and i not in seen_indices:
                    context.append(message)
                    seen_indices.add(i)

            # Add plan messages
            for message in plan_messages:
                idx = self.messages.index(message)
                if idx not in seen_indices:
                    context.append(message)
                    seen_indices.add(idx)

            # Add recent messages
            for message in recent_messages:
                idx = self.messages.index(message)
                if idx not in seen_indices:
                    context.append(message)
                    seen_indices.add(idx)

            return context

        # Fallback: recent messages plus summaries
        context = []
        for message in self.messages:
            if (
                message.get("role") == "system"
                or (message.get("metadata", {}).get("is_summary") is True)
                or message in self.get_latest_n_messages(15)
            ):
                context.append(message)

        return context

    def _get_reflect_decide_act_context(self) -> List[Dict[str, Any]]:
        """
        Get optimized context for reflect-decide-act strategy.

        Returns:
            List of messages
        """
        # Prioritize reflection messages and recent exchanges
        reflection_messages = [
            m
            for m in self.messages
            if m.get("metadata", {}).get("type") == "reflection"
        ]

        # Get recent messages
        recent_messages = self.get_latest_n_messages(12)

        # Combine and deduplicate
        context = []
        seen_indices = set()

        # Add system messages first
        for i, message in enumerate(self.messages):
            if message.get("role") == "system" and i not in seen_indices:
                context.append(message)
                seen_indices.add(i)

        # Add reflection messages
        for message in reflection_messages:
            idx = self.messages.index(message)
            if idx not in seen_indices:
                context.append(message)
                seen_indices.add(idx)

        # Add recent messages
        for message in recent_messages:
            idx = self.messages.index(message)
            if idx not in seen_indices:
                context.append(message)
                seen_indices.add(idx)

        return context
