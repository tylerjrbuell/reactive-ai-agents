import pytest
from unittest.mock import MagicMock, patch

from reactive_agents.core.context.context_manager import (
    ContextManager,
    MessageRole,
    ContextWindow,
)


class TestContextManager:
    """Tests for the ContextManager component."""

    @pytest.fixture
    def mock_agent_context(self):
        """Create a mock agent context."""
        mock_context = MagicMock()
        mock_context.session.messages = []
        mock_context.agent_logger = MagicMock()
        mock_context.verbose_logging = True
        mock_context.provider_model_name = "gpt-4-turbo"

        # Mock the estimate_context_tokens method
        mock_context.estimate_context_tokens.return_value = 1000

        return mock_context

    def test_context_manager_init(self, mock_agent_context):
        """Test context manager initialization."""
        context_manager = ContextManager(mock_agent_context)

        # Verify initial state
        assert context_manager.agent_context == mock_agent_context
        assert context_manager.windows == []
        assert context_manager._current_strategy is None

        # Verify strategy configs
        assert "reactive" in context_manager.strategy_configs
        assert "plan_execute_reflect" in context_manager.strategy_configs
        assert "reflect_decide_act" in context_manager.strategy_configs
        assert "default" in context_manager.strategy_configs

    def test_set_active_strategy(self, mock_agent_context):
        """Test setting active strategy."""
        context_manager = ContextManager(mock_agent_context)

        context_manager.set_active_strategy("reactive")
        assert context_manager._current_strategy == "reactive"

        context_manager.set_active_strategy("plan_execute_reflect")
        assert context_manager._current_strategy == "plan_execute_reflect"

    def test_add_message(self, mock_agent_context):
        """Test adding a message to context."""
        context_manager = ContextManager(mock_agent_context)

        # Add message with string role
        idx1 = context_manager.add_message("system", "System message")
        assert idx1 == 0
        assert mock_agent_context.session.messages[0] == {  # type: ignore
            "role": "system",
            "content": "System message",
        }

        # Add message with MessageRole enum
        idx2 = context_manager.add_message(MessageRole.USER, "User message")
        assert idx2 == 1
        assert mock_agent_context.session.messages[1] == {  # type: ignore
            "role": "user",
            "content": "User message",
        }

        # Add message with metadata
        idx3 = context_manager.add_message(
            MessageRole.ASSISTANT, "Assistant message", {"foo": "bar"}
        )
        assert idx3 == 2
        assert mock_agent_context.session.messages[2] == {  # type: ignore
            "role": "assistant",
            "content": "Assistant message",
            "metadata": {"foo": "bar"},
        }

    def test_add_window(self, mock_agent_context):
        """Test adding a context window."""
        context_manager = ContextManager(mock_agent_context)

        # Add messages first
        context_manager.add_message("system", "System message")
        context_manager.add_message("user", "User message")

        # Add window at latest message
        window = context_manager.add_window("test_window", importance=0.8)
        assert window.name == "test_window"
        assert window.start_idx == 2  # Start at the next message index
        assert window.end_idx == 2
        assert window.importance == 0.8

        # Add window at specific index
        window2 = context_manager.add_window("window2", start_idx=0, importance=0.5)
        assert window2.name == "window2"
        assert window2.start_idx == 0
        assert window2.end_idx == 0
        assert window2.importance == 0.5

        # Verify windows are stored
        assert len(context_manager.windows) == 2
        assert context_manager.windows[0] == window
        assert context_manager.windows[1] == window2

    def test_close_window(self, mock_agent_context):
        """Test closing a context window."""
        context_manager = ContextManager(mock_agent_context)

        # Add messages
        context_manager.add_message("system", "System message")
        context_manager.add_message("user", "User message")

        # Add and close window by object
        window = context_manager.add_window("test_window")
        context_manager.add_message("assistant", "Response 1")
        context_manager.close_window(window)

        assert window.start_idx == 2  # Start at the next message index
        assert window.end_idx == 2

        # Add and close window by name
        context_manager.add_window("second_window")
        context_manager.add_message("user", "Second question")
        context_manager.add_message("assistant", "Response 2")
        context_manager.close_window("second_window")

        # Find the window and check its indices
        second_window = None
        for w in context_manager.windows:
            if w.name == "second_window":
                second_window = w
                break

        assert second_window is not None
        assert second_window.start_idx == 3  # Start at the next message index
        assert second_window.end_idx == 4

    def test_get_messages_by_role(self, mock_agent_context):
        """Test getting messages by role."""
        context_manager = ContextManager(mock_agent_context)

        # Add messages with different roles
        context_manager.add_message("system", "System message")
        context_manager.add_message("user", "User message 1")
        context_manager.add_message("assistant", "Assistant message 1")
        context_manager.add_message("user", "User message 2")
        context_manager.add_message("assistant", "Assistant message 2")

        # Get messages by string role
        user_messages = context_manager.get_messages_by_role("user")
        assert len(user_messages) == 2
        assert user_messages[0]["content"] == "User message 1"
        assert user_messages[1]["content"] == "User message 2"

        # Get messages by enum role
        assistant_messages = context_manager.get_messages_by_role(MessageRole.ASSISTANT)
        assert len(assistant_messages) == 2
        assert assistant_messages[0]["content"] == "Assistant message 1"
        assert assistant_messages[1]["content"] == "Assistant message 2"

    def test_should_preserve_message(self, mock_agent_context):
        """Test message preservation rules."""
        context_manager = ContextManager(mock_agent_context)
        context_manager.set_active_strategy("reactive")

        # User messages should be preserved by default for reactive strategy
        user_msg = {"role": "user", "content": "Test"}
        assert context_manager.should_preserve_message(user_msg) is True

        # Assistant messages should not be preserved by default
        assistant_msg = {"role": "assistant", "content": "Test"}
        assert context_manager.should_preserve_message(assistant_msg) is False

        # Messages with preserve flag should be preserved
        meta_msg = {"role": "tool", "content": "Test", "metadata": {"preserve": True}}
        assert context_manager.should_preserve_message(meta_msg) is True

        # Add custom preservation rule
        context_manager.add_preservation_rule(lambda msg: msg.get("role") == "tool")
        tool_msg = {"role": "tool", "content": "Test"}
        assert context_manager.should_preserve_message(tool_msg) is True

    def test_get_pruning_config(self, mock_agent_context):
        """Test getting pruning configuration."""
        context_manager = ContextManager(mock_agent_context)

        # Test with different model names
        mock_agent_context.provider_model_name = "gpt-4"
        config = context_manager._get_pruning_config()
        assert config["max_tokens"] == 120000
        assert config["max_messages"] == 60

        mock_agent_context.provider_model_name = "gpt-3.5-turbo"
        config = context_manager._get_pruning_config()
        assert config["max_tokens"] == 12000
        assert config["max_messages"] == 40

        mock_agent_context.provider_model_name = "claude-3-opus"
        config = context_manager._get_pruning_config()
        assert config["max_tokens"] == 180000
        assert config["max_messages"] == 80

        mock_agent_context.provider_model_name = "unknown-model"
        config = context_manager._get_pruning_config()
        assert config["max_tokens"] == 8000
        assert config["max_messages"] == 30

    def test_generate_summary(self, mock_agent_context):
        """Test summary generation."""
        context_manager = ContextManager(mock_agent_context)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm fine"},
        ]

        summary = context_manager._generate_summary(messages, 2, 5)
        assert "[Summary of 4 messages from indices 2-5" in summary
        assert "2 user messages" in summary
        assert "2 assistant messages" in summary
