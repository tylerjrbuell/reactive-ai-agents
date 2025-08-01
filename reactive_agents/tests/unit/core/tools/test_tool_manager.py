"""
Tests for ToolManager.

Tests the tool management functionality including tool discovery, execution,
caching, validation, and integration with SOLID components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock, create_autospec
from reactive_agents.core.tools.tool_manager import ToolManager
from reactive_agents.core.tools.base import Tool
from reactive_agents.core.tools.abstractions import ToolProtocol, MCPToolWrapper
from reactive_agents.core.tools.tool_guard import ToolGuard
from reactive_agents.core.tools.tool_cache import ToolCache
from reactive_agents.core.tools.tool_confirmation import ToolConfirmation
from reactive_agents.core.tools.tool_validator import ToolValidator
from reactive_agents.core.tools.tool_executor import ToolExecutor
from reactive_agents.core.tools.default import FinalAnswerTool
from reactive_agents.core.context.agent_context import AgentContext
from reactive_agents.core.types.event_types import AgentStateEvent


class TestToolManager:
    """Test cases for ToolManager."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock agent context."""
        context = create_autospec(AgentContext, instance=True)
        context.agent_name = "TestAgent"
        context.enable_caching = True
        context.cache_ttl = 3600
        context.confirmation_callback = None
        context.confirmation_config = None
        context.tool_use_enabled = True
        context.collect_metrics_enabled = True
        context.tools = []
        context.mcp_client = None

        # Mock loggers
        context.agent_logger = Mock()
        context.tool_logger = Mock()
        context.model_provider = Mock()

        # Mock session
        context.session = Mock()
        context.session.successful_tools = set()

        # Mock metrics manager
        context.metrics_manager = Mock()
        context.metrics_manager.update_tool_metrics = Mock()
        context.metrics_manager.get_metrics = Mock(return_value={})

        # Mock event emission
        context.emit_event = Mock()

        return context

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        tool = Mock(spec=ToolProtocol)
        tool.name = "test_tool"
        tool.tool_definition = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "First parameter"}
                    },
                    "required": ["param1"],
                },
            },
        }
        return tool

    @pytest.fixture
    def tool_manager(self, mock_context):
        """Create a tool manager instance with mocked dependencies."""
        # Mock component instances first
        mock_guard = Mock(spec=ToolGuard)
        mock_guard.add_default_guards = Mock()
        mock_guard.can_use = Mock(return_value=True)
        mock_guard.needs_confirmation = Mock(return_value=False)
        mock_guard.record_use = Mock()

        mock_cache = Mock(spec=ToolCache)
        mock_cache.enabled = True
        mock_cache.ttl = 3600
        mock_cache.hits = 0
        mock_cache.misses = 0
        mock_cache.generate_cache_key = Mock(return_value="test_key")
        mock_cache.get = Mock(return_value=None)
        mock_cache.put = Mock()

        mock_confirmation = Mock(spec=ToolConfirmation)
        mock_confirmation.tool_requires_confirmation = Mock(return_value=False)
        mock_confirmation.request_confirmation = AsyncMock(return_value=(True, None))
        mock_confirmation.inject_user_feedback = Mock()

        mock_validator = Mock(spec=ToolValidator)
        mock_validator.validate_tool_result_usage = Mock(
            return_value={"valid": True, "warnings": [], "suggestions": []}
        )
        mock_validator.store_search_data = Mock()

        mock_executor = Mock(spec=ToolExecutor)

        # Make parse_tool_arguments dynamic based on the actual call
        def parse_tool_arguments_side_effect(tool_call):
            # Check for invalid format
            if "function" not in tool_call:
                raise ValueError("Tool call missing function name")
            function = tool_call.get("function", {})
            name = function.get("name", "test_tool")
            if not name:
                raise ValueError("Tool call missing function name")
            args = function.get("arguments", {"param1": "value1"})
            return (name, args)

        mock_executor.parse_tool_arguments = Mock(
            side_effect=parse_tool_arguments_side_effect
        )
        mock_executor.execute_tool = AsyncMock(
            return_value="Tool executed successfully"
        )
        mock_executor.add_reasoning_to_context = Mock()
        mock_executor._generate_tool_summary = AsyncMock(return_value="Tool summary")

        # Create ToolManager normally, then replace components after initialization
        manager = ToolManager(context=mock_context)

        # Replace the actual components with our mocks
        manager.guard = mock_guard
        manager.cache = mock_cache
        manager.confirmation = mock_confirmation
        manager.validator = mock_validator
        manager.executor = mock_executor

        # Store mocks for test access
        manager._mock_guard = mock_guard
        manager._mock_cache = mock_cache
        manager._mock_confirmation = mock_confirmation
        manager._mock_validator = mock_validator
        manager._mock_executor = mock_executor

        return manager

    def test_initialization(self, tool_manager, mock_context):
        """Test tool manager initialization."""
        assert tool_manager.context == mock_context
        assert tool_manager.tools == []
        assert tool_manager.tool_signatures == []
        assert tool_manager.tool_history == []

        # Verify components were initialized
        assert tool_manager.guard is not None
        assert tool_manager.cache is not None
        assert tool_manager.confirmation is not None
        assert tool_manager.validator is not None
        assert tool_manager.executor is not None

        # Verify that our mocks are properly set up
        assert hasattr(tool_manager.guard, "add_default_guards")
        assert hasattr(tool_manager.guard, "can_use")
        assert hasattr(tool_manager.guard, "needs_confirmation")
        assert hasattr(tool_manager.guard, "record_use")

    def test_properties(self, tool_manager, mock_context):
        """Test tool manager properties."""
        assert tool_manager.agent_logger == mock_context.agent_logger
        assert tool_manager.tool_logger == mock_context.tool_logger
        assert tool_manager.model_provider == mock_context.model_provider

        # Test cache properties
        assert tool_manager.cache_hits == 0
        assert tool_manager.cache_misses == 0
        assert tool_manager.enable_caching is True
        assert tool_manager.cache_ttl == 3600

    @pytest.mark.asyncio
    async def test_initialize_tools_empty(self, tool_manager):
        """Test tool initialization with no tools."""
        with patch.object(tool_manager, "_generate_tool_signatures"), patch(
            "reactive_agents.core.tools.tool_manager.FinalAnswerTool"
        ) as mock_final_answer:

            mock_tool = Mock()
            mock_tool.name = "final_answer"
            mock_final_answer.return_value = mock_tool

            await tool_manager._initialize_tools()

            # Should have injected final_answer tool
            assert len(tool_manager.tools) == 1
            assert tool_manager.tools[0].name == "final_answer"

    @pytest.mark.asyncio
    async def test_initialize_tools_with_mcp(self, tool_manager, mock_context):
        """Test tool initialization with MCP tools."""
        # Mock MCP client with tools
        mock_mcp_client = Mock()
        mock_tool1 = Mock()
        mock_tool1.name = "mcp_tool1"
        mock_tool2 = Mock()
        mock_tool2.name = "mcp_tool2"
        mock_mcp_tools = [mock_tool1, mock_tool2]
        mock_mcp_client.tools = mock_mcp_tools
        mock_mcp_client.tool_signatures = [
            {"type": "function", "function": {"name": "mcp_tool1"}}
        ]
        mock_mcp_client.server_tools = {"server1": mock_mcp_tools}
        mock_context.mcp_client = mock_mcp_client

        with patch.object(tool_manager, "_generate_tool_signatures"), patch(
            "reactive_agents.core.tools.tool_manager.MCPToolWrapper"
        ) as mock_wrapper, patch(
            "reactive_agents.core.tools.tool_manager.FinalAnswerTool"
        ) as mock_final_answer:

            def wrapper_side_effect(tool, client):
                wrapped = Mock()
                wrapped.name = f"wrapped_{tool.name}"
                return wrapped

            mock_wrapper.side_effect = wrapper_side_effect

            mock_final_tool = Mock()
            mock_final_tool.name = "final_answer"
            mock_final_answer.return_value = mock_final_tool

            await tool_manager._initialize_tools()

            # Should have MCP tools + final_answer tool
            assert len(tool_manager.tools) == 3
            assert len(tool_manager.tool_signatures) == 1

    @pytest.mark.asyncio
    async def test_initialize_tools_with_custom_tools(
        self, tool_manager, mock_context, mock_tool
    ):
        """Test tool initialization with custom tools."""
        mock_context.tools = [mock_tool]

        with patch.object(tool_manager, "_generate_tool_signatures"), patch(
            "reactive_agents.core.tools.tool_manager.FinalAnswerTool"
        ) as mock_final_answer:

            mock_final_tool = Mock()
            mock_final_tool.name = "final_answer"
            mock_final_answer.return_value = mock_final_tool

            await tool_manager._initialize_tools()

            # Should have custom tool + final_answer tool
            assert len(tool_manager.tools) == 2
            assert mock_tool in tool_manager.tools

    def test_get_tool(self, tool_manager, mock_tool):
        """Test getting a tool by name."""
        tool_manager.tools = [mock_tool]

        # Test finding existing tool
        found_tool = tool_manager.get_tool("test_tool")
        assert found_tool == mock_tool

        # Test tool not found
        not_found = tool_manager.get_tool("nonexistent_tool")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_use_tool_success(self, tool_manager, mock_tool):
        """Test successful tool execution."""
        tool_manager.tools = [mock_tool]

        tool_call = {
            "function": {"name": "test_tool", "arguments": {"param1": "value1"}}
        }

        with patch.object(
            tool_manager, "_actually_call_tool", return_value="Success"
        ) as mock_call:
            result = await tool_manager.use_tool(tool_call)

            assert result == "Success"
            mock_call.assert_called_once_with(tool_call)
            tool_manager._mock_guard.record_use.assert_called_once_with("test_tool")

    @pytest.mark.asyncio
    async def test_use_tool_rate_limited(self, tool_manager):
        """Test tool execution when rate limited."""
        tool_call = {"function": {"name": "test_tool", "arguments": {}}}

        # Mock guard to deny usage
        tool_manager._mock_guard.can_use.return_value = False

        result = await tool_manager.use_tool(tool_call)

        assert "rate-limited" in result
        tool_manager._mock_guard.record_use.assert_not_called()

    @pytest.mark.asyncio
    async def test_use_tool_confirmation_required(self, tool_manager, mock_tool):
        """Test tool execution with confirmation required."""
        tool_manager.tools = [mock_tool]

        tool_call = {
            "function": {"name": "test_tool", "arguments": {"param1": "value1"}}
        }

        # Mock guard to require confirmation
        tool_manager._mock_guard.needs_confirmation.return_value = True

        with patch.object(
            tool_manager, "_actually_call_tool", return_value="Success"
        ) as mock_call:
            result = await tool_manager.use_tool(tool_call)

            assert result == "Success"
            tool_manager._mock_confirmation.request_confirmation.assert_called_once()

    @pytest.mark.asyncio
    async def test_use_tool_confirmation_denied(self, tool_manager):
        """Test tool execution when confirmation is denied."""
        tool_call = {
            "function": {"name": "test_tool", "arguments": {"param1": "value1"}}
        }

        # Mock guard to require confirmation and confirmation to be denied
        tool_manager._mock_guard.needs_confirmation.return_value = True
        tool_manager._mock_confirmation.request_confirmation.return_value = (
            False,
            "User declined",
        )

        result = await tool_manager.use_tool(tool_call)

        assert "cancelled" in result
        assert "User declined" in result

    @pytest.mark.asyncio
    async def test_actually_call_tool_success(
        self, tool_manager, mock_tool, mock_context
    ):
        """Test successful tool execution through _actually_call_tool."""
        tool_manager.tools = [mock_tool]

        tool_call = {
            "function": {"name": "test_tool", "arguments": {"param1": "value1"}}
        }

        result = await tool_manager._actually_call_tool(tool_call)

        assert result == "Tool executed successfully"

        # Verify events were emitted
        mock_context.emit_event.assert_any_call(
            AgentStateEvent.TOOL_CALLED,
            {"tool_name": "test_tool", "parameters": {"param1": "value1"}},
        )

    @pytest.mark.asyncio
    async def test_actually_call_tool_not_found(self, tool_manager, mock_context):
        """Test tool execution when tool is not found."""
        tool_call = {"function": {"name": "nonexistent_tool", "arguments": {}}}

        result = await tool_manager._actually_call_tool(tool_call)

        assert "not found" in result

        # Verify TOOL_FAILED event was emitted
        mock_context.emit_event.assert_any_call(
            AgentStateEvent.TOOL_FAILED,
            {"tool_name": "nonexistent_tool", "parameters": {}, "error": result},
        )

    @pytest.mark.asyncio
    async def test_actually_call_tool_cached_result(self, tool_manager, mock_tool):
        """Test tool execution with cached result."""
        tool_manager.tools = [mock_tool]

        # Mock cache to return cached result
        tool_manager._mock_cache.get.return_value = "Cached result"

        tool_call = {
            "function": {"name": "test_tool", "arguments": {"param1": "value1"}}
        }

        result = await tool_manager._actually_call_tool(tool_call)

        assert result == "Cached result"
        tool_manager._mock_executor.execute_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_actually_call_tool_with_confirmation(self, tool_manager, mock_tool):
        """Test tool execution with confirmation flow."""
        tool_manager.tools = [mock_tool]

        # Mock confirmation to be required
        tool_manager._mock_confirmation.tool_requires_confirmation.return_value = True

        tool_call = {
            "function": {"name": "test_tool", "arguments": {"param1": "value1"}}
        }

        result = await tool_manager._actually_call_tool(tool_call)

        assert result == "Tool executed successfully"
        tool_manager._mock_confirmation.request_confirmation.assert_called_once()

    @pytest.mark.asyncio
    async def test_actually_call_tool_confirmation_denied(
        self, tool_manager, mock_tool
    ):
        """Test tool execution when confirmation is denied."""
        tool_manager.tools = [mock_tool]

        # Mock confirmation to be required and denied
        tool_manager._mock_confirmation.tool_requires_confirmation.return_value = True
        tool_manager._mock_confirmation.request_confirmation.return_value = (
            False,
            "User feedback",
        )

        tool_call = {
            "function": {"name": "test_tool", "arguments": {"param1": "value1"}}
        }

        result = await tool_manager._actually_call_tool(tool_call)

        assert "cancelled" in result
        assert "User feedback" in result
        tool_manager._mock_confirmation.inject_user_feedback.assert_called_once()

    def test_get_tool_description(self, tool_manager, mock_tool):
        """Test getting tool description."""
        description = tool_manager._get_tool_description(mock_tool)
        assert description == "A test tool"

    def test_emit_tool_completion_events(self, tool_manager, mock_context):
        """Test emitting tool completion events."""
        tool_manager._emit_tool_completion_events(
            "test_tool", {"param1": "value1"}, "result", 0.1
        )

        # Verify TOOL_COMPLETED event was emitted
        mock_context.emit_event.assert_any_call(
            AgentStateEvent.TOOL_COMPLETED,
            {
                "tool_name": "test_tool",
                "parameters": {"param1": "value1"},
                "result": "result",
                "execution_time": 0.1,
            },
        )

    def test_emit_tool_completion_events_final_answer(self, tool_manager, mock_context):
        """Test emitting events for final_answer tool."""
        tool_manager._emit_tool_completion_events(
            "final_answer", {"answer": "Final answer"}, "Final answer", 0.1
        )

        # Verify FINAL_ANSWER_SET event was emitted
        mock_context.emit_event.assert_any_call(
            AgentStateEvent.FINAL_ANSWER_SET,
            {
                "tool_name": "final_answer",
                "answer": "Final answer",
                "parameters": {"answer": "Final answer"},
            },
        )

    @pytest.mark.asyncio
    async def test_generate_and_log_summary(self, tool_manager):
        """Test generating tool summary."""
        summary = await tool_manager._generate_and_log_summary(
            "test_tool", {"param1": "value1"}, "result"
        )

        assert summary == "Tool summary"
        tool_manager._mock_executor._generate_tool_summary.assert_called_once_with(
            "test_tool", {"param1": "value1"}, "result"
        )

    def test_add_to_history_success(self, tool_manager, mock_context):
        """Test adding successful tool execution to history."""
        tool_manager._add_to_history(
            "test_tool", {"param1": "value1"}, "result", "summary", 0.1
        )

        assert len(tool_manager.tool_history) == 1
        entry = tool_manager.tool_history[0]
        assert entry["name"] == "test_tool"
        assert entry["params"] == {"param1": "value1"}
        assert entry["result"] == "result"
        assert entry["summary"] == "summary"
        assert entry["execution_time"] == 0.1
        assert entry["cached"] is False
        assert entry["cancelled"] is False
        assert entry["error"] is False

        # Verify tool was added to successful tools
        assert "test_tool" in mock_context.session.successful_tools

    def test_add_to_history_error(self, tool_manager, mock_context):
        """Test adding failed tool execution to history."""
        tool_manager._add_to_history("test_tool", {}, "Error occurred", error=True)

        assert len(tool_manager.tool_history) == 1
        entry = tool_manager.tool_history[0]
        assert entry["error"] is True

        # Verify tool was NOT added to successful tools
        assert "test_tool" not in mock_context.session.successful_tools

    def test_add_to_history_with_metrics(self, tool_manager, mock_context):
        """Test adding to history updates metrics."""
        tool_manager._add_to_history("test_tool", {}, "result", execution_time=0.1)

        # Verify metrics were updated
        mock_context.metrics_manager.update_tool_metrics.assert_called_once()
        mock_context.emit_event.assert_any_call(
            AgentStateEvent.METRICS_UPDATED, {"metrics": {}}
        )

    def test_generate_tool_signatures(self, tool_manager, mock_tool):
        """Test generating tool signatures."""
        tool_manager.tools = [mock_tool]

        tool_manager._generate_tool_signatures()

        assert len(tool_manager.tool_signatures) == 1
        assert mock_tool.tool_definition in tool_manager.tool_signatures

    def test_get_available_tools(self, tool_manager, mock_tool):
        """Test getting available tools."""
        tool_manager.tools = [mock_tool]

        tools = tool_manager.get_available_tools()
        assert tools == [mock_tool]

    def test_get_available_tool_names(self, tool_manager, mock_tool):
        """Test getting available tool names."""
        tool_manager.tools = [mock_tool]

        names = tool_manager.get_available_tool_names()
        assert names == {"test_tool"}

    def test_get_last_tool_action(self, tool_manager):
        """Test getting last tool action."""
        # Test with empty history
        assert tool_manager.get_last_tool_action() is None

        # Add entry to history
        tool_manager._add_to_history("test_tool", {}, "result")

        last_action = tool_manager.get_last_tool_action()
        assert last_action is not None
        assert last_action["name"] == "test_tool"

    def test_register_tools(self, tool_manager, mock_context, mock_tool):
        """Test registering tools."""
        # Add tool to context
        mock_context.tools = [mock_tool]

        with patch.object(tool_manager, "_generate_tool_signatures") as mock_gen:
            tool_manager.register_tools()

            assert mock_tool in tool_manager.tools
            mock_gen.assert_called_once()

    def test_register_tools_deduplication(self, tool_manager, mock_context):
        """Test tool registration deduplicates by name."""
        # Create two tools with same name
        tool1 = Mock()
        tool1.name = "duplicate_tool"
        tool2 = Mock()
        tool2.name = "duplicate_tool"

        tool_manager.tools = [tool1]
        mock_context.tools = [tool2]

        with patch.object(tool_manager, "_generate_tool_signatures"):
            tool_manager.register_tools()

            # Should only have one tool with that name
            names = [t.name for t in tool_manager.tools]
            assert names.count("duplicate_tool") == 1

    @pytest.mark.asyncio
    async def test_use_tool_invalid_format(self, tool_manager):
        """Test tool execution with invalid tool call format."""
        invalid_tool_call = {"invalid": "format"}

        result = await tool_manager.use_tool(invalid_tool_call)

        assert result.startswith("Error:")

    def test_load_plugin_tools_success(self, tool_manager):
        """Test loading tools from plugin system."""
        mock_plugin_manager = Mock()
        mock_tool_plugin = Mock()
        mock_plugin_tool = Mock()
        mock_plugin_tool.name = "plugin_tool"
        mock_plugin_tool.use = Mock()
        mock_tool_plugin.get_tools.return_value = {"plugin_tool": mock_plugin_tool}
        mock_plugin_manager.get_plugins_by_type.return_value = {
            "test_plugin": mock_tool_plugin
        }

        with patch(
            "reactive_agents.plugins.plugin_manager.get_plugin_manager",
            return_value=mock_plugin_manager,
        ), patch(
            "reactive_agents.plugins.plugin_manager.PluginType"
        ) as mock_plugin_type, patch(
            "reactive_agents.plugins.plugin_manager.ToolPlugin",
            mock_tool_plugin.__class__,
        ):

            mock_plugin_type.TOOL = "tool"  # Set the enum value

            tool_manager._load_plugin_tools()

            # Verify plugin tools were loaded
            plugin_tool_names = [
                t.name for t in tool_manager.tools if hasattr(t, "name")
            ]
            assert "plugin_tool" in plugin_tool_names

    def test_load_plugin_tools_import_error(self, tool_manager):
        """Test loading plugin tools when plugin system is not available."""
        with patch(
            "reactive_agents.plugins.plugin_manager.get_plugin_manager",
            side_effect=ImportError,
        ):
            # Should not raise exception
            tool_manager._load_plugin_tools()

            # Should log debug message about plugin system not being available
            if tool_manager.tool_logger:
                tool_manager.tool_logger.debug.assert_called()

    @pytest.mark.asyncio
    async def test_tool_execution_flow_integration(
        self, tool_manager, mock_tool, mock_context
    ):
        """Test complete tool execution flow integration."""
        tool_manager.tools = [mock_tool]

        tool_call = {
            "function": {"name": "test_tool", "arguments": {"param1": "value1"}}
        }

        # Execute tool
        result = await tool_manager.use_tool(tool_call)

        # Verify complete flow
        assert result == "Tool executed successfully"

        # Verify all components were called
        tool_manager._mock_guard.can_use.assert_called_with("test_tool")
        tool_manager._mock_executor.parse_tool_arguments.assert_called()
        tool_manager._mock_executor.execute_tool.assert_called()
        tool_manager._mock_cache.generate_cache_key.assert_called()
        tool_manager._mock_validator.validate_tool_result_usage.assert_called()

        # Verify history was updated
        assert len(tool_manager.tool_history) == 1

        # Verify events were emitted
        mock_context.emit_event.assert_called()
