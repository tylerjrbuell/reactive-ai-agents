"""
Tests for MemoryManager.

Tests the memory management functionality including memory storage,
retrieval, and lifecycle management.
"""

import pytest
import json
import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, mock_open
from reactive_agents.core.memory.memory_manager import MemoryManager
from reactive_agents.core.types.memory_types import AgentMemory
from reactive_agents.core.types.status_types import TaskStatus
from reactive_agents.core.types.session_types import AgentSession
from reactive_agents.core.context.agent_context import AgentContext


class TestMemoryManager:
    """Test cases for MemoryManager."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock agent context that satisfies Pydantic validation."""
        # Create a simpler mock that the MemoryManager will accept
        from unittest.mock import Mock, create_autospec
        from reactive_agents.core.context.agent_context import AgentContext
        
        # Create autospec mock that Pydantic will accept
        context = create_autospec(AgentContext, instance=True)
        context.agent_name = "TestAgent"
        context.use_memory_enabled = True
        context.reflection_manager = None
        
        # Mock logger
        mock_logger = Mock()
        mock_logger.debug = Mock()
        mock_logger.info = Mock()
        mock_logger.warning = Mock()
        mock_logger.error = Mock()
        context.agent_logger = mock_logger
        
        return context

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.get_memory_path.return_value = "/tmp/test_memory"
        return settings

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_initialization_with_memory_enabled(self, mock_context, mock_settings, temp_dir):
        """Test memory manager initialization with memory enabled."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            
            assert memory_manager.context == mock_context
            assert memory_manager.memory_enabled is True
            assert memory_manager.agent_memory is not None
            assert memory_manager.agent_memory.agent_name == "TestAgent"
            assert memory_manager.memory_file_path is not None

    def test_initialization_with_memory_disabled(self, mock_context):
        """Test memory manager initialization with memory disabled."""
        mock_context.use_memory_enabled = False
        
        memory_manager = MemoryManager(context=mock_context)
        
        assert memory_manager.memory_enabled is False
        assert memory_manager.agent_memory is None

    def test_initialization_loads_existing_memory(self, mock_context, mock_settings, temp_dir):
        """Test loading existing memory from file."""
        existing_memory = {
            "agent_name": "TestAgent",
            "session_history": [{"test": "data"}],
            "tool_preferences": {"test_tool": {"success_count": 5}},
            "reflections": []
        }
        
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(existing_memory))):
            
            memory_manager = MemoryManager(context=mock_context)
            assert memory_manager.agent_memory is not None
            assert memory_manager.agent_memory.agent_name == "TestAgent"
            assert len(memory_manager.agent_memory.session_history) == 1
            assert "test_tool" in memory_manager.agent_memory.tool_preferences

    def test_initialization_handles_corrupt_memory_file(self, mock_context, mock_settings):
        """Test handling corrupt memory file."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="invalid json")):
            
            memory_manager = MemoryManager(context=mock_context)
            
            # Should create new memory when file is corrupt
            assert memory_manager.agent_memory is not None
            assert memory_manager.agent_memory.agent_name == "TestAgent"
            assert len(memory_manager.agent_memory.session_history) == 0
            mock_context.agent_logger.warning.assert_called()

    def test_initialization_handles_agent_name_mismatch(self, mock_context, mock_settings):
        """Test handling agent name mismatch in memory file."""
        existing_memory = {
            "agent_name": "DifferentAgent",
            "session_history": [],
            "tool_preferences": {},
            "reflections": []
        }
        
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(existing_memory))):
            
            memory_manager = MemoryManager(context=mock_context)
            
            # Should create new memory when agent name doesn't match
            assert memory_manager.agent_memory is not None
            assert memory_manager.agent_memory.agent_name == "TestAgent"
            mock_context.agent_logger.warning.assert_called()

    def test_save_memory_success(self, mock_context, mock_settings):
        """Test successful memory saving."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            
            with patch('builtins.open', mock_open()) as mock_file:
                memory_manager.save_memory()
                
                mock_file.assert_called_once()
                mock_context.agent_logger.debug.assert_called()

    def test_save_memory_disabled(self, mock_context):
        """Test memory saving when disabled."""
        mock_context.use_memory_enabled = False
        memory_manager = MemoryManager(context=mock_context)
        
        memory_manager.save_memory()
        
        # Should not attempt to save when disabled
        mock_context.agent_logger.debug.assert_called_with(
            "Memory saving skipped (disabled, no path, or no memory object)."
        )

    def test_save_memory_handles_io_error(self, mock_context, mock_settings):
        """Test memory saving handles IO errors."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            
            with patch('builtins.open', side_effect=IOError("Disk full")):
                memory_manager.save_memory()
                
                mock_context.agent_logger.error.assert_called()

    def test_update_session_history(self, mock_context, mock_settings):
        """Test updating session history."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            
            # Create a mock session
            session = Mock(spec=AgentSession)
            session.session_id = "test-session-123"
            session.task_status = TaskStatus.COMPLETE
            session.iterations = 5
            session.evaluation = {"adherence_score": 0.9}
            session.initial_task = "Test task"
            session.current_task = "Test task"
            session.final_answer = "Test answer"
            session.successful_tools = ["tool1", "tool2"]
            
            memory_manager.update_session_history(session)
            
            # Verify session was added to history
            assert memory_manager.agent_memory is not None
            assert len(memory_manager.agent_memory.session_history) == 1
            history_entry = memory_manager.agent_memory.session_history[0]
            assert history_entry["session_id"] == "test-session-123"
            assert history_entry["task"] == "Test task"
            assert history_entry["success"] is True
            assert history_entry["iterations"] == 5
            assert set(history_entry["tools_used"]) == {"tool1", "tool2"}

    def test_update_session_history_limits_size(self, mock_context, mock_settings):
        """Test session history size limiting."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            
            # Add more than max history entries
            for i in range(25):
                session = Mock(spec=AgentSession)
                session.session_id = f"session-{i}"
                session.task_status = TaskStatus.COMPLETE
                session.iterations = 1
                session.evaluation = None
                session.initial_task = f"Task {i}"
                session.current_task = f"Task {i}"
                session.final_answer = f"Answer {i}"
                session.successful_tools = []
                
                memory_manager.update_session_history(session)
            
            # Should limit to max history (20)
            assert memory_manager.agent_memory is not None
            assert len(memory_manager.agent_memory.session_history) == 20
            # Should keep the most recent entries
            assert memory_manager.agent_memory.session_history[-1]["session_id"] == "session-24"

    def test_update_tool_preferences_new_tool(self, mock_context, mock_settings):
        """Test updating preferences for a new tool."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            assert memory_manager.agent_memory is not None
            memory_manager.update_tool_preferences("new_tool", success=True, feedback="Great tool!")
            
            # Verify tool preferences were created
            prefs = memory_manager.agent_memory.tool_preferences["new_tool"]
            assert prefs["success_count"] == 1
            assert prefs["failure_count"] == 0
            assert prefs["total_usage"] == 1
            assert prefs["success_rate"] == 1.0
            assert len(prefs["feedback"]) == 1
            assert prefs["feedback"][0]["message"] == "Great tool!"

    def test_update_tool_preferences_existing_tool(self, mock_context, mock_settings):
        """Test updating preferences for an existing tool."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            assert memory_manager.agent_memory is not None
            
            # Initialize tool preferences
            memory_manager.agent_memory.tool_preferences["existing_tool"] = {
                "success_count": 2,
                "failure_count": 1,
                "total_usage": 3,
                "feedback": [],
                "success_rate": 0.67
            }
            
            memory_manager.update_tool_preferences("existing_tool", success=False)
            
            # Verify preferences were updated
            prefs = memory_manager.agent_memory.tool_preferences["existing_tool"]
            assert prefs["success_count"] == 2
            assert prefs["failure_count"] == 2
            assert prefs["total_usage"] == 4
            assert prefs["success_rate"] == 0.5

    def test_update_tool_preferences_limits_feedback(self, mock_context, mock_settings):
        """Test that feedback history is limited."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            assert memory_manager.agent_memory is not None
            # Add more than max feedback entries
            for i in range(8):
                memory_manager.update_tool_preferences("test_tool", success=True, feedback=f"Feedback {i}")
            
            # Should limit to max feedback (5)
            prefs = memory_manager.agent_memory.tool_preferences["test_tool"]
            assert len(prefs["feedback"]) == 5
            # Should keep the most recent feedback
            assert prefs["feedback"][-1]["message"] == "Feedback 7"

    def test_get_reflections(self, mock_context, mock_settings):
        """Test getting reflections."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            assert memory_manager.agent_memory is not None
            # Add some reflections
            test_reflections = [
                {"id": 1, "content": "Reflection 1"},
                {"id": 2, "content": "Reflection 2"}
            ]
            memory_manager.agent_memory.reflections = test_reflections
            
            reflections = memory_manager.get_reflections()
            assert reflections == test_reflections

    def test_get_reflections_no_memory(self, mock_context):
        """Test getting reflections when memory is disabled."""
        mock_context.use_memory_enabled = False
        memory_manager = MemoryManager(context=mock_context)

        reflections = memory_manager.get_reflections()
        assert reflections == []

    def test_get_tool_preferences(self, mock_context, mock_settings):
        """Test getting tool preferences."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            assert memory_manager.agent_memory is not None
            # Add some tool preferences
            test_prefs = {
                "tool1": {"success_rate": 0.8},
                "tool2": {"success_rate": 0.6}
            }
            memory_manager.agent_memory.tool_preferences = test_prefs
            
            prefs = memory_manager.get_tool_preferences()
            assert prefs == test_prefs

    def test_get_session_history(self, mock_context, mock_settings):
        """Test getting session history."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            assert memory_manager.agent_memory is not None
            # Add some session history
            test_history = [
                {"session_id": "1", "task": "Task 1"},
                {"session_id": "2", "task": "Task 2"}
            ]
            memory_manager.agent_memory.session_history = test_history
            
            history = memory_manager.get_session_history()
            assert history == test_history

    def test_memory_sync_with_reflection_manager(self, mock_context, mock_settings):
        """Test memory syncing with reflection manager."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            
            # Mock reflection manager
            mock_reflection_manager = Mock()
            mock_reflection_manager.reflections = [{"reflection": "test"}]
            mock_context.reflection_manager = mock_reflection_manager
            
            with patch('builtins.open', mock_open()):
                memory_manager.save_memory()
            
            # Verify reflections were synced from reflection manager
            assert memory_manager.agent_memory is not None
            assert memory_manager.agent_memory.reflections == [{"reflection": "test"}]

    def test_agent_logger_property(self, mock_context, mock_settings):
        """Test agent logger property."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            
            assert memory_manager.agent_logger == mock_context.agent_logger

    def test_error_handling_in_update_session_history(self, mock_context, mock_settings):
        """Test error handling in update_session_history."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            
            # Pass invalid session data
            memory_manager.update_session_history(Mock(spec=AgentSession))
            
            # Should handle error gracefully
            mock_context.agent_logger.error.assert_called()

    def test_error_handling_in_update_tool_preferences(self, mock_context, mock_settings):
        """Test error handling in update_tool_preferences."""
        with patch('reactive_agents.config.settings.get_settings', return_value=mock_settings), \
             patch('os.makedirs'), \
             patch('os.path.exists', return_value=False):
            
            memory_manager = MemoryManager(context=mock_context)
            
            # Simulate error by making agent_memory None
            memory_manager.agent_memory = None
            memory_manager.memory_enabled = True
            
            memory_manager.update_tool_preferences("test_tool", success=True)
            
            # Should not raise exception when memory is None
