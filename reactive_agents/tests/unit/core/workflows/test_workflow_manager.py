"""
Tests for WorkflowManager.

Tests the workflow management functionality including dependency checking,
context updates, and workflow state management.
"""

import pytest
import json
from unittest.mock import Mock, patch, create_autospec
from datetime import datetime
from reactive_agents.core.workflows.workflow_manager import WorkflowManager
from reactive_agents.core.context.agent_context import AgentContext
from reactive_agents.core.types.status_types import TaskStatus


class TestWorkflowManager:
    """Test cases for WorkflowManager."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock agent context."""
        context = create_autospec(AgentContext, instance=True)
        context.agent_name = "TestAgent"

        # Mock session
        context.session = Mock()
        context.session.iterations = 1
        context.session.task_status = TaskStatus.INITIALIZED

        # Mock logger
        context.agent_logger = Mock()
        context.agent_logger.debug = Mock()
        context.agent_logger.info = Mock()
        context.agent_logger.warning = Mock()
        context.agent_logger.error = Mock()

        return context

    @pytest.fixture
    def workflow_context(self):
        """Create a shared workflow context."""
        return {
            "TestAgent": {
                "status": str(TaskStatus.INITIALIZED),
                "current_progress": "",
                "iterations": 0,
                "dependencies_met": True,
                "reflections": [],
                "last_updated": "",
            },
            "DependencyAgent": {
                "status": str(TaskStatus.COMPLETE),
                "current_progress": "Task completed",
                "iterations": 3,
                "dependencies_met": True,
                "reflections": [],
                "last_updated": "2024-01-01T12:00:00",
            },
        }

    def test_initialization_without_workflow_context(self, mock_context):
        """Test workflow manager initialization without workflow context."""
        manager = WorkflowManager(context=mock_context)

        assert manager.context == mock_context
        assert manager.workflow_context is None
        assert manager.workflow_dependencies == []

    def test_initialization_with_workflow_context(self, mock_context, workflow_context):
        """Test workflow manager initialization with workflow context."""
        manager = WorkflowManager(
            context=mock_context,
            workflow_context=workflow_context,
            workflow_dependencies=["DependencyAgent"],
        )

        assert manager.context == mock_context
        assert manager.workflow_context == workflow_context
        assert manager.workflow_dependencies == ["DependencyAgent"]

    def test_agent_logger_property(self, mock_context):
        """Test agent logger property."""
        manager = WorkflowManager(context=mock_context)
        assert manager.agent_logger == mock_context.agent_logger

    def test_check_dependencies_no_dependencies(self, mock_context, workflow_context):
        """Test dependency checking with no dependencies."""
        manager = WorkflowManager(
            context=mock_context, workflow_context=workflow_context
        )

        result = manager.check_dependencies()
        assert result is True

    def test_check_dependencies_no_workflow_context(self, mock_context):
        """Test dependency checking without workflow context."""
        manager = WorkflowManager(
            context=mock_context, workflow_dependencies=["SomeAgent"]
        )

        result = manager.check_dependencies()
        assert result is True

    def test_check_dependencies_met(self, mock_context, workflow_context):
        """Test dependency checking when dependencies are met."""
        manager = WorkflowManager(
            context=mock_context,
            workflow_context=workflow_context,
            workflow_dependencies=["DependencyAgent"],
        )

        result = manager.check_dependencies()
        assert result is True

    def test_check_dependencies_not_met(self, mock_context, workflow_context):
        """Test dependency checking when dependencies are not met."""
        # Set dependency agent to incomplete status
        workflow_context["DependencyAgent"]["status"] = str(TaskStatus.RUNNING)

        manager = WorkflowManager(
            context=mock_context,
            workflow_context=workflow_context,
            workflow_dependencies=["DependencyAgent"],
        )

        result = manager.check_dependencies()
        assert result is False
        mock_context.agent_logger.info.assert_called()

    def test_check_dependencies_updates_status_to_waiting(
        self, mock_context, workflow_context
    ):
        """Test that unmet dependencies update status to waiting."""
        workflow_context["DependencyAgent"]["status"] = str(TaskStatus.RUNNING)

        manager = WorkflowManager(
            context=mock_context,
            workflow_context=workflow_context,
            workflow_dependencies=["DependencyAgent"],
        )

        # Mock the update_context method at the class level to avoid Pydantic conflicts
        with patch(
            "reactive_agents.core.workflows.workflow_manager.WorkflowManager.update_context"
        ) as mock_update:
            result = manager.check_dependencies()
            assert result is False
            mock_update.assert_called_with(TaskStatus.WAITING_DEPENDENCIES)

    def test_check_dependencies_resets_status_when_met(
        self, mock_context, workflow_context
    ):
        """Test that met dependencies reset status from waiting."""
        mock_context.session.task_status = TaskStatus.WAITING_DEPENDENCIES

        manager = WorkflowManager(
            context=mock_context,
            workflow_context=workflow_context,
            workflow_dependencies=["DependencyAgent"],
        )

        with patch(
            "reactive_agents.core.workflows.workflow_manager.WorkflowManager.update_context"
        ) as mock_update:
            result = manager.check_dependencies()
            assert result is True
            mock_update.assert_called_with(TaskStatus.INITIALIZED)

    def test_update_context_basic_update(self, mock_context, workflow_context):
        """Test basic context update."""
        manager = WorkflowManager(
            context=mock_context, workflow_context=workflow_context
        )

        with patch(
            "reactive_agents.core.workflows.workflow_manager.WorkflowManager._check_dependencies_internal",
            return_value=True,
        ), patch(
            "reactive_agents.core.workflows.workflow_manager.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2024-01-01T12:00:00"
            )
            manager.update_context(TaskStatus.RUNNING, "Making progress")

        agent_context = workflow_context["TestAgent"]
        assert agent_context["status"] == str(TaskStatus.RUNNING)
        assert agent_context["current_progress"] == "Making progress"
        assert agent_context["iterations"] == 1
        assert agent_context["last_updated"] == "2024-01-01T12:00:00"

    def test_update_context_without_workflow_context(self, mock_context):
        """Test context update without workflow context."""
        manager = WorkflowManager(context=mock_context)

        manager.update_context(TaskStatus.RUNNING, "Progress")

        # Should not raise exception and should log debug message
        mock_context.agent_logger.debug.assert_called_with(
            "No workflow context provided, skipping update."
        )

    def test_update_context_agent_not_found(self, mock_context, workflow_context):
        """Test context update when agent not found in context."""
        mock_context.agent_name = "NonExistentAgent"

        manager = WorkflowManager(
            context=mock_context, workflow_context=workflow_context
        )

        with patch(
            "reactive_agents.core.workflows.workflow_manager.WorkflowManager._check_dependencies_internal",
            return_value=True,
        ):
            manager.update_context(TaskStatus.RUNNING)

        # The agent entry should be created automatically by _ensure_workflow_context_entry
        # Check the manager's workflow_context since Pydantic creates a copy
        assert manager.workflow_context is not None
        assert "NonExistentAgent" in manager.workflow_context
        # No warning should be called since the entry is created automatically

    def test_update_context_complete_status(self, mock_context, workflow_context):
        """Test context update with complete status."""
        manager = WorkflowManager(
            context=mock_context, workflow_context=workflow_context
        )

        with patch(
            "reactive_agents.core.workflows.workflow_manager.WorkflowManager._check_dependencies_internal",
            return_value=True,
        ):
            manager.update_context(TaskStatus.COMPLETE, "Task completed successfully")

        agent_context = workflow_context["TestAgent"]
        assert agent_context["status"] == str(TaskStatus.COMPLETE)
        assert agent_context["final_result"] == "Task completed successfully"
        assert (
            "current_progress" not in agent_context
            or agent_context.get("current_progress") == ""
        )

    def test_update_context_error_status(self, mock_context, workflow_context):
        """Test context update with error status."""
        manager = WorkflowManager(
            context=mock_context, workflow_context=workflow_context
        )

        with patch(
            "reactive_agents.core.workflows.workflow_manager.WorkflowManager._check_dependencies_internal",
            return_value=True,
        ):
            manager.update_context(TaskStatus.ERROR, error="Test error")

        agent_context = workflow_context["TestAgent"]
        assert agent_context["status"] == str(TaskStatus.ERROR)
        assert "error" in agent_context
        assert "Test error" in agent_context["error"]

    def test_update_context_missing_tools_status(self, mock_context, workflow_context):
        """Test context update with missing tools status."""
        manager = WorkflowManager(
            context=mock_context, workflow_context=workflow_context
        )

        missing_tools = ["tool1", "tool2"]
        with patch(
            "reactive_agents.core.workflows.workflow_manager.WorkflowManager._check_dependencies_internal",
            return_value=True,
        ):
            manager.update_context(
                TaskStatus.MISSING_TOOLS, missing_tools=missing_tools
            )

        agent_context = workflow_context["TestAgent"]
        assert agent_context["status"] == str(TaskStatus.MISSING_TOOLS)
        assert agent_context["missing_tools"] == missing_tools

    def test_update_context_with_optional_data(self, mock_context, workflow_context):
        """Test context update with optional data."""
        manager = WorkflowManager(
            context=mock_context, workflow_context=workflow_context
        )

        with patch(
            "reactive_agents.core.workflows.workflow_manager.WorkflowManager._check_dependencies_internal",
            return_value=True,
        ):
            manager.update_context(
                TaskStatus.RUNNING,
                "Progress",
                completion_score=0.8,
                plan="Test plan",
                reflection_data={"insight": "Test insight"},
            )

        agent_context = workflow_context["TestAgent"]
        assert agent_context["completion_score"] == 0.8
        assert agent_context["last_action_plan"] == "Test plan"
        assert agent_context["last_reflection"] == {"insight": "Test insight"}

    def test_update_context_truncates_long_results(
        self, mock_context, workflow_context
    ):
        """Test that long results are truncated."""
        manager = WorkflowManager(
            context=mock_context, workflow_context=workflow_context
        )

        long_result = "x" * 3000  # Longer than 2000 chars
        with patch(
            "reactive_agents.core.workflows.workflow_manager.WorkflowManager._check_dependencies_internal",
            return_value=True,
        ):
            manager.update_context(TaskStatus.COMPLETE, long_result)

        agent_context = workflow_context["TestAgent"]
        assert len(agent_context["final_result"]) <= 2003  # 2000 + "..."
        assert agent_context["final_result"].endswith("...")

    def test_update_context_handles_exceptions(self, mock_context, workflow_context):
        """Test that context update handles exceptions gracefully."""
        manager = WorkflowManager(
            context=mock_context, workflow_context=workflow_context
        )

        # Force an exception by making _check_dependencies_internal raise
        with patch(
            "reactive_agents.core.workflows.workflow_manager.WorkflowManager._check_dependencies_internal",
            side_effect=Exception("Test error"),
        ):
            manager.update_context(TaskStatus.RUNNING, "Progress")

        # Should handle exception gracefully and log error
        mock_context.agent_logger.error.assert_called()

    def test_get_agent_context(self, mock_context, workflow_context):
        """Test getting agent context."""
        manager = WorkflowManager(
            context=mock_context, workflow_context=workflow_context
        )

        context = manager.get_agent_context("DependencyAgent")
        assert context == workflow_context["DependencyAgent"]

        # Test non-existent agent
        context = manager.get_agent_context("NonExistentAgent")
        assert context is None

    def test_get_agent_context_no_workflow_context(self, mock_context):
        """Test getting agent context without workflow context."""
        manager = WorkflowManager(context=mock_context)

        context = manager.get_agent_context("SomeAgent")
        assert context is None

    def test_get_full_context(self, mock_context, workflow_context):
        """Test getting full workflow context."""
        manager = WorkflowManager(
            context=mock_context, workflow_context=workflow_context
        )

        full_context = manager.get_full_context()
        assert full_context == workflow_context

    def test_get_full_context_none(self, mock_context):
        """Test getting full context when none exists."""
        manager = WorkflowManager(context=mock_context)

        full_context = manager.get_full_context()
        assert full_context is None

    def test_ensure_workflow_context_entry_creates_entry(self, mock_context):
        """Test that _ensure_workflow_context_entry creates missing entry."""
        workflow_context = {}
        manager = WorkflowManager(
            context=mock_context, workflow_context=workflow_context
        )

        manager._ensure_workflow_context_entry()

        # Check the manager's workflow_context since Pydantic creates a copy
        assert manager.workflow_context is not None
        assert "TestAgent" in manager.workflow_context
        entry = manager.workflow_context["TestAgent"]
        assert entry["status"] == str(TaskStatus.INITIALIZED)
        assert entry["current_progress"] == ""
        assert entry["iterations"] == 0
        assert entry["dependencies_met"] is True
        assert entry["reflections"] == []
        assert entry["last_updated"] == ""
        mock_context.agent_logger.debug.assert_called()

    def test_ensure_workflow_context_entry_preserves_existing(
        self, mock_context, workflow_context
    ):
        """Test that _ensure_workflow_context_entry preserves existing entries."""
        original_entry = workflow_context["TestAgent"].copy()

        manager = WorkflowManager(
            context=mock_context, workflow_context=workflow_context
        )

        manager._ensure_workflow_context_entry()

        # Entry should remain unchanged
        assert workflow_context["TestAgent"] == original_entry

    def test_ensure_workflow_context_entry_no_context(self, mock_context):
        """Test _ensure_workflow_context_entry with no workflow context."""
        manager = WorkflowManager(context=mock_context)

        # Should not raise exception
        manager._ensure_workflow_context_entry()

    def test_ensure_workflow_context_entry_no_logger(self, mock_context):
        """Test _ensure_workflow_context_entry without logger."""
        mock_context.agent_logger = None
        workflow_context = {}  # Create empty context

        manager = WorkflowManager(
            context=mock_context, workflow_context=workflow_context
        )

        # Should not raise exception even without logger
        manager._ensure_workflow_context_entry()

        # Check the manager's workflow_context since Pydantic creates a copy
        assert manager.workflow_context is not None
        assert "TestAgent" in manager.workflow_context

    def test_workflow_manager_pydantic_validation(self, mock_context):
        """Test that WorkflowManager properly validates Pydantic fields."""
        # Test with valid context
        manager = WorkflowManager(context=mock_context)
        assert manager.context == mock_context

        # Test field exclusion for context
        model_dict = manager.model_dump()
        assert "context" not in model_dict  # Should be excluded

    def test_dependency_checking_with_rescoped_complete(
        self, mock_context, workflow_context
    ):
        """Test dependency checking recognizes RESCOPED_COMPLETE as met."""
        workflow_context["DependencyAgent"]["status"] = str(
            TaskStatus.RESCOPED_COMPLETE
        )

        manager = WorkflowManager(
            context=mock_context,
            workflow_context=workflow_context,
            workflow_dependencies=["DependencyAgent"],
        )

        result = manager.check_dependencies()
        assert result is True

    def test_multiple_dependencies_mixed_status(self, mock_context, workflow_context):
        """Test dependency checking with multiple dependencies having mixed status."""
        workflow_context["Agent2"] = {
            "status": str(TaskStatus.RUNNING),
            "current_progress": "In progress",
            "iterations": 2,
            "dependencies_met": True,
            "reflections": [],
            "last_updated": "2024-01-01T12:30:00",
        }

        manager = WorkflowManager(
            context=mock_context,
            workflow_context=workflow_context,
            workflow_dependencies=[
                "DependencyAgent",
                "Agent2",
            ],  # One complete, one running
        )

        result = manager.check_dependencies()
        assert result is False  # Should be false because Agent2 is still running
        mock_context.agent_logger.info.assert_called()
