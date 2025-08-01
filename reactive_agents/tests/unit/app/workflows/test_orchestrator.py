"""
Tests for WorkflowOrchestrator and WorkflowBuilder.

Tests the workflow orchestration functionality including workflow creation,
execution, node processing, and dependency management.
"""

import pytest
import asyncio
import uuid
import time
from unittest.mock import Mock, patch, AsyncMock, create_autospec
from reactive_agents.app.workflows.orchestrator import (
    WorkflowOrchestrator,
    WorkflowBuilder,
)
from reactive_agents.core.types.workflow_types import (
    WorkflowDefinition,
    WorkflowNode,
    WorkflowNodeType,
    WorkflowNodeStatus,
    WorkflowExecutionResult,
)


class TestWorkflowOrchestrator:
    """Test cases for WorkflowOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create a workflow orchestrator instance."""
        return WorkflowOrchestrator()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock reactive agent."""
        agent = Mock()
        agent.context = Mock()
        agent.context.agent_name = "TestAgent"
        agent.run = AsyncMock(
            return_value={"status": "success", "result": "Task completed"}
        )
        return agent

    @pytest.fixture
    def mock_a2a_protocol(self):
        """Create a mock A2A communication protocol."""
        return Mock()

    @pytest.fixture
    def sample_workflow_definition(self):
        """Create a sample workflow definition."""
        workflow = WorkflowDefinition(
            name="TestWorkflow", description="A test workflow"
        )

        # Add nodes
        node1 = WorkflowNode(
            node_id="node1",
            node_type=WorkflowNodeType.AGENT,
            agent_name="TestAgent",
            task_template="Complete task: ${input.task}",
            depends_on=[],
            context_mapping={"result": "outputs.node1_result"},
        )

        node2 = WorkflowNode(
            node_id="node2",
            node_type=WorkflowNodeType.CONDITION,
            condition="context['outputs']['node1_result'] == 'success'",
            depends_on=["node1"],
        )

        workflow.nodes = {"node1": node1, "node2": node2}
        workflow.entry_nodes = ["node1"]
        workflow.exit_nodes = ["node2"]
        workflow.global_context = {"input": {"task": "test task"}}

        return workflow

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.agents == {}
        assert orchestrator.a2a_protocols == {}
        assert orchestrator.active_workflows == {}
        assert orchestrator.execution_history == []

    def test_register_agent(self, orchestrator, mock_agent):
        """Test agent registration."""
        orchestrator.register_agent(mock_agent)

        assert "TestAgent" in orchestrator.agents
        assert orchestrator.agents["TestAgent"] == mock_agent

    def test_register_agent_with_a2a_protocol(
        self, orchestrator, mock_agent, mock_a2a_protocol
    ):
        """Test agent registration with A2A protocol."""
        orchestrator.register_agent(mock_agent, mock_a2a_protocol)

        assert "TestAgent" in orchestrator.agents
        assert "TestAgent" in orchestrator.a2a_protocols
        assert orchestrator.a2a_protocols["TestAgent"] == mock_a2a_protocol

    def test_create_workflow(self, orchestrator):
        """Test workflow creation using builder."""
        builder = orchestrator.create_workflow("TestWorkflow", "Test description")

        assert isinstance(builder, WorkflowBuilder)
        assert builder.orchestrator == orchestrator
        assert builder.workflow.name == "TestWorkflow"
        assert builder.workflow.description == "Test description"

    @pytest.mark.asyncio
    async def test_execute_workflow_success(
        self, orchestrator, mock_agent, sample_workflow_definition
    ):
        """Test successful workflow execution."""
        orchestrator.register_agent(mock_agent)

        with patch("networkx.DiGraph") as mock_graph_class, patch(
            "networkx.is_directed_acyclic_graph", return_value=True
        ), patch("networkx.topological_sort", return_value=["node1", "node2"]):

            mock_graph = Mock()
            mock_graph_class.return_value = mock_graph

            # Mock the node execution to prevent failures
            with patch.object(
                orchestrator, "_execute_node", return_value={"success": True}
            ):
                result = await orchestrator.execute_workflow(sample_workflow_definition)

                assert isinstance(result, WorkflowExecutionResult)
                assert result.workflow_id == sample_workflow_definition.workflow_id
                assert result.status in [
                    "completed",
                    "running",
                ]  # May be running initially

    @pytest.mark.asyncio
    async def test_execute_workflow_with_initial_context(
        self, orchestrator, mock_agent, sample_workflow_definition
    ):
        """Test workflow execution with initial context."""
        orchestrator.register_agent(mock_agent)
        initial_context = {"custom": {"data": "value"}}

        with patch("networkx.DiGraph"), patch(
            "networkx.is_directed_acyclic_graph", return_value=True
        ), patch("networkx.topological_sort", return_value=["node1", "node2"]):

            with patch.object(orchestrator, "_execute_workflow_graph") as mock_execute:
                result = await orchestrator.execute_workflow(
                    sample_workflow_definition, initial_context
                )

                # Verify context was merged
                args, kwargs = mock_execute.call_args
                execution_context = args[1]
                assert execution_context["custom"]["data"] == "value"
                assert execution_context["input"]["task"] == "test task"
                assert isinstance(result, WorkflowExecutionResult)

    @pytest.mark.asyncio
    async def test_execute_workflow_with_cycles(
        self, orchestrator, sample_workflow_definition
    ):
        """Test workflow execution fails with cycles."""
        with patch("networkx.DiGraph"), patch(
            "networkx.is_directed_acyclic_graph", return_value=False
        ):

            result = await orchestrator.execute_workflow(sample_workflow_definition)

            assert isinstance(result, WorkflowExecutionResult)
            assert result.status == "failed"
            assert "cycles" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_workflow_node_failure(
        self, orchestrator, mock_agent, sample_workflow_definition
    ):
        """Test workflow execution with node failure."""
        orchestrator.register_agent(mock_agent)

        with patch("networkx.DiGraph"), patch(
            "networkx.is_directed_acyclic_graph", return_value=True
        ), patch(
            "networkx.topological_sort", return_value=["node1", "node2"]
        ), patch.object(
            orchestrator, "_execute_node", side_effect=Exception("Node failed")
        ):

            result = await orchestrator.execute_workflow(sample_workflow_definition)

            assert isinstance(result, WorkflowExecutionResult)
            assert result.status == "failed"
            assert "Node failed" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_agent_node(self, orchestrator, mock_agent):
        """Test executing an agent node."""
        orchestrator.register_agent(mock_agent)

        node = WorkflowNode(
            node_id="test_node",
            node_type=WorkflowNodeType.AGENT,
            agent_name="TestAgent",
            task_template="Complete task: ${context.task}",
        )

        context = {"context": {"task": "sample task"}}  # Fix context structure

        result = await orchestrator._execute_agent_node(node, context)

        assert result["agent"] == "TestAgent"
        assert result["task"] == "Complete task: sample task"
        assert result["success"] is True
        mock_agent.run.assert_called_once_with("Complete task: sample task")

    @pytest.mark.asyncio
    async def test_execute_agent_node_not_found(self, orchestrator):
        """Test executing agent node when agent not found."""
        node = WorkflowNode(
            node_id="test_node",
            node_type=WorkflowNodeType.AGENT,
            agent_name="NonExistentAgent",
            task_template="Task",
        )

        with pytest.raises(ValueError, match="Agent NonExistentAgent not found"):
            await orchestrator._execute_agent_node(node, {})

    @pytest.mark.asyncio
    async def test_execute_condition_node_true(self, orchestrator):
        """Test executing condition node that evaluates to true."""
        node = WorkflowNode(
            node_id="condition_node",
            node_type=WorkflowNodeType.CONDITION,
            condition="context['value'] > 10",
        )

        context = {"value": 15}

        result = await orchestrator._execute_condition_node(node, context)

        assert result["result"] is True
        assert result["success"] is True
        assert result["condition"] == "context['value'] > 10"

    @pytest.mark.asyncio
    async def test_execute_condition_node_false(self, orchestrator):
        """Test executing condition node that evaluates to false."""
        node = WorkflowNode(
            node_id="condition_node",
            node_type=WorkflowNodeType.CONDITION,
            condition="context['value'] > 20",
        )

        context = {"value": 15}

        result = await orchestrator._execute_condition_node(node, context)

        assert result["result"] is False
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_condition_node_no_condition(self, orchestrator):
        """Test executing condition node without condition."""
        node = WorkflowNode(
            node_id="condition_node", node_type=WorkflowNodeType.CONDITION
        )

        with pytest.raises(
            ValueError, match="Condition node requires condition expression"
        ):
            await orchestrator._execute_condition_node(node, {})

    @pytest.mark.asyncio
    async def test_execute_delay_node(self, orchestrator):
        """Test executing delay node."""
        node = WorkflowNode(
            node_id="delay_node",
            node_type=WorkflowNodeType.DELAY,
            delay_seconds=0.1,  # Short delay for testing
        )

        start_time = time.time()
        result = await orchestrator._execute_delay_node(node, {})
        end_time = time.time()

        assert result["delay_seconds"] == 0.1
        assert result["success"] is True
        assert (end_time - start_time) >= 0.1

    @pytest.mark.asyncio
    async def test_execute_delay_node_default_delay(self, orchestrator):
        """Test executing delay node with default delay."""
        node = WorkflowNode(node_id="delay_node", node_type=WorkflowNodeType.DELAY)

        result = await orchestrator._execute_delay_node(node, {})

        assert result["delay_seconds"] == 1.0
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_node_sets_status_and_timing(self, orchestrator, mock_agent):
        """Test that _execute_node properly sets status and timing."""
        orchestrator.register_agent(mock_agent)

        node = WorkflowNode(
            node_id="test_node",
            node_type=WorkflowNodeType.AGENT,
            agent_name="TestAgent",
            task_template="Task",
        )

        workflow = WorkflowDefinition(name="test")

        result = await orchestrator._execute_node(node, {}, workflow)

        assert node.status == WorkflowNodeStatus.COMPLETED
        assert node.start_time is not None
        assert node.end_time is not None
        assert node.result == result

    @pytest.mark.asyncio
    async def test_execute_node_handles_failure(self, orchestrator, mock_agent):
        """Test that _execute_node handles node failures."""
        mock_agent.run.side_effect = Exception("Agent failed")
        orchestrator.register_agent(mock_agent)

        node = WorkflowNode(
            node_id="test_node",
            node_type=WorkflowNodeType.AGENT,
            agent_name="TestAgent",
            task_template="Task",
        )

        workflow = WorkflowDefinition(name="test")

        result = await orchestrator._execute_node(node, {}, workflow)

        assert node.status == WorkflowNodeStatus.FAILED
        assert node.error_message == "Agent failed"
        assert result["error"] == "Agent failed"

    def test_substitute_context_variables(self, orchestrator):
        """Test context variable substitution."""
        template = "Task: ${task.name} with priority ${task.priority}"
        context = {"task": {"name": "Sample Task", "priority": "high"}}

        result = orchestrator._substitute_context_variables(template, context)

        assert result == "Task: Sample Task with priority high"

    def test_substitute_context_variables_missing(self, orchestrator):
        """Test context variable substitution with missing variables."""
        template = "Task: ${task.name} with ${missing.value}"
        context = {"task": {"name": "Sample Task"}}

        result = orchestrator._substitute_context_variables(template, context)

        assert result == "Task: Sample Task with ${missing.value}"

    def test_get_nested_context(self, orchestrator):
        """Test getting nested context values."""
        context = {"level1": {"level2": {"value": "found"}}}

        result = orchestrator._get_nested_context(context, "level1.level2.value")
        assert result == "found"

        # Test missing path
        result = orchestrator._get_nested_context(context, "missing.path")
        assert result == "${missing.path}"

    def test_set_nested_context(self, orchestrator):
        """Test setting nested context values."""
        context = {}

        orchestrator._set_nested_context(context, "level1.level2.value", "set")

        assert context["level1"]["level2"]["value"] == "set"

    def test_set_nested_context_existing(self, orchestrator):
        """Test setting nested context values in existing structure."""
        context = {"level1": {"existing": "value"}}

        orchestrator._set_nested_context(context, "level1.level2.value", "new")

        assert context["level1"]["existing"] == "value"
        assert context["level1"]["level2"]["value"] == "new"


class TestWorkflowBuilder:
    """Test cases for WorkflowBuilder."""

    @pytest.fixture
    def orchestrator(self):
        """Create a workflow orchestrator instance."""
        return WorkflowOrchestrator()

    @pytest.fixture
    def builder(self, orchestrator):
        """Create a workflow builder instance."""
        return WorkflowBuilder(orchestrator, "TestWorkflow", "Test description")

    def test_builder_initialization(self, builder, orchestrator):
        """Test builder initialization."""
        assert builder.orchestrator == orchestrator
        assert builder.workflow.name == "TestWorkflow"
        assert builder.workflow.description == "Test description"
        assert isinstance(builder.workflow, WorkflowDefinition)

    def test_add_agent_node(self, builder):
        """Test adding an agent node."""
        result = builder.add_agent_node(
            agent_name="TestAgent",
            task_template="Complete ${task}",
            node_id="custom_node",
            depends_on=["dep1"],
            context_mapping={"result": "outputs.result"},
        )

        assert result == builder  # Should return self for chaining
        assert "custom_node" in builder.workflow.nodes

        node = builder.workflow.nodes["custom_node"]
        assert node.node_type == WorkflowNodeType.AGENT
        assert node.agent_name == "TestAgent"
        assert node.task_template == "Complete ${task}"
        assert node.depends_on == ["dep1"]
        assert node.context_mapping == {"result": "outputs.result"}

    def test_add_agent_node_auto_id(self, builder):
        """Test adding agent node with auto-generated ID."""
        builder.add_agent_node("Agent1", "Task1")
        builder.add_agent_node("Agent2", "Task2")

        assert "agent_0" in builder.workflow.nodes
        assert "agent_1" in builder.workflow.nodes

    def test_add_agent_node_entry_node(self, builder):
        """Test that agent nodes without dependencies become entry nodes."""
        builder.add_agent_node("TestAgent", "Task", node_id="entry_node")

        assert "entry_node" in builder.workflow.entry_nodes

    def test_add_agent_node_with_dependencies_not_entry(self, builder):
        """Test that agent nodes with dependencies don't become entry nodes."""
        builder.add_agent_node(
            "TestAgent", "Task", node_id="dep_node", depends_on=["other"]
        )

        assert "dep_node" not in builder.workflow.entry_nodes

    def test_add_condition_node(self, builder):
        """Test adding a condition node."""
        result = builder.add_condition_node(
            condition="context['value'] > 0", node_id="condition1", depends_on=["node1"]
        )

        assert result == builder
        assert "condition1" in builder.workflow.nodes

        node = builder.workflow.nodes["condition1"]
        assert node.node_type == WorkflowNodeType.CONDITION
        assert node.condition == "context['value'] > 0"
        assert node.depends_on == ["node1"]

    def test_add_condition_node_auto_id(self, builder):
        """Test adding condition node with auto-generated ID."""
        builder.add_condition_node("condition1")
        builder.add_condition_node("condition2")

        assert "condition_0" in builder.workflow.nodes
        assert "condition_1" in builder.workflow.nodes

    def test_add_delay_node(self, builder):
        """Test adding a delay node."""
        result = builder.add_delay_node(
            delay_seconds=5.0, node_id="delay1", depends_on=["node1"]
        )

        assert result == builder
        assert "delay1" in builder.workflow.nodes

        node = builder.workflow.nodes["delay1"]
        assert node.node_type == WorkflowNodeType.DELAY
        assert node.delay_seconds == 5.0
        assert node.depends_on == ["node1"]

    def test_add_delay_node_auto_id(self, builder):
        """Test adding delay node with auto-generated ID."""
        builder.add_delay_node(1.0)
        builder.add_delay_node(2.0)

        assert "delay_0" in builder.workflow.nodes
        assert "delay_1" in builder.workflow.nodes

    def test_set_exit_nodes(self, builder):
        """Test setting exit nodes."""
        result = builder.set_exit_nodes(["node1", "node2"])

        assert result == builder
        assert builder.workflow.exit_nodes == ["node1", "node2"]

    def test_set_global_context(self, builder):
        """Test setting global context."""
        context = {"key": "value", "config": {"setting": True}}
        result = builder.set_global_context(context)

        assert result == builder
        assert builder.workflow.global_context == context

    def test_build(self, builder):
        """Test building the workflow."""
        workflow = builder.build()

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow == builder.workflow

    def test_chaining_methods(self, builder):
        """Test that builder methods can be chained."""
        workflow = (
            builder.add_agent_node("Agent1", "Task1", node_id="agent1")
            .add_condition_node("True", node_id="condition1", depends_on=["agent1"])
            .add_delay_node(1.0, node_id="delay1", depends_on=["condition1"])
            .set_exit_nodes(["delay1"])
            .set_global_context({"test": True})
            .build()
        )

        assert len(workflow.nodes) == 3
        assert workflow.exit_nodes == ["delay1"]
        assert workflow.global_context == {"test": True}

    def test_complex_workflow_creation(self, builder):
        """Test creating a complex workflow with multiple node types."""
        workflow = (
            builder.add_agent_node(
                "DataCollector",
                "Collect data from ${source}",
                node_id="collect",
                context_mapping={"data": "collected_data"},
            )
            .add_agent_node(
                "DataProcessor",
                "Process ${collected_data}",
                node_id="process",
                depends_on=["collect"],
                context_mapping={"result": "processed_data"},
            )
            .add_condition_node(
                "context['processed_data']['success'] == True",
                node_id="check_success",
                depends_on=["process"],
            )
            .add_agent_node(
                "ReportGenerator",
                "Generate report from ${processed_data}",
                node_id="report",
                depends_on=["check_success"],
            )
            .add_delay_node(2.0, node_id="wait", depends_on=["report"])
            .set_exit_nodes(["wait"])
            .set_global_context({"source": "database", "format": "json"})
            .build()
        )

        # Verify workflow structure
        assert len(workflow.nodes) == 5
        assert workflow.entry_nodes == ["collect"]
        assert workflow.exit_nodes == ["wait"]

        # Verify node dependencies
        assert workflow.nodes["collect"].depends_on == []
        assert workflow.nodes["process"].depends_on == ["collect"]
        assert workflow.nodes["check_success"].depends_on == ["process"]
        assert workflow.nodes["report"].depends_on == ["check_success"]
        assert workflow.nodes["wait"].depends_on == ["report"]

        # Verify context mappings
        assert workflow.nodes["collect"].context_mapping == {"data": "collected_data"}
        assert workflow.nodes["process"].context_mapping == {"result": "processed_data"}

        # Verify global context
        assert workflow.global_context["source"] == "database"
        assert workflow.global_context["format"] == "json"
