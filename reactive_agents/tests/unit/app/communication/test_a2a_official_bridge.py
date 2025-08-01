"""
Tests for A2AOfficialBridge and related A2A integration components.

Tests the official A2A protocol bridge, atomic task delegation,
agent adapters, and capability-based agent discovery.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, create_autospec
from reactive_agents.app.communication.a2a_official_bridge import (
    A2AOfficialBridge,
    A2ATaskStatus,
    A2AAtomicTask,
    A2AAgentCapability,
    A2ACompatibleAgent,
    ReactiveAgentA2AAdapter,
)


class TestA2AOfficialBridge:
    """Test cases for A2AOfficialBridge."""

    @pytest.fixture
    def bridge(self):
        """Create a A2A official bridge instance."""
        return A2AOfficialBridge()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock reactive agent."""
        agent = Mock()
        agent.context = Mock()
        agent.context.agent_name = "TestAgent"
        agent.context.tool_use_enabled = True
        agent.context.tools = ["text_tool", "data_tool"]  # Make tools iterable
        agent.agent_logger = Mock()
        agent.run = AsyncMock(
            return_value={"status": "success", "result": "Task completed"}
        )
        agent.get_available_strategies = Mock(
            return_value=["reactive", "plan_execute_reflect"]
        )
        agent.get_current_strategy = Mock(return_value="reactive")
        agent.get_reasoning_context = Mock(return_value={"iteration_count": 1})
        return agent

    @pytest.fixture
    def mock_agent_2(self):
        """Create a second mock reactive agent."""
        agent = Mock()
        agent.context = Mock()
        agent.context.agent_name = "AnalysisAgent"
        agent.context.tool_use_enabled = True
        agent.context.tools = ["analysis_tool", "data_tool"]  # Make tools iterable
        agent.agent_logger = Mock()
        agent.run = AsyncMock(
            return_value={"status": "success", "result": "Analysis completed"}
        )
        agent.get_available_strategies = Mock(
            return_value=["reactive", "plan_execute_reflect"]
        )
        agent.get_current_strategy = Mock(return_value="plan_execute_reflect")
        agent.get_reasoning_context = Mock(return_value={"iteration_count": 2})
        return agent

    def test_bridge_initialization(self, bridge):
        """Test bridge initialization."""
        assert bridge.registered_agents == {}
        assert bridge.active_tasks == {}
        assert bridge.task_queue.empty()
        assert bridge.completed_tasks == []

        # Verify A2A endpoints are configured
        expected_endpoints = [
            "task_delegation",
            "task_status",
            "agent_discovery",
            "health_check",
        ]
        for endpoint in expected_endpoints:
            assert endpoint in bridge.a2a_endpoints
            assert bridge.a2a_endpoints[endpoint].startswith("/a2a/")  # type: ignore

    def test_register_agent(self, bridge, mock_agent):
        """Test agent registration."""
        adapter = bridge.register_agent(mock_agent)

        assert isinstance(adapter, ReactiveAgentA2AAdapter)
        assert adapter.agent_id in bridge.registered_agents
        assert bridge.registered_agents[adapter.agent_id] == adapter  # type: ignore

    @pytest.mark.asyncio
    async def test_delegate_atomic_task_success(self, bridge, mock_agent):
        """Test successful atomic task delegation."""
        # Register agent
        adapter = bridge.register_agent(mock_agent)

        # Mock the adapter methods
        adapter.execute_atomic_task = AsyncMock(
            return_value={"status": "success", "result": "Task executed successfully"}
        )

        task = await bridge.delegate_atomic_task(
            task_description="Process user request",
            target_agent_id=adapter.agent_id,
            required_capabilities=["data_processing"],
            input_data={"user_id": "123", "request_type": "analysis"},
        )

        assert isinstance(task, A2AAtomicTask)
        assert task.description == "Process user request"
        assert task.status == A2ATaskStatus.COMPLETED
        assert task.assigned_agent_id == adapter.agent_id
        assert task.result == {
            "status": "success",
            "result": "Task executed successfully",
        }
        assert task.started_at is not None
        assert task.completed_at is not None

        # Task should be in completed tasks
        assert task in bridge.completed_tasks
        assert task.task_id not in bridge.active_tasks

    @pytest.mark.asyncio
    async def test_delegate_atomic_task_failure(self, bridge, mock_agent):
        """Test atomic task delegation with failure."""
        adapter = bridge.register_agent(mock_agent)
        adapter.execute_atomic_task = AsyncMock(
            side_effect=Exception("Task execution failed")
        )

        task = await bridge.delegate_atomic_task(
            task_description="Process data", target_agent_id=adapter.agent_id
        )

        assert task.status == A2ATaskStatus.FAILED
        assert task.error_message == "Task execution failed"
        assert task.completed_at is not None
        assert task in bridge.completed_tasks

    @pytest.mark.asyncio
    async def test_delegate_atomic_task_auto_assignment(self, bridge, mock_agent):
        """Test atomic task delegation with automatic agent assignment."""
        adapter = bridge.register_agent(mock_agent)
        adapter.can_handle_task = AsyncMock(return_value=True)
        adapter.capabilities = [
            A2AAgentCapability(name="data_processing", description="Process data"),
            A2AAgentCapability(name="analysis", description="Analyze results"),
        ]
        adapter.execute_atomic_task = AsyncMock(
            return_value={"status": "success", "result": "Auto-assigned task completed"}
        )

        # Don't specify target_agent_id
        task = await bridge.delegate_atomic_task(
            task_description="Analyze dataset",
            required_capabilities=["data_processing", "analysis"],
        )

        assert task.status == A2ATaskStatus.COMPLETED
        assert task.assigned_agent_id == adapter.agent_id
        assert task.result == {
            "status": "success",
            "result": "Auto-assigned task completed",
        }

    @pytest.mark.asyncio
    async def test_delegate_atomic_task_no_suitable_agent(self, bridge):
        """Test atomic task delegation when no suitable agent is found."""
        task = await bridge.delegate_atomic_task(
            task_description="Complex AI task",
            required_capabilities=["advanced_ai", "machine_learning"],
        )

        assert task.status == A2ATaskStatus.FAILED
        assert "No suitable agent found" in task.error_message
        assert task.assigned_agent_id is None

    @pytest.mark.asyncio
    async def test_find_best_agent_capability_matching(
        self, bridge, mock_agent, mock_agent_2
    ):
        """Test finding the best agent based on capability matching."""
        # Register agents with different capabilities
        adapter1 = bridge.register_agent(mock_agent)
        adapter1.can_handle_task = AsyncMock(return_value=True)
        adapter1.capabilities = [
            A2AAgentCapability(name="basic_processing", description="Basic processing")
        ]

        adapter2 = bridge.register_agent(mock_agent_2)
        adapter2.can_handle_task = AsyncMock(return_value=True)
        adapter2.capabilities = [
            A2AAgentCapability(name="data_processing", description="Data processing"),
            A2AAgentCapability(name="analysis", description="Data analysis"),
            A2AAgentCapability(name="visualization", description="Data visualization"),
        ]

        task = A2AAtomicTask(
            description="Analyze and visualize data",
            required_capabilities=["data_processing", "analysis", "visualization"],
        )

        best_agent_id = await bridge._find_best_agent(task)

        assert (
            best_agent_id == adapter2.agent_id
        )  # Should prefer agent with more matching capabilities

    @pytest.mark.asyncio
    async def test_find_best_agent_no_match(self, bridge, mock_agent):
        """Test finding best agent when no agent can handle the task."""
        adapter = bridge.register_agent(mock_agent)
        adapter.can_handle_task = AsyncMock(return_value=False)

        task = A2AAtomicTask(
            description="Impossible task",
            required_capabilities=["impossible_capability"],
        )

        best_agent_id = await bridge._find_best_agent(task)

        assert best_agent_id is None

    @pytest.mark.asyncio
    async def test_get_task_status_active_task(self, bridge, mock_agent):
        """Test getting status of an active task."""
        adapter = bridge.register_agent(mock_agent)

        # Create and start a task without completing it
        task = A2AAtomicTask(description="Long running task")
        task.status = A2ATaskStatus.IN_PROGRESS
        task.assigned_agent_id = adapter.agent_id
        bridge.active_tasks[task.task_id] = task  # type: ignore

        retrieved_task = await bridge.get_task_status(task.task_id)

        assert retrieved_task == task
        assert retrieved_task.status == A2ATaskStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_get_task_status_completed_task(self, bridge, mock_agent):
        """Test getting status of a completed task."""
        adapter = bridge.register_agent(mock_agent)
        adapter.execute_atomic_task = AsyncMock(
            return_value={"status": "success", "result": "Completed"}
        )

        # Delegate and complete a task
        task = await bridge.delegate_atomic_task(
            task_description="Test task", target_agent_id=adapter.agent_id
        )

        retrieved_task = await bridge.get_task_status(task.task_id)

        assert retrieved_task == task
        assert retrieved_task.status == A2ATaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, bridge):
        """Test getting status of a non-existent task."""
        retrieved_task = await bridge.get_task_status("non_existent_task_id")

        assert retrieved_task is None

    def test_get_agent_discovery_info(self, bridge, mock_agent, mock_agent_2):
        """Test getting agent discovery information."""
        # Register agents with capabilities
        adapter1 = bridge.register_agent(mock_agent)
        adapter1.capabilities = [
            A2AAgentCapability(name="processing", description="Data processing")
        ]

        adapter2 = bridge.register_agent(mock_agent_2)
        adapter2.capabilities = [
            A2AAgentCapability(name="analysis", description="Data analysis"),
            A2AAgentCapability(name="visualization", description="Data visualization"),
        ]

        discovery_info = bridge.get_agent_discovery_info()

        assert "agents" in discovery_info
        assert "protocol_version" in discovery_info
        assert "supported_features" in discovery_info

        assert discovery_info["protocol_version"] == "a2a-v1"  # type: ignore

        expected_features = [
            "atomic_task_delegation",
            "capability_discovery",
            "status_monitoring",
        ]
        for feature in expected_features:
            assert feature in discovery_info["supported_features"]  # type: ignore

        # Verify agent information
        agents_info = discovery_info["agents"]  # type: ignore
        assert adapter1.agent_id in agents_info
        assert adapter2.agent_id in agents_info

        agent1_info = agents_info[adapter1.agent_id]
        assert "capabilities" in agent1_info
        assert "status" in agent1_info
        assert "endpoints" in agent1_info
        assert agent1_info["status"] == "available"


class TestA2AAtomicTask:
    """Test cases for A2AAtomicTask."""

    def test_atomic_task_creation(self):
        """Test creating an A2A atomic task."""
        task = A2AAtomicTask(
            description="Process user data",
            required_capabilities=["data_processing", "validation"],
            input_data={"user_id": "123", "data": "sample"},
        )

        assert task.description == "Process user data"
        assert task.required_capabilities == ["data_processing", "validation"]
        assert task.input_data == {"user_id": "123", "data": "sample"}
        assert task.status == A2ATaskStatus.PENDING
        assert task.task_id is not None
        assert task.assigned_agent_id is None
        assert task.result is None
        assert task.error_message is None
        assert task.started_at is None
        assert task.completed_at is None

    def test_atomic_task_status_transitions(self):
        """Test A2A task status transitions."""
        task = A2AAtomicTask(description="Test task")

        # Initial state
        assert task.status == A2ATaskStatus.PENDING

        # Start task
        task.status = A2ATaskStatus.IN_PROGRESS
        task.started_at = asyncio.get_event_loop().time()
        assert task.status == A2ATaskStatus.IN_PROGRESS
        assert task.started_at is not None

        # Complete task
        task.status = A2ATaskStatus.COMPLETED
        task.completed_at = asyncio.get_event_loop().time()
        task.result = {"status": "success", "result": "Task completed successfully"}
        assert task.status == A2ATaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.result is not None

    def test_atomic_task_failure(self):
        """Test A2A task failure handling."""
        task = A2AAtomicTask(description="Test task")

        task.status = A2ATaskStatus.FAILED
        task.error_message = "Task execution failed"
        task.completed_at = asyncio.get_event_loop().time()

        assert task.status == A2ATaskStatus.FAILED
        assert task.error_message == "Task execution failed"
        assert task.completed_at is not None


class TestReactiveAgentA2AAdapter:
    """Test cases for ReactiveAgentA2AAdapter."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock reactive agent."""
        agent = Mock()
        agent.context = Mock()
        agent.context.agent_name = "TestAgent"
        agent.context.tool_use_enabled = True
        agent.context.tools = ["text_tool", "data_tool"]  # Make tools iterable
        agent.agent_logger = Mock()
        agent.run = AsyncMock(
            return_value={"status": "success", "result": "Task completed"}
        )
        agent.get_available_strategies = Mock(
            return_value=["reactive", "plan_execute_reflect"]
        )
        agent.get_current_strategy = Mock(return_value="reactive")
        agent.get_reasoning_context = Mock(return_value={"iteration_count": 1})
        return agent

    @pytest.fixture
    def adapter(self, mock_agent):
        """Create an A2A adapter instance."""
        return ReactiveAgentA2AAdapter(mock_agent)

    def test_adapter_initialization(self, adapter, mock_agent):
        """Test adapter initialization."""
        assert adapter.reactive_agent == mock_agent
        assert adapter.agent_id == "TestAgent"
        assert len(adapter.capabilities) > 0
        assert all(isinstance(cap, A2AAgentCapability) for cap in adapter.capabilities)

    def test_adapter_default_capabilities(self, adapter):
        """Test adapter default capabilities."""
        capability_names = [cap.name for cap in adapter.capabilities]

        # The adapter creates these capabilities based on the mock setup:
        # - reasoning (from get_available_strategies)
        # - tool_text_tool, tool_data_tool (from context.tools)
        # - quick_response (from current_strategy being "reactive")
        expected_capabilities = [
            "reasoning",
            "tool_text_tool",
            "tool_data_tool",
            "quick_response",
        ]

        for expected_cap in expected_capabilities:
            assert expected_cap in capability_names

    @pytest.mark.asyncio
    async def test_adapter_can_handle_task_basic(self, adapter):
        """Test adapter can handle basic tasks."""
        task = A2AAtomicTask(
            description="Process some text", required_capabilities=["reasoning"]
        )

        can_handle = await adapter.can_handle_task(task)
        assert can_handle is True

    @pytest.mark.asyncio
    async def test_adapter_can_handle_task_missing_capability(self, adapter):
        """Test adapter cannot handle tasks requiring missing capabilities."""
        task = A2AAtomicTask(
            description="Advanced AI task",
            required_capabilities=["advanced_ai", "machine_learning"],
        )

        can_handle = await adapter.can_handle_task(task)
        assert can_handle is False

    @pytest.mark.asyncio
    async def test_adapter_can_handle_task_no_requirements(self, adapter):
        """Test adapter can handle tasks with no capability requirements."""
        task = A2AAtomicTask(description="Generic task", required_capabilities=[])

        can_handle = await adapter.can_handle_task(task)
        assert can_handle is True

    @pytest.mark.asyncio
    async def test_adapter_execute_atomic_task(self, adapter, mock_agent):
        """Test adapter executing an atomic task."""
        task = A2AAtomicTask(
            description="Process user data",
            input_data={"user_id": "123", "action": "analyze"},
        )

        result = await adapter.execute_atomic_task(task)

        assert result is not None
        mock_agent.run.assert_called_once_with(
            "Process user data\n\nContext:\nuser_id: 123\naction: analyze"
        )

    @pytest.mark.asyncio
    async def test_adapter_execute_atomic_task_with_context(self, adapter, mock_agent):
        """Test adapter executing atomic task with input data context."""
        task = A2AAtomicTask(
            description="Process ${action} for user ${user_id}",
            input_data={"user_id": "456", "action": "verification"},
        )

        result = await adapter.execute_atomic_task(task)

        assert result is not None
        # In a real implementation, the adapter might substitute variables
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_execute_atomic_task_error(self, adapter, mock_agent):
        """Test adapter handling execution errors."""
        mock_agent.run.side_effect = Exception("Agent execution failed")

        task = A2AAtomicTask(description="Failing task")

        with pytest.raises(Exception, match="Agent execution failed"):
            await adapter.execute_atomic_task(task)


class TestA2AIntegrationScenarios:
    """Integration test scenarios for A2A components."""

    @pytest.mark.asyncio
    async def test_multi_agent_task_delegation_workflow(self):
        """Test complete multi-agent task delegation workflow."""
        bridge = A2AOfficialBridge()

        # Create agents with different specializations
        data_agent = Mock()
        data_agent.context = Mock()
        data_agent.context.agent_name = "DataProcessor"
        data_agent.context.tool_use_enabled = True
        data_agent.context.tools = ["data_tool", "processing_tool"]
        data_agent.run = AsyncMock(
            return_value={"result": "data_processed", "data": [1, 2, 3, 4, 5]}
        )
        data_agent.get_available_strategies = Mock(
            return_value=["reactive", "plan_execute_reflect"]
        )
        data_agent.get_current_strategy = Mock(return_value="reactive")
        data_agent.get_reasoning_context = Mock(return_value={"iteration_count": 1})

        analysis_agent = Mock()
        analysis_agent.context = Mock()
        analysis_agent.context.agent_name = "AnalysisAgent"
        analysis_agent.context.tool_use_enabled = True
        analysis_agent.context.tools = ["analysis_tool", "visualization_tool"]
        analysis_agent.run = AsyncMock(
            return_value={
                "result": "analysis_complete",
                "insights": ["trend_up", "outliers_detected"],
            }
        )
        analysis_agent.get_available_strategies = Mock(
            return_value=["reactive", "plan_execute_reflect"]
        )
        analysis_agent.get_current_strategy = Mock(return_value="plan_execute_reflect")
        analysis_agent.get_reasoning_context = Mock(return_value={"iteration_count": 2})

        # Register agents
        data_adapter = bridge.register_agent(data_agent)
        analysis_adapter = bridge.register_agent(analysis_agent)

        # Configure capabilities
        data_adapter.capabilities = [
            A2AAgentCapability(name="data_processing", description="Process raw data"),
            A2AAgentCapability(
                name="data_validation", description="Validate data integrity"
            ),
        ]
        data_adapter.can_handle_task = AsyncMock(return_value=True)
        data_adapter.execute_atomic_task = AsyncMock(
            return_value="Data processed successfully"
        )

        analysis_adapter.capabilities = [
            A2AAgentCapability(
                name="data_analysis", description="Analyze processed data"
            ),
            A2AAgentCapability(
                name="insights_generation", description="Generate insights"
            ),
        ]
        analysis_adapter.can_handle_task = AsyncMock(return_value=True)
        analysis_adapter.execute_atomic_task = AsyncMock(
            return_value={
                "status": "success",
                "result": "Analysis completed successfully",
            }
        )

        # Execute workflow
        # Step 1: Process data
        data_task = await bridge.delegate_atomic_task(
            task_description="Process customer data",
            required_capabilities=["data_processing"],
            input_data={"dataset": "customer_data.csv"},
        )

        assert data_task.status == A2ATaskStatus.COMPLETED
        assert data_task.assigned_agent_id == data_adapter.agent_id

        # Step 2: Analyze processed data
        analysis_task = await bridge.delegate_atomic_task(
            task_description="Analyze processed customer data",
            required_capabilities=["data_analysis"],
            input_data={"processed_data": data_task.result},
        )

        assert analysis_task.status == A2ATaskStatus.COMPLETED
        assert analysis_task.assigned_agent_id == analysis_adapter.agent_id

        # Verify both tasks completed successfully
        assert len(bridge.completed_tasks) == 2
        assert len(bridge.active_tasks) == 0

    @pytest.mark.asyncio
    async def test_agent_discovery_and_capability_matching(self):
        """Test agent discovery and capability-based task routing."""
        bridge = A2AOfficialBridge()

        # Register multiple agents with different capabilities
        agents_config = [
            {
                "name": "GeneralistAgent",
                "capabilities": ["task_execution", "text_processing"],
            },
            {
                "name": "SpecialistAgent",
                "capabilities": ["data_analysis", "machine_learning", "visualization"],
            },
            {
                "name": "CommunicationAgent",
                "capabilities": ["communication", "translation", "summarization"],
            },
        ]

        registered_adapters = []
        for config in agents_config:
            mock_agent = Mock()
            mock_agent.context = Mock()
            mock_agent.context.agent_name = config["name"]
            mock_agent.context.tool_use_enabled = True
            mock_agent.context.tools = ["general_tool"]  # Add tools
            mock_agent.run = AsyncMock(return_value={"status": "success"})
            mock_agent.get_available_strategies = Mock(
                return_value=["reactive", "plan_execute_reflect"]
            )
            mock_agent.get_current_strategy = Mock(return_value="reactive")
            mock_agent.get_reasoning_context = Mock(return_value={"iteration_count": 1})

            adapter = bridge.register_agent(mock_agent)
            adapter.capabilities = [
                A2AAgentCapability(name=cap, description=f"{cap} capability")
                for cap in config["capabilities"]
            ]
            adapter.can_handle_task = AsyncMock(return_value=True)
            adapter.execute_atomic_task = AsyncMock(
                return_value=f"Task completed by {config['name']}"
            )

            registered_adapters.append(adapter)

        # Test discovery info
        discovery_info = bridge.get_agent_discovery_info()
        assert len(discovery_info["agents"]) == 3

        for config in agents_config:
            agent_name = config["name"]
            assert agent_name in discovery_info["agents"]
            agent_info = discovery_info["agents"][agent_name]
            assert len(agent_info["capabilities"]) == len(config["capabilities"])

        # Test capability-based routing
        # Task requiring ML capabilities should go to SpecialistAgent
        ml_task = await bridge.delegate_atomic_task(
            task_description="Build machine learning model",
            required_capabilities=["machine_learning", "data_analysis"],
        )

        assert ml_task.status == A2ATaskStatus.COMPLETED
        assert ml_task.assigned_agent_id == "SpecialistAgent"

        # Task requiring communication should go to CommunicationAgent
        comm_task = await bridge.delegate_atomic_task(
            task_description="Translate document",
            required_capabilities=["translation", "communication"],
        )

        assert comm_task.status == A2ATaskStatus.COMPLETED
        assert comm_task.assigned_agent_id == "CommunicationAgent"

    @pytest.mark.asyncio
    async def test_task_status_monitoring_lifecycle(self):
        """Test complete task status monitoring lifecycle."""
        bridge = A2AOfficialBridge()

        # Register agent
        mock_agent = Mock()
        mock_agent.context = Mock()
        mock_agent.context.agent_name = "MonitoredAgent"
        mock_agent.context.tool_use_enabled = True
        mock_agent.context.tools = ["monitoring_tool"]
        mock_agent.run = AsyncMock(return_value={"status": "success"})
        mock_agent.get_available_strategies = Mock(
            return_value=["reactive", "plan_execute_reflect"]
        )
        mock_agent.get_current_strategy = Mock(return_value="reactive")
        mock_agent.get_reasoning_context = Mock(return_value={"iteration_count": 1})

        adapter = bridge.register_agent(mock_agent)
        adapter.can_handle_task = AsyncMock(return_value=True)

        # Create a slow executing task to test monitoring
        async def slow_execute_task(task):
            await asyncio.sleep(0.1)  # Simulate work
            return {"status": "success", "result": "Slow task completed"}

        adapter.execute_atomic_task = slow_execute_task

        # Start task delegation in background
        task_future = asyncio.create_task(
            bridge.delegate_atomic_task(
                task_description="Long running task", target_agent_id=adapter.agent_id
            )
        )

        # Give task time to start
        await asyncio.sleep(0.05)

        # Task should be in progress
        task_id = None
        for tid in bridge.active_tasks:
            task_id = tid
            break

        assert task_id is not None
        active_task = await bridge.get_task_status(task_id)
        assert active_task is not None
        assert active_task.status == A2ATaskStatus.IN_PROGRESS

        # Wait for task completion
        completed_task = await task_future

        # Task should now be completed
        final_status = await bridge.get_task_status(completed_task.task_id)
        assert final_status is not None
        assert final_status.status == A2ATaskStatus.COMPLETED
        assert final_status.result == {
            "status": "success",
            "result": "Slow task completed",
        }
        assert completed_task.task_id not in bridge.active_tasks
        assert completed_task in bridge.completed_tasks
