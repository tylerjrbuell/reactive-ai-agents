"""
Tests for A2ACommunicationProtocol.

Tests the agent-to-agent communication functionality including message handling,
task delegation, broadcasting, and protocol lifecycle management.
"""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock, patch, create_autospec
from reactive_agents.app.communication.a2a_protocol import A2ACommunicationProtocol
from reactive_agents.core.types.communication_types import (
    A2AMessage,
    A2AResponse,
    MessageType,
    MessagePriority,
)


class TestA2ACommunicationProtocol:
    """Test cases for A2ACommunicationProtocol."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock reactive agent."""
        agent = Mock()
        agent.context = Mock()
        agent.context.agent_name = "TestAgent"
        agent.agent_logger = Mock()
        agent.run = AsyncMock(
            return_value={
                "status": "success",
                "result": "Task completed",
                "execution_time": 1.5,
            }
        )
        return agent

    @pytest.fixture
    def mock_agent_2(self):
        """Create a second mock reactive agent."""
        agent = Mock()
        agent.context = Mock()
        agent.context.agent_name = "Agent2"
        agent.agent_logger = Mock()
        agent.run = AsyncMock(
            return_value={
                "status": "success",
                "result": "Task completed by Agent2",
                "execution_time": 2.0,
            }
        )
        return agent

    @pytest.fixture
    def protocol(self, mock_agent):
        """Create a A2A communication protocol instance."""
        return A2ACommunicationProtocol(mock_agent)

    @pytest.fixture
    def protocol_2(self, mock_agent_2):
        """Create a second A2A communication protocol instance."""
        return A2ACommunicationProtocol(mock_agent_2)

    def test_initialization(self, protocol, mock_agent):
        """Test protocol initialization."""
        assert protocol.agent == mock_agent
        assert protocol.agent_id == "TestAgent"
        assert protocol.inbox.empty()
        assert protocol.outbox.empty()
        assert protocol.conversations == {}
        assert protocol.pending_responses == {}
        assert protocol.connected_agents == {}
        assert protocol._running is False
        assert protocol._message_processor_task is None

        # Verify message handlers are set up
        assert MessageType.REQUEST in protocol.message_handlers
        assert MessageType.DELEGATION in protocol.message_handlers
        assert MessageType.NOTIFICATION in protocol.message_handlers
        assert MessageType.BROADCAST in protocol.message_handlers

    @pytest.mark.asyncio
    async def test_start_protocol(self, protocol, mock_agent):
        """Test starting the protocol."""
        await protocol.start()

        assert protocol._running is True
        assert protocol._message_processor_task is not None
        assert not protocol._message_processor_task.done()
        mock_agent.agent_logger.info.assert_called()

        # Clean up
        await protocol.stop()

    @pytest.mark.asyncio
    async def test_stop_protocol(self, protocol):
        """Test stopping the protocol."""
        await protocol.start()
        await protocol.stop()

        assert protocol._running is False
        assert (
            protocol._message_processor_task.cancelled()
            or protocol._message_processor_task.done()
        )

    @pytest.mark.asyncio
    async def test_start_already_running(self, protocol, mock_agent):
        """Test starting protocol when already running."""
        await protocol.start()

        # Try to start again
        await protocol.start()

        # Should still be running and only one task
        assert protocol._running is True
        assert protocol._message_processor_task is not None

        await protocol.stop()

    def test_connect_agent(self, protocol, protocol_2, mock_agent):
        """Test connecting two agents."""
        protocol.connect_agent(protocol_2)

        assert protocol_2.agent_id in protocol.connected_agents
        assert protocol.agent_id in protocol_2.connected_agents
        assert protocol.connected_agents[protocol_2.agent_id] == protocol_2
        assert protocol_2.connected_agents[protocol.agent_id] == protocol

        mock_agent.agent_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_send_message_no_response(self, protocol):
        """Test sending a message that doesn't require response."""
        message = A2AMessage(
            message_type=MessageType.NOTIFICATION,
            sender_id=protocol.agent_id,
            recipient_id="TargetAgent",
            subject="Test notification",
            content={"data": "test"},
            requires_response=False,
        )

        with patch.object(protocol, "_deliver_message") as mock_deliver:
            response = await protocol.send_message(message)

            assert response is None
            # Message should be in outbox (but _deliver_message won't be called until _process_messages runs)
            assert not protocol.outbox.empty()

    @pytest.mark.asyncio
    async def test_send_message_with_response(self, protocol, protocol_2):
        """Test sending a message that requires response."""
        protocol.connect_agent(protocol_2)

        message = A2AMessage(
            message_type=MessageType.REQUEST,
            sender_id=protocol.agent_id,
            recipient_id=protocol_2.agent_id,
            subject="Test request",
            content={"task": "Complete this task"},
            requires_response=True,
            timeout_seconds=5.0,
        )

        # Start both protocols to handle message processing
        await protocol.start()
        await protocol_2.start()

        try:
            # Send message and wait for response
            response = await protocol.send_message(message)

            assert response is not None
            assert isinstance(response, A2AResponse)
            assert response.original_message_id == message.message_id
            assert response.sender_id == protocol_2.agent_id

        finally:
            await protocol.stop()
            await protocol_2.stop()

    @pytest.mark.asyncio
    async def test_send_message_timeout(self, protocol):
        """Test sending a message with timeout."""
        message = A2AMessage(
            message_type=MessageType.REQUEST,
            sender_id=protocol.agent_id,
            recipient_id="NonExistentAgent",
            subject="Test request",
            content={"task": "Complete this task"},
            requires_response=True,
            timeout_seconds=0.1,  # Very short timeout
        )

        await protocol.start()

        try:
            response = await protocol.send_message(message)

            assert response is not None
            assert response.success is False
            assert "timeout" in response.error_message.lower()

        finally:
            await protocol.stop()

    @pytest.mark.asyncio
    async def test_delegate_task(self, protocol, protocol_2, mock_agent_2):
        """Test task delegation."""
        protocol.connect_agent(protocol_2)

        await protocol.start()
        await protocol_2.start()

        try:
            response = await protocol.delegate_task(
                target_agent_id=protocol_2.agent_id,
                task="Complete data analysis",
                shared_context={"dataset": "sales_data"},
                success_criteria={"accuracy": ">95%"},
                timeout_seconds=5.0,
            )

            assert response is not None
            assert response.success is True
            assert response.sender_id == protocol_2.agent_id
            assert "result" in response.content

            # Verify the task was executed
            mock_agent_2.run.assert_called_with("Complete data analysis")

        finally:
            await protocol.stop()
            await protocol_2.stop()

    @pytest.mark.asyncio
    async def test_delegate_task_with_shared_context(
        self, protocol, protocol_2, mock_agent_2
    ):
        """Test task delegation with shared context."""
        protocol.connect_agent(protocol_2)

        await protocol.start()
        await protocol_2.start()

        try:
            shared_context = {"database_url": "test://db", "api_key": "secret"}

            response = await protocol.delegate_task(
                target_agent_id=protocol_2.agent_id,
                task="Process data",
                shared_context=shared_context,
                timeout_seconds=5.0,
            )

            assert response is not None
            assert response.success is True

            # Verify shared context was applied
            for key, value in shared_context.items():
                assert hasattr(mock_agent_2.context, f"shared_{key}")
                assert getattr(mock_agent_2.context, f"shared_{key}") == value

        finally:
            await protocol.stop()
            await protocol_2.stop()

    @pytest.mark.asyncio
    async def test_broadcast_message(self, protocol, protocol_2, mock_agent):
        """Test broadcasting message to all connected agents."""
        # Connect multiple agents
        protocol_3 = A2ACommunicationProtocol(
            Mock(context=Mock(agent_name="Agent3"), agent_logger=Mock())
        )
        protocol.connect_agent(protocol_2)
        protocol.connect_agent(protocol_3)

        await protocol.start()
        await protocol_2.start()
        await protocol_3.start()

        try:
            await protocol.broadcast_message(
                subject="System update",
                content={"version": "1.2.0", "changes": ["bug fixes", "new features"]},
                priority=MessagePriority.HIGH,
            )

            # Give some time for message processing
            await asyncio.sleep(0.1)

            # Verify message was sent to outbox
            mock_agent.agent_logger.info.assert_called()

        finally:
            await protocol.stop()
            await protocol_2.stop()
            await protocol_3.stop()

    @pytest.mark.asyncio
    async def test_message_processing_loop(self, protocol, protocol_2):
        """Test the message processing loop."""
        protocol.connect_agent(protocol_2)

        # Create a test message
        message = A2AMessage(
            message_type=MessageType.NOTIFICATION,
            sender_id=protocol.agent_id,
            recipient_id=protocol_2.agent_id,
            subject="Test notification",
            content={"data": "test"},
        )

        await protocol.start()
        await protocol_2.start()

        try:
            # Add message to outbox
            await protocol.outbox.put(message)

            # Give time for processing
            await asyncio.sleep(0.1)

            # Message should have been delivered to protocol_2's inbox
            # (Note: in actual implementation, it would be processed and removed from inbox)
            assert protocol.outbox.empty()

        finally:
            await protocol.stop()
            await protocol_2.stop()

    @pytest.mark.asyncio
    async def test_handle_request_message(self, protocol, mock_agent):
        """Test handling request messages."""
        message = A2AMessage(
            message_type=MessageType.REQUEST,
            sender_id="SenderAgent",
            recipient_id=protocol.agent_id,
            subject="Process data",
            content={"task": "Analyze sales data"},
            requires_response=True,
        )

        response = await protocol._handle_request(message)

        assert response is not None
        assert isinstance(response, A2AResponse)
        assert response.success is True
        assert response.sender_id == protocol.agent_id
        assert response.original_message_id == message.message_id
        assert "result" in response.content

        mock_agent.run.assert_called_with("Analyze sales data")

    @pytest.mark.asyncio
    async def test_handle_request_message_error(self, protocol, mock_agent):
        """Test handling request messages with errors."""
        mock_agent.run.side_effect = Exception("Processing failed")

        message = A2AMessage(
            message_type=MessageType.REQUEST,
            sender_id="SenderAgent",
            recipient_id=protocol.agent_id,
            subject="Process data",
            content={"task": "Analyze sales data"},
            requires_response=True,
        )

        response = await protocol._handle_request(message)

        assert response is not None
        assert response.success is False
        assert response.error_message == "Processing failed"

    @pytest.mark.asyncio
    async def test_handle_delegation_message(self, protocol, mock_agent):
        """Test handling delegation messages."""
        message = A2AMessage(
            message_type=MessageType.DELEGATION,
            sender_id="DelegatorAgent",
            recipient_id=protocol.agent_id,
            subject="Task delegation",
            content={
                "delegated_task": "Complete quarterly report",
                "shared_context": {"quarter": "Q4", "year": "2024"},
                "success_criteria": {"completeness": "100%"},
            },
            requires_response=True,
        )

        response = await protocol._handle_delegation(message)

        assert response is not None
        assert response.success is True
        assert response.content["task"] == "Complete quarterly report"
        assert "result" in response.content

        # Verify the task was executed
        mock_agent.run.assert_called_with("Complete quarterly report")

    @pytest.mark.asyncio
    async def test_handle_delegation_message_error(self, protocol, mock_agent):
        """Test handling delegation messages with errors."""
        mock_agent.run.side_effect = Exception("Delegation failed")

        message = A2AMessage(
            message_type=MessageType.DELEGATION,
            sender_id="DelegatorAgent",
            recipient_id=protocol.agent_id,
            subject="Task delegation error test",
            content={"delegated_task": "Complete report"},
            requires_response=True,
        )

        response = await protocol._handle_delegation(message)

        assert response is not None
        assert response.success is False
        assert response.error_message == "Delegation failed"

    @pytest.mark.asyncio
    async def test_handle_notification_message(self, protocol, mock_agent):
        """Test handling notification messages."""
        message = A2AMessage(
            message_type=MessageType.NOTIFICATION,
            sender_id="NotifierAgent",
            recipient_id=protocol.agent_id,
            subject="System alert",
            content={"alert": "High CPU usage detected"},
        )

        response = await protocol._handle_notification(message)

        assert response is None  # Notifications don't require responses
        mock_agent.agent_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_handle_broadcast_message(self, protocol, mock_agent):
        """Test handling broadcast messages."""
        message = A2AMessage(
            message_type=MessageType.BROADCAST,
            sender_id="BroadcasterAgent",
            subject="System maintenance",
            content={"maintenance_window": "2024-01-01 02:00-04:00"},
        )

        response = await protocol._handle_broadcast(message)

        assert response is None  # Broadcasts don't require responses
        mock_agent.agent_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_deliver_message_direct(self, protocol, protocol_2):
        """Test delivering direct messages."""
        protocol.connect_agent(protocol_2)

        message = A2AMessage(
            message_type=MessageType.NOTIFICATION,
            sender_id=protocol.agent_id,
            recipient_id=protocol_2.agent_id,
            subject="Direct message",
            content={"data": "test"},
        )

        await protocol._deliver_message(message)

        # Message should be in protocol_2's inbox
        assert not protocol_2.inbox.empty()
        received_message = await protocol_2.inbox.get()
        assert received_message.message_id == message.message_id

    @pytest.mark.asyncio
    async def test_deliver_message_broadcast(self, protocol, protocol_2):
        """Test delivering broadcast messages."""
        protocol_3 = A2ACommunicationProtocol(
            Mock(context=Mock(agent_name="Agent3"), agent_logger=Mock())
        )
        protocol.connect_agent(protocol_2)
        protocol.connect_agent(protocol_3)

        message = A2AMessage(
            message_type=MessageType.BROADCAST,
            sender_id=protocol.agent_id,
            recipient_id=None,  # Broadcast
            subject="Broadcast message",
            content={"data": "broadcast"},
        )

        await protocol._deliver_message(message)

        # Message should be in both connected agents' inboxes
        assert not protocol_2.inbox.empty()
        assert not protocol_3.inbox.empty()

    @pytest.mark.asyncio
    async def test_handle_incoming_message_stores_conversation(
        self, protocol, mock_agent
    ):
        """Test that incoming messages are stored in conversation history."""
        message = A2AMessage(
            message_type=MessageType.NOTIFICATION,
            sender_id="SenderAgent",
            recipient_id=protocol.agent_id,
            subject="Test message",
            content={"data": "test"},
            conversation_id="conv_123",
        )

        await protocol._handle_incoming_message(message)

        assert "conv_123" in protocol.conversations
        assert len(protocol.conversations["conv_123"]) == 1
        assert protocol.conversations["conv_123"][0] == message

    @pytest.mark.asyncio
    async def test_handle_incoming_message_generates_conversation_id(
        self, protocol, mock_agent
    ):
        """Test that conversation ID is generated if not provided."""
        message = A2AMessage(
            message_type=MessageType.NOTIFICATION,
            sender_id="SenderAgent",
            recipient_id=protocol.agent_id,
            subject="Test message",
            content={"data": "test"},
            # No conversation_id provided
        )

        await protocol._handle_incoming_message(message)

        # Should use message_id as conversation_id
        assert message.message_id in protocol.conversations
        assert len(protocol.conversations[message.message_id]) == 1

    @pytest.mark.asyncio
    async def test_send_response_completes_future(self, protocol, protocol_2):
        """Test that sending response completes pending futures."""
        protocol.connect_agent(protocol_2)

        # Create a pending response future
        message_id = str(uuid.uuid4())
        future = asyncio.Future()
        protocol.pending_responses[message_id] = future

        original_message = A2AMessage(
            message_type=MessageType.REQUEST,
            sender_id=protocol.agent_id,
            recipient_id=protocol_2.agent_id,
            subject="Test",
            content={},
            message_id=message_id,
        )

        response = A2AResponse(
            original_message_id=message_id,
            sender_id=protocol_2.agent_id,
            success=True,
            content={"result": "test"},
        )

        await protocol_2._send_response(original_message, response)

        # Future should be completed
        assert future.done()
        assert future.result() == response
        assert message_id not in protocol.pending_responses

    @pytest.mark.asyncio
    async def test_message_processing_error_handling(self, protocol, mock_agent):
        """Test error handling in message processing loop."""
        # Mock a processing error
        with patch.object(
            protocol, "_deliver_message", side_effect=Exception("Processing error")
        ):
            await protocol.start()

            # Add a message to trigger processing
            message = A2AMessage(
                message_type=MessageType.NOTIFICATION,
                sender_id=protocol.agent_id,
                recipient_id="TestAgent",
                subject="Test",
                content={},
            )
            await protocol.outbox.put(message)

            # Give time for processing
            await asyncio.sleep(0.1)

            # Error should be logged
            mock_agent.agent_logger.error.assert_called()

            await protocol.stop()

    @pytest.mark.asyncio
    async def test_delegate_task_no_response(self, protocol):
        """Test task delegation when no response is received."""
        # Don't connect any agents, so no response will come
        response = await protocol.delegate_task(
            target_agent_id="NonExistentAgent", task="Test task", timeout_seconds=0.1
        )

        assert response is not None
        assert response.success is False
        assert "timeout" in response.error_message.lower()

    def test_message_handlers_coverage(self, protocol):
        """Test that all message types have handlers."""
        expected_types = [
            MessageType.REQUEST,
            MessageType.DELEGATION,
            MessageType.NOTIFICATION,
            MessageType.BROADCAST,
        ]

        for msg_type in expected_types:
            assert msg_type in protocol.message_handlers
            assert callable(protocol.message_handlers[msg_type])

    @pytest.mark.asyncio
    async def test_protocol_lifecycle_integration(
        self, protocol, protocol_2, mock_agent, mock_agent_2
    ):
        """Test complete protocol lifecycle integration."""
        # Connect agents
        protocol.connect_agent(protocol_2)

        # Start protocols
        await protocol.start()
        await protocol_2.start()

        try:
            # Test various message types

            # 1. Broadcast
            await protocol.broadcast_message(
                subject="System status", content={"status": "operational"}
            )

            # 2. Task delegation
            response = await protocol.delegate_task(
                target_agent_id=protocol_2.agent_id,
                task="Process user request",
                timeout_seconds=5.0,
            )
            assert response.success is True

            # 3. Direct message
            direct_message = A2AMessage(
                message_type=MessageType.REQUEST,
                sender_id=protocol.agent_id,
                recipient_id=protocol_2.agent_id,
                subject="Direct request",
                content={"task": "Handle request"},
                requires_response=True,
            )

            response = await protocol.send_message(direct_message)
            assert response is not None
            assert response.success is True

            # Verify agents were called
            mock_agent_2.run.assert_called()

        finally:
            await protocol.stop()
            await protocol_2.stop()
