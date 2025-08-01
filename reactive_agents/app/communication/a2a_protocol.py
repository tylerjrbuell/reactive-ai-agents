from __future__ import annotations
import uuid
import time
import asyncio
from typing import Dict, Any, List, Optional, Literal, Union, TYPE_CHECKING
from pydantic import BaseModel, Field

# Import communication types from centralized location
from reactive_agents.core.types.communication_types import (
    MessageType,
    MessagePriority,
    A2AMessage,
    A2AResponse,
    A2ATaskStatus,
    A2AAtomicTask,
    A2AAgentCapability,
    A2AAgentProfile,
)

if TYPE_CHECKING:
    from reactive_agents.app.agents.reactive_agent import ReactiveAgent


class A2ACommunicationProtocol:
    """
    Handles agent-to-agent communication following Google's A2A protocol principles.

    Features:
    - Standard message schema
    - Task delegation
    - Broadcast communication
    - Shared context management
    - Async message handling
    """

    def __init__(self, agent: "ReactiveAgent"):
        self.agent = agent
        self.agent_id = agent.context.agent_name

        # Message queues
        self.inbox: asyncio.Queue[A2AMessage] = asyncio.Queue()
        self.outbox: asyncio.Queue[A2AMessage] = asyncio.Queue()

        # Active conversations
        self.conversations: Dict[str, List[A2AMessage]] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}

        # Connected agents registry
        self.connected_agents: Dict[str, "A2ACommunicationProtocol"] = {}

        # Message handlers
        self.message_handlers = {
            MessageType.REQUEST: self._handle_request,
            MessageType.DELEGATION: self._handle_delegation,
            MessageType.NOTIFICATION: self._handle_notification,
            MessageType.BROADCAST: self._handle_broadcast,
        }

        # Background tasks
        self._running = False
        self._message_processor_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the A2A communication protocol."""
        if self._running:
            return

        self._running = True
        self._message_processor_task = asyncio.create_task(self._process_messages())

        if self.agent.agent_logger:
            self.agent.agent_logger.info(
                f"🌐 A2A Protocol started for agent {self.agent_id}"
            )

    async def stop(self):
        """Stop the A2A communication protocol."""
        self._running = False

        if self._message_processor_task:
            self._message_processor_task.cancel()
            try:
                await self._message_processor_task
            except asyncio.CancelledError:
                pass

        if self.agent.agent_logger:
            self.agent.agent_logger.info(
                f"🌐 A2A Protocol stopped for agent {self.agent_id}"
            )

    def connect_agent(self, other_agent: "A2ACommunicationProtocol"):
        """Connect to another agent for communication."""
        self.connected_agents[other_agent.agent_id] = other_agent
        other_agent.connected_agents[self.agent_id] = self

        if self.agent.agent_logger:
            self.agent.agent_logger.info(
                f"🤝 Connected to agent {other_agent.agent_id}"
            )

    async def send_message(self, message: A2AMessage) -> Optional[A2AResponse]:
        """
        Send a message to another agent.

        Args:
            message: The message to send

        Returns:
            Response if requires_response is True, None otherwise
        """
        # Add to outbox
        await self.outbox.put(message)

        # Wait for response if required
        if message.requires_response:
            future = asyncio.Future()
            self.pending_responses[message.message_id] = future

            try:
                if message.timeout_seconds:
                    response = await asyncio.wait_for(
                        future, timeout=message.timeout_seconds
                    )
                else:
                    response = await future

                return response
            except asyncio.TimeoutError:
                del self.pending_responses[message.message_id]
                return A2AResponse(
                    original_message_id=message.message_id,
                    sender_id=self.agent_id,
                    success=False,
                    content={},
                    error_message="Response timeout",
                )

        return None

    async def delegate_task(
        self,
        target_agent_id: str,
        task: str,
        shared_context: Optional[Dict[str, Any]] = None,
        success_criteria: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 300.0,
    ) -> A2AResponse:
        """
        Delegate a task to another agent.

        Args:
            target_agent_id: ID of the agent to delegate to
            task: Task description
            shared_context: Context to share with the target agent
            success_criteria: Criteria for successful completion
            timeout_seconds: How long to wait for completion

        Returns:
            Response from the target agent
        """
        message = A2AMessage(
            message_type=MessageType.DELEGATION,
            priority=MessagePriority.HIGH,
            sender_id=self.agent_id,
            recipient_id=target_agent_id,
            subject=f"Task Delegation: {task[:50]}...",
            content={"task": task},
            delegated_task=task,
            shared_context=shared_context or {},
            success_criteria=success_criteria or {},
            requires_response=True,
            timeout_seconds=timeout_seconds,
            conversation_id=str(uuid.uuid4()),
        )

        response = await self.send_message(message)

        if self.agent.agent_logger:
            self.agent.agent_logger.info(
                f"📤 Delegated task to {target_agent_id}: {task[:100]}..."
            )

        # Handle case where send_message returns None (shouldn't happen with requires_response=True)
        if response is None:
            return A2AResponse(
                original_message_id=message.message_id,
                sender_id=self.agent_id,
                success=False,
                content={},
                error_message="No response received from target agent",
            )

        return response

    async def broadcast_message(
        self,
        subject: str,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ):
        """
        Broadcast a message to all connected agents.

        Args:
            subject: Message subject
            content: Message content
            priority: Message priority
        """
        message = A2AMessage(
            message_type=MessageType.BROADCAST,
            priority=priority,
            sender_id=self.agent_id,
            recipient_id=None,  # Broadcast
            subject=subject,
            content=content,
        )

        await self.outbox.put(message)

        if self.agent.agent_logger:
            self.agent.agent_logger.info(f"📢 Broadcasting: {subject}")

    async def _process_messages(self):
        """Background task to process incoming and outgoing messages."""
        while self._running:
            try:
                # Process outgoing messages
                try:
                    message = self.outbox.get_nowait()
                    await self._deliver_message(message)
                except asyncio.QueueEmpty:
                    pass

                # Process incoming messages
                try:
                    message = self.inbox.get_nowait()
                    await self._handle_incoming_message(message)
                except asyncio.QueueEmpty:
                    pass

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)

            except Exception as e:
                if self.agent.agent_logger:
                    self.agent.agent_logger.error(f"A2A message processing error: {e}")

    async def _deliver_message(self, message: A2AMessage):
        """Deliver a message to its recipient(s)."""
        if message.recipient_id:
            # Direct message
            if message.recipient_id in self.connected_agents:
                target = self.connected_agents[message.recipient_id]
                await target.inbox.put(message)
        else:
            # Broadcast message
            for agent_protocol in self.connected_agents.values():
                await agent_protocol.inbox.put(message)

    async def _handle_incoming_message(self, message: A2AMessage):
        """Handle an incoming message."""
        # Store in conversation history
        conversation_id = message.conversation_id or message.message_id
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        self.conversations[conversation_id].append(message)

        # Handle based on message type
        handler = self.message_handlers.get(message.message_type)
        if handler:
            response = await handler(message)

            # Send response if required
            if message.requires_response and response:
                await self._send_response(message, response)

    async def _handle_request(self, message: A2AMessage) -> Optional[A2AResponse]:
        """Handle a request message."""
        if self.agent.agent_logger:
            self.agent.agent_logger.info(
                f"📨 Received request from {message.sender_id}: {message.subject}"
            )

        # Process the request using the agent
        try:
            task = message.content.get("task", message.subject)
            result = await self.agent.run(task)

            return A2AResponse(
                original_message_id=message.message_id,
                sender_id=self.agent_id,
                success=True,
                content={"result": result},
            )
        except Exception as e:
            return A2AResponse(
                original_message_id=message.message_id,
                sender_id=self.agent_id,
                success=False,
                content={},
                error_message=str(e),
            )

    async def _handle_delegation(self, message: A2AMessage) -> Optional[A2AResponse]:
        """Handle a task delegation."""
        task = message.delegated_task or message.content.get(
            "delegated_task", message.content.get("task", "No task specified")
        )

        if self.agent.agent_logger:
            self.agent.agent_logger.info(
                f"🎯 Received delegation from {message.sender_id}: {task}"
            )

        # Merge shared context if provided
        if message.shared_context:
            # Add shared context to agent's context
            for key, value in message.shared_context.items():
                setattr(self.agent.context, f"shared_{key}", value)

        # Execute the delegated task
        try:
            result = await self.agent.run(task)

            return A2AResponse(
                original_message_id=message.message_id,
                sender_id=self.agent_id,
                success=True,
                content={"result": result, "task": task},
            )
        except Exception as e:
            return A2AResponse(
                original_message_id=message.message_id,
                sender_id=self.agent_id,
                success=False,
                content={},
                error_message=str(e),
            )

    async def _handle_notification(self, message: A2AMessage) -> Optional[A2AResponse]:
        """Handle a notification message."""
        if self.agent.agent_logger:
            self.agent.agent_logger.info(
                f"🔔 Notification from {message.sender_id}: {message.subject}"
            )
        # Notifications don't require responses
        return None

    async def _handle_broadcast(self, message: A2AMessage) -> Optional[A2AResponse]:
        """Handle a broadcast message."""
        if self.agent.agent_logger:
            self.agent.agent_logger.info(
                f"📢 Broadcast from {message.sender_id}: {message.subject}"
            )
        # Broadcasts don't require responses
        return None

    async def _send_response(self, original_message: A2AMessage, response: A2AResponse):
        """Send a response to an original message."""
        # If the original sender is connected, deliver the response
        if original_message.sender_id in self.connected_agents:
            sender_protocol = self.connected_agents[original_message.sender_id]

            # Complete any pending future for this message
            if original_message.message_id in sender_protocol.pending_responses:
                future = sender_protocol.pending_responses.pop(
                    original_message.message_id
                )
                if not future.done():
                    future.set_result(response)


# Convenience functions for easy A2A setup
async def create_agent_network(
    agents: List["ReactiveAgent"],
) -> List[A2ACommunicationProtocol]:
    """
    Create a fully connected network of agents with A2A communication.

    Args:
        agents: List of ReactiveAgentV2 instances

    Returns:
        List of A2ACommunicationProtocol instances
    """
    protocols = []

    # Create protocols for each agent
    for agent in agents:
        protocol = A2ACommunicationProtocol(agent)
        protocols.append(protocol)

    # Connect all agents to each other (full mesh)
    for i, protocol_a in enumerate(protocols):
        for j, protocol_b in enumerate(protocols):
            if i != j:
                protocol_a.connect_agent(protocol_b)

    # Start all protocols
    for protocol in protocols:
        await protocol.start()

    return protocols
