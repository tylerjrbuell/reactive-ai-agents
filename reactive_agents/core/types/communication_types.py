"""Types for agent-to-agent communication and messaging."""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import uuid
import time
import asyncio


class MessageType(Enum):
    """Types of inter-agent messages."""
    
    REQUEST = "request"
    RESPONSE = "response"
    DELEGATION = "delegation"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    ERROR = "error"


class MessagePriority(Enum):
    """Message priority levels."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class A2ATaskStatus(Enum):
    """Status of A2A tasks following official A2A patterns."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class A2AMessage(BaseModel):
    """Standard message format for agent-to-agent communication."""
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    message_type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL
    
    # Sender and recipient info
    sender_id: str
    recipient_id: Optional[str] = None  # None for broadcast messages
    
    # Message content
    subject: str
    content: Dict[str, Any]
    
    # Context and metadata
    conversation_id: Optional[str] = None
    reply_to: Optional[str] = None
    requires_response: bool = False
    timeout_seconds: Optional[float] = None
    
    # Delegation specific fields
    delegated_task: Optional[str] = None
    success_criteria: Optional[Dict[str, Any]] = None
    shared_context: Optional[Dict[str, Any]] = None


class A2AResponse(BaseModel):
    """Response to an A2A message."""
    
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_message_id: str
    timestamp: float = Field(default_factory=time.time)
    
    # Response content
    success: bool
    content: Dict[str, Any]
    error_message: Optional[str] = None
    
    # Sender info
    sender_id: str
    
    # Context
    metadata: Dict[str, Any] = Field(default_factory=dict)


class A2AAtomicTask(BaseModel):
    """
    Atomic task following official A2A design principles.
    
    Per A2A documentation: Tasks should be atomic and processed by a single
    selected agent from start to finish.
    """
    
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    
    # Task metadata
    priority: int = Field(default=1, ge=1, le=10)  # 1=low, 10=urgent
    estimated_duration: Optional[float] = None  # seconds
    required_capabilities: List[str] = Field(default_factory=list)
    
    # Execution context
    input_data: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    
    # Status tracking
    status: A2ATaskStatus = A2ATaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Timing
    created_at: float = Field(default_factory=lambda: asyncio.get_event_loop().time())
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class A2AAgentCapability(BaseModel):
    """Capability declaration for A2A agent registration."""
    
    capability_id: str
    name: str
    description: str
    
    # Capability metadata
    version: str = "1.0.0"
    supported_inputs: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)
    
    # Performance metrics
    average_execution_time: Optional[float] = None
    success_rate: Optional[float] = None
    max_concurrent_tasks: int = 1
    
    # Requirements and constraints
    required_resources: Dict[str, Any] = Field(default_factory=dict)
    cost_per_execution: Optional[float] = None
    
    # Availability
    is_available: bool = True
    last_heartbeat: float = Field(default_factory=time.time)


class A2AAgentProfile(BaseModel):
    """Profile information for an A2A agent."""
    
    agent_id: str
    name: str
    description: str
    
    # Agent metadata
    version: str = "1.0.0"
    agent_type: str = "reactive"
    
    # Capabilities
    capabilities: List[A2AAgentCapability] = Field(default_factory=list)
    
    # Communication preferences
    preferred_protocols: List[str] = Field(default_factory=list)
    max_message_size: int = 1048576  # 1MB
    supports_streaming: bool = False
    
    # Resource information
    max_concurrent_tasks: int = 1
    current_load: float = 0.0
    
    # Status
    is_online: bool = True
    last_seen: float = Field(default_factory=time.time)
    uptime_seconds: float = 0.0