"""Types for context management and conversation handling."""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from datetime import datetime


class MessageRole(Enum):
    """Roles for messages in conversation context."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class ContextScope(Enum):
    """Scope levels for context information."""
    
    GLOBAL = "global"
    SESSION = "session"
    TASK = "task"
    ITERATION = "iteration"
    TEMPORARY = "temporary"


class ContextPriority(Enum):
    """Priority levels for context elements."""
    
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class ContextStrategy(Enum):
    """Strategies for context management."""
    
    PRESERVE_ALL = "preserve_all"
    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    PRIORITY_BASED = "priority_based"
    RELEVANCE_BASED = "relevance_based"
    ADAPTIVE = "adaptive"


class ConversationMessage(BaseModel):
    """A single message in a conversation."""
    
    message_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Message content
    role: MessageRole
    content: str
    name: Optional[str] = None  # For function/tool messages
    
    # Metadata
    tokens: Optional[int] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    
    # Tool-related
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    # Context management
    priority: ContextPriority = ContextPriority.NORMAL
    scope: ContextScope = ContextScope.SESSION
    should_summarize: bool = True
    pinned: bool = False  # Pinned messages are never pruned
    
    # Relationships
    reply_to: Optional[str] = None
    thread_id: Optional[str] = None
    
    # Processing metadata
    processing_time_ms: Optional[float] = None
    cost: Optional[float] = None
    
    def token_count(self) -> int:
        """Get estimated token count for this message."""
        if self.tokens is not None:
            return self.tokens
        
        # Simple estimation if not provided
        return len(self.content.split()) + 10  # Rough estimate


class ContextWindow(BaseModel):
    """A window of conversation context."""
    
    window_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Window content
    messages: List[ConversationMessage] = Field(default_factory=list)
    summary: Optional[str] = None
    
    # Window metadata
    total_tokens: int = 0
    max_tokens: int = 4000
    priority: ContextPriority = ContextPriority.NORMAL
    scope: ContextScope = ContextScope.SESSION
    
    # Management
    is_summarized: bool = False
    is_archived: bool = False
    last_accessed: datetime = Field(default_factory=datetime.now)
    
    # Relationships
    parent_window_id: Optional[str] = None
    child_window_ids: List[str] = Field(default_factory=list)
    
    def add_message(self, message: ConversationMessage) -> None:
        """Add a message to this window."""
        self.messages.append(message)
        self.total_tokens += message.token_count()
        self.last_accessed = datetime.now()
    
    def is_full(self) -> bool:
        """Check if window is at capacity."""
        return self.total_tokens >= self.max_tokens
    
    def get_token_count(self) -> int:
        """Get total token count for this window."""
        return sum(msg.token_count() for msg in self.messages)


class ContextSummary(BaseModel):
    """Summary of conversation context."""
    
    summary_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Summary content
    text: str
    key_points: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    
    # Source information
    source_window_id: str
    source_message_count: int
    source_token_count: int
    
    # Summary metadata
    compression_ratio: float = 0.0  # tokens_saved / original_tokens
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    model_used: Optional[str] = None
    
    # Context preservation
    critical_information: List[str] = Field(default_factory=list)
    tool_usage_summary: List[str] = Field(default_factory=list)
    decision_points: List[str] = Field(default_factory=list)


class ContextState(BaseModel):
    """Current state of context management."""
    
    state_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Context metrics
    total_messages: int = 0
    total_tokens: int = 0
    active_windows: int = 0
    archived_windows: int = 0
    summaries_count: int = 0
    
    # Token distribution
    system_tokens: int = 0
    user_tokens: int = 0
    assistant_tokens: int = 0
    tool_tokens: int = 0
    
    # Management statistics
    pruning_events: int = 0
    summarization_events: int = 0
    last_pruning: Optional[datetime] = None
    last_summarization: Optional[datetime] = None
    
    # Performance metrics
    average_response_time_ms: float = 0.0
    context_efficiency_score: float = 0.0
    
    # Strategy information
    current_strategy: ContextStrategy = ContextStrategy.ADAPTIVE
    strategy_effectiveness: float = 0.8


class ContextConfiguration(BaseModel):
    """Configuration for context management."""
    
    # Token limits
    max_total_tokens: int = 10000
    max_window_tokens: int = 4000
    soft_limit_tokens: int = 8000  # Start managing at this point
    
    # Message limits
    max_messages_per_window: int = 50
    max_total_messages: int = 200
    
    # Summarization settings
    enable_summarization: bool = True
    summarization_threshold: int = 2000  # tokens
    summarization_target_ratio: float = 0.3  # target compression ratio
    preserve_recent_messages: int = 10
    
    # Pruning settings
    enable_pruning: bool = True
    pruning_strategy: ContextStrategy = ContextStrategy.ADAPTIVE
    preserve_critical_messages: bool = True
    preserve_pinned_messages: bool = True
    
    # Archival settings
    archive_old_windows: bool = True
    archive_threshold_hours: int = 24
    max_archived_windows: int = 100
    
    # Performance settings
    background_processing: bool = True
    lazy_summarization: bool = True
    cache_summaries: bool = True
    
    # Quality settings
    min_summary_confidence: float = 0.7
    retry_failed_summarizations: bool = True
    max_summarization_attempts: int = 3


@dataclass
class ContextMetrics:
    """Metrics for context management performance."""
    
    # Token efficiency
    tokens_saved: int = 0
    tokens_processed: int = 0
    compression_ratio: float = 0.0
    
    # Time efficiency
    total_processing_time_ms: float = 0.0
    average_operation_time_ms: float = 0.0
    
    # Operation counts
    summarizations_performed: int = 0
    pruning_operations: int = 0
    window_creations: int = 0
    message_additions: int = 0
    
    # Quality metrics
    summary_confidence_avg: float = 0.0
    context_relevance_score: float = 0.0
    information_retention_score: float = 0.0
    
    # Error tracking
    failed_operations: int = 0
    last_error: Optional[str] = None
    error_rate: float = 0.0
    
    def calculate_efficiency(self) -> float:
        """Calculate overall context management efficiency."""
        if self.tokens_processed == 0:
            return 0.0
        
        token_efficiency = self.tokens_saved / self.tokens_processed
        quality_efficiency = (
            self.summary_confidence_avg * 0.4 +
            self.context_relevance_score * 0.3 +
            self.information_retention_score * 0.3
        )
        error_penalty = self.error_rate * 0.2
        
        return max(0.0, min(1.0, 
            token_efficiency * 0.4 + 
            quality_efficiency * 0.6 - 
            error_penalty
        ))


class ContextNudge(BaseModel):
    """A nudge or hint to guide conversation direction."""
    
    nudge_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Nudge content
    message: str
    category: str = "general"  # general, tool_usage, task_completion, etc.
    
    # Priority and timing
    priority: ContextPriority = ContextPriority.NORMAL
    expires_at: Optional[datetime] = None
    trigger_condition: Optional[str] = None
    
    # Usage tracking
    times_shown: int = 0
    times_followed: int = 0
    effectiveness_score: float = 0.0
    
    # Metadata
    source: str = "system"  # system, user, agent, strategy
    context_when_created: Dict[str, Any] = Field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if nudge has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def should_show(self, context: Dict[str, Any]) -> bool:
        """Determine if nudge should be shown in current context."""
        if self.is_expired():
            return False
        
        if self.trigger_condition:
            # Simple condition evaluation - could be enhanced
            return self.trigger_condition in str(context)
        
        return True