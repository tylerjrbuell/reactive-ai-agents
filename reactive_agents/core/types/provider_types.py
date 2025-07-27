"""Types for LLM providers and external services."""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class ProviderType(Enum):
    """Types of providers."""
    
    LLM = "llm"
    EMBEDDING = "embedding"
    VECTOR_DB = "vector_db"
    EXTERNAL_API = "external_api"
    MCP_SERVER = "mcp_server"
    A2A_BRIDGE = "a2a_bridge"


class ProviderStatus(Enum):
    """Status of a provider."""
    
    AVAILABLE = "available"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"


class ModelCapability(Enum):
    """Capabilities of LLM models."""
    
    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    FUNCTION_CALLING = "function_calling"
    TOOL_USE = "tool_use"
    VISION = "vision"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    STREAMING = "streaming"
    EMBEDDINGS = "embeddings"
    FINE_TUNING = "fine_tuning"


class ResponseFormat(Enum):
    """Response formats for LLM completion."""
    
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"
    FUNCTION_CALL = "function_call"
    TOOL_CALLS = "tool_calls"


class CompletionMessage(BaseModel):
    """Message in a completion request/response."""
    
    content: str
    role: Optional[str] = None
    thinking: Optional[str] = None  # For models that support thinking
    tool_calls: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[str]] = None  # Base64 encoded images or URLs


class CompletionResponse(BaseModel):
    """Response from an LLM completion."""
    
    message: CompletionMessage
    model: str
    done: bool = True
    done_reason: Optional[str] = None
    
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Timing
    prompt_eval_duration: Optional[int] = None  # nanoseconds
    eval_duration: Optional[int] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    
    # Metadata
    created_at: Optional[str] = None
    context_length: Optional[int] = None
    
    @property
    def usage_summary(self) -> Dict[str, int]:
        """Get token usage summary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


class ModelInfo(BaseModel):
    """Information about an LLM model."""
    
    model_id: str
    name: str
    provider: str
    
    # Model characteristics
    context_length: int = 4096
    max_tokens: Optional[int] = None
    capabilities: List[ModelCapability] = Field(default_factory=list)
    
    # Model metadata
    version: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None
    
    # Performance characteristics
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    
    # Technical details
    temperature_range: tuple[float, float] = (0.0, 2.0)
    supports_streaming: bool = False
    supports_functions: bool = False
    supports_vision: bool = False
    
    # Availability
    is_available: bool = True
    deprecation_date: Optional[datetime] = None
    replacement_model: Optional[str] = None


class ProviderConfiguration(BaseModel):
    """Configuration for a provider."""
    
    provider_id: str
    provider_type: ProviderType
    
    # Connection settings
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: int = 30
    max_retries: int = 3
    
    # Rate limiting
    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    concurrent_requests: int = 1
    
    # Default settings
    default_model: Optional[str] = None
    default_temperature: float = 0.7
    default_max_tokens: Optional[int] = None
    
    # Provider-specific settings
    custom_headers: Dict[str, str] = Field(default_factory=dict)
    additional_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Health and monitoring
    health_check_enabled: bool = True
    health_check_interval: int = 300  # seconds
    
    # Features
    supports_streaming: bool = True
    supports_functions: bool = True
    enable_caching: bool = True
    
    # Security
    verify_ssl: bool = True
    proxy_url: Optional[str] = None


class ProviderHealth(BaseModel):
    """Health status of a provider."""
    
    provider_id: str
    status: ProviderStatus
    last_check: datetime = Field(default_factory=datetime.now)
    
    # Health metrics
    response_time_ms: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0
    
    # Usage statistics
    requests_today: int = 0
    tokens_used_today: int = 0
    cost_today: float = 0.0
    
    # Error information
    last_error: Optional[str] = None
    error_count: int = 0
    consecutive_errors: int = 0
    
    # Rate limiting info
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    
    # Availability
    uptime_percentage: float = 100.0
    maintenance_mode: bool = False
    
    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        return (
            self.status == ProviderStatus.CONNECTED and
            self.success_rate >= 0.95 and
            self.consecutive_errors < 3 and
            self.response_time_ms < 30000  # 30 seconds
        )


class ProviderUsage(BaseModel):
    """Usage statistics for a provider."""
    
    provider_id: str
    period_start: datetime
    period_end: datetime
    
    # Request statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Timing statistics
    total_duration_ms: float = 0.0
    average_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    
    # Cost information
    total_cost: float = 0.0
    cost_per_token: float = 0.0
    
    # Model breakdown
    model_usage: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Error breakdown
    error_types: Dict[str, int] = Field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return 1.0 - self.success_rate
    
    @property
    def cost_per_request(self) -> float:
        """Calculate average cost per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_cost / self.total_requests


class ProviderCapabilities(BaseModel):
    """Capabilities of a provider."""
    
    provider_id: str
    provider_type: ProviderType
    
    # Supported features
    supported_models: List[str] = Field(default_factory=list)
    supported_formats: List[ResponseFormat] = Field(default_factory=list)
    supported_capabilities: List[ModelCapability] = Field(default_factory=list)
    
    # Technical capabilities
    max_context_length: int = 4096
    max_tokens_per_request: Optional[int] = None
    supports_streaming: bool = False
    supports_batching: bool = False
    
    # Rate limits
    max_requests_per_minute: Optional[int] = None
    max_tokens_per_minute: Optional[int] = None
    max_concurrent_requests: int = 1
    
    # Cost information
    pricing_model: str = "pay_per_use"  # pay_per_use, subscription, free
    free_tier_limits: Optional[Dict[str, int]] = None
    
    # Geographic and legal
    available_regions: List[str] = Field(default_factory=list)
    data_residency: Optional[str] = None
    compliance_certifications: List[str] = Field(default_factory=list)


class ProviderEvent(BaseModel):
    """Event related to provider operations."""
    
    event_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Event details
    provider_id: str
    event_type: str  # connection, disconnection, error, rate_limit, etc.
    severity: str = "info"  # debug, info, warning, error, critical
    
    # Event data
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    
    # Context
    model_used: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Performance data
    response_time_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


class MCPServerConfig(BaseModel):
    """Configuration for a Model Context Protocol server."""
    
    server_id: str
    name: str
    
    # Connection details
    command: str = Field(..., description="Command to run the server")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    
    # Server configuration
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Server inputs")
    working_dir: Optional[str] = Field(default=None, description="Working directory")
    
    # Docker configuration (if using Docker)
    docker_config: Optional[Dict[str, Any]] = None
    
    # Management settings
    enabled: bool = Field(default=True, description="Whether this server is enabled")
    auto_restart: bool = True
    max_restart_attempts: int = 3
    
    # Timeout settings
    startup_timeout: int = 30
    request_timeout: int = 60
    shutdown_timeout: int = 10
    
    # Health monitoring
    health_check_interval: int = 60
    heartbeat_interval: int = 30


class ExternalServiceConfig(BaseModel):
    """Configuration for external services."""
    
    service_id: str
    service_type: str
    name: str
    
    # Connection details
    endpoint: str
    api_key: Optional[str] = None
    authentication: Dict[str, Any] = Field(default_factory=dict)
    
    # Request settings
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Rate limiting
    rate_limit: Optional[int] = None
    burst_limit: Optional[int] = None
    
    # Features
    supports_async: bool = False
    supports_streaming: bool = False
    supports_batching: bool = False
    
    # Data handling
    request_format: str = "json"
    response_format: str = "json"
    compression: bool = False
    
    # Monitoring
    logging_enabled: bool = True
    metrics_enabled: bool = True
    
    # Security
    verify_ssl: bool = True
    allowed_domains: List[str] = Field(default_factory=list)
    
    # Custom settings
    custom_headers: Dict[str, str] = Field(default_factory=dict)
    custom_params: Dict[str, Any] = Field(default_factory=dict)