"""Types for plugin management and extension system."""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class PluginType(Enum):
    """Types of plugins supported by the framework."""
    
    STRATEGY = "strategy"
    TOOL = "tool"
    PROVIDER = "provider"
    MEMORY = "memory"
    PROMPT = "prompt"
    AGENT = "agent"
    WORKFLOW = "workflow"


class PluginStatus(Enum):
    """Status of a plugin."""
    
    AVAILABLE = "available"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"
    OUTDATED = "outdated"


class PluginPriority(Enum):
    """Priority levels for plugins."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class PluginMetadata(BaseModel):
    """Metadata for a plugin."""
    
    plugin_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str
    description: str = ""
    
    # Plugin classification
    plugin_type: PluginType
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # Author and source information
    author: str = ""
    author_email: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: str = "MIT"
    
    # Dependencies and requirements
    dependencies: List[str] = Field(default_factory=list)
    python_requires: str = ">=3.8"
    framework_version: str = ""
    
    # Plugin configuration
    config_schema: Optional[Dict[str, Any]] = None
    default_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Runtime information
    entry_point: str = ""
    module_path: str = ""
    class_name: str = ""
    
    # Lifecycle hooks
    on_load: Optional[str] = None
    on_unload: Optional[str] = None
    on_enable: Optional[str] = None
    on_disable: Optional[str] = None
    
    # Validation and safety
    checksum: Optional[str] = None
    signature: Optional[str] = None
    trusted: bool = False
    
    # Status tracking
    status: PluginStatus = PluginStatus.AVAILABLE
    load_time: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    
    # Performance metrics
    average_load_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None


class PluginConfiguration(BaseModel):
    """Configuration for a plugin instance."""
    
    plugin_id: str
    instance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Configuration data
    config: Dict[str, Any] = Field(default_factory=dict)
    overrides: Dict[str, Any] = Field(default_factory=dict)
    
    # Instance settings
    enabled: bool = True
    priority: PluginPriority = PluginPriority.NORMAL
    auto_start: bool = True
    
    # Resource limits
    max_memory_mb: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    timeout_seconds: Optional[float] = None
    
    # Security settings
    sandbox_enabled: bool = True
    allowed_permissions: List[str] = Field(default_factory=list)
    denied_permissions: List[str] = Field(default_factory=list)
    
    # Lifecycle settings
    restart_on_failure: bool = True
    max_restart_attempts: int = 3
    
    # Monitoring
    logging_level: str = "INFO"
    metrics_enabled: bool = True
    health_check_interval: int = 60  # seconds


class PluginInterface(BaseModel):
    """Interface definition for a plugin."""
    
    interface_id: str
    name: str
    version: str
    description: str = ""
    
    # Interface specification
    methods: List[Dict[str, Any]] = Field(default_factory=list)
    properties: List[Dict[str, Any]] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Compatibility
    compatible_versions: List[str] = Field(default_factory=list)
    breaking_changes: List[str] = Field(default_factory=list)
    
    # Documentation
    documentation_url: Optional[str] = None
    examples: List[Dict[str, Any]] = Field(default_factory=list)


class PluginManifest(BaseModel):
    """Complete manifest for a plugin package."""
    
    # Plugin metadata
    metadata: PluginMetadata
    
    # Interface compliance
    implements: List[str] = Field(default_factory=list)  # Interface IDs
    provides: List[str] = Field(default_factory=list)    # Service names
    requires: List[str] = Field(default_factory=list)    # Required services
    
    # Asset information
    assets: List[Dict[str, Any]] = Field(default_factory=list)
    resources: Dict[str, str] = Field(default_factory=dict)
    
    # Installation and packaging
    install_requires: List[str] = Field(default_factory=list)
    extras_require: Dict[str, List[str]] = Field(default_factory=dict)
    package_data: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Testing and quality
    test_command: Optional[str] = None
    test_dependencies: List[str] = Field(default_factory=list)
    quality_gates: Dict[str, Any] = Field(default_factory=dict)


class PluginRegistry(BaseModel):
    """Registry for managing plugins."""
    
    registry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "default"
    
    # Plugin storage
    plugins: Dict[str, PluginMetadata] = Field(default_factory=dict)
    configurations: Dict[str, PluginConfiguration] = Field(default_factory=dict)
    interfaces: Dict[str, PluginInterface] = Field(default_factory=dict)
    
    # Registry settings
    auto_discovery: bool = True
    discovery_paths: List[str] = Field(default_factory=list)
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    
    # Security settings
    require_signatures: bool = False
    trusted_sources: List[str] = Field(default_factory=list)
    allowed_plugin_types: List[PluginType] = Field(default_factory=list)
    
    # Performance settings
    max_concurrent_loads: int = 5
    load_timeout_seconds: int = 30
    
    # Monitoring
    last_scan: Optional[datetime] = None
    scan_interval_seconds: int = 300
    health_check_enabled: bool = True


class PluginEvent(BaseModel):
    """Event related to plugin lifecycle."""
    
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Event details
    event_type: str  # loaded, unloaded, enabled, disabled, error, etc.
    plugin_id: str
    plugin_name: str
    
    # Event data
    data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    
    # Context
    registry_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class PluginHealthStatus(BaseModel):
    """Health status of a plugin."""
    
    plugin_id: str
    status: PluginStatus
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Health metrics
    is_responsive: bool = True
    response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Error information
    error_count: int = 0
    last_error: Optional[str] = None
    
    # Performance metrics
    throughput_per_second: float = 0.0
    average_execution_time_ms: float = 0.0
    
    # Additional metrics
    custom_metrics: Dict[str, float] = Field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if plugin is healthy."""
        return (
            self.status in [PluginStatus.LOADED, PluginStatus.ACTIVE] and
            self.is_responsive and
            self.response_time_ms < 5000 and  # 5 seconds
            self.memory_usage_mb < 1000  # 1GB
        )