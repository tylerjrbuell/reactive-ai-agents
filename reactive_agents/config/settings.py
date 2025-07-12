"""
Global Framework Settings

Centralized configuration management with type safety and validation.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import os
import json
from dataclasses import dataclass, field


def get_package_root() -> Path:
    """Get the reactive_agents package root directory"""
    # Try to find the reactive_agents package directory
    current_file = Path(__file__)

    # Navigate up from config/settings.py to find the reactive_agents root
    # config/settings.py -> config/ -> reactive_agents/
    package_root = current_file.parent.parent

    # Verify this is the reactive_agents package root
    if package_root.name == "reactive_agents" and (package_root / "core").exists():
        return package_root

    # Fallback: try to find it relative to current working directory
    cwd = Path.cwd()
    possible_roots = [
        cwd / "reactive_agents",
        cwd.parent / "reactive_agents",
    ]

    for root in possible_roots:
        if root.exists() and (root / "core").exists():
            return root

    # Last resort: use current directory
    return cwd


@dataclass
class DatabaseSettings:
    """Database and storage settings"""

    # Memory storage settings
    memory_backend: str = "json"
    memory_path: str = "storage/memory"

    # Vector memory settings
    vector_memory_persist_directory: str = "storage/memory/vector"

    # Vector database settings
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000
    chromadb_collection: str = "reactive_agents"

    # Redis settings (if used)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0


@dataclass
class LLMSettings:
    """LLM provider settings"""

    # Default model settings
    default_provider: str = "ollama"
    default_model: str = "qwen2:7b"

    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: int = 120

    # Groq settings
    groq_api_key: Optional[str] = None
    groq_model: str = "llama3-70b-8192"

    # OpenAI settings (for future use)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"


@dataclass
class AgentSettings:
    """Default agent settings"""

    # Reasoning settings
    default_strategy: str = "reactive"  # Changed from reflect_decide_act
    max_iterations: int = 10
    min_completion_score: float = 0.7

    # Tool settings
    tool_use_enabled: bool = True
    check_tool_feasibility: bool = True

    # Memory settings
    use_memory_enabled: bool = True
    enable_caching: bool = True

    # Workflow settings
    workflow_context_shared: bool = True


@dataclass
class PluginSettings:
    """Plugin system settings"""

    plugin_paths: List[str] = field(
        default_factory=lambda: ["plugins", "reactive_agents/plugins/examples"]
    )
    auto_load_plugins: bool = True
    plugin_timeout: int = 30


@dataclass
class LoggingSettings:
    """Logging configuration"""

    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    log_to_console: bool = True


@dataclass
class Settings:
    """Global framework settings"""

    # Core settings
    debug: bool = False
    version: str = "2.0.0"

    # Package root and storage paths
    package_root: Path = field(default_factory=get_package_root)
    storage_path: str = "storage"

    # Feature flags
    enable_a2a: bool = True
    enable_web_ui: bool = False
    enable_monitoring: bool = True

    # Sub-configurations
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    plugins: PluginSettings = field(default_factory=PluginSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    def __post_init__(self):
        """Post-initialization setup"""
        # Ensure storage path exists
        path = self.get_storage_path()
        path.mkdir(parents=True, exist_ok=True)

    def get_package_path(self, subpath: str = "") -> Path:
        """Get a path relative to the reactive_agents package root"""
        return self.package_root / subpath

    def get_storage_path(self, subpath: str = "") -> Path:
        """Get a path within the storage directory relative to package root"""
        path = self.package_root / "storage" / subpath
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_memory_path(self) -> Path:
        """Get the memory storage path relative to package root"""
        return self.get_storage_path("memory")

    def get_vector_memory_path(self) -> Path:
        """Get the vector memory persist directory relative to package root"""
        return self.get_storage_path("memory/vector")

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "debug": self.debug,
            "version": self.version,
            "package_root": str(self.package_root),
            "storage_path": self.storage_path,
            "enable_a2a": self.enable_a2a,
            "enable_web_ui": self.enable_web_ui,
            "enable_monitoring": self.enable_monitoring,
            "database": {
                "memory_backend": self.database.memory_backend,
                "memory_path": self.database.memory_path,
                "vector_memory_persist_directory": self.database.vector_memory_persist_directory,
                "chromadb_host": self.database.chromadb_host,
                "chromadb_port": self.database.chromadb_port,
                "chromadb_collection": self.database.chromadb_collection,
                "redis_host": self.database.redis_host,
                "redis_port": self.database.redis_port,
                "redis_db": self.database.redis_db,
            },
            "llm": {
                "default_provider": self.llm.default_provider,
                "default_model": self.llm.default_model,
                "ollama_host": self.llm.ollama_host,
                "ollama_timeout": self.llm.ollama_timeout,
                "groq_api_key": self.llm.groq_api_key,
                "groq_model": self.llm.groq_model,
                "openai_api_key": self.llm.openai_api_key,
                "openai_model": self.llm.openai_model,
            },
            "agent": {
                "default_strategy": self.agent.default_strategy,
                "max_iterations": self.agent.max_iterations,
                "min_completion_score": self.agent.min_completion_score,
                "tool_use_enabled": self.agent.tool_use_enabled,
                "check_tool_feasibility": self.agent.check_tool_feasibility,
                "use_memory_enabled": self.agent.use_memory_enabled,
                "enable_caching": self.agent.enable_caching,
                "workflow_context_shared": self.agent.workflow_context_shared,
            },
            "plugins": {
                "plugin_paths": self.plugins.plugin_paths,
                "auto_load_plugins": self.plugins.auto_load_plugins,
                "plugin_timeout": self.plugins.plugin_timeout,
            },
            "logging": {
                "log_level": self.logging.log_level,
                "log_format": self.logging.log_format,
                "log_file": self.logging.log_file,
                "log_to_console": self.logging.log_to_console,
            },
        }

    def save_to_file(self, file_path: str) -> None:
        """Save current settings to a JSON file"""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str) -> "Settings":
        """Load settings from a JSON file"""
        with open(file_path, "r") as f:
            data = json.load(f)

        # Reconstruct nested objects
        database = DatabaseSettings(**data.get("database", {}))
        llm = LLMSettings(**data.get("llm", {}))
        agent = AgentSettings(**data.get("agent", {}))
        plugins = PluginSettings(**data.get("plugins", {}))
        logging = LoggingSettings(**data.get("logging", {}))

        # Handle package_root - convert string back to Path
        package_root = (
            Path(data.get("package_root", ""))
            if data.get("package_root")
            else get_package_root()
        )

        return cls(
            debug=data.get("debug", False),
            version=data.get("version", "2.0.0"),
            package_root=package_root,
            storage_path=data.get("storage_path", "storage"),
            enable_a2a=data.get("enable_a2a", True),
            enable_web_ui=data.get("enable_web_ui", False),
            enable_monitoring=data.get("enable_monitoring", True),
            database=database,
            llm=llm,
            agent=agent,
            plugins=plugins,
            logging=logging,
        )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def initialize_settings(config_file: Optional[str] = None, **overrides) -> Settings:
    """Initialize settings with optional config file and overrides"""
    global _settings

    if config_file and os.path.exists(config_file):
        _settings = Settings.load_from_file(config_file)
    else:
        _settings = Settings()

    # Apply any overrides (simplified for now)
    if overrides:
        # This is a simplified override system
        for key, value in overrides.items():
            if hasattr(_settings, key):
                setattr(_settings, key, value)

    return _settings


def reset_settings() -> None:
    """Reset settings to default (useful for testing)"""
    global _settings
    _settings = None
