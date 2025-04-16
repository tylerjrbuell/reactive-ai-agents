from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, DirectoryPath, validator
import dotenv
import os
from pathlib import Path

dotenv.load_dotenv()


class DockerConfig(BaseModel):
    """Docker-specific configuration options"""

    host: str = Field(
        default="unix:///var/run/docker.sock", description="Docker host URL"
    )
    network: Optional[str] = Field(default=None, description="Docker network to use")
    extra_mounts: List[str] = Field(
        default_factory=list, description="Additional volume mounts"
    )
    extra_env: Dict[str, str] = Field(
        default_factory=dict, description="Additional environment variables"
    )


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server"""

    command: str = Field(..., description="Command to run the server")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )
    working_dir: Optional[DirectoryPath] = Field(
        default=None, description="Working directory"
    )
    docker: Optional[DockerConfig] = Field(
        default=None, description="Docker-specific configuration"
    )
    enabled: bool = Field(default=True, description="Whether this server is enabled")

    @validator("working_dir", pre=True)
    def validate_working_dir(cls, v):
        if v:
            # Support environment variable expansion
            return os.path.expandvars(str(v))
        return v


class MCPConfig(BaseModel):
    """Root configuration for all MCP servers"""

    servers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict, description="Map of server names to their configurations"
    )
    default_docker_config: Optional[DockerConfig] = Field(
        default=None,
        description="Default Docker configuration for all Docker-based servers",
    )


def load_server_config() -> MCPConfig:
    """Load server configuration with environment variable support"""
    config = MCPConfig(
        servers={
            "local": MCPServerConfig(
                command="python",
                args=["./agent_mcp/servers/server.py"],
                working_dir=Path.cwd(),
            ),
            "time": MCPServerConfig(
                command="docker",
                args=["run", "--name", "mcp-time", "-i", "--rm", "mcp/time"],
                docker=DockerConfig(),
            ),
            "filesystem": MCPServerConfig(
                command="docker",
                args=[
                    "run",
                    "--name",
                    "mcp-filesystem",
                    "-i",
                    "--rm",
                    "--mount",
                    f"type=bind,src={os.path.expandvars('$PWD')},dst=/projects",
                    "--workdir",
                    "/projects",
                    "mcp/filesystem",
                    "/projects",
                ],
                docker=DockerConfig(),
            ),
            "sqlite": MCPServerConfig(
                command="docker",
                args=[
                    "run",
                    "--name",
                    "mcp-sqlite",
                    "--rm",
                    "-i",
                    "-v",
                    f"{os.path.expandvars('$PWD')}/agent_mcp:/mcp",
                    "mcp/sqlite",
                    "--db-path",
                    "/mcp/agent.db",
                ],
                docker=DockerConfig(),
            ),
            "playwright": MCPServerConfig(
                command="npx",
                args=["-y", "@executeautomation/playwright-mcp-server"],
            ),
            "brave-search": MCPServerConfig(
                command="docker",
                args=[
                    "run",
                    "--name",
                    "mcp-brave-search",
                    "-i",
                    "--rm",
                    "-e",
                    "BRAVE_API_KEY",
                    "mcp/brave-search",
                ],
                env={"BRAVE_API_KEY": os.environ.get("BRAVE_API_KEY", "")},
                docker=DockerConfig(),
            ),
            "duckduckgo": MCPServerConfig(
                command="docker",
                args=[
                    "run",
                    "--name",
                    "mcp-duckduckgo",
                    "-i",
                    "--rm",
                    "mcp/duckduckgo",
                ],
                docker=DockerConfig(),
            ),
        }
    )

    # Load custom config from environment if specified
    custom_config_path = os.environ.get("MCP_CONFIG_PATH")
    if custom_config_path and os.path.exists(custom_config_path):
        import json

        with open(custom_config_path) as f:
            custom_config = json.load(f)
            # Merge custom config with default config
            for server_name, server_config in custom_config.get("servers", {}).items():
                if server_name in config.servers:
                    # Update existing server config
                    config.servers[server_name] = MCPServerConfig(
                        **{**config.servers[server_name].dict(), **server_config}
                    )
                else:
                    # Add new server config
                    config.servers[server_name] = MCPServerConfig(**server_config)

    return config


def get_mcp_servers(filter: Optional[List[str]] = None) -> Dict[str, Any]:
    """Get filtered server configurations in legacy format for backward compatibility"""
    config = load_server_config()

    # Convert to legacy format
    legacy_config = {
        "mcpServers": {
            name: {"command": server.command, "args": server.args, "env": server.env}
            for name, server in config.servers.items()
            if server.enabled and (not filter or name in filter)
        }
    }

    return legacy_config
