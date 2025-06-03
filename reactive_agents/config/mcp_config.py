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

    @classmethod
    def create_from_dict(cls, config_dict: Dict[str, Any]) -> "MCPConfig":
        """Create an MCPConfig instance from a dictionary with validation"""
        servers_dict = {}
        for server_name, server_info in config_dict.get("mcpServers", {}).items():
            # Handle Docker configuration if present
            docker_config = None
            if "docker" in server_info:
                docker_config = DockerConfig(**server_info["docker"])

            # Create server configuration with validation
            servers_dict[server_name] = MCPServerConfig(
                command=server_info["command"],
                args=server_info.get("args", []),
                env=server_info.get("env", {}),
                working_dir=server_info.get("working_dir"),
                docker=docker_config,
                enabled=server_info.get("enabled", True),
            )

        return cls(servers=servers_dict)

    def merge_config(self, other_config: "MCPConfig") -> "MCPConfig":
        """Merge another config into this one, with the other config taking precedence"""
        merged_servers = {**self.servers}

        for name, server in other_config.servers.items():
            if name in merged_servers:
                # Update existing server with new values, preserving existing ones if not specified
                current_dict = merged_servers[name].dict()
                update_dict = server.dict()
                merged_dict = {**current_dict, **update_dict}
                merged_servers[name] = MCPServerConfig(**merged_dict)
            else:
                # Add new server configuration
                merged_servers[name] = server

        return MCPConfig(
            servers=merged_servers,
            default_docker_config=other_config.default_docker_config
            or self.default_docker_config,
        )

    # @validator("servers")
    # def validate_server_names(cls, servers):
    #     """Validate server names and configurations"""
    #     for name, config in servers.items():
    #         if not name.isidentifier():
    #             raise ValueError(
    #                 f"Server name '{name}' must be a valid Python identifier"
    #             )

    #         if config.command == "docker" and not config.docker:
    #             raise ValueError(
    #                 f"Server '{name}' uses docker command but has no docker configuration"
    #             )

    #     return servers


def load_server_config(
    config_paths: Optional[List[str]] = None,
    env_configs: Optional[Dict[str, str]] = None,
) -> MCPConfig:
    """
    Load server configuration with support for multiple sources and environment variables

    Args:
        config_paths: List of paths to configuration files to load and merge
        env_configs: Dictionary of environment variable names and their corresponding config paths
    """
    # Start with base configuration
    config = MCPConfig(
        servers={
            "local": MCPServerConfig(
                command="python",
                args=["./agent_mcp/servers/server.py"],
                working_dir=Path.cwd(),
            )
        }
    )

    # Load configurations from provided paths
    if config_paths:
        for path in config_paths:
            if os.path.exists(path):
                try:
                    import json

                    with open(path) as f:
                        custom_config = MCPConfig.create_from_dict(json.load(f))
                        config = config.merge_config(custom_config)
                except Exception as e:
                    print(f"Error loading config from {path}: {str(e)}")

    # Load configurations from environment variables
    if env_configs:
        for env_var, default_path in env_configs.items():
            config_path = os.environ.get(env_var, default_path)
            if config_path and os.path.exists(config_path):
                try:
                    import json

                    with open(config_path) as f:
                        custom_config = MCPConfig.create_from_dict(json.load(f))
                        config = config.merge_config(custom_config)
                except Exception as e:
                    print(
                        f"Error loading config from {env_var} ({config_path}): {str(e)}"
                    )

    # Load default MCP servers if no other configurations were loaded
    if len(config.servers) <= 1:  # Only has local server
        config = config.merge_config(
            MCPConfig(
                servers={
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
        )

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
