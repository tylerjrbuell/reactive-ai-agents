from typing import List, Optional, Any, Dict
from contextlib import AsyncExitStack
import os
import uuid
import asyncio
from pydantic import AnyUrl
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult
from config.mcp_config import (
    load_server_config,
    MCPConfig,
    MCPServerConfig,
    DockerConfig,
)


class MCPClient:
    def __init__(
        self,
        config_file: str = "config/mcp.json",
        server_filter: Optional[List[str]] = None,
        server_config: Optional[Dict[str, Any]] = None,
    ):
        # Initialize session and client objects
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: List[Tool] = []
        self.tool_signatures: List[Dict[str, Any]] = []
        self.server_tools: Dict[str, List[Tool]] = {}
        self.exit_stack = AsyncExitStack()
        self.server_params: Optional[StdioServerParameters] = None
        self._closed = False
        self.server_filter = server_filter
        self.instance_id = str(uuid.uuid4())[:8]
        self.config: Optional[MCPConfig] = None
        self.config_file = config_file
        self.server_config = server_config

    def _prepare_docker_args(
        self, server_name: str, server_config: MCPServerConfig
    ) -> List[str]:
        """Prepare Docker arguments with any custom configuration"""
        args = list(server_config.args)  # Convert to list to allow modification

        if server_config.docker:
            # Modify container name for better identification
            if "--name" in args:
                name_index = args.index("--name") + 1
                if name_index < len(args):
                    args[name_index] = f"{args[name_index]}-{self.instance_id}"

            # Add network if specified
            if server_config.docker.network:
                args.extend(["--network", server_config.docker.network])

            # Add any extra mount points
            for mount in server_config.docker.extra_mounts:
                args.extend(["--mount", mount])

            # Add any extra environment variables
            for key, value in server_config.docker.extra_env.items():
                args.extend(["-e", f"{key}={value}"])

        return args

    def _prepare_environment(self, server_config: MCPServerConfig) -> Dict[str, str]:
        """Prepare environment variables for the server"""
        env = {**os.environ}  # Start with current environment

        # Add server-specific environment variables
        env.update(server_config.env)

        # Add Docker-specific environment variables if applicable
        if server_config.docker and server_config.docker.extra_env:
            env.update(server_config.docker.extra_env)

        return env

    async def connect_to_servers(self):
        """Connect to a group of MCP servers with unique container names"""
        if self.server_config:
            # Use provided server configuration
            servers_dict = {}
            for server_name, server_info in self.server_config.get(
                "mcpServers", {}
            ).items():
                docker_config = None
                if "docker" in server_info:
                    docker_config = DockerConfig(
                        network=server_info["docker"].get("network"),
                        extra_mounts=server_info["docker"].get("extra_mounts", []),
                        extra_env=server_info["docker"].get("extra_env", {}),
                    )

                servers_dict[server_name] = MCPServerConfig(
                    command=server_info["command"],
                    args=server_info.get("args", []),
                    env=server_info.get("env", {}),
                    working_dir=server_info.get("working_dir"),
                    docker=docker_config,
                    enabled=server_info.get("enabled", True),
                )
            self.config = MCPConfig(servers=servers_dict)
        else:
            # Load from config file
            self.config = load_server_config()

        for server_name, server_config in self.config.servers.items():
            if self._closed or not server_config.enabled:
                continue

            if self.server_filter and server_name not in self.server_filter:
                continue

            # Skip if already connected
            if server_name in self.sessions:
                continue

            try:
                # Prepare command arguments
                args = (
                    self._prepare_docker_args(server_name, server_config)
                    if server_config.command == "docker"
                    else server_config.args
                )

                # Prepare environment
                env = self._prepare_environment(server_config)

                # Create server parameters
                self.server_params = StdioServerParameters(
                    command=server_config.command,
                    args=args,
                    env=env,
                    cwd=server_config.working_dir,
                )

                # Connect to server
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(self.server_params)
                )
                self.stdio, self.write = stdio_transport
                session = await self.exit_stack.enter_async_context(
                    ClientSession(self.stdio, self.write)
                )

                # Initialize and store session
                await session.initialize()
                self.sessions[server_name] = session
                self.server_tools[server_name] = (await session.list_tools()).tools

            except Exception as e:
                print(f"Failed to connect to server {server_name}: {str(e)}")
                # Clean up any partial connections
                if server_name in self.sessions:
                    del self.sessions[server_name]
                    del self.server_tools[server_name]

    async def initialize(self):
        """Initialize the client and connect to servers"""
        await self.connect_to_servers()
        await self.get_tools()
        return self

    async def get_tools(self):
        """Get all available tools from connected servers"""
        if not self._closed:
            self.tools = []
            self.tool_signatures = []

            for session in self.sessions.values():
                try:
                    tools = (await session.list_tools()).tools
                    self.tools.extend(tools)

                    # Create tool signatures
                    for tool in tools:
                        self.tool_signatures.append(
                            {
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": tool.inputSchema,
                                },
                            }
                        )
                except Exception as e:
                    print(f"Error getting tools from session: {str(e)}")

        return self.tools

    async def call_tool(self, tool_name: str, params: dict) -> CallToolResult:
        """Call a tool on the appropriate server"""
        if self._closed:
            raise RuntimeError("Client is closed")

        # Find which server has the tool
        server_name = next(
            (
                server
                for server, tools in self.server_tools.items()
                if tool_name in [tool.name for tool in tools]
            ),
            None,
        )

        if not server_name:
            raise ValueError(f"Tool {tool_name} not found in any connected server")

        return await self.sessions[server_name].call_tool(
            name=tool_name, arguments=params
        )

    def get_session(self, server_name: str) -> ClientSession:
        """Get a specific server session"""
        if self._closed:
            raise RuntimeError("Client is closed")
        if server_name not in self.sessions:
            raise KeyError(f"No session found for server {server_name}")
        return self.sessions[server_name]

    async def get_resource(self, server: str, uri: AnyUrl):
        """Get a resource from a specific server"""
        if self._closed:
            raise RuntimeError("Client is closed")
        if server not in self.sessions:
            raise KeyError(f"No session found for server {server}")
        return await self.sessions[server].read_resource(uri)

    async def close(self):
        """Clean up all resources"""
        if not self._closed:
            self._closed = True

            # Stop Docker containers first
            if self.config:
                for server_name, server_config in self.config.servers.items():
                    if server_config.command == "docker":
                        container_name = None
                        if "--name" in server_config.args:
                            name_index = server_config.args.index("--name") + 1
                            if name_index < len(server_config.args):
                                container_name = f"{server_config.args[name_index]}-{self.instance_id}"

                        if container_name:
                            try:
                                import subprocess

                                subprocess.run(
                                    ["docker", "rm", "-f", container_name],
                                    timeout=5,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                )
                            except Exception as e:
                                print(
                                    f"Error cleaning up Docker container {container_name}: {e}"
                                )

            # Clear sessions and server tools
            self.sessions.clear()
            self.server_tools.clear()

            # Close the exit stack
            try:
                await asyncio.wait_for(self.exit_stack.aclose(), timeout=5)
            except asyncio.TimeoutError:
                print("MCPClient cleanup timed out after 5 seconds")
            except Exception as e:
                print(f"Error during MCPClient cleanup: {e}")
            finally:
                self.exit_stack = AsyncExitStack()

    async def __aenter__(self) -> "MCPClient":
        """Async context manager support"""
        return await self.initialize()

    def __await__(self):
        """Make MCPClient awaitable"""
        return self.initialize().__await__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup on context exit"""
        await self.close()
