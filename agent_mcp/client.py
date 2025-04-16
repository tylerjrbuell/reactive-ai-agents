from typing import Awaitable, List, Optional, Any
from contextlib import AsyncExitStack
import os
import uuid
import asyncio
from pydantic import AnyUrl
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult


class MCPClient:
    def __init__(
        self,
        config_file: str = "config/mcp.json",
        server_filter: Optional[List[str]] = None,
    ):
        # Initialize session and client objects
        self.sessions: dict[str, ClientSession] = {}
        self.tools: List[Tool] = []
        self.tool_signatures: List[dict[str, Any]] = []
        self.server_tools: dict[str, List[Tool]] = {}
        self.exit_stack = AsyncExitStack()
        self.server_params: Optional[StdioServerParameters] = None
        self._closed = False
        self.server_filter = server_filter
        self.instance_id = str(uuid.uuid4())[:8]

    async def connect_to_servers(self):
        """Connect to a group of MCP servers with unique container names"""
        for server_name, server_config in self.config["mcpServers"].items():
            if self._closed:
                break

            command = server_config["command"]
            args = list(server_config["args"])  # Convert to list to allow modification
            env = server_config["env"] if len(server_config["env"].keys()) else {}

            # Modify container name for docker-based servers
            if command == "docker" and "--name" in args:
                name_index = args.index("--name") + 1
                if name_index < len(args):
                    args[name_index] = f"{args[name_index]}-{self.instance_id}"

            if not self.sessions.get(server_name):
                self.server_params = StdioServerParameters(
                    command=command, args=args, env={**os.environ, **env}
                )

                try:
                    stdio_transport = await self.exit_stack.enter_async_context(
                        stdio_client(self.server_params)
                    )
                    self.stdio, self.write = stdio_transport
                    session = await self.exit_stack.enter_async_context(
                        ClientSession(self.stdio, self.write)
                    )
                    self.sessions[server_name] = session
                    await session.initialize()
                    self.server_tools[server_name] = (await session.list_tools()).tools
                except Exception as e:
                    print(f"Failed to connect to server {server_name}: {str(e)}")
                    # Clean up any partial connections
                    if server_name in self.sessions:
                        del self.sessions[server_name]
                        del self.server_tools[server_name]

    async def initialize(self):
        # Import config at runtime to avoid circular dependency
        from config.mcp_config import get_mcp_servers

        self.config = get_mcp_servers(filter=self.server_filter)
        await self.connect_to_servers()
        await self.get_tools()
        return self

    async def get_tools(self):
        if not self._closed:
            self.tools = sum(
                [(await tool.list_tools()).tools for tool in self.sessions.values()], []
            )
            self.tool_signatures = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in self.tools
            ]
        return self.tools

    async def call_tool(self, tool_name: str, params: dict) -> CallToolResult:
        if self._closed:
            raise RuntimeError("Client is closed")

        server_name = [
            server
            for server, tools in self.server_tools.items()
            if tool_name in [tool.name for tool in tools]
        ][0]
        return await self.sessions[server_name].call_tool(name=tool_name, arguments=params)  # type: ignore

    def get_session(self, server_name: str):
        if self._closed:
            raise RuntimeError("Client is closed")
        return self.sessions[server_name]

    async def get_resource(self, server: str, uri: AnyUrl):
        if self._closed:
            raise RuntimeError("Client is closed")

        server_name = [
            server for server, tools in self.server_tools.items() if server == server
        ][0]
        return await self.sessions[server_name].read_resource(uri)

    async def close(self):
        """Clean up all resources"""
        if not self._closed:
            self._closed = True

            # First, stop and remove any Docker containers
            for server_name, server_config in self.config["mcpServers"].items():
                if server_config["command"] == "docker":
                    container_name = None
                    args = server_config["args"]
                    if "--name" in args:
                        name_index = args.index("--name") + 1
                        if name_index < len(args):
                            container_name = f"{args[name_index]}-{self.instance_id}"

                    if container_name:
                        try:
                            # Use subprocess to forcefully stop and remove the container
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

            # Then clear sessions and server tools
            self.sessions.clear()
            self.server_tools.clear()

            # Finally close the exit stack
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
