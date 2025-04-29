from typing import Any, Awaitable, Dict, List, Protocol
from mcp import Tool as MCPTool
from mcp.types import TextContent
from tools.base import Tool
from agent_mcp.client import MCPClient


class ToolProtocol(Protocol):
    """Protocol defining the interface that all tools must implement"""

    name: str
    tool_definition: Dict[str, Any]

    async def use(self, params: dict) -> Any: ...


class ToolResult:
    """Standardized tool result wrapper"""

    def __init__(self, result: Any):
        self.raw_result = result

    def to_list(self) -> List[str]:
        """Convert the result to a list format"""
        if isinstance(self.raw_result, list):
            return [str(item) for item in self.raw_result]
        return [str(self.raw_result)]

    def to_string(self) -> str:
        """Convert the result to a string format"""
        if isinstance(self.raw_result, list):
            return (
                str(self.raw_result[0])
                if len(self.raw_result) == 1
                else str(self.raw_result)
            )
        return str(self.raw_result)

    @classmethod
    def wrap(cls, result: Any) -> "ToolResult":
        """Wrap any result in a ToolResult"""
        return cls(result)


class MCPToolWrapper(Tool):
    """Wrapper to adapt MCP tools to match our ToolProtocol interface"""

    def __init__(self, mcp_tool: MCPTool, client: MCPClient):
        self.mcp_tool = mcp_tool
        self.mcp_client = client
        self.name = mcp_tool.name
        self.tool_definition = {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": mcp_tool.inputSchema,
            },
        }

    async def use(self, params: dict) -> ToolResult:
        """Execute the MCP tool through its client"""
        try:
            result = await self.mcp_client.call_tool(
                tool_name=self.name,
                params=params,
            )
            return ToolResult(
                [r.text if type(r) is TextContent else r for r in result.content]
            )
        except Exception as e:
            return ToolResult(f"Tool Error: {e}")
