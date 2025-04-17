from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock


class MockModelProvider:
    def __init__(self, name: str = "mock", model: str = "mock:latest"):
        self.name = name
        self.model = model
        self.get_completion = AsyncMock(return_value={"response": "mock response"})
        self.get_chat_completion = AsyncMock(
            return_value={"message": {"content": "mock response", "role": "assistant"}}
        )


class MockMCPClient:
    def __init__(self, server_filter: Optional[List[str]] = None):
        self.server_filter = server_filter or ["local"]
        self.tools = []
        self.tool_signatures = []
        self.server_tools = {"local": []}
        self._closed = False

    async def initialize(self):
        return self

    async def close(self):
        self._closed = True

    async def __aenter__(self):
        return await self.initialize()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def call_tool(self, tool_name: str, params: dict) -> Dict[str, Any]:
        return {"result": "mock tool result"}
