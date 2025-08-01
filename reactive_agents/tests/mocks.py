from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock
from reactive_agents.providers.external.client import MCPClient
from reactive_agents.core.types.provider_types import (
    CompletionResponse,
    CompletionMessage,
)
from reactive_agents.providers.llm.base import BaseModelProvider


class MockModelProvider(BaseModelProvider):
    def __init__(
        self,
        model: str = "mock:latest",
        options: Optional[Dict[str, Any]] = None,
        context: Optional[Any] = None,
    ):
        self.name = "mock"
        self.model = model

        # Create proper CompletionResponse objects with JSON content
        mock_response = CompletionResponse(
            message=CompletionMessage(
                content='{"completion": false, "completion_score": 0.3, "reasoning": "Task not yet complete", "missing_requirements": []}',
                role="assistant",
            ),
            model=model,
            total_duration=1000000,  # 1ms in nanoseconds
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

        # Store the mock response for the methods to return
        self._mock_response = mock_response

    async def get_completion(self, **kwargs):
        """Mock implementation of get_completion."""
        return self._mock_response

    async def get_chat_completion(self, **kwargs):
        """Mock implementation of get_chat_completion."""
        return self._mock_response

    async def validate_model(self, model_name: str) -> bool:
        """Mock implementation of validate_model."""
        return True


# Register the mock provider with the factory
BaseModelProvider.register_provider(MockModelProvider)


class MockMCPClient(MCPClient):
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
