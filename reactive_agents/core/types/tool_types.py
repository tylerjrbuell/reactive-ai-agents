from typing import Any, Dict
from pydantic import BaseModel


class ToolCallFunction(BaseModel):
    name: str
    arguments: Dict[str, Any]


class ToolCall(BaseModel):
    function: ToolCallFunction
