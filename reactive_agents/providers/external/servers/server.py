from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator
from mcp.server.fastmcp import FastMCP
import dotenv

dotenv.load_dotenv()

if __name__ == "__main__":

    @dataclass
    class AppContext:
        agent_memory = {"messages": []}

    @asynccontextmanager
    async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
        """Manage application lifecycle with type-safe context"""

        # Initialize on startup
        try:
            yield AppContext()
        finally:
            # Clean up on shutdown
            pass

    mcp = FastMCP("local-agent-mcp", lifespan=app_lifespan)

    @mcp.tool()
    def provide_task_answer(answer: str) -> str:
        """
        Provide a complete and final answer to the given task

        Args:
            answer (str): The complete final answer to the task

        Returns:
            str: The final answer
        """
        return answer

    mcp.run(transport="stdio")
