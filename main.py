import json
from agent_mcp.client import MCPClient
from agents.react_agent import ReactAgent, ReactAgentConfig
import asyncio
import dotenv
import warnings
import tracemalloc
from typing import Any, Dict, Awaitable, Callable
from pydantic import PydanticDeprecatedSince211

warnings.simplefilter("ignore", ResourceWarning)
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince211)
tracemalloc.start()
dotenv.load_dotenv()


async def main():
    agent = None
    try:
        task = "Research the death of Chris Farley and add it to a table called celebrities_deaths(name, cause, date)."
        mcp_client = await MCPClient(
            server_filter=["brave-search", "sqlite", "time"]
        ).initialize()

        async def confirmation_callback(tool_name: str, params: Dict[str, Any]) -> bool:
            print(f"\n--- Tool Confirmation Request ---")
            print(f"Tool: {tool_name}")
            print(f"Parameters: {json.dumps(params, indent=2)}")
            user_input = (
                input("Proceed with this tool execution? (y/n) [y]: ").lower().strip()
            )
            return user_input == "y" or user_input == ""

        agent_config = ReactAgentConfig(
            agent_name="Task Agent",
            role="Task Executor",
            provider_model_name="ollama:cogito:14b",
            mcp_client=mcp_client,
            min_completion_score=1.0,
            instructions="Solve the given task as quickly as possible using the tools at your disposal.",
            max_iterations=10,
            reflect_enabled=True,
            log_level="debug",
            initial_task=None,
            tool_use_enabled=True,
            use_memory_enabled=True,
            collect_metrics_enabled=True,
            check_tool_feasibility=True,
            enable_caching=True,
            confirmation_callback=confirmation_callback,
            kwargs={},
        )

        agent = ReactAgent(config=agent_config)

        result_dict = await agent.run(initial_task=task)

        json_result = json.dumps(result_dict, indent=4, default=str)
        print("\n--- Agent Run Result ---")
        print(json_result)
        print("------------------------")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if agent:
            print("\nClosing Agent...")
            await agent.close()
            print("Agent closed.")


if __name__ == "__main__":
    asyncio.run(main())
