import json
from agent_mcp.client import MCPClient
from agents.react_agent import ReactAgent
import asyncio
import dotenv
import warnings
import tracemalloc
from typing import Any, Dict
from config.workflow import AgentConfig, WorkflowConfig, Workflow
from pydantic import PydanticDeprecatedSince211
from context.agent_context import AgentContext

warnings.simplefilter("ignore", ResourceWarning)
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince211)
tracemalloc.start()
dotenv.load_dotenv()


async def create_example_workflow() -> Workflow:
    workflow_config = WorkflowConfig()

    # Create a planner agent
    planner = AgentConfig(
        role="planner",
        model="ollama:cogito:14b",
        min_score=0.7,
        instructions="""You are a planning specialist. Your job is to break down a task into steps for the executor agent to follow and set them up for success to execute.
        Consider the following constraints:
        - Your deliverable is not to solve the task but rather to make a step by step plan for the executor agent.
        - You may use your tools to help give the executor agent the information it needs to complete the task.
        - You can only use tools that are available to you.
        - Assume the executor doesn't have research ability only execution ability.
        - If you have done some of the work exclude it from the plan to prevent the executor agent from repeating it.
        
        Here is the task you need to create a plan for: {task}
        """,
        max_iterations=3,
        instructions_as_task=True,
        reflect=True,
        mcp_servers=["brave-search", "time"],
    )

    # Create an executor agent
    executor = AgentConfig(
        role="executor",
        model="ollama:cogito:14b",
        mcp_servers=["filesystem", "sqlite"],
        min_score=1.0,
        instructions="You are an execution specialist. Implement solutions using available tools and planning steps provided by the planner",
        dependencies=["planner"],
        max_iterations=10,
        reflect=True,
    )

    # Add agents to workflow
    workflow_config.add_agent(planner)
    workflow_config.add_agent(executor)

    return Workflow(workflow_config)


async def run_workflow(task: str) -> Dict[str, str]:
    workflow = await create_example_workflow()
    return await workflow.run(task)


async def main():
    task = "Find the current price of xrp using a web search, then create a table called crypto_prices (currency, price, timestamp), then insert the price of xrp into the table."
    # task =" what day is it?"
    agent_context = None  # Initialize context to None
    try:
        mcp_client = await MCPClient(
            server_filter=["brave-search", "sqlite", "time"]
        ).initialize()

        async def confirmation_callback(tool_name: str, params: Dict[str, Any]) -> bool:
            return input("Continue with this tool? (y/n)") == "y"

        # Create AgentContext
        agent_context = AgentContext(
            agent_name="Task Agent",
            role="Task Executor",
            provider_model_name="ollama:cogito:14b",
            mcp_client=mcp_client,  # Pass initialized client
            min_completion_score=1.0,
            instructions="Solve the given task as quickly as possible using the tools at your disposal.",
            max_iterations=10,
            reflect_enabled=True,
            log_level="debug",
            initial_task=task,
            tool_use_enabled=True,
            use_memory_enabled=True,
            collect_metrics_enabled=True,
            check_tool_feasibility=True,
            enable_caching=False,
            confirmation_callback=confirmation_callback,
        )

        # Initialize ReactAgent with the context
        agent = ReactAgent(context=agent_context)

        # Run the agent
        result_dict = await agent.run(initial_task=task)

        # Convert result dict to JSON string for printing
        json_result = json.dumps(result_dict, indent=4, default=str)
        print("--- Agent Run Result ---")
        print(json_result)
        print("------------------------")

    finally:
        # Ensure context and its resources (like MCP client) are closed
        if agent_context:
            print("\nClosing AgentContext...")
            await agent_context.close()
            print("AgentContext closed.")

    # OR run the example workflow

    # results = await run_workflow(task)
    # print("\nWorkflow Results:")
    # for role, result in results.items():
    #     print(f"\n{role.upper()} Result:")
    #     print(result)


if __name__ == "__main__":
    asyncio.run(main())
