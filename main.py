from agents.react_agent import ReactAgent
from tools.general import (
    web_search,
    get_url_content,
    read_file,
    write_file,
    get_current_datetime,
    read_sqlite_records,
    get_sqlite_table_names,
    get_sqlite_table_schema,
    create_sqlite_table,
    create_sqlite_record,
    update_sqlite_record,
    execute_sqlite_query,
    alter_sqlite_table,
    get_user_input,
    get_current_cryptocurrency_market_data,
    list_directories,
)
from tools.decorators import tool
import asyncio
import dotenv
import warnings

warnings.simplefilter("ignore", ResourceWarning)

dotenv.load_dotenv()
research_agents: dict[str, ReactAgent] = {}


@tool()
async def run_ai_research_agent(
    task_prompt: str,
    provider_model: str = "ollama:qwen2.5:14b",
    max_iterations: int = 5,
):
    """
    Creates and runs an AI research agent to complete a specific task. May take some time to complete and return the result.

    Args:
        task_prompt (str): The prompt to the AI Agent for the task to be completed. The prompt should be specific and concise instruction for the AI Agent.
        provider_model (str, optional): The provider model to use for the AI task agent. Defaults to "ollama:qwen2.5:14b".
        max_iterations (int, optional): The maximum number of iterations to run the AI task agent. Defaults to 5.

    Returns:
        str: The response from the AI task agent.
    """
    from tools.general import web_search, read_file, write_file, get_url_content

    if research_agents.get(task_prompt):
        return await research_agents[task_prompt].run(task_prompt)
    agent = ReactAgent(
        name="ResearchAgent",
        provider_model=provider_model,
        tools=[web_search, read_file, write_file, get_url_content],
        reflect=True,
        max_iterations=max_iterations,
        min_completion_score=1,
        log_level="info",
    )
    research_agents[task_prompt] = agent
    return await agent.run(task_prompt)


async def main():
    agent = ReactAgent(
        name="TaskAgent",
        provider_model="ollama:qwen2.5:14b",
        # provider_model="groq:llama-3.3-70b-versatile",
        min_completion_score=1,
        log_level="debug",
        max_iterations=10,
        tools=[
            run_ai_research_agent,
            read_file,
            write_file,
            get_url_content,
            get_current_datetime,
            list_directories,
        ],
        reflect=True,
    )
    while True:
        initial_task = input("\nEnter your initial task: ")
        result = await agent.run(initial_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExiting...")
        exit()
