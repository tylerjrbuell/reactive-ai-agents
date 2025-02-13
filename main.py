from agents.react_agent import ReactAgent
from tools.general import *
import asyncio
import dotenv
import warnings

warnings.simplefilter("ignore", ResourceWarning)

dotenv.load_dotenv()


async def main():
    agent = ReactAgent(
        name="TaskAgent",
        provider_model="ollama:qwen2.5:14b",
        min_completion_score=1,
        reflect=True,
        log_level="debug",
        max_iterations=10,
        tools=[
            web_search,
            get_url_content,
            get_user_input,
            write_markdown_file,
            read_file,
            list_directories,
            get_current_datetime,
        ],
    )
    while True:
        initial_task = input("\nEnter your initial task: ")
        result = await agent.run(initial_task)
        print(f"{agent.name} result: {result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExiting...")
        exit()
