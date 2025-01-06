from agents.base_react_agent import BaseReactAgent
from tools import *
import asyncio
import dotenv
import warnings

warnings.simplefilter("ignore", ResourceWarning)

dotenv.load_dotenv()


if __name__ == "__main__":
    agent = BaseReactAgent(
        name="TaskAgent",
        provider_model="ollama:llama3.2",
        # provider_model="groq:llama-3.3-70b-versatile",
        purpose="Do anything in your power to complete the task",
        role="Task Agent",
        persona="You are very smart witty and resourceful AI",
        instructions="""
        Think through the steps to complete the task and select a tool or series of tools to complete the task as efficiently as possible.
        """,
        response_format="Respond in clear and concise Markdown format unless otherwise specified.",
        min_completion_score=0.8,
        reflect=True,
        max_iterations=5,
        tools=[web_search],
    )
    initial_task = input("\nEnter your initial task: ")
    result = asyncio.run(agent.run(initial_task))
    print(f"{agent.name} result: {result}")
