from agents.base_react_agent import BaseReactAgent
from tools import *
import asyncio
import dotenv

dotenv.load_dotenv()


if __name__ == "__main__":
    agent = BaseReactAgent(
        name="TaskAgent",
        provider_model="ollama:llama3.2",
        # provider_model="groq:llama-3.3-70b-versatile",
        purpose="Do anything in your power to complete the task",
        role="Task Agent",
        persona="You are very smart witty and resourceful AI",
        instructions="Respond with a clear and concise answer to the task. Do not do more than is required to complete the task. Do not make up answers, only use tool result context to help you complete the task.",
        response_format="Respond in clear and concise Markdown format",
        min_completion_score=0.9,
        reflect=True,
        max_iterations=5,
        tools=[web_search, get_url_content],
    )
    initial_task = input("\nEnter your initial task: ")
    result = asyncio.run(agent.run(initial_task))
    print(f"{agent.name} result: {result}")
