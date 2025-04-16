import json
from helpers.general import (
    google_search_api,
    google_search_scrape,
    summarize_webpage_markdown,
    url_to_markdown,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator
from mcp.server.fastmcp import FastMCP
import requests
from bs4 import BeautifulSoup
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

    mcp = FastMCP("reactive-agent-mcp", lifespan=app_lifespan)

    # @mcp.tool()
    # def query_json_file(
    #     file_path: str, key: str, result_cursor: int = 0, results: int = 10
    # ) -> dict:
    #     """
    #     Read a specific key from a JSON file and get back a specified number of results

    #     Args:
    #         file_path (str): The path to the JSON file.
    #         key (str): The key to retrieve from the JSON file.
    #         result_cursor (int, optional): The index of the first result to return. Defaults to 0.
    #         results (int, optional): The number of results to return from the specified index. Defaults to 10. If -1, return all results from the specified index.

    #     Returns:
    #         dict: The contents of the JSON file as a dictionary.
    #     """
    #     with open(file_path, "r") as f:
    #         data = json.load(f)
    #         if key in data:
    #             return data[key][int(result_cursor) : int(result_cursor) + int(results)]
    #         return data

    # @mcp.tool()
    # async def get_url_content(url: str) -> str | None:
    #     """
    #     Get the content of a URL in markdown format if possible otherwise raw HTML.

    #     Args:
    #         url (str): The URL to fetch the content from.

    #     Returns:
    #         str: The markdown or raw HTML content of the URL.
    #     """
    #     mkdown = await url_to_markdown(url)
    #     if mkdown:
    #         summarized_mkdown = await summarize_webpage_markdown(mkdown)
    #         return summarized_mkdown if summarized_mkdown else mkdown
    #     soup = BeautifulSoup(requests.get(url).text, "html.parser")
    #     return soup.get_text(separator=" ", strip=True)

    # @mcp.tool()
    # async def search_web(query: str):
    #     """
    #     Search the web using Google Custom Search JSON API.

    #     Args:
    #         query (str): The search query string.

    #     Returns:
    #         list: List of search results
    #     """
    #     results = await google_search_api(query, num_results=2)
    #     if not results:
    #         print("Scraping data from Google Search...")
    #         results = await google_search_scrape(query, num_results=2)
    #     # summary_coroutines = [summarize_search_result(result, query) for result in results]
    #     # search_summaries = await asyncio.gather(*summary_coroutines)
    #     # return "\n".join(search_summaries) if search_summaries else ""
    #     return results

    # @mcp.tool()
    # def final_answer(answer: str) -> str:
    #     """
    #     Provide final answer to the task

    #     Args:
    #         answer (str): The final answer to the task

    #     Returns:
    #         str: The final answer
    #     """
    #     return answer

    mcp.run(transport="stdio")
