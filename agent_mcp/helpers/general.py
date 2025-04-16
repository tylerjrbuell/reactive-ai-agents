import aiohttp
import fake_useragent
import os
from markitdown import MarkItDown
import ollama
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse


async def summarize_search_result(result: dict, query: str):
    if result.get("error"):
        return result["error"]
    if not result.get("link"):
        return ""
    print(f"Summarizing search result: {result['link']}")

    print("Fetching markdown content...")

    mkdown = await url_to_markdown(result["link"])
    if not mkdown:
        print("Fetching raw site content...")
        async with aiohttp.ClientSession() as session:
            async with session.get(result["link"]) as res:
                if res.status != 200:
                    return "Error fetching site content"
                text = await res.text()
        soup = BeautifulSoup(text, "html.parser")
        website_text = soup.get_text(separator=" ", strip=True)
        summary_content = website_text.strip()
    else:
        summary_content = mkdown.strip()
    print(f"Computing summary...")
    ollama_client = ollama.AsyncClient()
    summary = await ollama_client.chat(
        model="deepseek-r1:14b",
        messages=[
            {
                "role": "system",
                "content": """
                    You are an expert at extracting any relevant information from a website that would help answer a query.
                    Do not attempt to answer the query, rather focus on extracting relevant information to the query so that another AI Agent can answer the query with the information you extract.
                    Retain all source content, links and urls relevant to the query in your summary.
                    Try to be as concise as possible, only retaining information that is relevant to the query, but do not leave out any information that is relevant to the query.
                    Keep the summary as short as possible while still retaining all relevant information to the query.
                    """,
            },
            {
                "role": "user",
                "content": f"Summarize the following website content optimized for the QUERY: '{query}'\n CONTENT: '{summary_content}'.",
            },
        ],
        stream=False,
        options={"num_gpu": 256, "temperature": 0, "num_ctx": 10000},
    )
    if summary.get("message"):
        return summary["message"]["content"]
    return f"No Summary found for query {query}"


async def summarize_webpage_markdown(markdown):
    ollama_client = ollama.AsyncClient()
    summary = await ollama_client.chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": """
                    You are an expert at extracting any relevant information from markdown
                    Do not attempt to answer the query, rather focus on extracting relevant information to the query so that another AI Agent can answer the query with the information you extract.
                    Retain all source content, links and urls relevant to the query in your summary.
                    Try to be as concise as possible
                    Keep the summary as short as possible while still retaining all relevant information to the query.
                    Respond with the summary only as markdown
                    """,
            },
            {
                "role": "user",
                "content": f"Summarize the following website content: '{markdown}'.",
            },
        ],
        stream=False,
        options={"num_gpu": 256, "temperature": 0, "num_ctx": 10000},
    )
    if summary.get("message"):
        return summary["message"]["content"]
    return f"No Summary found for query {markdown}"


async def google_search_api(query, num_results=1):
    """
    Search the web using Google Custom Search JSON API.

    Args:
        query (str): The search query string.
        num_results (int): Number of search results to return (default is 5).

    Returns:
        list: A list of search results, where each result is a dictionary with title, link, and snippet.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": os.environ.get("GOOGLE_SEARCH_API_KEY"),
        "cx": "a60f759067ab94c36",
        "q": query.replace('"', ""),
        "num": num_results,
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                result = await response.json()
                return [
                    {
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet"),
                    }
                    for item in result.get("items", [])
                ]
            else:
                print(f"Google Search API Error: {response.status}")
                return []


async def google_search_scrape(query, num_results=2):
    headers = {"User-Agent": fake_useragent.UserAgent().random}
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    for d in soup.find_all("div"):
        if d.find("a"):
            links = d.find_all("a")
            for link in links:
                if link:
                    href = str(link.get("href"))
                    parsed_href = urlparse(href)
                    href = (
                        f"{parsed_href.scheme}://{parsed_href.netloc}{parsed_href.path}"
                    )
                    if "google.com" in parsed_href.netloc:
                        continue
                    if parsed_href.scheme == "https" and href not in [
                        result["link"] for result in results
                    ]:
                        results.append({"link": href})
    return results[:num_results]


async def url_to_markdown(url: str):
    """
    Convert a URL to Markdown format.

    Args:
        url (str): The URL to convert.

    Returns:
        str: The converted Markdown content.
    """
    # endpoint = f"https://urltomarkdown.herokuapp.com/?url={url}&title=true&links=true"
    try:
        md = MarkItDown()
        result = md.convert_url(url)
        return result.text_content
    except Exception as e:
        return f"Error converting URL to Markdown: {e}"
