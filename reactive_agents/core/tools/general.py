from typing import Any, Dict, List
from reactive_agents.core.tools.decorators import tool
from bs4 import BeautifulSoup
import ollama
from reactive_agents.app.agents.react_agent import ReactAgent
from reactive_agents.providers.external.client import MCPClient
import requests
from urllib.parse import urlparse
import os
import requests
import shutil
import sqlite3
from markitdown import MarkItDown
import json

conn = sqlite3.connect("./agent.db")


@tool()
async def execute_sqlite_query(query: str) -> List[Any] | str:
    """
    Execute a SQL query in the database.

    Args:
        query (str): The SQL query to execute.

    Returns:
        str: A message indicating the successful execution of the query.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        return cursor.fetchall()
    except Exception as e:
        return f"Error executing query: {e}"


@tool()
async def alter_sqlite_table(table_name: str, schema: str) -> str:
    """
    Alter the schema of a table in the database.

    Args:
        table_name (str): The name of the table to alter.
        schema (str): The new schema for the table.

    Returns:
        str: A message indicating the successful alteration of the table.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"ALTER TABLE {table_name} {schema}")
        conn.commit()
        return f"Table {table_name} altered successfully with schema: {schema}"
    except Exception as e:
        return f"Error altering table: {e}"


@tool()
async def count_sqlite_records(
    table_name: str, conditions: Dict[str, Any]
) -> int | str:
    """
    Count the number of records in a table in the database.

    Args:
        table_name (str): The name of the table to count the records in.
        conditions (Dict[str, Any]): A dictionary of conditions to filter the records.

    Returns:
        int: The number of records in the table or an error message.
    """
    try:
        conditions = eval(str(conditions)) if type(conditions) == str else conditions
        if conditions:
            where_clause = " AND ".join([f"{k} = ?" for k in conditions.keys()])
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT COUNT(*) FROM {table_name} { f'WHERE {where_clause}' if conditions.keys() else ''}",
            tuple(conditions.values()) if conditions else (),
        )
        return cursor.fetchone()[0]
    except Exception as e:
        return f"Error counting records: {e}"


@tool()
async def get_sqlite_table_names() -> List[str]:
    """
    Get the names of all tables in the database.

    Returns:
        List[str]: A list of table names.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cursor.fetchall()]


@tool()
async def get_sqlite_table_schema(table_name: str) -> str | None:
    """
    Get the schema of a table in the database.

    Args:
        table_name (str): The name of the table to get the schema for.

    Returns:
        str: The schema of the table or None if the table does not exist.
    """
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return ", ".join([f"{row[1]} {row[2]}" for row in cursor.fetchall()])


@tool()
async def create_sqlite_table(table_name: str, schema: str) -> str:
    """
    Create a new table in the database.

    Args:
        table_name (str): The name of the table to create.
        schema (str): The schema definition for the table columns.

    Returns:
        str: A message indicating the successful creation of the table.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"CREATE TABLE {table_name} ({schema})")
        conn.commit()
        return f"Table {table_name} created successfully with schema: {schema}"
    except Exception as e:
        return f"Error creating table: {e}"


@tool()
async def create_sqlite_record(table: str, data: Dict[str, Any]) -> str:
    """
    Create a new record in the specified table.

    Args:
        table (str): The name of the table to insert the record into.
        data (Dict[str, Any]): A dictionary containing the column names and values for the new record.

    Returns:
        str: A message indicating the successful creation of the record.
    """
    # serialize the data if it comes in as json
    data = eval(str(data)) if type(data) == str else data
    cursor = conn.cursor()
    placeholders = ", ".join(["?"] * len(data))
    columns = ", ".join(data.keys())
    cursor.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
        tuple(data.values()),
    )
    conn.commit()
    return "Record created successfully"


@tool()
async def read_sqlite_records(
    table: str, conditions: Dict[str, str] | None = None
) -> List[Dict[str, Any]]:
    """
    Read records from the specified table with optional conditions.

    Example usage:
        read_sqlite_records("table_name")
        read_sqlite_records("table_name", {"<column_name>": "<operator> <value>"})

    Real example:
        read_sqlite_records("students", {"age": "> 18"})
        read_sqlite_records("students", {"name": "= 'John'"})

    Args:
        table (str): The name of the table to read records from.
        conditions (Dict[str, str], optional): A dictionary of conditions to filter the records. Defaults to None.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the records.
    """
    # serialize the conditions if they come in as json
    conditions = eval(str(conditions)) if type(conditions) == str else conditions

    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if conditions:
        where_clause = " AND ".join(
            [
                f"{k} {v.split(' ')[0] or '='} {v.split(' ')[1]}"
                for k, v in conditions.items()
            ]
        )
        cursor.execute(
            f"SELECT * FROM {table} WHERE {where_clause}",
        )
    else:
        cursor.execute(f"SELECT * FROM {table}")
    return [dict(row) for row in cursor.fetchall()]


@tool()
async def update_sqlite_record(
    table: str, data: Dict[str, Any], conditions: Dict[str, Any]
) -> str:
    """
    Update a record in the specified table.

    Args:
        table (str): The name of the table to update the record in.
        data (Dict[str, Any]): A dictionary containing the column names and new values for the record.
        conditions (Dict[str, Any]): A dictionary of conditions to identify which records to update.

    Returns:
        str: A message indicating the successful update of the record.
    """
    data = eval(str(data)) if type(data) == str else data
    conditions = eval(str(conditions)) if type(conditions) == str else conditions
    cursor = conn.cursor()
    set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
    where_clause = " AND ".join([f"{k} = ?" for k in conditions.keys()])
    cursor.execute(
        f"UPDATE {table} SET {set_clause} WHERE {where_clause}",
        tuple(data.values()) + tuple(conditions.values()),
    )
    conn.commit()
    return "Record updated successfully"


@tool()
async def delete_sqlite_record(table: str, conditions: Dict[str, str]) -> str:
    """
    Delete a record from the specified table.

    Example usage:
        delete_sqlite_record("table_name")
        delete_sqlite_record("table_name", {"<column_name>": "<operator> <value>"})

    Real example:
        delete_sqlite_record("students", {"age": "> 18"})
        delete_sqlite_record("students", {"name": "= 'John'"})

    Args:
        table (str): The name of the table to delete the record from.
        conditions (Dict[str, str]): A dictionary of conditions to identify which records to delete.

    Returns:
        str: A message indicating the successful deletion of the record.
    """
    try:
        conditions = eval(str(conditions)) if type(conditions) == str else conditions
        cursor = conn.cursor()
        if conditions:
            where_clause = " AND ".join(
                [
                    f"{k} {v.split(' ')[0] or '='} {v.split(' ')[1]}"
                    for k, v in conditions.items()
                ]
            )
            cursor.execute(
                f"DELETE FROM {table} WHERE {where_clause}",
            )
            conn.commit()
            return "Record deleted successfully"
        raise Exception("No conditions provided")
    except Exception as e:
        return f"Error deleting record: {e}"


@tool()
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

    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("items", [])
        return [
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
            for item in results
        ]
    else:
        print(f"Google Search API Error: {response.status_code}")
        return []


@tool()
async def get_current_datetime():
    """
    Get the current date and time in the format "YYYY-MM-DD HH:MM:SS".

    Returns:
        str: The current date and time in the format "YYYY-MM-DD HH:MM:SS".
    """
    import datetime

    return f"Todays date and time is: {datetime.datetime.now().strftime('%A, %Y-%m-%d %H:%M:%S')}"


@tool()
async def get_current_cryptocurrency_market_data(currency_id: str):
    """
    Get the real-time market data for a specific coin identifier of crypto currency vs USD.

    Args:
        currency_id (str): Target coin identifier of a crypto currency.

    Returns:
        str: The current price of the crypto currency queried.
    """

    if type(currency_id) == str:
        try:
            url = f"https://api.coingecko.com/api/v3/coins/markets?ids={currency_id}&vs_currency=usd&order=market_cap_desc"
            response = requests.get(url)
            data = response.json()
            if not data:
                return f"Error: currency_id: '{currency_id}' was invalid. Try a different currency identifier."
            return data
        except Exception as e:
            return f"Error: {e}"
    elif eval(str(currency_id)) != type(str):
        return f"Error: Invalid identifier, expected a string with a single coin identifier but got: {type(eval(str(currency_id)))}"


@tool()
async def get_user_input(question: str) -> str:
    """
    Get input from user for for a specific question to help complete the task.
    Key phases to use this tool are when the user is asked a question that requires input from the user.
    Examples:
        - "Please provide me with some input to complete the task."
        - "What is your name?"
        - "I need your help to complete this task"

    Args:
        question (str): The question to the user. (e.g. "What is your name?") the question should be posed from the perspective of the assistant.

    Returns:
        str: The user's response to the question.
    """
    return f"User answer to the question: {question} is: {(input(f'{question}: '))}"


@tool()
async def execute_python_code(code: str, return_value_variable_name: str) -> Any:
    """
    Execute Python code and optionally save the return value in a variable with the same name as the return_value_variable_name
    so it can be used later. If the return_value_variable_name is not specified then the return value is not saved.

    Usage example:
    - "execute_python_code('x = 5', 'x')"
    - "execute_python_code('import random; result = random.randint(1, 10)', 'result')"

    Args:
        code (str): The Python code to execute.
        return_value_variable_name (str): The name of the variable to store the return value for retrieval after execution.

    Returns:
        Any: The value of the return variable.
    """
    import ast

    # User confirmation
    yn = input(
        f"Are you sure you want to execute this code: \n{code.strip()}\n\n(y/n): "
    )
    if yn.casefold() != "y".casefold():
        return "Code not executed."

    # Create isolated execution environment
    global_vars = {}
    local_vars = {}

    try:
        # Optional static analysis (if you want to detect invalid Python code early)
        ast.parse(code)

        # Execute the code
        exec(code, global_vars, local_vars)

        # Retrieve the return value if specified
        if return_value_variable_name:
            if return_value_variable_name in local_vars:
                return local_vars[return_value_variable_name]
            else:
                raise NameError(
                    f"Variable '{return_value_variable_name}' not defined in the code."
                )

        # No return variable specified
        return None
    except SyntaxError as syntax_error:
        return f"Syntax Error: {syntax_error}"
    except Exception as e:
        return f"Execution Error: {e}"


@tool()
async def get_url_content(url: str) -> str | None:
    """
    Get the content of a URL in markdown format if possible otherwise raw HTML.

    Args:
        url (str): The URL to fetch the content from.

    Returns:
        str: The markdown or raw HTML content of the URL.
    """
    mkdown = await url_to_markdown(url)
    if mkdown:
        return mkdown
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


@tool()
async def list_directories(path: str) -> list:
    """
    List the directories in a given path.

    Args:
        path (str): The path to list the directories from.

    Returns:
        list: A list of directories in the path.
    """
    return os.listdir(path)


@tool()
async def move_directory(source_path: str, destination_path: str) -> str:
    """
    Move a directory from one path to another.

    Args:
        source_path (str): The path of the directory to move.
        destination_path (str): The path to move the directory to.

    Returns:
        str: A message indicating the successful movement of the directory.
    """
    shutil.move(source_path, destination_path)
    return f"Directory moved from {source_path} to {destination_path}"


@tool()
async def new_directory(path: str) -> str:
    """
    Create a new directory at a given path.

    Args:
        path (str): The path to create the new directory at.

    Returns:
        str: A message indicating the successful creation of the directory.
    """
    os.mkdir(path)
    return f"Directory created at {path}"


@tool()
async def read_file(path: str) -> str:
    """
    Read the contents of a file and return it as a string.

    Args:
        path (str): The path to the file to read.

    Returns:
        str: The contents of the file as a string. If an error occurs, it returns the error message.
    """
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@tool()
async def write_file(path: str, content: str, mode: str = "w") -> str:
    """
    Write content to a file.

    Args:
        path (str): The path to the file to write.
        content (str): The content to write to the file.
        mode (str, optional): The python file mode to open the file in. Defaults to "w". Use this to change the file mode for the purpose of appending to a file or overwriting a file for example.

    Returns:
        str: A message indicating the successful writing of the file. If an error occurs, it returns the error message.
    """
    try:
        with open(path, mode) as f:
            f.write(content)
        return f"File '{path}' written successfully."
    except Exception as e:
        return f"Error writing file: {e}"


@tool()
def query_json_file(
    file_path: str, key: str, result_cursor: int = 0, results: int = 10
) -> dict:
    """
    Read a specific key from a JSON file and get back a specified number of results

    Args:
        file_path (str): The path to the JSON file.
        key (str): The key to retrieve from the JSON file.
        result_cursor (int, optional): The index of the first result to return. Defaults to 0.
        results (int, optional): The number of results to return from the specified index. Defaults to 10. If -1, return all results from the specified index.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
        if key in data:
            return data[key][int(result_cursor) : int(result_cursor) + int(results)]
        return data
