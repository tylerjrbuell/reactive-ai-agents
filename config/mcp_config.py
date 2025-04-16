from typing import Any, Literal, Optional
import dotenv
import os

dotenv.load_dotenv()

type MCPConfig = dict[
    Literal["mcpServers"],
    dict[str, dict[Literal["command", "args", "env"], Any]],
]

server_config: MCPConfig = {
    "mcpServers": {
        "local": {
            "command": "python",
            "args": ["./agent_mcp/server.py"],
            "env": {},
        },
        "time": {
            "command": "docker",
            "args": ["run", "--name", "mcp-time", "-i", "--rm", "mcp/time"],
            "env": {},
        },
        "filesystem": {
            "command": "docker",
            "args": [
                "run",
                "--name",
                "mcp-filesystem",
                "-i",
                "--rm",
                "--mount",
                "type=bind,src=/home/tylerbuell/Documents/AI Projects/reactive-ai-agent,dst=/projects",
                "--workdir",
                "/projects",
                "mcp/filesystem",
                "/projects",
            ],
            "env": {},
        },
        "sqlite": {
            "command": "docker",
            "args": [
                "run",
                "--name",
                "mcp-sqlite",
                "--rm",
                "-i",
                "-v",
                "/home/tylerbuell/Documents/AI Projects/reactive-ai-agent/agent_mcp:/mcp",
                "mcp/sqlite",
                "--db-path",
                "/mcp/agent.db",
            ],
            "env": {},
        },
        "playwright": {
            "command": "npx",
            "args": ["-y", "@executeautomation/playwright-mcp-server"],
            "env": {},
        },
        "brave-search": {
            "command": "docker",
            "args": [
                "run",
                "--name",
                "mcp-brave-search",
                "-i",
                "--rm",
                "-e",
                "BRAVE_API_KEY",
                "mcp/brave-search",
            ],
            "env": {"BRAVE_API_KEY": os.environ.get("BRAVE_API_KEY")},
        },
        "duckduckgo": {
            "command": "docker",
            "args": ["run", "--name", "mcp-duckduckgo", "-i", "--rm", "mcp/duckduckgo"],
            "env": {},
        },
    }
}


def get_mcp_servers(filter: Optional[list[str]] = None) -> MCPConfig:
    if filter:
        return {
            "mcpServers": {
                k: v for k, v in server_config["mcpServers"].items() if k in filter
            }
        }
    return server_config
