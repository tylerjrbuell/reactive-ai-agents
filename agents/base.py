import json
from loggers.base import Logger
from model_providers.base import BaseModelProvider
from prompts.agent_prompts import (
    TASK_PLANNING_SYSTEM_PROMPT,
    TASK_REFLECTION_SYSTEM_PROMPT,
)
from model_providers.factory import ModelProviderFactory
from tools.base import Tool
from typing import List, Dict, Any
from pydantic import BaseModel


class Agent:
    def __init__(
        self,
        name: str,
        provider_model: str,
        instructions: str = "",
        tools: List[Any] = [],
        tool_use: bool = True,
        min_completion_score: float = 1.0,
        max_iterations: int | None = None,
        log_level: str = "info",
    ):
        ## Agent Attributes
        self.name: str = name
        self.model_provider: BaseModelProvider = (
            ModelProviderFactory.get_model_provider(provider_model)
        )
        self.initial_task: str = ""
        self.instructions: str = instructions
        self.tools: List[Tool] = [Tool(tool) for tool in tools] if tools else []
        self.tool_use = tool_use
        self.tool_map: Dict[str, Tool] = {tool.name: tool for tool in self.tools}
        self.tool_signatures: List[Dict[str, Any]] = [
            tool.tool_definition for tool in self.tools
        ]
        self.tool_history: List[Dict[str, Any]] = []
        self.memory: str = ""
        self.messages: list = [{"role": "system", "content": self.instructions}]

        self.min_completion_score = min_completion_score
        self.max_iterations = max_iterations
        self.iterations: int = 0
        self.agent_logger = Logger(
            name=name,
            type="agent",
            level=log_level,
        )

        self.tool_logger = Logger(
            name=f"{self.name} Tool",
            type="tool",
            level=log_level,
        )

        self.result_logger = Logger(
            name=f"{self.name} Result", type="agent_response", level=log_level
        )

        self.agent_logger.info(f"{self.name} Initialized")
        self.agent_logger.info(f"Provider:Model: {provider_model}")
        self.agent_logger.info(
            f"Available Tools: {", ".join(list(self.tool_map.keys()))}"
        )

    async def _think(self, **kwargs) -> dict | None:
        try:
            self.agent_logger.info("Thinking...")
            return await self.model_provider.get_completion(**kwargs)
        except Exception as e:
            self.agent_logger.error(f"Completion Error: {e}")
            return

    async def _plan(self, **kwargs) -> str | None:
        try:
            self.agent_logger.info("Planning...")
            result = await self.model_provider.get_completion(
                system=TASK_PLANNING_SYSTEM_PROMPT,
                prompt="""
                Given this task: {task}
                Available tools: {tools}
                
                Plan a sequence of steps to complete the task as efficiently as possible. Keep the plan as short as possible.
                Respond in raw JSON format with no additional text
                """.format(
                    task=self.initial_task,
                    tools=[
                        f"Name: {tool['function']['name']}\nDescription: {tool['function']['description']}\nParameters: {tool['function']['parameters']}"
                        for tool in self.tool_signatures
                    ],
                ),
                format="json",
            )
            return result.get("response", None)
        except Exception as e:
            self.agent_logger.error(f"Completion Error: {e}")
            return

    async def _think_chain(
        self,
        tool_use: bool = True,
        remember_messages: bool = True,
        **kwargs,
    ) -> dict | None:
        try:
            kwargs.setdefault("messages", self.messages)
            self.agent_logger.info(f"Thinking...")

            result = await self.model_provider.get_chat_completion(
                tools=self.tool_signatures if tool_use else [],
                **kwargs,
            )
            if not result:
                return None
            message_content = result["message"].get("content")
            tool_calls = result["message"].get("tool_calls")
            if message_content and remember_messages:

                self.messages.append({"role": "assistant", "content": message_content})
            elif tool_calls:
                await self._process_tool_calls(tool_calls=tool_calls)
                return await self._think_chain(tool_use=False, **kwargs)

            return result
        except Exception as e:
            self.agent_logger.error(f"Chat Completion Error: {e}")
            return

    async def _process_tool_calls(self, tool_calls) -> None:
        processed_tool_calls = []
        for tool_call in tool_calls:
            if str(tool_call) in processed_tool_calls:
                self.tool_logger.info(
                    f"Tool Call Already Processed: {tool_call["function"]["name"]}"
                )
                continue
            tool_result = await self._use_tool(tool_call=tool_call)
            if tool_result is not None:
                self.messages.append(
                    {
                        **(
                            {"tool_call_id": tool_call["id"]}
                            if tool_call.get("id")
                            else {}
                        ),
                        "role": "tool",
                        "name": tool_call["function"]["name"],
                        "content": f"{tool_result}",
                    }
                )
                processed_tool_calls.append(str(tool_call))

    async def _use_tool(self, tool_call) -> dict | None:
        tool_name = tool_call["function"]["name"]
        tool = self.tool_map.get(tool_name)
        if tool:
            self.tool_logger.info(
                f"Executing tool: {tool_name} with parameters: {tool_call['function']['arguments']}"
            )
            params = (
                tool_call["function"]["arguments"]
                if type(tool_call["function"]["arguments"]) is dict
                else json.loads(tool_call["function"]["arguments"])
            )
            result = await tool.use(params=params)
            self.tool_history.append(
                {"name": tool.name, "params": params, "result": result}
            )
            self.tool_logger.debug(f"Tool Result: {result}")
            return result
        self.tool_logger.error(
            f"Tool {tool_name} not found in available tools: {self.tool_map.keys()}"
        )
        return None

    async def _run_task(self, task, tool_use: bool = True) -> dict | None:
        self.agent_logger.debug(f"MAIN TASK: {task}")
        self.messages.append(
            {
                "role": "user",
                "content": f"""
                
                MAIN TASK: {task}
                """.strip(),
            }
        )
        result = await self._think_chain(tool_use=tool_use)
        return result

    async def _run_task_iteration(self, task):
        running_task = task
        self.agent_logger.info(f"Starting New Task: {running_task}")
        self.iterations = 0
        while (
            True
            if self.max_iterations is None
            else self.iterations < self.max_iterations
        ):
            self.iterations += 1
            self.agent_logger.info(f"Running Iteration: {self.iterations}")
            result = await self._run_task(task=running_task)
            if not result:
                self.agent_logger.info(f"{self.name} Failed\n")
                self.agent_logger.info(f"Task: {task}\n")
                break
        return result["message"]["content"] if result else None

    def set_model_provider(self, provider_model: str):
        self.model_provider = ModelProviderFactory.get_model_provider(provider_model)
        self.agent_logger.info(f"Model Provider Set to: {self.model_provider.name}")

    async def run(self, initial_task):
        self.initial_task = initial_task
        result = await self._run_task_iteration(task=initial_task)
        self.result_logger.info(result)
        return result
