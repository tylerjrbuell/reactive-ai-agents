import json
import logging
from typing import List, Dict, Any

from pydantic import BaseModel
from model_providers.base import BaseModelProvider
from model_providers.factory import ModelProviderFactory
from prompts.agent_prompts import (
    REFLECTION_AGENT_SYSTEM_PROMPT,
    TASK_REFLECTION_SYSTEM_PROMPT,
    TASK_PLANNING_SYSTEM_PROMPT,
    TASK_TOOL_REVISION_SYSTEM_PROMPT,
)


class BaseReactAgent:
    def __init__(
        self,
        name: str,
        provider_model: str,
        role: str = "",
        purpose: str = "",
        persona: str = "",
        response_format: str = "",
        instructions: str = "",
        tools: List[Any] = [],
        tool_use: bool = True,
        reflect: bool = False,
        min_completion_score: float = 1.0,
        max_iterations: int | None = None,
    ):
        ## Agent Attributes
        self.name: str = name
        self.model_provider: BaseModelProvider = (
            ModelProviderFactory.get_model_provider(provider_model)
        )
        self.reflect: bool = reflect
        self.initial_task: str = ""
        self.purpose: str = purpose
        self.role: str = role
        self.persona: str = persona
        self.response_format: str = response_format
        self.instructions: str = instructions
        self.tools: List[Any] = tools if tools else []
        self.tool_use = tool_use
        self.tool_map: Dict[str, Any] = {tool.__name__: tool for tool in self.tools}
        self.tool_signatures: List[Dict[str, Any]] = [
            tool.ollama_tool_definition for tool in self.tools
        ]
        self.tool_history: List[Dict[str, Any]] = []
        self.attempt_history: list = []
        self.reflections: list = []
        self.reflection_messages: list = []
        self.messages: list = [
            {
                "role": "system",
                "content": REFLECTION_AGENT_SYSTEM_PROMPT.format(
                    agent_name=self.name or "Reflection Agent",
                    agent_role=self.role or "Task Solving Agent",
                    agent_purpose=self.purpose
                    or "Solve the given task to to the best of your ability",
                    agent_persona=self.persona or "You are a helpful assistant.",
                    agent_instructions=self.instructions
                    or "Solve the given task to to the best of your ability",
                    agent_response_format=self.response_format or "JSON",
                ),
            }
        ]

        self.min_completion_score = min_completion_score
        self.max_iterations = max_iterations

        ## Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(f"{self.name}:%(levelname)s - %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(f"{self.name} Initialized")
        self.logger.info(f"Provider:Model: {provider_model}")
        self.logger.info(f"Available Tools: {list(self.tool_map.keys())}")

    async def _think(self, **kwargs) -> dict | None:
        try:
            self.logger.info("Thinking...")
            return await self.model_provider.get_completion(**kwargs)
        except Exception as e:
            self.logger.error(f"Completion Error: {e}")
            return

    async def _think_chain(
        self,
        tool_use: bool = True,
        remember_messages: bool = True,
        reflect: bool = False,
        **kwargs,
    ) -> dict | None:
        try:
            kwargs.setdefault("messages", self.messages)
            self.logger.info("Thinking..." if not reflect else "Reflecting...")

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
                processed_tool_calls = []
                for tool_call in tool_calls:
                    if str(tool_call) in processed_tool_calls:
                        print("Tool call already processed")
                        continue
                    tool_result = await self._execute_tool(tool_call=tool_call)
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
                                "content": str(tool_result),
                            }
                        )
                        processed_tool_calls.append(str(tool_call))
                return await self._think_chain()

            return result
        except Exception as e:
            self.logger.error(f"Chat Completion Error: {e}")
            return

    async def _execute_tool(self, tool_call):
        tool_name = tool_call["function"]["name"]
        if tool_name in self.tool_map.keys():
            self.logger.info(f"Executing tool: {tool_name}")
            tool_function = self.tool_map.get(tool_name, None)
            params = (
                tool_call["function"]["arguments"]
                if type(tool_call["function"]["arguments"]) is dict
                else json.loads(tool_call["function"]["arguments"])
            )
            try:
                result = await tool_function(**params)
                self.tool_history.append(
                    {"name": tool_name, "params": params, "result": result}
                )
                print(f"Tool Result: {result}")
                return result
            except Exception as e:
                self.logger.error(f"Tool Execution Error: {e}")
                self.tool_history.append(
                    {
                        "name": tool_name,
                        "params": params,
                        "result": f"Tool Execution Error: {e}",
                    }
                )
                return f"Tool Execution Error: {e}"

    async def _run_task(self, task) -> dict | None:
        print(
            f"Observation: {self.attempt_history[-1]['tool_suggestion'] if self.attempt_history else ''}"
        )
        self.messages.append(
            {
                "role": "user",
                "content": f"""
                <task>
                    {task},
                    {self.attempt_history[-1]['tool_suggestion'] if self.attempt_history else ''}
                </task>
                
                """.strip(),
            }
        )
        result = await self._think_chain()
        return result if result else None

    async def _reflect(self, task_description, result):
        class format(BaseModel):
            completion_score: float
            reason: str
            tool_suggestion: str

        reflect_prompt = """
        {TASK_REFLECTION_SYSTEM_PROMPT}
        
        Reflect on the following task and result and determine if the task was completed or not. Use message history for context.
        If the task was not complete yet, provide tool suggestions to improve help the task agent (separate agent) to complete the task.
        
        <task>{task_description}</task>
        <result>{result}</result>
        <available-tools>{tools}</available-tools>
               """.format(
            task_description=task_description,
            result=result["message"]["content"],
            TASK_REFLECTION_SYSTEM_PROMPT=TASK_REFLECTION_SYSTEM_PROMPT,
            tools=json.dumps(self.tool_signatures),
        )
        reflection = await self._think_chain(
            format=(
                format.model_json_schema()
                if self.model_provider.name == "ollama"
                else "json"
            ),
            messages=[
                {
                    "role": "user",
                    "content": reflect_prompt,
                }
            ],
            reflect=True,
            tool_use=False,
            remember_messages=False,
        )
        return {
            "response": (reflection["message"]["content"] if reflection else ""),
        }

    async def _run_task_iteration(self, task):
        running_task = task
        self.logger.info(f"Starting New Task: {running_task}")
        iterations = 0
        while iterations < self.max_iterations if self.max_iterations else True:
            iterations += 1
            self.logger.info(f"Running Iteration: {iterations}")
            result = await self._run_task(task=running_task)
            if not result:
                self.logger.info(f"{self.name} Failed\n")
                self.logger.info(f"Task: {task}\n")
                break
            if self.reflect:
                reflection = await self._reflect(task, result=result)
                if not reflection:
                    self.logger.info(f"{self.name} Failed\n")
                    self.logger.info(f"Task: {task}\n")
                    break
                reflection = (
                    json.loads(reflection["response"]) if reflection["response"] else {}
                )
                print(f"Percent Complete: {reflection.get('completion_score', 0)}%")
                self.reflections.append(reflection)
                if reflection.get("completion_score", 0) >= self.min_completion_score:
                    self.logger.info(
                        f"{self.name} Task Completed Successfully because: {reflection['reason']}\n"
                    )
                    return result["message"]["content"]
                else:
                    self.logger.info(
                        f"{self.name} Task Failed because: {reflection['reason']}\n"
                    )
                    self.attempt_history.append(
                        {
                            "result": result["message"]["content"],
                            "failed_reason": reflection["reason"],
                            "tool_suggestion": reflection["tool_suggestion"],
                            "tool": (
                                self.tool_history[-1] if self.tool_history else None
                            ),
                        }
                    )
        return result["message"]["content"] if result else None

    def set_model_provider(self, provider_model: str):
        self.model_provider = ModelProviderFactory.get_model_provider(provider_model)
        self.logger.info(f"Model Provider Set to: {self.model_provider.name}")

    async def run(self, initial_task):
        self.initial_task = initial_task
        result = await self._run_task_iteration(task=initial_task)
        return result
