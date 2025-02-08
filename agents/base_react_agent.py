import json
import logging
from typing import List, Dict, Any

from pydantic import BaseModel
from model_providers.base import BaseModelProvider
from model_providers.factory import ModelProviderFactory
from prompts.agent_prompts import (
    REACT_AGENT_SYSTEM_PROMPT,
    TASK_REFLECTION_SYSTEM_PROMPT,
    TASK_PLANNING_SYSTEM_PROMPT,
    TASK_TOOL_REVISION_SYSTEM_PROMPT,
)


class BaseReactAgent:
    def __init__(
        self,
        name: str,
        provider_model: str,
        instructions: str = "",
        tools: List[Any] = [],
        tool_use: bool = True,
        reflect: bool = False,
        min_completion_score: float = 1.0,
        max_iterations: int | None = None,
        log_level: str = "info",
    ):
        ## Agent Attributes
        self.name: str = name
        self.model_provider: BaseModelProvider = (
            ModelProviderFactory.get_model_provider(provider_model)
        )
        self.reflect: bool = reflect
        self.initial_task: str = ""
        self.instructions: str = instructions
        self.tools: List[Any] = tools if tools else []
        self.tool_use = tool_use
        self.tool_map: Dict[str, Any] = {tool.__name__: tool for tool in self.tools}
        self.tool_signatures: List[Dict[str, Any]] = [
            tool.tool_definition for tool in self.tools
        ]
        self.tool_history: List[Dict[str, Any]] = []
        self.memory: list = []
        self.reflections: list = []
        self.reflection_messages: list = []
        self.messages: list = [{"role": "system", "content": REACT_AGENT_SYSTEM_PROMPT}]

        self.min_completion_score = min_completion_score
        self.max_iterations = max_iterations

        ## Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
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

    async def _plan(self, **kwargs) -> str | None:
        try:
            self.logger.info("Planning...")
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
            if not reflect:
                self.logger.debug(f"Result: {message_content}")
            tool_calls = result["message"].get("tool_calls")
            if message_content and remember_messages:
                self.messages.append({"role": "assistant", "content": message_content})
            elif tool_calls:
                await self._process_tool_calls(tool_calls=tool_calls)
                return await self._think_chain(tool_use=False)

            return result
        except Exception as e:
            self.logger.error(f"Chat Completion Error: {e}")
            return

    async def _process_tool_calls(self, tool_calls):
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
                self.logger.debug(self.messages[-1])
                processed_tool_calls.append(str(tool_call))

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
                self.logger.debug(f"Tool Result: {result}")
                return result
            except Exception as e:
                self.logger.debug(f"Tool Execution Error: {e}")
                self.tool_history.append(
                    {
                        "name": tool_name,
                        "params": params,
                        "result": f"Tool Execution Error: {e}",
                    }
                )
                return f"Tool Execution Error: {e}"

    async def _run_task(self, task, tool_use: bool = True) -> dict | None:
        print(
            f"Observation: {self.memory[-1]['tool_suggestion'] if self.memory else ''}"
        )
        self.messages.append(
            {
                "role": "user",
                "content": f"""
                
                 
                TASK: {task}
                
                {f"Previous Attempt Failure Reason: {self.memory[-1]['failed_reason']}" if self.memory else ''}
                {f"Reflection Agent Suggested Improvement: {self.memory[-1]['tool_suggestion'] if self.memory else ''}"}
                 
                """.strip(),
            }
        )
        result = await self._think_chain(tool_use=tool_use)
        return result if result else None

    async def _reflect(self, task_description, result):
        class format(BaseModel):
            completion_score: float
            reason: str
            tool_suggestion: str

        reflect_prompt = """
        {TASK_REFLECTION_SYSTEM_PROMPT}
        
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
                },
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
        while True if self.max_iterations is None else iterations < self.max_iterations:
            iterations += 1
            print(f"Running Iteration: {iterations}")
            # plan = await self._plan(task=task)
            # if not plan:
            #     self.logger.info(f"{self.name} Failed\n")
            #     self.logger.info(f"Task: {task}\n")
            #     return
            # plan = json.loads(plan)
            # print(f"Executing plan: ")
            # for step in plan["plan"]:
            #     print(f"{list(step.keys())[0]}: {step.get(list(step.keys())[0])}")
            # # plan["plan"].append(self.initial_task)
            # for step in plan["plan"]:
            #     iterations += 1
            #     step_type = list(step.keys())[0]
            #     self.logger.info(
            #         f"Running {step_type} Step {iterations}: {step.get(step_type)}"
            #     )
            #     tool_use = True if step_type == "action" else False
            #     running_task = step.get("action", step.get("thought", self.initial_task))
            #     result = await self._run_task(task=running_task, tool_use=tool_use)
            #     if not result:
            #         self.logger.info(f"{self.name} Failed\n")
            #         self.logger.info(f"Task: {task}\n")
            #         break
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
                    self.memory.append(
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
