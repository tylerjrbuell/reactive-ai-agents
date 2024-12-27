import json
import logging
from typing import List, Dict, Any
from model_providers.ModelProvider import BaseModelProvider
from prompts.agent_prompts import (
    REFLECTION_AGENT_SYSTEM_PROMPT,
    TASK_REFLECTION_SYSTEM_PROMPT,
)


class BaseAgent:
    def __init__(
        self,
        name: str,
        model_provider: BaseModelProvider,
        agent_instructions: str = "",
        model="llama3.2",
        tools: List[Any] = [],
        tool_use: bool = True,
    ):
        ## Agent Attributes
        self.name = name
        self.model = model
        self.iterations = 0
        self.model_provider = model_provider
        self.initial_task: str = ""
        self.agent_instructions = agent_instructions
        self.tools: List[Any] = tools if tools else []
        self.tool_use = tool_use
        self.tool_map: Dict[str, Any] = {tool.__name__: tool for tool in self.tools}
        self.tool_signatures: List[Dict[str, Any]] = [
            tool.ollama_tool_definition for tool in self.tools
        ]
        self.tool_history: List[Dict[str, Any]] = []
        self.reflection_observations: list = []
        self.reflection_messages: list = []
        self.messages: list = [
            {
                "role": "system",
                "content": REFLECTION_AGENT_SYSTEM_PROMPT.format(
                    agent_name=self.name,
                    task=self.initial_task,
                ),
            }
        ]

        ## Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(f"{self.name}:%(levelname)s - %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    async def chat(self, **kwargs) -> dict | None:
        try:
            if not kwargs.get("messages"):
                kwargs["messages"] = self.messages
            return await self.model_provider.get_chat_completion(
                tools=self.tool_signatures if self.tool_use else [], **kwargs
            )
        except Exception as e:
            self.logger.error(f"Chat Completion Error: {e}")
            return

    async def execute_tool(self, tool_call):
        tool_name = tool_call["function"]["name"]
        self.logger.info(f"Executing tool: {tool_name}")
        if tool_name in self.tool_map.keys():
            tool_function = self.tool_map.get(tool_name, None)
            params = (
                tool_call["function"]["arguments"]
                if type(tool_call["function"]["arguments"]) is dict
                else json.loads(tool_call["function"]["arguments"])
            )
            result = await tool_function(**params)
            self.tool_history.append(
                {"name": tool_name, "params": params, "result": result}
            )
            return result

    async def run_task(self, task_description):
        self.messages.append(
            {
                "role": "user",
                "content": f"""
                Complete the following <task> to the best of your ability.
                
                <task>{task_description}</task>

                {f"<instructions>{self.agent_instructions}</instructions>" if self.agent_instructions else ""}
                
                """.strip(),
            }
        )
        result = await self.chat()
        if not result:
            return
        if result.get("message") and result.get("message", {}).get("tool_calls"):
            self.logger.info("Using tools to assist with task...\n")
            for tool_call in result["message"]["tool_calls"]:
                tool_result = await self.execute_tool(tool_call=tool_call)
                if tool_result:
                    print(f"Tool Call result: {tool_result}")
                    self.messages.append(
                        {
                            **(
                                {"tool_call_id": tool_call["id"]}
                                if tool_call.get("id")
                                else {}
                            ),
                            "role": "tool",
                            "content": f"{tool_result}",
                        }
                    )
                    self.logger.info("Thinking how to solve task...")
                    return await self.chat()
        else:
            return result

    async def reflect(self, task_description, previous_task, result):
        reflect_system_prompt = TASK_REFLECTION_SYSTEM_PROMPT
        reflect_prompt = """
        <task>{task_description}</task>
        <result>{result}</result>
        <tool_history>{tool_history}</tool_history>
        <previous_task>{previous_task}</previous_task>
        <reason>{reason}</reason>
               """.format(
            previous_task=previous_task,
            task_description=task_description,
            result=result["message"]["content"],
            reason=(
                self.reflection_observations[-1]["reason"]
                if self.reflection_observations
                else ""
            ),
            tool_history=json.dumps(self.tool_history, indent=2),
        )
        user_prompt = reflect_prompt
        if "system" not in [msg["role"] for msg in self.reflection_messages]:
            self.reflection_messages.append(
                {
                    "role": "system",
                    "content": reflect_system_prompt,
                }
            )
        self.reflection_messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )
        reflection = await self.chat(
            messages=self.reflection_messages,
            tool_use=False,
            format="json",
            # options={"temperature": 0, "num_gpu": 256},
        )
        return reflection

    async def start(self, initial_task, max_iterations=None):
        self.initial_task = initial_task
        running_task = initial_task
        self.logger.info(f"Starting New Task: {running_task}")
        while True:
            self.iterations += 1
            print(f"Running Iteration: {self.iterations}")
            result = await self.run_task(running_task)
            if not result:
                self.logger.info(f"{self.name} Failed\n")
                self.logger.info(f"Task: {initial_task}\n")
                break
            if self.iterations == max_iterations:
                self.logger.info(f"{self.name} Task max iterations reached\n")
                return result["message"]["content"]
            # print(result["message"]["content"])
            reflection = await self.reflect(
                initial_task, previous_task=running_task, result=result
            )
            if not reflection:
                self.logger.info(f"{self.name} Failed\n")
                self.logger.info(f"Task: {initial_task}\n")
                break
            if reflection["message"]["content"]:
                reflection = json.loads(reflection["message"]["content"])
                if bool(reflection.get("completed")):
                    self.logger.info(
                        f"{self.name} Task Completed Successfully because: {reflection['reason']}\n"
                    )
                    return result["message"]["content"]
                elif reflection.get("refined_task"):
                    if reflection.get("reason"):
                        self.reflection_observations.append(reflection)
                        self.logger.info(
                            f"Refining Task due to the following Reason: {reflection['reason'].strip()}\n"
                        )
                    self.logger.info(f"Refined Task: {reflection['refined_task']}")
                    running_task = reflection["refined_task"]
                    continue
