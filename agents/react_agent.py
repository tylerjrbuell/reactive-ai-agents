import json
from typing import List, Dict, Any

from pydantic import BaseModel
from tools.base import Tool
from model_providers.base import BaseModelProvider
from model_providers.factory import ModelProviderFactory
from prompts.agent_prompts import (
    REACT_AGENT_SYSTEM_PROMPT,
    TASK_REFLECTION_SYSTEM_PROMPT,
    TASK_PLANNING_SYSTEM_PROMPT,
)
from agents.base import Agent


class ReactAgent(Agent):
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
        super().__init__(
            name=name,
            provider_model=provider_model,
            instructions=instructions,
            tools=tools,
            tool_use=tool_use,
            min_completion_score=min_completion_score,
            max_iterations=max_iterations,
            log_level=log_level,
        )
        self.reflect: bool = reflect
        self.messages: list = [{"role": "system", "content": REACT_AGENT_SYSTEM_PROMPT}]

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
            result = await self._run_task(task=running_task)
            if not result:
                self.logger.info(f"{self.name} Failed\n")
                self.logger.info(f"Task: {task}\n")
                break
            if self._reflect:
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
