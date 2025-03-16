import json
from typing import List, Any

from pydantic import BaseModel
from prompts.agent_prompts import (
    REACT_AGENT_SYSTEM_PROMPT,
    TASK_REFLECTION_SYSTEM_PROMPT,
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
        reflections: List[Any] = [],
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
        self.reflections: List[Any] = reflections
        self.messages: list = [{"role": "system", "content": REACT_AGENT_SYSTEM_PROMPT}]

    async def _run_task(self, task, tool_use: bool = True) -> dict | None:
        final_iteration = self.iterations >= (self.max_iterations or 0)
        working_task = self.reflections[-1]["next_step"] if self.reflections else task
        self.agent_logger.debug(f"MEMORY: {self.memory}")
        self.agent_logger.debug(f"MAIN TASK: {task}")
        self.agent_logger.debug(f"WORKING TASK: {working_task}")
        self.messages.append(
            {
                "role": "user",
                "content": f"""
                {"** This is the final iteration provide your best final response to the main task using everything you have learned **" if final_iteration else ""}
                {f"Your main task: { task }"}

                {f"Your working task: { working_task }" }
                 
                """.strip(),
            }
        )
        result = await self._think_chain(tool_use=tool_use)
        return result

    async def _reflect(self, task_description, result):
        class format(BaseModel):
            completion_score: float
            reason: str
            next_step: str

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

        self.agent_logger.info("Reflecting...")
        reflection = await self._think_chain(
            # model="deepseek-r1:14b",
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
            tool_use=False,
            remember_messages=False,
        )
        return {
            "response": (reflection["message"]["content"] if reflection else ""),
        }

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
            if self.reflect:
                reflection = await self._reflect(task, result=result)
                if not reflection:
                    self.agent_logger.info(f"{self.name} Failed\n")
                    self.agent_logger.info(f"Task: {task}\n")
                    break
                reflection = (
                    json.loads(reflection["response"]) if reflection["response"] else {}
                )
                self.agent_logger.info(
                    f"Percent Complete: {reflection.get('completion_score', 0)}%"
                )
                if reflection.get("completion_score", 0) >= self.min_completion_score:
                    self.agent_logger.info(
                        f"{self.name} Task Completed Successfully because: {reflection['reason']}\n"
                    )
                    return result["message"]["content"]
                else:
                    self.agent_logger.debug(
                        f"Iterating task again because: {reflection['reason']}\n"
                    )

                    self.reflections.append(
                        {
                            "result": result["message"]["content"],
                            "failed_reason": reflection["reason"],
                            "completion_score": reflection["completion_score"],
                            "next_step": reflection["next_step"],
                            "tool": (
                                self.tool_history[-1] if self.tool_history else None
                            ),
                        }
                    )
        best_reflection = self.reflections.index(
            max(self.reflections, key=lambda x: x["completion_score"])
        )
        self.result_logger.debug(
            f"""**Best Reflection**: 
            Reflection Reason: {self.reflections[best_reflection]['failed_reason']}
            Reflection Completion Score: {self.reflections[best_reflection]['completion_score']}%
            Reflection Next Step: {self.reflections[best_reflection]['next_step']}"""
        )
        return self.reflections[best_reflection]["result"] if result else None
