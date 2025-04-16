from __future__ import annotations
import json
import traceback
from typing import List, Dict, Any, Optional, Union, Sequence
from loggers.base import Logger
from model_providers.base import BaseModelProvider
from prompts.agent_prompts import (
    TASK_PLANNING_SYSTEM_PROMPT,
    TOOL_ACTION_SUMMARY_PROMPT,
    AGENT_ACTION_PLAN_PROMPT,
)
from model_providers.factory import ModelProviderFactory
from pydantic import BaseModel

from agent_mcp.client import MCPClient
from tools.abstractions import ToolProtocol, MCPToolWrapper, ToolResult


class Agent:
    def __init__(
        self,
        name: str,
        provider_model: str,
        mcp_client: Optional[MCPClient] = None,
        instructions: str = "",
        tools: Sequence[ToolProtocol] = (),  # Use empty tuple as default
        tool_use: bool = True,
        min_completion_score: float = 1.0,
        max_iterations: Optional[int] = None,
        log_level: str = "info",
        workflow_context: Optional[Dict[str, Any]] = None,
    ):
        try:
            ## Agent Attributes
            self.name: str = name
            self.model_provider: BaseModelProvider = (
                ModelProviderFactory.get_model_provider(provider_model)
            )
            self.initial_task: str = ""
            self.final_answer: Optional[str] = ""
            self.instructions: str = instructions
            self.mcp_client = mcp_client
            self.tool_use: bool = tool_use

            # Handle tools based on whether we have an MCP client or manual tools
            self.tools = ()  # Use empty tuple for initialization
            self.memory: list = []
            self.task_progress: str = ""
            self.messages: list = [{"role": "system", "content": self.instructions}]
            self.plan_prompt: str = ""
            self.min_completion_score = min_completion_score
            self.max_iterations = max_iterations
            self.iterations: int = 0

            # Initialize loggers
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
                name=f"{self.name} Result",
                type="agent_response",
                level=log_level,
            )

            self.tool_signatures: List[Dict[str, Any]] = []

            if self.mcp_client:
                # Import at runtime to avoid circular dependency
                from tools.abstractions import MCPToolWrapper

                self.tools = [
                    MCPToolWrapper(t, self.mcp_client) for t in self.mcp_client.tools
                ]
                self.tool_signatures = self.mcp_client.tool_signatures
            elif tools:
                self.tools = tools
                self.tool_signatures = [
                    tool.tool_definition
                    for tool in tools
                    if hasattr(tool, "tool_definition")
                ]

            self.tool_history: List[Dict[str, Any]] = []

            # Only initialize workflow context if provided
            if workflow_context is not None:
                self.workflow_context = workflow_context
                # Initialize agent's own context if workflow tracking is enabled
                if self.name:
                    if self.name not in self.workflow_context:
                        self.workflow_context[self.name] = {
                            "status": "initialized",
                            "current_progress": "",
                            "iterations": 0,
                            "dependencies_met": True,
                        }

            self.agent_logger.info(f"{self.name} Initialized")
            self.agent_logger.info(
                f"Provider:Model: {self.model_provider.name}:{self.model_provider.model}"
            )
            if self.mcp_client:
                self.agent_logger.info(
                    f"Connected MCP Servers: {', '.join(list(self.mcp_client.server_tools.keys()))}"
                )
            self.agent_logger.info(
                f"Available Tools: {', '.join([tool.name for tool in self.tools])}"
            )
        except Exception as e:
            # Get the full stack trace
            stack_trace = "".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
            self.agent_logger.error(
                f"Initialization Error: {str(e)}\nStack trace:\n{stack_trace}"
            )
            raise  # Re-raise the exception after logging

    async def _think(self, **kwargs) -> dict | None:
        try:
            self.agent_logger.info("Thinking...")
            return await self.model_provider.get_completion(**kwargs)
        except Exception as e:
            self.agent_logger.error(f"Completion Error: {e}")
            return

    async def _plan(self, **kwargs) -> str | None:
        class format(BaseModel):
            next_step: str

        try:
            self.agent_logger.info("Planning...")
            self.plan_prompt = """
            <context>
                <main-task> {task} </main-task>
                <available-tools> {tools} </available-tools>
                <previous-steps-performed> {steps} </previous-steps-performed>
            </context>
            
            """.format(
                task=self.initial_task,
                tools=[
                    f"Name: {tool['function']['name']}\nParameters: {tool['function']['parameters']}"
                    for tool in self.tool_signatures
                ],
                steps="\n".join(
                    [
                        f"Step {index}: {step}"
                        for index, step in enumerate(self.memory, start=1)
                    ]
                ),
            )
            result = await self.model_provider.get_completion(
                system=AGENT_ACTION_PLAN_PROMPT,
                prompt=self.plan_prompt,
                format=(
                    format.model_json_schema()
                    if self.model_provider.name == "ollama"
                    else "json"
                ),
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
            if self.workflow_context:
                # Include workflow context in messages if available
                context_message = {
                    "role": "system",
                    "content": f"Previous workflow steps:\n{json.dumps(self.workflow_context, indent=2)}",
                }
                if context_message not in self.messages:
                    self.messages.insert(1, context_message)

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
                    f"Tool Call Already Processed: {tool_call['function']['name']}"
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

    async def _use_tool(self, tool_call) -> Union[str, List[str], None]:
        tool_name = tool_call["function"]["name"]
        params = (
            tool_call["function"]["arguments"]
            if type(tool_call["function"]["arguments"]) is dict
            else json.loads(tool_call["function"]["arguments"])
        )

        try:
            # Find the tool in our unified tool sequence
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                available_tools = [t.name for t in self.tools]
                self.tool_logger.error(
                    f"Tool {tool_name} not found in available tools: {available_tools}"
                )
                return None

            self.tool_logger.info(
                f"Executing tool: {tool_name} with parameters: {params}"
            )

            # Use the tool and get standardized result
            result = await tool.use(params)
            if not isinstance(result, ToolResult):
                result = ToolResult.wrap(result)
            self.tool_logger.debug(f"Tool Result: {result.to_list()}")
            # Handle final answer
            if tool_name == "final_answer":
                self.final_answer = result.to_string()

            # Record tool history
            self.tool_history.append(
                {"name": tool_name, "params": params, "result": result.to_list()}
            )

            # Generate tool summary
            tool_action_summary = (
                await self.model_provider.get_completion(
                    system=TOOL_ACTION_SUMMARY_PROMPT,
                    prompt=f"""
                <context>
                    <tool_call>Tool Name: '{tool_name}' with parameters '{params}'</tool_call>
                    <tool_call_result>{result.to_list()}</tool_call_result>
                </context>
                """,
                )
            ).get("response")

            self.tool_logger.debug(f"Tool Action Summary: {tool_action_summary}")
            self.memory.append(tool_action_summary)
            self.task_progress = "\n".join(
                [
                    f"Step {index}: {step}"
                    for index, step in enumerate(self.memory, start=1)
                ]
            )

            return result.to_list()

        except Exception as e:
            self.tool_logger.error(f"Tool Error: {e}")
            return f"Tool Error: {e}"

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
        self.iterations = 0

        try:
            while True:
                if (
                    self.max_iterations is not None
                    and self.iterations >= self.max_iterations
                ):
                    self.agent_logger.info(
                        f"Reached maximum iterations ({self.max_iterations})"
                    )
                    break

                self.iterations += 1
                self.agent_logger.info(f"Running Iteration: {self.iterations}")

                result = await self._run_task(task=running_task)
                print(result)
                if not result:
                    self.agent_logger.info(f"{self.name} Failed\n")
                    self.agent_logger.info(f"Task: {task}\n")
                    break

                if self.final_answer:
                    return self.final_answer

                # Add continuation check
                if self.iterations > 1 and result.get("message", {}).get(
                    "content"
                ) == result.get("previous_content"):
                    self.agent_logger.info(
                        "No progress made in this iteration, stopping"
                    )
                    break

                result["previous_content"] = result.get("message", {}).get("content")

            return result["message"]["content"] if result else None

        except Exception as e:
            self.agent_logger.error(f"Iteration error: {str(e)}")
            return None
        finally:
            if (
                self.max_iterations is not None
                and self.iterations >= self.max_iterations
            ):
                self.agent_logger.info(
                    f"Reached maximum iterations ({self.max_iterations})"
                )

    async def run(self, initial_task):
        try:
            self.initial_task = initial_task
            self.agent_logger.info(f"Starting task: {initial_task}")

            result = await self._run_task_iteration(task=initial_task)

            if result:
                self.result_logger.info(result)
                return result
            else:
                self.agent_logger.warning("Task completed without result")
                return None

        except Exception as e:
            self.agent_logger.error(f"Agent Error: {e}")
            return None
        finally:
            if self.mcp_client:
                try:
                    await self.mcp_client.close()
                except Exception as cleanup_error:
                    self.agent_logger.error(f"Cleanup error: {cleanup_error}")

    def set_model_provider(self, provider_model: str):
        self.model_provider = ModelProviderFactory.get_model_provider(provider_model)
        self.agent_logger.info(f"Model Provider Set to: {self.model_provider.name}")
