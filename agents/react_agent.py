from __future__ import annotations
import json
import traceback
import asyncio
import os
from datetime import datetime
from enum import Enum, auto
from typing import List, Any, Optional, Dict, Set, Union, Tuple, Callable
import time
import random

from pydantic import BaseModel, Field
from agent_mcp.client import MCPClient
from prompts.agent_prompts import (
    PERCENTAGE_COMPLETE_TASK_REFLECTION_SYSTEM_PROMPT,
    REACT_AGENT_SYSTEM_PROMPT,
    AGENT_ACTION_PLAN_PROMPT,
    MISSING_TOOLS_PROMPT,
)
from agents.base import Agent
from tools.base import Tool as BaseTool
from tools.abstractions import ToolProtocol


class TaskStatus(Enum):
    """Standardized task status values"""

    INITIALIZED = "initialized"
    WAITING_DEPENDENCIES = "waiting_for_dependencies"
    RUNNING = "running"
    MISSING_TOOLS = "missing_tools"
    COMPLETE = "complete"
    RESCOPED_COMPLETE = "rescoped_complete"
    MAX_ITERATIONS = "max_iterations_reached"
    ERROR = "error"
    CANCELLED = "cancelled"

    def __str__(self):
        return self.value


class AgentMemory(BaseModel):
    """Model for agent memory storage"""

    agent_name: str
    session_history: List[Dict[str, Any]] = []
    tool_preferences: Dict[str, Any] = {}
    user_preferences: Dict[str, Any] = {}
    reflections: List[Dict[str, Any]] = []
    last_updated: datetime = Field(default_factory=datetime.now)


class ReactAgent(Agent):
    def __init__(
        self,
        name: str,
        provider_model: str,
        mcp_client: Optional[MCPClient] = None,
        instructions: str = "",
        role: str = "",
        role_instructions: Dict[str, Any] = {},
        tools: List[Any] = [],
        tool_use: bool = True,
        reflect: bool = False,
        reflections: Optional[List[Any]] = None,
        min_completion_score: float = 1.0,
        max_iterations: Optional[int] = None,
        log_level: str = "info",
        workflow_context: Optional[Dict[str, Any]] = None,
        workflow_dependencies: Optional[List[str]] = None,
        use_memory: bool = True,
        enable_caching: bool = True,
        cache_ttl: int = 3600,  # 1 hour default TTL
        offline_mode: bool = False,
        retry_config: Optional[Dict[str, Any]] = None,
        collect_metrics: bool = True,
    ):
        try:
            self.role = role
            self.role_instructions = role_instructions or {}

            # Combine base instructions with role-specific instructions
            combined_instructions = instructions
            if role and role_instructions:
                role_specific_instructions = role_instructions.get(role, "")
                if role_specific_instructions:
                    combined_instructions = (
                        f"{instructions}\nRole: {role}\n{role_specific_instructions}"
                    )
            # Initialize base Agent
            super().__init__(
                name=name,
                provider_model=provider_model,
                mcp_client=mcp_client,
                instructions=combined_instructions,
                tools=tools,
                tool_use=tool_use,
                min_completion_score=min_completion_score,
                max_iterations=max_iterations,
                log_level=log_level,
                workflow_context=workflow_context,
            )

            # ReactAgent specific attributes
            self.reflect = reflect
            self.workflow_dependencies = workflow_dependencies or []
            self.previous_content = None
            self.final_answer = None
            self.task_progress = ""

            # Initialize reflections with a proper empty list
            self.reflections = []

            # Initialize reflections either from parameter or workflow context
            try:
                if workflow_context is not None and name in workflow_context:
                    # Get reflections from workflow context if available
                    self.reflections = workflow_context[name].get("reflections", [])
                    if self.reflections is None:  # Ensure we never have None
                        self.reflections = []
                elif reflections is not None:
                    self.reflections = reflections
            except Exception as refl_error:
                stack_trace = "".join(
                    traceback.format_exception(
                        type(refl_error), refl_error, refl_error.__traceback__
                    )
                )
                self.agent_logger.error(
                    f"Error initializing reflections: {str(refl_error)}\nStack trace:\n{stack_trace}"
                )
                self.reflections = []  # Fallback to empty list on error

            self.messages = []

            # Initialize workflow context
            if workflow_context is not None:
                try:
                    self.workflow_context = workflow_context
                    if name in workflow_context:
                        # Update existing context
                        if "reflections" not in workflow_context[name]:
                            workflow_context[name]["reflections"] = self.reflections
                    else:
                        # Create new context
                        workflow_context[name] = {
                            "status": "initialized",
                            "current_progress": "",
                            "iterations": 0,
                            "dependencies_met": True,
                            "reflections": self.reflections,
                        }
                except Exception as ctx_error:
                    stack_trace = "".join(
                        traceback.format_exception(
                            type(ctx_error), ctx_error, ctx_error.__traceback__
                        )
                    )
                    self.agent_logger.error(
                        f"Error initializing workflow context: {str(ctx_error)}\nStack trace:\n{stack_trace}"
                    )
            else:
                # For standalone mode without workflow
                self.workflow_context = None

            self.agent_logger.debug(
                f"ReactAgent initialized with reflections: {self.reflections}"
            )
            self.agent_logger.debug(
                f"Workflow context for {name}: {workflow_context[name] if workflow_context and name in workflow_context else None}"
            )
            self.agent_logger.debug(f"Workflow Context: {self.workflow_context}")

            # Initialize memory if enabled
            self.use_memory = use_memory
            if use_memory:
                self._init_memory()

            # Add tool caching
            self.enable_caching = enable_caching
            self.cache_ttl = cache_ttl
            self.tool_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0

            # Offline mode and network resilience
            self.offline_mode = offline_mode
            self.retry_config = retry_config or {
                "max_retries": 3,
                "base_delay": 1.0,
                "max_delay": 10.0,
                "retry_network_errors": True,
            }

            # Set up network status tracking
            self.network_errors = 0
            self.last_network_check = time.time()
            self.network_available = not offline_mode

            if offline_mode:
                self.agent_logger.info(
                    "Agent running in offline mode - will use local models and cached tools only"
                )

                # Force enable caching in offline mode
                self.enable_caching = True

                # If using Ollama or other local provider, ensure it's configured
                if not any(
                    p in provider_model.lower() for p in ["ollama", "local", "offline"]
                ):
                    self.agent_logger.warning(
                        f"Offline mode enabled but provider '{provider_model}' may require network access"
                    )

            # Metrics collection for ZenRunners dashboard
            self.collect_metrics = collect_metrics
            self.metrics = {
                "start_time": time.time(),
                "end_time": None,
                "total_time": 0,
                "status": "initialized",
                "tool_calls": 0,
                "tool_errors": 0,
                "iterations": 0,
                "tokens": {
                    "prompt": 0,
                    "completion": 0,
                    "total": 0,
                },
                "model_calls": 0,
                "tools": {},
                "cache": {
                    "hits": 0,
                    "misses": 0,
                    "ratio": 0.0,
                },
                "latency": {
                    "avg_tool_latency": 0,
                    "max_tool_latency": 0,
                    "avg_model_latency": 0,
                    "max_model_latency": 0,
                    "tool_time": 0,
                    "model_time": 0,
                },
            }

        except Exception as e:
            stack_trace = "".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
            try:
                self.agent_logger.error(
                    f"ReactAgent initialization error: {str(e)}\nStack trace:\n{stack_trace}"
                )
            except:
                # Fallback if logger isn't initialized yet
                print(
                    f"Critical initialization error: {str(e)}\nStack trace:\n{stack_trace}"
                )
            raise

    def check_dependencies(self) -> bool:
        """Check if all workflow dependencies are satisfied"""
        # If no workflow context or dependencies, consider dependencies satisfied
        if not self.workflow_dependencies or self.workflow_context is None:
            return True

        for dep in self.workflow_dependencies:
            dep_status = self.workflow_context.get(dep, {}).get("status")
            if dep_status != "complete":
                self.agent_logger.info(
                    f"Waiting for dependency {dep} to complete (current status: {dep_status})"
                )
                return False
        return True

    async def _run_task(self, task, tool_use: bool = True) -> dict | None:
        system_message = {
            "role": "system",
            "content": REACT_AGENT_SYSTEM_PROMPT.format(
                task=task,
                task_progress=(
                    self.task_progress if hasattr(self, "task_progress") else ""
                ),
                role=self.role or "AI Assistant",
                instructions=(
                    self.role_instructions.get(self.role, "")
                    if self.role
                    else self.instructions
                ),
            ),
        }

        # Initialize or update messages list
        if not self.messages:
            self.messages = [system_message]
        else:
            self.messages[0] = system_message

        self.agent_logger.debug(f"Task Progress:\n{self.task_progress}")

        try:
            if hasattr(self, "reflections") and self.reflections:
                last_reflection = self.reflections[-1]
                feedback = (
                    last_reflection.get("reason", "")
                    if isinstance(last_reflection, dict)
                    else ""
                )
                next_step = (
                    last_reflection.get("next_step", "")
                    if isinstance(last_reflection, dict)
                    else ""
                )

                if feedback or next_step:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": f"""
                            
                            {feedback}
                            """,
                        }
                    )
                else:
                    self.messages.append({"role": "user", "content": task})
            else:
                self.messages.append({"role": "user", "content": task})

            result = await self._think_chain(tool_use=tool_use)
            if result:
                self.agent_logger.debug(f"RESULT: {result['message']['content']}")

            return result
        except (IndexError, KeyError, AttributeError) as e:
            self.agent_logger.warning(f"Error accessing reflection data: {str(e)}")
            # Reset messages to ensure we have a clean state
            self.messages = [system_message, {"role": "user", "content": task}]
            return await self._think_chain(tool_use=tool_use)

    async def _reflect(self, task_description, result):
        """Generate a reflection on the current state of task execution

        Args:
            task_description: The task being executed
            result: The result of the last execution step

        Returns:
            Dictionary with reflection data including completion_score, next_step, etc.
        """
        try:
            self.agent_logger.info("Reflecting on task progress...")

            class ReflectionFormat(BaseModel):
                completion_score: float = Field(..., ge=0.0, le=1.0)
                next_step: str
                required_tools: List[str]
                completed_tools: List[str]
                reason: str

            # Extract content from result if available
            result_content = ""
            if result:
                if isinstance(result, dict):
                    if "message" in result and "content" in result["message"]:
                        result_content = result["message"]["content"]
                    else:
                        result_content = str(result)
                else:
                    result_content = str(result)

            # Build reflection context
            reflection_context = {
                "task": task_description,
                "result": result_content,
                "tools_available": [t.name for t in self.tools],
                "tools_used": (
                    [t["name"] for t in self.tool_history] if self.tool_history else []
                ),
                "completion_criteria": self.min_completion_score,
                "current_iteration": self.iterations,
                "max_iterations": self.max_iterations,
            }

            # Get reflection from model
            result = await self.model_provider.get_completion(
                system="""
                You are a reflection assistant that evaluates task progress.
                
                For the given task and current result, you need to:
                1. Evaluate how complete the task is (0.0 to 1.0)
                2. Identify what tools were required
                3. Track which tools have been used successfully
                4. Suggest the next step (if any)
                5. Provide a clear reason for your assessment
                
                Assign a completion_score of 1.0 ONLY if the task has been fully completed.
                Assign 0.0-0.9 for partial completion based on how much meaningful progress has been made.
                """,
                prompt=json.dumps(reflection_context, indent=2),
                format=(
                    ReflectionFormat.model_json_schema()
                    if self.model_provider.name == "ollama"
                    else "json"
                ),
            )

            if not result or "response" not in result:
                self.agent_logger.warning("Reflection failed to produce output")
                return None

            # Parse and return the reflection data
            try:
                reflection_data = (
                    json.loads(result["response"])
                    if isinstance(result["response"], str)
                    else result["response"]
                )
                return reflection_data
            except (json.JSONDecodeError, AttributeError) as e:
                self.agent_logger.error(f"Error parsing reflection data: {e}")
                return None

        except Exception as e:
            self.agent_logger.error(f"Reflection error: {e}")
            return None

    async def _plan(self, task: str) -> dict | None:
        class PlanFormat(BaseModel):
            next_step: str
            continue_iteration: bool
            rationale: str
            suggested_tools: List[str] = []

        try:
            self.agent_logger.info("Planning next action...")

            # Safely get previous reflections
            if getattr(self, "reflections", None):
                previous_reflections = (
                    self.reflections[-3:]
                    if len(self.reflections) >= 3
                    else self.reflections
                )
            else:
                previous_reflections = []

            context = {
                "task": task,
                "workflow_context": getattr(self, "workflow_context", {}),
                "available_tools": [
                    f"{tool['function']['name']}: {tool['function']['parameters']}"
                    for tool in self.tool_signatures
                ],
                "previous_steps": self.task_progress,
                "previous_reflections": previous_reflections,
                "completion_criteria": self.min_completion_score,
                "current_iteration": self.iterations,
                "max_iterations": self.max_iterations,
            }

            result = await self.model_provider.get_completion(
                system=AGENT_ACTION_PLAN_PROMPT,
                prompt=json.dumps(context, indent=2),
                format=(
                    PlanFormat.model_json_schema()
                    if self.model_provider.name == "ollama"
                    else "json"
                ),
            )

            if result and result.get("response"):
                plan_data = json.loads(result["response"])
                self.agent_logger.debug(f"Action plan: {plan_data}")
                return plan_data

            return None

        except Exception as e:
            self.agent_logger.error(f"Planning error: {str(e)}")
            return None

    def should_continue(self) -> bool:
        """Determine if the agent should continue based on task completion criteria"""
        # If we already have a terminal status, don't continue
        if hasattr(self, "task_status") and self.task_status in [
            TaskStatus.COMPLETE,
            TaskStatus.ERROR,
            TaskStatus.RESCOPED_COMPLETE,
            TaskStatus.MAX_ITERATIONS,
        ]:
            return False

        # Check basic conditions
        if not self.reflect or not self.check_dependencies():
            return False

        if self.max_iterations and self.iterations >= self.max_iterations:
            self.agent_logger.info("Maximum iterations reached")
            if hasattr(self, "task_status"):
                self.task_status = TaskStatus.MAX_ITERATIONS
            return False

        # We have a final answer, no need to continue
        if self.final_answer:
            if hasattr(self, "task_status"):
                self.task_status = TaskStatus.COMPLETE
            return False

        # Check if we have reflections to analyze
        if not hasattr(self, "reflections") or not self.reflections:
            return True

        try:
            last_reflection = self.reflections[-1]
            if not isinstance(last_reflection, dict):
                return True

            # Check completion based on reflection score and tool usage
            score = last_reflection.get("completion_score", 0.0)
            required_tools = last_reflection.get("required_tools", [])
            completed_tools = last_reflection.get("completed_tools", [])

            if required_tools and completed_tools:
                tools_completion_ratio = len(completed_tools) / len(required_tools)
                effective_score = min(score, tools_completion_ratio)

                if effective_score >= self.min_completion_score:
                    if hasattr(self, "task_status"):
                        self.task_status = TaskStatus.COMPLETE
                    return False

            # Check for lack of progress
            if self.iterations > 1 and self.previous_content:
                if self.task_progress and len(self.task_progress) >= 2:
                    if self.task_progress[-1] == self.task_progress[-2]:
                        self.agent_logger.info(
                            "No progress detected between iterations"
                        )
                        return False
        except (IndexError, KeyError, AttributeError) as e:
            self.agent_logger.warning(f"Error checking continuation status: {str(e)}")
            # On any error accessing reflections or their data, continue if we haven't hit max iterations
            return True

        return True

    async def _run_task_iteration(self, task):
        """Execute a task with reflection and planning"""
        self.agent_logger.info("🔄 STARTING EXECUTION")

        if not self.reflect:
            self.agent_logger.info("⏩ Running in direct mode (no reflection)")
            try:
                # Run the task directly
                result = await self._run_task(task)

                # Extract content from result
                if result is not None and isinstance(result, dict):
                    if "message" in result and result["message"] is not None:
                        if "content" in result["message"]:
                            content = result["message"]["content"]
                            if content:
                                # Check for final answer pattern
                                content_lower = content.lower().strip()
                                if content_lower.startswith(
                                    "final answer:"
                                ) or content_lower.startswith("answer:"):
                                    self.final_answer = content
                                return content
                    # Try other common result patterns
                    for key in ["response", "content", "output", "result"]:
                        if key in result and result[key]:
                            return str(result[key])

                # Return string representation as fallback
                return (
                    str(result)
                    if result is not None
                    else "Task completed without result"
                )
            except Exception as e:
                self.agent_logger.error(f"Error in direct mode execution: {str(e)}")
                return f"Error during execution: {str(e)}"

        if not self.check_dependencies():
            self.agent_logger.warning("⚠️ Dependencies not met, cannot proceed")
            return None

        # Reset iteration state
        self.iterations = 0
        self.previous_content = None
        last_error = None
        last_reflection = None
        final_result = None

        # Ensure we have a valid reflections list
        if not hasattr(self, "reflections") or self.reflections is None:
            self.reflections = []
            # Update workflow context if it exists
            if self.workflow_context is not None and self.name in self.workflow_context:
                self.workflow_context[self.name]["reflections"] = self.reflections

        # Task execution loop
        while self.should_continue():
            # Log the current iteration
            self.iterations += 1
            self.agent_logger.info(
                f"🔄 ITERATION {self.iterations}/{self.max_iterations or 'unlimited'}"
            )

            # Update workflow context if available
            if self.workflow_context is not None and self.name in self.workflow_context:
                self.workflow_context[self.name]["iterations"] = self.iterations

            # Get dependencies
            try:
                # Execute the task
                self.agent_logger.info("🎯 Running task")
                result = await self._run_task(task)
                final_result = result  # Keep track of the last result

                if not result:
                    self.agent_logger.warning("⚠️ Task execution returned no result")
                    break

                # Check for final answer
                if self.final_answer:
                    self.agent_logger.info("✅ Final answer received")
                    return self.final_answer

                # Reflect on the current progress
                self.agent_logger.info("🤔 Reflecting on task progress...")
                reflection = await self._reflect(task, result)

                if not reflection or not isinstance(reflection, dict):
                    self.agent_logger.warning(
                        "⚠️ Reflection failed, continuing without it"
                    )
                    continue

                # Store reflection
                self.reflections.append(reflection)
                last_reflection = reflection

                # Update workflow context if it exists
                if (
                    self.workflow_context is not None
                    and self.name in self.workflow_context
                ):
                    # Consider making a copy to avoid shared references
                    self.workflow_context[self.name]["reflections"] = self.reflections

                # Check if we're done based on completion score
                score = reflection.get("completion_score", 0)
                self.agent_logger.info(
                    f"📊 Task completion score: {score:.2f} / {self.min_completion_score:.2f} required"
                )
                self.agent_logger.debug(
                    f"📋 Reflection feedback: {reflection.get('reason', 'No reason provided')}"
                )
                self.agent_logger.debug(
                    f"🔧 Required tools: {reflection.get('required_tools', [])}"
                )
                self.agent_logger.debug(
                    f"✅ Completed tools: {reflection.get('completed_tools', [])}"
                )
                self.agent_logger.debug(
                    f"👉 Next step: {reflection.get('next_step', '')}"
                )

                # If score meets or exceeds target, we're done
                if score >= self.min_completion_score:
                    self.agent_logger.info(f"✅ Task complete with score {score:.2f}")
                    break

            except Exception as e:
                last_error = str(e)
                self.agent_logger.error(f"❌ Error in task iteration: {last_error}")
                break

        # Log completion
        self.agent_logger.info(
            f"🏁 Task execution completed after {self.iterations} iterations"
        )

        # Save memory if we've completed and memory is enabled
        if hasattr(self, "agent_memory") and self.agent_memory:
            self.agent_logger.debug("💾 Saving agent memory")
            self.save_memory()

        # Return final state
        if self.final_answer:
            self.agent_logger.info("✅ Final answer received")
            return self.final_answer
        elif (
            final_result
            and isinstance(final_result, dict)
            and "message" in final_result
            and "content" in final_result["message"]
        ):
            self.agent_logger.info("✅ Result content available")
            return final_result["message"]["content"]
        elif last_reflection and "reason" in last_reflection:
            # If we have a reflection with completion score ≥ min_score but no final_answer,
            # return the reasoning as a result
            self.agent_logger.info("✅ Using reflection reason as result")
            return last_reflection.get("reason")
        elif last_error:
            self.agent_logger.warning(f"⚠️ Returning error as result: {last_error}")
            return f"Error: {last_error}"
        else:
            self.agent_logger.info("✅ Task completed without explicit result")
            return "Task completed without explicit result."

    def _update_workflow_context(
        self, status: Union[TaskStatus, str], result: Optional[str], **kwargs
    ):
        """Update workflow context with current status and additional data"""
        if not hasattr(self, "workflow_context"):
            return

        # Skip workflow updates if workflow context isn't being used
        if self.workflow_context is None:
            return

        # Convert string status to TaskStatus if needed
        if isinstance(status, str):
            try:
                status = TaskStatus(status)
            except ValueError:
                # If invalid status string, default to RUNNING
                status = TaskStatus.RUNNING

        # Ensure agent's context exists
        if self.name not in self.workflow_context:
            self.workflow_context[self.name] = {
                "status": TaskStatus.INITIALIZED.value,
                "current_progress": "",
                "iterations": 0,
                "dependencies_met": True,
            }

        context_update = {
            "status": str(status),
            "iterations": self.iterations,
            "dependencies_met": self.check_dependencies(),
        }

        if result:
            context_update[
                (
                    "final_result"
                    if status == TaskStatus.COMPLETE
                    or (isinstance(status, str) and status == "complete")
                    else "current_progress"
                )
            ] = result

        if status == TaskStatus.ERROR:
            context_update["error"] = kwargs.get("error")
        else:
            if "completion_score" in kwargs:
                context_update["completion_score"] = kwargs["completion_score"]
            if "plan" in kwargs:
                context_update["last_action_plan"] = kwargs["plan"]
            if "reflection_data" in kwargs:
                context_update["last_reflection"] = kwargs["reflection_data"]

        self.workflow_context[self.name].update(context_update)

    async def check_tool_feasibility(self, task: str) -> Dict[str, Any]:
        """
        Check if all required tools for a task are available.

        Args:
            task: The task description

        Returns:
            Dict with 'feasible' (bool) and 'missing_tools' (list) keys
        """
        try:
            # Get available tool names
            available_tools = {tool.name for tool in self.tools}

            # Ask the model to analyze required tools for this task
            system_prompt = MISSING_TOOLS_PROMPT
            prompt = f"""
            <context>
            <task>{task}</task>
            <available_tools>{list(available_tools)}</available_tools>
            </context>
            """

            class ToolAnalysisFormat(BaseModel):
                required_tools: List[str] = Field(
                    ..., description="List of tools required for this task"
                )
                optional_tools: List[str] = Field(
                    ...,
                    description="List of tools that would be helpful but aren't essential",
                )
                explanation: str = Field(
                    ..., description="Brief explanation of the tool requirements"
                )

            result = await self.model_provider.get_completion(
                system=system_prompt,
                prompt=prompt,
                format=(
                    ToolAnalysisFormat.model_json_schema()
                    if self.model_provider.name == "ollama"
                    else "json"
                ),
            )

            if not result or "response" not in result:
                return {
                    "feasible": True,
                    "missing_tools": [],
                    "explanation": "Could not analyze tool requirements",
                }

            # Parse the response
            try:
                tool_analysis = (
                    json.loads(result["response"])
                    if isinstance(result["response"], str)
                    else result["response"]
                )
                required_tools = set(tool_analysis.get("required_tools", []))

                # Find missing tools
                missing_tools = required_tools - available_tools

                return {
                    "feasible": len(missing_tools) == 0,
                    "missing_tools": list(missing_tools),
                    "available_tools": list(available_tools),
                    "required_tools": list(required_tools),
                    "explanation": tool_analysis.get("explanation", ""),
                }
            except (json.JSONDecodeError, AttributeError) as e:
                self.agent_logger.error(f"Error parsing tool analysis: {e}")
                return {
                    "feasible": True,
                    "missing_tools": [],
                    "explanation": "Error analyzing tool requirements",
                }

        except Exception as e:
            self.agent_logger.error(f"Tool feasibility check error: {e}")
            return {
                "feasible": True,
                "missing_tools": [],
                "explanation": f"Error during analysis: {str(e)}",
            }

    async def _safe_close_mcp_client(self):
        """Safely close the MCP client without causing cancellation issues."""
        if not hasattr(self, "mcp_client") or not self.mcp_client:
            return

        self.agent_logger.info("Safely closing MCP client connection")
        try:
            # Create a new task in the same event loop to close the client
            # This ensures the cancel scope is managed properly
            loop = asyncio.get_running_loop()
            close_task = loop.create_task(self.mcp_client.close())

            # Wait for the close task to complete with a timeout
            try:
                await asyncio.wait_for(close_task, timeout=5.0)
                self.agent_logger.info("MCP client closed successfully")
            except asyncio.TimeoutError:
                self.agent_logger.warning(
                    "MCP client close timed out, client may not be fully closed"
                )
            except asyncio.CancelledError:
                # If this task gets cancelled, detach the close task so it can finish
                close_task.add_done_callback(
                    lambda _: self.agent_logger.info(
                        "Detached MCP client close completed"
                    )
                )
                # Don't wait for it since we're being cancelled
                raise
        except Exception as e:
            self.agent_logger.warning(f"Error during safe MCP client close: {str(e)}")
        finally:
            # Clear the reference regardless of success
            self.mcp_client = None

    async def run(self, initial_task, check_tools=True, cancellation_event=None):
        """
        Run the agent with the given task, with optional tool checking.

        Args:
            initial_task: The task to execute
            check_tools: Whether to check if required tools are available
            cancellation_event: Optional asyncio Event that can be set to cancel execution

        Returns:
            Dict with execution results
        """
        # Create a list to track any tasks we spawn
        self._active_tasks = set()

        # Initialize variables to avoid UnboundLocalError
        result_content = None
        summary = "No execution summary available."
        evaluation = {
            "adherence_score": 0.5,
            "matches_intent": False,
            "explanation": "No evaluation performed.",
        }

        # Reset metrics for this run
        if self.collect_metrics:
            self.metrics = {
                "start_time": time.time(),
                "end_time": None,
                "total_time": 0,
                "status": "initialized",
                "tool_calls": 0,
                "tool_errors": 0,
                "iterations": 0,
                "tokens": {
                    "prompt": 0,
                    "completion": 0,
                    "total": 0,
                },
                "model_calls": 0,
                "tools": {},
                "cache": {
                    "hits": 0,
                    "misses": 0,
                    "ratio": 0.0,
                },
                "latency": {
                    "avg_tool_latency": 0,
                    "max_tool_latency": 0,
                    "avg_model_latency": 0,
                    "max_model_latency": 0,
                    "tool_time": 0,
                    "model_time": 0,
                },
            }

        try:
            self.initial_task = initial_task
            self.agent_logger.info(f"Starting task: {initial_task}")

            # Track current task status
            self.task_status = TaskStatus.INITIALIZED

            if self.collect_metrics:
                self._update_metrics("status", {"status": str(self.task_status)})

            # Check if all dependencies are satisfied
            if not self.check_dependencies():
                self.agent_logger.info("Dependencies not met, skipping execution")
                self.task_status = TaskStatus.WAITING_DEPENDENCIES
                if self.workflow_context and self.name in self.workflow_context:
                    self._update_workflow_context(TaskStatus.WAITING_DEPENDENCIES, None)

                if self.collect_metrics:
                    self._update_metrics("status", {"status": str(self.task_status)})

                return {"status": str(TaskStatus.WAITING_DEPENDENCIES), "result": None}

            # Check if we have all needed tools for this task
            if check_tools:
                feasibility = await self.check_tool_feasibility(initial_task)
                if not feasibility["feasible"]:
                    self.agent_logger.warning(
                        f"Missing required tools: {feasibility['missing_tools']}"
                    )

                    # Add this to our reasoning log
                    self.reasoning_log.append(
                        f"I cannot complete this task because I'm missing tools: {feasibility['missing_tools']}. {feasibility['explanation']}"
                    )

                    # Update task status
                    self.task_status = TaskStatus.MISSING_TOOLS

                    # Update workflow context if available
                    if self.workflow_context and self.name in self.workflow_context:
                        self._update_workflow_context(
                            TaskStatus.MISSING_TOOLS,
                            None,
                            missing_tools=feasibility["missing_tools"],
                        )

                    return {
                        "status": str(TaskStatus.MISSING_TOOLS),
                        "result": f"Cannot complete task. Missing tools: {feasibility['missing_tools']}",
                        "missing_tools": feasibility["missing_tools"],
                        "explanation": feasibility["explanation"],
                    }

            # Reset state for a fresh run
            self.iterations = 0
            self.task_progress = ""
            result = None
            failure_count = 0
            max_failures = 3
            rescoped_task = None

            # Update task status to running
            self.task_status = TaskStatus.RUNNING

            # Notify workflow context we're running
            if self.workflow_context and self.name in self.workflow_context:
                self._update_workflow_context(TaskStatus.RUNNING, None)

            # Handle direct execution (non-reflection mode)
            if not self.reflect:
                self.iterations = 1  # Just one iteration in direct mode
                self.agent_logger.info("Running task in direct mode (no reflection)")

                try:
                    current_task = rescoped_task or initial_task
                    result_content = await self._run_task_iteration(current_task)

                    # Always record the result if available
                    if result_content:
                        self.reasoning_log.append(result_content)

                    # Set completion status
                    if self.final_answer:
                        self.task_status = TaskStatus.COMPLETE
                    else:
                        # If we got here with no errors, consider it complete
                        self.task_status = TaskStatus.COMPLETE

                    # Simplified summary generation in non-reflection mode
                    if self.tool_history:
                        summary = await self.generate_summary()
                    else:
                        summary = "Task completed in direct mode with no tool usage."

                    # Generate a basic evaluation
                    evaluation = await self.compare_goal_vs_result()

                    # Update workflow context if available
                    if self.workflow_context and self.name in self.workflow_context:
                        self._update_workflow_context(
                            self.task_status,
                            result_content,
                            adherence_score=evaluation.get("adherence_score", 0.5),
                            matches_intent=evaluation.get("matches_intent", False),
                        )

                except Exception as e:
                    self.task_status = TaskStatus.ERROR
                    error_message = f"Error in direct execution: {str(e)}"
                    self.agent_logger.error(error_message)
                    self.reasoning_log.append(error_message)

                    if self.workflow_context and self.name in self.workflow_context:
                        self._update_workflow_context(TaskStatus.ERROR, error_message)

                    return {"status": str(TaskStatus.ERROR), "result": error_message}
            else:
                # Main execution loop with reflection
                while self.should_continue():
                    # Check for cancellation
                    if cancellation_event and cancellation_event.is_set():
                        self.agent_logger.info("Task execution cancelled by user")
                        self.task_status = TaskStatus.CANCELLED
                        break

                    try:
                        current_task = rescoped_task or initial_task
                        result_content = await self._run_task_iteration(current_task)

                        # Store the result content, which might be the final_answer or the last result
                        if result_content:
                            # Add to reasoning log if it's not already there
                            if result_content not in self.reasoning_log:
                                self.reasoning_log.append(result_content)

                        # Update iteration metrics
                        if self.collect_metrics:
                            self._update_metrics(
                                "iteration", {"count": self.iterations}
                            )

                        if self.final_answer:
                            self.task_status = TaskStatus.COMPLETE
                            break

                        failure_count = (
                            0  # Reset failure counter on successful iteration
                        )

                    except Exception as iter_error:
                        failure_count += 1
                        self.agent_logger.error(f"Iteration error: {str(iter_error)}")

                        # Add to reasoning log
                        self.reasoning_log.append(
                            f"Error during execution: {str(iter_error)}"
                        )

                        if failure_count >= max_failures:
                            # Try to rescope the task if we keep failing
                            if not rescoped_task:  # Only try rescoping once
                                error_context = f"Multiple failures during execution: {str(iter_error)}"
                                rescope_result = await self.rescope_goal(
                                    initial_task, error_context
                                )

                                if rescope_result["rescoped_task"]:
                                    rescoped_task = rescope_result["rescoped_task"]
                                    self.agent_logger.info(
                                        f"Task rescoped: {rescoped_task}"
                                    )
                                    failure_count = (
                                        0  # Reset failure counter after rescoping
                                    )

                                    # Add to reasoning log
                                    self.reasoning_log.append(
                                        f"Task rescoped due to failures: {rescope_result['explanation']}"
                                    )

                                    # Update workflow context
                                    if (
                                        self.workflow_context
                                        and self.name in self.workflow_context
                                    ):
                                        self.workflow_context[self.name][
                                            "rescoped"
                                        ] = True
                                        self.workflow_context[self.name][
                                            "original_task"
                                        ] = initial_task
                                        self.workflow_context[self.name][
                                            "rescoped_task"
                                        ] = rescoped_task
                                    continue
                                else:
                                    # Couldn't rescope, bail out
                                    self.task_status = TaskStatus.ERROR
                                    raise Exception(
                                        f"Multiple execution failures and could not rescope task: {str(iter_error)}"
                                    )
                            else:
                                # Already tried rescoping, bail out
                                self.task_status = TaskStatus.ERROR
                                raise Exception(
                                    f"Multiple execution failures even after rescoping: {str(iter_error)}"
                                )

                # Generate summary of actions taken
                summary = await self.generate_summary()

                # Evaluate how well we met the goal
                evaluation = await self.compare_goal_vs_result()

                # Determine final status
                if self.final_answer:
                    self.task_status = TaskStatus.COMPLETE
                elif rescoped_task:
                    self.task_status = TaskStatus.RESCOPED_COMPLETE
                elif self.max_iterations and self.iterations >= self.max_iterations:
                    self.task_status = TaskStatus.MAX_ITERATIONS
                else:
                    # If we got here successfully but don't have a final answer, consider it complete
                    self.task_status = TaskStatus.COMPLETE

            # If we don't have a final_answer but have a result_content, that's our result
            result_to_return = self.final_answer or result_content

            # Update workflow context if available
            if self.workflow_context and self.name in self.workflow_context:
                self._update_workflow_context(
                    self.task_status,
                    result_to_return,
                    adherence_score=evaluation.get("adherence_score", 0.5),
                    matches_intent=evaluation.get("matches_intent", False),
                    rescoped=rescoped_task is not None,
                )

            # Final metrics update before returning
            if self.collect_metrics:
                # Always ensure we have an end time and the status is correctly set
                end_time = time.time()
                self._update_metrics(
                    "run",
                    {
                        "status": str(self.task_status),
                        "end_time": end_time,
                        "iterations": self.iterations,
                    },
                )

                # Update metrics based on actual tool history
                self._update_metrics_from_history()

                # Update cache metrics
                self._update_metrics(
                    "cache", {"hits": self.cache_hits, "misses": self.cache_misses}
                )

            # Prepare final result
            final_result = {
                "status": str(self.task_status),
                "result": result_to_return,
                "iterations": self.iterations,
                "summary": summary,
                "reasoning_log": self.reasoning_log,
                "evaluation": evaluation,
                "rescoped": rescoped_task is not None,
                "original_task": initial_task,
                "rescoped_task": rescoped_task,
                "metrics": self.get_metrics() if self.collect_metrics else None,
            }

            # Update memory if enabled
            if self.use_memory and hasattr(self, "agent_memory"):
                self.update_session_history(initial_task, final_result)

                # Update tool preference stats for tools used in this session
                for tool_usage in self.tool_history:
                    tool_name = tool_usage.get("name", "unknown")
                    # Simplistic success detection - could be made more sophisticated
                    success = "error" not in str(tool_usage.get("result", "")).lower()
                    self.update_tool_preferences(tool_name, success)

            return final_result
        except Exception as e:
            self.task_status = TaskStatus.ERROR
            error_message = f"Error running agent: {str(e)}"
            stack_trace = "".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
            self.agent_logger.error(f"{error_message}\n{stack_trace}")

            if self.workflow_context and self.name in self.workflow_context:
                self._update_workflow_context(TaskStatus.ERROR, error_message)

            return {"status": str(TaskStatus.ERROR), "result": error_message}
        finally:
            # Ensure metrics get finalized properly
            if self.collect_metrics:
                # Always ensure we have an end time and the status is correctly set
                end_time = time.time()
                self._update_metrics(
                    "run",
                    {
                        "status": str(self.task_status),
                        "end_time": end_time,
                        "iterations": self.iterations,
                    },
                )

                # Update metrics based on actual tool history
                self._update_metrics_from_history()

            # Clean up resources
            try:
                # Cancel any active tasks we might have created
                if hasattr(self, "_active_tasks"):
                    for task in list(self._active_tasks):
                        if not task.done():
                            task.cancel()
                    self._active_tasks.clear()

                # Close MCP client properly using our safe method
                if hasattr(self, "mcp_client") and self.mcp_client:
                    await self._safe_close_mcp_client()
            except Exception as cleanup_error:
                self.agent_logger.error(f"Error during agent cleanup: {cleanup_error}")
                # Don't re-raise cleanup errors - just log them

    async def generate_summary(self) -> str:
        """Generate a summary of the agent's actions and results."""
        try:
            if not self.tool_history:
                return "No actions were taken."

            # Extract key information from tool history
            tools_used = []
            for tool_call in self.tool_history:
                tool_name = tool_call.get("name", "unknown")
                if tool_name not in tools_used:
                    tools_used.append(tool_name)

            # Create a default summary based on tool usage
            default_summary = f"Used {len(tools_used)} tools ({', '.join(tools_used)}) across {len(self.tool_history)} operations."

            summary_prompt = f"""
            <context>
            <task>{self.initial_task}</task>
            <tools_used>{', '.join(tools_used)}</tools_used>
            <actions>
            {json.dumps(self.tool_history, indent=2)}
            </actions>
            <final_result>
            {self.final_answer or "Task was executed through the tools shown above without a separate final answer."}
            </final_result>
            </context>
            """

            summary_system_prompt = """
            You are a summarization assistant. Create a concise summary of the actions taken and results achieved.
            Focus on:
            1. What was attempted
            2. What tools were used
            3. What was accomplished
            4. Any issues encountered
            
            Keep the summary short and user-friendly, avoiding technical details unless they're critical.
            """

            result = await self.model_provider.get_completion(
                system=summary_system_prompt, prompt=summary_prompt
            )

            # If the model completion fails, return the default summary
            if not result or "response" not in result:
                self.agent_logger.warning("Could not generate summary, using default")
                return default_summary

            return result.get("response", default_summary)
        except Exception as e:
            self.agent_logger.error(f"Error generating summary: {e}")
            # Return a simple summary based on available information
            if self.tool_history:
                tools_used = set(t.get("name", "unknown") for t in self.tool_history)
                return f"Performed {len(self.tool_history)} operations using {len(tools_used)} tools: {', '.join(tools_used)}."
            return "Error generating summary."

    async def rescope_goal(self, task: str, error_context: str) -> Dict[str, Any]:
        """
        Attempt to simplify or rescope the task when the original task is too ambitious.

        Args:
            task: The original task that couldn't be completed
            error_context: What went wrong during the attempt

        Returns:
            Dict with rescoped task and explanation
        """
        try:
            rescope_system_prompt = """
            You are an AI assistant that helps simplify and rescope tasks that are too ambitious or complex.
            
            When presented with a task that couldn't be completed and the error context,
            your job is to:
            
            1. Analyze what went wrong
            2. Suggest a simpler alternative task that might be achievable
            3. Explain your reasoning clearly
            
            Important: The rescoped task should still be useful to the user but more achievable given the constraints.
            """

            rescope_prompt = f"""
            <context>
            <original_task>{task}</original_task>
            <error_context>{error_context}</error_context>
            <available_tools>{[tool.name for tool in self.tools]}</available_tools>
            </context>
            """

            class RescopeFormat(BaseModel):
                rescoped_task: str = Field(
                    ...,
                    description="A simplified version of the original task that is more likely to succeed",
                )
                explanation: str = Field(
                    ..., description="Why this simplified task is more achievable"
                )
                expected_tools: List[str] = Field(
                    [], description="Tools expected to be needed for the rescoped task"
                )

            result = await self.model_provider.get_completion(
                system=rescope_system_prompt,
                prompt=rescope_prompt,
                format=(
                    RescopeFormat.model_json_schema()
                    if self.model_provider.name == "ollama"
                    else "json"
                ),
            )

            if not result or "response" not in result:
                return {
                    "rescoped_task": None,
                    "explanation": "Could not rescope the task",
                    "original_task": task,
                }

            # Parse the response
            try:
                rescope_result = (
                    json.loads(result["response"])
                    if isinstance(result["response"], str)
                    else result["response"]
                )

                # Add to reasoning log
                self.reasoning_log.append(
                    f"Task rescoped: {rescope_result.get('explanation', '')}"
                )

                return {
                    "rescoped_task": rescope_result.get("rescoped_task"),
                    "explanation": rescope_result.get("explanation"),
                    "expected_tools": rescope_result.get("expected_tools", []),
                    "original_task": task,
                }
            except (json.JSONDecodeError, AttributeError) as e:
                self.agent_logger.error(f"Error parsing rescope result: {e}")
                return {
                    "rescoped_task": None,
                    "explanation": f"Error during rescoping: {str(e)}",
                    "original_task": task,
                }

        except Exception as e:
            self.agent_logger.error(f"Goal rescoping error: {e}")
            return {
                "rescoped_task": None,
                "explanation": f"Error during rescoping: {str(e)}",
                "original_task": task,
            }

    async def _handle_tool_failure(
        self, tool_name: str, error: str, attempt: int = 1, max_attempts: int = 3
    ) -> Dict[str, Any]:
        """Handle a tool failure with retry logic and potential goal rescoping."""
        if attempt < max_attempts:
            self.agent_logger.warning(
                f"Tool '{tool_name}' failed (attempt {attempt}/{max_attempts}): {error}"
            )
            self.reasoning_log.append(
                f"Tool failure: {tool_name} - {error}. Retrying..."
            )

            # Wait briefly before retry (could be made more sophisticated with exponential backoff)
            await asyncio.sleep(1)

            return {
                "status": "retry",
                "message": f"Retrying tool {tool_name} (attempt {attempt+1}/{max_attempts})",
            }
        else:
            # We've reached max attempts, need to rescope the goal
            self.agent_logger.error(
                f"Tool '{tool_name}' failed after {max_attempts} attempts: {error}"
            )

            error_context = (
                f"The tool '{tool_name}' failed repeatedly with error: {error}"
            )
            rescope_result = await self.rescope_goal(self.initial_task, error_context)

            if rescope_result["rescoped_task"]:
                self.agent_logger.info(
                    f"Task rescoped: {rescope_result['rescoped_task']}"
                )

                # Update workflow context if available
                if self.workflow_context and self.name in self.workflow_context:
                    self.workflow_context[self.name]["rescoped"] = True
                    self.workflow_context[self.name][
                        "original_task"
                    ] = self.initial_task
                    self.workflow_context[self.name]["rescoped_task"] = rescope_result[
                        "rescoped_task"
                    ]

                return {
                    "status": "rescoped",
                    "message": f"Original task was too ambitious. Simplified to: {rescope_result['rescoped_task']}",
                    "explanation": rescope_result["explanation"],
                    "rescoped_task": rescope_result["rescoped_task"],
                }
            else:
                return {
                    "status": "failed",
                    "message": f"Tool '{tool_name}' failed and task could not be rescoped: {error}",
                }

    async def compare_goal_vs_result(self) -> Dict[str, Any]:
        """
        Rate how well the final output matched the user's intent.

        Returns:
            Dict with adherence score and evaluation details
        """
        # First check if we have tool history, it means actions were taken
        if not self.tool_history:
            # Only if there's truly been no activity
            return {
                "adherence_score": 0.0,
                "explanation": "No actions were taken or results produced.",
                "matches_intent": False,
                "strengths": [],
                "weaknesses": [
                    "No tools were used",
                    "No progress was made on the task",
                ],
            }

        try:
            evaluation_system_prompt = """
            You are an objective evaluator that assesses how well a task result matches the original goal.
            
            Rate the adherence on a scale from 0.0 to 1.0, where:
            - 0.0: Completely failed to address the goal or made things worse
            - 0.3: Partially addressed the goal but with significant gaps
            - 0.5: Met some key aspects of the goal
            - 0.7: Successfully addressed most parts of the goal
            - 1.0: Perfectly achieved the goal
            
            Provide specific reasons for your rating, noting both strengths and weaknesses.
            """

            # Extract tool names for more clarity
            tool_names_used = [t.get("name", "unknown") for t in self.tool_history]

            evaluation_prompt = f"""
            <context>
            <original_goal>{self.initial_task}</original_goal>
            <tools_used>{', '.join(sorted(set(tool_names_used)))}</tools_used>
            <actions_taken>
            {json.dumps(self.tool_history, indent=2)}
            </actions_taken>
            <final_result>
            {self.final_answer or "The task was executed through the tools shown above."}
            </final_result>
            <reasoning_log>
            {json.dumps(self.reasoning_log, indent=2) if self.reasoning_log else "No explicit reasoning recorded."}
            </reasoning_log>
            </context>
            """

            # If we don't get a response, provide a reasonable default based on tool usage
            if len(self.tool_history) > 0:
                default_response = {
                    "adherence_score": min(
                        0.7, len(self.tool_history) * 0.1
                    ),  # Base score on tool activity
                    "explanation": f"Task was partially addressed with {len(self.tool_history)} tool calls.",
                    "matches_intent": True,
                    "strengths": [f"Used {len(set(tool_names_used))} different tools"],
                    "weaknesses": ["No final explicit answer provided"],
                }
            else:
                default_response = {
                    "adherence_score": 0.1,
                    "explanation": "Minimal progress was made on the task.",
                    "matches_intent": False,
                    "strengths": [],
                    "weaknesses": ["Insufficient actions taken"],
                }

            class EvaluationFormat(BaseModel):
                adherence_score: float = Field(
                    ...,
                    ge=0.0,
                    le=1.0,
                    description="Score from 0.0 to 1.0 indicating how well the result matches the goal",
                )
                strengths: List[str] = Field(
                    ...,
                    description="List of ways the result successfully addressed the goal",
                )
                weaknesses: List[str] = Field(
                    ..., description="List of ways the result fell short of the goal"
                )
                explanation: str = Field(
                    ..., description="Overall explanation of the rating"
                )
                matches_intent: bool = Field(
                    ...,
                    description="Whether the result fundamentally addresses the user's core intent",
                )

            try:
                result = await self.model_provider.get_completion(
                    system=evaluation_system_prompt,
                    prompt=evaluation_prompt,
                    format=(
                        EvaluationFormat.model_json_schema()
                        if self.model_provider.name == "ollama"
                        else "json"
                    ),
                )

                if not result or "response" not in result:
                    self.agent_logger.warning(
                        "Could not evaluate goal adherence, using default"
                    )
                    return default_response

                # Parse the response
                evaluation = (
                    json.loads(result["response"])
                    if isinstance(result["response"], str)
                    else result["response"]
                )

                # Add to workflow context if available
                if self.workflow_context and self.name in self.workflow_context:
                    self.workflow_context[self.name]["adherence_score"] = (
                        evaluation.get("adherence_score", 0.5)
                    )
                    self.workflow_context[self.name]["matches_intent"] = evaluation.get(
                        "matches_intent", False
                    )

                # Add to reasoning log
                evaluation_summary = (
                    f"Goal adherence evaluation: {evaluation.get('explanation', '')}"
                )
                if evaluation_summary not in self.reasoning_log:
                    self.reasoning_log.append(evaluation_summary)

                return evaluation
            except Exception as e:
                self.agent_logger.error(f"Error in evaluation model call: {e}")
                return default_response

        except Exception as e:
            self.agent_logger.error(f"Goal adherence evaluation error: {e}")
            return default_response

    def _init_memory(self):
        """Initialize the agent's persistent memory"""
        try:
            memory_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory"
            )
            os.makedirs(memory_dir, exist_ok=True)

            self.memory_file = os.path.join(
                memory_dir, f"{self.name.replace(' ', '_')}_memory.json"
            )
            self.agent_memory = None

            # Try to load existing memory
            if os.path.exists(self.memory_file):
                try:
                    with open(self.memory_file, "r") as f:
                        memory_data = json.load(f)
                        self.agent_memory = AgentMemory(**memory_data)
                        self.agent_logger.debug(
                            f"Loaded memory from {self.memory_file}"
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    self.agent_logger.warning(f"Error loading memory file: {e}")
                    # Create new memory if file is corrupted
                    self.agent_memory = AgentMemory(agent_name=self.name)
            else:
                # Create new memory if file doesn't exist
                self.agent_memory = AgentMemory(agent_name=self.name)

        except Exception as e:
            self.agent_logger.error(f"Error initializing memory: {e}")
            # Create in-memory version that won't be persisted
            self.agent_memory = AgentMemory(agent_name=self.name)
            self.memory_file = None

    def save_memory(self):
        """Save the agent's memory to disk"""
        if (
            not hasattr(self, "memory_file")
            or not self.memory_file
            or not hasattr(self, "agent_memory")
            or self.agent_memory is None
        ):
            return

        try:
            # Update last updated timestamp
            self.agent_memory.last_updated = datetime.now()

            # Convert to dictionary and then to JSON
            memory_dict = self.agent_memory.dict()
            memory_json = json.dumps(memory_dict, indent=2, default=str)

            with open(self.memory_file, "w") as f:
                f.write(memory_json)
            self.agent_logger.debug(f"Saved memory to {self.memory_file}")
        except Exception as e:
            self.agent_logger.error(f"Error saving memory: {e}")

    def update_session_history(self, task: str, result: Dict[str, Any]):
        """Update the agent's session history"""
        if not hasattr(self, "agent_memory") or not self.agent_memory:
            return

        try:
            # Parse status from result or use current task_status
            status = result.get("status", "unknown")
            if hasattr(self, "task_status"):
                status = str(self.task_status)

            # Create a session summary
            session_entry = {
                "timestamp": datetime.now().isoformat(),
                "task": task,
                "status": status,
                "tools_used": (
                    [t.get("name", "unknown") for t in self.tool_history]
                    if hasattr(self, "tool_history")
                    else []
                ),
                "success": status
                in [str(TaskStatus.COMPLETE), str(TaskStatus.RESCOPED_COMPLETE)],
                "iterations": self.iterations,
                "adherence_score": (
                    result.get("evaluation", {}).get("adherence_score")
                    if "evaluation" in result
                    else None
                ),
            }

            # Add to memory
            self.agent_memory.session_history.append(session_entry)

            # Keep only the last 20 sessions
            if len(self.agent_memory.session_history) > 20:
                self.agent_memory.session_history = self.agent_memory.session_history[
                    -20:
                ]

            # Save to disk
            self.save_memory()
        except Exception as e:
            self.agent_logger.error(f"Error updating session history: {e}")

    def update_tool_preferences(
        self, tool_name: str, success: bool, feedback: Optional[str] = None
    ):
        """Update tool preferences based on usage success"""
        if not hasattr(self, "agent_memory") or not self.agent_memory:
            return

        try:
            if tool_name not in self.agent_memory.tool_preferences:
                self.agent_memory.tool_preferences[tool_name] = {
                    "success_count": 0,
                    "failure_count": 0,
                    "feedback": [],
                }

            # Update counts
            if success:
                self.agent_memory.tool_preferences[tool_name]["success_count"] += 1
            else:
                self.agent_memory.tool_preferences[tool_name]["failure_count"] += 1

            # Add feedback if provided
            if feedback:
                self.agent_memory.tool_preferences[tool_name]["feedback"].append(
                    {"timestamp": datetime.now().isoformat(), "message": feedback}
                )

                # Keep only the last 5 feedback items
                if len(self.agent_memory.tool_preferences[tool_name]["feedback"]) > 5:
                    self.agent_memory.tool_preferences[tool_name]["feedback"] = (
                        self.agent_memory.tool_preferences[tool_name]["feedback"][-5:]
                    )

            # Save to disk
            self.save_memory()
        except Exception as e:
            self.agent_logger.error(f"Error updating tool preferences: {e}")

    async def execute_tools_in_parallel(self, tool_calls: List[Dict]) -> List[Dict]:
        """
        Execute multiple tools in parallel for better performance.
        Only used for tools that are known to be independent.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            List of results in the same order as the calls
        """
        if not tool_calls:
            return []

        self.agent_logger.info(f"Executing {len(tool_calls)} tools in parallel")

        async def execute_single_tool(tool_call):
            try:
                result = await self._use_tool(tool_call)
                return {
                    "tool_name": tool_call["function"]["name"],
                    "result": result,
                    "success": True,
                }
            except Exception as e:
                self.agent_logger.error(f"Parallel tool execution error: {e}")
                return {
                    "tool_name": tool_call["function"]["name"],
                    "result": f"Error: {str(e)}",
                    "success": False,
                }

        # Execute all tools in parallel
        tasks = [execute_single_tool(call) for call in tool_calls]
        results = await asyncio.gather(*tasks)

        # Record statistics
        successful = sum(1 for r in results if r["success"])
        self.agent_logger.info(
            f"Parallel execution complete: {successful}/{len(results)} tools succeeded"
        )

        return results

    async def _check_network_availability(self) -> bool:
        """Check if network is available for online operations"""
        if self.offline_mode:
            return False

        # Only check occasionally to avoid unnecessary overhead
        if time.time() - self.last_network_check < 60:  # Check once per minute at most
            return self.network_available

        self.last_network_check = time.time()

        try:
            # Try to connect to a reliable host with a short timeout
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("1.1.1.1", 53), timeout=1.0
            )
            writer.close()
            await writer.wait_closed()

            # Reset error count and update availability
            self.network_errors = 0
            self.network_available = True
            return True
        except Exception as e:
            self.network_errors += 1
            self.network_available = False
            self.agent_logger.warning(f"Network appears to be unavailable: {e}")
            return False

    async def _execute_with_retry(
        self, operation_name: str, coro, *args, **kwargs
    ) -> Any:
        """Execute an async operation with retry logic for network resilience"""
        max_retries = self.retry_config.get("max_retries", 3)
        base_delay = self.retry_config.get("base_delay", 1.0)
        max_delay = self.retry_config.get("max_delay", 10.0)

        for attempt in range(max_retries + 1):
            try:
                return await coro(*args, **kwargs)
            except (asyncio.TimeoutError, ConnectionError, OSError) as e:
                if attempt >= max_retries:
                    self.agent_logger.error(
                        f"{operation_name} failed after {max_retries} attempts: {e}"
                    )
                    raise

                # Exponential backoff with jitter
                delay = min(
                    base_delay * (2**attempt) * (0.5 + random.random()), max_delay
                )
                self.agent_logger.warning(
                    f"{operation_name} attempt {attempt+1} failed: {e}. Retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)

                # Check network before retrying
                if not await self._check_network_availability():
                    self.agent_logger.warning(
                        "Network unavailable, using cached results if possible"
                    )
                    break

    def _update_metrics(self, metric_type, data):
        """Update the metrics based on the type and data provided.

        Args:
            metric_type (str): The type of metric to update (tool, model, etc.)
            data (dict): The data to update the metrics with
        """
        if not self.collect_metrics:
            return

        if metric_type == "tool":
            # Update tool call metrics
            self.metrics["tool_calls"] += 1

            if data.get("error"):
                self.metrics["tool_errors"] += 1

            # Track tool-specific metrics
            tool_name = data.get("name", "unknown")
            if tool_name not in self.metrics["tools"]:
                self.metrics["tools"][tool_name] = {
                    "calls": 0,
                    "errors": 0,
                    "total_time": 0,
                }

            self.metrics["tools"][tool_name]["calls"] += 1
            if data.get("error"):
                self.metrics["tools"][tool_name]["errors"] += 1

            if "time" in data:
                self.metrics["tools"][tool_name]["total_time"] += data["time"]

                # Ensure latency and tool_time keys exist
                if "latency" not in self.metrics:
                    self.metrics["latency"] = {}
                if "tool_time" not in self.metrics["latency"]:
                    self.metrics["latency"]["tool_time"] = 0

                self.metrics["latency"]["tool_time"] += data["time"]

        elif metric_type == "model":
            # Update model call metrics
            self.metrics["model_calls"] += 1

            # Update token counts
            prompt_tokens = data.get("prompt_tokens", 0)
            completion_tokens = data.get("completion_tokens", 0)

            self.metrics["tokens"]["prompt"] += prompt_tokens
            self.metrics["tokens"]["completion"] += completion_tokens
            self.metrics["tokens"]["total"] = (
                self.metrics["tokens"]["prompt"] + self.metrics["tokens"]["completion"]
            )

            # Update model latency
            if "time" in data:
                # Ensure latency and model_time keys exist
                if "latency" not in self.metrics:
                    self.metrics["latency"] = {}
                if "model_time" not in self.metrics["latency"]:
                    self.metrics["latency"]["model_time"] = 0

                self.metrics["latency"]["model_time"] += data["time"]

        elif metric_type == "cache":
            # Update cache metrics
            if data.get("hit", False):
                self.metrics["cache"]["hits"] += 1
            else:
                self.metrics["cache"]["misses"] += 1

        elif metric_type == "run":
            # Update run metrics
            if "status" in data:
                self.metrics["status"] = data["status"]

            if "iterations" in data:
                self.metrics["iterations"] = data["iterations"]

            if "start_time" in data:
                self.metrics["start_time"] = data["start_time"]

            if "end_time" in data:
                self.metrics["end_time"] = data["end_time"]
                self.metrics["total_time"] = (
                    self.metrics["end_time"] - self.metrics["start_time"]
                )

    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics for the agent"""
        if not self.collect_metrics:
            return {}

        # Update cache ratio
        total_cache_calls = self.cache_hits + self.cache_misses
        if total_cache_calls > 0:
            self.metrics["cache"]["ratio"] = self.cache_hits / total_cache_calls

        # Make sure we have the latest iteration count
        self.metrics["iterations"] = self.iterations

        # If metrics don't have an end time yet, calculate current duration
        if not self.metrics["end_time"]:
            self.metrics["total_time"] = time.time() - self.metrics["start_time"]

        return self.metrics

    async def _think_chain(
        self, tool_use: bool = True, remember_messages: bool = True, **kwargs
    ):
        """Override base _think_chain to track metrics"""
        self.agent_logger.info("🧠 MODEL CALL: Generating response...")

        start_time = time.time()
        result = await super()._think_chain(
            tool_use=tool_use, remember_messages=remember_messages, **kwargs
        )
        execution_time = time.time() - start_time

        # Log completion
        if result:
            message = result.get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            if content:
                self.agent_logger.info(f"💬 MODEL RESPONSE: ({execution_time:.2f}s)")
                self.agent_logger.debug(
                    f"📝 CONTENT: {content[:1000]}{' [truncated]' if len(content) > 1000 else ''}"
                )

            if tool_calls:
                tool_names = [
                    t.get("function", {}).get("name", "unknown") for t in tool_calls
                ]
                self.agent_logger.info(
                    f"🔧 MODEL REQUESTED TOOLS: {', '.join(tool_names)}"
                )

        # Track metrics if enabled
        if self.collect_metrics and result:
            # Extract token counts if available
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            self._update_metrics(
                "model",
                {
                    "time": execution_time,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            )

        return result

    async def _think(self, **kwargs):
        """Override base _think to track metrics"""
        self.agent_logger.info("🧠 MODEL COMPLETION: Generating completion...")

        start_time = time.time()
        result = await super()._think(**kwargs)
        execution_time = time.time() - start_time

        # Log completion
        if result:
            response = result.get("response", "")
            if response:
                self.agent_logger.info(f"💡 COMPLETION: ({execution_time:.2f}s)")
                self.agent_logger.debug(
                    f"📝 CONTENT: {response[:1000]}{' [truncated]' if len(response) > 1000 else ''}"
                )

        # Track metrics if enabled
        if self.collect_metrics and result:
            # Extract token counts if available
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            self._update_metrics(
                "model",
                {
                    "time": execution_time,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            )

        return result

    async def _use_tool(self, tool_call) -> Union[str, List[str], None]:
        """Execute a tool and track metrics."""
        tool_name = tool_call["function"]["name"]
        tool_args = (
            json.loads(tool_call["function"]["arguments"])
            if isinstance(tool_call["function"]["arguments"], str)
            else tool_call["function"]["arguments"]
        )

        if not tool_name:
            error_msg = "Tool name not provided"
            self._update_metrics(
                "tool", {"name": "unknown", "error": error_msg, "time": 0}
            )
            return error_msg

        try:
            tool = self.get_tool(tool_name)
            if not tool:
                error_msg = f"Tool '{tool_name}' not found"
                self._update_metrics(
                    "tool", {"name": tool_name, "error": error_msg, "time": 0}
                )
                return error_msg

            # Add detailed logging
            self.agent_logger.info(f"📦 TOOL CALL: {tool_name}")
            self.agent_logger.debug(f"🔍 TOOL ARGS: {json.dumps(tool_args, indent=2)}")

            # Track tool execution time
            start_time = time.time()
            result = await tool.use(tool_args)
            execution_time = time.time() - start_time

            # Format result for logging
            formatted_result = (
                result.to_list() if hasattr(result, "to_list") else result
            )

            # Log the result
            self.agent_logger.info(
                f"✅ TOOL RESULT: {tool_name} completed in {execution_time:.2f}s"
            )

            # Log a truncated version of the result if it's very long
            result_str = str(formatted_result)
            if len(result_str) > 1000:
                self.agent_logger.debug(
                    f"📄 RESULT PREVIEW: {result_str[:1000]}... (truncated, total length: {len(result_str)})"
                )
            else:
                self.agent_logger.debug(f"📄 RESULT: {result_str}")

            # Update metrics
            self._update_metrics("tool", {"name": tool_name, "time": execution_time})

            # Update tool history with this tool call and result
            if not hasattr(self, "tool_history"):
                self.tool_history = []

            self.tool_history.append(
                {
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": formatted_result,
                    "execution_time": execution_time,
                    "timestamp": time.time(),
                }
            )

            return formatted_result
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            # Log the error
            self.agent_logger.error(f"❌ TOOL ERROR: {error_msg}")
            self.agent_logger.debug(f"🐞 EXCEPTION: {traceback.format_exc()}")

            self._update_metrics(
                "tool",
                {
                    "name": tool_name,
                    "error": error_msg,
                    "time": 0,  # No execution time for errors
                },
            )

            # Record failed tool calls in history too
            if not hasattr(self, "tool_history"):
                self.tool_history = []

            self.tool_history.append(
                {
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": error_msg,
                    "error": True,
                    "timestamp": time.time(),
                }
            )

            return error_msg

    def get_tool(self, tool_name: str) -> Optional[ToolProtocol]:
        """Get a tool by name from the agent's toolset.

        Args:
            tool_name: The name of the tool to retrieve

        Returns:
            The tool if found, None otherwise
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def _update_metrics_from_history(self):
        """Update metrics based on the actual tool_history before returning results"""
        if not self.collect_metrics:
            return

        # Make sure tool_calls matches actual history
        self.metrics["tool_calls"] = len(self.tool_history)

        # Reset tool-specific metrics to avoid double counting
        self.metrics["tools"] = {}
        self.metrics["tool_errors"] = 0

        # Update tool-specific metrics
        for tool_call in self.tool_history:
            tool_name = tool_call.get("name", "unknown")

            # Ensure the tool entry exists
            if tool_name not in self.metrics["tools"]:
                self.metrics["tools"][tool_name] = {
                    "calls": 0,
                    "errors": 0,
                    "total_time": 0,
                }

            # Increment call count
            self.metrics["tools"][tool_name]["calls"] += 1

            # Track errors
            if "error" in str(tool_call.get("result", "")).lower():
                self.metrics["tools"][tool_name]["errors"] += 1
                self.metrics["tool_errors"] += 1

            # Track execution time if available
            if "execution_time" in tool_call:
                time_value = tool_call.get("execution_time", 0)
                self.metrics["tools"][tool_name]["total_time"] += time_value
