import json
import traceback
from typing import List, Any, Optional, Dict

from pydantic import BaseModel, Field
from agent_mcp.client import MCPClient
from prompts.agent_prompts import (
    PERCENTAGE_COMPLETE_TASK_REFLECTION_SYSTEM_PROMPT,
    REACT_AGENT_SYSTEM_PROMPT,
    AGENT_ACTION_PLAN_PROMPT,
)
from agents.base import Agent


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

                if "final answer:" in result["message"]["content"].lower():
                    self.final_answer = result["message"]["content"]

            return result
        except (IndexError, KeyError, AttributeError) as e:
            self.agent_logger.warning(f"Error accessing reflection data: {str(e)}")
            # Reset messages to ensure we have a clean state
            self.messages = [system_message, {"role": "user", "content": task}]
            return await self._think_chain(tool_use=tool_use)

    async def _reflect(self, task_description, result):
        """Reflect on task progress and determine next steps."""
        try:

            class ReflectionFormat(BaseModel):
                completion_score: float = Field(..., ge=0.0, le=1.0)
                next_step: str
                required_tools: List[str]
                completed_tools: List[str]
                reason: str

            # Get previous reflection safely
            previous_reflection = (
                self.reflections[-1].get("reason", "None")
                if self.reflections
                else "None"
            )

            reflection_prompt = f"""
            Goal: Evaluate completion status and required tools for task: {task_description}
            Context:
            - Current Result: {result["message"]["content"]}
            - Role Of Agent: {self.role or "AI Assistant"}
            - Steps Taken: {self.task_progress}
            - Available Tools: {[f"Name: {tool['function']['name']}  Parameters: {tool['function']['parameters']}" for tool in self.tool_signatures]}
            - Previous Workflow Steps: {json.dumps(self.workflow_context if self.workflow_context is not None else {})}
            - Previous Reflection: {previous_reflection}

            Evaluate the task completion status and provide guidance on next steps.
            Output must include:
            1. Precise completion percentage (0.00-1.00)
            2. List of all tools required to complete the task
            3. List of tools that have been successfully used
            4. Specific next tool action with parameters
            5. Detailed reasoning explaining:
            - Why the completion score was given
            - Which required tools are still needed
            - What the next step should be

            Remember:
            - Task is not complete until all required tools have been used *successfully*
            - Tools should be added to the completed tools list only if they were used properly to accomplish what it was intended to do
            - Depending on the role, not all tools may be required for example if the role is to plan then execution steps may not be required
            - Completion score should reflect both progress and tool usage
            - Next step must use an available tool
            - Do not add a next step that is not an available tool
            - Do not keep repeating the same next step
            - Don't mark task complete until all required actions are verified
            """

            self.agent_logger.info("Reflecting on task progress...")
            try:
                reflection = await self.model_provider.get_completion(
                    format=(
                        ReflectionFormat.model_json_schema()
                        if self.model_provider.name == "ollama"
                        else "json"
                    ),
                    system=PERCENTAGE_COMPLETE_TASK_REFLECTION_SYSTEM_PROMPT,
                    prompt=reflection_prompt,
                )
            except Exception as model_error:
                self.agent_logger.error(
                    f"Model provider error during reflection: {str(model_error)}"
                )
                return None

            if not reflection or not reflection.get("response"):
                self.agent_logger.warning("No reflection response received")
                return None

            try:
                reflection_data = json.loads(reflection["response"])
            except json.JSONDecodeError as decode_error:
                self.agent_logger.error(
                    f"Failed to parse reflection response: {str(decode_error)}"
                )
                return None

            # Only update workflow context if it exists
            if self.workflow_context is not None and self.name in self.workflow_context:
                self.workflow_context[self.name].update(
                    {
                        "status": "in_progress",
                        "current_progress": result["message"]["content"],
                        "completion_status": reflection_data,
                        "required_tools": reflection_data.get("required_tools", []),
                        "completed_tools": reflection_data.get("completed_tools", []),
                    }
                )

            # Always update reflections list
            self.reflections.append(reflection_data)

            # Calculate effective completion score based on tool usage
            required_tools = reflection_data.get("required_tools", [])
            completed_tools = reflection_data.get("completed_tools", [])
            if required_tools and completed_tools:
                tools_completion_ratio = len(completed_tools) / len(required_tools)
                reflection_data["completion_score"] = min(
                    reflection_data["completion_score"], tools_completion_ratio
                )

            return reflection

        except Exception as e:
            self.agent_logger.error(f"Reflection process error: {str(e)}")
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
        # Check basic conditions first
        if not self.reflect or not self.check_dependencies():
            return False

        if self.max_iterations and self.iterations >= self.max_iterations:
            self.agent_logger.info("Maximum iterations reached")
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
                    self.agent_logger.info(
                        f"Task complete with score {effective_score}"
                    )
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
        if not self.reflect:
            result = await self._run_task(task)
            return result["message"]["content"] if result else None

        if not self.check_dependencies():
            self.agent_logger.warning("Dependencies not met, cannot proceed")
            return None

        # Reset iteration state
        self.iterations = 0
        self.previous_content = None
        last_error = None
        last_reflection = None

        # Ensure we have a valid reflections list
        if not hasattr(self, "reflections") or self.reflections is None:
            self.reflections = []
            # Update workflow context if it exists
            if self.workflow_context is not None and self.name in self.workflow_context:
                self.workflow_context[self.name]["reflections"] = self.reflections

        try:
            while self.should_continue():
                self.iterations += 1
                self.agent_logger.info(f"Running Iteration: {self.iterations}")

                try:
                    # Execute task
                    result = await self._run_task(task)
                    if not result or not result.get("message", {}).get("content"):
                        self.agent_logger.warning(
                            "Task execution failed or returned no content"
                        )
                        break

                    current_content = result["message"]["content"]

                    # Check for final answer
                    if self.final_answer:
                        self._update_workflow_context("complete", self.final_answer)
                        return self.final_answer

                    # Reflect on progress
                    reflection = await self._reflect(
                        task_description=task, result=result
                    )

                    # Store last valid reflection data before potential parsing
                    if reflection and reflection.get("response"):
                        try:
                            reflection_data = json.loads(reflection["response"])
                            last_reflection = reflection_data
                            self.reflections.append(reflection_data)
                            completion_score = reflection_data.get(
                                "completion_score", 0.0
                            )

                            self.agent_logger.info(
                                f"Completion Percentage: {completion_score * 100}%"
                            )
                            self.agent_logger.debug(
                                f"Reflection Feedback: {reflection_data.get('reason')}"
                            )
                            self.agent_logger.debug(
                                f"Required Tools: {reflection_data.get('required_tools')}"
                            )
                            self.agent_logger.debug(
                                f"Completed Tools: {reflection_data.get('completed_tools')}"
                            )
                            self.agent_logger.debug(
                                f"Next Step: {reflection_data.get('next_step')}"
                            )

                            # Sync workflow context with current reflection state
                            if (
                                self.workflow_context is not None
                                and self.name in self.workflow_context
                            ):
                                self.workflow_context[self.name][
                                    "reflections"
                                ] = self.reflections

                            # Update workflow context with current status
                            status = (
                                "complete"
                                if completion_score >= self.min_completion_score
                                else "in_progress"
                            )
                            self._update_workflow_context(
                                status,
                                current_content,
                                completion_score=completion_score,
                                reflection_data=reflection_data,
                            )

                            # Return immediately if task is complete
                            if status == "complete":
                                return current_content

                            # Store the current content for the next iteration
                            self.previous_content = current_content

                        except json.JSONDecodeError as e:
                            self.agent_logger.error(
                                f"Reflection data parsing error: {str(e)}"
                            )
                            if self.iterations > 1:  # Only break if not first iteration
                                last_error = e
                                break
                    else:
                        self.agent_logger.warning(
                            "Reflection failed or returned no data"
                        )
                        if self.iterations > 1:  # Only break if not first iteration
                            break

                    self.previous_content = current_content

                except Exception as iteration_error:
                    self.agent_logger.error(
                        f"Iteration error: {str(iteration_error)}", exc_info=True
                    )
                    last_error = iteration_error
                    if self.iterations > 1:  # Only break if not first iteration
                        break
                    continue

            # Handle final state
            if last_error:
                self._update_workflow_context(
                    "error",
                    None,
                    error=str(last_error),
                    last_reflection=last_reflection,
                )
                return None

            if self.previous_content:
                # Update final state with last known good state
                self._update_workflow_context(
                    (
                        "complete"
                        if last_reflection
                        and last_reflection.get("completion_score", 0)
                        >= self.min_completion_score
                        else "stopped"
                    ),
                    self.previous_content,
                    completion_score=(
                        last_reflection.get("completion_score", 0)
                        if last_reflection
                        else None
                    ),
                    last_reflection=last_reflection,
                )
                return self.previous_content

            return None

        except Exception as e:
            self.agent_logger.error(f"Task execution error: {str(e)}", exc_info=True)
            self._update_workflow_context(
                "error", None, error=str(e), last_reflection=last_reflection
            )
            return None

    def _update_workflow_context(self, status: str, result: Optional[str], **kwargs):
        """Update workflow context with current status and additional data"""
        if not hasattr(self, "workflow_context"):
            return

        # Skip workflow updates if workflow context isn't being used
        if self.workflow_context is None:
            return

        # Ensure agent's context exists
        if self.name not in self.workflow_context:
            self.workflow_context[self.name] = {
                "status": "initialized",
                "current_progress": "",
                "iterations": 0,
                "dependencies_met": True,
            }

        context_update = {
            "status": status,
            "iterations": self.iterations,
            "dependencies_met": self.check_dependencies(),
        }

        if result:
            context_update[
                "final_result" if status == "complete" else "current_progress"
            ] = result

        if status == "error":
            context_update["error"] = kwargs.get("error")
        else:
            if "completion_score" in kwargs:
                context_update["completion_score"] = kwargs["completion_score"]
            if "plan" in kwargs:
                context_update["last_action_plan"] = kwargs["plan"]
            if "reflection_data" in kwargs:
                context_update["last_reflection"] = kwargs["reflection_data"]

        self.workflow_context[self.name].update(context_update)
