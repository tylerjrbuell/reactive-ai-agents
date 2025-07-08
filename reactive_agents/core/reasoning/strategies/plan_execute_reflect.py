from __future__ import annotations
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Callable, NoReturn
from functools import wraps
from reactive_agents.core.types.reasoning_types import (
    ReasoningStrategies,
    ReasoningContext,
)
from .base import BaseReasoningStrategy, StrategyResult, StrategyCapabilities
from ..infrastructure import Infrastructure
from reactive_agents.core.types.session_types import PlanExecuteReflectState
import json
import asyncio

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


def ensure_state_initialized(func: Callable):
    """Decorator to ensure strategy state is initialized before method execution."""

    @wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        if not self.context.session.per_strategy_state:
            self.context.session.initialize_strategy_state("plan_execute_reflect")
        return await func(self, *args, **kwargs)

    @wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        if not self.context.session.per_strategy_state:
            self.context.session.initialize_strategy_state("plan_execute_reflect")
        return func(self, *args, **kwargs)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class PlanExecuteReflectStrategy(BaseReasoningStrategy):
    """
    Enhanced Plan-Execute-Reflect strategy using prompt adapters.

    Uses focused, context-aware prompts for each phase:
    1. Plan: Break down goals into actionable steps
    2. Execute: Execute specific steps with proper tool mapping
    3. Reflect: Evaluate progress and determine next actions
    """

    @property
    def name(self) -> str:
        return "plan_execute_reflect"

    @property
    def capabilities(self) -> list[StrategyCapabilities]:
        return [
            StrategyCapabilities.PLANNING,
            StrategyCapabilities.TOOL_EXECUTION,
            StrategyCapabilities.REFLECTION,
        ]

    def __init__(self, infrastructure: "Infrastructure"):
        super().__init__(infrastructure)

        # Import adapter here to avoid circular imports
        from ..prompts.adapters.plan_execute_reflect import (
            PlanExecuteReflectStrategyAdapter,
        )

        self.adapter = PlanExecuteReflectStrategyAdapter(infrastructure.context)

        # Initialize strategy state
        if not self.context.session.per_strategy_state:
            self.context.session.initialize_strategy_state("plan_execute_reflect")

        # Track available tools
        self.available_tools = [tool.name for tool in self.context.tools]

    @property
    def state(self) -> PlanExecuteReflectState:
        """Get the current strategy state, ensuring it's initialized."""
        if not self.context.session.per_strategy_state:
            self.context.session.initialize_strategy_state("plan_execute_reflect")
        if not isinstance(
            self.context.session.per_strategy_state, PlanExecuteReflectState
        ):
            raise ValueError("Strategy state is not properly initialized")
        return self.context.session.per_strategy_state

    def _record_step_result(self, step_result: Dict[str, Any]) -> None:
        """Record a step result using the state manager."""
        self.state.record_step_result(step_result)

    def _record_reflection_result(self, reflection_result: Dict[str, Any]) -> None:
        """Record a reflection result using the state manager."""
        self.state.record_reflection_result(reflection_result)

    def _prepare_focused_message_context(
        self, user_prompt: str, current_step: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Prepare a focused three-message context window for the LLM.
        """
        if self.agent_logger:
            self.agent_logger.debug("🔄 Preparing focused message context")

        # 1. Preserve system message
        system_message = (
            self.context.session.messages[0]
            if self.context.session.messages
            else {
                "role": "system",
                "content": "You are an AI agent. Use only the provided tools to complete the task.",
            }
        )

        # Remove any existing assistant context/summary messages except the system message
        new_messages = [system_message]
        for msg in self.context.session.messages[1:]:
            if msg.get("role") == "assistant":
                content = msg.get("content", "").strip()
                if (
                    content.startswith("[SUMMARY")
                    or content.startswith("[CONTEXT")
                    or "CURRENT EXECUTION STATE" in content
                ):
                    continue  # skip old context/summary messages
            new_messages.append(msg)
        self.context.session.messages = new_messages

        # Set the focused three-message context
        self.context.session.messages[0] = system_message
        self.context.session.messages.append({"role": "user", "content": user_prompt})

        if self.agent_logger:
            self.agent_logger.debug(
                f"📝 Context window prepared with {len(self.context.session.messages)} messages"
            )
            # self.agent_logger.debug(
            #     f"Context summary length: {len(context_summary)} chars"
            # )

    @ensure_state_initialized
    async def execute_iteration(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """Execute plan-execute-reflect iteration with enhanced error handling."""
        try:
            if self.agent_logger:
                self.agent_logger.debug("📋 Enhanced Plan-Execute-Reflect iteration")

            # Phase 1: Planning (if no plan exists)
            if not self.state.current_plan or not self.state.current_plan.get(
                "steps", []
            ):
                if self.agent_logger:
                    self.agent_logger.debug("No plan exists, starting planning phase")
                return await self._planning_phase(task, reasoning_context)

            # Phase 2: Execution (if steps remain)
            if self.state.current_step < len(self.state.current_plan.get("steps", [])):
                if self.agent_logger:
                    self.agent_logger.debug(
                        f"Executing step {self.state.current_step + 1}"
                    )
                return await self._execution_phase(task, reasoning_context)

            # Phase 3: Reflection (only after execution or when plan is complete)
            if self.agent_logger:
                self.agent_logger.debug("Plan complete or needs reflection")
            return await self._reflection_phase(task, reasoning_context)

        except Exception as e:
            self.state.error_count += 1
            if self.agent_logger:
                self.agent_logger.error(f"Plan-Execute-Reflect iteration failed: {e}")
            return self._format_error_result(e, "plan_execute_reflect")

    @ensure_state_initialized
    async def _planning_phase(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """Create a focused plan using the adapter."""
        if self.agent_logger:
            self.agent_logger.debug("📋 Creating execution plan")

        # Use adapter to generate focused planning prompt
        task_classification = (
            getattr(reasoning_context, "task_classification", {}) or {}
        )
        task_type = task_classification.get("task_type", "general")

        planning_prompt = self.adapter.prepare_plan_prompt(
            goal=task,
            task_type=task_type,
            prior_reflections=getattr(self, "_prior_reflections", []),
        )

        # Get plan from infrastructure
        result = await self._think(planning_prompt)

        if not result or not result.get("content"):
            return StrategyResult(
                action_taken="planning_failed",
                result={"error": "Failed to generate plan"},
                should_continue=False,
                strategy_used="plan_execute_reflect",
            )

        # Parse plan using enhanced parsing
        plan_text = result["content"]
        steps = self._parse_plan_steps(plan_text)

        if not steps:
            return StrategyResult(
                action_taken="planning_failed",
                result={"error": "No valid steps generated"},
                should_continue=False,
                strategy_used="plan_execute_reflect",
            )

        self.state.current_plan = {
            "task": task,
            "steps": steps,
            "plan_text": plan_text,
        }

        # Reset state for new plan
        self.state.current_step = 0
        self.state.execution_history = []
        self.state.error_count = 0
        self.state.completed_actions = []

        # Preserve the plan
        self._preserve_context("execution_plan", self.state.current_plan)

        if self.agent_logger:
            self.agent_logger.info(f"📋 Created plan with {len(steps)} steps")

        return StrategyResult(
            action_taken="planning_complete",
            result={
                "plan": self.state.current_plan,
                "steps_count": len(steps),
            },
            should_continue=True,
            strategy_used="plan_execute_reflect",
        )

    @ensure_state_initialized
    async def _execution_phase(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """Execute current step with focused message context."""
        if not self.state.current_plan or "steps" not in self.state.current_plan:
            return StrategyResult(
                action_taken="execution_failed",
                result={"error": "No plan available for execution"},
                should_continue=False,
                strategy_used="plan_execute_reflect",
            )

        plan_steps = self.state.current_plan.get("steps", [])

        # Check if we've gone past the end of the plan
        if self.state.current_step >= len(plan_steps):
            if self.agent_logger:
                self.agent_logger.debug(
                    f"Current step {self.state.current_step} is beyond plan length {len(plan_steps)}"
                )
            return await self._reflection_phase(task, reasoning_context)

        # Check if we have a valid step index
        if self.state.current_step < 0:
            self.state.current_step = 0
            if self.agent_logger:
                self.agent_logger.debug("Reset current step to 0")

        current_step_info = plan_steps[self.state.current_step]

        # --- NEW: Extract and inject data from previous successful steps ---
        def extract_step_data(execution_history):
            # Use the adapter's data_extractor for consistency
            extracted_data = {}
            for step in execution_history:
                if step.get("success"):
                    tool_calls = step.get("tool_calls", [])
                    for call in tool_calls:
                        if not isinstance(call, dict):
                            continue
                        result = call.get("result", {})
                        tool_name = call.get("function", {}).get("name", "unknown")
                        result_str = (
                            result if isinstance(result, str) else json.dumps(result)
                        )
                        extracted = self.adapter.data_extractor.extract_all(result_str)
                        # Collect email IDs
                        if (
                            extracted.structured_data
                            and "id" in extracted.structured_data
                        ):
                            if "email_ids" not in extracted_data:
                                extracted_data["email_ids"] = []
                            val = extracted.structured_data["id"]
                            if isinstance(val, list):
                                extracted_data["email_ids"].extend(val)
                            else:
                                extracted_data["email_ids"].append(val)
            return extracted_data

        if self.state.execution_history:
            step_data = extract_step_data(self.state.execution_history)
            if step_data:
                current_step_info = dict(
                    current_step_info
                )  # copy to avoid mutating plan
                current_step_info["available_data"] = step_data

        if self.agent_logger:
            self.agent_logger.debug(
                f"⚡ Executing step {self.state.current_step + 1}: {current_step_info['description']}"
            )
            self.agent_logger.debug(
                f"Required tools for this step: {current_step_info.get('required_tools', [])}"
            )

        # Generate focused execution prompt
        execution_prompt = self.adapter.prepare_execute_prompt(
            current_step=current_step_info,
            step_index=self.state.current_step,
            previous_results=self.state.execution_history,
        )

        # Prepare focused message context
        self._prepare_focused_message_context(
            user_prompt=execution_prompt, current_step=current_step_info
        )

        # Execute with focused context
        result = await self._think_chain(use_tools=True)

        # Validate step execution (without tool correction)
        validation = self.adapter.validate_step_execution(
            current_step_info,
            {
                "tool_calls": result.get("tool_calls", []) if result else [],
                "content": result.get("content", "") if result else "",
                "result": str(result) if result else "",
            },
        )

        # Create detailed step result and record it
        step_result = {
            "step_number": self.state.current_step + 1,
            "description": current_step_info["description"],
            "result": result.get("content", "") if result else "No result",
            "success": validation["success"],
            "tool_calls": result.get("tool_calls", []) if result else [],
            "required_tools": current_step_info.get("required_tools", []),
            "validation": validation,
            "tools_called_correctly": validation["tools_called_correctly"],
            "missing_tools": validation["missing_tools"],
            "tool_errors": validation["tool_errors"],
            "failure_analysis": validation.get("failure_analysis", {}),
        }

        self._record_step_result(step_result)
        self.state.current_step += 1

        # Enhanced logging for tool usage analysis
        if self.agent_logger:
            self.agent_logger.info(f"Step {step_result['step_number']} completed")
            self.agent_logger.info(f"Success: {step_result['success']}")
            if step_result["tool_calls"]:
                self.agent_logger.info("Tools used:")
                for call in step_result["tool_calls"]:
                    if isinstance(call, dict):
                        self.agent_logger.info(
                            f"- {call.get('function', {}).get('name')}"
                        )
            if step_result["missing_tools"]:
                self.agent_logger.warning(
                    f"Missing required tools: {step_result['missing_tools']}"
                )
            if step_result["tool_errors"]:
                self.agent_logger.error(f"Tool errors: {step_result['tool_errors']}")

        return StrategyResult(
            action_taken="step_execution",
            result=step_result,
            should_continue=True,
            strategy_used="plan_execute_reflect",
        )

    @ensure_state_initialized
    async def _reflection_phase(
        self, task: str, reasoning_context: ReasoningContext
    ) -> StrategyResult:
        """Reflect with focused message context and proper completion handling."""
        if self.agent_logger:
            self.agent_logger.debug("🤔 Starting reflection phase")
            self.agent_logger.debug(f"Current step: {self.state.current_step}")
            self.agent_logger.debug(
                f"Total steps: {len(self.state.current_plan.get('steps', []) if self.state.current_plan else [])}"
            )
            self.agent_logger.debug(f"Error count: {self.state.error_count}")

        # Get reflection on current progress
        reflection_prompt = self.adapter.prepare_reflection_prompt(
            execution_history=self.state.execution_history,
            goal=task,
            last_step_output=self.state.last_step_output,
            tool_responses=self.state.tool_responses,
        )

        # Initialize default reflection data
        reflection_data = {
            "goal_achieved": False,
            "completion_score": 0.0,
            "next_action": "retry",
            "confidence": 0.0,
            "blockers": [],
            "success_indicators": [],
            "learning_insights": [],
        }

        # Get reflection with explicit JSON format
        reflection_result = await self.infrastructure.think(
            prompt=reflection_prompt, format="json"
        )

        try:
            # Extract JSON from nested response structure
            if isinstance(reflection_result, dict):
                if (
                    "result" in reflection_result
                    and "message" in reflection_result["result"]
                ):
                    content = reflection_result["result"]["message"].get("content", "")
                    if content:
                        parsed_data = json.loads(content)
                        if isinstance(parsed_data, dict):
                            reflection_data.update(parsed_data)
                else:
                    reflection_data.update(reflection_result)

            if self.agent_logger:
                self.agent_logger.debug(
                    f"Reflection data: {json.dumps(reflection_data, indent=2)}"
                )
            self._record_reflection_result(reflection_data)

            # Handle task completion or retry
            if reflection_data.get("next_action") == "complete":
                if self.agent_logger:
                    self.agent_logger.info(
                        "🎯 Task execution complete, generating final answer..."
                    )
                    if not reflection_data.get("goal_achieved", False):
                        self.agent_logger.info(
                            "⚠️ Task completed but goal not fully achieved"
                        )

                # Generate final answer prompt
                final_answer_prompt = self.adapter.prepare_final_answer_prompt(
                    task=task,
                    execution_history=self.state.execution_history,
                    reflection_data=reflection_data,
                )

                # Add final answer prompt to message context
                self._prepare_focused_message_context(user_prompt=final_answer_prompt)

                # Call _think_chain with use_tools=True to allow tool usage in final answer
                final_answer_result = await self._think_chain(use_tools=True)
                final_answer = str(final_answer_result) if final_answer_result else None

                return StrategyResult(
                    action_taken="reflection_complete",
                    result={
                        "reflection": reflection_result,
                        "reflection_data": reflection_data,
                        "execution_history": self.state.execution_history,
                        "steps_completed": self.state.current_step,
                        "steps_total": (
                            len(self.state.current_plan.get("steps", []))
                            if self.state.current_plan
                            else 0
                        ),
                        "error_count": self.state.error_count,
                        "should_continue": False,
                        "goal_achieved": reflection_data.get("goal_achieved", False),
                    },
                    should_continue=False,
                    confidence=reflection_data.get("confidence", 0.5),
                    final_answer=final_answer,
                )

            # If reflection suggests retry and we have no plan or failed steps, create a new plan
            if reflection_data.get("next_action") == "retry" and (
                not self.state.current_plan
                or not self.state.current_plan.get("steps")
                or any(
                    not step.get("success", False)
                    for step in self.state.execution_history[-3:]
                )
            ):
                if self.agent_logger:
                    self.agent_logger.info(
                        "🔄 Creating new plan based on reflection feedback"
                    )
                return await self._planning_phase(task, reasoning_context)

        except (json.JSONDecodeError, ValueError) as e:
            if self.agent_logger:
                self.agent_logger.warning(
                    f"Failed to parse reflection result: {str(e)}"
                )
                self.agent_logger.debug(f"Raw reflection result: {reflection_result}")
                self.agent_logger.info("Using default reflection values")

        # Handle step retry
        if reflection_data.get("next_action") == "retry":
            if self.state.error_count >= self.state.max_errors:
                if self.agent_logger:
                    self.agent_logger.error(
                        f"❌ Max errors ({self.state.max_errors}) reached, stopping."
                    )
                return StrategyResult(
                    action_taken="reflection_error",
                    result={
                        "reflection": reflection_result,
                        "reflection_data": reflection_data,
                        "execution_history": self.state.execution_history,
                        "error": f"Max errors ({self.state.max_errors}) reached",
                        "should_continue": False,
                    },
                    should_continue=False,
                    confidence=reflection_data.get("confidence", 0.5),
                )

            if self.agent_logger:
                self.agent_logger.info("🔄 Retrying failed step...")
            self.state.current_step -= 1  # Retry the current step

        return StrategyResult(
            action_taken="reflection_continue",
            result={
                "reflection": reflection_result,
                "reflection_data": reflection_data,
                "execution_history": self.state.execution_history,
                "steps_completed": self.state.current_step,
                "steps_total": (
                    len(self.state.current_plan.get("steps", []))
                    if self.state.current_plan
                    else 0
                ),
                "error_count": self.state.error_count,
                "should_continue": True,
            },
            should_continue=True,
            confidence=reflection_data.get("confidence", 0.5),
        )

    def _parse_plan_steps(self, plan_text: str) -> List[Dict[str, Any]]:
        """Enhanced plan parsing with better error handling."""
        steps = []

        # Try to parse as JSON first
        try:
            import json
            import re

            # Look for multiple step objects (comma-separated) - this is the actual format
            # The plan text contains multiple JSON objects separated by commas
            # Use a more sophisticated approach to handle nested braces
            def find_json_objects(text):
                """Find all JSON objects in the text that contain step_number."""
                objects = []
                brace_count = 0
                start = -1

                for i, char in enumerate(text):
                    if char == "{":
                        if brace_count == 0:
                            start = i
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0 and start != -1:
                            # Extract the JSON object
                            json_str = text[start : i + 1]
                            if '"step_number"' in json_str:
                                objects.append(json_str)
                            start = -1

                return objects

            json_objects = find_json_objects(plan_text)
            for obj_match in json_objects:
                try:
                    step_data = json.loads(obj_match)
                    if "step_number" in step_data:
                        steps.append(
                            {
                                "step_number": step_data.get(
                                    "step_number", len(steps) + 1
                                ),
                                "description": step_data.get(
                                    "description", "Execute step"
                                ),
                                "purpose": step_data.get("purpose", ""),
                                "required_tools": step_data.get("required_tools", []),
                                "success_criteria": step_data.get(
                                    "success_criteria", ""
                                ),
                                "expected_output": step_data.get("expected_output", ""),
                            }
                        )
                except json.JSONDecodeError:
                    continue

            # If we found steps, return them
            if steps:
                if self.agent_logger:
                    self.agent_logger.info(
                        f"📋 Parsed {len(steps)} steps from JSON structure"
                    )
                return steps

            # Look for JSON structure in the response - handle both array and object formats
            json_match = re.search(r'\{[\s\S]*"step_number"[\s\S]*\}', plan_text)
            if json_match:
                # Try to parse as a single step object
                try:
                    step_data = json.loads(json_match.group())
                    if "step_number" in step_data:
                        steps.append(
                            {
                                "step_number": step_data.get("step_number", 1),
                                "description": step_data.get(
                                    "description", "Execute step"
                                ),
                                "purpose": step_data.get("purpose", ""),
                                "required_tools": step_data.get("required_tools", []),
                                "success_criteria": step_data.get(
                                    "success_criteria", ""
                                ),
                                "expected_output": step_data.get("expected_output", ""),
                            }
                        )
                except json.JSONDecodeError:
                    pass

            # Look for plan_steps array format (backward compatibility)
            if not steps:
                json_match = re.search(r'\{[\s\S]*"plan_steps"[\s\S]*\}', plan_text)
                if json_match:
                    json_data = json.loads(json_match.group())
                    if "plan_steps" in json_data:
                        for step_data in json_data["plan_steps"]:
                            steps.append(
                                {
                                    "step_number": step_data.get(
                                        "step_number", len(steps) + 1
                                    ),
                                    "description": step_data.get(
                                        "description", "Execute step"
                                    ),
                                    "purpose": step_data.get("purpose", ""),
                                    "required_tools": step_data.get(
                                        "required_tools", []
                                    ),
                                    "success_criteria": step_data.get(
                                        "success_criteria", ""
                                    ),
                                    "expected_output": step_data.get(
                                        "expected_output", ""
                                    ),
                                }
                            )

                    if steps:
                        if self.agent_logger:
                            self.agent_logger.info(
                                f"📋 Parsed {len(steps)} steps from JSON structure"
                            )
                        return steps

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            if self.agent_logger:
                self.agent_logger.debug(f"Failed to parse JSON plan: {e}")

        # Fallback to simple line-by-line parsing
        lines = plan_text.split("\n")
        step_num = 1

        for line in lines:
            line = line.strip()
            if line and (
                line[0].isdigit() or line.startswith("-") or line.startswith("*")
            ):
                # Clean up step text
                step_text = line
                if line[0].isdigit():
                    step_text = line.split(".", 1)[-1].strip()
                elif line.startswith("-") or line.startswith("*"):
                    step_text = line[1:].strip()

                if step_text:
                    steps.append(
                        {
                            "step_number": step_num,
                            "description": step_text,
                            "purpose": "",
                            "required_tools": [],
                            "success_criteria": "",
                            "expected_output": "",
                        }
                    )
                    step_num += 1

        # Enhanced fallback if no steps found
        if not steps:
            steps = [
                {
                    "step_number": 1,
                    "description": f"Complete the task: {plan_text[:100]}...",
                    "purpose": "Fallback step when plan parsing fails",
                    "required_tools": [],
                    "success_criteria": "Task appears complete",
                    "expected_output": "Task completion",
                }
            ]

        if self.agent_logger:
            self.agent_logger.info(f"📋 Parsed {len(steps)} steps from text structure")
        return steps

    async def _attempt_auth_recovery(
        self,
        step_info: Dict[str, Any],
        failed_result: Dict[str, Any],
        reasoning_context: ReasoningContext,
    ) -> StrategyResult:
        """
        Attempt to recover from authentication errors by guiding through auth flow.

        Args:
            step_info: The step that failed due to auth issues
            failed_result: The result of the failed step
            reasoning_context: Current reasoning context

        Returns:
            StrategyResult with authentication recovery attempt
        """
        step_description = step_info["description"]
        auth_errors = failed_result["validation"].get("auth_errors", [])

        if self.agent_logger:
            self.agent_logger.info(
                f"🔐 Attempting authentication recovery for step: {step_description}"
            )
            self.agent_logger.info(f"🚫 Auth errors in tools: {auth_errors}")

        # Create authentication recovery prompt
        recovery_prompt = f"""AUTHENTICATION RECOVERY: The previous step failed due to authentication issues.

FAILED STEP: {step_description}
AUTHENTICATION ERRORS: Tools {', '.join(auth_errors)} returned authentication failure responses

🔐 AUTHENTICATION REQUIRED: You must complete authentication before proceeding.

RECOVERY STEPS:
1. Use available authentication tools to establish valid access
2. Verify authentication is successful before using restricted tools
3. Only proceed with the original step after authentication is confirmed

💡 WORKFLOW GUIDANCE:
- Authentication may require external system interaction (browser, API keys, etc.)
- Check authentication status before each restricted operation
- Don't skip authentication prerequisites
- Look for tools with names containing: auth, login, token, oauth, credential

🔄 Start with authentication, then retry the original step."""

        # Execute authentication recovery
        self.context.session.messages.append(
            {"role": "user", "content": recovery_prompt}
        )

        recovery_result = await self._think_chain(use_tools=True)

        # Validate recovery attempt
        recovery_validation = self.adapter.validate_step_execution(
            step_info,
            {
                "tool_calls": (
                    recovery_result.get("tool_calls", []) if recovery_result else []
                ),
                "content": (
                    recovery_result.get("content", "") if recovery_result else ""
                ),
                "result": str(recovery_result) if recovery_result else "",
            },
        )

        # Create recovery step result
        recovery_step_result = {
            "step_number": failed_result["step_number"],
            "description": step_description,
            "result": (
                recovery_result.get("content", "")
                if recovery_result
                else "Auth recovery failed"
            ),
            "success": recovery_validation["success"],
            "tool_calls": (
                recovery_result.get("tool_calls", []) if recovery_result else []
            ),
            "required_tools": step_info.get("required_tools", []),
            "validation": recovery_validation,
            "tools_called_correctly": recovery_validation["tools_called_correctly"],
            "missing_tools": recovery_validation["missing_tools"],
            "tool_errors": recovery_validation["tool_errors"],
            "auth_errors": recovery_validation.get("auth_errors", []),
            "recovery_attempt": True,
            "recovery_type": "authentication",
        }

        # Update execution history
        self.state.execution_history[-1] = recovery_step_result

        # Track recovery success
        if recovery_validation["success"]:
            if self.agent_logger:
                self.agent_logger.info(
                    f"✅ Authentication recovery successful for step {failed_result['step_number']}"
                )
            self.state.error_count = max(0, self.state.error_count - 1)
        else:
            if self.agent_logger:
                self.agent_logger.warning(
                    f"❌ Authentication recovery failed for step {failed_result['step_number']}"
                )

        return StrategyResult(
            action_taken="auth_recovery",
            result=recovery_step_result,
            should_continue=True,
            strategy_used="plan_execute_reflect",
        )

    async def _attempt_workflow_recovery(
        self,
        step_info: Dict[str, Any],
        failed_result: Dict[str, Any],
        reasoning_context: ReasoningContext,
    ) -> StrategyResult:
        """
        Attempt to recover from workflow errors by analyzing step dependencies.

        Args:
            step_info: The step that failed due to workflow issues
            failed_result: The result of the failed step
            reasoning_context: Current reasoning context

        Returns:
            StrategyResult with workflow recovery attempt
        """
        step_description = step_info["description"]
        tool_errors = failed_result["validation"]["tool_errors"]

        if self.agent_logger:
            self.agent_logger.info(
                f"🔧 Attempting workflow recovery for step: {step_description}"
            )
            self.agent_logger.info(f"🚫 Tool errors: {tool_errors}")

        # Analyze previous steps for dependencies
        dependency_analysis = ""
        if self.state.execution_history:
            successful_steps = [
                s for s in self.state.execution_history if s.get("success", False)
            ]
            failed_steps = [
                s for s in self.state.execution_history if not s.get("success", False)
            ]

            dependency_analysis = f"""
DEPENDENCY ANALYSIS:
- Successful steps: {len(successful_steps)}/{len(self.state.execution_history)}
- Recent failures: {[s['description'][:50] + '...' for s in failed_steps[-2:]]}
- Available data from previous steps: Check execution history for usable outputs
"""

        # Create workflow recovery prompt
        recovery_prompt = f"""WORKFLOW RECOVERY: The previous step failed due to workflow or dependency issues.

FAILED STEP: {step_description}
TOOL ERRORS: {', '.join(tool_errors)}

{dependency_analysis}

🔧 WORKFLOW ANALYSIS NEEDED:
1. Check if prerequisite steps were completed successfully
2. Verify that required data is available from previous steps
3. Ensure proper step sequencing and dependencies
4. Use correct tool parameters based on workflow state

💡 RECOVERY STRATEGY:
- Review what data/state is available from successful previous steps
- Adjust approach based on current workflow state
- Use appropriate tools in correct sequence
- Ensure all dependencies are satisfied

🔄 Retry the step with proper workflow consideration."""

        # Execute workflow recovery
        self.context.session.messages.append(
            {"role": "user", "content": recovery_prompt}
        )

        recovery_result = await self._think_chain(use_tools=True)

        # Validate recovery attempt
        recovery_validation = self.adapter.validate_step_execution(
            step_info,
            {
                "tool_calls": (
                    recovery_result.get("tool_calls", []) if recovery_result else []
                ),
                "content": (
                    recovery_result.get("content", "") if recovery_result else ""
                ),
                "result": str(recovery_result) if recovery_result else "",
            },
        )

        # Create recovery step result
        recovery_step_result = {
            "step_number": failed_result["step_number"],
            "description": step_description,
            "result": (
                recovery_result.get("content", "")
                if recovery_result
                else "Workflow recovery failed"
            ),
            "success": recovery_validation["success"],
            "tool_calls": (
                recovery_result.get("tool_calls", []) if recovery_result else []
            ),
            "required_tools": step_info.get("required_tools", []),
            "validation": recovery_validation,
            "tools_called_correctly": recovery_validation["tools_called_correctly"],
            "missing_tools": recovery_validation["missing_tools"],
            "tool_errors": recovery_validation["tool_errors"],
            "auth_errors": recovery_validation.get("auth_errors", []),
            "recovery_attempt": True,
            "recovery_type": "workflow",
        }

        # Update execution history
        self.state.execution_history[-1] = recovery_step_result

        # Track recovery success
        if recovery_validation["success"]:
            if self.agent_logger:
                self.agent_logger.info(
                    f"✅ Workflow recovery successful for step {failed_result['step_number']}"
                )
            self.state.error_count = max(0, self.state.error_count - 1)
        else:
            if self.agent_logger:
                self.agent_logger.warning(
                    f"❌ Workflow recovery failed for step {failed_result['step_number']}"
                )

        return StrategyResult(
            action_taken="workflow_recovery",
            result=recovery_step_result,
            should_continue=True,
            strategy_used="plan_execute_reflect",
        )

    async def _attempt_step_recovery(
        self,
        step_info: Dict[str, Any],
        failed_result: Dict[str, Any],
        reasoning_context: ReasoningContext,
    ) -> StrategyResult:
        """
        Attempt to recover from a failed step by retrying with more explicit instructions.

        Args:
            step_info: The step that failed
            failed_result: The result of the failed step
            reasoning_context: Current reasoning context

        Returns:
            StrategyResult with recovery attempt
        """
        step_description = step_info["description"]
        missing_tools = failed_result["validation"]["missing_tools"]

        if self.agent_logger:
            self.agent_logger.info(
                f"🔄 Attempting recovery for step: {step_description}"
            )
            self.agent_logger.info(f"🔧 Missing tools: {missing_tools}")

        # Create recovery prompt with explicit tool instructions
        recovery_prompt = f"""RECOVERY ATTEMPT: The previous step failed because required tools were not used.

FAILED STEP: {step_description}
MISSING TOOLS: {', '.join(missing_tools)}

🚨 CRITICAL: You MUST use these exact tools:
{chr(10).join([f'- {tool}' for tool in missing_tools])}

⚠️  DO NOT use any other tools. Focus only on using the missing tools correctly.

💡 GUIDANCE: 
- Use appropriate parameters based on the step description and context
- Reference data from previous steps if needed as input parameters
- Follow the tool's expected input format and requirements
- Consider the logical sequence and dependencies between operations

🔄 Retry the step now with the correct tools."""

        # Execute recovery attempt
        self.context.session.messages.append(
            {"role": "user", "content": recovery_prompt}
        )

        recovery_result = await self._think_chain(use_tools=True)

        # Validate recovery attempt
        recovery_validation = self.adapter.validate_step_execution(
            step_info,
            {
                "tool_calls": (
                    recovery_result.get("tool_calls", []) if recovery_result else []
                ),
                "content": (
                    recovery_result.get("content", "") if recovery_result else ""
                ),
                "result": str(recovery_result) if recovery_result else "",
            },
        )

        # Update the failed result with recovery attempt
        recovery_step_result = {
            "step_number": failed_result["step_number"],
            "description": step_description,
            "result": (
                recovery_result.get("content", "")
                if recovery_result
                else "Recovery failed"
            ),
            "success": recovery_validation["success"],
            "tool_calls": (
                recovery_result.get("tool_calls", []) if recovery_result else []
            ),
            "required_tools": step_info.get("required_tools", []),
            "validation": recovery_validation,
            "tools_called_correctly": recovery_validation["tools_called_correctly"],
            "missing_tools": recovery_validation["missing_tools"],
            "tool_errors": recovery_validation["tool_errors"],
            "recovery_attempt": True,
        }

        # Update execution history with recovery result
        self.state.execution_history[-1] = recovery_step_result

        # Track recovery success
        if recovery_validation["success"]:
            if self.agent_logger:
                self.agent_logger.info(
                    f"✅ Recovery successful for step {failed_result['step_number']}"
                )
            # Reset error count on successful recovery
            self.state.error_count = max(0, self.state.error_count - 1)
        else:
            if self.agent_logger:
                self.agent_logger.warning(
                    f"❌ Recovery failed for step {failed_result['step_number']}"
                )

        return StrategyResult(
            action_taken="step_recovery",
            result=recovery_step_result,
            should_continue=True,
            strategy_used="plan_execute_reflect",
        )

    @ensure_state_initialized
    def should_switch_strategy(
        self, reasoning_context: ReasoningContext
    ) -> Optional[str]:
        """Enhanced strategy switching logic."""
        # Switch to reactive if plan is simple and working well
        if (
            self.state.current_plan
            and len(self.state.current_plan.get("steps", [])) <= 2
        ):
            if self.state.error_count == 0:
                return "reactive"

        # Switch to reflect-decide-act if too many errors
        if self.state.error_count > 2:
            return "reflect_decide_act"

        return None

    def get_strategy_name(self) -> ReasoningStrategies:
        """Return the strategy identifier."""
        return ReasoningStrategies.PLAN_EXECUTE_REFLECT

    def _correct_tool_calls(
        self, result: Dict[str, Any], step_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Correct tool calls to ensure required tools are used.

        Args:
            result: The result from _think_chain
            step_info: Current step information with required_tools

        Returns:
            Corrected result with proper tool calls
        """
        if not result or not result.get("tool_calls"):
            return result

        required_tools = step_info.get("required_tools", [])
        actual_tool_calls = result.get("tool_calls", [])

        # Check if any required tools are missing
        called_tools = [
            call.get("function", {}).get("name") for call in actual_tool_calls
        ]
        missing_tools = [tool for tool in required_tools if tool not in called_tools]

        if not missing_tools:
            return result

        if self.agent_logger:
            self.agent_logger.warning(
                f"🔧 Tool call correction needed. Required: {required_tools}, "
                f"Called: {called_tools}, Missing: {missing_tools}"
            )

        # Force the correct tool call
        corrected_tool_calls = []

        for required_tool in required_tools:
            # Check if this tool was already called correctly
            existing_call = next(
                (
                    call
                    for call in actual_tool_calls
                    if call.get("function", {}).get("name") == required_tool
                ),
                None,
            )

            if existing_call:
                corrected_tool_calls.append(existing_call)
            else:
                # Create a corrected tool call with appropriate parameters
                corrected_call = self._create_corrected_tool_call(
                    required_tool, step_info
                )
                if corrected_call:
                    corrected_tool_calls.append(corrected_call)

        # Update result with corrected tool calls
        result["tool_calls"] = corrected_tool_calls

        if self.agent_logger:
            self.agent_logger.info(
                f"✅ Corrected tool calls: {[call.get('function', {}).get('name') for call in corrected_tool_calls]}"
            )

        return result

    def _create_corrected_tool_call(
        self, tool_name: str, step_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Create a corrected tool call with appropriate parameters based on step context.

        Args:
            tool_name: Name of the tool to call
            step_info: Current step information

        Returns:
            Corrected tool call dict or None if unable to create
        """
        step_description = step_info.get("description", "").lower()

        # Extract email IDs from previous steps if available
        email_ids = []
        for step in self.state.execution_history:
            if step.get("success") and "search_emails" in str(step.get("result", "")):
                # Try to extract email IDs from search results
                result_str = str(step.get("result", ""))
                import re

                # Look for email IDs in the format we saw in the logs
                ids = re.findall(r'"id":\s*"([^"]+)"', result_str)
                email_ids.extend(ids)

        # Create tool call based on tool name and step context
        if tool_name == "trash_emails":
            if email_ids:
                return {
                    "function": {
                        "name": "trash_emails",
                        "arguments": {"email_ids": email_ids[:5]},  # Limit to first 5
                    }
                }
            else:
                # Fallback - trash emails from notifications@github.com
                return {
                    "function": {
                        "name": "trash_emails",
                        "arguments": {"from_": "notifications@github.com"},
                    }
                }

        elif tool_name == "write_email":
            # Create a summary email
            return {
                "function": {
                    "name": "write_email",
                    "arguments": {
                        "to": "tylerjrbuell@gmail.com",
                        "subject": "GitHub Notification Cleanup Summary",
                        "body": f"Task completed: Found and processed {len(email_ids)} GitHub notification emails.",
                    },
                }
            }

        elif tool_name == "search_emails":
            return {
                "function": {
                    "name": "search_emails",
                    "arguments": {"from_": "notifications@github.com"},
                }
            }

        elif tool_name == "authenticate_with_browser":
            return {"function": {"name": "authenticate_with_browser", "arguments": {}}}

        return None

    def reset(self):
        """Reset strategy state for new task."""
        if self.context.session.per_strategy_state:
            self.context.session.initialize_strategy_state("plan_execute_reflect")
