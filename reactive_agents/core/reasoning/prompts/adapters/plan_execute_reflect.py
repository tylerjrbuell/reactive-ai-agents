"""
Plan-Execute-Reflect Strategy Adapter

Provides focused, context-aware prompts for the three phases of plan-execute-reflect reasoning:
1. Plan Phase: Break down goals into actionable steps with dependencies
2. Execute Phase: Execute specific steps with proper tool mapping
3. Reflect Phase: Evaluate progress and determine next actions

This adapter bridges AgentContext with minimal, targeted prompts.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import json
import re
from datetime import datetime

from .base_adapter import BaseStrategyAdapter
from reactive_agents.core.tools.data_extractor import DataExtractor
from reactive_agents.core.reasoning.prompts.base import (
    PlanGenerationPrompt,
    StepExecutionPrompt,
    ReflectionPrompt,
)

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class PlanExecuteReflectStrategyAdapter(BaseStrategyAdapter):
    """
    Adapter for Plan-Execute-Reflect strategy prompts with enhanced failure analysis.

    Focuses on minimal, context-aware prompts while tracking and learning from failures.
    """

    def __init__(self, context: "AgentContext"):
        """Initialize the adapter with tracking capabilities."""
        super().__init__(context)
        self.current_plan = None  # Will be set when a plan is created
        self._prior_reflections: List[str] = []
        self.data_extractor = DataExtractor()

    def prepare_plan_prompt(self, goal: str, task_type: str, **kwargs) -> str:
        """
        Generate a focused planning prompt with dependency tracking using PlanGenerationPrompt.
        """
        prompt = PlanGenerationPrompt(self.context)
        return prompt.generate(task=goal, task_type=task_type)

    def prepare_execute_prompt(
        self,
        current_step: Dict[str, Any],
        step_index: int,
        previous_results: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a focused execution prompt for the current step using StepExecutionPrompt.
        """
        step_number = current_step.get("step_number", step_index + 1)
        description = current_step.get("description", "")
        required_tools = current_step.get("required_tools", [])
        available_data = current_step.get("available_data", {})
        # Optionally, pass previous_results or available_data as context
        prompt = StepExecutionPrompt(self.context)
        return prompt.generate(
            step=description,
            required_tools=required_tools,
            step_number=step_number,
            available_data=available_data,
            context=previous_results,
        )

    def _check_dependencies(
        self, dependencies: Dict[str, Any], previous_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check if all dependencies for a step are met.

        Args:
            dependencies: Dictionary of dependencies
            previous_results: List of previous step results

        Returns:
            Dict with dependency status
        """
        required_steps = dependencies.get("required_steps", [])
        required_outputs = dependencies.get("required_outputs", [])

        # Track missing dependencies
        missing = {"steps": [], "outputs": [], "details": {}}

        # Check required steps
        for step in required_steps:
            if step >= len(previous_results):
                missing["steps"].append(step)
                continue

            if not previous_results[step].get("success", False):
                missing["steps"].append(step)
                missing["details"][f"step_{step}"] = previous_results[step].get(
                    "failure_analysis", {}
                )

        # Check required outputs
        available_outputs = set()
        for result in previous_results:
            if result.get("success", False):
                available_outputs.update(result.get("outputs", {}).keys())

        for output in required_outputs:
            if output not in available_outputs:
                missing["outputs"].append(output)

        return {"met": not (missing["steps"] or missing["outputs"]), "missing": missing}

    def _generate_dependency_error_prompt(
        self, step_num: int, description: str, dependency_status: Dict[str, Any]
    ) -> str:
        """
        Generate an error prompt when dependencies are not met.

        Args:
            step_num: Step number
            description: Step description
            dependency_status: Dependency check results

        Returns:
            Error prompt string
        """
        missing = dependency_status["missing"]
        prompt = [
            f"❌ Cannot execute Step {step_num}: {description}",
            "",
            "🚫 Missing Dependencies:",
        ]

        if missing["steps"]:
            prompt.extend(
                [
                    "Required steps not completed:",
                    *[f"- Step {step}" for step in missing["steps"]],
                ]
            )

            # Add failure details
            for step, details in missing["details"].items():
                if details:
                    prompt.extend(
                        [
                            f"\nFailure details for {step}:",
                            f"Type: {details.get('failure_type', 'unknown')}",
                            f"Cause: {details.get('root_cause', 'unknown')}",
                        ]
                    )

        if missing["outputs"]:
            prompt.extend(
                [
                    "",
                    "Required outputs not available:",
                    *[f"- {output}" for output in missing["outputs"]],
                ]
            )

        prompt.extend(
            [
                "",
                "⚠️ Resolution Required:",
                "1. Review and fix failed dependencies",
                "2. Ensure all required outputs are available",
                "3. Try the step again after dependencies are met",
            ]
        )

        return "\n".join(prompt)

    def _extract_previous_step_data(
        self, previous_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract relevant data from previous steps.

        Args:
            previous_results: List of previous step results

        Returns:
            Dict of extracted data
        """
        extracted = {}

        for result in previous_results:
            if not result.get("success", False):
                continue

            # Extract IDs and references
            content = result.get("content", "")
            if isinstance(content, str):
                # Look for IDs
                id_matches = re.finditer(r"(?:id|ID|Id):\s*([a-zA-Z0-9-]+)", content)
                for match in id_matches:
                    key = f"extracted_id_{len(extracted)}"
                    extracted[key] = match.group(1)

                # Look for email addresses
                email_matches = re.finditer(r"[\w\.-]+@[\w\.-]+\.\w+", content)
                for match in email_matches:
                    key = f"extracted_email_{len(extracted)}"
                    extracted[key] = match.group(0)

            # Extract tool outputs
            tool_calls = result.get("tool_calls", [])
            for call in tool_calls:
                if isinstance(call, dict) and "result" in call:
                    try:
                        result_data = json.loads(call["result"])
                        if isinstance(result_data, dict):
                            extracted.update(result_data)
                    except (json.JSONDecodeError, TypeError):
                        continue

        return extracted

    def validate_step_execution(
        self, step_info: Dict[str, Any], execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate step execution with enhanced dependency checking and intelligent tool mapping.

        Args:
            step_info: The step that was executed
            execution_result: The result of the execution

        Returns:
            Dict with validation results
        """
        required_tools = step_info.get("required_tools", [])
        success_criteria = step_info.get("success_criteria", "")
        dependencies = step_info.get("dependencies", {})
        provides_outputs = dependencies.get("provides_outputs", [])

        tool_calls = execution_result.get("tool_calls", [])
        content = execution_result.get("content", "")
        result = execution_result.get("result", "")

        # Extract actual tools used
        actual_tools_used = []
        for call in tool_calls:
            if isinstance(call, dict):
                tool_name = call.get("name", "")
                if tool_name:
                    actual_tools_used.append(tool_name)

        # Enhanced tool mapping for validation
        tool_mapping = {
            # Email operations
            "send_email": ["write_email", "compose_email"],
            "write_email": ["send_email", "compose_email"],
            "compose_email": ["write_email", "send_email"],
            "trash_emails": ["delete_emails", "move_to_trash"],
            "delete_emails": ["trash_emails", "move_to_trash"],
            "move_to_trash": ["trash_emails", "delete_emails"],
            # Authentication
            "authenticate_with_browser": ["browser", "login", "auth"],
            "browser": ["authenticate_with_browser", "login", "auth"],
            # Search operations
            "search_emails": ["find_emails", "query_emails"],
            "find_emails": ["search_emails", "query_emails"],
            # Label operations
            "list_labels": ["get_labels", "show_labels"],
            "create_label": ["add_label", "make_label"],
        }

        # Check if required tools were used (with mapping)
        tools_called_correctly = True
        missing_tools = []
        tool_errors = []
        auth_errors = []

        for required_tool in required_tools:
            tool_used = False

            # Direct match
            if required_tool in actual_tools_used:
                tool_used = True
            else:
                # Check mapped equivalents
                mapped_tools = tool_mapping.get(required_tool, [])
                for mapped_tool in mapped_tools:
                    if mapped_tool in actual_tools_used:
                        tool_used = True
                        break

                # Check reverse mapping
                for mapped_key, mapped_values in tool_mapping.items():
                    if (
                        required_tool in mapped_values
                        and mapped_key in actual_tools_used
                    ):
                        tool_used = True
                        break

            if not tool_used:
                tools_called_correctly = False
                missing_tools.append(required_tool)

        # Check for authentication errors
        for call in tool_calls:
            if isinstance(call, dict):
                result_data = call.get("result", "")
                if isinstance(result_data, str) and any(
                    auth_error in result_data.lower()
                    for auth_error in [
                        "unauthorized",
                        "unauthenticated",
                        "auth",
                        "login",
                    ]
                ):
                    auth_errors.append(call.get("name", "unknown"))

        # Check for tool execution errors
        for call in tool_calls:
            if isinstance(call, dict):
                if call.get("error"):
                    tool_errors.append(
                        f"{call.get('name', 'unknown')}: {call.get('error')}"
                    )

        # Determine overall success
        # A step is successful if:
        # 1. Required tools were used (or equivalents)
        # 2. No critical errors occurred
        # 3. At least one tool call succeeded
        success = (
            tools_called_correctly
            and not auth_errors
            and any(
                call.get("success", False)
                for call in tool_calls
                if isinstance(call, dict)
            )
        )

        # Enhanced failure analysis
        failure_analysis = {}
        if not success:
            failure_analysis = self.analyze_failure(step_info, execution_result)

        return {
            "success": success,
            "tools_called_correctly": tools_called_correctly,
            "missing_tools": missing_tools,
            "tool_errors": tool_errors,
            "auth_errors": auth_errors,
            "actual_tools_used": actual_tools_used,
            "required_tools": required_tools,
            "failure_analysis": failure_analysis,
        }

    def prepare_reflection_prompt(
        self,
        execution_history: List[Dict[str, Any]],
        goal: str,
        last_step_output: str,
        tool_responses: List[str],
    ) -> str:
        """Generate a focused reflection prompt that evaluates task completion."""
        from reactive_agents.core.reasoning.prompts.base import ReflectionPrompt

        # Get PER-specific state
        per_state = self.context.session.per_strategy_state
        if not per_state:
            raise ValueError("PlanExecuteReflectState not initialized")

        # Get execution summary from PER state
        state_summary = per_state.get_execution_summary()

        # Build detailed execution summary from state's execution history
        execution_summary = []
        for step in per_state.execution_history:
            status = "✅" if step.get("success", False) else "❌"
            desc = step.get("description", "Unknown step")
            tool_calls = step.get("tool_calls", [])
            tool_results = []

            for tc in tool_calls:
                if isinstance(tc, dict):
                    # Handle both direct tool calls and function-wrapped calls
                    tool_name = tc.get("name") or tc.get("function", {}).get(
                        "name", "unknown"
                    )
                    result = tc.get("result", "No result")
                    if isinstance(result, list) and result:
                        result = result[0]  # Take first result if it's a list
                    tool_results.append(f"{tool_name}: {str(result)[:100]}...")

            execution_summary.append(
                {"step": desc, "status": status, "tool_results": tool_results}
            )

        # Create reflection prompt with state-based context
        reflection_prompt = ReflectionPrompt(self.context)
        return reflection_prompt.generate(
            task=goal,
            last_result={
                "execution_summary": execution_summary,
                "total_steps": state_summary["total_steps"],
                "successful_steps": state_summary["successful_steps"],
                "failed_steps": state_summary["failed_steps"],
                "error_count": state_summary["error_count"],
                "reflection_count": state_summary["reflection_count"],
                "current_step": state_summary["current_step"],
                "plan_success_rate": state_summary["plan_success_rate"],
                "step_success_rate": state_summary["step_success_rate"],
                "recovery_success_rate": state_summary["recovery_success_rate"],
                "last_step_output": per_state.last_step_output,
                "tool_responses": per_state.tool_responses,
                "last_reflection": per_state.last_reflection_result,
            },
        )

    def _analyze_dependencies(
        self, execution_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze dependencies across execution history.

        Args:
            execution_history: List of executed steps

        Returns:
            Dict with dependency analysis
        """
        analysis = {
            "unmet_dependencies": [],
            "required_actions": [],
            "available_outputs": set(),
        }

        # Track available outputs
        for step in execution_history:
            if step.get("success", False):
                outputs = step.get("outputs", {})
                analysis["available_outputs"].update(outputs.keys())

        # Check dependencies
        for step in execution_history:
            if not step.get("success", False):
                dependencies = step.get("dependencies", {})

                # Check required steps
                for req_step in dependencies.get("required_steps", []):
                    if req_step >= len(execution_history) or not execution_history[
                        req_step
                    ].get("success", False):
                        analysis["unmet_dependencies"].append(
                            f"Step {step['step_number']} requires step {req_step}"
                        )
                        analysis["required_actions"].append(
                            f"Complete step {req_step} successfully"
                        )

                # Check required outputs
                for output in dependencies.get("required_outputs", []):
                    if output not in analysis["available_outputs"]:
                        analysis["unmet_dependencies"].append(
                            f"Step {step['step_number']} requires output: {output}"
                        )
                        analysis["required_actions"].append(
                            f"Ensure output '{output}' is provided"
                        )

        return analysis

    def should_continue(
        self,
        reflection_data: Dict[str, Any],
        max_iterations: int = 10,
        current_iteration: int = 0,
        error_count: int = 0,
    ) -> bool:
        """
        Determine if execution should continue based on reflection data and execution limits.

        Args:
            reflection_data: Data from reflection phase
            max_iterations: Maximum number of iterations allowed
            current_iteration: Current iteration count
            error_count: Number of errors encountered

        Returns:
            True if execution should continue, False otherwise
        """
        # Check iteration limits
        if current_iteration >= max_iterations:
            return False

        # Check if goal is achieved
        if reflection_data.get("goal_achieved", False):
            return False

        # Check confidence level
        confidence = reflection_data.get("confidence", 0.5)
        if confidence >= 0.9:  # High confidence threshold
            return False

        # Check for critical failures
        if error_count >= 3:
            return False

        # Check next action
        next_action = reflection_data.get("next_action", "continue")
        if next_action.lower() in ["stop", "complete", "done"]:
            return False

        # Get insights about tool reliability
        insights = self.get_failure_insights()
        problematic_tools = insights["tool_reliability"]["problematic_tools"]
        if len(problematic_tools) >= 2:
            return False

        return True

    def prepare_final_answer_prompt(
        self,
        task: str,
        execution_history: List[Dict[str, Any]],
        reflection_data: Dict[str, Any],
    ) -> str:
        """
        Generate a final answer prompt when task is complete.
        Uses the FinalAnswerPrompt template for consistent formatting.
        """
        from reactive_agents.core.reasoning.prompts.base import FinalAnswerPrompt

        # Create success indicators from execution history
        success_indicators = []
        for step in execution_history:
            if step.get("success", False):
                success_indicators.append(
                    f"✅ {step.get('description', 'Step completed')}"
                )

        # Get the final answer prompt using the template
        prompt = FinalAnswerPrompt(self.context).generate(
            task=task,
            reflection=reflection_data,
            success_indicators=success_indicators,
            execution_history=execution_history,
        )

        return prompt
