from reactive_agents.core.context.agent_context import AgentContext
from reactive_agents.core.types.reasoning_types import (
    TaskGoalEvaluationContext,
    TaskGoalEvaluationResult,
)
from reactive_agents.core.reasoning.prompts.base import (
    TaskGoalEvaluationPrompt,
    BasePrompt,
)
import json
from typing import Any, Dict, Optional, Type
from reactive_agents.providers.llm.base import BaseModelProvider, CompletionResponse


class TaskGoalEvaluator:
    """
    Generalized LLM-powered task completion evaluator.
    Can be used in any agent strategy phase to dynamically determine if a task is complete.

    Example usage:
        evaluator = TaskGoalEvaluator(llm, context)
        result = evaluator.is_complete()
        # result: TaskGoalEvaluationResult
    """

    def __init__(
        self,
        agent_context: AgentContext,
        eval_context: TaskGoalEvaluationContext,
        min_completion_score: float = 0.85,
        prompt_class: Optional[BasePrompt] = None,
        model_provider: Optional[BaseModelProvider] = None,
    ):
        """
        Args:
            model_provider: An OpenAI-compatible LLM client or wrapper with a chat_completion method.
            agent_context: The agent's context object (for prompt class instantiation).
            eval_context: TaskGoalEvaluationContext with task_description, progress_summary, latest_output, execution_log, and meta fields.
            prompt_class: An instance of a prompt class (e.g., TaskGoalEvaluationPrompt) for prompt generation.
            min_completion_score: Threshold for considering a task complete.
        """
        self.model_provider = model_provider
        self.agent_context = agent_context
        self.eval_context = eval_context
        self.min_completion_score = min_completion_score
        self.prompt = prompt_class or TaskGoalEvaluationPrompt(context=agent_context)

    async def get_goal_evaluation(self) -> TaskGoalEvaluationResult:
        """
        Evaluate task completion using the LLM.
        Returns:
            TaskGoalEvaluationResult
        """
        system = """
            You are an expert task evaluator. You will be provided with the following information:
            - Task description
            - Progress summary
            - Latest output
            - Execution log
            - Meta data

            Your job is to determine whether the task has been completed successfully.
            You will also be given a set of success criteria that the task must meet to be considered complete.

            For your evaluation response, provide:
            - A completion status (True/False)
            - A completion score (confidence between 0 and 1)
            - A reasoning explanation for your evaluation
            - Any missing requirements, if applicable
            
            Respond in JSON format.
        """
        prompt = self._build_prompt()
        if not self.model_provider:
            raise ValueError("Model is not set in the context")
        response = await self.model_provider.get_completion(
            system=system,
            prompt=prompt,
            format="json",
        )
        result = self._parse_response(response)
        return TaskGoalEvaluationResult(
            completion=result.get("completion", False),
            completion_score=result.get("completion_score", 0.0),
            reasoning=result.get("reasoning", ""),
            missing_requirements=result.get("missing_requirements", []),
        )

    def _build_prompt(self) -> str:
        prompt_str = self.prompt.generate(
            task=self.eval_context.task_description,
            progress_summary=self.eval_context.progress_summary,
            latest_output=self.eval_context.latest_output,
            execution_log=self.eval_context.execution_log,
            meta=self.eval_context.meta,
        )
        return prompt_str

    def _parse_response(self, response: CompletionResponse) -> Dict[str, Any]:
        try:
            content = response.message.content
            return json.loads(content)
        except Exception as e:
            print("Evaluation parse error:", e)
            return {
                "completion": False,
                "completion_score": 0.0,
                "reasoning": "Failed to parse model response.",
                "missing_requirements": [],
            }
