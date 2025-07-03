from __future__ import annotations
import json
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from reactive_agents.core.types.task_types import (
    TaskType,
    TaskClassification,
)

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class TaskClassifier:
    """Classifies tasks at runtime to inform reasoning strategy and tool usage."""

    def __init__(self, context: "AgentContext"):
        self.context = context
        self.agent_logger = context.agent_logger
        self.model_provider = context.model_provider

    async def classify_task(
        self, task: str, context_messages: Optional[List[Dict[str, Any]]] = None
    ) -> TaskClassification:
        """
        Classify a task to determine its type, complexity, and requirements.

        Args:
            task: The task description to classify
            context_messages: Optional conversation context for better classification

        Returns:
            TaskClassification with type, confidence, and metadata
        """
        if self.agent_logger:
            self.agent_logger.debug(f"Classifying task: {task[:100]}...")

        # Get available tools for classification context
        available_tools = [
            tool.get("function", {}).get("name", "")
            for tool in self.context.get_tool_signatures()
        ]

        classification_prompt = self._build_classification_prompt(
            task, available_tools, context_messages
        )

        try:
            if not self.model_provider:
                raise Exception("No model provider available")
            response = await self.model_provider.get_completion(
                system=classification_prompt,
                prompt=f"Classify this task: {task}",
                options=self.context.model_provider_options,
            )

            if response and response.message.content:
                classification_data = json.loads(response.message.content)
                return TaskClassification(**classification_data)

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.warning(f"Task classification failed: {e}")

        # Fallback classification
        return self._fallback_classification(task, available_tools)

    def _build_classification_prompt(
        self,
        task: str,
        available_tools: List[str],
        context_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build the classification prompt based on task and available context."""

        task_types_info = {
            "simple_lookup": "Quick information retrieval, single answer",
            "tool_required": "Needs specific tools to complete",
            "creative_generation": "Requires creative writing, brainstorming, or content creation",
            "multi_step": "Complex task requiring multiple sequential actions",
            "agent_collaboration": "Would benefit from multiple agents or perspectives",
            "external_context_required": "Needs external data or real-time information",
            "analysis": "Requires data analysis, interpretation, or evaluation",
            "planning": "Involves creating plans, strategies, or roadmaps",
            "execution": "Direct action or implementation task",
        }

        context_info = ""
        if context_messages:
            context_info = (
                f"\nConversation context: {json.dumps(context_messages[-3:], indent=2)}"
            )

        return f"""You are a task classification expert. Analyze the given task and classify it.

Available task types:
{json.dumps(task_types_info, indent=2)}

Available tools: {', '.join(available_tools)}

{context_info}

Respond with JSON in this exact format:
{{
    "task_type": "<one of the task types above>",
    "confidence": <float 0.0-1.0>,
    "reasoning": "<explanation of classification>",
    "suggested_tools": ["<tool_name>", ...],
    "complexity_score": <float 0.0-1.0>,
    "requires_collaboration": <boolean>,
    "estimated_steps": <integer>
}}

Guidelines:
- Consider the task's inherent complexity and requirements
- Match available tools to task needs
- Be conservative with collaboration requirements
- Estimate realistic step counts (1-10 typical range)
- Confidence should reflect certainty of classification"""

    def _fallback_classification(
        self, task: str, available_tools: List[str]
    ) -> TaskClassification:
        """Provide a fallback classification when LLM classification fails."""

        # Simple heuristics for fallback
        task_lower = task.lower()

        if any(word in task_lower for word in ["search", "find", "lookup", "what is"]):
            task_type = TaskType.SIMPLE_LOOKUP
            complexity = 0.2
        elif any(word in task_lower for word in ["analyze", "compare", "evaluate"]):
            task_type = TaskType.ANALYSIS
            complexity = 0.6
        elif any(
            word in task_lower for word in ["create", "write", "generate", "compose"]
        ):
            task_type = TaskType.CREATIVE_GENERATION
            complexity = 0.5
        elif any(
            word in task_lower for word in ["plan", "strategy", "roadmap", "steps"]
        ):
            task_type = TaskType.PLANNING
            complexity = 0.7
        else:
            task_type = TaskType.MULTI_STEP
            complexity = 0.5

        # Check if tools are likely needed
        if available_tools and any(
            word in task_lower for word in ["file", "search", "data", "calculate"]
        ):
            task_type = TaskType.TOOL_REQUIRED

        return TaskClassification(
            task_type=task_type,
            confidence=0.5,  # Low confidence for fallback
            reasoning="Fallback classification based on keyword heuristics",
            suggested_tools=[],
            complexity_score=complexity,
            requires_collaboration=False,
            estimated_steps=max(1, int(complexity * 5)),
        )
