from __future__ import annotations
import json
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from reactive_agents.core.types.task_types import (
    TaskType,
    TaskClassification,
)
from reactive_agents.core.reasoning.performance_monitor import (
    StrategyPerformanceMonitor,
)

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext


class TaskClassifier:
    """Classifies tasks at runtime to inform reasoning strategy and tool usage."""

    def __init__(self, context: "AgentContext"):
        self.context = context
        self.agent_logger = context.agent_logger
        self.model_provider = context.model_provider
        self.performance_monitor: Optional[StrategyPerformanceMonitor] = None

    def set_performance_monitor(
        self, performance_monitor: StrategyPerformanceMonitor
    ) -> None:
        """
        Set the performance monitor for strategy recommendation enhancement.

        Args:
            performance_monitor: The performance monitor instance
        """
        self.performance_monitor = performance_monitor

    async def classify_task_with_performance(
        self, task: str, context_messages: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced task classification that includes performance-based strategy recommendations.

        Args:
            task: The task description to classify
            context_messages: Optional conversation context for better classification

        Returns:
            Extended classification with performance-based recommendations
        """
        # Get base classification
        base_classification = await self.classify_task(task, context_messages)

        # Create enhanced result with performance data
        enhanced_result = {
            "base_classification": base_classification.model_dump(),
            "performance_recommendations": {},
            "strategy_rankings": [],
            "recommended_strategy": None,
            "confidence_adjustment": 0.0,
        }

        # Add performance-based recommendations if monitor is available
        if self.performance_monitor:
            strategy_rankings = self.performance_monitor.get_strategy_rankings()
            enhanced_result["strategy_rankings"] = strategy_rankings

            # Get task-type specific recommendations
            task_type = base_classification.task_type
            performance_recommendations = self._get_performance_recommendations(
                task_type, base_classification.complexity_score, strategy_rankings
            )
            enhanced_result["performance_recommendations"] = performance_recommendations

            # Recommend best performing strategy for this task type
            recommended_strategy = self._recommend_strategy_by_performance(
                task_type, base_classification, strategy_rankings
            )
            enhanced_result["recommended_strategy"] = recommended_strategy

            # Adjust confidence based on performance data availability
            if strategy_rankings:
                enhanced_result["confidence_adjustment"] = (
                    0.1  # Boost confidence when we have data
                )

        return enhanced_result

    def _get_performance_recommendations(
        self,
        task_type: TaskType,
        complexity_score: float,
        strategy_rankings: List[tuple],
    ) -> Dict[str, Any]:
        """
        Generate performance-based recommendations for a task type.

        Args:
            task_type: The classified task type
            complexity_score: Task complexity score
            strategy_rankings: List of (strategy_name, performance_score) tuples

        Returns:
            Dictionary containing performance recommendations
        """
        recommendations = {
            "high_performing_strategies": [],
            "avoid_strategies": [],
            "complexity_considerations": {},
            "performance_insights": [],
        }

        if not strategy_rankings:
            recommendations["performance_insights"].append(
                "No performance data available - using heuristic recommendations"
            )
            return recommendations

        # Identify high and low performing strategies
        high_threshold = 0.7
        low_threshold = 0.4

        for strategy_name, performance_score in strategy_rankings:
            if performance_score >= high_threshold:
                recommendations["high_performing_strategies"].append(
                    {
                        "strategy": strategy_name,
                        "score": performance_score,
                        "reason": f"High performance score: {performance_score:.2f}",
                    }
                )
            elif performance_score <= low_threshold:
                recommendations["avoid_strategies"].append(
                    {
                        "strategy": strategy_name,
                        "score": performance_score,
                        "reason": f"Low performance score: {performance_score:.2f}",
                    }
                )

        # Add complexity-specific recommendations
        if complexity_score > 0.8:
            recommendations["complexity_considerations"][
                "high_complexity"
            ] = "Task has high complexity - consider strategies with planning capabilities"
        elif complexity_score < 0.3:
            recommendations["complexity_considerations"][
                "low_complexity"
            ] = "Task has low complexity - reactive strategies may be sufficient"

        # Add general performance insights
        if len(strategy_rankings) >= 2:
            best_strategy, best_score = strategy_rankings[0]
            worst_strategy, worst_score = strategy_rankings[-1]
            performance_gap = best_score - worst_score

            if performance_gap > 0.3:
                recommendations["performance_insights"].append(
                    f"Significant performance gap detected: {best_strategy} "
                    f"({best_score:.2f}) vs {worst_strategy} ({worst_score:.2f})"
                )

        return recommendations

    def _recommend_strategy_by_performance(
        self,
        task_type: TaskType,
        classification: TaskClassification,
        strategy_rankings: List[tuple],
    ) -> Optional[Dict[str, Any]]:
        """
        Recommend the best strategy based on performance data and task characteristics.

        Args:
            task_type: The classified task type
            classification: The base task classification
            strategy_rankings: List of (strategy_name, performance_score) tuples

        Returns:
            Strategy recommendation with reasoning
        """
        if not strategy_rankings:
            return None

        # Task type to preferred strategy mapping (heuristic baseline)
        task_strategy_preferences = {
            TaskType.SIMPLE_LOOKUP: ["reactive"],
            TaskType.TOOL_REQUIRED: ["reactive", "plan_execute_reflect"],
            TaskType.CREATIVE_GENERATION: ["reactive", "reflect_decide_act"],
            TaskType.MULTI_STEP: ["plan_execute_reflect", "reflect_decide_act"],
            TaskType.ANALYSIS: ["reflect_decide_act", "plan_execute_reflect"],
            TaskType.PLANNING: ["plan_execute_reflect"],
            TaskType.EXECUTION: ["reactive", "plan_execute_reflect"],
        }

        preferred_strategies = task_strategy_preferences.get(task_type, [])

        # Find the best performing strategy among preferred ones
        best_preferred = None
        best_preferred_score = 0.0

        for strategy_name, performance_score in strategy_rankings:
            if strategy_name in preferred_strategies:
                if performance_score > best_preferred_score:
                    best_preferred = strategy_name
                    best_preferred_score = performance_score

        # Get the overall best performing strategy
        best_overall, best_overall_score = strategy_rankings[0]

        # Decision logic
        if best_preferred and best_preferred_score >= 0.6:
            # Use preferred strategy if it performs reasonably well
            return {
                "strategy": best_preferred,
                "confidence": min(0.9, classification.confidence + 0.1),
                "reasoning": f"Best performing strategy ({best_preferred_score:.2f}) "
                f"for {task_type.value} tasks",
                "performance_score": best_preferred_score,
                "selection_reason": "task_type_and_performance",
            }
        elif best_overall_score > 0.7:
            # Use best overall strategy if it's significantly better
            return {
                "strategy": best_overall,
                "confidence": classification.confidence,
                "reasoning": f"Best overall performing strategy ({best_overall_score:.2f})",
                "performance_score": best_overall_score,
                "selection_reason": "best_performance",
            }
        else:
            # Fall back to heuristic recommendation
            fallback_strategy = (
                preferred_strategies[0] if preferred_strategies else "reactive"
            )
            return {
                "strategy": fallback_strategy,
                "confidence": max(0.3, classification.confidence - 0.2),
                "reasoning": f"Heuristic fallback for {task_type.value} tasks "
                f"(limited performance data)",
                "performance_score": None,
                "selection_reason": "heuristic_fallback",
            }

    def get_strategy_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of strategy performance data.

        Returns:
            Summary of strategy performance information
        """
        if not self.performance_monitor:
            return {"available": False, "message": "No performance monitor configured"}

        rankings = self.performance_monitor.get_strategy_rankings()
        summary = self.performance_monitor.get_performance_summary()

        return {
            "available": True,
            "total_strategies": len(rankings),
            "strategy_rankings": rankings,
            "performance_summary": summary,
            "recommendations": {
                "high_performing": [name for name, score in rankings if score > 0.7],
                "needs_improvement": [name for name, score in rankings if score < 0.4],
            },
        }

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
                format="json",  # Request JSON format to trigger cleaning
                options=self.context.model_provider_options,
            )

            if response and response.message.content:
                # Log the raw response for debugging
                if self.agent_logger:
                    self.agent_logger.debug(
                        f"Raw classification response: {response.message.content[:500]}..."
                    )

                try:
                    classification_data = json.loads(response.message.content)

                    # Validate that we have the required fields
                    if not isinstance(classification_data, dict):
                        raise ValueError("Response is not a dictionary")

                    # Check for required fields
                    required_fields = ["task_type", "confidence", "reasoning"]
                    missing_fields = [
                        field
                        for field in required_fields
                        if field not in classification_data
                    ]

                    if missing_fields:
                        raise ValueError(f"Missing required fields: {missing_fields}")

                    # Validate task_type is a valid enum value
                    task_type_value = classification_data.get("task_type")
                    if task_type_value not in [t.value for t in TaskType]:
                        raise ValueError(f"Invalid task_type: {task_type_value}")

                    # Validate confidence is a number between 0 and 1
                    confidence = classification_data.get("confidence")
                    if not isinstance(confidence, (int, float)) or not (
                        0.0 <= confidence <= 1.0
                    ):
                        raise ValueError(f"Invalid confidence value: {confidence}")

                    # Validate reasoning is a string
                    reasoning = classification_data.get("reasoning")
                    if not isinstance(reasoning, str) or not reasoning.strip():
                        raise ValueError("Reasoning must be a non-empty string")

                    if self.agent_logger:
                        self.agent_logger.debug(
                            f"Successfully classified task as: {task_type_value} (confidence: {confidence})"
                        )

                    return TaskClassification(**classification_data)

                except (json.JSONDecodeError, ValueError) as parse_error:
                    if self.agent_logger:
                        self.agent_logger.warning(
                            f"JSON parsing/validation failed: {parse_error}. Content: {response.message.content[:200]}..."
                        )
                    raise
            else:
                if self.agent_logger:
                    self.agent_logger.warning("No content received from model provider")

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

CRITICAL: You MUST respond with valid JSON in this exact format. Do not include any text before or after the JSON object.

{{
    "task_type": "<one of the task types above>",
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<explanation of classification>",
    "suggested_tools": ["<tool_name>", ...],
    "complexity_score": <float between 0.0 and 1.0>,
    "requires_collaboration": <boolean true or false>,
    "estimated_steps": <integer>
}}

Examples of valid responses:

For a simple lookup task:
{{
    "task_type": "simple_lookup",
    "confidence": 0.9,
    "reasoning": "This is a straightforward information retrieval task that requires a single answer",
    "suggested_tools": [],
    "complexity_score": 0.2,
    "requires_collaboration": false,
    "estimated_steps": 1
}}

For a complex analysis task:
{{
    "task_type": "analysis",
    "confidence": 0.8,
    "reasoning": "This task requires data analysis and interpretation of multiple factors",
    "suggested_tools": ["data_analyzer", "calculator"],
    "complexity_score": 0.7,
    "requires_collaboration": false,
    "estimated_steps": 5
}}

Guidelines:
- task_type: Must be one of the exact values listed above
- confidence: Must be a number between 0.0 and 1.0
- reasoning: Must be a non-empty string explaining your classification
- suggested_tools: Array of tool names that might be useful (can be empty)
- complexity_score: Number between 0.0 and 1.0 indicating task complexity
- requires_collaboration: Boolean indicating if multiple agents would help
- estimated_steps: Integer representing expected number of steps (1-10 typical)

Remember: Respond ONLY with the JSON object, no additional text."""

    def _fallback_classification(
        self, task: str, available_tools: List[str]
    ) -> TaskClassification:
        """Provide a fallback classification when LLM classification fails."""

        if self.agent_logger:
            self.agent_logger.info(
                "Using fallback classification due to LLM classification failure"
            )

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

        fallback_result = TaskClassification(
            task_type=task_type,
            confidence=0.5,  # Low confidence for fallback
            reasoning="Fallback classification based on keyword heuristics",
            suggested_tools=[],
            complexity_score=complexity,
            requires_collaboration=False,
            estimated_steps=max(1, int(complexity * 5)),
        )

        if self.agent_logger:
            self.agent_logger.info(
                f"Fallback classification result: {task_type.value} (confidence: 0.5)"
            )

        return fallback_result
