from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from .status_types import StepStatus


class TaskType(Enum):
    """Classification of task types to inform reasoning strategy selection."""

    SIMPLE_LOOKUP = "simple_lookup"
    TOOL_REQUIRED = "tool_required"
    CREATIVE_GENERATION = "creative_generation"
    MULTI_STEP = "multi_step"
    AGENT_COLLABORATION = "agent_collaboration"
    EXTERNAL_CONTEXT_REQUIRED = "external_context_required"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"


class TaskClassification(BaseModel):
    """Result of task classification with confidence and reasoning."""

    task_type: TaskType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    suggested_tools: List[str] = []
    complexity_score: float = 0.0  # 0.0 to 1.0
    requires_collaboration: bool = False
    estimated_steps: int = 1

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom model_dump to ensure TaskType enum is serialized to its value."""
        data = super().model_dump(**kwargs)
        # Ensure task_type is serialized to its value, not its name
        if "task_type" in data and isinstance(data["task_type"], TaskType):
            data["task_type"] = data["task_type"].value
        return data


class TaskComplexity(Enum):
    """Task complexity levels."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"
