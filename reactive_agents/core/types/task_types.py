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


class PlanStep(BaseModel):
    """A single step in a task execution plan."""

    description: str = Field(..., description="Description of what this step does")
    status: StepStatus = Field(
        default=StepStatus.PENDING, description="Current status of this step"
    )
    suggested_tools: List[str] = Field(
        default_factory=list, description="Tools that might be needed for this step"
    )
    dependencies: List[int] = Field(
        default_factory=list,
        description="Step indices that must complete before this step",
    )
    result: Optional[str] = None
    error: Optional[str] = None
    retries: int = Field(
        default=0, description="Number of times this step has been retried"
    )
    max_retries: int = Field(default=3, description="Maximum number of retries allowed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional step metadata"
    )
    index: int = Field(default=0, description="Index of this step in the plan")
    completed_at: Optional[float] = Field(
        default=None, description="Timestamp when step was completed"
    )
    tool_used: Optional[str] = Field(
        default=None, description="Name of the tool used in this step"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Parameters used with the tool"
    )
