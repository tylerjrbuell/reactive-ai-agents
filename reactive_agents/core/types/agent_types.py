from typing import List, Literal, Optional, Dict, Any, Set
from pydantic import BaseModel, Field, model_validator, ConfigDict

from reactive_agents.core.types.tool_types import ProcessedToolCall
from reactive_agents.providers.external.client import MCPClient
from reactive_agents.core.types.confirmation_types import ConfirmationCallbackProtocol
from reactive_agents.config.mcp_config import MCPConfig


class AgentThinkResult(BaseModel):
    """Result of a single thinking step."""

    result: dict = Field(description="The result of the thinking step.")
    result_json: dict = Field(
        default={}, description="The result of the thinking step as a JSON object."
    )
    content: str = Field(description="The content of the thinking step.")


class AgentThinkChainResult(BaseModel):
    """Result of a chain of thinking steps."""

    result: dict = Field(description="The result of the thinking chain.")
    content: str = Field(description="The content of the thinking chain.")
    result_json: dict = Field(
        default={}, description="The result of the thinking chain as a JSON object."
    )
    tool_calls: List[ProcessedToolCall] = Field(
        default_factory=list, description="The tool calls of the thinking chain."
    )


# --- Agent Configuration Model ---
class ReactiveAgentConfig(BaseModel):
    # Required parameters
    agent_name: str = Field(description="Name of the agent.")
    provider_model_name: str = Field(description="Name of the LLM provider and model.")

    # Optional parameters
    model_provider_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Options for the LLM provider."
    )
    role: Optional[str] = Field(
        default="Task Executor", description="Role of the agent."
    )
    mcp_client: Optional[MCPClient] = Field(
        default=None, description="An initialized MCPClient instance."
    )
    mcp_config: Optional[MCPConfig] = Field(
        default=None, description="MCP config dict or file path to use for MCPClient."
    )
    mcp_server_filter: Optional[List[str]] = Field(
        default_factory=list,
        description="Filter List of MCP servers for the agent to use in the MCPClient.",
    )
    min_completion_score: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Minimum score for task completion evaluation.",
    )
    instructions: Optional[str] = Field(
        default="Solve the given task.",
        description="High-level instructions for the agent.",
    )
    max_iterations: Optional[int] = Field(
        default=10, description="Maximum number of iterations allowed."
    )
    reflect_enabled: Optional[bool] = Field(
        default=True, description="Whether reflection mechanism is enabled."
    )
    log_level: Optional[Literal["debug", "info", "warning", "error", "critical"]] = (
        Field(
            default="info",
            description="Logging level ('debug', 'info', 'warning', 'error' or 'critical').",
        )
    )
    initial_task: Optional[str] = Field(
        default=None,
        description="The initial task description (can also be passed to run).",
    )
    tool_use_enabled: Optional[bool] = Field(
        default=True, description="Whether the agent can use tools."
    )
    tools: Optional[List[Any]] = Field(
        default_factory=list,
        description="List of custom tool instances to use with the agent.",
    )
    use_memory_enabled: Optional[bool] = Field(
        default=True, description="Whether the agent uses long-term memory."
    )
    vector_memory_enabled: Optional[bool] = Field(
        default=False,
        description="Whether to use ChromaDB vector memory for semantic search.",
    )
    vector_memory_collection: Optional[str] = Field(
        default=None, description="Name of the ChromaDB collection for vector memory."
    )
    collect_metrics_enabled: Optional[bool] = Field(
        default=True, description="Whether to collect performance metrics."
    )
    check_tool_feasibility: Optional[bool] = Field(
        default=True, description="Whether to check tool feasibility before starting."
    )
    enable_caching: Optional[bool] = Field(
        default=True, description="Whether to enable LLM response caching."
    )
    confirmation_callback: Optional[ConfirmationCallbackProtocol] = Field(
        default=None,
        description="Callback for confirming tool use. Can return bool or (bool, feedback).",
    )
    confirmation_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration for tool confirmation behavior. If None, defaults will be used.",
    )
    # Add workflow_context_shared field to ReactAgentConfig
    workflow_context_shared: Optional[Dict[str, Any]] = Field(
        default=None, description="Shared workflow context data."
    )

    # Response format configuration
    response_format: Optional[str] = Field(
        default=None, description="Format specification for the agent's final response."
    )

    # --- Context Management and Tool Use Policy Configuration ---
    max_context_messages: int = Field(
        default=20, description="Maximum number of context messages to retain."
    )
    max_context_tokens: Optional[int] = Field(
        default=None, description="Maximum number of context tokens to retain."
    )
    enable_context_pruning: bool = Field(
        default=True, description="Whether to enable context pruning."
    )
    enable_context_summarization: bool = Field(
        default=True, description="Whether to enable context summarization."
    )
    context_pruning_strategy: Literal["conservative", "balanced", "aggressive"] = Field(
        default="balanced", description="Context pruning strategy."
    )
    context_token_budget: int = Field(
        default=4000, description="Token budget for context management."
    )
    context_pruning_aggressiveness: Literal[
        "conservative", "balanced", "aggressive"
    ] = Field(default="balanced", description="Aggressiveness of context pruning.")
    context_summarization_frequency: int = Field(
        default=3, description="Number of iterations between context summarizations."
    )
    tool_use_policy: Literal["always", "required_only", "adaptive", "never"] = Field(
        default="adaptive", description="Policy for tool use in the agent loop."
    )
    tool_use_max_consecutive_calls: int = Field(
        default=3,
        description="Maximum consecutive tool calls before forcing reflection/summarization.",
    )

    # Advanced reasoning strategy configuration
    reasoning_strategy: str = Field(
        default="reactive", description="Initial reasoning strategy to use."
    )
    enable_reactive_execution: bool = Field(
        default=True, description="Whether to enable reactive execution engine."
    )
    enable_dynamic_strategy_switching: bool = Field(
        default=True,
        description="Whether to enable dynamic strategy switching based on task classification.",
    )

    # Strategy execution mode configuration
    strategy_mode: str = Field(
        default="adaptive",
        description="Strategy mode: 'static' for fixed strategy, 'adaptive' for dynamic switching.",
    )
    static_strategy: str = Field(
        default="reactive",
        description="Strategy to use when strategy_mode is 'static'.",
    )

    # Store extra kwargs passed, e.g. for specific context managers
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to AgentContext.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow MCPClient etc.

    @model_validator(mode="before")
    def capture_extra_kwargs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Get all known fields from the model
        known_fields = set(cls.model_fields.keys())

        # Fields that should be processed normally (not captured as extra kwargs)
        normal_fields = known_fields - {"kwargs"}

        extra_kwargs = {}
        processed_values = {}

        for key, value in values.items():
            if key in normal_fields:
                processed_values[key] = value
            else:
                extra_kwargs[key] = value

        # Only add extra kwargs for truly unknown fields
        if extra_kwargs:  # Only add kwargs if there are actually extra fields
            processed_values["kwargs"] = extra_kwargs

        return processed_values


class PlanFormat(BaseModel):
    """Format for agent planning data."""

    next_step: str
    rationale: str
    suggested_tools: List[str] = []


class ToolAnalysisFormat(BaseModel):
    """Format for tool analysis data."""

    required_tools: List[str] = Field(
        ..., description="List of tools essential for this task"
    )
    optional_tools: List[str] = Field(
        [], description="List of tools helpful but not essential"
    )
    explanation: str = Field(
        ..., description="Brief explanation of the tool requirements"
    )


class RescopeFormat(BaseModel):
    """Format for task rescoping data."""

    rescoped_task: Optional[str] = Field(
        None,
        description="A simplified, achievable task, or null if no rescope possible.",
    )
    explanation: str = Field(
        ..., description="Why this task was/wasn't rescoped and justification."
    )
    expected_tools: List[str] = Field(
        [], description="Tools expected for the rescoped task (if any)."
    )


class EvaluationFormat(BaseModel):
    """Format for goal evaluation data."""

    adherence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score 0.0-1.0: how well the result matches the goal",
    )
    strengths: List[str] = Field(
        [], description="Ways the result successfully addressed the goal"
    )
    weaknesses: List[str] = Field(
        [], description="Ways the result fell short of the goal"
    )
    explanation: str = Field(..., description="Overall explanation of the rating")
    matches_intent: bool = Field(
        ...,
        description="Whether the result fundamentally addresses the user's core intent",
    )


class TaskSuccessCriteria(BaseModel):
    """Model for task-specific success criteria."""

    required_tools: Set[str] = Field(
        default_factory=set,
        description="Set of tools that must be used for successful completion",
    )
    min_completion_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum completion score required for success",
    )
    required_answer_format: Optional[str] = Field(
        default=None,
        description="Expected format of the final answer (e.g., 'json', 'list', 'number')",
    )
    required_answer_content: Optional[List[str]] = Field(
        default=None, description="Required content elements in the final answer"
    )
    max_iterations: Optional[int] = Field(
        default=None, description="Maximum number of iterations allowed"
    )
    time_limit: Optional[float] = Field(
        default=None, description="Maximum time allowed for task completion in seconds"
    )
    success_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Overall success threshold combining all criteria",
    )
