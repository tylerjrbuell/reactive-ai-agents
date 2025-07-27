import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

import asyncio

from reactive_agents.core.reasoning.strategy_components import (
    ThinkingComponent,
    PlanningComponent,
    ToolExecutionComponent,
    ReflectionComponent,
    TaskEvaluationComponent,
    CompletionComponent,
    ErrorHandlingComponent,
    MemoryIntegrationComponent,
    StrategyTransitionComponent,
)
from reactive_agents.core.types.reasoning_component_types import (
    Plan,
    ReflectionResult,
    ToolExecutionResult,
    CompletionResult,
    StrategyTransitionResult,
)


class DummyEngine:
    def __init__(self):
        self.context = MagicMock()
        self.context.agent_logger = MagicMock()
        self.context.agent_logger.debug = MagicMock()
        self.context.agent_logger.warning = MagicMock()
        self.context.agent_logger.info = MagicMock()
        self.context.memory_manager = MagicMock()
        self.preserve_context = MagicMock()
        self.get_preserved_context = MagicMock(return_value=None)
        self.get_context_manager = MagicMock()
        
        # Mock think methods to return appropriate structures
        self.think = AsyncMock(return_value=MagicMock(result_json={"thought": "test"}))
        
        # Mock think_chain to return the expected structure with tool_calls
        think_chain_result = MagicMock()
        think_chain_result.tool_calls = [{"function": {"name": "test_tool"}}]
        self.think_chain = AsyncMock(return_value=think_chain_result)
        
        self.execute_tools = AsyncMock(return_value=[{"result": "ok"}])
        self.generate_final_answer = AsyncMock(return_value={"final_answer": "done"})
        self.complete_task_if_ready = AsyncMock(return_value={"is_complete": True})
        
        # Mock get_prompt to return a prompt object with async get_completion
        mock_completion_result = MagicMock()
        mock_completion_result.result_json = {
            "plan_steps": [], 
            "metadata": {},
            "reflection": "test reflection",
            "insights": ["test insight"],
            "confidence": 0.8,
            "is_complete": True,
            "completion_score": 0.9,
            "reasoning": "test reasoning",
            "should_switch": False,
            "recommended_strategy": None,
            "rationale": "test rationale"
        }
        mock_prompt = MagicMock()
        mock_prompt.get_completion = AsyncMock(return_value=mock_completion_result)
        self.get_prompt = MagicMock(return_value=mock_prompt)


@pytest_asyncio.fixture
def infra():
    return DummyEngine()


@pytest.mark.asyncio
async def test_thinking_component(infra):
    comp = ThinkingComponent(infra)
    result = await comp.think("test prompt")
    assert isinstance(result, dict)
    # TODO: Add more behavioral assertions


@pytest.mark.asyncio
async def test_planning_component(infra):
    comp = PlanningComponent(infra)
    result = await comp.generate_plan("test task", MagicMock())
    assert isinstance(result, Plan)
    # TODO: Add more behavioral assertions


@pytest.mark.asyncio
async def test_tool_execution_component(infra):
    comp = ToolExecutionComponent(infra)
    result = await comp.execute_tools([{"function": {"name": "dummy"}}])
    assert isinstance(result, list)
    # TODO: Add more behavioral assertions
    result2 = await comp.select_and_execute_tool("test task", "step desc")
    assert isinstance(result2, ToolExecutionResult)


@pytest.mark.asyncio
async def test_reflection_component(infra):
    comp = ReflectionComponent(infra)
    result = await comp.reflect_on_progress("test task", {"result": "ok"}, MagicMock())
    assert isinstance(result, ReflectionResult)
    result2 = await comp.reflect_on_plan_progress(
        "test task", 0, [{"description": "step1"}], {"result": "ok"}
    )
    assert isinstance(result2, ReflectionResult)


@pytest.mark.asyncio
async def test_task_evaluation_component(infra):
    comp = TaskEvaluationComponent(infra)
    # Patch context with real values for prompt context
    comp.context.role = "test_role"
    comp.context.instructions = "test_instructions"
    comp.context.provider_model_name = "test_model"
    comp.context.get_tool_signatures = MagicMock(return_value=[])
    # Patch session with required fields
    session_mock = MagicMock()
    session_mock.current_task = "test task"
    session_mock.messages = []
    session_mock.iterations = 0
    comp.context.session = session_mock
    # Patch memory_manager as a MagicMock with get_memory_stats
    memory_manager_mock = MagicMock()
    memory_manager_mock.get_memory_stats = MagicMock(return_value={})
    comp.context.memory_manager = memory_manager_mock
    # Patch model_provider as an AsyncMock with get_completion
    comp.context.model_provider = AsyncMock()
    comp.context.model_provider.get_completion = AsyncMock(
        return_value={
            "completion": True,
            "completion_score": 1.0,
            "reasoning": "Test reasoning",
            "missing_requirements": [],
        }
    )
    result = await comp.evaluate_task_completion("test task")
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_completion_component(infra):
    comp = CompletionComponent(infra)
    result = await comp.generate_final_answer("test task")
    assert isinstance(result, dict)
    result2 = await comp.complete_task_if_ready("test task")
    assert isinstance(result2, dict)


@pytest.mark.asyncio
async def test_error_handling_component(infra):
    comp = ErrorHandlingComponent(infra)
    result = await comp.handle_error("test task", "context", 1, "last error")
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_memory_integration_component(infra):
    comp = MemoryIntegrationComponent(infra)
    # Patch memory_manager on the Dummyengine context directly
    memory_manager_mock = MagicMock()
    setattr(memory_manager_mock, "is_ready", MagicMock(return_value=True))
    setattr(
        memory_manager_mock,
        "get_context_memories",
        AsyncMock(return_value=[{"memory": "test"}]),
    )
    comp.context.memory_manager = memory_manager_mock
    result = await comp.get_relevant_memories("test task")
    assert isinstance(result, list)
    comp.preserve_context("key", "value")
    comp.get_preserved_context("key")
    # No assertion needed, just ensure no error


@pytest.mark.asyncio
async def test_strategy_transition_component(infra):
    comp = StrategyTransitionComponent(infra)
    result = await comp.should_switch_strategy(
        "current", ["a", "b"], MagicMock(), {"error_count": 0}
    )
    assert isinstance(result, dict)
