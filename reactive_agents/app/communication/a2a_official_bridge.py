from __future__ import annotations
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable, TYPE_CHECKING
from pydantic import BaseModel, Field
from enum import Enum

if TYPE_CHECKING:
    from reactive_agents.app.agents.reactive_agent import ReactiveAgentV2

# Note: This is a bridge implementation that should be updated to use the official a2a-sdk
# when integrating with the Google A2A ecosystem


class A2ATaskStatus(Enum):
    """Status of A2A tasks following official A2A patterns."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class A2AAtomicTask(BaseModel):
    """
    Atomic task following official A2A design principles.

    Per A2A documentation: Tasks should be atomic and processed by a single
    selected agent from start to finish.
    """

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str

    # Task metadata
    priority: int = Field(default=1, ge=1, le=10)  # 1=low, 10=urgent
    estimated_duration: Optional[float] = None  # seconds
    required_capabilities: List[str] = Field(default_factory=list)

    # Execution context
    input_data: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)

    # Status tracking
    status: A2ATaskStatus = A2ATaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    # Timing
    created_at: float = Field(default_factory=lambda: asyncio.get_event_loop().time())
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class A2AAgentCapability(BaseModel):
    """Agent capability definition for A2A protocol."""

    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    cost_estimate: Optional[float] = None  # Relative cost 0.0-1.0


@runtime_checkable
class A2ACompatibleAgent(Protocol):
    """Protocol for A2A compatible agents."""

    agent_id: str
    capabilities: List[A2AAgentCapability]

    async def can_handle_task(self, task: A2AAtomicTask) -> bool:
        """Check if agent can handle the given atomic task."""
        ...

    async def execute_atomic_task(self, task: A2AAtomicTask) -> Dict[str, Any]:
        """Execute an atomic task and return results."""
        ...

    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        ...


class A2AOfficialBridge:
    """
    Bridge to integrate with official Google A2A protocol.

    This class provides compatibility with the official A2A SDK and follows
    the established patterns from the Google A2A project.

    TODO: Replace with official a2a-sdk when available in environment
    """

    def __init__(self):
        self.registered_agents: Dict[str, A2ACompatibleAgent] = {}
        self.active_tasks: Dict[str, A2AAtomicTask] = {}

        # Task queue following A2A patterns
        self.task_queue: asyncio.Queue[A2AAtomicTask] = asyncio.Queue()
        self.completed_tasks: List[A2AAtomicTask] = []

        # Official A2A endpoints (to be implemented with official SDK)
        self.a2a_endpoints = {
            "task_delegation": "/a2a/delegate",
            "task_status": "/a2a/status",
            "agent_discovery": "/a2a/discover",
            "health_check": "/a2a/health",
        }

    def register_agent(self, agent: "ReactiveAgentV2") -> A2ACompatibleAgent:
        """
        Register a ReactiveAgentV2 as an A2A compatible agent.

        This creates an adapter that makes our agents compatible with
        the official A2A protocol.
        """
        adapter = ReactiveAgentA2AAdapter(agent)
        self.registered_agents[adapter.agent_id] = adapter
        return adapter

    async def delegate_atomic_task(
        self,
        task_description: str,
        target_agent_id: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> A2AAtomicTask:
        """
        Delegate an atomic task following official A2A patterns.

        Args:
            task_description: Clear, atomic task description
            target_agent_id: Specific agent to assign (optional)
            required_capabilities: Required agent capabilities
            input_data: Task input data

        Returns:
            A2AAtomicTask object
        """

        # Create atomic task
        task = A2AAtomicTask(
            description=task_description,
            required_capabilities=required_capabilities or [],
            input_data=input_data or {},
        )

        # Find suitable agent if not specified
        if target_agent_id is None:
            target_agent_id = await self._find_best_agent(task)

        if target_agent_id and target_agent_id in self.registered_agents:
            task.assigned_agent_id = target_agent_id
            task.status = A2ATaskStatus.IN_PROGRESS
            task.started_at = asyncio.get_event_loop().time()

            # Store active task
            self.active_tasks[task.task_id] = task

            # Execute task
            try:
                agent = self.registered_agents[target_agent_id]
                result = await agent.execute_atomic_task(task)

                task.result = result
                task.status = A2ATaskStatus.COMPLETED
                task.completed_at = asyncio.get_event_loop().time()

            except Exception as e:
                task.error_message = str(e)
                task.status = A2ATaskStatus.FAILED
                task.completed_at = asyncio.get_event_loop().time()

            # Move to completed
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.completed_tasks.append(task)

        else:
            task.status = A2ATaskStatus.FAILED
            task.error_message = f"No suitable agent found for task: {task_description}"

        return task

    async def _find_best_agent(self, task: A2AAtomicTask) -> Optional[str]:
        """Find the best agent for a task based on capabilities."""
        best_agent_id = None
        best_score = 0.0

        for agent_id, agent in self.registered_agents.items():
            if await agent.can_handle_task(task):
                # Simple scoring based on capability match
                score = len(
                    set(task.required_capabilities)
                    & set(cap.name for cap in agent.capabilities)
                )
                if score > best_score:
                    best_score = score
                    best_agent_id = agent_id

        return best_agent_id

    async def get_task_status(self, task_id: str) -> Optional[A2AAtomicTask]:
        """Get status of a task."""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]

        for task in self.completed_tasks:
            if task.task_id == task_id:
                return task

        return None

    def get_agent_discovery_info(self) -> Dict[str, Any]:
        """Get agent discovery information for A2A protocol."""
        agents_info = {}

        for agent_id, agent in self.registered_agents.items():
            agents_info[agent_id] = {
                "capabilities": [cap.model_dump() for cap in agent.capabilities],
                "status": "available",  # Could be enhanced with real status
                "endpoints": self.a2a_endpoints,
            }

        return {
            "agents": agents_info,
            "protocol_version": "a2a-v1",  # Should match official version
            "supported_features": [
                "atomic_task_delegation",
                "capability_discovery",
                "status_monitoring",
            ],
        }


class ReactiveAgentA2AAdapter(A2ACompatibleAgent):
    """
    Adapter to make ReactiveAgentV2 compatible with official A2A protocol.
    """

    def __init__(self, agent: "ReactiveAgentV2"):
        self.reactive_agent = agent
        self.agent_id = agent.context.agent_name

        # Map ReactiveAgent capabilities to A2A capabilities
        self.capabilities = self._map_capabilities()

    def _map_capabilities(self) -> List[A2AAgentCapability]:
        """Map ReactiveAgent features to A2A capabilities."""
        capabilities = []

        # Base reasoning capability
        capabilities.append(
            A2AAgentCapability(
                name="reasoning",
                description=f"Dynamic reasoning with strategies: {self.reactive_agent.get_available_strategies()}",
                cost_estimate=0.5,
            )
        )

        # Tool capabilities
        if (
            self.reactive_agent.context.tool_use_enabled
            and self.reactive_agent.context.tools
        ):
            for tool in self.reactive_agent.context.tools:
                capabilities.append(
                    A2AAgentCapability(
                        name=f"tool_{tool}",
                        description=f"Can use tool: {tool}",
                        cost_estimate=0.3,
                    )
                )

        # Strategy-specific capabilities
        current_strategy = self.reactive_agent.get_current_strategy()
        if current_strategy == "plan_execute_reflect":
            capabilities.append(
                A2AAgentCapability(
                    name="complex_planning",
                    description="Can handle complex multi-step tasks with planning",
                    cost_estimate=0.8,
                )
            )
        elif current_strategy == "reactive":
            capabilities.append(
                A2AAgentCapability(
                    name="quick_response",
                    description="Fast responses for simple tasks",
                    cost_estimate=0.2,
                )
            )

        return capabilities

    async def can_handle_task(self, task: A2AAtomicTask) -> bool:
        """Check if this agent can handle the atomic task."""
        # Check if required capabilities are available
        agent_cap_names = {cap.name for cap in self.capabilities}

        for required_cap in task.required_capabilities:
            if required_cap not in agent_cap_names:
                return False

        # Additional checks could include:
        # - Task complexity vs agent capability
        # - Current agent load
        # - Resource availability

        return True

    async def execute_atomic_task(self, task: A2AAtomicTask) -> Dict[str, Any]:
        """Execute an atomic task using the ReactiveAgent."""

        # Prepare task for ReactiveAgent
        agent_task = task.description

        # Add input data as context if available
        if task.input_data:
            context_str = "\n".join([f"{k}: {v}" for k, v in task.input_data.items()])
            agent_task = f"{agent_task}\n\nContext:\n{context_str}"

        # Execute using ReactiveAgent
        result = await self.reactive_agent.run(agent_task)

        # Format result for A2A protocol
        return {
            "task_id": task.task_id,
            "agent_id": self.agent_id,
            "execution_result": result,
            "strategy_used": self.reactive_agent.get_current_strategy(),
            "reasoning_context": self.reactive_agent.get_reasoning_context(),
            "success": result.get("status") != "error",
        }

    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "status": "available",  # Could be enhanced with real status
            "current_strategy": self.reactive_agent.get_current_strategy(),
            "capabilities": [cap.name for cap in self.capabilities],
            "reasoning_context": self.reactive_agent.get_reasoning_context(),
        }


# Helper functions for easy integration
async def create_a2a_compatible_agent_network(
    agents: List["ReactiveAgentV2"],
) -> A2AOfficialBridge:
    """
    Create an A2A compatible network from ReactiveAgentV2 instances.

    This function creates a bridge that makes the reactive agents
    compatible with the official A2A protocol.
    """
    bridge = A2AOfficialBridge()

    for agent in agents:
        bridge.register_agent(agent)

    return bridge


# Example usage pattern following official A2A samples
async def demonstrate_official_a2a_pattern():
    """
    Demonstrate usage pattern following official A2A samples.

    This follows the patterns seen in the GitHub documentation for
    multi-agent orchestration.
    """

    # This would typically use the official a2a-sdk
    # For now, we use our bridge implementation

    print("🔗 Creating A2A compatible agent network...")

    # In a real implementation, this would integrate with:
    # - Official a2a-sdk
    # - Standard A2A endpoints
    # - Official authentication mechanisms

    return {
        "message": "A2A bridge created - ready for official SDK integration",
        "next_steps": [
            "Install official a2a-sdk",
            "Replace bridge with official SDK calls",
            "Implement standard A2A endpoints",
            "Add proper authentication",
        ],
    }
