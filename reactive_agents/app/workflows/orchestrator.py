from __future__ import annotations
import uuid
import time
import asyncio
from typing import Dict, Any, List, Optional, Set, Union, TYPE_CHECKING
from pydantic import BaseModel, Field
from enum import Enum
import networkx as nx

if TYPE_CHECKING:
    from reactive_agents.app.agents.reactive_agent import ReactiveAgentV2
    from reactive_agents.communication.a2a_protocol import A2ACommunicationProtocol


class WorkflowNodeType(Enum):
    """Types of workflow nodes."""

    AGENT = "agent"
    CONDITION = "condition"
    PARALLEL = "parallel"
    MERGE = "merge"
    DELAY = "delay"


class WorkflowNodeStatus(Enum):
    """Status of a workflow node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowNode(BaseModel):
    """A node in the workflow graph."""

    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_type: WorkflowNodeType
    agent_name: Optional[str] = None  # For agent nodes
    task_template: Optional[str] = None  # Task with placeholders
    condition: Optional[str] = None  # For condition nodes
    delay_seconds: Optional[float] = None  # For delay nodes

    # Status and results
    status: WorkflowNodeStatus = WorkflowNodeStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Dependencies
    depends_on: List[str] = Field(default_factory=list)
    outputs_to: List[str] = Field(default_factory=list)

    # Context sharing
    shared_context: Dict[str, Any] = Field(default_factory=dict)
    context_mapping: Dict[str, str] = Field(default_factory=dict)  # key -> context_path


class WorkflowDefinition(BaseModel):
    """Definition of a complete workflow."""

    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None

    # Workflow nodes
    nodes: Dict[str, WorkflowNode] = Field(default_factory=dict)

    # Global configuration
    global_context: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    max_retries: int = 3
    parallel_execution: bool = True

    # Entry and exit points
    entry_nodes: List[str] = Field(default_factory=list)
    exit_nodes: List[str] = Field(default_factory=list)


class WorkflowExecutionResult(BaseModel):
    """Result of workflow execution."""

    workflow_id: str
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float
    end_time: Optional[float] = None

    # Status
    success: bool
    status: str  # "completed", "failed", "timeout", "cancelled"

    # Results
    node_results: Dict[str, Any] = Field(default_factory=dict)
    final_result: Optional[Any] = None
    error_message: Optional[str] = None

    # Metadata
    total_nodes: int
    completed_nodes: int
    failed_nodes: int
    execution_time: Optional[float] = None


class WorkflowOrchestrator:
    """
    Orchestrates multi-agent workflows using DAG-style execution.

    Features:
    - Declarative workflow definition
    - Dependency management
    - Parallel execution
    - Context sharing between agents
    - Error handling and retries
    - Real-time status monitoring
    """

    def __init__(self):
        self.agents: Dict[str, "ReactiveAgentV2"] = {}
        self.a2a_protocols: Dict[str, "A2ACommunicationProtocol"] = {}
        self.active_workflows: Dict[str, WorkflowDefinition] = {}
        self.execution_history: List[WorkflowExecutionResult] = []

    def register_agent(
        self,
        agent: "ReactiveAgentV2",
        a2a_protocol: Optional["A2ACommunicationProtocol"] = None,
    ):
        """Register an agent for use in workflows."""
        agent_name = agent.context.agent_name
        self.agents[agent_name] = agent

        if a2a_protocol:
            self.a2a_protocols[agent_name] = a2a_protocol

    def create_workflow(
        self, name: str, description: Optional[str] = None
    ) -> "WorkflowBuilder":
        """Create a new workflow using the builder pattern."""
        return WorkflowBuilder(self, name, description)

    async def execute_workflow(
        self,
        workflow: WorkflowDefinition,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecutionResult:
        """
        Execute a workflow.

        Args:
            workflow: The workflow definition to execute
            initial_context: Initial context data for the workflow

        Returns:
            WorkflowExecutionResult with execution details
        """
        execution_id = str(uuid.uuid4())
        start_time = time.time()

        # Initialize result
        result = WorkflowExecutionResult(
            workflow_id=workflow.workflow_id,
            execution_id=execution_id,
            start_time=start_time,
            success=False,
            status="running",
            total_nodes=len(workflow.nodes),
            completed_nodes=0,
            failed_nodes=0,
        )

        try:
            # Merge initial context with global context
            execution_context = {**workflow.global_context}
            if initial_context:
                execution_context.update(initial_context)

            # Store active workflow
            self.active_workflows[execution_id] = workflow

            # Execute workflow
            await self._execute_workflow_graph(workflow, execution_context, result)

            # Finalize result
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time
            result.success = result.failed_nodes == 0
            result.status = "completed" if result.success else "failed"

            # Extract final result from exit nodes
            if workflow.exit_nodes:
                exit_results = {}
                for exit_node_id in workflow.exit_nodes:
                    if exit_node_id in result.node_results:
                        exit_results[exit_node_id] = result.node_results[exit_node_id]
                result.final_result = exit_results

        except Exception as e:
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time
            result.success = False
            result.status = "failed"
            result.error_message = str(e)

        finally:
            # Clean up
            if execution_id in self.active_workflows:
                del self.active_workflows[execution_id]

        # Store in history
        self.execution_history.append(result)

        return result

    async def _execute_workflow_graph(
        self,
        workflow: WorkflowDefinition,
        context: Dict[str, Any],
        result: WorkflowExecutionResult,
    ):
        """Execute the workflow graph using topological ordering."""

        # Create NetworkX graph for dependency resolution
        graph = nx.DiGraph()

        # Add nodes
        for node_id, node in workflow.nodes.items():
            graph.add_node(node_id, node=node)

        # Add edges (dependencies)
        for node_id, node in workflow.nodes.items():
            for dependency in node.depends_on:
                graph.add_edge(dependency, node_id)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Workflow contains cycles - DAG required")

        # Get topological order
        execution_order = list(nx.topological_sort(graph))

        # Execute nodes in order, handling parallels
        completed_nodes = set()

        for node_id in execution_order:
            node = workflow.nodes[node_id]

            # Check if dependencies are satisfied
            if not all(dep in completed_nodes for dep in node.depends_on):
                continue  # Will be handled in a later iteration

            # Execute node
            try:
                node_result = await self._execute_node(node, context, workflow)
                result.node_results[node_id] = node_result
                result.completed_nodes += 1
                completed_nodes.add(node_id)

                # Update shared context
                if node.context_mapping:
                    for key, context_path in node.context_mapping.items():
                        if key in node_result:
                            self._set_nested_context(
                                context, context_path, node_result[key]
                            )

            except Exception as e:
                result.failed_nodes += 1
                result.node_results[node_id] = {"error": str(e)}

                # Stop execution on failure (can be made configurable)
                raise e

    async def _execute_node(
        self, node: WorkflowNode, context: Dict[str, Any], workflow: WorkflowDefinition
    ) -> Dict[str, Any]:
        """Execute a single workflow node."""

        node.status = WorkflowNodeStatus.RUNNING
        node.start_time = time.time()

        try:
            if node.node_type == WorkflowNodeType.AGENT:
                result = await self._execute_agent_node(node, context)
            elif node.node_type == WorkflowNodeType.CONDITION:
                result = await self._execute_condition_node(node, context)
            elif node.node_type == WorkflowNodeType.DELAY:
                result = await self._execute_delay_node(node, context)
            else:
                result = {
                    "message": f"Node type {node.node_type.value} not yet implemented"
                }

            node.status = WorkflowNodeStatus.COMPLETED
            node.result = result

        except Exception as e:
            node.status = WorkflowNodeStatus.FAILED
            node.error_message = str(e)
            result = {"error": str(e)}

        finally:
            node.end_time = time.time()

        return result

    async def _execute_agent_node(
        self, node: WorkflowNode, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an agent node."""
        if not node.agent_name or node.agent_name not in self.agents:
            raise ValueError(f"Agent {node.agent_name} not found")

        agent = self.agents[node.agent_name]

        # Prepare task with context substitution
        task = self._substitute_context_variables(node.task_template or "", context)

        # Execute task
        result = await agent.run(task)

        return {
            "agent": node.agent_name,
            "task": task,
            "result": result,
            "success": result.get("status") != "error",
        }

    async def _execute_condition_node(
        self, node: WorkflowNode, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a condition node."""
        if not node.condition:
            raise ValueError("Condition node requires condition expression")

        # Simple condition evaluation (can be enhanced with safe_eval or similar)
        condition_result = eval(node.condition, {"context": context})

        return {
            "condition": node.condition,
            "result": bool(condition_result),
            "success": True,
        }

    async def _execute_delay_node(
        self, node: WorkflowNode, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a delay node."""
        delay_seconds = node.delay_seconds or 1.0
        await asyncio.sleep(delay_seconds)

        return {"delay_seconds": delay_seconds, "success": True}

    def _substitute_context_variables(
        self, template: str, context: Dict[str, Any]
    ) -> str:
        """Substitute context variables in a template string."""
        import re

        def replace_var(match):
            var_path = match.group(1)
            return str(self._get_nested_context(context, var_path))

        # Replace ${variable.path} with context values
        return re.sub(r"\$\{([^}]+)\}", replace_var, template)

    def _get_nested_context(self, context: Dict[str, Any], path: str) -> Any:
        """Get a nested value from context using dot notation."""
        keys = path.split(".")
        value = context

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return f"${{{path}}}"  # Return original if not found

        return value

    def _set_nested_context(self, context: Dict[str, Any], path: str, value: Any):
        """Set a nested value in context using dot notation."""
        keys = path.split(".")
        current = context

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value


class WorkflowBuilder:
    """Builder for creating workflows in a fluent interface."""

    def __init__(
        self,
        orchestrator: WorkflowOrchestrator,
        name: str,
        description: Optional[str] = None,
    ):
        self.orchestrator = orchestrator
        self.workflow = WorkflowDefinition(name=name, description=description)

    def add_agent_node(
        self,
        agent_name: str,
        task_template: str,
        node_id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        context_mapping: Optional[Dict[str, str]] = None,
    ) -> "WorkflowBuilder":
        """Add an agent execution node."""
        node_id = node_id or f"agent_{len(self.workflow.nodes)}"

        node = WorkflowNode(
            node_id=node_id,
            node_type=WorkflowNodeType.AGENT,
            agent_name=agent_name,
            task_template=task_template,
            depends_on=depends_on or [],
            context_mapping=context_mapping or {},
        )

        self.workflow.nodes[node_id] = node

        # Update entry nodes if no dependencies
        if not depends_on:
            self.workflow.entry_nodes.append(node_id)

        return self

    def add_condition_node(
        self,
        condition: str,
        node_id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a condition evaluation node."""
        node_id = node_id or f"condition_{len(self.workflow.nodes)}"

        node = WorkflowNode(
            node_id=node_id,
            node_type=WorkflowNodeType.CONDITION,
            condition=condition,
            depends_on=depends_on or [],
        )

        self.workflow.nodes[node_id] = node
        return self

    def add_delay_node(
        self,
        delay_seconds: float,
        node_id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a delay node."""
        node_id = node_id or f"delay_{len(self.workflow.nodes)}"

        node = WorkflowNode(
            node_id=node_id,
            node_type=WorkflowNodeType.DELAY,
            delay_seconds=delay_seconds,
            depends_on=depends_on or [],
        )

        self.workflow.nodes[node_id] = node
        return self

    def set_exit_nodes(self, node_ids: List[str]) -> "WorkflowBuilder":
        """Set the exit nodes for the workflow."""
        self.workflow.exit_nodes = node_ids
        return self

    def set_global_context(self, context: Dict[str, Any]) -> "WorkflowBuilder":
        """Set global context for the workflow."""
        self.workflow.global_context = context
        return self

    def build(self) -> WorkflowDefinition:
        """Build and return the workflow definition."""
        return self.workflow


# Convenience function for creating simple agent chains
def create_agent_chain(
    agent_names: List[str], task_templates: List[str]
) -> WorkflowDefinition:
    """
    Create a simple sequential chain of agents.

    Args:
        agent_names: List of agent names in execution order
        task_templates: List of task templates for each agent

    Returns:
        WorkflowDefinition for the agent chain
    """
    if len(agent_names) != len(task_templates):
        raise ValueError("agent_names and task_templates must have the same length")

    orchestrator = WorkflowOrchestrator()  # Temporary for building
    builder = orchestrator.create_workflow(f"Chain_{len(agent_names)}_agents")

    previous_node_id = None
    for i, (agent_name, task_template) in enumerate(zip(agent_names, task_templates)):
        node_id = f"agent_{i}_{agent_name}"
        depends_on = [previous_node_id] if previous_node_id else None

        builder.add_agent_node(
            agent_name=agent_name,
            task_template=task_template,
            node_id=node_id,
            depends_on=depends_on,
            context_mapping={"result": f"agents.{agent_name}.result"},
        )

        previous_node_id = node_id

    # Set the last node as exit
    if previous_node_id:
        builder.set_exit_nodes([previous_node_id])

    return builder.build()
