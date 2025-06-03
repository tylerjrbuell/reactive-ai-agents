from __future__ import annotations
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from contextlib import AsyncExitStack

if TYPE_CHECKING:
    from agents.react_agent import ReactAgent
    from agent_mcp.client import MCPClient


@dataclass
class AgentConfig:
    role: str
    model: str
    min_score: float
    instructions: str
    dependencies: List[str] = field(default_factory=list)
    max_iterations: int = 5
    mcp_servers: Optional[List[str]] = None
    tools: Optional[List[str]] = None
    instructions_as_task: bool = False
    reflect: bool = False
    log_level: str = "info"


@dataclass
class WorkflowConfig:
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    _results: Dict[str, str] = field(default_factory=dict)

    def add_agent(self, agent_config: AgentConfig) -> None:
        """Add an agent configuration to the workflow"""
        self.agents[agent_config.role] = agent_config

    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort on the dependency graph"""
        visited = set()
        temp_mark = set()
        order = []

        def visit(node):
            if node in temp_mark:
                raise ValueError(f"Circular dependency detected involving {node}")
            if node not in visited:
                temp_mark.add(node)
                # Only visit dependencies that exist in the graph
                for dep in graph.get(node, []):
                    if dep not in graph:
                        raise ValueError(
                            f"Dependency '{dep}' referenced by '{node}' does not exist in the workflow"
                        )
                    visit(dep)
                temp_mark.remove(node)
                visited.add(node)
                order.append(node)

        for node in graph:
            if node not in visited:
                visit(node)

        # Return the order as is - nodes without dependencies will be first
        return order


class Workflow:
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self._exit_stack = AsyncExitStack()
        self._active_agents: Dict[str, ReactAgent] = {}
        self._mcp_client: Optional[MCPClient] = None
        self._workflow_context: Dict[str, Any] = {}

    async def __aenter__(self):
        """Setup the workflow context"""
        await self._exit_stack.__aenter__()

        return self

    async def _cleanup_mcp_client(self):
        """Cleanup MCP client resources"""
        if self._mcp_client:
            await self._mcp_client.__aexit__(None, None, None)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup workflow resources"""
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def _create_agent(
        self, agent_config: AgentConfig, dependencies: List[str]
    ) -> ReactAgent:
        """Create an agent instance from configuration"""
        from agents.react_agent import ReactAgent  # Import at runtime
        from agent_mcp.client import MCPClient  # Import at runtime

        # Initialize agent's context in the workflow context if it doesn't exist
        if agent_config.role not in self._workflow_context:
            self._workflow_context[agent_config.role] = {
                "status": "pending",
                "dependencies_met": False,
                "reflections": [],  # Initialize empty reflections list
                "current_progress": "",
                "iterations": 0,
            }

        # Only create MCP client if servers are configured
        filtered_mcp_client = None
        if agent_config.mcp_servers:
            # Create an MCP client with filtered servers if needed otherwise use default
            filtered_mcp_client = (
                await MCPClient(server_filter=agent_config.mcp_servers).__aenter__()
                if agent_config.mcp_servers
                else await MCPClient().__aenter__()
            )
            # Add filtered client to exit stack for cleanup
            self._exit_stack.push_async_callback(
                filtered_mcp_client.__aexit__, None, None, None
            )

        return ReactAgent(
            name=agent_config.role,
            provider_model=agent_config.model,
            mcp_client=filtered_mcp_client,  # Will be None if no servers configured
            instructions=agent_config.instructions,
            role=agent_config.role,
            min_completion_score=agent_config.min_score,
            max_iterations=agent_config.max_iterations,
            workflow_context=self._workflow_context,
            workflow_dependencies=dependencies,
            reflect=agent_config.reflect,
            log_level=agent_config.log_level,
        )

    async def _execute_workflow(self, task: str) -> Dict[str, str]:
        """Execute the workflow with the given task"""
        results = {}

        try:
            # Pre-initialize workflow context for all agents
            for role, agent_config in self.config.agents.items():
                if role not in self._workflow_context:
                    self._workflow_context[role] = {
                        "status": "pending",
                        "dependencies_met": False,
                        "reflections": [],
                        "current_progress": "",
                        "iterations": 0,
                    }

            # Find execution order based on dependencies
            execution_order = self.config._topological_sort(
                {
                    role: config.dependencies
                    for role, config in self.config.agents.items()
                }
            )

            # Execute agents in order
            for role in execution_order:
                agent_config = self.config.agents[role]

                # Create agent with its dependencies
                agent = await self._create_agent(
                    agent_config, agent_config.dependencies
                )
                self._active_agents[role] = agent

                # Update workflow context with status
                self._workflow_context[role].update(
                    {
                        "status": "running",
                        "dependencies_met": True,  # Dependencies are met since we're running in order
                    }
                )

                try:
                    # Set task to instructions if instructions_as_task is True
                    if agent_config.instructions_as_task:
                        try:
                            formatted_instructions = agent_config.instructions.format(
                                task=task
                            )
                        except KeyError:
                            # If {task} placeholder isn't in instructions, use instructions as-is
                            formatted_instructions = agent_config.instructions

                        result = await agent.run(formatted_instructions)
                    else:
                        # Run agent with task and automatically picks up context from previous agents
                        result = await agent.run(task)

                    if result:
                        results[role] = result
                        # Update workflow context with completion status
                        self._workflow_context[role].update(
                            {"status": "complete", "final_result": result}
                        )
                    else:
                        self._workflow_context[role].update({"status": "failed"})
                except Exception as e:
                    import traceback

                    print(
                        f"Error executing agent {role}: {str(e)}\n{traceback.format_exc()}"
                    )
                    self._workflow_context[role].update(
                        {
                            "status": "error",
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )

            return results

        except Exception as e:
            import traceback

            print(f"Workflow execution error: {str(e)}\n{traceback.format_exc()}")
            return {}

    async def run(self, task: str) -> Dict[str, str]:
        """Run the workflow with proper resource management"""
        async with self:
            return await self._execute_workflow(task)

    async def cleanup(self):
        """Explicit cleanup method"""
        await self.__aexit__(None, None, None)
