import asyncio
import json
from typing import Dict, Any, Optional, List

from reactive_agents.app.agents.base import Agent
from reactive_agents.core.types.agent_types import ReactAgentConfig
from reactive_agents.core.context.agent_context import AgentContext
from reactive_agents.core.engine.reactive_execution_engine import (
    ReactiveExecutionEngine,
)
from reactive_agents.core.events.event_manager import EventManager
from reactive_agents.core.engine.task_executor import TaskExecutor
from reactive_agents.config.validators.config_validator import ConfigValidator
from reactive_agents.core.tools.tool_processor import ToolProcessor


# Import reasoning types for strategy management
from reactive_agents.core.types.reasoning_types import ReasoningStrategies
from reactive_agents.core.types.event_types import AgentStateEvent


class ReactiveAgent(Agent):
    """
    Next-Generation Reactive Agent with comprehensive capabilities.

    This agent consolidates the best features from ReactAgent and ReactiveAgentV2:

    **Core Features:**
    - Dynamic reasoning strategies (reactive, reflect_decide_act, plan_execute_reflect, adaptive)
    - Comprehensive event system with typed callbacks
    - Advanced tool management with feasibility checking and rescoping
    - Intelligent context management and memory
    - Real-time control operations (pause, resume, stop, terminate)
    - Metrics collection and performance tracking
    - MCP integration for external tool access

    **Key Improvements:**
    - Unified architecture with clear separation of concerns
    - Protocol-based design for extensibility
    - Enhanced error handling and recovery
    - Optimized tool execution with parallel processing
    - Adaptive reasoning based on task classification
    - Comprehensive logging and debugging support
    - Dynamic event system that automatically handles all event types
    """

    execution_engine: ReactiveExecutionEngine
    event_manager: EventManager
    task_executor: TaskExecutor
    tool_processor: ToolProcessor

    def __init__(self, config: ReactAgentConfig):
        """
        Initialize the ReactiveAgent with enhanced reactive capabilities.

        Args:
            config: ReactAgentConfig containing all agent settings
        """
        # Initialize config validator
        self.config_validator = ConfigValidator(str(config.log_level or "info"))

        # Validate configuration
        validated_config = self.config_validator.validate_agent_config(
            agent_name=config.agent_name,
            provider_model_name=config.provider_model_name,
            model_provider_options=config.model_provider_options,
            role=config.role,
            mcp_client=config.mcp_client,
            mcp_config=config.mcp_config,
            mcp_server_filter=config.mcp_server_filter,
            min_completion_score=config.min_completion_score,
            instructions=config.instructions,
            max_iterations=config.max_iterations,
            reflect_enabled=config.reflect_enabled,
            log_level=str(config.log_level or "info"),
            initial_task=config.initial_task,
            tool_use_enabled=config.tool_use_enabled,
            custom_tools=config.custom_tools,
            use_memory_enabled=config.use_memory_enabled,
            vector_memory_enabled=config.vector_memory_enabled,
            vector_memory_collection=config.vector_memory_collection,
            collect_metrics_enabled=config.collect_metrics_enabled,
            check_tool_feasibility=config.check_tool_feasibility,
            enable_caching=config.enable_caching,
            confirmation_callback=config.confirmation_callback,
            confirmation_config=config.confirmation_config,
            workflow_context_shared=config.workflow_context_shared,
            response_format=config.response_format,
        )

        # Enable reactive execution and preserve configured strategy
        validated_config["enable_reactive_execution"] = True

        # Only set default strategy if none was configured
        if (
            "reasoning_strategy" not in validated_config
            or not validated_config["reasoning_strategy"]
        ):
            validated_config["reasoning_strategy"] = "reflect_decide_act"  # Default

        # Preserve dynamic strategy switching setting from config
        if hasattr(config, "enable_dynamic_strategy_switching"):
            validated_config["enable_dynamic_strategy_switching"] = (
                config.enable_dynamic_strategy_switching
            )

        # Create the context with reactive execution enabled
        context = AgentContext(**validated_config)

        # Initialize the base Agent class
        super().__init__(context)

        # Store config for reference
        self.config = config

        # Set agent reference in context for strategy use
        self.context._agent = self  # type: ignore

        # Initialize components
        self.tool_processor = ToolProcessor(self)

        # Process custom tools
        processed_tools = self.tool_processor.process_custom_tools(
            validated_config["custom_tools"]
        )
        self.context.tools = processed_tools

        # Initialize the reactive execution engine
        self.execution_engine = ReactiveExecutionEngine(self)

        # Initialize event manager and task executor
        self.event_manager = EventManager(self)
        self.task_executor = TaskExecutor(self)

    async def _execute_task(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a task using the reactive execution engine.

        This is the core execution method that implements the abstract method
        from the base Agent class.

        Args:
            task: The task to execute
            **kwargs: Additional execution parameters

        Returns:
            Execution results with metadata
        """
        cancellation_event = kwargs.get("cancellation_event")

        try:
            # Use the reactive execution engine for enhanced execution
            result = await self.execution_engine.execute(
                initial_task=task, cancellation_event=cancellation_event
            )

            if self.agent_logger:
                self.agent_logger.info(
                    f"✅ Task completed with status: {result.get('status')} "
                    f"using strategy: {result.get('reasoning_strategy')}"
                )

            return result

        except Exception as e:
            if self.agent_logger:
                self.agent_logger.error(f"ReactiveAgent execution failed: {e}")

            return {
                "status": "error",
                "error": str(e),
                "final_answer": None,
                "completion_score": 0.0,
                "iterations": 0,
                "reasoning_strategy": "unknown",
            }

    # --- Strategy Management Methods ---
    async def run_with_strategy(
        self,
        initial_task: str,
        strategy: str = "reflect_decide_act",
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent with a specific reasoning strategy.

        Args:
            initial_task: The task to execute
            strategy: Specific reasoning strategy to use
            cancellation_event: Optional cancellation event

        Returns:
            Execution results
        """
        strategy_map = {
            "reactive": ReasoningStrategies.REACTIVE,
            "reflect_decide_act": ReasoningStrategies.REFLECT_DECIDE_ACT,
            "plan_execute_reflect": ReasoningStrategies.PLAN_EXECUTE_REFLECT,
            "adaptive": ReasoningStrategies.ADAPTIVE,
        }

        if strategy in strategy_map:
            self.execution_engine.strategy_manager.current_strategy = strategy_map[
                strategy
            ]
            self.execution_engine.strategy_manager.reasoning_context.current_strategy = strategy_map[
                strategy
            ]

            if self.agent_logger:
                self.agent_logger.info(f"🎯 Using forced strategy: {strategy}")

        return await self.run(initial_task, cancellation_event=cancellation_event)

    async def initialize(self) -> "ReactiveAgent":
        """
        Initialize the ReactiveAgent and wait for vector memory to be ready.

        This ensures that vector memory is fully initialized before any tasks
        are executed, preventing race conditions where memories might be lost.
        """
        if self._initialized:
            self.agent_logger.warning("ReactiveAgent already initialized")
            return self

        try:
            self.agent_logger.info(
                f"Initializing ReactiveAgent '{self.context.agent_name}'..."
            )

            # Call parent initialization first
            await super().initialize()

            # Wait for vector memory to be ready if enabled
            if self.context.vector_memory_enabled and self.context.memory_manager:

                # Check if this is a VectorMemoryManager that supports await_ready
                from reactive_agents.core.memory.vector_memory import (
                    VectorMemoryManager,
                )

                if isinstance(self.context.memory_manager, VectorMemoryManager):
                    self.agent_logger.info(
                        "Waiting for vector memory initialization..."
                    )
                    memory_ready = await self.context.memory_manager.await_ready(
                        timeout=30.0
                    )

                    if memory_ready:
                        self.agent_logger.info(
                            "Vector memory ready - all memories will be stored in vector database"
                        )
                    else:
                        self.agent_logger.warning(
                            "Vector memory not ready - falling back to JSON memory storage"
                        )
                else:
                    self.agent_logger.debug(
                        "Memory manager is not VectorMemoryManager, skipping await_ready"
                    )

            self._initialized = True
            self.agent_logger.info(
                f"ReactiveAgent '{self.context.agent_name}' initialized successfully"
            )
            return self

        except Exception as e:
            self.agent_logger.error(
                f"Error initializing ReactiveAgent '{self.context.agent_name}': {e}"
            )
            raise e

    def get_available_strategies(self) -> List[str]:
        """Get list of available reasoning strategies."""
        return ["reactive", "reflect_decide_act", "plan_execute_reflect", "adaptive"]

    def get_current_strategy(self) -> str:
        """Get the currently active reasoning strategy."""
        if hasattr(self.execution_engine, "strategy_manager"):
            current = self.execution_engine.strategy_manager.current_strategy
            return current.value if current else "unknown"
        return "unknown"

    def get_reasoning_context(self) -> Dict[str, Any]:
        """Get the current reasoning context information."""
        if hasattr(self.execution_engine, "strategy_manager"):
            context = self.execution_engine.strategy_manager.get_reasoning_context()
            return {
                "current_strategy": (
                    context.current_strategy.value
                    if context.current_strategy
                    else "unknown"
                ),
                "iteration_count": context.iteration_count,
                "error_count": context.error_count,
                "stagnation_count": context.stagnation_count,
                "tool_usage_history": context.tool_usage_history,
                "task_classification": context.task_classification,
            }
        return {}

    # --- Advanced Tool Management Methods (from ReactAgent) ---
    async def check_tool_feasibility(self, task: str) -> Dict[str, Any]:
        """
        Check if the task requires tools and which tools are feasible.

        This method analyzes the task to determine:
        - Whether tools are needed
        - Which tools are most appropriate
        - Tool feasibility scores

        Args:
            task: The task to analyze

        Returns:
            Analysis results with tool recommendations
        """
        if not self.context.check_tool_feasibility:
            return {"feasible": True, "tools_needed": False, "recommendations": []}

        try:
            # Get available tools
            available_tools = self.context.get_tool_names()
            if not available_tools:
                return {"feasible": True, "tools_needed": False, "recommendations": []}

            # Create analysis prompt
            analysis_prompt = f"""
            Task: {task}
            
            Available tools: {', '.join(available_tools)}
            
            Analyze whether this task requires tools and which tools are most appropriate.
            Consider:
            1. Does the task require external data or actions?
            2. Which tools would be most useful?
            3. What is the feasibility score (0-1) for using tools?
            
            Provide your analysis in JSON format:
            {{
                "tools_needed": boolean,
                "feasible": boolean,
                "feasibility_score": float,
                "recommended_tools": [list of tool names],
                "reasoning": "explanation"
            }}
            """

            # Get analysis from model
            result = await self._think(
                messages=[{"role": "user", "content": analysis_prompt}]
            )

            if result and result.get("content"):
                try:
                    analysis = json.loads(result["content"])
                    return analysis
                except json.JSONDecodeError:
                    self.agent_logger.warning(
                        "Failed to parse tool feasibility analysis"
                    )

            return {"feasible": True, "tools_needed": False, "recommendations": []}

        except Exception as e:
            self.agent_logger.error(f"Tool feasibility check failed: {e}")
            return {"feasible": True, "tools_needed": False, "recommendations": []}

    async def rescope_goal(
        self, original_task: str, error_context: str
    ) -> Dict[str, Any]:
        """
        Rescope the original goal based on errors or new information.

        This method helps the agent adapt when the original task cannot be completed
        as specified, allowing it to propose alternative approaches.

        Args:
            original_task: The original task that failed
            error_context: Information about why the task failed

        Returns:
            Rescoped task and reasoning
        """
        try:
            rescope_prompt = f"""
            Original Task: {original_task}
            
            Error Context: {error_context}
            
            Based on the error context, rescope the original task to make it more achievable.
            Consider:
            1. What aspects of the task can still be accomplished?
            2. What alternative approaches could work?
            3. What constraints or limitations should be acknowledged?
            
            Provide your rescoped task in JSON format:
            {{
                "rescoped_task": "new task description",
                "reasoning": "explanation of changes",
                "constraints": ["list of acknowledged limitations"],
                "confidence": float (0-1)
            }}
            """

            result = await self._think(
                messages=[{"role": "user", "content": rescope_prompt}]
            )

            if result and result.get("content"):
                try:
                    rescoped = json.loads(result["content"])
                    return rescoped
                except json.JSONDecodeError:
                    self.agent_logger.warning("Failed to parse rescoped task")

            return {
                "rescoped_task": original_task,
                "reasoning": "Could not rescope task",
                "constraints": [],
                "confidence": 0.0,
            }

        except Exception as e:
            self.agent_logger.error(f"Goal rescoping failed: {e}")
            return {
                "rescoped_task": original_task,
                "reasoning": f"Rescoping failed: {e}",
                "constraints": [],
                "confidence": 0.0,
            }

    # --- Enhanced Control Methods ---
    async def pause(self) -> None:
        """Pause the agent execution with event emission."""
        if self.agent_logger:
            self.agent_logger.info(f"{self.context.agent_name} pause requested")

        # Emit pause requested event
        if self.event_manager:
            self.event_manager.emit_event(AgentStateEvent.PAUSE_REQUESTED, {})

        # Pause execution engine
        if self.execution_engine:
            await self.execution_engine.pause()

    async def resume(self) -> None:
        """Resume the agent execution with event emission."""
        if self.agent_logger:
            self.agent_logger.info(f"{self.context.agent_name} resume requested")

        # Emit resume requested event
        if self.event_manager:
            self.event_manager.emit_event(AgentStateEvent.RESUME_REQUESTED, {})

        # Resume execution engine
        if self.execution_engine:
            await self.execution_engine.resume()

    async def stop(self) -> None:
        """Stop the agent execution with event emission."""
        if self.agent_logger:
            self.agent_logger.info(f"{self.context.agent_name} stop requested")

        # Emit stop requested event
        if self.event_manager:
            self.event_manager.emit_event(AgentStateEvent.STOP_REQUESTED, {})

        # Stop execution engine
        if self.execution_engine:
            await self.execution_engine.stop()

    async def terminate(self) -> None:
        """Terminate the agent execution with event emission."""
        if self.agent_logger:
            self.agent_logger.info(f"{self.context.agent_name} terminate requested")

        # Emit terminate requested event
        if self.event_manager:
            self.event_manager.emit_event(AgentStateEvent.TERMINATE_REQUESTED, {})

        # Terminate execution engine
        if self.execution_engine:
            await self.execution_engine.terminate()

    # --- Dynamic Event System ---
    @property
    def events(self):
        """Access to the event manager for event subscription."""
        return self.event_manager

    def __dir__(self):
        """
        Include dynamic event methods in dir() and auto-completion.

        This ensures that IDEs and tools can discover the dynamically available
        on_<event_name> methods for better IntelliSense support.
        """
        # Get the default attributes
        attrs = set(super().__dir__())

        # Add all the dynamic event methods
        for event in AgentStateEvent:
            attrs.add(f"on_{event.value}")

        return sorted(attrs)

    def __getattr__(self, name: str):
        """
        Dynamic event handler registration with full IntelliSense support.

        This method allows for dynamic event subscription using the pattern:
        agent.on_<event_name>(callback)

        Examples:
            >>> agent.on_session_started(my_callback)
            >>> agent.on_tool_called(lambda event: print(event))
            >>> agent.on_pause_requested(handle_pause)

        Available events:
            - on_session_started: When a new agent session begins
            - on_session_ended: When an agent session completes
            - on_task_status_changed: When task status changes
            - on_iteration_started: When a new iteration begins
            - on_iteration_completed: When an iteration completes
            - on_tool_called: When a tool is invoked
            - on_tool_completed: When a tool completes successfully
            - on_tool_failed: When a tool execution fails
            - on_reflection_generated: When agent generates a reflection
            - on_final_answer_set: When the final answer is determined
            - on_metrics_updated: When performance metrics are updated
            - on_error_occurred: When an error occurs
            - on_pause_requested: When pause is requested
            - on_paused: When agent is paused
            - on_resume_requested: When resume is requested
            - on_resumed: When agent resumes execution
            - on_terminate_requested: When termination is requested
            - on_terminated: When agent is terminated
            - on_stop_requested: When stop is requested
            - on_stopped: When agent is stopped
            - on_cancelled: When operation is cancelled

        Args:
            name: The attribute name being accessed

        Returns:
            A callable function that registers the event callback

        Raises:
            AttributeError: If the attribute doesn't exist or isn't a valid event
        """
        if name.startswith("on_"):
            event_name = name[3:]  # Remove 'on_' prefix

            # Check if this is a valid event type (AgentStateEvent values are in snake_case)
            try:
                event_type = AgentStateEvent(event_name)
            except ValueError:
                # If not a valid event, raise AttributeError with helpful message
                available_events = [f"on_{e.value}" for e in AgentStateEvent]
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                    f"Available events: {available_events}"
                )

            # Return a function that registers the callback
            def register_callback(callback):
                """
                Register a callback for the event.

                Args:
                    callback: Function to call when the event occurs.
                             Should accept a single argument (event_data dict).

                Returns:
                    The subscription object or None if registration failed.
                """
                if self.event_manager:
                    # Dynamically get the method from event_manager
                    method_name = f"on_{event_name}"

                    if hasattr(self.event_manager, method_name):
                        method = getattr(self.event_manager, method_name)
                        return method(callback)
                    else:
                        self.agent_logger.warning(
                            f"EventManager does not have method '{method_name}'"
                        )
                        return None
                else:
                    self.agent_logger.warning("Event manager not initialized")
                    return None

            # Add docstring to the returned function for better IntelliSense
            register_callback.__doc__ = f"""
            Register a callback for the {event_name} event.
            
            Event Type: {event_type.value}
            
            Args:
                callback: Function that will be called when the event occurs.
                         The callback should accept one argument containing event data.
            
            Returns:
                Subscription object for managing the callback, or None if failed.
            
            Example:
                >>> def my_callback(event_data):
                ...     print(f"Event: {{event_data['event_type']}}")
                >>> agent.{name}(my_callback)
            """

            return register_callback

        # If not an event handler, raise AttributeError
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def get_available_events(self) -> List[str]:
        """Get list of all available event types for subscription."""
        return [event.value for event in AgentStateEvent]

    def subscribe_to_all_events(self, callback) -> Dict[str, Any]:
        """
        Subscribe to all available events with a single callback.

        Args:
            callback: The callback function to register for all events

        Returns:
            Dictionary mapping event names to subscription results
        """
        subscriptions = {}
        for event_name in self.get_available_events():
            try:
                handler_name = f"on_{event_name}"
                handler = getattr(self, handler_name)
                subscriptions[event_name] = handler(callback)
            except Exception as e:
                self.agent_logger.warning(f"Failed to subscribe to {event_name}: {e}")
                subscriptions[event_name] = None
        return subscriptions

    # --- Vector Memory Methods ---
    async def search_memory(
        self, query: str, n_results: int = 5, memory_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search through agent memories using semantic similarity.

        Args:
            query: The search query
            n_results: Maximum number of results to return
            memory_types: Optional filter by memory types

        Returns:
            List of relevant memory items with metadata
        """
        if not self.context.memory_manager:
            return []

        if hasattr(self.context.memory_manager, "search_memory"):
            # Vector memory manager
            memory_manager = self.context.memory_manager
            if hasattr(memory_manager, "is_ready"):
                # This is a VectorMemoryManager
                if not memory_manager.is_ready():  # type: ignore
                    self.agent_logger.warning("Vector memory not ready, cannot search")
                    return []
            return await self.context.memory_manager.search_memory(  # type: ignore
                query, n_results, memory_types
            )
        else:
            # Traditional memory manager - return empty for now
            return []

    async def get_context_memories(
        self, task: str, max_items: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get memories relevant to the current task context.

        Args:
            task: The current task description
            max_items: Maximum number of context memories to return

        Returns:
            List of contextually relevant memories
        """
        if not self.context.memory_manager:
            return []

        if hasattr(self.context.memory_manager, "get_context_memories"):
            # Vector memory manager
            memory_manager = self.context.memory_manager
            if hasattr(memory_manager, "is_ready"):
                # This is a VectorMemoryManager
                if not memory_manager.is_ready():  # type: ignore
                    self.agent_logger.warning(
                        "Vector memory not ready, cannot get context memories"
                    )
                    return []
            return await self.context.memory_manager.get_context_memories(  # type: ignore
                task, max_items
            )
        else:
            # Traditional memory manager - return empty for now
            return []

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        if not self.context.memory_manager:
            return {"error": "Memory not initialized"}

        if hasattr(self.context.memory_manager, "get_memory_stats"):
            # Vector memory manager
            memory_manager = self.context.memory_manager
            if hasattr(memory_manager, "is_ready"):
                # This is a VectorMemoryManager
                if not memory_manager.is_ready():  # type: ignore
                    return {
                        "error": "Vector memory not ready",
                        "memory_type": "vector",
                        "vector_memory_enabled": True,
                        "ready": False,
                    }
            return self.context.memory_manager.get_memory_stats()  # type: ignore
        else:
            # Traditional memory manager
            return {
                "memory_type": "traditional",
                "vector_memory_enabled": False,
                "session_history_count": len(self.context.get_session_history()),
                "reflections_count": len(self.context.get_reflections()),
            }

    # --- Utility Methods ---
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the agent."""
        return {
            "agent_name": self.context.agent_name,
            "model_provider": self.context.provider_model_name,
            "role": self.context.role,
            "instructions": self.context.instructions,
            "max_iterations": self.context.max_iterations,
            "reflect_enabled": self.context.reflect_enabled,
            "tool_use_enabled": self.context.tool_use_enabled,
            "use_memory_enabled": self.context.use_memory_enabled,
            "vector_memory_enabled": getattr(
                self.context, "vector_memory_enabled", False
            ),
            "collect_metrics_enabled": self.context.collect_metrics_enabled,
            "current_strategy": self.get_current_strategy(),
            "available_strategies": self.get_available_strategies(),
            "available_events": self.get_available_events(),
            "is_initialized": self.is_initialized,
            "is_closed": self.is_closed,
            "tools_count": len(self.context.tools) if self.context.tools else 0,
        }

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        if not self.context.session:
            return {}

        return {
            "initial_task": self.context.session.initial_task,
            "current_task": self.context.session.current_task,
            "iterations": self.context.session.iterations,
            "final_answer": self.context.session.final_answer,
            "completion_score": self.context.session.completion_score,
            "task_status": (
                self.context.session.task_status.value
                if self.context.session.task_status
                else None
            ),
            "successful_tools": list(self.context.session.successful_tools),
            "start_time": self.context.session.start_time,
            "messages_count": len(self.context.session.messages),
        }

    async def __aenter__(self):
        """Async context manager entry - initialize the agent."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()
        return False
