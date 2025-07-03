"""
Official EventBus Implementation

This module provides a centralized, middleware-capable event system that replaces
the current AgentStateObserver with enhanced functionality including:

- Event middleware for processing and filtering
- Event persistence and replay
- Distributed event handling
- Type-safe event subscriptions
- Performance monitoring and debugging
"""

from __future__ import annotations
import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Callable,
    Awaitable,
    Set,
    Union,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
)
from datetime import datetime
from dataclasses import dataclass, field
from pydantic import BaseModel
from enum import Enum

from reactive_agents.core.types.event_types import AgentStateEvent

if TYPE_CHECKING:
    from reactive_agents.utils.logging import Logger


class EventPriority(Enum):
    """Event priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Event:
    """Enhanced event structure with metadata and routing information"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure timestamp is set if not provided"""
        if not self.timestamp:
            self.timestamp = time.time()


@runtime_checkable
class EventHandler(Protocol):
    """Protocol for event handlers"""

    async def handle(self, event: Event) -> None:
        """Handle an event"""
        ...


@runtime_checkable
class EventMiddleware(Protocol):
    """Protocol for event middleware"""

    async def process(
        self, event: Event, next_handler: Callable[[Event], Awaitable[None]]
    ) -> None:
        """Process an event before it reaches handlers"""
        ...


class EventPersistence(ABC):
    """Abstract base for event persistence"""

    @abstractmethod
    async def store_event(self, event: Event) -> None:
        """Store an event for replay/audit purposes"""
        pass

    @abstractmethod
    async def get_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Retrieve stored events"""
        pass


class MemoryEventPersistence(EventPersistence):
    """In-memory event persistence for development/testing"""

    def __init__(self, max_events: int = 10000):
        self.events: List[Event] = []
        self.max_events = max_events

    async def store_event(self, event: Event) -> None:
        """Store event in memory with size limit"""
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

    async def get_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Filter and return stored events"""
        filtered_events = self.events

        if event_type:
            filtered_events = [e for e in filtered_events if e.type == event_type]

        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]

        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]

        return filtered_events[-limit:] if limit else filtered_events


class EventSubscription:
    """Represents a subscription to an event type"""

    def __init__(self, event_type: str, handler: EventHandler, event_bus: "EventBus"):
        self.event_type = event_type
        self.handler = handler
        self.event_bus = event_bus
        self.active = True

    def unsubscribe(self) -> None:
        """Unsubscribe from the event"""
        self.active = False
        self.event_bus._remove_subscription(self)


class LoggingMiddleware:
    """Middleware that logs all events"""

    def __init__(self, logger: Optional["Logger"] = None):
        self.logger = logger

    async def process(
        self, event: Event, next_handler: Callable[[Event], Awaitable[None]]
    ) -> None:
        """Log the event and continue processing"""
        if self.logger:
            self.logger.debug(f"Processing event: {event.type} (ID: {event.id[:8]})")

        start_time = time.time()
        try:
            await next_handler(event)
        finally:
            processing_time = time.time() - start_time
            if self.logger:
                self.logger.debug(
                    f"Event {event.type} processed in {processing_time:.3f}s"
                )


class FilteringMiddleware:
    """Middleware that filters events based on criteria"""

    def __init__(self, filter_func: Callable[[Event], bool]):
        self.filter_func = filter_func

    async def process(
        self, event: Event, next_handler: Callable[[Event], Awaitable[None]]
    ) -> None:
        """Filter events based on criteria"""
        if self.filter_func(event):
            await next_handler(event)


class RateLimitingMiddleware:
    """Middleware that implements rate limiting"""

    def __init__(self, max_events_per_second: float = 100.0):
        self.max_events_per_second = max_events_per_second
        self.event_times: List[float] = []

    async def process(
        self, event: Event, next_handler: Callable[[Event], Awaitable[None]]
    ) -> None:
        """Apply rate limiting"""
        current_time = time.time()

        # Clean old entries
        cutoff_time = current_time - 1.0
        self.event_times = [t for t in self.event_times if t > cutoff_time]

        # Check rate limit
        if len(self.event_times) < self.max_events_per_second:
            self.event_times.append(current_time)
            await next_handler(event)
        # If rate limited, we silently drop the event


class EventBus:
    """
    Centralized event management system with middleware support.

    This replaces the current AgentStateObserver with enhanced functionality:
    - Middleware pipeline for event processing
    - Event persistence and replay capabilities
    - Performance monitoring and debugging
    - Type-safe subscriptions with proper cleanup
    """

    def __init__(self, logger: Optional["Logger"] = None):
        self.logger = logger
        self.middleware: List[EventMiddleware] = []
        self.subscribers: Dict[str, List[EventHandler]] = {}
        self.subscriptions: List[EventSubscription] = []
        self.persistence: Optional[EventPersistence] = None
        self.stats = {
            "events_emitted": 0,
            "events_processed": 0,
            "middleware_errors": 0,
            "handler_errors": 0,
            "processing_times": [],
            "events_by_type": {},
            "last_event_time": None,
        }
        self._running = True

        # Add default logging middleware
        if logger:
            self.add_middleware(LoggingMiddleware(logger))

    def add_middleware(self, middleware: EventMiddleware) -> None:
        """Add event processing middleware"""
        self.middleware.append(middleware)
        if self.logger:
            self.logger.debug(f"Added middleware: {type(middleware).__name__}")

    def set_persistence(self, persistence: EventPersistence) -> None:
        """Set event persistence handler"""
        self.persistence = persistence
        if self.logger:
            self.logger.info(f"Event persistence enabled: {type(persistence).__name__}")

    def subscribe(self, event_type: str, handler: EventHandler) -> EventSubscription:
        """Subscribe to an event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []

        self.subscribers[event_type].append(handler)
        subscription = EventSubscription(event_type, handler, self)
        self.subscriptions.append(subscription)

        if self.logger:
            self.logger.debug(f"New subscription: {event_type}")

        return subscription

    def subscribe_function(
        self, event_type: str, func: Callable[[Event], Awaitable[None]]
    ) -> EventSubscription:
        """Subscribe a function as an event handler"""

        class FunctionHandler:
            def __init__(self, func: Callable[[Event], Awaitable[None]]):
                self.func = func

            async def handle(self, event: Event) -> None:
                await self.func(event)

        return self.subscribe(event_type, FunctionHandler(func))

    def _remove_subscription(self, subscription: EventSubscription) -> None:
        """Internal method to remove a subscription"""
        if subscription in self.subscriptions:
            self.subscriptions.remove(subscription)

        if subscription.event_type in self.subscribers:
            if subscription.handler in self.subscribers[subscription.event_type]:
                self.subscribers[subscription.event_type].remove(subscription.handler)

    async def emit(
        self, event: Union[Event, str], data: Optional[Dict[str, Any]] = None, **kwargs
    ) -> None:
        """
        Emit an event to all subscribers through the middleware pipeline.

        Args:
            event: Event object or event type string
            data: Event data (if event is a string)
            **kwargs: Additional event properties
        """
        if not self._running:
            return

        # Create Event object if string provided
        if isinstance(event, str):
            event_obj = Event(type=event, data=data or {}, **kwargs)
        else:
            event_obj = event

        # Update stats
        self.stats["events_emitted"] += 1
        self.stats["last_event_time"] = event_obj.timestamp
        self.stats["events_by_type"][event_obj.type] = (
            self.stats["events_by_type"].get(event_obj.type, 0) + 1
        )

        # Process through middleware pipeline
        try:
            await self._process_event(event_obj)
        except Exception as e:
            self.stats["middleware_errors"] += 1
            if self.logger:
                self.logger.error(f"Error processing event {event_obj.type}: {e}")

    async def _process_event(self, event: Event) -> None:
        """Process event through middleware pipeline"""

        async def final_handler(processed_event: Event) -> None:
            """Final handler that dispatches to subscribers"""
            await self._dispatch_to_subscribers(processed_event)

        # Build middleware chain
        handler = final_handler
        for middleware in reversed(self.middleware):
            current_middleware = middleware
            current_handler = handler

            async def chained_handler(
                e: Event, m=current_middleware, h=current_handler
            ) -> None:
                await m.process(e, h)

            handler = chained_handler

        # Execute the chain
        start_time = time.time()
        try:
            await handler(event)
            processing_time = time.time() - start_time
            self.stats["processing_times"].append(processing_time)

            # Keep only last 1000 processing times for memory efficiency
            if len(self.stats["processing_times"]) > 1000:
                self.stats["processing_times"] = self.stats["processing_times"][-1000:]

            # Store event if persistence is enabled
            if self.persistence:
                await self.persistence.store_event(event)

        except Exception as e:
            self.stats["middleware_errors"] += 1
            if self.logger:
                self.logger.error(f"Middleware error for event {event.type}: {e}")

    async def _dispatch_to_subscribers(self, event: Event) -> None:
        """Dispatch event to all subscribers"""

        if event.type not in self.subscribers:
            return

        handlers = self.subscribers[event.type][
            :
        ]  # Copy to avoid modification during iteration

        # Dispatch to all handlers concurrently
        if handlers:
            tasks = []
            for handler in handlers:
                tasks.append(self._safe_handle(handler, event))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                self.stats["events_processed"] += 1

    async def _safe_handle(self, handler: EventHandler, event: Event) -> None:
        """Safely handle an event, catching and logging errors"""
        try:
            await handler.handle(event)
        except Exception as e:
            self.stats["handler_errors"] += 1
            if self.logger:
                self.logger.error(f"Handler error for event {event.type}: {e}")

    async def replay_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> int:
        """
        Replay stored events through the current subscriber set.

        Returns:
            Number of events replayed
        """
        if not self.persistence:
            if self.logger:
                self.logger.warning("No persistence configured, cannot replay events")
            return 0

        events = await self.persistence.get_events(
            event_type, start_time, end_time, limit
        )

        replayed_count = 0
        for event in events:
            # Mark as replay to avoid re-storing
            event.metadata["replayed"] = True
            await self._dispatch_to_subscribers(event)
            replayed_count += 1

        if self.logger:
            self.logger.info(f"Replayed {replayed_count} events")

        return replayed_count

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        avg_processing_time = (
            sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
            if self.stats["processing_times"]
            else 0
        )

        return {
            **self.stats,
            "avg_processing_time": avg_processing_time,
            "active_subscribers": sum(
                len(handlers) for handlers in self.subscribers.values()
            ),
            "middleware_count": len(self.middleware),
            "persistence_enabled": self.persistence is not None,
        }

    def get_subscriber_count(self, event_type: str) -> int:
        """Get number of subscribers for an event type"""
        return len(self.subscribers.get(event_type, []))

    async def close(self) -> None:
        """Close the event bus and clean up resources"""
        self._running = False

        # Cancel all subscriptions
        for subscription in self.subscriptions:
            subscription.active = False

        self.subscriptions.clear()
        self.subscribers.clear()

        if self.logger:
            self.logger.info("EventBus closed")


class AgentEventBus(EventBus):
    """
    Specialized EventBus for ReactiveAgent with pre-configured middleware
    and AgentStateEvent integration.
    """

    def __init__(self, agent_name: str, logger: Optional["Logger"] = None):
        super().__init__(logger)
        self.agent_name = agent_name

        # Add agent-specific middleware
        self.add_middleware(FilteringMiddleware(self._agent_event_filter))
        self.add_middleware(RateLimitingMiddleware(max_events_per_second=50.0))

        # Enable memory persistence for development
        self.set_persistence(MemoryEventPersistence(max_events=5000))

    def _agent_event_filter(self, event: Event) -> bool:
        """Filter events specific to this agent"""
        # Allow events without agent context or events for this agent
        agent_name = event.data.get("agent_name")
        return agent_name is None or agent_name == self.agent_name

    async def emit_agent_event(
        self,
        event_type: AgentStateEvent,
        data: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        """Emit an agent state event with proper context"""

        # Add agent context to event data
        enhanced_data = {
            "agent_name": self.agent_name,
            "timestamp": time.time(),
            **data,
        }

        event = Event(
            type=event_type.value,
            data=enhanced_data,
            priority=priority,
            source=f"agent:{self.agent_name}",
            tags={"agent_event", self.agent_name},
        )

        await self.emit(event)

    def subscribe_to_agent_event(
        self,
        event_type: AgentStateEvent,
        handler: Union[EventHandler, Callable[[Event], Awaitable[None]]],
    ) -> EventSubscription:
        """Subscribe to a specific agent state event"""

        if callable(handler) and not hasattr(handler, "handle"):
            # Handler is a function
            return self.subscribe_function(event_type.value, handler)
        else:
            # Handler is an EventHandler protocol object
            # Type assertion to help the type checker
            event_handler: EventHandler = handler  # type: ignore
            return self.subscribe(event_type.value, event_handler)


# Convenience function for creating agent event buses
def create_agent_event_bus(
    agent_name: str, logger: Optional["Logger"] = None
) -> AgentEventBus:
    """Create a pre-configured event bus for an agent"""
    return AgentEventBus(agent_name, logger)
