"""
WebSocket Event Manager

Provides real-time event broadcasting for framework operations, monitoring,
and system events via WebSocket connections.
"""

import asyncio
import json
import logging
import uuid
import weakref
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect


class EventType(str, Enum):
    """Types of events that can be broadcast."""

    FRAMEWORK_OPERATION = "framework_operation"
    BATCH_OPERATION = "batch_operation"
    BENCHMARK_OPERATION = "benchmark_operation"
    MONITORING_ALERT = "monitoring_alert"
    SYSTEM_STATUS = "system_status"
    AGENT_STATUS = "agent_status"
    ERROR = "error"
    INFO = "info"


class EventPriority(str, Enum):
    """Priority levels for events."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Event:
    """Represents a system event."""

    id: str
    type: EventType
    priority: EventPriority
    source: str
    title: str
    message: str
    data: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.data is None:
            self.data = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "source": self.source,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class WebSocketConnection:
    """Represents a WebSocket connection with filtering capabilities."""

    def __init__(self, websocket: WebSocket, connection_id: str):
        self.websocket = websocket
        self.connection_id = connection_id
        self.connected_at = datetime.now()
        self.last_ping = datetime.now()

        # Event filtering
        self.event_types: Set[EventType] = set()
        self.priority_filter: Optional[EventPriority] = None
        self.source_filter: Set[str] = set()

        # Statistics
        self.events_sent = 0
        self.events_filtered = 0

    def should_receive_event(self, event: Event) -> bool:
        """Check if this connection should receive the event based on filters."""
        # Check event type filter
        if self.event_types and event.type not in self.event_types:
            self.events_filtered += 1
            return False

        # Check priority filter
        if self.priority_filter:
            priority_order = {
                EventPriority.LOW: 0,
                EventPriority.NORMAL: 1,
                EventPriority.HIGH: 2,
                EventPriority.CRITICAL: 3,
            }
            if priority_order[event.priority] < priority_order[self.priority_filter]:
                self.events_filtered += 1
                return False

        # Check source filter
        if self.source_filter and event.source not in self.source_filter:
            self.events_filtered += 1
            return False

        return True

    async def send_event(self, event: Event):
        """Send an event to this connection."""
        if not self.should_receive_event(event):
            return

        try:
            await self.websocket.send_text(event.to_json())
            self.events_sent += 1
        except Exception as e:
            logging.error(f"Failed to send event to connection {self.connection_id}: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "connection_id": self.connection_id,
            "connected_at": self.connected_at.isoformat(),
            "uptime_seconds": (datetime.now() - self.connected_at).total_seconds(),
            "events_sent": self.events_sent,
            "events_filtered": self.events_filtered,
            "last_ping": self.last_ping.isoformat(),
            "filters": {
                "event_types": [t.value for t in self.event_types],
                "priority_filter": self.priority_filter.value if self.priority_filter else None,
                "source_filter": list(self.source_filter),
            },
        }


class WebSocketEventManager:
    """Manages WebSocket connections and event broadcasting."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connections: Dict[str, WebSocketConnection] = {}
        self.event_history: List[Event] = []
        self.max_history_size = 1000

        # Event listeners
        self.event_listeners: Dict[EventType, List[Callable]] = {}

        # Statistics
        self.total_events_sent = 0
        self.total_connections = 0

        self.logger.info("WebSocket Event Manager initialized")

    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new WebSocket connection."""
        await websocket.accept()

        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(websocket, connection_id)
        self.connections[connection_id] = connection
        self.total_connections += 1

        self.logger.info(f"WebSocket connection established: {connection_id}")

        # Send welcome message
        welcome_event = Event(
            id=str(uuid.uuid4()),
            type=EventType.INFO,
            priority=EventPriority.NORMAL,
            source="websocket_manager",
            title="Connection Established",
            message=f"WebSocket connection established with ID: {connection_id}",
            data={"connection_id": connection_id},
        )

        await connection.send_event(welcome_event)

        # Send recent events to new connection
        await self._send_recent_events(connection)

        return connection_id

    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
        if connection_id in self.connections:
            del self.connections[connection_id]
            self.logger.info(f"WebSocket connection closed: {connection_id}")

    async def broadcast_event(self, event: Event):
        """Broadcast an event to all connected clients."""
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size :]

        # Send to all connections
        disconnected_connections = []

        for connection_id, connection in self.connections.items():
            try:
                await connection.send_event(event)
                self.total_events_sent += 1
            except Exception as e:
                self.logger.error(f"Failed to send event to connection {connection_id}: {e}")
                disconnected_connections.append(connection_id)

        # Clean up disconnected connections
        for connection_id in disconnected_connections:
            self.disconnect(connection_id)

        # Trigger event listeners
        await self._trigger_event_listeners(event)

    async def send_framework_event(
        self,
        framework: str,
        operation: str,
        success: bool,
        execution_time_ms: float = None,
        data: Dict[str, Any] = None,
    ):
        """Send a framework operation event."""
        event = Event(
            id=str(uuid.uuid4()),
            type=EventType.FRAMEWORK_OPERATION,
            priority=EventPriority.NORMAL,
            source=f"framework_{framework}",
            title=f"{framework.title()} {operation.title()}",
            message=f"{framework} {operation} {'succeeded' if success else 'failed'}",
            data={
                "framework": framework,
                "operation": operation,
                "success": success,
                "execution_time_ms": execution_time_ms,
                **(data or {}),
            },
        )

        await self.broadcast_event(event)

    async def send_batch_event(
        self,
        batch_id: str,
        batch_type: str,
        status: str,
        framework: str,
        processed_items: int,
        total_items: int,
        data: Dict[str, Any] = None,
    ):
        """Send a batch operation event."""
        event = Event(
            id=str(uuid.uuid4()),
            type=EventType.BATCH_OPERATION,
            priority=EventPriority.NORMAL,
            source="batch_processor",
            title=f"Batch {batch_type.title()} {status.title()}",
            message=f"Batch {batch_type} ({batch_id}) is {status} - {processed_items}/{total_items} items",
            data={
                "batch_id": batch_id,
                "batch_type": batch_type,
                "status": status,
                "framework": framework,
                "processed_items": processed_items,
                "total_items": total_items,
                "progress_percent": (processed_items / total_items * 100) if total_items > 0 else 0,
                **(data or {}),
            },
        )

        await self.broadcast_event(event)

    async def send_benchmark_event(
        self,
        benchmark_id: str,
        benchmark_type: str,
        status: str,
        frameworks: List[str],
        winner: str = None,
        data: Dict[str, Any] = None,
    ):
        """Send a benchmark operation event."""
        event = Event(
            id=str(uuid.uuid4()),
            type=EventType.BENCHMARK_OPERATION,
            priority=EventPriority.HIGH,
            source="benchmarker",
            title=f"Benchmark {benchmark_type.title()} {status.title()}",
            message=f"Benchmark {benchmark_type} ({benchmark_id}) {status}"
            + (f" - Winner: {winner}" if winner else ""),
            data={
                "benchmark_id": benchmark_id,
                "benchmark_type": benchmark_type,
                "status": status,
                "frameworks": frameworks,
                "winner": winner,
                **(data or {}),
            },
        )

        await self.broadcast_event(event)

    async def send_monitoring_alert(
        self, alert_type: str, severity: str, message: str, framework: str = None, data: Dict[str, Any] = None
    ):
        """Send a monitoring alert event."""
        priority_map = {
            "info": EventPriority.LOW,
            "warning": EventPriority.NORMAL,
            "error": EventPriority.HIGH,
            "critical": EventPriority.CRITICAL,
        }

        event = Event(
            id=str(uuid.uuid4()),
            type=EventType.MONITORING_ALERT,
            priority=priority_map.get(severity.lower(), EventPriority.NORMAL),
            source=f"monitor_{framework}" if framework else "monitor",
            title=f"{alert_type.title()} Alert",
            message=message,
            data={"alert_type": alert_type, "severity": severity, "framework": framework, **(data or {})},
        )

        await self.broadcast_event(event)

    async def send_system_status_event(self, status: str, message: str, data: Dict[str, Any] = None):
        """Send a system status event."""
        event = Event(
            id=str(uuid.uuid4()),
            type=EventType.SYSTEM_STATUS,
            priority=EventPriority.HIGH if status != "healthy" else EventPriority.NORMAL,
            source="system",
            title=f"System Status: {status.title()}",
            message=message,
            data={"status": status, **(data or {})},
        )

        await self.broadcast_event(event)

    async def update_connection_filters(
        self,
        connection_id: str,
        event_types: List[str] = None,
        priority_filter: str = None,
        source_filter: List[str] = None,
    ):
        """Update event filters for a connection."""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]

        if event_types is not None:
            connection.event_types = {EventType(t) for t in event_types}

        if priority_filter is not None:
            connection.priority_filter = EventPriority(priority_filter)

        if source_filter is not None:
            connection.source_filter = set(source_filter)

        self.logger.info(f"Updated filters for connection {connection_id}")
        return True

    async def _send_recent_events(self, connection: WebSocketConnection):
        """Send recent events to a new connection."""
        # Send last 10 events
        recent_events = self.event_history[-10:] if len(self.event_history) > 10 else self.event_history

        for event in recent_events:
            try:
                await connection.send_event(event)
            except Exception as e:
                self.logger.error(f"Failed to send recent event to {connection.connection_id}: {e}")
                break

    async def _trigger_event_listeners(self, event: Event):
        """Trigger registered event listeners."""
        listeners = self.event_listeners.get(event.type, [])

        for listener in listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                else:
                    listener(event)
            except Exception as e:
                self.logger.error(f"Event listener failed: {e}")

    def add_event_listener(self, event_type: EventType, listener: Callable):
        """Add an event listener for a specific event type."""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []

        self.event_listeners[event_type].append(listener)
        self.logger.info(f"Added event listener for {event_type.value}")

    def get_connection_stats(self, connection_id: str = None) -> Dict[str, Any]:
        """Get statistics for a specific connection or all connections."""
        if connection_id:
            if connection_id in self.connections:
                return self.connections[connection_id].get_stats()
            return None
        else:
            return {
                "total_connections": len(self.connections),
                "total_connections_ever": self.total_connections,
                "total_events_sent": self.total_events_sent,
                "events_in_history": len(self.event_history),
                "active_connections": [conn.get_stats() for conn in self.connections.values()],
            }

    async def ping_connections(self):
        """Send ping to all connections to check if they're alive."""
        ping_event = Event(
            id=str(uuid.uuid4()),
            type=EventType.INFO,
            priority=EventPriority.LOW,
            source="websocket_manager",
            title="Ping",
            message="Connection health check",
            data={"ping": True},
        )

        disconnected_connections = []

        for connection_id, connection in self.connections.items():
            try:
                await connection.websocket.ping()
                connection.last_ping = datetime.now()
            except Exception as e:
                self.logger.info(f"Connection {connection_id} is no longer responsive")
                disconnected_connections.append(connection_id)

        # Clean up unresponsive connections
        for connection_id in disconnected_connections:
            self.disconnect(connection_id)


# Global event manager instance
websocket_event_manager = WebSocketEventManager()
