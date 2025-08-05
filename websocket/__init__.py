"""
WebSocket Package

Provides real-time event broadcasting and WebSocket management.
"""

from .event_manager import (
    Event,
    EventPriority,
    EventType,
    WebSocketConnection,
    WebSocketEventManager,
    websocket_event_manager,
)

__all__ = [
    'WebSocketEventManager',
    'websocket_event_manager',
    'Event',
    'EventType',
    'EventPriority',
    'WebSocketConnection',
]
