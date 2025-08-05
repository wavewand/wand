"""
WebSocket Integration for Real-time Health Monitoring
Streams health updates to connected clients in real-time
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Set

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from ..monitoring.integration_health_monitor import HealthMetrics, get_health_monitor

logger = logging.getLogger(__name__)


class HealthWebSocketManager:
    """
    Manages WebSocket connections for real-time health monitoring
    """

    def __init__(self):
        # Active WebSocket connections
        self.active_connections: Set[WebSocket] = set()

        # Client subscriptions (websocket -> set of integration names)
        self.subscriptions: Dict[WebSocket, Set[str]] = {}

        # Health broadcast task
        self.broadcast_task: Optional[asyncio.Task] = None
        self.is_broadcasting = False

        # Health state cache for change detection
        self.last_health_state: Dict[str, HealthMetrics] = {}

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscriptions[websocket] = set()

        logger.info(f"🔌 Health WebSocket client connected. Total: {len(self.active_connections)}")

        # Start broadcasting if this is the first connection
        if len(self.active_connections) == 1 and not self.is_broadcasting:
            await self.start_health_broadcasting()

        # Send current health summary to new client
        try:
            await self.send_current_health_summary(websocket)
        except Exception as e:
            logger.error(f"Failed to send initial health summary: {e}")

    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if websocket in self.subscriptions:
            del self.subscriptions[websocket]

        logger.info(f"🔌 Health WebSocket client disconnected. Total: {len(self.active_connections)}")

        # Stop broadcasting if no clients remain
        if len(self.active_connections) == 0 and self.is_broadcasting:
            await self.stop_health_broadcasting()

    async def handle_message(self, websocket: WebSocket, message: dict):
        """Handle incoming WebSocket messages"""
        try:
            msg_type = message.get('type')
            action = message.get('action', 'request')
            data = message.get('data', {})

            if msg_type == 'subscribe' and action == 'request':
                await self.handle_subscription(websocket, data)
            elif msg_type == 'unsubscribe' and action == 'request':
                await self.handle_unsubscription(websocket, data)
            elif msg_type == 'health_request' and action == 'request':
                await self.handle_health_request(websocket, data)
            elif msg_type == 'ping':
                await self.send_pong(websocket)
            else:
                logger.warning(f"Unknown message type: {msg_type}")

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self.send_error(websocket, f"Message handling error: {str(e)}")

    async def handle_subscription(self, websocket: WebSocket, data: dict):
        """Handle subscription requests"""
        integration_names = data.get('integrations', [])

        if 'all' in integration_names:
            # Subscribe to all integrations
            monitor = await get_health_monitor()
            all_health = await monitor.get_all_health_metrics()
            integration_names = list(all_health.keys())

        # Add to subscriptions
        if websocket not in self.subscriptions:
            self.subscriptions[websocket] = set()

        self.subscriptions[websocket].update(integration_names)

        # Confirm subscription
        await self.send_message(
            websocket,
            {
                'type': 'subscription',
                'action': 'confirmed',
                'data': {
                    'integrations': list(self.subscriptions[websocket]),
                    'total_subscribed': len(self.subscriptions[websocket]),
                },
                'timestamp': datetime.utcnow().isoformat(),
            },
        )

        logger.info(f"📡 Client subscribed to {len(integration_names)} integrations")

    async def handle_unsubscription(self, websocket: WebSocket, data: dict):
        """Handle unsubscription requests"""
        integration_names = data.get('integrations', [])

        if websocket in self.subscriptions:
            if 'all' in integration_names:
                # Unsubscribe from all
                self.subscriptions[websocket].clear()
            else:
                # Remove specific integrations
                self.subscriptions[websocket].difference_update(integration_names)

        # Confirm unsubscription
        await self.send_message(
            websocket,
            {
                'type': 'unsubscription',
                'action': 'confirmed',
                'data': {
                    'integrations': list(self.subscriptions[websocket]),
                    'total_subscribed': len(self.subscriptions[websocket]),
                },
                'timestamp': datetime.utcnow().isoformat(),
            },
        )

    async def handle_health_request(self, websocket: WebSocket, data: dict):
        """Handle specific health data requests"""
        integration_name = data.get('integration')

        try:
            monitor = await get_health_monitor()

            if integration_name:
                # Request specific integration health
                health = await monitor.get_integration_health(integration_name)
                if health:
                    await self.send_message(
                        websocket,
                        {
                            'type': 'health_data',
                            'action': 'response',
                            'data': {'integration': integration_name, 'health': health.to_dict()},
                            'timestamp': datetime.utcnow().isoformat(),
                        },
                    )
                else:
                    await self.send_error(websocket, f"Integration '{integration_name}' not found")
            else:
                # Request all health data
                all_health = await monitor.get_all_health_metrics()
                health_data = {name: health.to_dict() for name, health in all_health.items()}

                await self.send_message(
                    websocket,
                    {
                        'type': 'health_data',
                        'action': 'response',
                        'data': {'integrations': health_data, 'total': len(health_data)},
                        'timestamp': datetime.utcnow().isoformat(),
                    },
                )

        except Exception as e:
            logger.error(f"Error handling health request: {e}")
            await self.send_error(websocket, f"Health request error: {str(e)}")

    async def send_current_health_summary(self, websocket: WebSocket):
        """Send current health summary to a client"""
        try:
            monitor = await get_health_monitor()
            summary = await monitor.get_health_summary()

            await self.send_message(
                websocket,
                {
                    'type': 'health_summary',
                    'action': 'update',
                    'data': summary,
                    'timestamp': datetime.utcnow().isoformat(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to send health summary: {e}")

    async def send_pong(self, websocket: WebSocket):
        """Send pong response"""
        await self.send_message(websocket, {'type': 'pong', 'timestamp': datetime.utcnow().isoformat()})

    async def send_error(self, websocket: WebSocket, error_message: str):
        """Send error message to client"""
        await self.send_message(
            websocket,
            {
                'type': 'error',
                'action': 'notification',
                'data': {'message': error_message},
                'timestamp': datetime.utcnow().isoformat(),
            },
        )

    async def send_message(self, websocket: WebSocket, message: dict):
        """Send a message to a specific WebSocket client"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            # Remove broken connection
            if websocket in self.active_connections:
                await self.disconnect(websocket)

    async def broadcast_message(self, message: dict, target_integrations: Set[str] = None):
        """Broadcast a message to all connected clients (with subscription filtering)"""
        if not self.active_connections:
            return

        # Send to all clients who are subscribed to relevant integrations
        for websocket in list(self.active_connections):
            try:
                # Check if client is subscribed to any of the target integrations
                if target_integrations is None or not self.subscriptions.get(websocket):
                    # Send to all if no specific targets or client has no subscriptions
                    await self.send_message(websocket, message)
                else:
                    # Check if client is subscribed to any target integration
                    client_subscriptions = self.subscriptions.get(websocket, set())
                    if target_integrations.intersection(client_subscriptions):
                        await self.send_message(websocket, message)

            except Exception as e:
                logger.error(f"Failed to broadcast to client: {e}")
                await self.disconnect(websocket)

    async def start_health_broadcasting(self):
        """Start the background health broadcasting task"""
        if not self.is_broadcasting:
            self.is_broadcasting = True
            self.broadcast_task = asyncio.create_task(self.health_broadcast_loop())
            logger.info("🎯 Started health broadcasting task")

    async def stop_health_broadcasting(self):
        """Stop the background health broadcasting task"""
        if self.is_broadcasting:
            self.is_broadcasting = False
            if self.broadcast_task:
                self.broadcast_task.cancel()
                try:
                    await self.broadcast_task
                except asyncio.CancelledError:
                    pass
            logger.info("🎯 Stopped health broadcasting task")

    async def health_broadcast_loop(self):
        """Main broadcasting loop that detects and streams health changes"""
        try:
            while self.is_broadcasting:
                try:
                    monitor = await get_health_monitor()
                    current_health = await monitor.get_all_health_metrics()

                    # Detect changes and broadcast updates
                    for integration_name, current_metrics in current_health.items():
                        previous_metrics = self.last_health_state.get(integration_name)

                        # Check if there's a significant change
                        if self.has_significant_change(previous_metrics, current_metrics):
                            await self.broadcast_health_update(integration_name, current_metrics)

                    # Update state cache
                    self.last_health_state = current_health.copy()

                    # Broadcast summary every 30 seconds
                    await self.broadcast_health_summary(monitor)

                except Exception as e:
                    logger.error(f"Error in health broadcast loop: {e}")

                # Wait before next broadcast cycle
                await asyncio.sleep(10)  # 10 second intervals

        except asyncio.CancelledError:
            logger.info("Health broadcasting task cancelled")
        except Exception as e:
            logger.error(f"Health broadcasting task error: {e}")

    def has_significant_change(self, previous: Optional[HealthMetrics], current: HealthMetrics) -> bool:
        """Detect if there's a significant change worth broadcasting"""
        if not previous:
            return True  # First time we see this integration

        # Check for status changes
        if previous.status != current.status:
            return True

        # Check for significant score changes (> 5 points)
        if abs(previous.score - current.score) > 5.0:
            return True

        # Check for connection status changes
        if previous.connection_status != current.connection_status:
            return True

        # Check for configuration changes
        if previous.config_valid != current.config_valid:
            return True

        return False

    async def broadcast_health_update(self, integration_name: str, health_metrics: HealthMetrics):
        """Broadcast health update for a specific integration"""
        await self.broadcast_message(
            {
                'type': 'integration_health',
                'action': 'update',
                'data': {'integration_name': integration_name, 'health': health_metrics.to_dict()},
                'timestamp': datetime.utcnow().isoformat(),
            },
            {integration_name},
        )

    async def broadcast_health_summary(self, monitor):
        """Broadcast system-wide health summary"""
        try:
            summary = await monitor.get_health_summary()
            await self.broadcast_message(
                {
                    'type': 'health_summary',
                    'action': 'update',
                    'data': summary,
                    'timestamp': datetime.utcnow().isoformat(),
                }
            )
        except Exception as e:
            logger.error(f"Failed to broadcast health summary: {e}")


# Global health WebSocket manager instance
health_ws_manager = HealthWebSocketManager()


async def get_health_ws_manager() -> HealthWebSocketManager:
    """Get the global health WebSocket manager instance"""
    return health_ws_manager


# WebSocket endpoint handler


async def health_websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for health monitoring"""
    manager = await get_health_ws_manager()

    try:
        await manager.connect(websocket)

        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                await manager.handle_message(websocket, message)

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from WebSocket client: {e}")
                await manager.send_error(websocket, "Invalid JSON format")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await manager.send_error(websocket, f"Server error: {str(e)}")

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        await manager.disconnect(websocket)


# Export main components
__all__ = ['health_ws_manager', 'get_health_ws_manager', 'health_websocket_endpoint']
