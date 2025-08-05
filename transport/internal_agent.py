#!/usr/bin/env python3
"""
Internal Agent Communication

Handles communication between orchestrator and agent instances using internal interfaces.
Agents run in the same process but communicate through clean, async interfaces.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from agents.agent import Agent

logger = logging.getLogger(__name__)


class InternalAgentClient:
    """Client for communicating with agent instances in the same process"""

    def __init__(self, agent_id: str, agent_instance: Optional['Agent'] = None):
        self.agent_id = agent_id
        self.agent_instance = agent_instance
        self.connected = False

    async def connect(self) -> bool:
        """Connect to the agent instance"""
        if self.agent_instance:
            self.connected = True
            logger.info(f"Connected to internal agent {self.agent_id}")
            return True
        else:
            logger.error(f"No agent instance provided for {self.agent_id}")
            return False

    async def disconnect(self):
        """Disconnect from the agent"""
        self.connected = False
        logger.info(f"Disconnected from agent {self.agent_id}")

    async def execute_task(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Send task execution request to agent"""
        if not self.connected or not self.agent_instance:
            return {"success": False, "error": "Agent not connected"}

        try:
            logger.info(f"Executing {tool_name} on internal agent {self.agent_id}")
            result = await self.agent_instance.execute_tool(tool_name, arguments)
            logger.info(f"Task completed on agent {self.agent_id}: {result.get('success', False)}")
            return result

        except Exception as e:
            logger.error(f"Task execution failed for agent {self.agent_id}: {e}")
            return {"success": False, "error": str(e)}

    async def heartbeat(self) -> bool:
        """Check if agent is healthy"""
        if not self.connected or not self.agent_instance:
            return False

        try:
            status = self.agent_instance.get_status()
            return status.get("running", False)
        except Exception as e:
            logger.error(f"Heartbeat failed for agent {self.agent_id}: {e}")
            return False

    async def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        if not self.connected or not self.agent_instance:
            return {"agent_id": self.agent_id, "connected": False}

        try:
            status = self.agent_instance.get_status()
            status["connected"] = True
            return status
        except Exception as e:
            logger.error(f"Failed to get status for agent {self.agent_id}: {e}")
            return {"agent_id": self.agent_id, "connected": False, "error": str(e)}


class InternalAgentServer:
    """Server interface for agents running in the same process"""

    def __init__(self, agent_instance: 'Agent'):
        self.agent_instance = agent_instance
        self.running = False

    async def start(self):
        """Start the internal agent server"""
        self.running = True
        logger.info(f"Started internal agent server for {self.agent_instance.agent_id}")

    async def stop(self):
        """Stop the internal agent server"""
        self.running = False
        logger.info(f"Stopped internal agent server for {self.agent_instance.agent_id}")


class InternalAgentRegistry:
    """Registry for managing internal agent instances"""

    def __init__(self):
        self.agents: Dict[str, 'Agent'] = {}
        self.clients: Dict[str, InternalAgentClient] = {}

    def register_agent(self, agent_id: str, agent_instance: 'Agent') -> InternalAgentClient:
        """Register an agent instance and create a client"""
        self.agents[agent_id] = agent_instance
        client = InternalAgentClient(agent_id, agent_instance)
        self.clients[agent_id] = client
        logger.info(f"Registered internal agent {agent_id}")
        return client

    def get_client(self, agent_id: str) -> Optional[InternalAgentClient]:
        """Get client for an agent"""
        return self.clients.get(agent_id)

    def get_agent(self, agent_id: str) -> Optional['Agent']:
        """Get agent instance"""
        return self.agents.get(agent_id)

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all registered agents"""
        agent_list = {}
        for agent_id, agent in self.agents.items():
            try:
                status = agent.get_status()
                agent_list[agent_id] = status
            except Exception as e:
                agent_list[agent_id] = {"agent_id": agent_id, "error": str(e)}
        return agent_list

    def remove_agent(self, agent_id: str):
        """Remove an agent from the registry"""
        if agent_id in self.agents:
            del self.agents[agent_id]
        if agent_id in self.clients:
            del self.clients[agent_id]
        logger.info(f"Removed internal agent {agent_id}")


# Global registry instance
_agent_registry = InternalAgentRegistry()


def get_agent_registry() -> InternalAgentRegistry:
    """Get the global agent registry"""
    return _agent_registry
