"""
Base Agent Implementation for MCP Distributed System
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from .types import AgentStatus, AgentType


class BaseAgent(ABC):
    """Base class for all agents in the distributed system."""

    def __init__(
        self,
        agent_id: str = None,
        agent_type: AgentType = AgentType.BACKEND,
        coordinator_address: str = "localhost:50051",
        capabilities: List[str] = None,
    ):
        self.agent_id = agent_id or f"{agent_type.value}-{str(uuid.uuid4())[:8]}"
        self.agent_type = agent_type
        self.coordinator_address = coordinator_address
        self.capabilities = capabilities or []
        self.status = AgentStatus.OFFLINE

        # Setup logging
        self.logger = logging.getLogger(f"agent.{self.agent_id}")

        # Task tracking
        self.current_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[str] = []

        # Performance metrics
        self.metrics = {"tasks_completed": 0, "tasks_failed": 0, "total_execution_time": 0.0, "uptime_start": None}

        self.logger.info(f"Initialized {agent_type.value} agent: {self.agent_id}")

    @abstractmethod
    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a task assigned to this agent. Must be implemented by subclasses."""
        pass

    async def start(self):
        """Start the agent and connect to coordinator."""
        self.logger.info(f"Starting agent {self.agent_id}")
        self.status = AgentStatus.ONLINE
        self.metrics["uptime_start"] = datetime.now()

        # gRPC connection to coordinator initialized when needed
        # Agent marked as online for immediate availability
        self.logger.info(f"Agent {self.agent_id} is online")

    async def stop(self):
        """Stop the agent gracefully."""
        self.logger.info(f"Stopping agent {self.agent_id}")
        self.status = AgentStatus.OFFLINE

        # Cancel any running tasks
        for task_id in list(self.current_tasks.keys()):
            await self._cancel_task(task_id)

        self.logger.info(f"Agent {self.agent_id} stopped")

    async def assign_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Assign a task to this agent."""
        task_id = task.get("id", str(uuid.uuid4()))

        self.logger.info(f"Received task {task_id}: {task.get('type', 'unknown')}")

        # Store task
        self.current_tasks[task_id] = {"task": task, "start_time": datetime.now(), "status": "in_progress"}

        try:
            # Set status to busy
            self.status = AgentStatus.BUSY

            # Handle the task
            result = await self.handle_task(task)

            # Update metrics
            execution_time = (datetime.now() - self.current_tasks[task_id]["start_time"]).total_seconds()
            self.metrics["tasks_completed"] += 1
            self.metrics["total_execution_time"] += execution_time

            # Move to completed
            del self.current_tasks[task_id]
            self.completed_tasks.append(task_id)

            # Update status
            self.status = AgentStatus.IDLE if not self.current_tasks else AgentStatus.BUSY

            self.logger.info(f"Completed task {task_id} in {execution_time:.2f}s")

            return {"success": True, "task_id": task_id, "result": result, "execution_time": execution_time}

        except Exception as e:
            self.logger.error(f"Task {task_id} failed: {e}")

            # Update metrics
            self.metrics["tasks_failed"] += 1

            # Clean up
            if task_id in self.current_tasks:
                del self.current_tasks[task_id]

            self.status = AgentStatus.IDLE if not self.current_tasks else AgentStatus.BUSY

            return {"success": False, "task_id": task_id, "error": str(e)}

    async def _cancel_task(self, task_id: str):
        """Cancel a running task."""
        if task_id in self.current_tasks:
            self.logger.info(f"Cancelling task {task_id}")
            del self.current_tasks[task_id]

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        uptime = None
        if self.metrics["uptime_start"]:
            uptime = (datetime.now() - self.metrics["uptime_start"]).total_seconds()

        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "current_tasks": len(self.current_tasks),
            "completed_tasks": len(self.completed_tasks),
            "metrics": {
                **self.metrics,
                "uptime_seconds": uptime,
                "avg_execution_time": (
                    self.metrics["total_execution_time"] / self.metrics["tasks_completed"]
                    if self.metrics["tasks_completed"] > 0
                    else 0
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }

    def can_handle_capability(self, capability: str) -> bool:
        """Check if this agent can handle a specific capability."""
        return capability in self.capabilities

    def __str__(self):
        return f"{self.agent_type.value}Agent({self.agent_id})"

    def __repr__(self):
        return self.__str__()
