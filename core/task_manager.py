#!/usr/bin/env python3
"""
Core Task Management

Handles distributed task creation, assignment, and tracking.
Extracted from distributed_server.py to be part of the core layer.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents available in the system"""

    MANAGER = "manager"
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    DEVOPS = "devops"
    INTEGRATION = "integration"
    HAYSTACK = "haystack"


class TaskStatus(Enum):
    """Status of tasks in the system"""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(Enum):
    """Priority levels for tasks"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Task:
    """Represents a task in the distributed system"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    task_type: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    """Represents an agent in the distributed system"""

    id: str
    agent_type: AgentType
    capabilities: List[str] = field(default_factory=list)
    current_tasks: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    status: str = "available"


class DistributedTaskManager:
    """Manages tasks and agents in the distributed system"""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.agents: Dict[str, Agent] = {}
        self.task_queue: List[str] = []
        self.completed_tasks: List[str] = []

        # Initialize default agents
        self._initialize_default_agents()

    def _initialize_default_agents(self):
        """Initialize default set of agents"""
        default_agents = [
            Agent("manager", AgentType.MANAGER, ["project_management", "coordination"]),
            Agent("frontend_1", AgentType.FRONTEND, ["react", "vue", "angular", "html", "css", "javascript"]),
            Agent("backend_1", AgentType.BACKEND, ["python", "node", "django", "fastapi", "flask"]),
            Agent("database", AgentType.DATABASE, ["postgresql", "mysql", "mongodb", "redis"]),
            Agent("devops", AgentType.DEVOPS, ["docker", "kubernetes", "ci_cd", "aws", "monitoring"]),
            Agent("integration", AgentType.INTEGRATION, ["api", "webhooks", "messaging", "etl"]),
            Agent("haystack", AgentType.HAYSTACK, ["ai", "ml", "nlp", "document_processing"]),
        ]

        for agent in default_agents:
            self.agents[agent.id] = agent
            logger.info(f"Registered agent: {agent.id} ({agent.agent_type.value})")

    def create_task(
        self, title: str, description: str, task_type: str, priority: TaskPriority = TaskPriority.MEDIUM
    ) -> Task:
        """Create a new task"""
        task = Task(title=title, description=description, task_type=task_type, priority=priority)

        self.tasks[task.id] = task
        self.task_queue.append(task.id)

        logger.info(f"Created task: {task.id} - {title}")
        return task

    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to an agent"""
        if task_id not in self.tasks or agent_id not in self.agents:
            return False

        task = self.tasks[task_id]
        agent = self.agents[agent_id]

        # Check if agent can take more tasks
        if len(agent.current_tasks) >= agent.max_concurrent_tasks:
            return False

        # Assign task
        task.assigned_to = agent_id
        task.status = TaskStatus.ASSIGNED
        task.updated_at = datetime.now()

        agent.current_tasks.append(task_id)

        # Remove from queue
        if task_id in self.task_queue:
            self.task_queue.remove(task_id)

        logger.info(f"Assigned task {task_id} to agent {agent_id}")
        return True

    def find_best_agent(self, task: Task) -> Optional[str]:
        """Find the best available agent for a task"""
        # Simple matching based on task type and agent type
        task_type_mapping = {
            "frontend": AgentType.FRONTEND,
            "backend": AgentType.BACKEND,
            "database": AgentType.DATABASE,
            "devops": AgentType.DEVOPS,
            "integration": AgentType.INTEGRATION,
            "haystack": AgentType.HAYSTACK,
            "ai": AgentType.HAYSTACK,
            "ml": AgentType.HAYSTACK,
        }

        preferred_type = task_type_mapping.get(task.task_type.lower(), AgentType.MANAGER)

        # Find available agents of preferred type
        available_agents = [
            agent
            for agent in self.agents.values()
            if (
                agent.agent_type == preferred_type
                and len(agent.current_tasks) < agent.max_concurrent_tasks
                and agent.status == "available"
            )
        ]

        # If no preferred type available, try any available agent
        if not available_agents:
            available_agents = [
                agent
                for agent in self.agents.values()
                if (len(agent.current_tasks) < agent.max_concurrent_tasks and agent.status == "available")
            ]

        # Return the agent with least current tasks
        if available_agents:
            best_agent = min(available_agents, key=lambda a: len(a.current_tasks))
            return best_agent.id

        return None

    def complete_task(self, task_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """Mark a task as completed"""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.updated_at = datetime.now()

        if result:
            task.metadata["result"] = result

        # Remove from agent's current tasks
        if task.assigned_to and task.assigned_to in self.agents:
            agent = self.agents[task.assigned_to]
            if task_id in agent.current_tasks:
                agent.current_tasks.remove(task_id)

        # Add to completed tasks
        self.completed_tasks.append(task_id)

        logger.info(f"Completed task: {task_id}")
        return True

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        return {
            "id": task.id,
            "title": task.title,
            "status": task.status.value,
            "assigned_to": task.assigned_to,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
        }

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent"""
        if agent_id not in self.agents:
            return None

        agent = self.agents[agent_id]
        return {
            "id": agent.id,
            "type": agent.agent_type.value,
            "status": agent.status,
            "current_tasks": len(agent.current_tasks),
            "max_tasks": agent.max_concurrent_tasks,
            "capabilities": agent.capabilities,
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "tasks": {
                "total": len(self.tasks),
                "pending": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
                "in_progress": len([t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS]),
                "completed": len(self.completed_tasks),
                "failed": len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED]),
            },
            "agents": {
                "total": len(self.agents),
                "available": len([a for a in self.agents.values() if a.status == "available"]),
                "busy": len([a for a in self.agents.values() if len(a.current_tasks) > 0]),
            },
            "queue_size": len(self.task_queue),
        }

    def get_project_status(self) -> Dict[str, Any]:
        """Get project status - placeholder for future project management"""
        return {"projects": {}, "total_projects": 0, "active_projects": 0}
