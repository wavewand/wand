"""
Core types for the distributed MCP system.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentType(Enum):
    """Types of agents in the distributed system."""

    MANAGER = "manager"
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    DEVOPS = "devops"
    INTEGRATION = "integration"
    QA = "qa"
    HAYSTACK = "haystack"
    LLAMAINDEX = "llamaindex"


class TaskStatus(Enum):
    """Status of a task in the system."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Priority levels for tasks."""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class AgentStatus(Enum):
    """Status of an agent in the system."""

    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    IDLE = "idle"


@dataclass
class Task:
    """Represents a work item in the distributed system."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    type: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_status(self, status: TaskStatus) -> None:
        """Update task status and timestamp."""
        self.status = status
        self.updated_at = datetime.now()


@dataclass
class Agent:
    """Represents a worker agent in the distributed system."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: AgentType = AgentType.BACKEND
    capabilities: Dict[str, bool] = field(default_factory=dict)
    current_tasks: List[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    max_concurrent_tasks: int = 5
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    def can_take_task(self) -> bool:
        """Check if agent can accept a new task."""
        return (
            self.status in [AgentStatus.IDLE, AgentStatus.ONLINE]
            and len(self.current_tasks) < self.max_concurrent_tasks
        )

    def assign_task(self, task_id: str) -> None:
        """Assign a task to this agent."""
        if self.can_take_task():
            self.current_tasks.append(task_id)
            self.status = AgentStatus.BUSY

    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed for this agent."""
        if task_id in self.current_tasks:
            self.current_tasks.remove(task_id)
            if not self.current_tasks:
                self.status = AgentStatus.IDLE


@dataclass
class DistributedAgent:
    """Enhanced agent with logging and communication capabilities."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    capabilities: List[str] = field(default_factory=list)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def can_handle(self, capability: str) -> bool:
        """Check if agent can handle a specific capability."""
        return capability in self.capabilities


def generate_id() -> str:
    """Generate a unique ID for tasks and agents."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = str(uuid.uuid4())[:8]
    return f"{timestamp}-{random_suffix}"


# Capability mappings for different task types
CAPABILITY_MAPPINGS = {
    "frontend": ["react", "vue", "angular", "typescript", "css", "responsive_design"],
    "backend": ["python", "go", "nodejs", "api_design", "microservices"],
    "database": ["postgresql", "mysql", "mongodb", "redis", "database_design", "optimization"],
    "devops": ["aws", "docker", "kubernetes", "jenkins", "terraform", "monitoring"],
    "integration": ["slack", "git", "youtrack", "api_integration", "webhooks"],
    "manager": ["planning", "coordination", "reporting", "risk_assessment"],
    "qa": ["testing", "automation", "selenium", "pytest", "quality_assurance"],
    "haystack": [
        "rag",
        "document_search",
        "question_answering",
        "document_processing",
        "pipeline_management",
        "semantic_search",
        "summarization",
        "multimodal",
    ],
    "llamaindex": [
        "rag",
        "document_search",
        "question_answering",
        "document_processing",
        "index_management",
        "semantic_search",
        "summarization",
        "query_engines",
    ],
}


def get_default_capabilities(agent_type: AgentType) -> Dict[str, bool]:
    """Get default capabilities for an agent type."""
    type_name = agent_type.value
    capabilities = CAPABILITY_MAPPINGS.get(type_name, [])
    return {cap: True for cap in capabilities}
