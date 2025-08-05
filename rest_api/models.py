"""
Pydantic models for REST API requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskPriorityModel(str, Enum):
    """Task priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatusModel(str, Enum):
    """Task status values."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class AgentTypeModel(str, Enum):
    """Agent types."""

    MANAGER = "manager"
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    DEVOPS = "devops"
    INTEGRATION = "integration"
    QA = "qa"


class AgentStatusModel(str, Enum):
    """Agent status values."""

    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    IDLE = "idle"


# Request Models
class TaskCreateModel(BaseModel):
    """Model for creating a new task."""

    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Task description")
    type: str = Field(..., description="Task type (frontend, backend, etc.)")
    priority: TaskPriorityModel = Field(TaskPriorityModel.MEDIUM, description="Task priority")
    dependencies: List[str] = Field(default_factory=list, description="List of dependency task IDs")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional task metadata")


class ProjectCreateModel(BaseModel):
    """Model for creating a new project."""

    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    requirements: str = Field(..., description="Project requirements")
    tech_stack: Dict[str, str] = Field(default_factory=dict, description="Technology stack")
    priority: TaskPriorityModel = Field(TaskPriorityModel.MEDIUM, description="Project priority")


class TaskUpdateModel(BaseModel):
    """Model for updating a task."""

    status: Optional[TaskStatusModel] = Field(None, description="New task status")
    metadata: Optional[Dict[str, str]] = Field(None, description="Updated metadata")


class SlackOperationModel(BaseModel):
    """Model for Slack operations."""

    operation: str = Field(..., description="Slack operation to perform")
    parameters: Dict[str, str] = Field(default_factory=dict, description="Operation parameters")


class GitOperationModel(BaseModel):
    """Model for Git operations."""

    operation: str = Field(..., description="Git operation to perform")
    parameters: Dict[str, str] = Field(default_factory=dict, description="Operation parameters")


class AWSOperationModel(BaseModel):
    """Model for AWS operations."""

    service: str = Field(..., description="AWS service (ec2, s3, lambda)")
    operation: str = Field(..., description="AWS operation to perform")
    parameters: Dict[str, str] = Field(default_factory=dict, description="Operation parameters")


class JenkinsOperationModel(BaseModel):
    """Model for Jenkins operations."""

    operation: str = Field(..., description="Jenkins operation to perform")
    parameters: Dict[str, str] = Field(default_factory=dict, description="Operation parameters")


class YouTrackOperationModel(BaseModel):
    """Model for YouTrack operations."""

    operation: str = Field(..., description="YouTrack operation to perform")
    parameters: Dict[str, str] = Field(default_factory=dict, description="Operation parameters")


# Response Models
class TaskModel(BaseModel):
    """Task response model."""

    id: str
    title: str
    description: str
    type: str
    priority: TaskPriorityModel
    status: TaskStatusModel
    assigned_to: Optional[str] = None
    dependencies: List[str] = []
    subtasks: List[str] = []
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, str] = {}


class AgentModel(BaseModel):
    """Agent response model."""

    id: str
    name: str
    type: AgentTypeModel
    capabilities: Dict[str, bool] = {}
    current_tasks: List[str] = []
    status: AgentStatusModel
    max_concurrent_tasks: int
    performance_metrics: Dict[str, str] = {}
    port: int


class ProjectModel(BaseModel):
    """Project response model."""

    id: str
    name: str
    description: str
    requirements: str
    tech_stack: Dict[str, str] = {}
    priority: TaskPriorityModel
    created_at: datetime
    task_ids: List[str] = []


class TaskResponseModel(BaseModel):
    """Generic task operation response."""

    task_id: str
    status: str
    message: str
    timestamp: datetime


class ProjectResponseModel(BaseModel):
    """Project creation response."""

    project_id: str
    tasks_created: int
    tasks_distributed: int
    task_ids: List[str] = []


class ProjectStatusModel(BaseModel):
    """Project status response."""

    project_id: str
    status: str
    tasks: List[TaskModel] = []
    completion_percentage: float
    task_status_counts: Dict[str, int] = {}


class SystemStatusModel(BaseModel):
    """System status response."""

    total_agents: int
    active_agents: int
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    agents: Dict[str, AgentModel] = {}
    task_counts_by_status: Dict[str, int] = {}


class IntegrationResponseModel(BaseModel):
    """Integration operation response."""

    success: bool
    message: str
    result_data: Dict[str, str] = {}
    timestamp: datetime


class IntegrationStatusModel(BaseModel):
    """Integration status response."""

    integration_name: str
    status: str
    message: str
    last_check: datetime
    metrics: Dict[str, str] = {}


class ErrorResponseModel(BaseModel):
    """Error response model."""

    error: str
    message: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthCheckModel(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    version: str = "3.0.0"
    services: Dict[str, str] = {}


# Pagination Models
class PaginatedTasksModel(BaseModel):
    """Paginated tasks response."""

    tasks: List[TaskModel]
    total_count: int
    page: int
    page_size: int
    total_pages: int


class PaginatedAgentsModel(BaseModel):
    """Paginated agents response."""

    agents: List[AgentModel]
    total_count: int
    page: int
    page_size: int
    total_pages: int
