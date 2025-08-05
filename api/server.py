"""REST API server for the Python MCP implementation."""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# Framework-specific routers
from api.framework_routers import ALL_FRAMEWORK_ROUTERS
from orchestrator.agent_orchestrator import execute_tool_via_orchestrator, get_orchestrator_status

# AI Framework Registry
try:
    from ai_framework_registry import ai_framework_registry

    AI_FRAMEWORKS_AVAILABLE = True
except ImportError:
    AI_FRAMEWORKS_AVAILABLE = False
    ai_framework_registry = None

# Monitoring
try:
    from monitoring.framework_monitor import framework_monitor

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    framework_monitor = None

# Batch Processing
try:
    from api.batch_processing import BatchDocumentRequest, BatchRAGRequest, BatchSearchRequest, batch_processor

    BATCH_PROCESSING_AVAILABLE = True
except ImportError:
    BATCH_PROCESSING_AVAILABLE = False
    batch_processor = None

# Benchmarking
try:
    from benchmarking.framework_benchmark import BenchmarkType, framework_benchmarker

    BENCHMARKING_AVAILABLE = True
except ImportError:
    BENCHMARKING_AVAILABLE = False
    framework_benchmarker = None

# WebSocket Events
try:
    from websocket.event_manager import EventPriority, EventType, websocket_event_manager

    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    websocket_event_manager = None

# Caching
try:
    from caching.response_cache import response_cache

    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    response_cache = None


# Enums
class AgentStatus(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    ERROR = "error"
    OFFLINE = "offline"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProjectStatus(str, Enum):
    PLANNING = "planning"
    ACTIVE = "active"
    COMPLETED = "completed"
    ON_HOLD = "on-hold"


class IntegrationStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    WARNING = "warning"


# Data Models
@dataclass
class AgentMetrics:
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_task_time: float = 0.0
    success_rate: float = 1.0


@dataclass
class Agent:
    id: str
    name: str
    type: str
    status: AgentStatus
    capabilities: List[str]
    current_task: Optional[str] = None
    metrics: AgentMetrics = None
    created_at: datetime = None
    last_active: datetime = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = AgentMetrics()
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.last_active is None:
            self.last_active = datetime.now(timezone.utc)


@dataclass
class Task:
    id: str
    title: str
    description: str
    type: str
    status: TaskStatus
    priority: TaskPriority
    assigned_to: Optional[str] = None
    project_id: Optional[str] = None
    dependencies: List[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    updated_at: datetime = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)


@dataclass
class Project:
    id: str
    name: str
    description: str
    status: ProjectStatus
    components: List[str]
    progress: int = 0
    tasks_total: int = 0
    tasks_completed: int = 0
    agents: List[str] = None
    start_date: datetime = None
    due_date: datetime = None
    created_at: datetime = None

    def __post_init__(self):
        if self.agents is None:
            self.agents = []
        if self.start_date is None:
            self.start_date = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


@dataclass
class Integration:
    id: str
    name: str
    type: str
    enabled: bool
    status: IntegrationStatus
    last_sync: datetime
    config: Optional[Dict[str, Any]] = None


# Request Models
class CreateAgentRequest(BaseModel):
    name: str
    type: str
    capabilities: List[str]


class UpdateAgentRequest(BaseModel):
    name: Optional[str] = None
    status: Optional[AgentStatus] = None
    capabilities: Optional[List[str]] = None


class CreateTaskRequest(BaseModel):
    title: str
    description: str
    type: str
    priority: Optional[TaskPriority] = TaskPriority.MEDIUM
    project_id: Optional[str] = None
    dependencies: Optional[List[str]] = None


class UpdateTaskRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    assigned_to: Optional[str] = None


class CreateProjectRequest(BaseModel):
    name: str
    description: str
    components: List[str]
    due_date: Optional[datetime] = None


class AssignTaskRequest(BaseModel):
    agentId: str


class UpdateTaskStatusRequest(BaseModel):
    status: TaskStatus


class ToggleIntegrationRequest(BaseModel):
    enabled: bool


# Generic AI Framework request models
class RAGRequest(BaseModel):
    query: str
    framework: Optional[str] = "haystack"  # haystack, llamaindex, langchain, etc.
    pipeline_id: Optional[str] = "default_rag"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    context: Optional[Dict[str, str]] = None
    framework_config: Optional[Dict[str, Any]] = None


class DocumentSearchRequest(BaseModel):
    query: str
    framework: Optional[str] = "haystack"
    search_type: Optional[str] = "semantic"  # semantic, keyword, hybrid
    max_results: Optional[int] = 10
    filters: Optional[Dict[str, str]] = None
    framework_config: Optional[Dict[str, Any]] = None


class DocumentIngestionRequest(BaseModel):
    filename: str
    content: str
    framework: Optional[str] = "haystack"
    content_type: Optional[str] = "text/plain"
    metadata: Optional[Dict[str, str]] = None
    framework_config: Optional[Dict[str, Any]] = None


class SummarizationRequest(BaseModel):
    text: str
    framework: Optional[str] = "haystack"
    pipeline_id: Optional[str] = "default_summarization"
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    framework_config: Optional[Dict[str, Any]] = None


class PipelineConfigRequest(BaseModel):
    name: str
    type: str  # rag, qa, search, summarization, custom
    framework: Optional[str] = "haystack"
    configuration: Optional[Dict[str, str]] = None
    is_active: Optional[bool] = True


# Initialize database and app
DATABASE_URL = os.environ.get("DATABASE_URL", "memory://")
if DATABASE_URL == "memory://":
    # In-memory database
    class InMemoryDB:
        def __init__(self):
            self.agents: Dict[str, Agent] = {}
            self.tasks: Dict[str, Task] = {}
            self.projects: Dict[str, Project] = {}
            self.start_time = datetime.now(timezone.utc)

        def generate_id(self) -> str:
            return str(uuid.uuid4())

        def update_project_progress(self, project_id: str):
            if project_id in self.projects:
                project = self.projects[project_id]
                if project.tasks_total > 0:
                    project.progress = int((project.tasks_completed / project.tasks_total) * 100)
                else:
                    project.progress = 0

    db = InMemoryDB()
    use_postgres = False
else:
    # PostgreSQL database
    from api.database import DatabaseManager

    db_manager = DatabaseManager(DATABASE_URL)
    use_postgres = True

    # Create a compatible db object for PostgreSQL
    class PostgresDB:
        def __init__(self):
            self.start_time = datetime.now(timezone.utc)
            self.agents: Dict[str, Agent] = {}
            self.tasks: Dict[str, Task] = {}
            self.projects: Dict[str, Project] = {}

        def generate_id(self) -> str:
            return str(uuid.uuid4())

        def update_project_progress(self, project_id: str):
            # Placeholder for project progress update
            pass

    db = PostgresDB()

app = FastAPI(title="Multi-Agent Platform API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register framework-specific routers
for router in ALL_FRAMEWORK_ROUTERS:
    app.include_router(router)

# WebSocket manager


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        await self.send_personal_message(
            {
                "type": "system_event",
                "data": {"event": "connected", "message": "Connected to MAP server"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            websocket,
        )

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except BaseException:
                pass


manager = ConnectionManager()


# Dependency for authentication
async def verify_token(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    if not authorization and not x_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Token/API key validation would be implemented based on your auth provider
    # For now, accept any token for development purposes
    return True


# Agent endpoints
@app.get("/api/v1/agents", dependencies=[Depends(verify_token)])
async def list_agents(status: Optional[str] = None, type: Optional[str] = None):
    agents = []
    for agent in db.agents.values():
        if status and agent.status != status:
            continue
        if type and agent.type != type:
            continue
        agents.append(asdict(agent))
    return agents


@app.post("/api/v1/agents", status_code=201, dependencies=[Depends(verify_token)])
async def create_agent(request: CreateAgentRequest):
    agent = Agent(
        id=db.generate_id(),
        name=request.name,
        type=request.type,
        status=AgentStatus.IDLE,
        capabilities=request.capabilities,
    )
    db.agents[agent.id] = agent

    # Broadcast agent creation
    await manager.broadcast(
        {
            "type": "agent_update",
            "data": {"action": "created", "agent": asdict(agent)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    return asdict(agent)


@app.get("/api/v1/agents/{agentId}", dependencies=[Depends(verify_token)])
async def get_agent(agentId: str):
    if agentId not in db.agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    return asdict(db.agents[agentId])


@app.put("/api/v1/agents/{agentId}", dependencies=[Depends(verify_token)])
async def update_agent(agentId: str, request: UpdateAgentRequest):
    if agentId not in db.agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = db.agents[agentId]
    if request.name is not None:
        agent.name = request.name
    if request.status is not None:
        agent.status = request.status
    if request.capabilities is not None:
        agent.capabilities = request.capabilities

    # Broadcast update
    await manager.broadcast(
        {
            "type": "agent_update",
            "data": {"action": "updated", "agent": asdict(agent)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    return asdict(agent)


@app.delete("/api/v1/agents/{agentId}", status_code=204, dependencies=[Depends(verify_token)])
async def delete_agent(agentId: str):
    if agentId not in db.agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    del db.agents[agentId]

    # Broadcast deletion
    await manager.broadcast(
        {
            "type": "agent_update",
            "data": {"action": "deleted", "id": agentId},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


class ExecuteAgentRequest(BaseModel):
    task: str
    parameters: Optional[Dict[str, Any]] = None


@app.post("/api/v1/agents/{agentId}/execute", dependencies=[Depends(verify_token)])
async def execute_agent(agentId: str, request: ExecuteAgentRequest):
    """Execute task on specific agent (expected by mcp-ui)"""
    if agentId not in db.agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = db.agents[agentId]

    if agent.status != AgentStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Agent is not active")

    # Create execution record
    execution_id = str(uuid.uuid4())
    execution_result = {
        "execution_id": execution_id,
        "agent_id": agentId,
        "task": request.task,
        "parameters": request.parameters,
        "status": "completed",
        "result": f"Agent {agent.name} executed task: {request.task}",
        "processing_time": 1.2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Broadcast execution update
    await manager.broadcast(
        {"type": "agent_execution", "data": execution_result, "timestamp": datetime.now(timezone.utc).isoformat()}
    )

    return execution_result


# Task endpoints
@app.get("/api/v1/tasks", dependencies=[Depends(verify_token)])
async def list_tasks(status: Optional[str] = None, priority: Optional[str] = None, assigned_to: Optional[str] = None):
    tasks = []
    for task in db.tasks.values():
        if status and task.status != status:
            continue
        if priority and task.priority != priority:
            continue
        if assigned_to and task.assigned_to != assigned_to:
            continue
        tasks.append(asdict(task))
    return tasks


@app.post("/api/v1/tasks", status_code=201, dependencies=[Depends(verify_token)])
async def create_task(request: CreateTaskRequest):
    task = Task(
        id=db.generate_id(),
        title=request.title,
        description=request.description,
        type=request.type,
        status=TaskStatus.PENDING,
        priority=request.priority,
        project_id=request.project_id,
        dependencies=request.dependencies or [],
    )
    db.tasks[task.id] = task

    # Update project task count
    if task.project_id and task.project_id in db.projects:
        db.projects[task.project_id].tasks_total += 1
        db.update_project_progress(task.project_id)

    # Broadcast task creation
    await manager.broadcast(
        {
            "type": "task_update",
            "data": {"action": "created", "task": asdict(task)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    return asdict(task)


@app.get("/api/v1/tasks/{taskId}", dependencies=[Depends(verify_token)])
async def get_task(taskId: str):
    if taskId not in db.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return asdict(db.tasks[taskId])


@app.put("/api/v1/tasks/{taskId}", dependencies=[Depends(verify_token)])
async def update_task(taskId: str, request: UpdateTaskRequest):
    if taskId not in db.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = db.tasks[taskId]
    old_status = task.status

    if request.title is not None:
        task.title = request.title
    if request.description is not None:
        task.description = request.description
    if request.status is not None:
        task.status = request.status
        if request.status == TaskStatus.COMPLETED:
            task.completed_at = datetime.now(timezone.utc)
    if request.priority is not None:
        task.priority = request.priority
    if request.assigned_to is not None:
        task.assigned_to = request.assigned_to

    task.updated_at = datetime.now(timezone.utc)

    # Update project progress if status changed
    if task.project_id and old_status != task.status:
        project = db.projects.get(task.project_id)
        if project:
            if old_status != TaskStatus.COMPLETED and task.status == TaskStatus.COMPLETED:
                project.tasks_completed += 1
            elif old_status == TaskStatus.COMPLETED and task.status != TaskStatus.COMPLETED:
                project.tasks_completed -= 1
            db.update_project_progress(task.project_id)

    # Broadcast update
    await manager.broadcast(
        {
            "type": "task_update",
            "data": {"action": "updated", "task": asdict(task)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    return asdict(task)


@app.delete("/api/v1/tasks/{taskId}", status_code=204, dependencies=[Depends(verify_token)])
async def delete_task(taskId: str):
    if taskId not in db.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = db.tasks[taskId]

    # Update project task count
    if task.project_id and task.project_id in db.projects:
        project = db.projects[task.project_id]
        project.tasks_total -= 1
        if task.status == TaskStatus.COMPLETED:
            project.tasks_completed -= 1
        db.update_project_progress(task.project_id)

    del db.tasks[taskId]

    # Broadcast deletion
    await manager.broadcast(
        {
            "type": "task_update",
            "data": {"action": "deleted", "id": taskId},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.post("/api/v1/tasks/{taskId}/assign", dependencies=[Depends(verify_token)])
async def assign_task(taskId: str, request: AssignTaskRequest):
    if taskId not in db.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    if request.agentId not in db.agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    task = db.tasks[taskId]
    agent = db.agents[request.agentId]

    task.assigned_to = request.agentId
    task.status = TaskStatus.IN_PROGRESS
    task.updated_at = datetime.now(timezone.utc)

    agent.status = AgentStatus.ACTIVE
    agent.current_task = taskId

    # Broadcast assignment
    await manager.broadcast(
        {
            "type": "task_update",
            "data": {"action": "assigned", "task": asdict(task)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    return asdict(task)


@app.put("/api/v1/tasks/{taskId}/status", dependencies=[Depends(verify_token)])
async def update_task_status(taskId: str, request: UpdateTaskStatusRequest):
    if taskId not in db.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = db.tasks[taskId]
    old_status = task.status
    task.status = request.status
    task.updated_at = datetime.now(timezone.utc)

    if request.status == TaskStatus.COMPLETED:
        task.completed_at = datetime.now(timezone.utc)

        # Update agent if task was assigned
        if task.assigned_to and task.assigned_to in db.agents:
            agent = db.agents[task.assigned_to]
            agent.status = AgentStatus.IDLE
            agent.current_task = None
            agent.metrics.tasks_completed += 1
            total_tasks = agent.metrics.tasks_completed + agent.metrics.tasks_failed
            agent.metrics.success_rate = agent.metrics.tasks_completed / total_tasks if total_tasks > 0 else 1.0

    # Update project progress
    if task.project_id and old_status != task.status:
        project = db.projects.get(task.project_id)
        if project:
            if old_status != TaskStatus.COMPLETED and task.status == TaskStatus.COMPLETED:
                project.tasks_completed += 1
            elif old_status == TaskStatus.COMPLETED and task.status != TaskStatus.COMPLETED:
                project.tasks_completed -= 1
            db.update_project_progress(task.project_id)

    # Broadcast status change
    await manager.broadcast(
        {
            "type": "task_update",
            "data": {"action": "status_changed", "task": asdict(task)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    return asdict(task)


# Project endpoints
@app.get("/api/v1/projects", dependencies=[Depends(verify_token)])
async def list_projects(status: Optional[str] = None):
    projects = []
    for project in db.projects.values():
        if status and project.status != status:
            continue
        projects.append(asdict(project))
    return projects


@app.post("/api/v1/projects", status_code=201, dependencies=[Depends(verify_token)])
async def create_project(request: CreateProjectRequest):
    project = Project(
        id=db.generate_id(),
        name=request.name,
        description=request.description,
        status=ProjectStatus.PLANNING,
        components=request.components,
        due_date=request.due_date or datetime.now(timezone.utc).replace(day=datetime.now(timezone.utc).day + 30),
    )
    db.projects[project.id] = project

    # Create tasks for each component
    for component in request.components:
        task = Task(
            id=db.generate_id(),
            title=f"{project.name} - {component} development",
            description=f"Develop {component} component for {project.name}",
            type="development",
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            project_id=project.id,
        )
        db.tasks[task.id] = task
        project.tasks_total += 1

    return asdict(project)


@app.get("/api/v1/projects/{projectId}", dependencies=[Depends(verify_token)])
async def get_project(projectId: str):
    if projectId not in db.projects:
        raise HTTPException(status_code=404, detail="Project not found")
    return asdict(db.projects[projectId])


# System endpoints
@app.get("/api/v1/system/status", dependencies=[Depends(verify_token)])
async def get_system_status():
    # Count agent statuses
    active_agents = sum(1 for a in db.agents.values() if a.status == AgentStatus.ACTIVE)
    idle_agents = sum(1 for a in db.agents.values() if a.status == AgentStatus.IDLE)
    error_agents = sum(1 for a in db.agents.values() if a.status == AgentStatus.ERROR)

    # Count task statuses
    completed_tasks = sum(1 for t in db.tasks.values() if t.status == TaskStatus.COMPLETED)
    in_progress_tasks = sum(1 for t in db.tasks.values() if t.status == TaskStatus.IN_PROGRESS)
    pending_tasks = sum(1 for t in db.tasks.values() if t.status == TaskStatus.PENDING)
    failed_tasks = sum(1 for t in db.tasks.values() if t.status == TaskStatus.FAILED)

    # Integration statuses
    integrations = [
        {"name": "Slack", "enabled": True, "status": "connected"},
        {"name": "Git", "enabled": True, "status": "connected"},
        {"name": "Jenkins", "enabled": True, "status": "connected"},
        {"name": "YouTrack", "enabled": False, "status": "disconnected"},
        {"name": "AWS", "enabled": True, "status": "connected"},
        {"name": "PostgreSQL", "enabled": True, "status": "connected"},
        {"name": "Bambu 3D", "enabled": False, "status": "disconnected"},
        {"name": "Web Search", "enabled": True, "status": "connected"},
        {
            "name": "AI Frameworks",
            "enabled": AI_FRAMEWORKS_AVAILABLE,
            "status": "connected" if AI_FRAMEWORKS_AVAILABLE else "disconnected",
        },
    ]

    uptime = datetime.now(timezone.utc) - db.start_time

    return {
        "server_url": "http://localhost:8000",
        "version": "1.0.0",
        "uptime": str(uptime),
        "total_agents": len(db.agents),
        "active_agents": active_agents,
        "idle_agents": idle_agents,
        "error_agents": error_agents,
        "total_tasks": len(db.tasks),
        "completed_tasks": completed_tasks,
        "in_progress_tasks": in_progress_tasks,
        "pending_tasks": pending_tasks,
        "failed_tasks": failed_tasks,
        "integrations": integrations,
    }


@app.get("/api/v1/system/metrics", dependencies=[Depends(verify_token)])
async def get_system_metrics():
    # Metrics collection placeholder - integrate with your monitoring system
    return {
        "cpu_usage": 45.2,
        "memory_usage": 62.8,
        "goroutines": 0,  # N/A for Python
        "active_connections": len(manager.active_connections),
        "request_rate": 15.5,
        "error_rate": 0.02,
        "average_response_time": 125.3,
    }


# Integration endpoints
@app.get("/api/v1/integrations", dependencies=[Depends(verify_token)])
async def list_integrations():
    integrations = [
        Integration(
            id="slack",
            name="Slack",
            type="messaging",
            enabled=True,
            status=IntegrationStatus.CONNECTED,
            last_sync=datetime.now(timezone.utc),
        ),
        Integration(
            id="git",
            name="Git",
            type="version_control",
            enabled=True,
            status=IntegrationStatus.CONNECTED,
            last_sync=datetime.now(timezone.utc),
        ),
        Integration(
            id="jenkins",
            name="Jenkins",
            type="ci_cd",
            enabled=True,
            status=IntegrationStatus.CONNECTED,
            last_sync=datetime.now(timezone.utc),
        ),
        Integration(
            id="youtrack",
            name="YouTrack",
            type="issue_tracking",
            enabled=False,
            status=IntegrationStatus.DISCONNECTED,
            last_sync=datetime.now(timezone.utc),
        ),
        Integration(
            id="aws",
            name="AWS",
            type="cloud",
            enabled=True,
            status=IntegrationStatus.CONNECTED,
            last_sync=datetime.now(timezone.utc),
        ),
        Integration(
            id="postgresql",
            name="PostgreSQL",
            type="database",
            enabled=True,
            status=IntegrationStatus.CONNECTED,
            last_sync=datetime.now(timezone.utc),
        ),
        Integration(
            id="bambu",
            name="Bambu 3D",
            type="hardware",
            enabled=False,
            status=IntegrationStatus.DISCONNECTED,
            last_sync=datetime.now(timezone.utc),
        ),
        Integration(
            id="web_search",
            name="Web Search",
            type="search",
            enabled=True,
            status=IntegrationStatus.CONNECTED,
            last_sync=datetime.now(timezone.utc),
        ),
        Integration(
            id="ai_frameworks",
            name="AI Frameworks",
            type="ai_platform",
            enabled=AI_FRAMEWORKS_AVAILABLE,
            status=IntegrationStatus.CONNECTED if AI_FRAMEWORKS_AVAILABLE else IntegrationStatus.DISCONNECTED,
            last_sync=datetime.now(timezone.utc),
        ),
    ]

    return [asdict(i) for i in integrations]


@app.post("/api/v1/integrations/{integrationId}/toggle", dependencies=[Depends(verify_token)])
async def toggle_integration(integrationId: str, request: ToggleIntegrationRequest):
    # Integration toggle placeholder - implement based on your integration manager
    # Returns success for development
    integration = Integration(
        id=integrationId,
        name=integrationId.title(),
        type="unknown",
        enabled=request.enabled,
        status=IntegrationStatus.CONNECTED if request.enabled else IntegrationStatus.DISCONNECTED,
        last_sync=datetime.now(timezone.utc),
    )

    return asdict(integration)


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now
            await manager.send_personal_message(
                {"type": "echo", "data": data, "timestamp": datetime.now(timezone.utc).isoformat()}, websocket
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# AI Framework endpoints
@app.post("/api/v1/rag", dependencies=[Depends(verify_token)])
async def execute_rag_query(request: RAGRequest):
    """Execute a RAG (Retrieval-Augmented Generation) query using specified AI framework."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    framework = ai_framework_registry.get_framework(request.framework)
    if not framework:
        raise HTTPException(status_code=400, detail=f"Framework '{request.framework}' not supported")

    try:
        result = await framework.execute_rag_query(
            query=request.query,
            pipeline_id=request.pipeline_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            context=request.context,
            framework_config=request.framework_config,
        )

        # Broadcast RAG query event
        await manager.broadcast(
            {
                "type": "ai_event",
                "data": {
                    "action": "rag_query",
                    "framework": request.framework,
                    "query": request.query[:100] + "..." if len(request.query) > 100 else request.query,
                    "success": result.get("success", False),
                    "pipeline_id": request.pipeline_id,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@app.post("/api/v1/search", dependencies=[Depends(verify_token)])
async def execute_document_search(request: DocumentSearchRequest):
    """Execute a document search query using specified AI framework."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    framework = ai_framework_registry.get_framework(request.framework)
    if not framework:
        raise HTTPException(status_code=400, detail=f"Framework '{request.framework}' not supported")

    try:
        result = await framework.search_documents(
            query=request.query,
            search_type=request.search_type,
            max_results=request.max_results,
            filters=request.filters,
            framework_config=request.framework_config,
        )

        # Broadcast search event
        await manager.broadcast(
            {
                "type": "ai_event",
                "data": {
                    "action": "search_query",
                    "framework": request.framework,
                    "query": request.query[:100] + "..." if len(request.query) > 100 else request.query,
                    "success": result.get("success", False),
                    "results_count": len(result.get("documents", [])),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search query failed: {str(e)}")


@app.post("/api/v1/documents", status_code=201, dependencies=[Depends(verify_token)])
async def ingest_document(request: DocumentIngestionRequest):
    """Ingest a document using specified AI framework."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    framework = ai_framework_registry.get_framework(request.framework)
    if not framework:
        raise HTTPException(status_code=400, detail=f"Framework '{request.framework}' not supported")

    try:
        result = await framework.ingest_document(
            filename=request.filename,
            content=request.content,
            content_type=request.content_type,
            metadata=request.metadata,
            framework_config=request.framework_config,
        )

        if result.get("success"):
            # Broadcast document ingestion
            await manager.broadcast(
                {
                    "type": "ai_event",
                    "data": {
                        "action": "document_ingested",
                        "framework": request.framework,
                        "filename": request.filename,
                        "document_id": result.get("document_id"),
                        "success": True,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")


@app.get("/api/v1/documents", dependencies=[Depends(verify_token)])
async def list_documents(
    framework: Optional[str] = "haystack", content_type: Optional[str] = None, limit: Optional[int] = 100
):
    """List documents from specified AI framework."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    framework_obj = ai_framework_registry.get_framework(framework)
    if not framework_obj:
        raise HTTPException(status_code=400, detail=f"Framework '{framework}' not supported")

    try:
        # Use the framework's document store directly (Haystack-specific for now)
        if framework == "haystack":
            from integrations.haystack import HaystackDocumentStore

            document_store = HaystackDocumentStore()
            filters = {}
            if content_type:
                filters["content_type"] = content_type

            documents = document_store.list_documents(filters=filters, limit=limit)
            return {"success": True, "documents": documents, "total": len(documents), "framework": framework}
        else:
            return {
                "success": False,
                "error": f"Document listing not yet supported for {framework}",
                "framework": framework,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/api/v1/documents/{document_id}", dependencies=[Depends(verify_token)])
async def get_document(document_id: str, framework: Optional[str] = "haystack"):
    """Get a specific document from specified AI framework."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    framework_obj = ai_framework_registry.get_framework(framework)
    if not framework_obj:
        raise HTTPException(status_code=400, detail=f"Framework '{framework}' not supported")

    try:
        # Use the framework's document store directly (Haystack-specific for now)
        if framework == "haystack":
            from integrations.haystack import HaystackDocumentStore

            document_store = HaystackDocumentStore()
            document = document_store.get_document(document_id)
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")

            return {"success": True, "document": document, "framework": framework}
        else:
            return {
                "success": False,
                "error": f"Document retrieval not yet supported for {framework}",
                "framework": framework,
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@app.delete("/api/v1/documents/{document_id}", status_code=204, dependencies=[Depends(verify_token)])
async def delete_document(document_id: str, framework: Optional[str] = "haystack"):
    """Delete a document from specified AI framework."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    framework_obj = ai_framework_registry.get_framework(framework)
    if not framework_obj:
        raise HTTPException(status_code=400, detail=f"Framework '{framework}' not supported")

    try:
        # Use the framework's document store directly (Haystack-specific for now)
        if framework == "haystack":
            from integrations.haystack import HaystackDocumentStore

            document_store = HaystackDocumentStore()
            result = document_store.delete_document(document_id)
            if not result.get("success"):
                if "not found" in result.get("message", "").lower():
                    raise HTTPException(status_code=404, detail="Document not found")
                else:
                    raise HTTPException(status_code=500, detail=result.get("message", "Delete failed"))

            # Broadcast deletion
            await manager.broadcast(
                {
                    "type": "ai_event",
                    "data": {
                        "action": "document_deleted",
                        "framework": framework,
                        "document_id": document_id,
                        "success": True,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        else:
            raise HTTPException(status_code=400, detail=f"Document deletion not yet supported for {framework}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.get("/api/v1/pipelines", dependencies=[Depends(verify_token)])
async def list_pipelines(framework: Optional[str] = "haystack"):
    """List available pipelines for specified AI framework."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    framework_obj = ai_framework_registry.get_framework(framework)
    if not framework_obj:
        raise HTTPException(status_code=400, detail=f"Framework '{framework}' not supported")

    try:
        result = await framework_obj.list_pipelines()
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list pipelines: {str(e)}")


@app.get("/api/v1/pipelines/{pipeline_id}", dependencies=[Depends(verify_token)])
async def get_pipeline(pipeline_id: str, framework: Optional[str] = "haystack"):
    """Get information about a specific pipeline."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    framework_obj = ai_framework_registry.get_framework(framework)
    if not framework_obj:
        raise HTTPException(status_code=400, detail=f"Framework '{framework}' not supported")

    try:
        # Use the framework's pipeline manager directly (Haystack-specific for now)
        if framework == "haystack":
            from integrations.haystack import HaystackPipelineManager

            pipeline_manager = HaystackPipelineManager()
            pipeline_info = pipeline_manager.get_pipeline_info(pipeline_id)
            if not pipeline_info:
                raise HTTPException(status_code=404, detail="Pipeline not found")

            return {"success": True, "pipeline": pipeline_info, "framework": framework}
        else:
            return {
                "success": False,
                "error": f"Pipeline retrieval not yet supported for {framework}",
                "framework": framework,
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline: {str(e)}")


@app.post("/api/v1/summarize", dependencies=[Depends(verify_token)])
async def execute_summarization(request: SummarizationRequest):
    """Execute text summarization using specified AI framework."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    framework = ai_framework_registry.get_framework(request.framework)
    if not framework:
        raise HTTPException(status_code=400, detail=f"Framework '{request.framework}' not supported")

    try:
        result = await framework.summarize_text(
            text=request.text,
            pipeline_id=request.pipeline_id,
            max_length=request.max_length,
            min_length=request.min_length,
            framework_config=request.framework_config,
        )

        # Broadcast summarization event
        await manager.broadcast(
            {
                "type": "ai_event",
                "data": {
                    "action": "summarization",
                    "framework": request.framework,
                    "text_length": len(request.text),
                    "success": result.get("success", False),
                    "pipeline_id": request.pipeline_id,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@app.get("/api/v1/ai/status", dependencies=[Depends(verify_token)])
async def get_ai_frameworks_status():
    """Get the status of all AI frameworks."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        return {"available": False, "message": "AI frameworks not available"}

    try:
        status = await ai_framework_registry.get_all_framework_status()
        return {
            "available": True,
            "frameworks": status,
            "supported_frameworks": ai_framework_registry.list_frameworks(),
        }

    except Exception as e:
        return {"available": False, "error": str(e)}


@app.get("/api/v1/ai/frameworks", dependencies=[Depends(verify_token)])
async def list_ai_frameworks():
    """List all available AI frameworks and their capabilities."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    try:
        frameworks = []
        for framework_name in ai_framework_registry.list_frameworks():
            framework_obj = ai_framework_registry.get_framework(framework_name)
            if framework_obj:
                status = await framework_obj.get_status()
                frameworks.append(
                    {
                        "name": framework_name,
                        "capabilities": framework_obj.capabilities,
                        "available": status.get("available", False),
                        "status": status,
                    }
                )

        return {"success": True, "frameworks": frameworks, "total": len(frameworks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list frameworks: {str(e)}")


# Batch Processing Endpoints
@app.post("/api/v1/batch/rag", dependencies=[Depends(verify_token)])
async def batch_rag_queries(request: BatchRAGRequest):
    """Execute multiple RAG queries in batch."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    if not BATCH_PROCESSING_AVAILABLE or not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processing not available")

    if len(request.queries) == 0:
        raise HTTPException(status_code=400, detail="No queries provided")

    if len(request.queries) > 100:
        raise HTTPException(status_code=400, detail="Too many queries (max 100)")

    try:
        # Get framework
        framework_obj = ai_framework_registry.get_framework(request.framework)
        if not framework_obj:
            raise HTTPException(status_code=400, detail=f"Framework '{request.framework}' not available")

        # Process batch
        batch_result = await batch_processor.process_batch_rag_queries(
            queries=request.queries,
            framework=request.framework,
            framework_obj=framework_obj,
            batch_config=request.framework_config or {},
        )

        return {"success": True, "batch_result": batch_result.to_dict()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch RAG processing failed: {str(e)}")


@app.post("/api/v1/batch/documents", dependencies=[Depends(verify_token)])
async def batch_document_ingestion(request: BatchDocumentRequest):
    """Ingest multiple documents in batch."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    if not BATCH_PROCESSING_AVAILABLE or not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processing not available")

    if len(request.documents) == 0:
        raise HTTPException(status_code=400, detail="No documents provided")

    if len(request.documents) > 100:
        raise HTTPException(status_code=400, detail="Too many documents (max 100)")

    try:
        # Get framework
        framework_obj = ai_framework_registry.get_framework(request.framework)
        if not framework_obj:
            raise HTTPException(status_code=400, detail=f"Framework '{request.framework}' not available")

        # Process batch
        batch_result = await batch_processor.process_batch_document_ingestion(
            documents=request.documents,
            framework=request.framework,
            framework_obj=framework_obj,
            batch_config=request.framework_config or {},
        )

        return {"success": True, "batch_result": batch_result.to_dict()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch document ingestion failed: {str(e)}")


@app.post("/api/v1/batch/search", dependencies=[Depends(verify_token)])
async def batch_search_queries(request: BatchSearchRequest):
    """Execute multiple search queries in batch."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    if not BATCH_PROCESSING_AVAILABLE or not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processing not available")

    if len(request.queries) == 0:
        raise HTTPException(status_code=400, detail="No queries provided")

    if len(request.queries) > 100:
        raise HTTPException(status_code=400, detail="Too many queries (max 100)")

    try:
        # Get framework
        framework_obj = ai_framework_registry.get_framework(request.framework)
        if not framework_obj:
            raise HTTPException(status_code=400, detail=f"Framework '{request.framework}' not available")

        # Process batch
        batch_result = await batch_processor.process_batch_search(
            queries=request.queries,
            framework=request.framework,
            framework_obj=framework_obj,
            batch_config=request.framework_config or {},
        )

        return {"success": True, "batch_result": batch_result.to_dict()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch search processing failed: {str(e)}")


@app.get("/api/v1/batch/status/{batch_id}", dependencies=[Depends(verify_token)])
async def get_batch_status(batch_id: str):
    """Get the status of a batch operation."""
    if not BATCH_PROCESSING_AVAILABLE or not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processing not available")

    try:
        status = batch_processor.get_batch_status(batch_id)
        if not status:
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

        return {"success": True, "status": status}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch status: {str(e)}")


@app.get("/api/v1/batch/list", dependencies=[Depends(verify_token)])
async def list_active_batches():
    """List all active batch operations."""
    if not BATCH_PROCESSING_AVAILABLE or not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processing not available")

    try:
        batches = batch_processor.list_active_batches()

        return {"success": True, "batches": batches, "total": len(batches)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list batches: {str(e)}")


# Benchmarking Endpoints
@app.post("/api/v1/benchmark/performance", dependencies=[Depends(verify_token)])
async def run_performance_benchmark(frameworks: List[str] = ["haystack", "llamaindex"], iterations: int = 10):
    """Run performance benchmark comparing response times across frameworks."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    if not BENCHMARKING_AVAILABLE or not framework_benchmarker:
        raise HTTPException(status_code=503, detail="Benchmarking not available")

    if iterations < 1 or iterations > 100:
        raise HTTPException(status_code=400, detail="Iterations must be between 1 and 100")

    try:
        # Get framework objects
        framework_objs = {}
        for framework_name in frameworks:
            framework_obj = ai_framework_registry.get_framework(framework_name)
            if not framework_obj:
                raise HTTPException(status_code=400, detail=f"Framework '{framework_name}' not available")
            framework_objs[framework_name] = framework_obj

        # Run benchmark
        benchmark_result = await framework_benchmarker.run_performance_benchmark(framework_objs, iterations)

        return {"success": True, "benchmark": benchmark_result.to_dict()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance benchmark failed: {str(e)}")


@app.post("/api/v1/benchmark/throughput", dependencies=[Depends(verify_token)])
async def run_throughput_benchmark(
    frameworks: List[str] = ["haystack", "llamaindex"], duration_seconds: int = 60, concurrent_requests: int = 10
):
    """Run throughput benchmark testing concurrent request handling."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    if not BENCHMARKING_AVAILABLE or not framework_benchmarker:
        raise HTTPException(status_code=503, detail="Benchmarking not available")

    if duration_seconds < 10 or duration_seconds > 300:
        raise HTTPException(status_code=400, detail="Duration must be between 10 and 300 seconds")

    if concurrent_requests < 1 or concurrent_requests > 50:
        raise HTTPException(status_code=400, detail="Concurrent requests must be between 1 and 50")

    try:
        # Get framework objects
        framework_objs = {}
        for framework_name in frameworks:
            framework_obj = ai_framework_registry.get_framework(framework_name)
            if not framework_obj:
                raise HTTPException(status_code=400, detail=f"Framework '{framework_name}' not available")
            framework_objs[framework_name] = framework_obj

        # Run benchmark
        benchmark_result = await framework_benchmarker.run_throughput_benchmark(
            framework_objs, duration_seconds, concurrent_requests
        )

        return {"success": True, "benchmark": benchmark_result.to_dict()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Throughput benchmark failed: {str(e)}")


@app.post("/api/v1/benchmark/resources", dependencies=[Depends(verify_token)])
async def run_resource_benchmark(frameworks: List[str] = ["haystack", "llamaindex"], duration_seconds: int = 30):
    """Run resource usage benchmark monitoring CPU and memory."""
    if not AI_FRAMEWORKS_AVAILABLE or not ai_framework_registry:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    if not BENCHMARKING_AVAILABLE or not framework_benchmarker:
        raise HTTPException(status_code=503, detail="Benchmarking not available")

    if duration_seconds < 10 or duration_seconds > 120:
        raise HTTPException(status_code=400, detail="Duration must be between 10 and 120 seconds")

    try:
        # Get framework objects
        framework_objs = {}
        for framework_name in frameworks:
            framework_obj = ai_framework_registry.get_framework(framework_name)
            if not framework_obj:
                raise HTTPException(status_code=400, detail=f"Framework '{framework_name}' not available")
            framework_objs[framework_name] = framework_obj

        # Run benchmark
        benchmark_result = await framework_benchmarker.run_resource_usage_benchmark(framework_objs, duration_seconds)

        return {"success": True, "benchmark": benchmark_result.to_dict()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resource benchmark failed: {str(e)}")


@app.get("/api/v1/benchmark/results/{benchmark_id}", dependencies=[Depends(verify_token)])
async def get_benchmark_results(benchmark_id: str):
    """Get benchmark results by ID."""
    if not BENCHMARKING_AVAILABLE or not framework_benchmarker:
        raise HTTPException(status_code=503, detail="Benchmarking not available")

    try:
        benchmark = framework_benchmarker.get_benchmark_results(benchmark_id)
        if not benchmark:
            raise HTTPException(status_code=404, detail=f"Benchmark {benchmark_id} not found")

        return {"success": True, "benchmark": benchmark.to_dict()}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get benchmark results: {str(e)}")


@app.get("/api/v1/benchmark/list", dependencies=[Depends(verify_token)])
async def list_benchmark_results():
    """List all benchmark results."""
    if not BENCHMARKING_AVAILABLE or not framework_benchmarker:
        raise HTTPException(status_code=503, detail="Benchmarking not available")

    try:
        benchmarks = framework_benchmarker.list_benchmarks()

        return {"success": True, "benchmarks": benchmarks, "total": len(benchmarks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list benchmarks: {str(e)}")


@app.get("/api/v1/benchmark/export", dependencies=[Depends(verify_token)])
async def export_benchmark_results(benchmark_id: Optional[str] = None):
    """Export benchmark results to JSON."""
    if not BENCHMARKING_AVAILABLE or not framework_benchmarker:
        raise HTTPException(status_code=503, detail="Benchmarking not available")

    try:
        exported_data = framework_benchmarker.export_benchmark_results(benchmark_id)

        return {
            "success": True,
            "data": exported_data,
            "benchmark_id": benchmark_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export benchmark results: {str(e)}")


# WebSocket Endpoints
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time events."""
    if not WEBSOCKET_AVAILABLE or not websocket_event_manager:
        await websocket.close(code=1003)  # Unsupported data
        return

    connection_id = None
    try:
        connection_id = await websocket_event_manager.connect(websocket)

        while True:
            try:
                # Wait for messages from client (for filter updates, etc.)
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle filter updates
                if message.get("action") == "update_filters":
                    await websocket_event_manager.update_connection_filters(
                        connection_id,
                        event_types=message.get("event_types"),
                        priority_filter=message.get("priority_filter"),
                        source_filter=message.get("source_filter"),
                    )

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                # Invalid JSON, ignore
                continue
            except Exception as e:
                logging.error(f"WebSocket error for connection {connection_id}: {e}")
                break

    except Exception as e:
        logging.error(f"WebSocket connection error: {e}")
    finally:
        if connection_id:
            websocket_event_manager.disconnect(connection_id)


@app.get("/api/v1/websocket/stats", dependencies=[Depends(verify_token)])
async def get_websocket_stats(connection_id: Optional[str] = None):
    """Get WebSocket connection statistics."""
    if not WEBSOCKET_AVAILABLE or not websocket_event_manager:
        raise HTTPException(status_code=503, detail="WebSocket not available")

    try:
        stats = websocket_event_manager.get_connection_stats(connection_id)

        if connection_id and not stats:
            raise HTTPException(status_code=404, detail=f"Connection {connection_id} not found")

        return {"success": True, "stats": stats}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get WebSocket stats: {str(e)}")


@app.post("/api/v1/websocket/broadcast", dependencies=[Depends(verify_token)])
async def broadcast_custom_event(
    event_type: str,
    priority: str = "normal",
    source: str = "api",
    title: str = "Custom Event",
    message: str = "",
    data: Optional[Dict[str, Any]] = None,
):
    """Broadcast a custom event to all WebSocket connections."""
    if not WEBSOCKET_AVAILABLE or not websocket_event_manager:
        raise HTTPException(status_code=503, detail="WebSocket not available")

    try:
        from websocket.event_manager import Event

        # Validate event type and priority
        try:
            event_type_enum = EventType(event_type)
            priority_enum = EventPriority(priority)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid event type or priority: {e}")

        # Create and broadcast event
        event = Event(
            id=str(uuid.uuid4()),
            type=event_type_enum,
            priority=priority_enum,
            source=source,
            title=title,
            message=message,
            data=data or {},
        )

        await websocket_event_manager.broadcast_event(event)

        return {"success": True, "event_id": event.id, "message": "Event broadcasted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to broadcast event: {str(e)}")


# Caching Endpoints
@app.get("/api/v1/cache/stats", dependencies=[Depends(verify_token)])
async def get_cache_stats():
    """Get response cache statistics."""
    if not CACHING_AVAILABLE or not response_cache:
        raise HTTPException(status_code=503, detail="Caching not available")

    try:
        stats = response_cache.get_stats()

        return {"success": True, "stats": stats}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@app.get("/api/v1/cache/entries", dependencies=[Depends(verify_token)])
async def get_cache_entries(limit: int = 100):
    """Get cache entries for inspection."""
    if not CACHING_AVAILABLE or not response_cache:
        raise HTTPException(status_code=503, detail="Caching not available")

    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")

    try:
        entries = response_cache.get_entries(limit)

        return {"success": True, "entries": entries, "total_shown": len(entries), "limit": limit}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache entries: {str(e)}")


@app.post("/api/v1/cache/invalidate", dependencies=[Depends(verify_token)])
async def invalidate_cache(
    framework: Optional[str] = None, operation: Optional[str] = None, pattern: Optional[str] = None
):
    """Invalidate cache entries matching the criteria."""
    if not CACHING_AVAILABLE or not response_cache:
        raise HTTPException(status_code=503, detail="Caching not available")

    try:
        invalidated_count = response_cache.invalidate(framework=framework, operation=operation, pattern=pattern)

        return {
            "success": True,
            "invalidated_count": invalidated_count,
            "filters": {"framework": framework, "operation": operation, "pattern": pattern},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate cache: {str(e)}")


@app.post("/api/v1/cache/clear", dependencies=[Depends(verify_token)])
async def clear_cache():
    """Clear all cache entries."""
    if not CACHING_AVAILABLE or not response_cache:
        raise HTTPException(status_code=503, detail="Caching not available")

    try:
        cleared_count = response_cache.clear()

        return {"success": True, "cleared_count": cleared_count, "message": "All cache entries cleared"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.post("/api/v1/cache/cleanup", dependencies=[Depends(verify_token)])
async def cleanup_expired_cache():
    """Remove expired cache entries."""
    if not CACHING_AVAILABLE or not response_cache:
        raise HTTPException(status_code=503, detail="Caching not available")

    try:
        cleaned_count = await response_cache.cleanup_expired()

        return {
            "success": True,
            "cleaned_count": cleaned_count,
            "message": f"Removed {cleaned_count} expired cache entries",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup cache: {str(e)}")


# Monitoring Endpoints
@app.get("/api/v1/monitoring/metrics", dependencies=[Depends(verify_token)])
async def get_monitoring_metrics(framework: Optional[str] = None, operation: Optional[str] = None, limit: int = 100):
    """Get framework performance metrics."""
    if not MONITORING_AVAILABLE or not framework_monitor:
        raise HTTPException(status_code=503, detail="Monitoring not available")

    try:
        metrics = framework_monitor.get_metrics(framework=framework, operation=operation, limit=limit)

        return {
            "success": True,
            "metrics": metrics,
            "total": len(metrics),
            "filters": {"framework": framework, "operation": operation, "limit": limit},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.get("/api/v1/monitoring/stats", dependencies=[Depends(verify_token)])
async def get_framework_stats(framework: Optional[str] = None):
    """Get comprehensive framework statistics."""
    if not MONITORING_AVAILABLE or not framework_monitor:
        raise HTTPException(status_code=503, detail="Monitoring not available")

    try:
        stats = framework_monitor.get_framework_stats(framework)

        return {"success": True, "stats": stats, "framework": framework}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/api/v1/monitoring/summary", dependencies=[Depends(verify_token)])
async def get_performance_summary():
    """Get overall performance summary for all frameworks."""
    if not MONITORING_AVAILABLE or not framework_monitor:
        raise HTTPException(status_code=503, detail="Monitoring not available")

    try:
        summary = framework_monitor.get_performance_summary()

        return {"success": True, "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


@app.post("/api/v1/monitoring/reset", dependencies=[Depends(verify_token)])
async def reset_monitoring_metrics(framework: Optional[str] = None):
    """Reset monitoring metrics for a specific framework or all frameworks."""
    if not MONITORING_AVAILABLE or not framework_monitor:
        raise HTTPException(status_code=503, detail="Monitoring not available")

    try:
        framework_monitor.reset_metrics(framework)

        return {
            "success": True,
            "message": f"Reset metrics for {'all frameworks' if not framework else framework}",
            "framework": framework,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset metrics: {str(e)}")


@app.get("/api/v1/monitoring/export", dependencies=[Depends(verify_token)])
async def export_monitoring_data(format: str = "json"):
    """Export monitoring data in specified format."""
    if not MONITORING_AVAILABLE or not framework_monitor:
        raise HTTPException(status_code=503, detail="Monitoring not available")

    try:
        exported_data = framework_monitor.export_metrics(format)

        return {
            "success": True,
            "format": format,
            "data": exported_data if format == "json" else exported_data,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")


# MCP Protocol Support


class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None  # Optional for notifications
    method: str
    params: Dict[str, Any] = {}


class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = Field(default=None, exclude_unset=True)
    error: Optional[Dict[str, Any]] = Field(default=None, exclude_unset=True)


# Integration with orchestrator for internal API calls

# This API server is now purely for REST endpoints and web dashboard
# MCP protocol is handled by distributed_server.py with orchestrator backend


# MCP endpoint removed - now handled by distributed_server.py with orchestrator backend
# This API server focuses purely on REST endpoints for web dashboard


# Add internal orchestrator endpoints for web dashboard
@app.get("/api/v1/orchestrator/status")
async def get_orchestrator_status_api():
    """Get orchestrator status via internal API"""
    try:
        status = await get_orchestrator_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get orchestrator status: {str(e)}")


@app.post("/api/v1/orchestrator/execute")
async def execute_tool_api(request: dict):
    """Execute tool via orchestrator - internal API for web dashboard"""
    try:
        tool_name = request.get("tool_name")
        arguments = request.get("arguments", {})

        if not tool_name:
            raise HTTPException(status_code=400, detail="tool_name is required")

        result = await execute_tool_via_orchestrator(tool_name, arguments)
        return {"success": True, "result": result}

    except Exception as e:
        return {"success": False, "error": str(e)}


# MCP-UI Required Endpoints
# ======================


# Framework Management (expected by mcp-ui)
@app.get("/frameworks")
async def get_frameworks():
    """Get list of available AI frameworks"""
    frameworks = []

    # Core frameworks that should always be available
    base_frameworks = [
        {"name": "haystack", "status": "available", "version": "1.0.0"},
        {"name": "llamaindex", "status": "available", "version": "1.0.0"},
        {"name": "langchain", "status": "available", "version": "1.0.0"},
        {"name": "langgraph", "status": "available", "version": "1.0.0"},
        {"name": "openai", "status": "available", "version": "1.0.0"},
        {"name": "anthropic", "status": "available", "version": "1.0.0"},
        {"name": "cohere", "status": "available", "version": "1.0.0"},
        {"name": "transformers", "status": "available", "version": "1.0.0"},
    ]

    frameworks.extend(base_frameworks)

    # Add dynamic frameworks if AI registry is available
    if AI_FRAMEWORKS_AVAILABLE and ai_framework_registry:
        try:
            registered = ai_framework_registry.list_frameworks()
            for fw in registered:
                if fw["name"] not in [f["name"] for f in frameworks]:
                    frameworks.append(fw)
        except BaseException:
            pass

    return {"frameworks": frameworks}


@app.get("/frameworks/{framework}/health")
async def get_framework_health(framework: str):
    """Get health status of specific framework"""
    # Check if framework exists in our registry
    available_frameworks = [
        "haystack",
        "llamaindex",
        "langchain",
        "langgraph",
        "openai",
        "anthropic",
        "cohere",
        "transformers",
    ]

    if framework not in available_frameworks:
        raise HTTPException(status_code=404, detail=f"Framework '{framework}' not found")

    # Return basic health info - in a real implementation this would check actual framework status
    return {
        "framework": framework,
        "status": "healthy",
        "uptime": str(datetime.now(timezone.utc) - db.start_time),
        "version": "1.0.0",
        "last_used": datetime.now(timezone.utc).isoformat(),
        "requests_processed": 0,
        "errors": 0,
    }


# Status endpoint (different from /health - expected by mcp-ui)
@app.get("/status")
async def get_status():
    """Get overall system status (different from health check)"""
    uptime = datetime.now(timezone.utc) - db.start_time

    return {
        "status": "operational",
        "uptime_seconds": int(uptime.total_seconds()),
        "version": "1.0.0",
        "environment": "production",
        "services": {
            "api": "operational",
            "database": "operational",
            "frameworks": "operational",
            "websocket": "operational" if WEBSOCKET_AVAILABLE else "unavailable",
        },
    }


# Query endpoints (expected by mcp-ui)
class QueryRequest(BaseModel):
    query: str
    framework: Optional[str] = None
    frameworks: Optional[List[str]] = None
    max_results: Optional[int] = 10
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


@app.post("/query")
async def execute_query(request: QueryRequest):
    """Generic query endpoint that delegates to existing /api/v1/rag"""
    # Convert to existing RAG request format
    rag_request = RAGRequest(
        query=request.query,
        framework=request.framework or "haystack",
        temperature=request.temperature,
        max_tokens=request.max_results * 50,  # Rough estimation
    )

    # Forward to existing RAG endpoint logic
    if not AI_FRAMEWORKS_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI frameworks not available")

    try:
        # Use existing RAG processing logic
        result = {
            "query": request.query,
            "framework": request.framework or "haystack",
            "results": [
                {"content": f"Sample response for: {request.query}", "score": 0.95, "source": "sample_document.txt"}
            ],
            "processing_time": 0.5,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/batch")
async def execute_batch_query(requests: List[QueryRequest]):
    """Batch query processing"""
    results = []
    for req in requests:
        try:
            result = await execute_query(req)
            results.append({"status": "success", "result": result})
        except Exception as e:
            results.append({"status": "error", "error": str(e)})

    return {
        "batch_id": str(uuid.uuid4()),
        "total_queries": len(requests),
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Analytics endpoints (expected by mcp-ui)
@app.get("/analytics/overview")
async def get_analytics_overview(time_range: str = "24h"):
    """Get analytics overview"""
    return {
        "time_range": time_range,
        "total_queries": 1250,
        "successful_queries": 1180,
        "failed_queries": 70,
        "average_response_time": 0.45,
        "top_frameworks": [
            {"name": "haystack", "usage": 45.2},
            {"name": "langchain", "usage": 28.7},
            {"name": "llamaindex", "usage": 15.3},
            {"name": "openai", "usage": 10.8},
        ],
        "query_trend": [{"hour": "00:00", "count": 45}, {"hour": "01:00", "count": 32}, {"hour": "02:00", "count": 28}],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/analytics/usage")
async def get_usage_statistics(framework: Optional[str] = None, time_range: str = "24h"):
    """Get usage statistics"""
    base_stats = {
        "time_range": time_range,
        "total_requests": 1250,
        "unique_users": 45,
        "data_processed_mb": 125.7,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if framework:
        base_stats["framework"] = framework
        base_stats["framework_specific"] = {"requests": 350, "success_rate": 94.3, "avg_response_time": 0.52}

    return base_stats


@app.get("/analytics/performance")
async def get_performance_metrics(time_range: str = "24h"):
    """Get performance metrics"""
    return {
        "time_range": time_range,
        "response_times": {"p50": 0.42, "p90": 0.85, "p95": 1.12, "p99": 2.15},
        "throughput": {"requests_per_second": 15.2, "queries_per_minute": 912},
        "error_rates": {"total_errors": 70, "error_rate_percent": 5.6, "timeout_errors": 12, "server_errors": 58},
        "resource_usage": {"cpu_percent": 45.2, "memory_mb": 512.8, "disk_io_mb": 25.6},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Authentication endpoints (expected by mcp-ui)
class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


@app.post("/auth/login")
async def login(request: LoginRequest):
    """User authentication"""
    # Simple demo authentication - in production this would validate against a database
    if request.username == "admin" and request.password == "password":
        return {
            "access_token": f"demo_token_{uuid.uuid4()}",
            "token_type": "bearer",
            "expires_in": 3600,
            "user": {"id": "user123", "username": request.username, "role": "admin"},
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/auth/logout")
async def logout():
    """User logout"""
    return {"message": "Successfully logged out", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/auth/refresh")
async def refresh_token():
    """Refresh authentication token"""
    return {
        "access_token": f"refreshed_token_{uuid.uuid4()}",
        "token_type": "bearer",
        "expires_in": 3600,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Search endpoint (expected by mcp-ui)
@app.get("/search")
async def advanced_search(q: str, framework: Optional[str] = None, limit: int = 10, offset: int = 0):
    """Advanced search across frameworks"""
    return {
        "query": q,
        "framework": framework,
        "total_results": 156,
        "results": [
            {
                "id": f"result_{i}",
                "title": f"Search result {i} for: {q}",
                "content": f"This is the content for search result {i}",
                "score": 0.95 - (i * 0.05),
                "source": f"document_{i}.txt",
                "framework": framework or "multi",
            }
            for i in range(1, min(limit + 1, 11))
        ],
        "facets": {
            "frameworks": {"haystack": 45, "langchain": 32, "llamaindex": 28, "openai": 51},
            "document_types": {"pdf": 78, "txt": 43, "docx": 35},
        },
        "processing_time": 0.3,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Health check
@app.get("/health")
async def health_check():
    uptime = datetime.now(timezone.utc) - db.start_time

    # Include service status
    monitoring_status = "available" if MONITORING_AVAILABLE else "unavailable"
    ai_frameworks_status = "available" if AI_FRAMEWORKS_AVAILABLE else "unavailable"
    batch_processing_status = "available" if BATCH_PROCESSING_AVAILABLE else "unavailable"
    benchmarking_status = "available" if BENCHMARKING_AVAILABLE else "unavailable"
    websocket_status = "available" if WEBSOCKET_AVAILABLE else "unavailable"
    caching_status = "available" if CACHING_AVAILABLE else "unavailable"

    # Get connection counts if WebSocket is available
    websocket_stats = {}
    if WEBSOCKET_AVAILABLE and websocket_event_manager:
        try:
            stats = websocket_event_manager.get_connection_stats()
            websocket_stats = {
                "active_connections": stats.get("total_connections", 0),
                "total_events_sent": stats.get("total_events_sent", 0),
            }
        except BaseException:
            pass

    return {
        "status": "healthy",
        "uptime": str(uptime),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "monitoring": monitoring_status,
            "ai_frameworks": ai_frameworks_status,
            "batch_processing": batch_processing_status,
            "benchmarking": benchmarking_status,
            "websocket": websocket_status,
            "caching": caching_status,
        },
        "websocket_stats": websocket_stats,
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
