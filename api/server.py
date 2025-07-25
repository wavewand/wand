"""REST API server for the Python MCP implementation."""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


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
    
    db = InMemoryDB()
    use_postgres = False
else:
    # PostgreSQL database
    from api.database import DatabaseManager
    db_manager = DatabaseManager(DATABASE_URL)
    use_postgres = True

app = FastAPI(title="Multi-Agent Platform API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        await self.send_personal_message({
            "type": "system_event",
            "data": {
                "event": "connected",
                "message": "Connected to MAP server"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# Dependency for authentication
async def verify_token(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    if not authorization and not x_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # TODO: Implement actual token/API key validation
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
        capabilities=request.capabilities
    )
    db.agents[agent.id] = agent
    
    # Broadcast agent creation
    await manager.broadcast({
        "type": "agent_update",
        "data": {
            "action": "created",
            "agent": asdict(agent)
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
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
    await manager.broadcast({
        "type": "agent_update",
        "data": {
            "action": "updated",
            "agent": asdict(agent)
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    return asdict(agent)


@app.delete("/api/v1/agents/{agentId}", status_code=204, dependencies=[Depends(verify_token)])
async def delete_agent(agentId: str):
    if agentId not in db.agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    del db.agents[agentId]
    
    # Broadcast deletion
    await manager.broadcast({
        "type": "agent_update",
        "data": {
            "action": "deleted",
            "id": agentId
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


# Task endpoints
@app.get("/api/v1/tasks", dependencies=[Depends(verify_token)])
async def list_tasks(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    assigned_to: Optional[str] = None
):
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
        dependencies=request.dependencies or []
    )
    db.tasks[task.id] = task
    
    # Update project task count
    if task.project_id and task.project_id in db.projects:
        db.projects[task.project_id].tasks_total += 1
        db.update_project_progress(task.project_id)
    
    # Broadcast task creation
    await manager.broadcast({
        "type": "task_update",
        "data": {
            "action": "created",
            "task": asdict(task)
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
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
    await manager.broadcast({
        "type": "task_update",
        "data": {
            "action": "updated",
            "task": asdict(task)
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
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
    await manager.broadcast({
        "type": "task_update",
        "data": {
            "action": "deleted",
            "id": taskId
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


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
    await manager.broadcast({
        "type": "task_update",
        "data": {
            "action": "assigned",
            "task": asdict(task)
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
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
    await manager.broadcast({
        "type": "task_update",
        "data": {
            "action": "status_changed",
            "task": asdict(task)
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
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
        due_date=request.due_date or datetime.now(timezone.utc).replace(day=datetime.now(timezone.utc).day + 30)
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
            project_id=project.id
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
        "integrations": integrations
    }


@app.get("/api/v1/system/metrics", dependencies=[Depends(verify_token)])
async def get_system_metrics():
    # TODO: Implement real metrics collection
    return {
        "cpu_usage": 45.2,
        "memory_usage": 62.8,
        "goroutines": 0,  # N/A for Python
        "active_connections": len(manager.active_connections),
        "request_rate": 15.5,
        "error_rate": 0.02,
        "average_response_time": 125.3
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
            last_sync=datetime.now(timezone.utc)
        ),
        Integration(
            id="git",
            name="Git",
            type="version_control",
            enabled=True,
            status=IntegrationStatus.CONNECTED,
            last_sync=datetime.now(timezone.utc)
        ),
        Integration(
            id="jenkins",
            name="Jenkins",
            type="ci_cd",
            enabled=True,
            status=IntegrationStatus.CONNECTED,
            last_sync=datetime.now(timezone.utc)
        ),
        Integration(
            id="youtrack",
            name="YouTrack",
            type="issue_tracking",
            enabled=False,
            status=IntegrationStatus.DISCONNECTED,
            last_sync=datetime.now(timezone.utc)
        ),
        Integration(
            id="aws",
            name="AWS",
            type="cloud",
            enabled=True,
            status=IntegrationStatus.CONNECTED,
            last_sync=datetime.now(timezone.utc)
        ),
        Integration(
            id="postgresql",
            name="PostgreSQL",
            type="database",
            enabled=True,
            status=IntegrationStatus.CONNECTED,
            last_sync=datetime.now(timezone.utc)
        ),
        Integration(
            id="bambu",
            name="Bambu 3D",
            type="hardware",
            enabled=False,
            status=IntegrationStatus.DISCONNECTED,
            last_sync=datetime.now(timezone.utc)
        ),
        Integration(
            id="web_search",
            name="Web Search",
            type="search",
            enabled=True,
            status=IntegrationStatus.CONNECTED,
            last_sync=datetime.now(timezone.utc)
        ),
    ]
    
    return [asdict(i) for i in integrations]


@app.post("/api/v1/integrations/{integrationId}/toggle", dependencies=[Depends(verify_token)])
async def toggle_integration(integrationId: str, request: ToggleIntegrationRequest):
    # TODO: Actually toggle the integration
    # For now, just return success
    integration = Integration(
        id=integrationId,
        name=integrationId.title(),
        type="unknown",
        enabled=request.enabled,
        status=IntegrationStatus.CONNECTED if request.enabled else IntegrationStatus.DISCONNECTED,
        last_sync=datetime.now(timezone.utc)
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
            await manager.send_personal_message({
                "type": "echo",
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Health check
@app.get("/health")
async def health_check():
    uptime = datetime.now(timezone.utc) - db.start_time
    return {
        "status": "healthy",
        "uptime": str(uptime),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()