#!/usr/bin/env python3

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
import uuid
import argparse
from pathlib import Path

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from mcp.types import TextContent

# Import config management
from config import load_config, Config

# Initialize config - will be loaded in main() or use defaults for tests
config = None

def init_config(config_path='config.json'):
    """Initialize configuration"""
    global config
    config = load_config(config_path)
    return config

# Use default config if not initialized (for tests)
if config is None:
    try:
        config = load_config('config.json')
    except:
        # Use a minimal default config for tests
        from config import ServerConfig, DistributedConfig
        config = Config(
            server=ServerConfig(),
            distributed=DistributedConfig()
        )

# Configure logging based on config
log_level = getattr(logging, config.server.log_level.upper(), logging.INFO)
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Initialize the MCP server with config
mcp = FastMCP("distributed-mcp-server", config.server.version)

# Agent Types
class AgentType(Enum):
    MANAGER = "manager"
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    DEVOPS = "devops"
    INTEGRATION = "integration"
    QA = "qa"

# Task Status
class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

# Task Priority
class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    type: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Agent:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: AgentType = AgentType.INTEGRATION
    capabilities: Set[str] = field(default_factory=set)
    current_tasks: List[str] = field(default_factory=list)
    status: str = "idle"
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class DistributedTaskManager:
    def __init__(self, config: Config):
        self.config = config
        self.tasks: Dict[str, Task] = {}
        self.agents: Dict[str, Agent] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=config.distributed.task_queue_size)
        self.max_concurrent_tasks = config.distributed.max_concurrent_tasks
        self.initialize_agents()
        
    def initialize_agents(self):
        """Initialize the default agent pool"""
        # Manager Agent
        self.register_agent(Agent(
            name="Project Manager",
            type=AgentType.MANAGER,
            capabilities={"planning", "coordination", "reporting", "risk_assessment"}
        ))
        
        # Frontend Agents
        self.register_agent(Agent(
            name="Frontend Developer 1",
            type=AgentType.FRONTEND,
            capabilities={"react", "vue", "angular", "typescript", "css", "responsive_design"}
        ))
        
        # Backend Agents
        self.register_agent(Agent(
            name="Backend Developer 1",
            type=AgentType.BACKEND,
            capabilities={"python", "go", "nodejs", "api_design", "microservices"}
        ))
        
        # Database Agent
        self.register_agent(Agent(
            name="Database Engineer",
            type=AgentType.DATABASE,
            capabilities={"postgresql", "mysql", "mongodb", "redis", "database_design", "optimization"}
        ))
        
        # DevOps Agent
        self.register_agent(Agent(
            name="DevOps Engineer",
            type=AgentType.DEVOPS,
            capabilities={"aws", "docker", "kubernetes", "jenkins", "terraform", "monitoring"}
        ))
        
        # Integration Agent
        self.register_agent(Agent(
            name="Integration Specialist",
            type=AgentType.INTEGRATION,
            capabilities={"slack", "git", "youtrack", "api_integration", "webhooks"}
        ))
        
    def register_agent(self, agent: Agent):
        """Register a new agent"""
        self.agents[agent.id] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.type.value})")
        
    def create_task(self, title: str, description: str, task_type: str, 
                   priority: TaskPriority = TaskPriority.MEDIUM) -> Task:
        """Create a new task"""
        task = Task(
            title=title,
            description=description,
            type=task_type,
            priority=priority
        )
        self.tasks[task.id] = task
        return task
        
    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to an agent"""
        if task_id in self.tasks and agent_id in self.agents:
            task = self.tasks[task_id]
            agent = self.agents[agent_id]
            
            task.assigned_to = agent_id
            task.status = TaskStatus.ASSIGNED
            agent.current_tasks.append(task_id)
            agent.status = "working"
            
            return True
        return False
        
    def find_best_agent(self, task: Task) -> Optional[str]:
        """Find the best available agent for a task based on capabilities"""
        # Map task types to required capabilities
        capability_map = {
            "frontend": {"react", "vue", "angular", "typescript"},
            "backend": {"python", "go", "nodejs", "api_design"},
            "database": {"postgresql", "mysql", "database_design"},
            "devops": {"aws", "docker", "kubernetes", "jenkins"},
            "integration": {"api_integration", "webhooks"},
            "slack": {"slack"},
            "git": {"git"},
            "aws": {"aws", "terraform"},
            "3d_printing": {"bambu", "3d_printing"}
        }
        
        required_capabilities = capability_map.get(task.type.lower(), set())
        
        best_agent = None
        best_score = 0
        
        for agent_id, agent in self.agents.items():
            # Skip busy agents
            if len(agent.current_tasks) >= 3:  # Max 3 concurrent tasks
                continue
                
            # Calculate capability match score
            score = len(agent.capabilities.intersection(required_capabilities))
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
                
        return best_agent

# Global task manager instance
task_manager = DistributedTaskManager(config)

# Tools for task management and distribution
@mcp.tool()
async def create_project(ctx: Context, 
                        name: str, 
                        description: str,
                        components: List[str]) -> str:
    """
    Create a new project with multiple components
    
    Args:
        name: Project name
        description: Project description
        components: List of components (frontend, backend, database, etc.)
    """
    project_id = str(uuid.uuid4())
    tasks = []
    
    # Create tasks for each component
    for component in components:
        task = task_manager.create_task(
            title=f"{name} - {component.capitalize()} Development",
            description=f"Develop {component} for {name}",
            task_type=component,
            priority=TaskPriority.HIGH
        )
        tasks.append(task)
        
        # Auto-assign to best available agent
        agent_id = task_manager.find_best_agent(task)
        if agent_id:
            task_manager.assign_task(task.id, agent_id)
    
    return f"Created project '{name}' with {len(tasks)} tasks. Tasks distributed to agents."

@mcp.tool()
async def distribute_task(ctx: Context,
                         title: str,
                         description: str,
                         task_type: str,
                         priority: str = "medium") -> str:
    """
    Create and distribute a task to the best available agent
    
    Args:
        title: Task title
        description: Task description
        task_type: Type of task (frontend, backend, database, devops, integration)
        priority: Task priority (critical, high, medium, low)
    """
    priority_map = {
        "critical": TaskPriority.CRITICAL,
        "high": TaskPriority.HIGH,
        "medium": TaskPriority.MEDIUM,
        "low": TaskPriority.LOW
    }
    
    task = task_manager.create_task(
        title=title,
        description=description,
        task_type=task_type,
        priority=priority_map.get(priority.lower(), TaskPriority.MEDIUM)
    )
    
    agent_id = task_manager.find_best_agent(task)
    if agent_id:
        task_manager.assign_task(task.id, agent_id)
        agent = task_manager.agents[agent_id]
        return f"Task '{title}' assigned to {agent.name} ({agent.type.value})"
    else:
        return f"Task '{title}' created but no suitable agent available. Task queued."

@mcp.tool()
async def get_project_status(ctx: Context) -> str:
    """Get the current status of all tasks and agents"""
    status_report = {
        "agents": {},
        "tasks": {
            "total": len(task_manager.tasks),
            "by_status": {},
            "by_priority": {}
        }
    }
    
    # Agent status
    for agent_id, agent in task_manager.agents.items():
        status_report["agents"][agent.name] = {
            "type": agent.type.value,
            "status": agent.status,
            "current_tasks": len(agent.current_tasks),
            "capabilities": list(agent.capabilities)
        }
    
    # Task status
    for task in task_manager.tasks.values():
        status = task.status.value
        priority = task.priority.name
        
        status_report["tasks"]["by_status"][status] = \
            status_report["tasks"]["by_status"].get(status, 0) + 1
        status_report["tasks"]["by_priority"][priority] = \
            status_report["tasks"]["by_priority"].get(priority, 0) + 1
    
    return json.dumps(status_report, indent=2)

# Integration Tools
@mcp.tool()
async def slack_notify(ctx: Context,
                      channel: str,
                      message: str,
                      thread_ts: Optional[str] = None) -> str:
    """
    Send a notification to Slack
    
    Args:
        channel: Slack channel name or ID
        message: Message to send
        thread_ts: Thread timestamp for replies (optional)
    """
    # In production, this would use the Slack SDK
    return f"Slack notification sent to {channel}: {message}"

@mcp.tool()
async def git_operation(ctx: Context,
                       operation: str,
                       repository: str,
                       branch: Optional[str] = None,
                       message: Optional[str] = None) -> str:
    """
    Perform Git operations
    
    Args:
        operation: Git operation (clone, pull, push, commit, branch)
        repository: Repository URL or path
        branch: Branch name (optional)
        message: Commit message (for commit operation)
    """
    operations = {
        "clone": f"Cloned repository {repository}",
        "pull": f"Pulled latest changes from {repository}",
        "push": f"Pushed changes to {repository}",
        "commit": f"Committed changes: {message}",
        "branch": f"Created/switched to branch {branch}"
    }
    
    return operations.get(operation, f"Performed {operation} on {repository}")

@mcp.tool()
async def jenkins_trigger(ctx: Context,
                         job_name: str,
                         parameters: Optional[Dict[str, Any]] = None) -> str:
    """
    Trigger a Jenkins job
    
    Args:
        job_name: Name of the Jenkins job
        parameters: Job parameters (optional)
    """
    params = json.dumps(parameters) if parameters else "default parameters"
    return f"Triggered Jenkins job '{job_name}' with {params}"

@mcp.tool()
async def youtrack_issue(ctx: Context,
                        operation: str,
                        project: str,
                        summary: Optional[str] = None,
                        description: Optional[str] = None,
                        issue_id: Optional[str] = None) -> str:
    """
    Manage YouTrack issues
    
    Args:
        operation: Operation to perform (create, update, close)
        project: YouTrack project ID
        summary: Issue summary (for create)
        description: Issue description (for create/update)
        issue_id: Issue ID (for update/close)
    """
    if operation == "create":
        return f"Created YouTrack issue in {project}: {summary}"
    elif operation == "update":
        return f"Updated YouTrack issue {issue_id}"
    elif operation == "close":
        return f"Closed YouTrack issue {issue_id}"
    else:
        return f"Performed {operation} on YouTrack project {project}"

@mcp.tool()
async def postgres_query(ctx: Context,
                        database: str,
                        query: str,
                        params: Optional[List[Any]] = None) -> str:
    """
    Execute PostgreSQL queries
    
    Args:
        database: Database name
        query: SQL query to execute
        params: Query parameters (optional)
    """
    # In production, this would use asyncpg or psycopg
    return f"Executed query on {database}: {query[:50]}..."

@mcp.tool()
async def aws_operation(ctx: Context,
                       service: str,
                       operation: str,
                       parameters: Dict[str, Any]) -> str:
    """
    Perform AWS operations
    
    Args:
        service: AWS service (ec2, s3, lambda, etc.)
        operation: Operation to perform
        parameters: Operation parameters
    """
    return f"Performed {operation} on AWS {service} with parameters: {json.dumps(parameters)}"

@mcp.tool()
async def bambu_print(ctx: Context,
                     printer_id: str,
                     file_path: str,
                     settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Send a print job to Bambu 3D printer
    
    Args:
        printer_id: Printer identifier
        file_path: Path to 3MF/STL file
        settings: Print settings (optional)
    """
    settings_str = json.dumps(settings) if settings else "default settings"
    return f"Sent {file_path} to Bambu printer {printer_id} with {settings_str}"

@mcp.tool()
async def web_search(ctx: Context,
                    query: str,
                    num_results: int = 5) -> str:
    """
    Search the internet for information
    
    Args:
        query: Search query
        num_results: Number of results to return
    """
    # In production, this would use a search API
    return f"Found {num_results} results for '{query}'"

@mcp.tool()
async def api_request(ctx: Context,
                     url: str,
                     method: str = "GET",
                     headers: Optional[Dict[str, str]] = None,
                     body: Optional[Dict[str, Any]] = None) -> str:
    """
    Make arbitrary API requests
    
    Args:
        url: API endpoint URL
        method: HTTP method (GET, POST, PUT, DELETE)
        headers: Request headers (optional)
        body: Request body (optional)
    """
    return f"Made {method} request to {url}"

# Resources for monitoring
@mcp.resource("agents://status")
async def get_agents_status() -> str:
    """Current status of all agents"""
    agents_info = []
    for agent_id, agent in task_manager.agents.items():
        agents_info.append({
            "id": agent.id,
            "name": agent.name,
            "type": agent.type.value,
            "status": agent.status,
            "current_tasks": agent.current_tasks,
            "capabilities": list(agent.capabilities)
        })
    return json.dumps(agents_info, indent=2)

@mcp.resource("tasks://all")
async def get_all_tasks() -> str:
    """All tasks in the system"""
    tasks_info = []
    for task_id, task in task_manager.tasks.items():
        tasks_info.append({
            "id": task.id,
            "title": task.title,
            "type": task.type,
            "status": task.status.value,
            "priority": task.priority.name,
            "assigned_to": task.assigned_to,
            "created_at": task.created_at.isoformat()
        })
    return json.dumps(tasks_info, indent=2)

@mcp.resource("integrations://config")
async def get_integrations_config() -> str:
    """Current integrations configuration"""
    config = {
        "slack": {
            "enabled": True,
            "workspace": "your-workspace",
            "default_channel": "#dev-updates"
        },
        "git": {
            "enabled": True,
            "default_branch": "main",
            "auto_commit": False
        },
        "jenkins": {
            "enabled": True,
            "url": "https://jenkins.example.com",
            "default_pipeline": "ci-cd"
        },
        "youtrack": {
            "enabled": True,
            "url": "https://youtrack.example.com",
            "default_project": "DEV"
        },
        "postgres": {
            "enabled": True,
            "default_database": "production"
        },
        "aws": {
            "enabled": True,
            "region": "us-east-1",
            "services": ["ec2", "s3", "lambda", "rds"]
        },
        "bambu": {
            "enabled": True,
            "printers": ["X1-Carbon-01", "P1S-01"]
        }
    }
    return json.dumps(config, indent=2)

def main():
    """Run the distributed MCP server"""
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Distributed MCP Server')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file (default: config.json)')
    args = parser.parse_args()
    
    # Initialize configuration
    init_config(args.config)
    
    # Configure logging based on config
    log_level = getattr(logging, config.server.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, force=True)
    
    asyncio.run(mcp.run(transport="stdio"))

if __name__ == "__main__":
    main()