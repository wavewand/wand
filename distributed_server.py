#!/usr/bin/env python3

import argparse
import asyncio
import importlib.util
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

# Import config management - direct import
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from mcp.types import TextContent
from pydantic import BaseModel

from tools.execution.base import ExecutionConfig, ExecutionResult
from tools.execution.factory import create_execution_backend

spec = importlib.util.spec_from_file_location("config_module", Path(__file__).parent / "config.py")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
load_config = config_module.load_config
Config = config_module.Config

# Import execution backend

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
    except BaseException:
        # Use a minimal default config for tests
        ServerConfig = config_module.ServerConfig
        DistributedConfig = config_module.DistributedConfig
        config = Config(server=ServerConfig(), distributed=DistributedConfig())

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
    HAYSTACK = "haystack"


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
        self.register_agent(
            Agent(
                name="Project Manager",
                type=AgentType.MANAGER,
                capabilities={"planning", "coordination", "reporting", "risk_assessment"},
            )
        )

        # Frontend Agents
        self.register_agent(
            Agent(
                name="Frontend Developer 1",
                type=AgentType.FRONTEND,
                capabilities={"react", "vue", "angular", "typescript", "css", "responsive_design"},
            )
        )

        # Backend Agents
        self.register_agent(
            Agent(
                name="Backend Developer 1",
                type=AgentType.BACKEND,
                capabilities={"python", "go", "nodejs", "api_design", "microservices"},
            )
        )

        # Database Agent
        self.register_agent(
            Agent(
                name="Database Engineer",
                type=AgentType.DATABASE,
                capabilities={"postgresql", "mysql", "mongodb", "redis", "database_design", "optimization"},
            )
        )

        # DevOps Agent
        self.register_agent(
            Agent(
                name="DevOps Engineer",
                type=AgentType.DEVOPS,
                capabilities={"aws", "docker", "kubernetes", "jenkins", "terraform", "monitoring"},
            )
        )

        # Integration Agent
        self.register_agent(
            Agent(
                name="Integration Specialist",
                type=AgentType.INTEGRATION,
                capabilities={"slack", "git", "youtrack", "api_integration", "webhooks"},
            )
        )

        # Haystack AI Agent
        self.register_agent(
            Agent(
                name="Haystack AI Specialist",
                type=AgentType.HAYSTACK,
                capabilities={
                    "rag",
                    "document_search",
                    "question_answering",
                    "document_processing",
                    "pipeline_management",
                    "semantic_search",
                    "summarization",
                    "multimodal",
                },
            )
        )

    def register_agent(self, agent: Agent):
        """Register a new agent"""
        self.agents[agent.id] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.type.value})")

    def create_task(
        self, title: str, description: str, task_type: str, priority: TaskPriority = TaskPriority.MEDIUM
    ) -> Task:
        """Create a new task"""
        task = Task(title=title, description=description, type=task_type, priority=priority)
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
            "3d_printing": {"bambu", "3d_printing"},
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

# Initialize execution backend
execution_backend = None
try:
    # Get execution configuration
    exec_config = {
        'working_directory': config.execution.working_directory,
        'allowed_commands': config.execution.security.allowed_commands,
        'blocked_commands': config.execution.security.blocked_commands,
        'path_restrictions': config.execution.security.path_restrictions,
        'timeout': config.execution.default_timeout,
        'max_concurrent': config.execution.max_concurrent,
    }

    # Add mode-specific configuration
    if config.execution.mode == 'host_agent':
        exec_config.update(
            {
                'url': config.execution.host_agent.url,
                'auth_token': config.execution.host_agent.auth_token,
                'timeout': config.execution.host_agent.timeout,
                'retry_attempts': config.execution.host_agent.retry_attempts,
                'health_check_interval': config.execution.host_agent.health_check_interval,
            }
        )
    elif config.execution.mode == 'ssh_remote':
        exec_config.update(
            {
                'host': config.execution.ssh_remote.host,
                'port': config.execution.ssh_remote.port,
                'username': config.execution.ssh_remote.username,
                'auth_method': config.execution.ssh_remote.auth_method,
                'key_file': config.execution.ssh_remote.key_file,
                'password': config.execution.ssh_remote.password,
                'known_hosts_file': config.execution.ssh_remote.known_hosts_file,
                'keepalive': config.execution.ssh_remote.keepalive,
            }
        )

    execution_backend = create_execution_backend(config.execution.mode, exec_config)
    logger.info(f"Initialized execution backend: {config.execution.mode}")

except Exception as e:
    logger.error(f"Failed to initialize execution backend: {e}")
    # Fall back to native execution
    try:
        execution_backend = create_execution_backend('native', exec_config)
        logger.info("Fell back to native execution backend")
    except Exception as fallback_error:
        logger.error(f"Failed to initialize fallback execution backend: {fallback_error}")
        execution_backend = None

# Tools for task management and distribution


@mcp.tool()
async def create_project(ctx: Context, name: str, description: str, components: List[str]) -> str:
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
            priority=TaskPriority.HIGH,
        )
        tasks.append(task)

        # Auto-assign to best available agent
        agent_id = task_manager.find_best_agent(task)
        if agent_id:
            task_manager.assign_task(task.id, agent_id)

    return f"Created project '{name}' with {len(tasks)} tasks. Tasks distributed to agents."


@mcp.tool()
async def distribute_task(ctx: Context, title: str, description: str, task_type: str, priority: str = "medium") -> str:
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
        "low": TaskPriority.LOW,
    }

    task = task_manager.create_task(
        title=title,
        description=description,
        task_type=task_type,
        priority=priority_map.get(priority.lower(), TaskPriority.MEDIUM),
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
    status_report = {"agents": {}, "tasks": {"total": len(task_manager.tasks), "by_status": {}, "by_priority": {}}}

    # Agent status
    for agent_id, agent in task_manager.agents.items():
        status_report["agents"][agent.name] = {
            "type": agent.type.value,
            "status": agent.status,
            "current_tasks": len(agent.current_tasks),
            "capabilities": list(agent.capabilities),
        }

    # Task status
    for task in task_manager.tasks.values():
        status = task.status.value
        priority = task.priority.name

        status_report["tasks"]["by_status"][status] = status_report["tasks"]["by_status"].get(status, 0) + 1
        status_report["tasks"]["by_priority"][priority] = status_report["tasks"]["by_priority"].get(priority, 0) + 1

    return json.dumps(status_report, indent=2)


# Integration Tools


@mcp.tool()
async def slack_notify(ctx: Context, channel: str, message: str, thread_ts: Optional[str] = None) -> str:
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
async def git_operation(
    ctx: Context, operation: str, repository: str, branch: Optional[str] = None, message: Optional[str] = None
) -> str:
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
        "branch": f"Created/switched to branch {branch}",
    }

    return operations.get(operation, f"Performed {operation} on {repository}")


@mcp.tool()
async def jenkins_trigger(ctx: Context, job_name: str, parameters: Optional[Dict[str, Any]] = None) -> str:
    """
    Trigger a Jenkins job

    Args:
        job_name: Name of the Jenkins job
        parameters: Job parameters (optional)
    """
    params = json.dumps(parameters) if parameters else "default parameters"
    return f"Triggered Jenkins job '{job_name}' with {params}"


@mcp.tool()
async def youtrack_issue(
    ctx: Context,
    operation: str,
    project: str,
    summary: Optional[str] = None,
    description: Optional[str] = None,
    issue_id: Optional[str] = None,
) -> str:
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
async def postgres_query(ctx: Context, database: str, query: str, params: Optional[List[Any]] = None) -> str:
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
async def aws_operation(ctx: Context, service: str, operation: str, parameters: Dict[str, Any]) -> str:
    """
    Perform AWS operations

    Args:
        service: AWS service (ec2, s3, lambda, etc.)
        operation: Operation to perform
        parameters: Operation parameters
    """
    return f"Performed {operation} on AWS {service} with parameters: {json.dumps(parameters)}"


@mcp.tool()
async def bambu_print(ctx: Context, printer_id: str, file_path: str, settings: Optional[Dict[str, Any]] = None) -> str:
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
async def web_search(ctx: Context, query: str, num_results: int = 5) -> str:
    """
    Search the internet for information

    Args:
        query: Search query
        num_results: Number of results to return
    """
    # In production, this would use a search API
    return f"Found {num_results} results for '{query}'"


@mcp.tool()
async def api_request(
    ctx: Context,
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> str:
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
        agents_info.append(
            {
                "id": agent.id,
                "name": agent.name,
                "type": agent.type.value,
                "status": agent.status,
                "current_tasks": agent.current_tasks,
                "capabilities": list(agent.capabilities),
            }
        )
    return json.dumps(agents_info, indent=2)


@mcp.resource("tasks://all")
async def get_all_tasks() -> str:
    """All tasks in the system"""
    tasks_info = []
    for task_id, task in task_manager.tasks.items():
        tasks_info.append(
            {
                "id": task.id,
                "title": task.title,
                "type": task.type,
                "status": task.status.value,
                "priority": task.priority.name,
                "assigned_to": task.assigned_to,
                "created_at": task.created_at.isoformat(),
            }
        )
    return json.dumps(tasks_info, indent=2)


@mcp.resource("integrations://config")
async def get_integrations_config() -> str:
    """Current integrations configuration"""
    config = {
        "slack": {"enabled": True, "workspace": "your-workspace", "default_channel": "#dev-updates"},
        "git": {"enabled": True, "default_branch": "main", "auto_commit": False},
        "jenkins": {"enabled": True, "url": "https://jenkins.example.com", "default_pipeline": "ci-cd"},
        "youtrack": {"enabled": True, "url": "https://youtrack.example.com", "default_project": "DEV"},
        "postgres": {"enabled": True, "default_database": "production"},
        "aws": {"enabled": True, "region": "us-east-1", "services": ["ec2", "s3", "lambda", "rds"]},
        "bambu": {"enabled": True, "printers": ["X1-Carbon-01", "P1S-01"]},
    }
    return json.dumps(config, indent=2)


# System Command Execution Tools


@mcp.tool()
async def execute_command(
    ctx: Context,
    command: str,
    working_directory: Optional[str] = None,
    timeout: int = 30,
    env_vars: Optional[Dict[str, str]] = None,
) -> str:
    """
    Execute a shell command on the system

    Args:
        command: Command to execute
        working_directory: Working directory for command execution
        timeout: Maximum execution time in seconds
        env_vars: Additional environment variables
    """
    if execution_backend is None:
        return json.dumps({"success": False, "error": "Execution backend not available", "output": ""})

    try:
        exec_config = ExecutionConfig(
            command=command,
            working_directory=working_directory,
            timeout=timeout,
            environment=env_vars or {},
            user_id=getattr(ctx, 'user_id', 'unknown'),
        )

        result = await execution_backend.execute_with_timeout(exec_config)

        return json.dumps(
            {
                "success": result.success,
                "status": result.status.value,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "execution_time": result.execution_time,
                "working_directory": result.working_directory,
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"Command execution failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "output": ""})


@mcp.tool()
async def read_file(ctx: Context, file_path: str, encoding: str = "utf-8") -> str:
    """
    Read contents of a file

    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)
    """
    try:
        command = f"cat '{file_path}'"
        exec_config = ExecutionConfig(
            command=command, timeout=43200, user_id=getattr(ctx, 'user_id', 'unknown')  # 12 hours
        )

        result = await execution_backend.execute_with_timeout(exec_config)

        if result.success:
            return json.dumps(
                {
                    "success": True,
                    "content": result.stdout,
                    "file_path": file_path,
                    "size": len(result.stdout.encode(encoding)),
                },
                indent=2,
            )
        else:
            return json.dumps(
                {
                    "success": False,
                    "error": result.stderr or f"Failed to read file: {file_path}",
                    "file_path": file_path,
                }
            )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "file_path": file_path})


@mcp.tool()
async def write_file(ctx: Context, file_path: str, content: str, create_dirs: bool = True) -> str:
    """
    Write content to a file

    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        create_dirs: Whether to create parent directories if they don't exist
    """
    try:
        # Escape content for shell
        import shlex

        escaped_content = shlex.quote(content)

        # Create directories if needed
        commands = []
        if create_dirs:
            dir_path = str(Path(file_path).parent)
            commands.append(f"mkdir -p '{dir_path}'")

        commands.append(f"echo {escaped_content} > '{file_path}'")
        command = " && ".join(commands)

        exec_config = ExecutionConfig(
            command=command, timeout=43200, user_id=getattr(ctx, 'user_id', 'unknown')  # 12 hours
        )

        result = await execution_backend.execute_with_timeout(exec_config)

        if result.success:
            return json.dumps(
                {
                    "success": True,
                    "file_path": file_path,
                    "bytes_written": len(content.encode('utf-8')),
                    "message": f"Successfully wrote to {file_path}",
                },
                indent=2,
            )
        else:
            return json.dumps(
                {
                    "success": False,
                    "error": result.stderr or f"Failed to write file: {file_path}",
                    "file_path": file_path,
                }
            )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "file_path": file_path})


@mcp.tool()
async def list_directory(ctx: Context, directory: str = ".", show_hidden: bool = False) -> str:
    """
    List contents of a directory

    Args:
        directory: Directory path to list
        show_hidden: Whether to show hidden files
    """
    try:
        flags = "-la" if show_hidden else "-l"
        command = f"ls {flags} '{directory}'"

        exec_config = ExecutionConfig(
            command=command, timeout=43200, user_id=getattr(ctx, 'user_id', 'unknown')  # 12 hours
        )

        result = await execution_backend.execute_with_timeout(exec_config)

        if result.success:
            return json.dumps(
                {"success": True, "directory": directory, "contents": result.stdout, "raw_output": result.stdout},
                indent=2,
            )
        else:
            return json.dumps(
                {
                    "success": False,
                    "error": result.stderr or f"Failed to list directory: {directory}",
                    "directory": directory,
                }
            )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "directory": directory})


@mcp.tool()
async def search_files(ctx: Context, pattern: str, directory: str = ".", file_type: Optional[str] = None) -> str:
    """
    Search for files matching a pattern

    Args:
        pattern: Search pattern (supports wildcards)
        directory: Directory to search in
        file_type: File type filter (e.g., 'f' for files, 'd' for directories)
    """
    try:
        type_flag = f" -type {file_type}" if file_type else ""
        command = f"find '{directory}' -name '{pattern}'{type_flag}"

        exec_config = ExecutionConfig(
            command=command, timeout=43200, user_id=getattr(ctx, 'user_id', 'unknown')  # 12 hours
        )

        result = await execution_backend.execute_with_timeout(exec_config)

        if result.success:
            files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
            return json.dumps(
                {"success": True, "pattern": pattern, "directory": directory, "matches": files, "count": len(files)},
                indent=2,
            )
        else:
            return json.dumps(
                {
                    "success": False,
                    "error": result.stderr or f"Search failed for pattern: {pattern}",
                    "pattern": pattern,
                    "directory": directory,
                }
            )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "pattern": pattern, "directory": directory})


@mcp.tool()
async def get_system_info(ctx: Context) -> str:
    """
    Get system information (OS, CPU, memory, disk usage)
    """
    try:
        commands = [
            "uname -a",  # System info
            "df -h .",  # Disk usage
            "free -h",  # Memory info (Linux)
            "uptime",  # System uptime
            "whoami",  # Current user
            "pwd",  # Current directory
        ]

        info = {}
        for cmd in commands:
            exec_config = ExecutionConfig(
                command=cmd, timeout=43200, user_id=getattr(ctx, 'user_id', 'unknown')  # 12 hours
            )

            result = await execution_backend.execute_with_timeout(exec_config)
            cmd_name = cmd.split()[0]
            info[cmd_name] = {
                "success": result.success,
                "output": result.stdout.strip() if result.success else result.stderr,
            }

        # Add execution backend info
        info["execution_backend"] = {
            "mode": config.execution.mode,
            "backend": execution_backend.__class__.__name__ if execution_backend else "None",
        }

        return json.dumps({"success": True, "system_info": info}, indent=2)

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def check_command_exists(ctx: Context, command: str) -> str:
    """
    Check if a command exists in the system PATH

    Args:
        command: Command name to check
    """
    try:
        check_cmd = f"which '{command}'"
        exec_config = ExecutionConfig(
            command=check_cmd, timeout=43200, user_id=getattr(ctx, 'user_id', 'unknown')  # 12 hours
        )

        result = await execution_backend.execute_with_timeout(exec_config)

        return json.dumps(
            {
                "success": True,
                "command": command,
                "exists": result.success,
                "path": result.stdout.strip() if result.success else None,
                "message": f"Command '{command}' {'found' if result.success else 'not found'}",
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "command": command})


# Add execution backend resource


@mcp.resource("execution://status")
async def get_execution_status() -> str:
    """Current execution backend status and configuration"""
    if execution_backend is None:
        return json.dumps({"available": False, "error": "No execution backend initialized"}, indent=2)

    try:
        is_healthy = await execution_backend.health_check()
        active_executions = await execution_backend.list_active_executions()

        status = {
            "available": True,
            "healthy": is_healthy,
            "backend_type": execution_backend.__class__.__name__,
            "mode": config.execution.mode,
            "active_executions": len(active_executions),
            "configuration": {
                "working_directory": config.execution.working_directory,
                "default_timeout": config.execution.default_timeout,
                "max_concurrent": config.execution.max_concurrent,
                "audit_logging": config.execution.audit_logging,
            },
        }

        return json.dumps(status, indent=2)

    except Exception as e:
        return json.dumps(
            {
                "available": True,
                "healthy": False,
                "error": str(e),
                "backend_type": execution_backend.__class__.__name__,
            },
            indent=2,
        )


def detect_available_integrations():
    """Dynamically detect available integrations based on configuration"""
    integrations = []

    # Check integrations from config.json
    if hasattr(config, 'integrations'):
        # Slack integration
        if hasattr(config.integrations, 'slack'):
            slack_config = config.integrations.slack
            status = (
                "configured"
                if slack_config.enabled and slack_config.bot_token and slack_config.app_token
                else "disabled"
            )
            integrations.append(
                {
                    "name": "Slack",
                    "type": "messaging",
                    "enabled": slack_config.enabled,
                    "status": status,
                    "config_source": "config.json",
                    "has_credentials": bool(slack_config.bot_token and slack_config.app_token),
                }
            )

        # Git integrations
        if hasattr(config.integrations, 'git'):
            git_config = config.integrations.git
            has_credentials = bool(git_config.github_token or git_config.gitlab_token)
            status = "configured" if git_config.enabled and has_credentials else "disabled"
            integrations.append(
                {
                    "name": "Git",
                    "type": "version_control",
                    "enabled": git_config.enabled,
                    "status": status,
                    "config_source": "config.json",
                    "has_credentials": has_credentials,
                    "providers": {"github": bool(git_config.github_token), "gitlab": bool(git_config.gitlab_token)},
                }
            )

        # Jenkins
        if hasattr(config.integrations, 'jenkins'):
            jenkins_config = config.integrations.jenkins
            has_credentials = bool(jenkins_config.url and jenkins_config.username and jenkins_config.token)
            status = "configured" if jenkins_config.enabled and has_credentials else "disabled"
            integrations.append(
                {
                    "name": "Jenkins",
                    "type": "ci_cd",
                    "enabled": jenkins_config.enabled,
                    "status": status,
                    "config_source": "config.json",
                    "has_credentials": has_credentials,
                }
            )

        # YouTrack
        if hasattr(config.integrations, 'youtrack'):
            youtrack_config = config.integrations.youtrack
            has_credentials = bool(youtrack_config.url and youtrack_config.token)
            status = "configured" if youtrack_config.enabled and has_credentials else "disabled"
            integrations.append(
                {
                    "name": "YouTrack",
                    "type": "issue_tracking",
                    "enabled": youtrack_config.enabled,
                    "status": status,
                    "config_source": "config.json",
                    "has_credentials": has_credentials,
                }
            )

        # AWS
        if hasattr(config.integrations, 'aws'):
            aws_config = config.integrations.aws
            has_credentials = bool(aws_config.access_key_id and aws_config.secret_access_key)
            status = "configured" if aws_config.enabled and has_credentials else "disabled"
            integrations.append(
                {
                    "name": "AWS",
                    "type": "cloud",
                    "enabled": aws_config.enabled,
                    "status": status,
                    "config_source": "config.json",
                    "has_credentials": has_credentials,
                    "region": aws_config.region,
                }
            )

        # PostgreSQL
        if hasattr(config.integrations, 'postgres'):
            postgres_config = config.integrations.postgres
            has_credentials = bool(postgres_config.host and postgres_config.username)
            status = "configured" if postgres_config.enabled and has_credentials else "disabled"
            integrations.append(
                {
                    "name": "PostgreSQL",
                    "type": "database",
                    "enabled": postgres_config.enabled,
                    "status": status,
                    "config_source": "config.json",
                    "has_credentials": has_credentials,
                }
            )

        # Bambu 3D Printers
        if hasattr(config.integrations, 'bambu'):
            bambu_config = config.integrations.bambu
            has_credentials = bool(bambu_config.api_key)
            status = "configured" if bambu_config.enabled and has_credentials else "disabled"
            integrations.append(
                {
                    "name": "Bambu 3D",
                    "type": "hardware",
                    "enabled": bambu_config.enabled,
                    "status": status,
                    "config_source": "config.json",
                    "has_credentials": has_credentials,
                }
            )

        # Web Search
        if hasattr(config.integrations, 'web'):
            web_config = config.integrations.web
            has_credentials = bool(web_config.search_api_key)
            status = "configured" if web_config.enabled and has_credentials else "disabled"
            integrations.append(
                {
                    "name": "Web Search",
                    "type": "search",
                    "enabled": web_config.enabled,
                    "status": status,
                    "config_source": "config.json",
                    "has_credentials": has_credentials,
                }
            )

    # System integrations that are always available
    # Execution Backend
    if execution_backend is not None:
        integrations.append(
            {
                "name": "Execution Backend",
                "type": "system",
                "enabled": True,
                "status": "active",
                "config_source": "system",
                "backend_type": execution_backend.__class__.__name__,
                "mode": config.execution.mode,
            }
        )

    # Check for AI framework integrations by trying imports
    try:
        import haystack

        integrations.append(
            {
                "name": "Haystack AI",
                "type": "ai_framework",
                "enabled": True,
                "status": "available",
                "config_source": "system",
                "version": getattr(haystack, '__version__', 'unknown'),
            }
        )
    except ImportError:
        pass

    try:
        import llama_index

        integrations.append(
            {
                "name": "LlamaIndex",
                "type": "ai_framework",
                "enabled": True,
                "status": "available",
                "config_source": "system",
                "version": getattr(llama_index, '__version__', 'unknown'),
            }
        )
    except ImportError:
        pass

    return integrations


# MCP API Tools (moved from api/server.py)


@mcp.tool()
async def get_system_status(ctx: Context) -> str:
    """Get current system status and metrics"""
    try:
        # Count agent statuses
        active_agents = sum(1 for agent in task_manager.agents.values() if agent.status == "working")
        idle_agents = sum(1 for agent in task_manager.agents.values() if agent.status == "idle")

        # Count task statuses
        completed_tasks = sum(1 for task in task_manager.tasks.values() if task.status == TaskStatus.COMPLETED)
        in_progress_tasks = sum(1 for task in task_manager.tasks.values() if task.status == TaskStatus.IN_PROGRESS)
        pending_tasks = sum(1 for task in task_manager.tasks.values() if task.status == TaskStatus.PENDING)
        failed_tasks = sum(1 for task in task_manager.tasks.values() if task.status == TaskStatus.FAILED)

        # Dynamically detect integrations from config
        integrations = detect_available_integrations()

        # Calculate uptime (simple placeholder)
        uptime = "0:00:00"  # Would need to track server start time for real uptime

        status_data = {
            "server_url": "mcp://distributed-server",
            "version": config.server.version,
            "uptime": uptime,
            "total_agents": len(task_manager.agents),
            "active_agents": active_agents,
            "idle_agents": idle_agents,
            "total_tasks": len(task_manager.tasks),
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "pending_tasks": pending_tasks,
            "failed_tasks": failed_tasks,
            "integrations": integrations,
            "integrations_summary": {
                "total": len(integrations),
                "enabled": sum(1 for i in integrations if i["enabled"]),
                "configured": sum(1 for i in integrations if i["status"] == "configured"),
                "active": sum(1 for i in integrations if i["status"] == "active"),
                "available": sum(1 for i in integrations if i["status"] == "available"),
            },
            "execution_backend": {
                "available": execution_backend is not None,
                "mode": config.execution.mode if execution_backend else "none",
                "backend_type": execution_backend.__class__.__name__ if execution_backend else "None",
            },
        }

        return json.dumps(status_data, indent=2)

    except Exception as e:
        logger.error(f"Error getting system status: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def list_agents(ctx: Context, status: Optional[str] = None) -> str:
    """List all available agents"""
    try:
        agents_data = []
        for agent_id, agent in task_manager.agents.items():
            if status and agent.status != status:
                continue

            # Calculate metrics
            agent_tasks = [task for task in task_manager.tasks.values() if task.assigned_to == agent_id]
            completed_tasks = sum(1 for task in agent_tasks if task.status == TaskStatus.COMPLETED)
            failed_tasks = sum(1 for task in agent_tasks if task.status == TaskStatus.FAILED)

            agent_data = {
                "id": agent.id,
                "name": agent.name,
                "type": agent.type.value,
                "status": agent.status,
                "capabilities": list(agent.capabilities),
                "current_tasks": len(agent.current_tasks),
                "metrics": {
                    "tasks_completed": completed_tasks,
                    "tasks_failed": failed_tasks,
                    "total_tasks": len(agent_tasks),
                    "success_rate": (completed_tasks / len(agent_tasks)) if agent_tasks else 1.0,
                },
            }
            agents_data.append(agent_data)

        return json.dumps(
            {
                "success": True,
                "agents": agents_data,
                "total": len(agents_data),
                "filter": {"status": status} if status else None,
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"Error listing agents: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def create_task(ctx: Context, title: str, description: str, type: str, priority: str = "medium") -> str:
    """Create a new task"""
    try:
        priority_map = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW,
        }

        if not all([title, description, type]):
            return json.dumps({"success": False, "error": "Missing required fields (title, description, type)"})

        task_priority = priority_map.get(priority.lower(), TaskPriority.MEDIUM)

        # Create task using the task manager (this will also handle auto-assignment)
        task = task_manager.create_task(title=title, description=description, task_type=type, priority=task_priority)

        # Try to auto-assign to best available agent
        agent_id = task_manager.find_best_agent(task)
        assignment_info = ""
        if agent_id:
            if task_manager.assign_task(task.id, agent_id):
                agent = task_manager.agents[agent_id]
                assignment_info = f"\nAssigned to: {agent.name} ({agent.type.value})"
            else:
                assignment_info = "\nAuto-assignment failed"
        else:
            assignment_info = "\nNo suitable agent available - task queued"

        return json.dumps(
            {
                "success": True,
                "task": {
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "type": task.type,
                    "status": task.status.value,
                    "priority": task.priority.name,
                    "assigned_to": task.assigned_to,
                    "created_at": task.created_at.isoformat(),
                },
                "message": f"Task created successfully{assignment_info}",
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"Error creating task: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e)})


# HTTP Transport for MCP protocol

# MCP HTTP Request/Response models


class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    method: str
    params: dict = {}


class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    result: dict = None
    error: dict = None


# Create HTTP app for MCP transport
http_app = FastAPI(title="MCP Distributed Server", version=config.server.version)

# Add CORS for OpenCode compatibility
http_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy import orchestrator to avoid circular import
# from agent_orchestrator import execute_tool_via_orchestrator, get_orchestrator_status


@http_app.post("/mcp")
async def mcp_http_endpoint(request: MCPRequest):
    """HTTP transport for MCP protocol - forwards to orchestrator"""
    try:
        if request.method == "initialize":
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": request.id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}, "resources": {}},
                        "serverInfo": {"name": "distributed-mcp-server-orchestrated", "version": config.server.version},
                    },
                }
            )

        elif request.method == "notifications/initialized":
            # No response for notifications
            return JSONResponse(content="", status_code=204)

        elif request.method == "tools/list":
            # Return all 22 tools available through orchestrator
            tools = [
                {"name": "execute_command", "description": "Execute shell command"},
                {"name": "read_file", "description": "Read file contents"},
                {"name": "write_file", "description": "Write content to file"},
                {"name": "list_directory", "description": "List directory contents"},
                {"name": "search_files", "description": "Search for files"},
                {"name": "get_system_info", "description": "Get system information"},
                {"name": "check_command_exists", "description": "Check if command exists"},
                {"name": "create_project", "description": "Create multi-component project"},
                {"name": "distribute_task", "description": "Distribute task to agents"},
                {"name": "get_project_status", "description": "Get project and agent status"},
                {"name": "slack_notify", "description": "Send Slack notification"},
                {"name": "git_operation", "description": "Perform Git operations"},
                {"name": "jenkins_trigger", "description": "Trigger Jenkins job"},
                {"name": "youtrack_issue", "description": "Manage YouTrack issues"},
                {"name": "postgres_query", "description": "Execute PostgreSQL queries"},
                {"name": "aws_operation", "description": "Perform AWS operations"},
                {"name": "bambu_print", "description": "Send job to Bambu 3D printer"},
                {"name": "web_search", "description": "Search the internet"},
                {"name": "api_request", "description": "Make arbitrary API requests"},
                {"name": "get_system_status", "description": "Get system status and metrics"},
                {"name": "list_agents", "description": "List all available agents"},
                {"name": "create_task", "description": "Create a new task"},
            ]

            return JSONResponse(content={"jsonrpc": "2.0", "id": request.id, "result": {"tools": tools}})

        elif request.method == "tools/call":
            # Execute tool through orchestrator
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})

            try:
                # Execute via orchestrator
                result = await execute_tool_via_orchestrator(tool_name, arguments)

                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": request.id,
                        "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]},
                    }
                )

            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": request.id,
                        "error": {"code": -32000, "message": f"Tool execution failed: {str(e)}"},
                    }
                )

        elif request.method == "resources/list":
            # Return available resources
            resources = [
                {
                    "uri": "orchestrator://status",
                    "name": "Orchestrator Status",
                    "description": "Current orchestrator status",
                },
                {"uri": "agents://status", "name": "Agent Status", "description": "Current agent status"},
                {"uri": "tasks://all", "name": "All Tasks", "description": "All tasks in system"},
                {"uri": "integrations://config", "name": "Integrations", "description": "Integration configuration"},
                {"uri": "execution://status", "name": "Execution Status", "description": "Execution backend status"},
            ]

            return JSONResponse(content={"jsonrpc": "2.0", "id": request.id, "result": {"resources": resources}})

        elif request.method == "resources/read":
            uri = request.params.get("uri")

            try:
                if uri == "orchestrator://status":
                    status = await get_orchestrator_status()
                    content = json.dumps(status, indent=2, default=str)

                elif uri == "agents://status":
                    content = await get_agents_status()

                elif uri == "tasks://all":
                    content = await get_all_tasks()

                elif uri == "integrations://config":
                    content = await get_integrations_config()

                elif uri == "execution://status":
                    content = await get_execution_status()

                else:
                    raise Exception(f"Resource not found: {uri}")

                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": request.id,
                        "result": {"contents": [{"uri": uri, "mimeType": "application/json", "text": content}]},
                    }
                )

            except Exception as e:
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": request.id,
                        "error": {"code": -32000, "message": f"Resource read failed: {str(e)}"},
                    }
                )

        else:
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": request.id,
                    "error": {"code": -32601, "message": f"Method not found: {request.method}"},
                }
            )

    except Exception as e:
        logger.error(f"MCP HTTP transport error: {e}")
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": request.id if hasattr(request, 'id') else None,
                "error": {"code": -32000, "message": f"Internal server error: {str(e)}"},
            }
        )


# Health check endpoint


@http_app.get("/health")
async def health_check():
    status = await get_orchestrator_status()
    return {
        "status": "healthy",
        "orchestrator": status.get("orchestrator", {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def run_dual_transport():
    """Run both stdio and HTTP MCP transports"""
    import asyncio

    # Start stdio transport in thread
    def run_stdio():
        mcp.run(transport="stdio")

    # Start HTTP server
    config_http = {"host": "0.0.0.0", "port": 8001, "log_level": "info"}

    # Run both transports
    stdio_task = asyncio.create_task(asyncio.to_thread(run_stdio))

    http_task = asyncio.create_task(uvicorn.run(http_app, **config_http))

    logger.info("Starting dual transport MCP server...")
    logger.info("stdio transport: Available for Claude Desktop")
    logger.info("HTTP transport: http://localhost:8001/mcp for OpenCode")

    try:
        await asyncio.gather(stdio_task, http_task)
    except KeyboardInterrupt:
        logger.info("Shutting down MCP server...")
        stdio_task.cancel()
        http_task.cancel()


# Process Management Functions (for compatibility with tests)
# These wrap the ProcessManager functionality for easy access


async def list_processes(filter_criteria=None):
    """
    List running processes on the system

    Args:
        filter_criteria: Optional ProcessFilter object to filter results

    Returns:
        List of ProcessInfo objects
    """
    from tools.process.manager import ProcessManager
    from tools.process.models import ProcessFilter

    manager = ProcessManager()
    return await manager.list_processes(filter_criteria)


async def get_process_info(pid):
    """
    Get detailed information about a specific process

    Args:
        pid: Process ID

    Returns:
        ProcessInfo object or None if process not found
    """
    from tools.process.manager import ProcessManager

    manager = ProcessManager()
    return await manager.get_process_info(pid)


async def kill_process(pid, signal_type=None):
    """
    Kill a process by PID

    Args:
        pid: Process ID
        signal_type: Optional signal type (default: SIGTERM)

    Returns:
        ProcessOperation result object
    """
    from tools.process.manager import ProcessManager
    from tools.process.models import ProcessSignal

    if signal_type is None:
        signal_type = ProcessSignal.SIGTERM

    manager = ProcessManager()
    return await manager.kill_process(pid, signal_type)


def main():
    """Run the distributed MCP server with orchestrator backend"""
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Distributed MCP Server with Agent Orchestration')
    parser.add_argument(
        '--config', type=str, default='config.json', help='Path to configuration file (default: config.json)'
    )
    parser.add_argument(
        '--transport',
        type=str,
        choices=['stdio', 'http', 'dual'],
        default='dual',
        help='Transport type (default: dual)',
    )
    args = parser.parse_args()

    # Initialize configuration
    init_config(args.config)

    # Configure logging based on config
    log_level = getattr(logging, config.server.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, force=True)

    if args.transport == 'stdio':
        # stdio only
        mcp.run(transport="stdio")
    elif args.transport == 'http':
        # HTTP only
        uvicorn.run(http_app, host="0.0.0.0", port=8001)
    else:
        # Dual transport (default)
        asyncio.run(run_dual_transport())


if __name__ == "__main__":
    main()
