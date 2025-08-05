#!/usr/bin/env python3

import asyncio
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from mcp.types import TextContent

from distributed_server import Agent, AgentType, DistributedTaskManager, Task, TaskPriority, TaskStatus, task_manager
from integrations import (
    api_integration,
    aws_integration,
    bambu_integration,
    git_integration,
    jenkins_integration,
    postgres_integration,
    slack_integration,
    web_integration,
    youtrack_integration,
)
from integrations_config import integrations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MCP server
mcp = FastMCP("enhanced-distributed-mcp-server", "2.1.0")

# Enhanced Tools with real integrations


@mcp.tool()
async def slack_send(ctx: Context, channel: str, message: str, thread_ts: Optional[str] = None) -> str:
    """
    Send a message to Slack

    Args:
        channel: Slack channel (e.g., #general or C1234567890)
        message: Message text
        thread_ts: Thread timestamp for replies (optional)
    """
    try:
        result = await slack_integration.send_message(channel, message, thread_ts)
        if result.get("ok"):
            return f"Message sent to {channel} (ts: {result['ts']})"
        else:
            return f"Failed to send message: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error sending Slack message: {str(e)}"


@mcp.tool()
async def git_create_pr(
    ctx: Context, repo: str, title: str, body: str, head_branch: str, base_branch: str = "main"
) -> str:
    """
    Create a pull request on GitHub

    Args:
        repo: Repository (owner/name format)
        title: PR title
        body: PR description
        head_branch: Source branch
        base_branch: Target branch (default: main)
    """
    try:
        result = await git_integration.create_pr(repo, title, body, head_branch, base_branch)
        if "number" in result:
            return f"Created PR #{result['number']}: {result['html_url']}"
        else:
            return f"Failed to create PR: {result}"
    except Exception as e:
        return f"Error creating PR: {str(e)}"


@mcp.tool()
async def jenkins_build(ctx: Context, job_name: str, parameters: Optional[Dict[str, Any]] = None) -> str:
    """
    Trigger a Jenkins build

    Args:
        job_name: Jenkins job name
        parameters: Build parameters (optional)
    """
    try:
        result = await jenkins_integration.trigger_job(job_name, parameters)
        if result["status"] == 201:
            return f"Jenkins job '{job_name}' triggered successfully. Queue URL: {result['location']}"
        else:
            return f"Failed to trigger job: HTTP {result['status']}"
    except Exception as e:
        return f"Error triggering Jenkins job: {str(e)}"


@mcp.tool()
async def youtrack_create(
    ctx: Context, project: str, summary: str, description: str = "", priority: str = "Normal"
) -> str:
    """
    Create a YouTrack issue

    Args:
        project: Project ID
        summary: Issue summary
        description: Issue description (optional)
        priority: Issue priority (optional)
    """
    try:
        result = await youtrack_integration.create_issue(project, summary, description)
        if "id" in result:
            return f"Created YouTrack issue {result['id']}: {result.get('summary', summary)}"
        else:
            return f"Failed to create issue: {result}"
    except Exception as e:
        return f"Error creating YouTrack issue: {str(e)}"


@mcp.tool()
async def postgres_execute(
    ctx: Context, query: str, params: Optional[List[Any]] = None, database: Optional[str] = None
) -> str:
    """
    Execute a PostgreSQL query

    Args:
        query: SQL query to execute
        params: Query parameters (optional)
        database: Database name (optional, uses default if not specified)
    """
    try:
        # For SELECT queries
        if query.strip().upper().startswith("SELECT"):
            results = await postgres_integration.execute_query(query, params)
            return json.dumps(results, indent=2, default=str)
        else:
            # For INSERT, UPDATE, DELETE
            result = await postgres_integration.execute_command(query, params)
            return f"Query executed successfully: {result}"
    except Exception as e:
        return f"Error executing PostgreSQL query: {str(e)}"


@mcp.tool()
async def aws_ec2(
    ctx: Context, operation: str, instance_id: Optional[str] = None, instance_type: Optional[str] = None
) -> str:
    """
    Manage AWS EC2 instances

    Args:
        operation: Operation to perform (list, start, stop, create)
        instance_id: Instance ID (for start/stop)
        instance_type: Instance type (for create)
    """
    try:
        params = {}
        if instance_id:
            params["instance_id"] = instance_id
        if instance_type:
            params["instance_type"] = instance_type

        result = await aws_integration.ec2_operation(operation, params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error with EC2 operation: {str(e)}"


@mcp.tool()
async def aws_s3(
    ctx: Context,
    operation: str,
    bucket: Optional[str] = None,
    key: Optional[str] = None,
    local_path: Optional[str] = None,
) -> str:
    """
    Manage AWS S3 storage

    Args:
        operation: Operation (list_buckets, upload, download, delete)
        bucket: Bucket name
        key: Object key
        local_path: Local file path (for upload/download)
    """
    try:
        params = {}
        if bucket:
            params["bucket"] = bucket
        if key:
            params["key"] = key
        if local_path:
            params["local_path"] = local_path

        result = await aws_integration.s3_operation(operation, params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error with S3 operation: {str(e)}"


@mcp.tool()
async def bambu_send_print(
    ctx: Context, printer_id: str, file_path: str, material: str = "PLA", quality: str = "standard"
) -> str:
    """
    Send a print job to Bambu 3D printer

    Args:
        printer_id: Printer ID (e.g., X1-Carbon-01)
        file_path: Path to 3MF/STL file
        material: Filament material (default: PLA)
        quality: Print quality (default: standard)
    """
    try:
        settings = {"material": material, "quality": quality}
        result = await bambu_integration.send_print_job(printer_id, file_path, settings)

        if "error" in result:
            return f"Error: {result['error']}"
        else:
            return f"Print job sent to {printer_id}. Job ID: {result['job_id']}, Estimated time: {result['estimated_time']}"
    except Exception as e:
        return f"Error sending print job: {str(e)}"


@mcp.tool()
async def bambu_printer_status(ctx: Context, printer_id: str) -> str:
    """
    Get Bambu printer status

    Args:
        printer_id: Printer ID
    """
    try:
        result = await bambu_integration.get_printer_status(printer_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting printer status: {str(e)}"


@mcp.tool()
async def web_search(ctx: Context, query: str, num_results: int = 5) -> str:
    """
    Search the web for information

    Args:
        query: Search query
        num_results: Number of results to return (default: 5)
    """
    try:
        results = await web_integration.search(query, num_results)
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching web: {str(e)}"


@mcp.tool()
async def api_call(
    ctx: Context,
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Make an arbitrary API call

    Args:
        url: API endpoint URL
        method: HTTP method (GET, POST, PUT, DELETE)
        headers: Request headers (optional)
        body: Request body for POST/PUT (optional)
    """
    try:
        result = await api_integration.request(url, method, headers, body)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return f"Error making API call: {str(e)}"


# Agent coordination tools


@mcp.tool()
async def coordinate_project(ctx: Context, project_name: str, requirements: str, tech_stack: Dict[str, str]) -> str:
    """
    Coordinate a full-stack project across multiple agents

    Args:
        project_name: Name of the project
        requirements: Project requirements description
        tech_stack: Technology choices (frontend, backend, database, etc.)
    """
    try:
        # Create project in YouTrack
        issue_result = await youtrack_integration.create_issue(
            integrations.youtrack.default_project, f"{project_name} - Main Project", requirements
        )

        # Notify Slack
        await slack_integration.send_message(
            integrations.slack.default_channel,
            f"🚀 New project started: *{project_name}*\nTech Stack: {json.dumps(tech_stack)}",
        )

        # Create tasks for each component
        tasks_created = []

        # Frontend task
        if "frontend" in tech_stack:
            task = task_manager.create_task(
                f"{project_name} - Frontend",
                f"Develop frontend using {tech_stack['frontend']}",
                "frontend",
                TaskPriority.HIGH,
            )
            tasks_created.append(task)

        # Backend task
        if "backend" in tech_stack:
            task = task_manager.create_task(
                f"{project_name} - Backend",
                f"Develop backend API using {tech_stack['backend']}",
                "backend",
                TaskPriority.HIGH,
            )
            tasks_created.append(task)

        # Database task
        if "database" in tech_stack:
            task = task_manager.create_task(
                f"{project_name} - Database",
                f"Design and implement database using {tech_stack['database']}",
                "database",
                TaskPriority.HIGH,
            )
            tasks_created.append(task)

        # DevOps task
        task = task_manager.create_task(
            f"{project_name} - DevOps Setup",
            "Set up CI/CD pipeline and deployment infrastructure",
            "devops",
            TaskPriority.MEDIUM,
        )
        tasks_created.append(task)

        # Assign tasks to agents
        assigned_count = 0
        for task in tasks_created:
            agent_id = task_manager.find_best_agent(task)
            if agent_id:
                task_manager.assign_task(task.id, agent_id)
                assigned_count += 1

        return (
            f"Project '{project_name}' initialized:\n"
            f"- Created {len(tasks_created)} tasks\n"
            f"- Assigned {assigned_count} tasks to agents\n"
            f"- YouTrack issue: {issue_result.get('id', 'N/A')}\n"
            f"- Slack notification sent"
        )

    except Exception as e:
        return f"Error coordinating project: {str(e)}"


@mcp.tool()
async def deploy_application(ctx: Context, app_name: str, environment: str = "staging", branch: str = "main") -> str:
    """
    Deploy application through full CI/CD pipeline

    Args:
        app_name: Application name
        environment: Target environment (staging, production)
        branch: Git branch to deploy
    """
    try:
        steps = []

        # Trigger Jenkins build
        jenkins_result = await jenkins_integration.trigger_job(
            f"{app_name}-{environment}", {"BRANCH": branch, "ENVIRONMENT": environment}
        )
        steps.append(f"Jenkins build triggered: {jenkins_result}")

        # Notify Slack
        await slack_integration.send_message(
            integrations.slack.default_channel, f"🚀 Deploying {app_name} to {environment} from branch {branch}"
        )
        steps.append("Slack notification sent")

        # Create deployment task
        task = task_manager.create_task(
            f"Deploy {app_name} to {environment}",
            f"Monitor and verify deployment of {app_name}",
            "devops",
            TaskPriority.CRITICAL,
        )

        # Assign to DevOps agent
        devops_agents = [a for a in task_manager.agents.values() if a.type == AgentType.DEVOPS]
        if devops_agents:
            task_manager.assign_task(task.id, devops_agents[0].id)
            steps.append(f"Monitoring task assigned to {devops_agents[0].name}")

        return f"Deployment initiated:\n" + "\n".join(f"- {step}" for step in steps)

    except Exception as e:
        return f"Error deploying application: {str(e)}"


# Resources for monitoring integrations


@mcp.resource("integrations://status")
async def get_integrations_status() -> str:
    """Current status of all integrations"""
    status = integrations.validate_config()
    summary = integrations.get_config_summary()

    return json.dumps({"configured": status, "summary": summary}, indent=2)


@mcp.resource("agents://workload")
async def get_agent_workload() -> str:
    """Current workload distribution across agents"""
    workload = {}

    for agent_id, agent in task_manager.agents.items():
        tasks = []
        for task_id in agent.current_tasks:
            if task_id in task_manager.tasks:
                task = task_manager.tasks[task_id]
                tasks.append(
                    {"id": task.id, "title": task.title, "priority": task.priority.name, "status": task.status.value}
                )

        workload[agent.name] = {
            "type": agent.type.value,
            "task_count": len(agent.current_tasks),
            "status": agent.status,
            "tasks": tasks,
        }

    return json.dumps(workload, indent=2)


def main():
    """Run the enhanced distributed MCP server"""
    import sys

    # Load environment variables if .env file exists
    integrations.load_env_file()

    # Validate configurations
    config_status = integrations.validate_config()
    logger.info(f"Integration status: {config_status}")

    asyncio.run(mcp.run(transport="stdio"))


if __name__ == "__main__":
    main()
