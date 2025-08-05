#!/usr/bin/env python3
"""
Demo client for the MCP Distributed System

This script demonstrates the capabilities of the distributed MCP system
by creating projects, tasks, and executing integrations.
"""

import asyncio
import json
import time
from typing import Any, Dict

import aiohttp


class MCPDemoClient:
    """Demo client for testing MCP system functionality."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def check_health(self) -> Dict[str, Any]:
        """Check system health."""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()

    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        async with self.session.get(f"{self.base_url}/api/v1/system/status") as response:
            return await response.json()

    async def create_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new project."""
        async with self.session.post(f"{self.base_url}/api/v1/projects", json=project_data) as response:
            return await response.json()

    async def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """Get project status."""
        async with self.session.get(f"{self.base_url}/api/v1/projects/{project_id}/status") as response:
            return await response.json()

    async def create_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a single task."""
        async with self.session.post(f"{self.base_url}/api/v1/tasks", json=task_data) as response:
            return await response.json()

    async def list_agents(self) -> Dict[str, Any]:
        """List all agents."""
        async with self.session.get(f"{self.base_url}/api/v1/agents") as response:
            return await response.json()

    async def list_integrations(self) -> Dict[str, Any]:
        """List all integrations."""
        async with self.session.get(f"{self.base_url}/api/v1/integrations") as response:
            return await response.json()

    async def execute_slack_operation(self, operation: str, parameters: Dict[str, str]) -> Dict[str, Any]:
        """Execute Slack operation."""
        async with self.session.post(
            f"{self.base_url}/api/v1/integrations/slack", json={"operation": operation, "parameters": parameters}
        ) as response:
            return await response.json()

    async def execute_git_operation(self, operation: str, parameters: Dict[str, str]) -> Dict[str, Any]:
        """Execute Git operation."""
        async with self.session.post(
            f"{self.base_url}/api/v1/integrations/git", json={"operation": operation, "parameters": parameters}
        ) as response:
            return await response.json()

    async def execute_aws_operation(self, service: str, operation: str, parameters: Dict[str, str]) -> Dict[str, Any]:
        """Execute AWS operation."""
        async with self.session.post(
            f"{self.base_url}/api/v1/integrations/aws",
            json={"service": service, "operation": operation, "parameters": parameters},
        ) as response:
            return await response.json()


async def demo_basic_functionality():
    """Demonstrate basic system functionality."""
    print("🚀 MCP Distributed System Demo")
    print("=" * 50)

    async with MCPDemoClient() as client:
        try:
            # Check system health
            print("\n1. Checking system health...")
            health = await client.check_health()
            print(f"   Status: {health['status']}")
            print(f"   Services: {health['services']}")

            # Get system status
            print("\n2. Getting system status...")
            status = await client.get_system_status()
            print(f"   Total agents: {status['total_agents']}")
            print(f"   Active agents: {status['active_agents']}")
            print(f"   Total tasks: {status['total_tasks']}")

            # List agents
            print("\n3. Listing available agents...")
            agents = await client.list_agents()
            for agent in agents['agents']:
                print(f"   • {agent['type'].title()} Agent ({agent['status']}) - {len(agent['current_tasks'])} tasks")

            # List integrations
            print("\n4. Listing available integrations...")
            integrations = await client.list_integrations()
            for integration in integrations:
                print(f"   • {integration['integration_name'].title()}: {integration['status']}")

        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_project_creation():
    """Demonstrate project creation and management."""
    print("\n🏗️  Project Creation Demo")
    print("=" * 50)

    async with MCPDemoClient() as client:
        try:
            # Create a sample project
            project_data = {
                "name": "E-Commerce Platform",
                "description": "Full-stack e-commerce application with user authentication, product catalog, and payment processing",
                "requirements": "User registration/login, product browsing, shopping cart, secure checkout, order management, admin dashboard",
                "tech_stack": {
                    "frontend": "React + TypeScript",
                    "backend": "Python + FastAPI",
                    "database": "PostgreSQL",
                    "cache": "Redis",
                    "deployment": "Docker + Kubernetes",
                },
                "priority": "high",
            }

            print("\n1. Creating e-commerce project...")
            project_result = await client.create_project(project_data)
            project_id = project_result['project_id']
            print(f"   ✅ Project created: {project_id}")
            print(f"   📋 Tasks created: {project_result['tasks_created']}")
            print(f"   🎯 Tasks distributed: {project_result['tasks_distributed']}")

            # Wait a moment for tasks to be processed
            print("\n2. Waiting for task processing...")
            await asyncio.sleep(3)

            # Check project status
            print("\n3. Checking project status...")
            project_status = await client.get_project_status(project_id)
            print(f"   📊 Project status: {project_status['status']}")
            print(f"   🔄 Completion: {project_status['completion_percentage']:.1f}%")

            if project_status['tasks']:
                print(f"   📋 Tasks breakdown:")
                for task in project_status['tasks']:
                    print(f"      • {task['title']} ({task['type']}) - {task['status']}")

        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_individual_tasks():
    """Demonstrate individual task creation."""
    print("\n📋 Individual Task Demo")
    print("=" * 50)

    async with MCPDemoClient() as client:
        try:
            tasks = [
                {
                    "title": "Implement user authentication API",
                    "description": "Create JWT-based authentication endpoints with OAuth2 support",
                    "type": "backend",
                    "priority": "high",
                },
                {
                    "title": "Design responsive login interface",
                    "description": "Create mobile-friendly login and registration forms",
                    "type": "frontend",
                    "priority": "medium",
                },
                {
                    "title": "Set up user database schema",
                    "description": "Design and implement user tables with proper indexing",
                    "type": "database",
                    "priority": "high",
                },
                {
                    "title": "Configure deployment pipeline",
                    "description": "Set up CI/CD pipeline with automated testing and deployment",
                    "type": "devops",
                    "priority": "medium",
                },
            ]

            print("\n1. Creating individual tasks...")
            for i, task_data in enumerate(tasks, 1):
                result = await client.create_task(task_data)
                print(f"   {i}. ✅ {task_data['title']}")
                print(f"      Task ID: {result['task_id']}")
                print(f"      Status: {result['status']}")
                await asyncio.sleep(0.5)  # Small delay between tasks

        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_integrations():
    """Demonstrate integration capabilities."""
    print("\n🔗 Integration Demo")
    print("=" * 50)

    async with MCPDemoClient() as client:
        try:
            # Slack integration
            print("\n1. Testing Slack integration...")
            slack_result = await client.execute_slack_operation(
                "send_message", {"channel": "#development", "message": "🚀 MCP Distributed System is now online!"}
            )
            print(f"   ✅ Slack: {slack_result['message']}")
            if slack_result['success']:
                print(f"      Message ID: {slack_result['result_data'].get('message_id', 'N/A')}")

            # Git integration
            print("\n2. Testing Git integration...")
            git_result = await client.execute_git_operation(
                "create_pr",
                {
                    "repo": "company/mcp-project",
                    "title": "Add distributed task management",
                    "branch": "feature/distributed-system",
                },
            )
            print(f"   ✅ Git: {git_result['message']}")
            if git_result['success']:
                print(f"      PR URL: {git_result['result_data'].get('url', 'N/A')}")

            # AWS integration
            print("\n3. Testing AWS integration...")
            aws_result = await client.execute_aws_operation("ec2", "list_instances", {})
            print(f"   ✅ AWS: {aws_result['message']}")
            if aws_result['success']:
                print(f"      Instances: {aws_result['result_data'].get('instance_count', 'N/A')}")

        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_system_monitoring():
    """Demonstrate system monitoring capabilities."""
    print("\n📊 System Monitoring Demo")
    print("=" * 50)

    async with MCPDemoClient() as client:
        try:
            print("\n1. Monitoring system for 10 seconds...")

            for i in range(5):
                status = await client.get_system_status()

                print(f"\n   📊 Status Check {i + 1}:")
                print(f"      Active Agents: {status['active_agents']}/{status['total_agents']}")
                print(f"      Total Tasks: {status['total_tasks']}")
                print(f"      Completed: {status['completed_tasks']}")
                print(f"      Failed: {status['failed_tasks']}")

                # Show agent status
                if status['agents']:
                    busy_agents = [
                        f"{agent['type'].title()}" for agent in status['agents'].values() if agent['status'] == 'busy'
                    ]
                    if busy_agents:
                        print(f"      Busy Agents: {', '.join(busy_agents)}")

                if i < 4:  # Don't wait after the last iteration
                    await asyncio.sleep(2)

        except Exception as e:
            print(f"❌ Error: {e}")


async def run_full_demo():
    """Run the complete demonstration."""
    demos = [
        ("Basic Functionality", demo_basic_functionality),
        ("Project Creation", demo_project_creation),
        ("Individual Tasks", demo_individual_tasks),
        ("Integrations", demo_integrations),
        ("System Monitoring", demo_system_monitoring),
    ]

    print("🎯 MCP Distributed System - Complete Demo")
    print("=" * 60)
    print("This demo will showcase all major features of the system.")
    print("Make sure the MCP system is running: python main.py")
    print("=" * 60)

    for name, demo_func in demos:
        try:
            await demo_func()
            await asyncio.sleep(1)  # Brief pause between demos
        except Exception as e:
            print(f"\n❌ Demo '{name}' failed: {e}")
            continue

    print("\n🎉 Demo completed!")
    print("=" * 60)
    print("Visit http://localhost:8000/docs for interactive API documentation")


if __name__ == "__main__":
    asyncio.run(run_full_demo())
