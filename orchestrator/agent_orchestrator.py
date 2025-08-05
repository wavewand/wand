#!/usr/bin/env python3
"""
Agent Orchestrator - Central coordination for multi-agent system

Reuses existing components from distributed_server.py and adds orchestration layer.
"""

import asyncio
import json
import logging
import os

# Import configuration - use simple default config for now
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import core Agent for task management
# Import from new core modules
from core.task_manager import Agent as CoreAgent
from core.task_manager import AgentType, DistributedTaskManager, Task, TaskPriority, TaskStatus

# Import internal agent transport
from transport.internal_agent import InternalAgentClient, get_agent_registry

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Simple config class for orchestrator


class SimpleConfig:
    def __init__(self):
        self.server = SimpleServerConfig()
        self.distributed = SimpleDistributedConfig()


class SimpleServerConfig:
    def __init__(self):
        self.version = "1.0.0"
        self.http_port = "8080"


class SimpleDistributedConfig:
    def __init__(self):
        self.max_concurrent_tasks = 5
        self.task_timeout = 43200  # 12 hours


logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Central orchestrator that manages multiple agent instances.
    Reuses existing DistributedTaskManager and extends with multi-agent capabilities.
    """

    def __init__(self, config_override: Optional[Dict] = None):
        # Use simple configuration
        self.config = SimpleConfig()
        if config_override:
            # Override specific config values
            for key, value in config_override.items():
                setattr(self.config, key, value)

        # Reuse existing task manager
        self.task_manager = DistributedTaskManager()

        # New orchestration components
        self.agent_instances: Dict[str, Any] = {}  # Agent instances
        self.agent_connections: Dict[str, asyncio.Queue] = {}
        self.internal_clients: Dict[str, InternalAgentClient] = {}  # Internal clients for agents
        self.agent_registry = get_agent_registry()
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False

        # Agent pool management
        self.min_agents = 3
        self.max_agents = 20
        self.current_agent_count = 0
        self.next_agent_port = 50052
        self.round_robin_counter = 0  # For load balancing

    async def start(self) -> None:
        """Start the orchestrator and initial agent pool"""
        logger.info("Starting Agent Orchestrator...")
        self.running = True

        # Create initial agent pool
        for i in range(self.min_agents):
            await self.create_agent(f"initial-{i}")

        # Start orchestration workers
        asyncio.create_task(self._task_dispatcher())
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._message_handler())

        logger.info(f"Agent Orchestrator started with {self.min_agents} agents")

    async def stop(self) -> None:
        """Stop orchestrator and all agents"""
        logger.info("Stopping Agent Orchestrator...")
        self.running = False

        # Stop all internal agents
        for agent_id, client in self.internal_clients.items():
            try:
                await client.disconnect()
                logger.info(f"Stopped agent {agent_id}")
            except Exception as e:
                logger.error(f"Error stopping agent {agent_id}: {e}")

        # Clean up references
        self.agent_instances.clear()
        self.internal_clients.clear()
        self.agent_connections.clear()

    async def create_agent(self, agent_id: Optional[str] = None) -> str:
        """Create new internal agent instance"""
        if self.current_agent_count >= self.max_agents:
            raise Exception(f"Maximum agents ({self.max_agents}) reached")

        if not agent_id:
            agent_id = f"agent-{self.current_agent_count + 1:03d}"

        # Create agent configuration
        agent_config = {
            "agent_id": agent_id,
            "orchestrator_port": 50051,
            "agent_port": self.next_agent_port,
            "config": self.config.__dict__ if hasattr(self.config, '__dict__') else {},
        }
        self.next_agent_port += 1

        try:
            # Import and create agent instance
            from agents.agent import Agent

            agent_instance = Agent(agent_id, agent_config)

            # Register agent in the registry and get client
            internal_client = self.agent_registry.register_agent(agent_id, agent_instance)
            await internal_client.connect()

            # Store references
            self.agent_instances[agent_id] = agent_instance
            self.internal_clients[agent_id] = internal_client
            self.agent_connections[agent_id] = asyncio.Queue()

            self.current_agent_count += 1

            # Register agent in existing task manager
            agent = CoreAgent(
                id=agent_id,
                agent_type=AgentType.INTEGRATION,  # All agents can do all tasks
                capabilities=["all_tools"],  # All 22 tools available
                max_concurrent_tasks=5,
            )
            self.task_manager.agents[agent_id] = agent

            logger.info(f"Created internal agent {agent_id}")
            return agent_id

        except Exception as e:
            logger.error(f"Failed to create agent {agent_id}: {e}")
            raise

    async def execute_task(
        self, tool_name: str, arguments: Dict[str, Any], priority: TaskPriority = TaskPriority.MEDIUM
    ) -> Dict[str, Any]:
        """Execute task through agent system - reuses existing task system"""

        # Create task using existing system
        task = self.task_manager.create_task(
            title=f"Orchestrated {tool_name}",
            description=f"Execute {tool_name} via orchestrator",
            task_type="orchestrated",
            priority=priority,
        )

        # Store tool info in task metadata
        task.metadata = {"tool_name": tool_name, "arguments": arguments}

        # Find and assign agent - use round-robin load balancing
        available_agents = list(self.internal_clients.keys())
        if available_agents:
            # Round-robin load balancing across agents
            agent_id = available_agents[self.round_robin_counter % len(available_agents)]
            self.round_robin_counter += 1
        else:
            # Create new agent if needed and under limit
            if self.current_agent_count < self.max_agents:
                new_agent_id = await self.create_agent()
                agent_id = new_agent_id
            else:
                raise Exception("No available agents and max capacity reached")

        # Assign task
        self.task_manager.assign_task(task.id, agent_id)

        # Send task to agent via internal client
        await self._send_task_to_agent_internal(agent_id, task, tool_name, arguments)

        # Wait for completion
        while task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            await asyncio.sleep(0.1)

        if task.status == TaskStatus.COMPLETED:
            return task.metadata.get("result", {})
        else:
            raise Exception(f"Task failed: {task.metadata.get('error', 'Unknown error')}")

    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], priority: TaskPriority = TaskPriority.MEDIUM
    ) -> Dict[str, Any]:
        """Execute a single tool - alias for execute_task for API compatibility"""
        return await self.execute_task(tool_name, arguments, priority)

    async def execute_parallel_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple tasks in parallel"""
        # Create all tasks
        task_futures = []
        for task_def in tasks:
            future = asyncio.create_task(
                self.execute_task(
                    task_def["tool_name"],
                    task_def.get("arguments", {}),
                    TaskPriority(task_def.get("priority", TaskPriority.MEDIUM.value)),
                )
            )
            task_futures.append(future)

        # Wait for all to complete
        results = await asyncio.gather(*task_futures, return_exceptions=True)

        # Convert exceptions to error results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({"success": False, "error": str(result)})
            else:
                processed_results.append(result)

        return processed_results

    async def _send_task_to_agent(self, agent_id: str, task: Task, tool_name: str, arguments: Dict) -> None:
        """Send task to agent process via message queue"""
        try:
            if agent_id not in self.agent_connections:
                raise Exception(f"No connection to agent {agent_id}")

            # Create task message
            message = {
                "type": "execute_tool",
                "task_id": task.id,
                "tool_name": tool_name,
                "arguments": arguments,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Send to agent (will be implemented with proper IPC)
            await self.agent_connections[agent_id].put(message)

            # For now, simulate task execution with actual tool call
            # This will be replaced with proper agent communication
            result = await self._simulate_tool_execution(tool_name, arguments)

            # Update task with result
            task.metadata["result"] = result
            task.status = TaskStatus.COMPLETED

        except Exception as e:
            logger.error(f"Error sending task to agent {agent_id}: {e}")
            task.metadata["error"] = str(e)
            task.status = TaskStatus.FAILED

    async def _simulate_tool_execution(self, tool_name: str, arguments: Dict) -> Dict[str, Any]:
        """
        Temporary simulation - will be replaced with actual agent communication.
        For now, directly call the tools from distributed_server.py
        """
        try:
            # Import and call the actual tool functions
            from distributed_server import mcp

            # Create a mock context
            class MockContext:
                pass

            ctx = MockContext()

            # Get the tool function and call it
            if hasattr(mcp, '_tools') and tool_name in mcp._tools:
                tool_func = mcp._tools[tool_name]
                result = await tool_func(ctx, **arguments)
                return {"success": True, "result": result}
            else:
                return {"success": False, "error": f"Tool {tool_name} not found"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _task_dispatcher(self) -> None:
        """Worker that monitors and dispatches tasks"""
        while self.running:
            try:
                # Check for overloaded agents and scale up if needed
                agent_stats = self.get_agent_stats()
                avg_load = agent_stats.get("average_load", 0)

                if avg_load > 0.8 and self.current_agent_count < self.max_agents:
                    await self.create_agent()
                    logger.info(f"Auto-scaled up: created new agent (load: {avg_load:.2f})")

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error in task dispatcher: {e}")
                await asyncio.sleep(1)

    async def _health_monitor(self) -> None:
        """Monitor agent health and restart failed agents"""
        while self.running:
            try:
                for agent_id, client in list(self.internal_clients.items()):
                    # Check agent health via heartbeat
                    healthy = await client.heartbeat()
                    if not healthy:
                        logger.warning(f"Agent {agent_id} failed health check, restarting...")

                        # Remove from tracking
                        if agent_id in self.agent_instances:
                            del self.agent_instances[agent_id]
                        if agent_id in self.internal_clients:
                            del self.internal_clients[agent_id]
                        if agent_id in self.agent_connections:
                            del self.agent_connections[agent_id]
                        self.current_agent_count -= 1

                        # Restart agent
                        try:
                            await self.create_agent(agent_id)
                        except Exception as e:
                            logger.error(f"Failed to restart agent {agent_id}: {e}")

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(10)

    async def _message_handler(self) -> None:
        """Handle inter-agent messages"""
        while self.running:
            try:
                # Process inter-agent messages
                if not self.message_queue.empty():
                    message = await self.message_queue.get()
                    await self._route_message(message)

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in message handler: {e}")
                await asyncio.sleep(1)

    async def _route_message(self, message: Dict) -> None:
        """Route message between agents"""
        try:
            target_agent = message.get("to_agent")
            if target_agent and target_agent in self.agent_connections:
                await self.agent_connections[target_agent].put(message)
            elif target_agent == "broadcast":
                # Broadcast to all agents
                for agent_queue in self.agent_connections.values():
                    await agent_queue.put(message)
        except Exception as e:
            logger.error(f"Error routing message: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status - reuses existing task manager stats"""
        task_status = self.task_manager.get_system_status()
        project_status = self.task_manager.get_project_status()

        return {
            "orchestrator": {
                "running": self.running,
                "agents": {
                    "total": self.current_agent_count,
                    "min": self.min_agents,
                    "max": self.max_agents,
                    "processes": len(self.agent_instances),
                },
            },
            "tasks": task_status.get("tasks", {}),
            "agents_detail": task_status.get("agents", {}),
            "projects": project_status,
            "message_queue_size": self.message_queue.qsize(),
        }

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        total_tasks = sum(len(agent.current_tasks) for agent in self.task_manager.agents.values())
        avg_load = total_tasks / self.current_agent_count if self.current_agent_count > 0 else 0

        return {
            "total_agents": self.current_agent_count,
            "active_processes": len(self.agent_instances),
            "total_tasks_in_progress": total_tasks,
            "average_load": avg_load,
        }

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tools with their JSON schemas"""
        # Define tool schemas matching the agent's tools
        return [
            {
                "name": "execute_command",
                "description": "Execute shell command",
                "inputSchema": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "Shell command to execute"}},
                    "required": ["command"],
                },
            },
            {
                "name": "read_file",
                "description": "Read file contents",
                "inputSchema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string", "description": "Path to file to read"}},
                    "required": ["file_path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write content to file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to file to write"},
                        "content": {"type": "string", "description": "Content to write to file"},
                    },
                    "required": ["file_path", "content"],
                },
            },
            {
                "name": "list_directory",
                "description": "List directory contents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "Directory path to list", "default": "."}
                    },
                    "required": [],
                },
            },
            {
                "name": "search_files",
                "description": "Search for files",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Search pattern"},
                        "directory": {"type": "string", "description": "Directory to search in", "default": "."},
                    },
                    "required": ["pattern"],
                },
            },
            {
                "name": "get_system_info",
                "description": "Get system information",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_command_exists",
                "description": "Check if command exists",
                "inputSchema": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "Command to check"}},
                    "required": ["command"],
                },
            },
            {
                "name": "create_project",
                "description": "Create multi-component project",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Project name"},
                        "description": {"type": "string", "description": "Project description"},
                        "components": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Project components",
                        },
                    },
                    "required": ["name", "description", "components"],
                },
            },
            {
                "name": "distribute_task",
                "description": "Distribute task to agents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Task title"},
                        "description": {"type": "string", "description": "Task description"},
                        "task_type": {"type": "string", "description": "Task type"},
                    },
                    "required": ["title", "description", "task_type"],
                },
            },
            {
                "name": "get_project_status",
                "description": "Get project and agent status",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "slack_notify",
                "description": "Send Slack notification",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string", "description": "Slack channel"},
                        "message": {"type": "string", "description": "Message to send"},
                    },
                    "required": ["channel", "message"],
                },
            },
            {
                "name": "git_operation",
                "description": "Perform Git operations",
                "inputSchema": {
                    "type": "object",
                    "properties": {"operation": {"type": "string", "description": "Git operation to perform"}},
                    "required": ["operation"],
                },
            },
            {
                "name": "jenkins_trigger",
                "description": "Trigger Jenkins build",
                "inputSchema": {
                    "type": "object",
                    "properties": {"job_name": {"type": "string", "description": "Jenkins job name"}},
                    "required": ["job_name"],
                },
            },
            {
                "name": "youtrack_issue",
                "description": "Create YouTrack issue",
                "inputSchema": {
                    "type": "object",
                    "properties": {"operation": {"type": "string", "description": "YouTrack operation"}},
                    "required": ["operation"],
                },
            },
            {
                "name": "postgres_query",
                "description": "Execute PostgreSQL query",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database": {"type": "string", "description": "Database name"},
                        "query": {"type": "string", "description": "SQL query"},
                    },
                    "required": ["database", "query"],
                },
            },
            {
                "name": "aws_operation",
                "description": "Execute AWS operation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string", "description": "AWS service"},
                        "operation": {"type": "string", "description": "AWS operation"},
                    },
                    "required": ["service", "operation"],
                },
            },
            {
                "name": "bambu_print",
                "description": "Start Bambu Lab 3D print",
                "inputSchema": {
                    "type": "object",
                    "properties": {"printer_id": {"type": "string", "description": "Printer ID"}},
                    "required": ["printer_id"],
                },
            },
            {
                "name": "web_search",
                "description": "Search the web",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
            },
            {
                "name": "api_request",
                "description": "Make HTTP API request",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "API URL"},
                        "method": {"type": "string", "description": "HTTP method", "default": "GET"},
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "get_system_status",
                "description": "Get system status",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "list_agents",
                "description": "List available agents",
                "inputSchema": {
                    "type": "object",
                    "properties": {"status": {"type": "string", "description": "Filter by status", "default": "all"}},
                    "required": [],
                },
            },
            {
                "name": "create_task",
                "description": "Create new task",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Task title"},
                        "description": {"type": "string", "description": "Task description"},
                        "task_type": {"type": "string", "description": "Task type"},
                    },
                    "required": ["title", "description", "task_type"],
                },
            },
            {
                "name": "ollama",
                "description": "Execute local Ollama AI operations (generate, chat, embed, list_models, pull_model, show_model)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "description": "Ollama operation to perform",
                            "enum": ["generate", "chat", "embed", "list_models", "pull_model", "show_model"],
                        },
                        "prompt": {"type": "string", "description": "Text prompt (for generate operation)"},
                        "messages": {
                            "type": "array",
                            "description": "Chat messages array (for chat operation)",
                            "items": {
                                "type": "object",
                                "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
                                "required": ["role", "content"],
                            },
                        },
                        "input_text": {"type": "string", "description": "Text to embed (for embed operation)"},
                        "model_name": {
                            "type": "string",
                            "description": "Model name (for pull_model and show_model operations)",
                        },
                        "model": {"type": "string", "description": "Model to use for operation"},
                        "temperature": {"type": "number", "description": "Temperature for text generation"},
                        "max_tokens": {"type": "integer", "description": "Maximum tokens to generate"},
                    },
                    "required": ["operation"],
                },
            },
        ]

    async def _send_task_to_agent_internal(self, agent_id: str, task: Task, tool_name: str, arguments: Dict[str, Any]):
        """Send task to agent via internal communication"""
        try:
            # Get internal client for this agent
            internal_client = self.internal_clients.get(agent_id)
            if not internal_client:
                logger.error(f"No internal client found for agent {agent_id}")
                task.status = TaskStatus.FAILED
                task.metadata["error"] = "No connection to agent"
                return

            # Execute task via internal client
            logger.info(f"Sending {tool_name} to agent {agent_id} via internal client")
            result = await internal_client.execute_task(tool_name, arguments)

            # Update task status based on result
            if result.get("success", False):
                task.status = TaskStatus.COMPLETED
                task.metadata["result"] = result
                logger.info(f"Task {task.id} completed successfully on agent {agent_id}")
            else:
                task.status = TaskStatus.FAILED
                task.metadata["error"] = result.get("error", "Unknown error")
                logger.error(f"Task {task.id} failed on agent {agent_id}: {task.metadata['error']}")

        except Exception as e:
            logger.error(f"Error sending task to agent {agent_id}: {e}")
            task.status = TaskStatus.FAILED
            task.metadata["error"] = str(e)


# Convenience functions for external use
_orchestrator_instance: Optional[AgentOrchestrator] = None


async def get_orchestrator() -> AgentOrchestrator:
    """Get or create global orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AgentOrchestrator()
        await _orchestrator_instance.start()
    return _orchestrator_instance


async def execute_tool_via_orchestrator(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute tool through orchestrator - main entry point for external calls"""
    orchestrator = await get_orchestrator()
    return await orchestrator.execute_tool(tool_name, arguments)


async def execute_parallel_tools(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Execute multiple tools in parallel"""
    orchestrator = await get_orchestrator()
    return await orchestrator.execute_parallel_tasks(tasks)


async def get_orchestrator_status() -> Dict[str, Any]:
    """Get orchestrator status"""
    if _orchestrator_instance:
        return _orchestrator_instance.get_system_status()
    return {"orchestrator": {"running": False}}


# Main function for testing
async def main():
    """Test the orchestrator"""
    logging.basicConfig(level=logging.INFO)

    orchestrator = AgentOrchestrator()

    try:
        await orchestrator.start()

        # Test single task
        result = await orchestrator.execute_task("execute_command", {"command": "echo 'Hello from orchestrator!'"})
        print(f"Single task result: {result}")

        # Test parallel tasks
        parallel_tasks = [
            {"tool_name": "get_system_info", "arguments": {}},
            {"tool_name": "list_directory", "arguments": {"directory": "."}},
            {"tool_name": "check_command_exists", "arguments": {"command": "python"}},
        ]

        parallel_results = await orchestrator.execute_parallel_tasks(parallel_tasks)
        print(f"Parallel results: {len(parallel_results)} tasks completed")

        # Show status
        status = orchestrator.get_system_status()
        print(f"System status: {json.dumps(status, indent=2, default=str)}")

        await asyncio.sleep(2)

    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
