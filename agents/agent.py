#!/usr/bin/env python3
"""
Agent Instance - Individual agent with all tools and integrations

Clean architecture with tools contained within the agent.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.task_manager import Task, TaskPriority, TaskStatus

from .tools import AgentTools

# Import from new architecture
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)


class Agent:
    """
    Individual agent instance that can execute all 22 MCP tools.
    Communicates with orchestrator via message queues.
    """

    def __init__(self, agent_id: str, config_data: Dict[str, Any]):
        self.agent_id = agent_id
        self.config_data = config_data
        self.orchestrator_port = config_data.get("orchestrator_port", 50051)
        self.agent_port = config_data.get("agent_port", 50052)

        # Communication queues
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.running = False

        # Initialize tools
        self.agent_tools = AgentTools(agent_id, config_data)

        # Tool registry - all 22 tools available
        self.tools = {
            # System tools
            "execute_command": self.agent_tools.execute_command,
            "get_system_info": self.agent_tools.get_system_info,
            "check_command_exists": self.agent_tools.check_command_exists,
            "get_system_status": self.agent_tools.get_system_status,
            # File operations
            "read_file": self.agent_tools.read_file,
            "write_file": self.agent_tools.write_file,
            "list_directory": self.agent_tools.list_directory,
            "search_files": self.agent_tools.search_files,
            # Project management
            "create_project": self.agent_tools.create_project,
            # API tools
            "api_request": self.agent_tools.api_request,
            # Integration tools (placeholder implementations)
            "slack_notify": self.agent_tools.slack_notify,
            "git_operation": self.agent_tools.git_operation,
            "jenkins_trigger": self.agent_tools.jenkins_trigger,
            "youtrack_issue": self.agent_tools.youtrack_issue,
            "postgres_query": self.agent_tools.postgres_query,
            "aws_operation": self.agent_tools.aws_operation,
            "bambu_print": self.agent_tools.bambu_print,
            "web_search": self.agent_tools.web_search,
            "claude_api_call": self.agent_tools.claude_api_call,
            # Design and development tools
            "sketch_transpile": self.agent_tools.sketch_transpile,
            # Task management tools (these delegate to orchestrator)
            "distribute_task": self._distribute_task,
            "get_project_status": self._get_project_status,
            "list_agents": self._list_agents,
            "create_task": self._create_task,
        }

        logger.info(f"Agent {agent_id} initialized with {len(self.tools)} tools (including Claude API)")

    async def start(self) -> None:
        """Start the agent as internal instance"""
        logger.info(f"Starting internal agent {self.agent_id}...")
        self.running = True

        # Start worker tasks
        asyncio.create_task(self._message_processor())
        asyncio.create_task(self._heartbeat_sender())

        # Connect to orchestrator (internal)
        await self._connect_to_orchestrator()

        logger.info(f"Internal agent {self.agent_id} started and ready")

        # For internal agents, we don't need to keep running a loop
        # The agent is ready to receive calls via direct method invocation

    async def stop(self) -> None:
        """Stop the agent"""
        logger.info(f"Stopping internal agent {self.agent_id}...")
        self.running = False

    async def _connect_to_orchestrator(self) -> None:
        """Connect to orchestrator (placeholder for gRPC/IPC)"""
        # For now, just log connection
        logger.info(f"Agent {self.agent_id} connected to orchestrator on port {self.orchestrator_port}")

    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute a tool with given arguments"""
        try:
            if tool_name not in self.tools:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}",
                }

            # Execute tool directly (new architecture doesn't use Context)
            tool_func = self.tools[tool_name]
            result = await tool_func(**arguments)

            # Handle different result types
            if isinstance(result, str):
                # Most tools return JSON strings
                try:
                    parsed_result = json.loads(result)
                    return {"success": True, "result": parsed_result}
                except json.JSONDecodeError:
                    return {"success": True, "result": {"output": result}}
            elif isinstance(result, dict):
                return {"success": True, "result": result}
            else:
                return {"success": True, "result": {"output": str(result)}}

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return {"success": False, "error": str(e), "tool_name": tool_name, "arguments": arguments}

    async def _message_processor(self) -> None:
        """Process incoming messages from orchestrator"""
        while self.running:
            try:
                if not self.inbox.empty():
                    message = await self.inbox.get()
                    await self._handle_message(message)

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await asyncio.sleep(1)

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle specific message types"""
        try:
            message_type = message.get("type")

            if message_type == "execute_tool":
                # Execute tool request
                task_id = message.get("task_id")
                tool_name = message.get("tool_name")
                arguments = message.get("arguments", {})

                logger.info(f"Agent {self.agent_id} executing {tool_name} for task {task_id}")

                # Execute the tool
                result = await self.execute_tool(tool_name, arguments)

                # Send result back to orchestrator
                response = {
                    "type": "tool_result",
                    "task_id": task_id,
                    "agent_id": self.agent_id,
                    "result": result,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                await self.outbox.put(response)

            elif message_type == "ping":
                # Health check
                response = {
                    "type": "pong",
                    "agent_id": self.agent_id,
                    "status": "healthy",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.outbox.put(response)

            elif message_type == "inter_agent_message":
                # Message from another agent
                await self._handle_inter_agent_message(message)

            else:
                logger.warning(f"Unknown message type: {message_type}")

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def _handle_inter_agent_message(self, message: Dict[str, Any]) -> None:
        """Handle message from another agent"""
        from_agent = message.get("from_agent")
        payload = message.get("payload", {})

        logger.info(f"Agent {self.agent_id} received message from {from_agent}: {payload}")

        # Process inter-agent message (placeholder)
        # In the future, this could handle complex agent-to-agent workflows

    async def _heartbeat_sender(self) -> None:
        """Send periodic heartbeat to orchestrator"""
        while self.running:
            try:
                heartbeat = {
                    "type": "heartbeat",
                    "agent_id": self.agent_id,
                    "status": "running",
                    "tools_count": len(self.tools),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                await self.outbox.put(heartbeat)
                await asyncio.sleep(30)  # Send every 30 seconds

            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                await asyncio.sleep(30)

    async def send_message_to_agent(self, target_agent: str, message_type: str, payload: Dict[str, Any]) -> None:
        """Send message to another agent via orchestrator"""
        message = {
            "type": "inter_agent_message",
            "from_agent": self.agent_id,
            "to_agent": target_agent,
            "message_type": message_type,
            "payload": payload,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await self.outbox.put(message)
        logger.info(f"Agent {self.agent_id} sent message to {target_agent}")

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "running": self.running,
            "tools_available": len(self.tools),
            "inbox_size": self.inbox.qsize(),
            "outbox_size": self.outbox.qsize(),
            "tools": list(self.tools.keys()),
        }

    # Delegation methods for task management tools
    async def _distribute_task(
        self, title: str, description: str, task_type: str, priority: str = "medium"
    ) -> Dict[str, Any]:
        """Delegate task distribution to orchestrator"""
        return {"success": False, "error": "Task distribution requires orchestrator connection"}

    async def _get_project_status(self) -> Dict[str, Any]:
        """Delegate project status to orchestrator"""
        return {"success": False, "error": "Project status requires orchestrator connection"}

    async def _list_agents(self, status: Optional[str] = None) -> Dict[str, Any]:
        """Delegate agent listing to orchestrator"""
        return {"success": False, "error": "Agent listing requires orchestrator connection"}

    async def _create_task(
        self, title: str, description: str, task_type: str, priority: str = "medium"
    ) -> Dict[str, Any]:
        """Delegate task creation to orchestrator"""
        return {"success": False, "error": "Task creation requires orchestrator connection"}


# Mock message simulation for testing without full IPC
class MockMessageSystem:
    """Simulate message passing for testing"""

    def __init__(self, agent: Agent):
        self.agent = agent

    async def simulate_tool_execution(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate receiving a tool execution request"""
        message = {
            "type": "execute_tool",
            "task_id": f"test-{datetime.now().timestamp()}",
            "tool_name": tool_name,
            "arguments": arguments,
        }

        await self.agent.inbox.put(message)

        # Wait for result
        timeout = 0
        while self.agent.outbox.empty() and timeout < 100:
            await asyncio.sleep(0.1)
            timeout += 1

        if not self.agent.outbox.empty():
            result = await self.agent.outbox.get()
            return result
        else:
            return {"success": False, "error": "Timeout waiting for result"}


async def main():
    """Test agent functionality"""
    parser = argparse.ArgumentParser(description='MCP Agent Instance')
    parser.add_argument('--config', type=str, help='Path to agent configuration file')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    if args.test:
        # Test mode - run agent with mock configuration
        agent_config = {"agent_id": "test-agent-001", "orchestrator_port": 50051, "agent_port": 50052, "config": {}}

        agent = Agent("test-agent-001", agent_config)

        # Test tool execution
        mock_system = MockMessageSystem(agent)

        try:
            # Start agent in background
            agent_task = asyncio.create_task(agent.start())

            # Give agent time to start
            await asyncio.sleep(1)

            # Test some tools
            print("Testing execute_command...")
            result1 = await mock_system.simulate_tool_execution(
                "execute_command", {"command": "echo 'Hello from agent!'"}
            )
            print(f"Result: {result1}")

            print("\nTesting get_system_info...")
            result2 = await mock_system.simulate_tool_execution("get_system_info", {})
            print(f"Result: {result2}")

            print("\nTesting list_directory...")
            result3 = await mock_system.simulate_tool_execution("list_directory", {"directory": "."})
            print(f"Result: {result3}")

            # Show agent status
            status = agent.get_status()
            print(f"\nAgent status: {json.dumps(status, indent=2)}")

            await asyncio.sleep(2)

        finally:
            await agent.stop()
            agent_task.cancel()

    elif args.config:
        # Production mode - load config from file
        with open(args.config, 'r') as f:
            agent_config = json.load(f)

        agent_id = agent_config.get("agent_id", "unknown-agent")
        agent = Agent(agent_id, agent_config)

        try:
            await agent.start()
        except KeyboardInterrupt:
            await agent.stop()

    else:
        print("Usage: python agent.py [--config config_file] [--test]")
        print("  --config: Run agent with configuration file")
        print("  --test: Run agent in test mode with mock setup")


if __name__ == "__main__":
    asyncio.run(main())
