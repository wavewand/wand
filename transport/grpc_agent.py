#!/usr/bin/env python3
"""
gRPC Agent Transport - Communication between orchestrator and agents

Simplified gRPC communication focused on task execution:
- Orchestrator sends tasks to agents via gRPC
- Agents execute tasks and return results
- Heartbeat monitoring for agent health
"""

import asyncio
import logging
import os

# Import generated gRPC code
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import grpc

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from generated import agent_pb2, agent_pb2_grpc
except ImportError:
    # Fallback for missing grpc files
    agent_pb2 = None
    agent_pb2_grpc = None

logger = logging.getLogger(__name__)


class GrpcAgentClient:
    """Client for communicating with agent gRPC servers"""

    def __init__(self, agent_id: str, host: str = "localhost", port: int = 50052):
        self.agent_id = agent_id
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None

    async def connect(self) -> bool:
        """Connect to the agent gRPC server"""
        try:
            if not agent_pb2_grpc:
                logger.warning("gRPC protobuf files not available, using mock client")
                return True

            self.channel = grpc.aio.insecure_channel(f"{self.host}:{self.port}")
            self.stub = agent_pb2_grpc.AgentServiceStub(self.channel)

            # Test connection with heartbeat
            await self.heartbeat()
            logger.info(f"Connected to agent {self.agent_id} at {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to agent {self.agent_id}: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the agent"""
        if self.channel:
            await self.channel.close()
            logger.info(f"Disconnected from agent {self.agent_id}")

    async def execute_task(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Send task execution request to agent"""
        try:
            if not self.stub:
                # Mock execution for testing
                return await self._mock_execute_task(tool_name, arguments)

            # Create task proto message
            task = agent_pb2.Task(
                id=f"task-{datetime.now().timestamp()}",
                title=f"Execute {tool_name}",
                description=f"Execute tool {tool_name} with arguments",
                type="tool_execution",
                priority=agent_pb2.TaskPriority.MEDIUM,
                status=agent_pb2.TaskStatus.PENDING,
                metadata={"tool_name": tool_name, "arguments": str(arguments)},
            )

            # Submit task request
            request = agent_pb2.SubmitTaskRequest(task=task, target_agent_id=self.agent_id)

            response = await self.stub.SubmitTask(request)

            return {
                "success": True,
                "task_id": response.task_id,
                "status": response.status,
                "message": response.message,
            }

        except Exception as e:
            logger.error(f"Task execution failed for agent {self.agent_id}: {e}")
            return {"success": False, "error": str(e)}

    async def _mock_execute_task(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Mock task execution for testing without full gRPC"""
        logger.info(f"Mock executing {tool_name} on agent {self.agent_id}")

        # Simulate some common tool responses
        if tool_name == "get_system_info":
            return {
                "success": True,
                "result": {
                    "agent_id": self.agent_id,
                    "system": {"platform": "Mock", "uptime": "1h"},
                    "timestamp": datetime.now().isoformat(),
                },
            }
        elif tool_name == "execute_command":
            command = arguments.get("command", "")
            return {
                "success": True,
                "result": {"stdout": f"Mock execution of: {command}", "stderr": "", "exit_code": 0},
            }
        else:
            return {
                "success": True,
                "result": {
                    "tool": tool_name,
                    "arguments": arguments,
                    "agent_id": self.agent_id,
                    "message": f"Mock execution of {tool_name}",
                },
            }

    async def heartbeat(self) -> bool:
        """Send heartbeat to check agent health"""
        try:
            if not self.stub:
                # Mock heartbeat
                return True

            request = agent_pb2.HeartbeatRequest(agent_id=self.agent_id)
            response = await self.stub.Heartbeat(request)

            return response.status == "healthy"

        except Exception as e:
            logger.error(f"Heartbeat failed for agent {self.agent_id}: {e}")
            return False

    async def get_agent_status(self) -> Dict[str, Any]:
        """Get detailed agent status"""
        try:
            if not self.stub:
                # Mock status
                return {"agent_id": self.agent_id, "status": "online", "current_tasks": 0, "max_concurrent_tasks": 5}

            request = agent_pb2.AgentStatusRequest(agent_id=self.agent_id)
            response = await self.stub.GetAgentStatus(request)

            if response.found:
                agent = response.agent
                return {
                    "agent_id": agent.id,
                    "name": agent.name,
                    "type": agent.type,
                    "status": agent.status,
                    "current_tasks": len(agent.current_tasks),
                    "max_concurrent_tasks": agent.max_concurrent_tasks,
                }
            else:
                return {"agent_id": self.agent_id, "found": False}

        except Exception as e:
            logger.error(f"Status check failed for agent {self.agent_id}: {e}")
            return {"agent_id": self.agent_id, "error": str(e)}


class GrpcAgentServer:
    """gRPC server implementation for agent instances"""

    def __init__(self, agent_instance, port: int = 50052):
        self.agent_instance = agent_instance
        self.port = port
        self.server = None

    async def start(self):
        """Start the gRPC server"""
        try:
            if not agent_pb2_grpc:
                logger.info("gRPC protobuf files not available, using mock server mode")
                return

            self.server = grpc.aio.server()
            # Only add servicer if protobuf files are available
            if hasattr(agent_pb2_grpc, 'add_AgentServiceServicer_to_server'):
                agent_pb2_grpc.add_AgentServiceServicer_to_server(GrpcAgentServicer(self.agent_instance), self.server)

            listen_addr = f'[::]:{self.port}'
            self.server.add_insecure_port(listen_addr)

            await self.server.start()
            logger.info(f"Agent gRPC server started on port {self.port}")

        except Exception as e:
            logger.error(f"Failed to start gRPC server: {e}")

    async def stop(self):
        """Stop the gRPC server"""
        if self.server:
            await self.server.stop(grace=5)
            logger.info(f"Agent gRPC server stopped")


class GrpcAgentServicer:
    """gRPC service implementation for agents"""

    def __init__(self, agent_instance):
        self.agent_instance = agent_instance

    async def SubmitTask(self, request, context):
        """Handle task submission from orchestrator"""
        try:
            task = request.task
            tool_name = task.metadata.get("tool_name", "unknown")
            arguments_str = task.metadata.get("arguments", "{}")

            # Parse arguments
            import json

            try:
                arguments = json.loads(arguments_str.replace("'", '"'))
            except BaseException:
                arguments = {}

            # Execute task using agent's tools
            result = await self.agent_instance.execute_tool(tool_name, arguments)

            return agent_pb2.TaskResponse(
                task_id=task.id,
                status="completed" if result.get("success") else "failed",
                message=str(result),
                timestamp=agent_pb2.google.protobuf.timestamp_pb2.Timestamp(),
            )

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return agent_pb2.TaskResponse(task_id=request.task.id, status="failed", message=str(e))

    async def GetAgentStatus(self, request, context):
        """Return agent status"""
        try:
            status = self.agent_instance.get_status()

            # Create agent proto message
            agent = agent_pb2.Agent(
                id=status["agent_id"],
                name=f"Agent-{status['agent_id']}",
                type=agent_pb2.AgentType.INTEGRATION,
                status=agent_pb2.AgentStatus.ONLINE if status["running"] else agent_pb2.AgentStatus.OFFLINE,
                current_tasks=[],  # Convert from status
                max_concurrent_tasks=5,
            )

            return agent_pb2.AgentStatusResponse(agent=agent, found=True)

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return agent_pb2.AgentStatusResponse(found=False)

    async def Heartbeat(self, request, context):
        """Handle heartbeat requests"""
        try:
            return agent_pb2.HeartbeatResponse(
                agent_id=request.agent_id,
                status="healthy",
                timestamp=agent_pb2.google.protobuf.timestamp_pb2.Timestamp(),
            )
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            return agent_pb2.HeartbeatResponse(agent_id=request.agent_id, status="error")
