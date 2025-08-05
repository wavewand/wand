"""
gRPC Agent Service Implementation
"""

import asyncio
import logging
import os

# Import generated protobuf classes
import sys
import uuid
from concurrent import futures
from datetime import datetime
from typing import Dict, List, Optional

import grpc

from distributed.types import AgentStatus, AgentType, TaskPriority, TaskStatus, get_default_capabilities
from generated import agent_pb2, agent_pb2_grpc

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'generated'))


class AgentGRPCServer(agent_pb2_grpc.AgentServiceServicer):
    """gRPC server implementation for individual agents."""

    def __init__(self, agent_type: AgentType, agent_id: str, port: int):
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.port = port
        self.active_tasks: Dict[str, agent_pb2.Task] = {}
        self.message_queue = asyncio.Queue()
        self.capabilities = get_default_capabilities(agent_type)
        self.status = AgentStatus.ONLINE
        self.logger = logging.getLogger(f"agent.{agent_type.value}")
        self.max_concurrent_tasks = 5

        # Performance metrics
        self.performance_metrics = {
            "tasks_completed": "0",
            "tasks_failed": "0",
            "average_task_time": "0",
            "uptime": "0",
        }

        self.logger.info(f"Agent {self.agent_id} ({agent_type.value}) initialized")

    async def SubmitTask(self, request: agent_pb2.SubmitTaskRequest, context) -> agent_pb2.TaskResponse:
        """Handle task submission."""
        task = request.task

        self.logger.info(f"Received task submission: {task.id} - {task.title}")

        # Check if agent can handle this task
        if not self._can_handle_task(task):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Agent {self.agent_type.value} cannot handle task type {task.type}")
            return agent_pb2.TaskResponse(
                task_id=task.id, status="rejected", message=f"Cannot handle task type {task.type}"
            )

        # Check capacity
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("Agent at maximum capacity")
            return agent_pb2.TaskResponse(task_id=task.id, status="rejected", message="Agent at maximum capacity")

        # Accept task
        task.status = agent_pb2.TaskStatus.ASSIGNED
        task.assigned_to = self.agent_id
        self.active_tasks[task.id] = task
        self.status = AgentStatus.BUSY

        # Start processing asynchronously
        asyncio.create_task(self._process_task(task))

        self.logger.info(f"Accepted task {task.id}: {task.title}")

        return agent_pb2.TaskResponse(
            task_id=task.id,
            status="accepted",
            message=f"Task accepted by {self.agent_type.value} agent",
            timestamp=self._current_timestamp(),
        )

    async def GetTaskStatus(self, request: agent_pb2.TaskStatusRequest, context) -> agent_pb2.TaskStatusResponse:
        """Get status of a specific task."""
        task_id = request.task_id

        if task_id in self.active_tasks:
            return agent_pb2.TaskStatusResponse(task=self.active_tasks[task_id], found=True)

        return agent_pb2.TaskStatusResponse(found=False)

    async def UpdateTaskStatus(self, request: agent_pb2.UpdateTaskStatusRequest, context) -> agent_pb2.TaskResponse:
        """Update task status."""
        task_id = request.task_id
        new_status = request.status

        if task_id not in self.active_tasks:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Task {task_id} not found")
            return agent_pb2.TaskResponse()

        task = self.active_tasks[task_id]
        task.status = new_status
        task.updated_at.CopyFrom(self._current_timestamp())

        # Update metadata if provided
        if request.metadata:
            task.metadata.update(request.metadata)

        self.logger.info(f"Updated task {task_id} status to {new_status}")

        return agent_pb2.TaskResponse(
            task_id=task_id,
            status="updated",
            message=f"Task status updated to {new_status}",
            timestamp=self._current_timestamp(),
        )

    async def GetAgentTasks(self, request: agent_pb2.AgentTasksRequest, context) -> agent_pb2.AgentTasksResponse:
        """Get all tasks assigned to this agent."""
        tasks = list(self.active_tasks.values())
        return agent_pb2.AgentTasksResponse(tasks=tasks)

    async def SendMessage(self, request: agent_pb2.SendMessageRequest, context) -> agent_pb2.MessageResponse:
        """Handle incoming messages from other agents."""
        message = request.message
        await self.message_queue.put(message)

        self.logger.info(f"Received message from {message.from_}: {message.subject}")

        return agent_pb2.MessageResponse(message_id=message.id, status="received", timestamp=self._current_timestamp())

    async def GetMessages(self, request: agent_pb2.GetMessagesRequest, context):
        """Stream messages to requesting agent."""
        limit = request.limit if request.limit > 0 else 100
        count = 0

        while count < limit:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                yield message
                count += 1
            except asyncio.TimeoutError:
                # Send keepalive or check if client disconnected
                if context.cancelled():
                    break

    async def RequestCollaboration(
        self, request: agent_pb2.CollaborationRequest, context
    ) -> agent_pb2.CollaborationResponse:
        """Handle collaboration requests."""
        task_id = request.task_id
        required_caps = request.required_capabilities

        self.logger.info(f"Collaboration request from {request.from_agent} for task {task_id}")

        # Check if this agent can help with required capabilities
        can_collaborate = any(self.capabilities.get(cap, False) for cap in required_caps)

        if can_collaborate and len(self.active_tasks) < self.max_concurrent_tasks:
            # Accept collaboration
            collaboration_context = await self._prepare_collaboration_context(request)

            return agent_pb2.CollaborationResponse(
                accepted=True,
                message=f"{self.agent_type.value} agent available for collaboration",
                shared_context=collaboration_context,
            )

        return agent_pb2.CollaborationResponse(accepted=False, message="Agent unavailable for collaboration")

    async def GetAgentStatus(self, request: agent_pb2.AgentStatusRequest, context) -> agent_pb2.AgentStatusResponse:
        """Get current agent status."""
        # Convert capabilities to protobuf format
        pb_capabilities = {}
        for cap, enabled in self.capabilities.items():
            pb_capabilities[cap] = enabled

        agent = agent_pb2.Agent(
            id=self.agent_id,
            name=f"{self.agent_type.value}_agent",
            type=self._convert_agent_type(self.agent_type),
            capabilities=pb_capabilities,
            current_tasks=list(self.active_tasks.keys()),
            status=self._convert_agent_status(self.status),
            max_concurrent_tasks=self.max_concurrent_tasks,
            performance_metrics=self.performance_metrics,
            port=self.port,
        )

        return agent_pb2.AgentStatusResponse(agent=agent, found=True)

    async def Heartbeat(self, request: agent_pb2.HeartbeatRequest, context) -> agent_pb2.HeartbeatResponse:
        """Health check heartbeat."""
        return agent_pb2.HeartbeatResponse(
            agent_id=self.agent_id, status="healthy", timestamp=self._current_timestamp()
        )

    async def _process_task(self, task: agent_pb2.Task):
        """Process task logic."""
        try:
            self.logger.info(f"Starting task processing: {task.id}")

            # Update task status
            task.status = agent_pb2.TaskStatus.IN_PROGRESS
            task.updated_at.CopyFrom(self._current_timestamp())

            # Execute task based on agent type and task type
            result = await self._execute_task_logic(task)

            # Update completion status
            task.status = agent_pb2.TaskStatus.COMPLETED
            task.metadata["result"] = str(result)
            task.metadata["completed_by"] = self.agent_id
            task.updated_at.CopyFrom(self._current_timestamp())

            # Update performance metrics
            self.performance_metrics["tasks_completed"] = str(int(self.performance_metrics["tasks_completed"]) + 1)

            self.logger.info(f"Completed task {task.id}")

        except Exception as e:
            task.status = agent_pb2.TaskStatus.FAILED
            task.metadata["error"] = str(e)
            task.updated_at.CopyFrom(self._current_timestamp())

            # Update performance metrics
            self.performance_metrics["tasks_failed"] = str(int(self.performance_metrics["tasks_failed"]) + 1)

            self.logger.error(f"Failed task {task.id}: {e}")

        finally:
            # Remove from active tasks
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]

            # Update agent status
            if not self.active_tasks:
                self.status = AgentStatus.IDLE

            # Notify coordinator of completion
            await self._notify_task_completion(task)

    async def _execute_task_logic(self, task: agent_pb2.Task) -> str:
        """Execute the actual task logic based on agent type."""
        task_type = task.type.lower()

        if self.agent_type == AgentType.FRONTEND:
            return await self._handle_frontend_task(task)
        elif self.agent_type == AgentType.BACKEND:
            return await self._handle_backend_task(task)
        elif self.agent_type == AgentType.DATABASE:
            return await self._handle_database_task(task)
        elif self.agent_type == AgentType.DEVOPS:
            return await self._handle_devops_task(task)
        elif self.agent_type == AgentType.INTEGRATION:
            return await self._handle_integration_task(task)
        elif self.agent_type == AgentType.MANAGER:
            return await self._handle_manager_task(task)
        elif self.agent_type == AgentType.QA:
            return await self._handle_qa_task(task)
        else:
            return f"Task {task.id} processed by {self.agent_type.value} agent"

    async def _handle_frontend_task(self, task: agent_pb2.Task) -> str:
        """Handle frontend-specific tasks."""
        # Simulate frontend work
        await asyncio.sleep(2)  # Simulate processing time

        if "react" in task.type:
            return "React component created and tested"
        elif "vue" in task.type:
            return "Vue component implemented"
        elif "css" in task.type:
            return "Styles implemented and responsive design verified"
        else:
            return "Frontend task completed"

    async def _handle_backend_task(self, task: agent_pb2.Task) -> str:
        """Handle backend-specific tasks."""
        await asyncio.sleep(3)  # Simulate processing time

        if "api" in task.type:
            return "REST API endpoints implemented and documented"
        elif "database" in task.type:
            return "Database integration completed"
        elif "microservice" in task.type:
            return "Microservice deployed and configured"
        else:
            return "Backend task completed"

    async def _handle_database_task(self, task: agent_pb2.Task) -> str:
        """Handle database-specific tasks."""
        await asyncio.sleep(2.5)

        if "migration" in task.type:
            return "Database migration executed successfully"
        elif "optimization" in task.type:
            return "Database queries optimized"
        elif "backup" in task.type:
            return "Database backup completed"
        else:
            return "Database task completed"

    async def _handle_devops_task(self, task: agent_pb2.Task) -> str:
        """Handle DevOps-specific tasks."""
        await asyncio.sleep(4)

        if "deployment" in task.type:
            return "Application deployed to production"
        elif "monitoring" in task.type:
            return "Monitoring and alerting configured"
        elif "infrastructure" in task.type:
            return "Infrastructure provisioned"
        else:
            return "DevOps task completed"

    async def _handle_integration_task(self, task: agent_pb2.Task) -> str:
        """Handle integration-specific tasks."""
        await asyncio.sleep(1.5)

        if "slack" in task.type:
            return "Slack integration configured"
        elif "git" in task.type:
            return "Git workflow implemented"
        elif "api" in task.type:
            return "External API integration completed"
        else:
            return "Integration task completed"

    async def _handle_manager_task(self, task: agent_pb2.Task) -> str:
        """Handle manager-specific tasks."""
        await asyncio.sleep(1)

        if "planning" in task.type:
            return "Project plan created and reviewed"
        elif "coordination" in task.type:
            return "Team coordination completed"
        elif "reporting" in task.type:
            return "Status report generated"
        else:
            return "Management task completed"

    async def _handle_qa_task(self, task: agent_pb2.Task) -> str:
        """Handle QA-specific tasks."""
        await asyncio.sleep(3)

        if "testing" in task.type:
            return "Test suite executed - all tests passed"
        elif "automation" in task.type:
            return "Test automation framework implemented"
        elif "review" in task.type:
            return "Code review completed"
        else:
            return "QA task completed"

    def _can_handle_task(self, task: agent_pb2.Task) -> bool:
        """Check if this agent can handle the given task."""
        task_type = task.type.lower()

        # Map task types to required capabilities
        capability_map = {
            'frontend': ['react', 'vue', 'angular', 'css'],
            'backend': ['python', 'go', 'nodejs', 'api_design'],
            'database': ['postgresql', 'mysql', 'database_design'],
            'devops': ['docker', 'kubernetes', 'aws'],
            'integration': ['slack', 'git', 'api_integration'],
            'manager': ['planning', 'coordination'],
            'qa': ['testing', 'automation'],
        }

        required_caps = capability_map.get(task_type, [])
        if not required_caps:
            # If no specific requirements, any agent can handle it
            return True

        # Check if agent has any of the required capabilities
        return any(self.capabilities.get(cap, False) for cap in required_caps)

    async def _prepare_collaboration_context(self, request: agent_pb2.CollaborationRequest) -> Dict[str, str]:
        """Prepare context for collaboration."""
        context = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "available_capabilities": ",".join([cap for cap, enabled in self.capabilities.items() if enabled]),
            "current_load": str(len(self.active_tasks)),
        }
        return context

    async def _notify_task_completion(self, task: agent_pb2.Task):
        """Notify coordinator about task completion."""
        # This would normally send a gRPC call to the coordinator
        # For now, just log it
        self.logger.info(f"Task {task.id} completed with status {task.status}")

    def _convert_agent_type(self, agent_type: AgentType) -> agent_pb2.AgentType:
        """Convert internal AgentType to protobuf AgentType."""
        mapping = {
            AgentType.MANAGER: agent_pb2.AgentType.MANAGER,
            AgentType.FRONTEND: agent_pb2.AgentType.FRONTEND,
            AgentType.BACKEND: agent_pb2.AgentType.BACKEND,
            AgentType.DATABASE: agent_pb2.AgentType.DATABASE,
            AgentType.DEVOPS: agent_pb2.AgentType.DEVOPS,
            AgentType.INTEGRATION: agent_pb2.AgentType.INTEGRATION,
            AgentType.QA: agent_pb2.AgentType.QA,
        }
        return mapping.get(agent_type, agent_pb2.AgentType.BACKEND)

    def _convert_agent_status(self, status: AgentStatus) -> agent_pb2.AgentStatus:
        """Convert internal AgentStatus to protobuf AgentStatus."""
        mapping = {
            AgentStatus.ONLINE: agent_pb2.AgentStatus.ONLINE,
            AgentStatus.OFFLINE: agent_pb2.AgentStatus.OFFLINE,
            AgentStatus.BUSY: agent_pb2.AgentStatus.BUSY,
            AgentStatus.IDLE: agent_pb2.AgentStatus.IDLE,
        }
        return mapping.get(status, agent_pb2.AgentStatus.ONLINE)

    def _current_timestamp(self):
        """Get current timestamp in protobuf format."""
        from google.protobuf.timestamp_pb2 import Timestamp

        timestamp = Timestamp()
        timestamp.GetCurrentTime()
        return timestamp


async def register_with_coordinator(agent_service: AgentGRPCServer, port: int):
    """Register agent with coordinator."""
    try:
        # Wait a moment for coordinator to be ready
        await asyncio.sleep(2)

        # Create connection to coordinator
        channel = grpc.aio.insecure_channel('127.0.0.1:50051')
        coordinator_stub = agent_pb2_grpc.CoordinatorServiceStub(channel)

        # Convert enum values to protobuf enums
        agent_type_pb = getattr(agent_pb2.AgentType, agent_service.agent_type.name)
        agent_status_pb = getattr(agent_pb2.AgentStatus, agent_service.status.name)

        # Create agent info
        agent_proto = agent_pb2.Agent(
            id=agent_service.agent_id,
            name=f"{agent_service.agent_type.value.title()} Agent",
            type=agent_type_pb,
            capabilities={cap: True for cap in agent_service.capabilities},
            current_tasks=[],
            status=agent_status_pb,
            max_concurrent_tasks=agent_service.max_concurrent_tasks,
            performance_metrics=agent_service.performance_metrics,
            port=port,
        )

        # Register with coordinator
        request = agent_pb2.RegisterAgentRequest(agent=agent_proto, port=port)
        response = await coordinator_stub.RegisterAgent(request)

        if response.success:
            agent_service.logger.info(f"Successfully registered with coordinator: {response.message}")
        else:
            agent_service.logger.error(f"Failed to register with coordinator: {response.message}")

        await channel.close()

    except Exception as e:
        agent_service.logger.error(f"Error registering with coordinator: {e}")


async def start_agent_grpc_server(agent_type: AgentType, port: int):
    """Start gRPC server for an agent."""
    agent_id = f"{agent_type.value}_{port}"

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create server
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))

    # Create and add agent service
    agent_service = AgentGRPCServer(agent_type, agent_id, port)
    agent_pb2_grpc.add_AgentServiceServicer_to_server(agent_service, server)

    # Configure server address
    listen_addr = f'127.0.0.1:{port}'
    server.add_insecure_port(listen_addr)

    print(f"Starting {agent_type.value} agent gRPC server on {listen_addr}")

    try:
        await server.start()

        # Register with coordinator
        asyncio.create_task(register_with_coordinator(agent_service, port))

        await server.wait_for_termination()
    except KeyboardInterrupt:
        print(f"\nShutting down {agent_type.value} agent server...")
        await server.stop(5)


def run_agent_process(agent_type: AgentType, port: int):
    """Run agent in separate process."""
    asyncio.run(start_agent_grpc_server(agent_type, port))
