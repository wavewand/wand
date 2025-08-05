"""
REST API Gateway Implementation

Provides HTTP/REST interface to the gRPC-based distributed MCP system.
"""

import asyncio
import logging
import os

# Import generated protobuf classes
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

import grpc
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from generated import agent_pb2, agent_pb2_grpc

from .models import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'generated'))


class RestAPIGateway:
    """REST API Gateway for the distributed MCP system."""

    def __init__(
        self,
        coordinator_host: str = "127.0.0.1",
        coordinator_port: int = 50051,
        integration_host: str = "127.0.0.1",
        integration_port: int = 50200,
    ):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.integration_host = integration_host
        self.integration_port = integration_port

        # gRPC clients
        self.coordinator_client: Optional[agent_pb2_grpc.CoordinatorServiceStub] = None
        self.integration_client: Optional[agent_pb2_grpc.IntegrationServiceStub] = None
        self.coordinator_channel: Optional[grpc.aio.Channel] = None
        self.integration_channel: Optional[grpc.aio.Channel] = None

        self.logger = logging.getLogger("rest_api_gateway")

        # Create FastAPI app
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self._startup()
            yield
            # Shutdown
            await self._shutdown()

        app = FastAPI(
            title="MCP Distributed System API",
            description="REST API Gateway for the distributed MCP system with multi-agent task management",
            version="3.0.0",
            lifespan=lifespan,
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add routes
        self._setup_routes(app)

        return app

    async def _startup(self):
        """Initialize gRPC connections on startup."""
        self.logger.info("Starting REST API Gateway...")

        try:
            # Create gRPC channels
            self.coordinator_channel = grpc.aio.insecure_channel(f'{self.coordinator_host}:{self.coordinator_port}')
            self.integration_channel = grpc.aio.insecure_channel(f'{self.integration_host}:{self.integration_port}')

            # Create gRPC clients
            self.coordinator_client = agent_pb2_grpc.CoordinatorServiceStub(self.coordinator_channel)
            self.integration_client = agent_pb2_grpc.IntegrationServiceStub(self.integration_channel)

            # Test connections
            await self._test_connections()

            self.logger.info("REST API Gateway started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start REST API Gateway: {e}")
            raise

    async def _shutdown(self):
        """Clean up on shutdown."""
        self.logger.info("Shutting down REST API Gateway...")

        if self.coordinator_channel:
            await self.coordinator_channel.close()
        if self.integration_channel:
            await self.integration_channel.close()

    async def _test_connections(self):
        """Test gRPC connections."""
        try:
            # Test coordinator connection
            await asyncio.wait_for(
                self.coordinator_client.GetSystemStatus(agent_pb2.SystemStatusRequest()), timeout=5.0
            )
            self.logger.info("Coordinator connection OK")

            # Test integration connection
            await asyncio.wait_for(
                self.integration_client.ListIntegrations(agent_pb2.ListIntegrationsRequest()), timeout=5.0
            )
            self.logger.info("Integration service connection OK")

        except asyncio.TimeoutError:
            self.logger.warning("gRPC services not yet available - will retry")
        except Exception as e:
            self.logger.error(f"gRPC connection test failed: {e}")

    def _setup_routes(self, app: FastAPI):
        """Setup all API routes."""

        # Health check
        @app.get("/health", response_model=HealthCheckModel)
        async def health_check():
            """Health check endpoint."""
            services = {}

            # Check coordinator
            try:
                await asyncio.wait_for(
                    self.coordinator_client.GetSystemStatus(agent_pb2.SystemStatusRequest()), timeout=2.0
                )
                services["coordinator"] = "healthy"
            except BaseException:
                services["coordinator"] = "unhealthy"

            # Check integration service
            try:
                await asyncio.wait_for(
                    self.integration_client.ListIntegrations(agent_pb2.ListIntegrationsRequest()), timeout=2.0
                )
                services["integration"] = "healthy"
            except BaseException:
                services["integration"] = "unhealthy"

            overall_status = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"

            return HealthCheckModel(status=overall_status, timestamp=datetime.now(), services=services)

        # Project endpoints
        @app.post("/api/v1/projects", response_model=ProjectResponseModel)
        async def create_project(project: ProjectCreateModel):
            """Create a new project."""
            try:
                # Convert to gRPC project
                grpc_project = self._convert_to_grpc_project(project)

                # Call coordinator
                request = agent_pb2.CreateProjectRequest(project=grpc_project)
                response = await self.coordinator_client.CreateProject(request)

                return ProjectResponseModel(
                    project_id=response.project_id,
                    tasks_created=response.tasks_created,
                    tasks_distributed=response.tasks_distributed,
                    task_ids=list(response.task_ids),
                )

            except grpc.RpcError as e:
                self.logger.error(f"gRPC error: {e}")
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/api/v1/projects/{project_id}/status", response_model=ProjectStatusModel)
        async def get_project_status(project_id: str):
            """Get project status."""
            try:
                request = agent_pb2.ProjectStatusRequest(project_id=project_id)
                response = await self.coordinator_client.GetProjectStatus(request)

                return ProjectStatusModel(
                    project_id=project_id,
                    status=response.status,
                    tasks=[self._convert_grpc_task_to_model(task) for task in response.tasks],
                    completion_percentage=response.completion_percentage,
                    task_status_counts=dict(response.task_status_counts),
                )

            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.NOT_FOUND:
                    raise HTTPException(status_code=404, detail="Project not found")
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

        # Task endpoints
        @app.post("/api/v1/tasks", response_model=TaskResponseModel)
        async def create_task(task: TaskCreateModel):
            """Create a single task."""
            try:
                # Convert to gRPC task
                grpc_task = self._convert_to_grpc_task(task)

                # Distribute via coordinator
                request = agent_pb2.DistributeTaskRequest(task=grpc_task)
                response = await self.coordinator_client.DistributeTask(request)

                return TaskResponseModel(
                    task_id=response.task_id,
                    status=response.status,
                    message=response.message,
                    timestamp=self._convert_grpc_timestamp(response.timestamp),
                )

            except grpc.RpcError as e:
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

        @app.get("/api/v1/tasks", response_model=PaginatedTasksModel)
        async def get_tasks(
            status: Optional[TaskStatusModel] = Query(None, description="Filter by status"),
            page: int = Query(1, ge=1, description="Page number"),
            page_size: int = Query(50, ge=1, le=100, description="Page size"),
        ):
            """Get all tasks with pagination."""
            try:
                # Convert status filter
                status_filter = None
                if status:
                    status_filter = self._convert_task_status_to_grpc(status)

                offset = (page - 1) * page_size
                request = agent_pb2.AllTasksRequest(status_filter=status_filter, limit=page_size, offset=offset)
                response = await self.coordinator_client.GetAllTasks(request)

                tasks = [self._convert_grpc_task_to_model(task) for task in response.tasks]
                total_pages = (response.total_count + page_size - 1) // page_size

                return PaginatedTasksModel(
                    tasks=tasks,
                    total_count=response.total_count,
                    page=page,
                    page_size=page_size,
                    total_pages=total_pages,
                )

            except grpc.RpcError as e:
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

        @app.get("/api/v1/tasks/{task_id}", response_model=TaskModel)
        async def get_task(task_id: str):
            """Get a specific task."""
            # This would require querying individual agents
            # For now, return error
            raise HTTPException(status_code=501, detail="Not implemented - requires agent queries")

        @app.put("/api/v1/tasks/{task_id}", response_model=TaskResponseModel)
        async def update_task(task_id: str, update: TaskUpdateModel):
            """Update a task."""
            # This would require finding the agent and updating
            raise HTTPException(status_code=501, detail="Not implemented - requires agent queries")

        # Agent endpoints
        @app.get("/api/v1/agents", response_model=PaginatedAgentsModel)
        async def get_agents(
            agent_type: Optional[AgentTypeModel] = Query(None, description="Filter by agent type"),
            page: int = Query(1, ge=1, description="Page number"),
            page_size: int = Query(50, ge=1, le=100, description="Page size"),
        ):
            """Get all agents."""
            try:
                # Convert type filter
                type_filter = None
                if agent_type:
                    type_filter = self._convert_agent_type_to_grpc(agent_type)

                request = agent_pb2.AllAgentsRequest(type_filter=type_filter)
                response = await self.coordinator_client.GetAllAgents(request)

                agents = [self._convert_grpc_agent_to_model(agent) for agent in response.agents]

                # Apply pagination
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                paginated_agents = agents[start_idx:end_idx]

                total_pages = (len(agents) + page_size - 1) // page_size

                return PaginatedAgentsModel(
                    agents=paginated_agents,
                    total_count=len(agents),
                    page=page,
                    page_size=page_size,
                    total_pages=total_pages,
                )

            except grpc.RpcError as e:
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

        # System status
        @app.get("/api/v1/system/status", response_model=SystemStatusModel)
        async def get_system_status():
            """Get overall system status."""
            try:
                request = agent_pb2.SystemStatusRequest(include_metrics=True)
                response = await self.coordinator_client.GetSystemStatus(request)

                agents = {}
                for agent_id, agent in response.agent_statuses.items():
                    agents[agent_id] = self._convert_grpc_agent_to_model(agent)

                return SystemStatusModel(
                    total_agents=response.total_agents,
                    active_agents=response.active_agents,
                    total_tasks=response.total_tasks,
                    completed_tasks=response.completed_tasks,
                    failed_tasks=response.failed_tasks,
                    agents=agents,
                    task_counts_by_status=dict(response.task_counts_by_status),
                )

            except grpc.RpcError as e:
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

        # Integration endpoints
        @app.get("/api/v1/integrations", response_model=List[IntegrationStatusModel])
        async def list_integrations(active_only: bool = Query(False, description="Show only active integrations")):
            """List all integrations."""
            try:
                request = agent_pb2.ListIntegrationsRequest(active_only=active_only)
                response = await self.integration_client.ListIntegrations(request)

                integrations = []
                for name, status in response.statuses.items():
                    integrations.append(
                        IntegrationStatusModel(
                            integration_name=status.integration_name,
                            status=status.status,
                            message=status.message,
                            last_check=self._convert_grpc_timestamp(status.last_check),
                            metrics=dict(status.metrics),
                        )
                    )

                return integrations

            except grpc.RpcError as e:
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

        @app.get("/api/v1/integrations/{integration_name}/status", response_model=IntegrationStatusModel)
        async def get_integration_status(integration_name: str):
            """Get status of a specific integration."""
            try:
                request = agent_pb2.IntegrationStatusRequest(integration_name=integration_name)
                response = await self.integration_client.GetIntegrationStatus(request)

                return IntegrationStatusModel(
                    integration_name=response.integration_name,
                    status=response.status,
                    message=response.message,
                    last_check=self._convert_grpc_timestamp(response.last_check),
                    metrics=dict(response.metrics),
                )

            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.NOT_FOUND:
                    raise HTTPException(status_code=404, detail="Integration not found")
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

        @app.post("/api/v1/integrations/slack", response_model=IntegrationResponseModel)
        async def execute_slack_operation(operation: SlackOperationModel):
            """Execute Slack operation."""
            try:
                request = agent_pb2.SlackOperationRequest(
                    operation=operation.operation, parameters=operation.parameters
                )
                response = await self.integration_client.ExecuteSlackOperation(request)

                return IntegrationResponseModel(
                    success=response.success,
                    message=response.message,
                    result_data=dict(response.result_data),
                    timestamp=self._convert_grpc_timestamp(response.timestamp),
                )

            except grpc.RpcError as e:
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

        @app.post("/api/v1/integrations/git", response_model=IntegrationResponseModel)
        async def execute_git_operation(operation: GitOperationModel):
            """Execute Git operation."""
            try:
                request = agent_pb2.GitOperationRequest(operation=operation.operation, parameters=operation.parameters)
                response = await self.integration_client.ExecuteGitOperation(request)

                return IntegrationResponseModel(
                    success=response.success,
                    message=response.message,
                    result_data=dict(response.result_data),
                    timestamp=self._convert_grpc_timestamp(response.timestamp),
                )

            except grpc.RpcError as e:
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

        @app.post("/api/v1/integrations/aws", response_model=IntegrationResponseModel)
        async def execute_aws_operation(operation: AWSOperationModel):
            """Execute AWS operation."""
            try:
                request = agent_pb2.AWSOperationRequest(
                    service=operation.service, operation=operation.operation, parameters=operation.parameters
                )
                response = await self.integration_client.ExecuteAWSOperation(request)

                return IntegrationResponseModel(
                    success=response.success,
                    message=response.message,
                    result_data=dict(response.result_data),
                    timestamp=self._convert_grpc_timestamp(response.timestamp),
                )

            except grpc.RpcError as e:
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

        @app.post("/api/v1/integrations/jenkins", response_model=IntegrationResponseModel)
        async def execute_jenkins_operation(operation: JenkinsOperationModel):
            """Execute Jenkins operation."""
            try:
                request = agent_pb2.JenkinsOperationRequest(
                    operation=operation.operation, parameters=operation.parameters
                )
                response = await self.integration_client.ExecuteJenkinsOperation(request)

                return IntegrationResponseModel(
                    success=response.success,
                    message=response.message,
                    result_data=dict(response.result_data),
                    timestamp=self._convert_grpc_timestamp(response.timestamp),
                )

            except grpc.RpcError as e:
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

        @app.post("/api/v1/integrations/youtrack", response_model=IntegrationResponseModel)
        async def execute_youtrack_operation(operation: YouTrackOperationModel):
            """Execute YouTrack operation."""
            try:
                request = agent_pb2.YouTrackOperationRequest(
                    operation=operation.operation, parameters=operation.parameters
                )
                response = await self.integration_client.ExecuteYouTrackOperation(request)

                return IntegrationResponseModel(
                    success=response.success,
                    message=response.message,
                    result_data=dict(response.result_data),
                    timestamp=self._convert_grpc_timestamp(response.timestamp),
                )

            except grpc.RpcError as e:
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

    # Conversion methods
    def _convert_to_grpc_project(self, project: ProjectCreateModel) -> agent_pb2.Project:
        """Convert REST project model to gRPC."""
        from google.protobuf.timestamp_pb2 import Timestamp

        current_time = Timestamp()
        current_time.GetCurrentTime()

        grpc_project = agent_pb2.Project(
            id=str(uuid.uuid4()),
            name=project.name,
            description=project.description,
            requirements=project.requirements,
            tech_stack=project.tech_stack,
            priority=self._convert_task_priority_to_grpc(project.priority),
            created_at=current_time,
        )

        return grpc_project

    def _convert_to_grpc_task(self, task: TaskCreateModel) -> agent_pb2.Task:
        """Convert REST task model to gRPC."""
        from google.protobuf.timestamp_pb2 import Timestamp

        current_time = Timestamp()
        current_time.GetCurrentTime()

        grpc_task = agent_pb2.Task(
            id=str(uuid.uuid4()),
            title=task.title,
            description=task.description,
            type=task.type,
            priority=self._convert_task_priority_to_grpc(task.priority),
            status=agent_pb2.TaskStatus.PENDING,
            dependencies=task.dependencies,
            created_at=current_time,
            updated_at=current_time,
            metadata=task.metadata,
        )

        return grpc_task

    def _convert_grpc_task_to_model(self, grpc_task: agent_pb2.Task) -> TaskModel:
        """Convert gRPC task to REST model."""
        return TaskModel(
            id=grpc_task.id,
            title=grpc_task.title,
            description=grpc_task.description,
            type=grpc_task.type,
            priority=self._convert_grpc_task_priority(grpc_task.priority),
            status=self._convert_grpc_task_status(grpc_task.status),
            assigned_to=grpc_task.assigned_to if grpc_task.assigned_to else None,
            dependencies=list(grpc_task.dependencies),
            subtasks=list(grpc_task.subtasks),
            created_at=self._convert_grpc_timestamp(grpc_task.created_at),
            updated_at=self._convert_grpc_timestamp(grpc_task.updated_at),
            metadata=dict(grpc_task.metadata),
        )

    def _convert_grpc_agent_to_model(self, grpc_agent: agent_pb2.Agent) -> AgentModel:
        """Convert gRPC agent to REST model."""
        return AgentModel(
            id=grpc_agent.id,
            name=grpc_agent.name,
            type=self._convert_grpc_agent_type(grpc_agent.type),
            capabilities=dict(grpc_agent.capabilities),
            current_tasks=list(grpc_agent.current_tasks),
            status=self._convert_grpc_agent_status(grpc_agent.status),
            max_concurrent_tasks=grpc_agent.max_concurrent_tasks,
            performance_metrics=dict(grpc_agent.performance_metrics),
            port=grpc_agent.port,
        )

    def _convert_task_priority_to_grpc(self, priority: TaskPriorityModel) -> agent_pb2.TaskPriority:
        """Convert REST task priority to gRPC."""
        mapping = {
            TaskPriorityModel.CRITICAL: agent_pb2.TaskPriority.CRITICAL,
            TaskPriorityModel.HIGH: agent_pb2.TaskPriority.HIGH,
            TaskPriorityModel.MEDIUM: agent_pb2.TaskPriority.MEDIUM,
            TaskPriorityModel.LOW: agent_pb2.TaskPriority.LOW,
        }
        return mapping.get(priority, agent_pb2.TaskPriority.MEDIUM)

    def _convert_grpc_task_priority(self, priority: agent_pb2.TaskPriority) -> TaskPriorityModel:
        """Convert gRPC task priority to REST."""
        mapping = {
            agent_pb2.TaskPriority.CRITICAL: TaskPriorityModel.CRITICAL,
            agent_pb2.TaskPriority.HIGH: TaskPriorityModel.HIGH,
            agent_pb2.TaskPriority.MEDIUM: TaskPriorityModel.MEDIUM,
            agent_pb2.TaskPriority.LOW: TaskPriorityModel.LOW,
        }
        return mapping.get(priority, TaskPriorityModel.MEDIUM)

    def _convert_task_status_to_grpc(self, status: TaskStatusModel) -> agent_pb2.TaskStatus:
        """Convert REST task status to gRPC."""
        mapping = {
            TaskStatusModel.PENDING: agent_pb2.TaskStatus.PENDING,
            TaskStatusModel.ASSIGNED: agent_pb2.TaskStatus.ASSIGNED,
            TaskStatusModel.IN_PROGRESS: agent_pb2.TaskStatus.IN_PROGRESS,
            TaskStatusModel.COMPLETED: agent_pb2.TaskStatus.COMPLETED,
            TaskStatusModel.FAILED: agent_pb2.TaskStatus.FAILED,
            TaskStatusModel.BLOCKED: agent_pb2.TaskStatus.BLOCKED,
        }
        return mapping.get(status, agent_pb2.TaskStatus.PENDING)

    def _convert_grpc_task_status(self, status: agent_pb2.TaskStatus) -> TaskStatusModel:
        """Convert gRPC task status to REST."""
        mapping = {
            agent_pb2.TaskStatus.PENDING: TaskStatusModel.PENDING,
            agent_pb2.TaskStatus.ASSIGNED: TaskStatusModel.ASSIGNED,
            agent_pb2.TaskStatus.IN_PROGRESS: TaskStatusModel.IN_PROGRESS,
            agent_pb2.TaskStatus.COMPLETED: TaskStatusModel.COMPLETED,
            agent_pb2.TaskStatus.FAILED: TaskStatusModel.FAILED,
            agent_pb2.TaskStatus.BLOCKED: TaskStatusModel.BLOCKED,
        }
        return mapping.get(status, TaskStatusModel.PENDING)

    def _convert_agent_type_to_grpc(self, agent_type: AgentTypeModel) -> agent_pb2.AgentType:
        """Convert REST agent type to gRPC."""
        mapping = {
            AgentTypeModel.MANAGER: agent_pb2.AgentType.MANAGER,
            AgentTypeModel.FRONTEND: agent_pb2.AgentType.FRONTEND,
            AgentTypeModel.BACKEND: agent_pb2.AgentType.BACKEND,
            AgentTypeModel.DATABASE: agent_pb2.AgentType.DATABASE,
            AgentTypeModel.DEVOPS: agent_pb2.AgentType.DEVOPS,
            AgentTypeModel.INTEGRATION: agent_pb2.AgentType.INTEGRATION,
            AgentTypeModel.QA: agent_pb2.AgentType.QA,
        }
        return mapping.get(agent_type, agent_pb2.AgentType.BACKEND)

    def _convert_grpc_agent_type(self, agent_type: agent_pb2.AgentType) -> AgentTypeModel:
        """Convert gRPC agent type to REST."""
        mapping = {
            agent_pb2.AgentType.MANAGER: AgentTypeModel.MANAGER,
            agent_pb2.AgentType.FRONTEND: AgentTypeModel.FRONTEND,
            agent_pb2.AgentType.BACKEND: AgentTypeModel.BACKEND,
            agent_pb2.AgentType.DATABASE: AgentTypeModel.DATABASE,
            agent_pb2.AgentType.DEVOPS: AgentTypeModel.DEVOPS,
            agent_pb2.AgentType.INTEGRATION: AgentTypeModel.INTEGRATION,
            agent_pb2.AgentType.QA: AgentTypeModel.QA,
        }
        return mapping.get(agent_type, AgentTypeModel.BACKEND)

    def _convert_grpc_agent_status(self, status: agent_pb2.AgentStatus) -> AgentStatusModel:
        """Convert gRPC agent status to REST."""
        mapping = {
            agent_pb2.AgentStatus.ONLINE: AgentStatusModel.ONLINE,
            agent_pb2.AgentStatus.OFFLINE: AgentStatusModel.OFFLINE,
            agent_pb2.AgentStatus.BUSY: AgentStatusModel.BUSY,
            agent_pb2.AgentStatus.IDLE: AgentStatusModel.IDLE,
        }
        return mapping.get(status, AgentStatusModel.ONLINE)

    def _convert_grpc_timestamp(self, timestamp) -> datetime:
        """Convert gRPC timestamp to datetime."""
        if not timestamp:
            return datetime.now()
        return timestamp.ToDatetime()


def create_rest_api_gateway(coordinator_port: int = 50051, integration_port: int = 50200) -> RestAPIGateway:
    """Create REST API Gateway instance."""
    return RestAPIGateway(coordinator_port=coordinator_port, integration_port=integration_port)


async def run_rest_api_gateway(
    host: str = "0.0.0.0", port: int = 8000, coordinator_port: int = 50051, integration_port: int = 50200
):
    """Run REST API Gateway server."""
    import uvicorn

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    gateway = create_rest_api_gateway(coordinator_port, integration_port)

    print(f"Starting REST API Gateway on {host}:{port}")
    print(f"Coordinator: 127.0.0.1:{coordinator_port}")
    print(f"Integration: 127.0.0.1:{integration_port}")

    # Run with uvicorn
    config = uvicorn.Config(app=gateway.app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def run_rest_api_process(
    host: str = "0.0.0.0", port: int = 8000, coordinator_port: int = 50051, integration_port: int = 50200
):
    """Run REST API Gateway in separate process."""
    asyncio.run(run_rest_api_gateway(host, port, coordinator_port, integration_port))
