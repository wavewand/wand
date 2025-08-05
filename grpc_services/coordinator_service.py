"""
gRPC Coordinator Service Implementation
"""

import asyncio
import logging
import os

# Import generated protobuf classes
import sys
import uuid
from concurrent import futures
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import grpc

from distributed.types import AgentType, TaskPriority, TaskStatus
from generated import agent_pb2, agent_pb2_grpc

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'generated'))


class CoordinatorGRPCServer(agent_pb2_grpc.CoordinatorServiceServicer):
    """gRPC server implementation for the coordinator."""

    def __init__(self):
        self.agents: Dict[str, agent_pb2.Agent] = {}
        self.tasks: Dict[str, agent_pb2.Task] = {}
        self.projects: Dict[str, agent_pb2.Project] = {}
        self.agent_clients: Dict[str, agent_pb2_grpc.AgentServiceStub] = {}
        self.agent_channels: Dict[str, grpc.aio.Channel] = {}
        self.logger = logging.getLogger("coordinator")

        # Task dependency tracking
        self.task_dependencies: Dict[str, List[str]] = {}
        self.dependent_tasks: Dict[str, List[str]] = {}

        self.logger.info("Coordinator service initialized")

    async def RegisterAgent(self, request: agent_pb2.RegisterAgentRequest, context) -> agent_pb2.RegisterAgentResponse:
        """Register a new agent with the coordinator."""
        agent = request.agent
        port = request.port

        self.logger.info(f"Registering agent {agent.id} ({agent.type}) on port {port}")

        try:
            # Store agent info
            self.agents[agent.id] = agent

            # Create gRPC client for this agent
            channel = grpc.aio.insecure_channel(f'127.0.0.1:{port}')
            self.agent_clients[agent.id] = agent_pb2_grpc.AgentServiceStub(channel)
            self.agent_channels[agent.id] = channel

            # Test connection
            try:
                await asyncio.wait_for(
                    self.agent_clients[agent.id].Heartbeat(agent_pb2.HeartbeatRequest(agent_id=agent.id)), timeout=5.0
                )
                self.logger.info(f"Successfully connected to agent {agent.id}")

            except asyncio.TimeoutError:
                self.logger.warning(f"Could not connect to agent {agent.id} - will retry later")

            return agent_pb2.RegisterAgentResponse(
                success=True, message=f"Agent {agent.id} registered successfully", agent_id=agent.id
            )

        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.id}: {e}")
            return agent_pb2.RegisterAgentResponse(
                success=False, message=f"Registration failed: {str(e)}", agent_id=agent.id
            )

    async def UnregisterAgent(
        self, request: agent_pb2.UnregisterAgentRequest, context
    ) -> agent_pb2.UnregisterAgentResponse:
        """Unregister an agent."""
        agent_id = request.agent_id

        if agent_id in self.agents:
            # Close gRPC channel
            if agent_id in self.agent_channels:
                await self.agent_channels[agent_id].close()
                del self.agent_channels[agent_id]

            # Remove from tracking
            del self.agents[agent_id]
            if agent_id in self.agent_clients:
                del self.agent_clients[agent_id]

            self.logger.info(f"Unregistered agent {agent_id}")

            return agent_pb2.UnregisterAgentResponse(
                success=True, message=f"Agent {agent_id} unregistered successfully"
            )

        return agent_pb2.UnregisterAgentResponse(success=False, message=f"Agent {agent_id} not found")

    async def DistributeTask(self, request: agent_pb2.DistributeTaskRequest, context) -> agent_pb2.TaskResponse:
        """Distribute task to the best available agent."""
        task = request.task
        preferred_agent_type = request.preferred_agent_type or ""

        self.logger.info(f"Distributing task {task.id}: {task.title}")

        # Check dependencies
        if task.dependencies:
            ready, pending = await self._check_dependencies(task.id, task.dependencies)
            if not ready:
                self.logger.info(f"Task {task.id} blocked by dependencies: {pending}")
                task.status = agent_pb2.TaskStatus.BLOCKED
                self.tasks[task.id] = task
                return agent_pb2.TaskResponse(
                    task_id=task.id, status="blocked", message=f"Task blocked by dependencies: {', '.join(pending)}"
                )

        # Find best agent for this task
        best_agent_id = await self._find_best_agent(task, preferred_agent_type)

        if not best_agent_id:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("No suitable agent available")
            return agent_pb2.TaskResponse(task_id=task.id, status="rejected", message="No suitable agent available")

        # Submit task to selected agent
        agent_client = self.agent_clients[best_agent_id]

        try:
            response = await agent_client.SubmitTask(
                agent_pb2.SubmitTaskRequest(task=task, target_agent_id=best_agent_id)
            )

            # Store task info
            task.assigned_to = best_agent_id
            task.status = agent_pb2.TaskStatus.ASSIGNED
            self.tasks[task.id] = task

            # Update dependency tracking
            self._update_dependency_tracking(task)

            self.logger.info(f"Task {task.id} assigned to agent {best_agent_id}")

            return response

        except grpc.RpcError as e:
            self.logger.error(f"Failed to submit task to agent {best_agent_id}: {e}")
            context.set_code(e.code())
            context.set_details(f"Failed to submit task to agent: {e.details()}")
            return agent_pb2.TaskResponse(
                task_id=task.id, status="failed", message=f"Failed to submit task: {e.details()}"
            )

    async def GetSystemStatus(self, request: agent_pb2.SystemStatusRequest, context) -> agent_pb2.SystemStatusResponse:
        """Get overall system status."""
        include_metrics = request.include_metrics

        agent_statuses = {}
        active_count = 0

        # Query all agents for their status
        for agent_id, agent_client in self.agent_clients.items():
            try:
                status_response = await asyncio.wait_for(
                    agent_client.GetAgentStatus(agent_pb2.AgentStatusRequest(agent_id=agent_id)), timeout=5.0
                )

                if status_response.found:
                    agent_statuses[agent_id] = status_response.agent
                    active_count += 1

            except (grpc.RpcError, asyncio.TimeoutError) as e:
                self.logger.warning(f"Agent {agent_id} unavailable: {e}")
                # Create offline agent status
                offline_agent = agent_pb2.Agent(
                    id=agent_id, name=f"offline_agent_{agent_id}", status=agent_pb2.AgentStatus.OFFLINE
                )
                agent_statuses[agent_id] = offline_agent

        # Calculate task status counts
        task_counts = {"PENDING": 0, "ASSIGNED": 0, "IN_PROGRESS": 0, "COMPLETED": 0, "FAILED": 0, "BLOCKED": 0}

        completed_tasks = 0
        failed_tasks = 0

        for task in self.tasks.values():
            status_name = agent_pb2.TaskStatus.Name(task.status)
            task_counts[status_name] = task_counts.get(status_name, 0) + 1

            if task.status == agent_pb2.TaskStatus.COMPLETED:
                completed_tasks += 1
            elif task.status == agent_pb2.TaskStatus.FAILED:
                failed_tasks += 1

        return agent_pb2.SystemStatusResponse(
            total_agents=len(self.agents),
            active_agents=active_count,
            total_tasks=len(self.tasks),
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            agent_statuses=agent_statuses,
            task_counts_by_status=task_counts,
        )

    async def GetAllTasks(self, request: agent_pb2.AllTasksRequest, context) -> agent_pb2.AllTasksResponse:
        """Get all tasks with optional filtering."""
        status_filter = request.status_filter
        limit = request.limit if request.limit > 0 else 100
        offset = request.offset

        # Filter tasks
        filtered_tasks = []
        for task in self.tasks.values():
            if status_filter and task.status != status_filter:
                continue
            filtered_tasks.append(task)

        # Sort by creation time (newest first)
        filtered_tasks.sort(key=lambda t: t.created_at.seconds, reverse=True)

        # Apply pagination
        paginated_tasks = filtered_tasks[offset : offset + limit]

        return agent_pb2.AllTasksResponse(tasks=paginated_tasks, total_count=len(filtered_tasks))

    async def GetAllAgents(self, request: agent_pb2.AllAgentsRequest, context) -> agent_pb2.AllAgentsResponse:
        """Get all registered agents."""
        type_filter = request.type_filter

        agents = []
        for agent in self.agents.values():
            if type_filter and agent.type != type_filter:
                continue
            agents.append(agent)

        return agent_pb2.AllAgentsResponse(agents=agents)

    async def CreateProject(self, request: agent_pb2.CreateProjectRequest, context) -> agent_pb2.CreateProjectResponse:
        """Create a new project and break it down into tasks."""
        project = request.project

        self.logger.info(f"Creating project {project.id}: {project.name}")

        try:
            # Break down project into tasks
            tasks = await self._break_down_project(project)

            # Store project
            project.task_ids.extend([task.id for task in tasks])
            self.projects[project.id] = project

            # Distribute tasks to agents
            task_responses = []
            distributed_count = 0

            for task in tasks:
                distribute_request = agent_pb2.DistributeTaskRequest(task=task)
                try:
                    response = await self.DistributeTask(distribute_request, context)
                    task_responses.append(response)
                    if response.status == "accepted":
                        distributed_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to distribute task {task.id}: {e}")

            self.logger.info(f"Project {project.id} created with {len(tasks)} tasks, {distributed_count} distributed")

            return agent_pb2.CreateProjectResponse(
                project_id=project.id,
                tasks_created=len(tasks),
                tasks_distributed=distributed_count,
                task_ids=[task.id for task in tasks],
            )

        except Exception as e:
            self.logger.error(f"Failed to create project {project.id}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to create project: {str(e)}")
            return agent_pb2.CreateProjectResponse()

    async def GetProjectStatus(
        self, request: agent_pb2.ProjectStatusRequest, context
    ) -> agent_pb2.ProjectStatusResponse:
        """Get project status and progress."""
        project_id = request.project_id

        if project_id not in self.projects:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Project {project_id} not found")
            return agent_pb2.ProjectStatusResponse()

        project = self.projects[project_id]
        project_tasks = [self.tasks[task_id] for task_id in project.task_ids if task_id in self.tasks]

        # Calculate status counts and completion percentage
        status_counts = {}
        completed_count = 0

        for task in project_tasks:
            status_name = agent_pb2.TaskStatus.Name(task.status)
            status_counts[status_name] = status_counts.get(status_name, 0) + 1

            if task.status == agent_pb2.TaskStatus.COMPLETED:
                completed_count += 1

        completion_percentage = (completed_count / len(project_tasks) * 100) if project_tasks else 0

        # Determine overall project status
        if completion_percentage == 100:
            project_status = "completed"
        elif any(task.status == agent_pb2.TaskStatus.FAILED for task in project_tasks):
            project_status = "failed"
        elif any(task.status == agent_pb2.TaskStatus.IN_PROGRESS for task in project_tasks):
            project_status = "in_progress"
        else:
            project_status = "pending"

        return agent_pb2.ProjectStatusResponse(
            project=project,
            status=project_status,
            tasks=project_tasks,
            completion_percentage=completion_percentage,
            task_status_counts=status_counts,
        )

    async def CheckDependencies(
        self, request: agent_pb2.CheckDependenciesRequest, context
    ) -> agent_pb2.CheckDependenciesResponse:
        """Check if task dependencies are satisfied."""
        task_id = request.task_id

        if task_id not in self.tasks:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Task {task_id} not found")
            return agent_pb2.CheckDependenciesResponse()

        task = self.tasks[task_id]
        ready, pending = await self._check_dependencies(task_id, list(task.dependencies))

        return agent_pb2.CheckDependenciesResponse(ready=ready, pending_dependencies=pending)

    async def NotifyTaskCompletion(
        self, request: agent_pb2.TaskCompletionNotification, context
    ) -> agent_pb2.TaskCompletionResponse:
        """Handle task completion notifications."""
        task_id = request.task_id
        agent_id = request.agent_id
        final_status = request.final_status

        self.logger.info(f"Task completion notification: {task_id} by {agent_id} - {final_status}")

        # Update task status
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = final_status

            # Update metadata
            for key, value in request.result_metadata.items():
                task.metadata[key] = value

        # Check for dependent tasks that can now be triggered
        triggered_tasks = []
        if final_status == agent_pb2.TaskStatus.COMPLETED:
            triggered_tasks = await self._trigger_dependent_tasks(task_id)

        return agent_pb2.TaskCompletionResponse(acknowledged=True, dependent_tasks_triggered=triggered_tasks)

    async def _find_best_agent(self, task: agent_pb2.Task, preferred_agent_type: str = "") -> Optional[str]:
        """Find the best agent for a task based on capabilities and load."""
        suitable_agents = []

        # Define capability requirements based on task type
        task_type = task.type.lower()
        capability_requirements = self._get_capability_requirements(task_type)

        for agent_id, agent in self.agents.items():
            # Skip if preferred type specified and doesn't match
            if preferred_agent_type and agent_pb2.AgentType.Name(agent.type) != preferred_agent_type:
                continue

            # Check if agent can handle this task type
            if self._agent_can_handle_task(agent, task, capability_requirements):
                # Check current load
                current_load = len(agent.current_tasks)
                if current_load < agent.max_concurrent_tasks:
                    # Calculate suitability score
                    score = self._calculate_agent_score(agent, task, capability_requirements, current_load)
                    suitable_agents.append((agent_id, score, current_load))

        if not suitable_agents:
            return None

        # Sort by score (higher is better), then by load (lower is better)
        suitable_agents.sort(key=lambda x: (-x[1], x[2]))
        return suitable_agents[0][0]

    def _get_capability_requirements(self, task_type: str) -> List[str]:
        """Get required capabilities for a task type."""
        capability_map = {
            'frontend': ['react', 'vue', 'angular', 'css', 'typescript'],
            'backend': ['python', 'go', 'nodejs', 'api_design', 'microservices'],
            'database': ['postgresql', 'mysql', 'mongodb', 'database_design'],
            'devops': ['docker', 'kubernetes', 'aws', 'terraform'],
            'integration': ['slack', 'git', 'api_integration', 'webhooks'],
            'manager': ['planning', 'coordination', 'reporting'],
            'qa': ['testing', 'automation', 'selenium', 'pytest'],
        }
        return capability_map.get(task_type, [])

    def _agent_can_handle_task(self, agent: agent_pb2.Agent, task: agent_pb2.Task, required_caps: List[str]) -> bool:
        """Check if an agent can handle a specific task."""
        if not required_caps:
            return True  # Any agent can handle tasks with no specific requirements

        # Check if agent has any of the required capabilities
        return any(agent.capabilities.get(cap, False) for cap in required_caps)

    def _calculate_agent_score(
        self, agent: agent_pb2.Agent, task: agent_pb2.Task, required_caps: List[str], current_load: int
    ) -> float:
        """Calculate suitability score for an agent."""
        score = 0.0

        # Capability match score (0-100)
        if required_caps:
            matching_caps = sum(1 for cap in required_caps if agent.capabilities.get(cap, False))
            score += (matching_caps / len(required_caps)) * 100
        else:
            score += 50  # Base score for general tasks

        # Load penalty (0-50)
        load_penalty = (current_load / max(agent.max_concurrent_tasks, 1)) * 50
        score -= load_penalty

        # Priority bonus
        if task.priority == agent_pb2.TaskPriority.CRITICAL:
            score += 20
        elif task.priority == agent_pb2.TaskPriority.HIGH:
            score += 10

        return max(score, 0.0)

    async def _break_down_project(self, project: agent_pb2.Project) -> List[agent_pb2.Task]:
        """Break down a project into individual tasks."""
        tasks = []

        # This is a simplified project breakdown
        # In a real implementation, this would use AI/ML to intelligently break down projects

        project_type = project.tech_stack.get('type', 'web_application')

        if project_type == 'web_application':
            tasks.extend(self._create_web_app_tasks(project))
        elif project_type == 'api_service':
            tasks.extend(self._create_api_service_tasks(project))
        elif project_type == 'data_pipeline':
            tasks.extend(self._create_data_pipeline_tasks(project))
        else:
            # Generic project tasks
            tasks.extend(self._create_generic_tasks(project))

        return tasks

    def _create_web_app_tasks(self, project: agent_pb2.Project) -> List[agent_pb2.Task]:
        """Create tasks for a web application project."""
        tasks = []
        base_id = project.id

        # Planning tasks
        planning_task = self._create_task(
            f"{base_id}_planning",
            "Project Planning and Architecture",
            "Create technical specifications and project timeline",
            "manager",
            project.priority,
        )
        tasks.append(planning_task)

        # Database tasks
        db_design_task = self._create_task(
            f"{base_id}_db_design",
            "Database Schema Design",
            "Design database schema and relationships",
            "database",
            project.priority,
            dependencies=[planning_task.id],
        )
        tasks.append(db_design_task)

        # Backend tasks
        api_task = self._create_task(
            f"{base_id}_api",
            "REST API Development",
            "Implement REST API endpoints",
            "backend",
            project.priority,
            dependencies=[db_design_task.id],
        )
        tasks.append(api_task)

        # Frontend tasks
        ui_task = self._create_task(
            f"{base_id}_ui",
            "User Interface Development",
            "Create responsive user interface",
            "frontend",
            project.priority,
            dependencies=[api_task.id],
        )
        tasks.append(ui_task)

        # Integration tasks
        integration_task = self._create_task(
            f"{base_id}_integration",
            "Third-party Integrations",
            "Implement external service integrations",
            "integration",
            project.priority,
            dependencies=[api_task.id],
        )
        tasks.append(integration_task)

        # Testing tasks
        testing_task = self._create_task(
            f"{base_id}_testing",
            "Testing and Quality Assurance",
            "Implement comprehensive test suite",
            "qa",
            project.priority,
            dependencies=[ui_task.id, integration_task.id],
        )
        tasks.append(testing_task)

        # Deployment tasks
        deployment_task = self._create_task(
            f"{base_id}_deployment",
            "Deployment and DevOps",
            "Set up deployment pipeline and infrastructure",
            "devops",
            project.priority,
            dependencies=[testing_task.id],
        )
        tasks.append(deployment_task)

        return tasks

    def _create_api_service_tasks(self, project: agent_pb2.Project) -> List[agent_pb2.Task]:
        """Create tasks for an API service project."""
        # Similar to web app but focused on backend
        return self._create_generic_tasks(project)

    def _create_data_pipeline_tasks(self, project: agent_pb2.Project) -> List[agent_pb2.Task]:
        """Create tasks for a data pipeline project."""
        # Data-focused tasks
        return self._create_generic_tasks(project)

    def _create_generic_tasks(self, project: agent_pb2.Project) -> List[agent_pb2.Task]:
        """Create generic project tasks."""
        tasks = []
        base_id = project.id

        # Basic tasks for any project
        planning_task = self._create_task(
            f"{base_id}_planning", "Project Planning", f"Plan and design {project.name}", "manager", project.priority
        )
        tasks.append(planning_task)

        implementation_task = self._create_task(
            f"{base_id}_implementation",
            "Implementation",
            f"Implement core functionality for {project.name}",
            "backend",
            project.priority,
            dependencies=[planning_task.id],
        )
        tasks.append(implementation_task)

        testing_task = self._create_task(
            f"{base_id}_testing",
            "Testing",
            f"Test and validate {project.name}",
            "qa",
            project.priority,
            dependencies=[implementation_task.id],
        )
        tasks.append(testing_task)

        return tasks

    def _create_task(
        self,
        task_id: str,
        title: str,
        description: str,
        task_type: str,
        priority: agent_pb2.TaskPriority,
        dependencies: List[str] = None,
    ) -> agent_pb2.Task:
        """Create a task with the given parameters."""
        from google.protobuf.timestamp_pb2 import Timestamp

        current_time = Timestamp()
        current_time.GetCurrentTime()

        task = agent_pb2.Task(
            id=task_id,
            title=title,
            description=description,
            type=task_type,
            priority=priority,
            status=agent_pb2.TaskStatus.PENDING,
            dependencies=dependencies or [],
            created_at=current_time,
            updated_at=current_time,
        )

        return task

    async def _check_dependencies(self, task_id: str, dependencies: List[str]) -> Tuple[bool, List[str]]:
        """Check if task dependencies are satisfied."""
        pending_dependencies = []

        for dep_id in dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.status != agent_pb2.TaskStatus.COMPLETED:
                    pending_dependencies.append(dep_id)
            else:
                # Dependency task doesn't exist
                pending_dependencies.append(dep_id)

        return len(pending_dependencies) == 0, pending_dependencies

    def _update_dependency_tracking(self, task: agent_pb2.Task):
        """Update dependency tracking structures."""
        task_id = task.id
        dependencies = list(task.dependencies)

        # Update dependency mappings
        self.task_dependencies[task_id] = dependencies

        for dep_id in dependencies:
            if dep_id not in self.dependent_tasks:
                self.dependent_tasks[dep_id] = []
            self.dependent_tasks[dep_id].append(task_id)

    async def _trigger_dependent_tasks(self, completed_task_id: str) -> List[str]:
        """Trigger tasks that were waiting for the completed task."""
        triggered_tasks = []

        if completed_task_id not in self.dependent_tasks:
            return triggered_tasks

        dependent_task_ids = self.dependent_tasks[completed_task_id]

        for task_id in dependent_task_ids:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status == agent_pb2.TaskStatus.BLOCKED:
                    # Check if all dependencies are now satisfied
                    ready, pending = await self._check_dependencies(task_id, list(task.dependencies))
                    if ready:
                        # Re-distribute the task
                        distribute_request = agent_pb2.DistributeTaskRequest(task=task)
                        try:
                            response = await self.DistributeTask(distribute_request, None)
                            if response.status == "accepted":
                                triggered_tasks.append(task_id)
                                self.logger.info(f"Triggered dependent task {task_id}")
                        except Exception as e:
                            self.logger.error(f"Failed to trigger dependent task {task_id}: {e}")

        return triggered_tasks


async def start_coordinator_grpc_server(port: int = 50051):
    """Start gRPC server for the coordinator."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create server
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=20))

    # Create and add coordinator service
    coordinator_service = CoordinatorGRPCServer()
    agent_pb2_grpc.add_CoordinatorServiceServicer_to_server(coordinator_service, server)

    # Configure server address
    listen_addr = f'127.0.0.1:{port}'
    server.add_insecure_port(listen_addr)

    print(f"Starting Coordinator gRPC server on {listen_addr}")

    try:
        await server.start()
        await server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down Coordinator server...")
        await server.stop(5)


def run_coordinator_process(port: int = 50051):
    """Run coordinator in separate process."""
    asyncio.run(start_coordinator_grpc_server(port))
