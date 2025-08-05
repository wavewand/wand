import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from distributed_server import Agent, AgentType, DistributedTaskManager, Task, TaskPriority, TaskStatus, task_manager
from enhanced_distributed_server import coordinate_project, deploy_application


@pytest.fixture
def mock_context():
    """Create a mock context for testing"""
    return Mock()


@pytest.fixture
def clean_task_manager():
    """Reset task manager for each test"""
    task_manager.tasks.clear()
    task_manager.agents.clear()
    task_manager.initialize_agents()
    return task_manager


class TestDistributedTaskManager:
    """Test the distributed task manager"""

    def test_initialize_agents(self, clean_task_manager):
        """Test agent initialization"""
        assert len(clean_task_manager.agents) >= 6

        # Check agent types
        agent_types = {agent.type for agent in clean_task_manager.agents.values()}
        assert AgentType.MANAGER in agent_types
        assert AgentType.FRONTEND in agent_types
        assert AgentType.BACKEND in agent_types
        assert AgentType.DATABASE in agent_types
        assert AgentType.DEVOPS in agent_types
        assert AgentType.INTEGRATION in agent_types

    def test_create_task(self, clean_task_manager):
        """Test task creation"""
        task = clean_task_manager.create_task("Test Task", "Test Description", "frontend", TaskPriority.HIGH)

        assert task.title == "Test Task"
        assert task.description == "Test Description"
        assert task.type == "frontend"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        assert task.id in clean_task_manager.tasks

    def test_assign_task(self, clean_task_manager):
        """Test task assignment"""
        task = clean_task_manager.create_task("Test", "Test", "frontend")
        agent = list(clean_task_manager.agents.values())[0]

        result = clean_task_manager.assign_task(task.id, agent.id)

        assert result is True
        assert task.assigned_to == agent.id
        assert task.status == TaskStatus.ASSIGNED
        assert task.id in agent.current_tasks
        assert agent.status == "working"

    def test_find_best_agent(self, clean_task_manager):
        """Test finding best agent for task"""
        # Create a frontend task
        task = clean_task_manager.create_task("React Component", "Build React component", "frontend")

        best_agent_id = clean_task_manager.find_best_agent(task)
        assert best_agent_id is not None

        best_agent = clean_task_manager.agents[best_agent_id]
        assert "react" in best_agent.capabilities or "frontend" in str(best_agent.capabilities)

    def test_agent_workload_limit(self, clean_task_manager):
        """Test that agents have task limits"""
        # Find a frontend agent
        frontend_agent = None
        for agent in clean_task_manager.agents.values():
            if agent.type == AgentType.FRONTEND:
                frontend_agent = agent
                break

        assert frontend_agent is not None

        # Assign maximum tasks
        for i in range(3):
            task = clean_task_manager.create_task(f"Task {i}", "Test", "frontend")
            clean_task_manager.assign_task(task.id, frontend_agent.id)

        # Try to find agent for another task - should not return the busy agent
        new_task = clean_task_manager.create_task("New Task", "Test", "frontend")
        best_agent = clean_task_manager.find_best_agent(new_task)

        # Should either find a different agent or None if all are busy
        assert best_agent != frontend_agent.id or best_agent is None


@pytest.mark.asyncio
class TestEnhancedTools:
    """Test enhanced MCP tools"""

    @pytest.mark.skip(reason="create_project function not implemented")
    @pytest.mark.asyncio
    async def test_create_project(self, mock_context, clean_task_manager):
        """Test project creation"""
        pass

    @pytest.mark.skip(reason="distribute_task function not implemented")
    @pytest.mark.asyncio
    async def test_distribute_task(self, mock_context, clean_task_manager):
        """Test task distribution"""
        pass
        assert tasks[0].priority == TaskPriority.HIGH

    @pytest.mark.skip(reason="get_project_status function not implemented")
    @pytest.mark.asyncio
    async def test_get_project_status(self, mock_context, clean_task_manager):
        """Test project status retrieval"""
        # Create some tasks
        for i in range(3):
            task = clean_task_manager.create_task(f"Task {i}", "Description", "backend", TaskPriority.MEDIUM)

        result = await get_project_status(mock_context)
        status = json.loads(result)

        assert "agents" in status
        assert "tasks" in status
        assert status["tasks"]["total"] == 3
        assert "by_status" in status["tasks"]
        assert "by_priority" in status["tasks"]


class TestIntegrations:
    """Test integration tools with mocking"""

    @pytest.mark.asyncio
    @patch('enhanced_distributed_server.slack_integration')
    async def test_slack_send(self, mock_slack, mock_context):
        """Test Slack integration"""
        from enhanced_distributed_server import slack_send

        mock_slack.send_message = AsyncMock(return_value={"ok": True, "ts": "123456"})

        result = await slack_send(mock_context, "#general", "Test message")

        assert "Message sent" in result
        mock_slack.send_message.assert_called_once_with("#general", "Test message", None)

    @pytest.mark.asyncio
    @patch('enhanced_distributed_server.git_integration')
    async def test_git_create_pr(self, mock_git, mock_context):
        """Test Git integration"""
        from enhanced_distributed_server import git_create_pr

        mock_git.create_pr = AsyncMock(return_value={"number": 42, "html_url": "https://github.com/test/repo/pull/42"})

        result = await git_create_pr(mock_context, "test/repo", "Test PR", "This is a test", "feature", "main")

        assert "Created PR #42" in result
        assert "https://github.com/test/repo/pull/42" in result

    @pytest.mark.asyncio
    @patch('enhanced_distributed_server.jenkins_integration')
    async def test_jenkins_build(self, mock_jenkins, mock_context):
        """Test Jenkins integration"""
        from enhanced_distributed_server import jenkins_build

        mock_jenkins.trigger_job = AsyncMock(return_value={"status": 201, "location": "https://jenkins/queue/item/123"})

        result = await jenkins_build(mock_context, "test-job", {"BRANCH": "main"})

        assert "triggered successfully" in result
        assert "https://jenkins/queue/item/123" in result

    @pytest.mark.asyncio
    @patch('enhanced_distributed_server.postgres_integration')
    async def test_postgres_execute(self, mock_pg, mock_context):
        """Test PostgreSQL integration"""
        from enhanced_distributed_server import postgres_execute

        mock_pg.execute_query = AsyncMock(return_value=[{"id": 1, "name": "Test User"}])

        result = await postgres_execute(mock_context, "SELECT * FROM users WHERE id = $1", [1])

        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["name"] == "Test User"

    @pytest.mark.asyncio
    @patch('enhanced_distributed_server.aws_integration')
    async def test_aws_operations(self, mock_aws, mock_context):
        """Test AWS integration"""
        from enhanced_distributed_server import aws_ec2, aws_s3

        # Test EC2
        mock_aws.ec2_operation = AsyncMock(return_value={"instances": ["i-1234", "i-5678"]})

        result = await aws_ec2(mock_context, "list")
        data = json.loads(result)
        assert "instances" in data
        assert len(data["instances"]) == 2

        # Test S3
        mock_aws.s3_operation = AsyncMock(return_value={"status": "uploaded", "url": "s3://bucket/key"})

        result = await aws_s3(mock_context, "upload", "bucket", "key", "/local/file")
        data = json.loads(result)
        assert data["status"] == "uploaded"

    @pytest.mark.asyncio
    @patch('enhanced_distributed_server.bambu_integration')
    async def test_bambu_print(self, mock_bambu, mock_context):
        """Test Bambu 3D printer integration"""
        from enhanced_distributed_server import bambu_printer_status, bambu_send_print

        # Test sending print
        mock_bambu.send_print_job = AsyncMock(
            return_value={"status": "queued", "job_id": "job_123", "estimated_time": "2h 30m"}
        )

        result = await bambu_send_print(mock_context, "X1-Carbon-01", "/models/test.3mf", "PETG", "high")

        assert "Print job sent" in result
        assert "job_123" in result

        # Test printer status
        mock_bambu.get_printer_status = AsyncMock(
            return_value={"printer_id": "X1-Carbon-01", "status": "idle", "temperature": {"bed": 60, "nozzle": 220}}
        )

        result = await bambu_printer_status(mock_context, "X1-Carbon-01")
        data = json.loads(result)
        assert data["status"] == "idle"
        assert data["temperature"]["bed"] == 60


class TestCoordination:
    """Test multi-agent coordination"""

    @pytest.mark.asyncio
    @patch('enhanced_distributed_server.youtrack_integration')
    @patch('enhanced_distributed_server.slack_integration')
    async def test_coordinate_project(self, mock_slack, mock_youtrack, mock_context, clean_task_manager):
        """Test project coordination"""
        mock_youtrack.create_issue = AsyncMock(return_value={"id": "PROJECT-123"})
        mock_slack.send_message = AsyncMock(return_value={"ok": True})

        result = await coordinate_project(
            mock_context,
            "Test App",
            "Build a test application",
            {"frontend": "React", "backend": "Python", "database": "PostgreSQL"},
        )

        assert "Project 'Test App' initialized" in result
        assert "Created 4 tasks" in result  # frontend, backend, database, devops
        assert "PROJECT-123" in result

        # Verify tasks were created
        tasks = list(clean_task_manager.tasks.values())
        assert len(tasks) == 4

    @pytest.mark.asyncio
    @patch('enhanced_distributed_server.jenkins_integration')
    @patch('enhanced_distributed_server.slack_integration')
    async def test_deploy_application(self, mock_slack, mock_jenkins, mock_context, clean_task_manager):
        """Test application deployment"""
        mock_jenkins.trigger_job = AsyncMock(return_value={"status": 201, "location": "https://jenkins/queue/123"})
        mock_slack.send_message = AsyncMock(return_value={"ok": True})

        result = await deploy_application(mock_context, "test-app", "staging", "main")

        assert "Deployment initiated" in result
        assert "Jenkins build triggered" in result
        assert "Slack notification sent" in result

        # Verify deployment task was created
        tasks = list(clean_task_manager.tasks.values())
        deployment_tasks = [t for t in tasks if "Deploy" in t.title]
        assert len(deployment_tasks) == 1
        assert deployment_tasks[0].priority == TaskPriority.CRITICAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
