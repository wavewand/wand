"""
Tests for Docker Socket Execution Backend
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import docker
import pytest

from tools.execution.base import ExecutionConfig, ExecutionResult
from tools.execution.docker_socket import DockerSocketExecutionBackend, create_docker_socket_backend

# Skip entire module if docker is not available
pytest.importorskip("docker")

# Check if Docker daemon is running
try:
    client = docker.from_env()
    client.ping()
    DOCKER_AVAILABLE = True
except BaseException:
    DOCKER_AVAILABLE = False

pytestmark = pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker daemon not available")


class TestDockerSocketExecutionBackend:
    """Test Docker Socket execution backend"""

    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            "default_image": "ubuntu:22.04",
            "network": "bridge",
            "auto_remove": True,
            "memory_limit": "512MB",
            "cpu_limit": "0.5",
            "working_dir": "/workspace",
            "socket_path": "/var/run/docker.sock",
            "volume_mounts": {"/tmp": {"container_path": "/tmp", "mode": "rw"}},
        }

    @pytest.fixture
    def mock_docker_client(self):
        """Mock Docker client"""
        client = Mock()
        client.ping.return_value = True
        client.info.return_value = {"ContainersRunning": 2, "Containers": 10, "Images": 50, "Driver": "overlay2"}
        client.version.return_value = {"Version": "20.10.8", "ApiVersion": "1.41"}
        return client

    @pytest.fixture
    def mock_container(self):
        """Mock Docker container"""
        container = Mock()
        container.attrs = {"State": {"ExitCode": 0}}
        container.logs.return_value = b"test output\n"
        container.wait.return_value = {"StatusCode": 0}
        container.id = "abc123def456"
        return container

    @patch('tools.execution.docker_socket.docker.DockerClient')
    def test_backend_initialization(self, mock_docker_class, config, mock_docker_client):
        """Test backend initialization"""
        mock_docker_class.return_value = mock_docker_client

        backend = DockerSocketExecutionBackend(config)

        assert backend.default_image == "ubuntu:22.04"
        assert backend.network == "bridge"
        assert backend.auto_remove is True
        assert backend.memory_limit == "512MB"
        assert backend.cpu_limit == "0.5"
        assert backend.container_working_dir == "/workspace"

        mock_docker_class.assert_called_once_with(base_url="unix:///var/run/docker.sock")
        mock_docker_client.ping.assert_called_once()

    def test_factory_function(self, config):
        """Test factory function"""
        with patch('tools.execution.docker_socket.docker.DockerClient'):
            backend = create_docker_socket_backend(config)
            assert isinstance(backend, DockerSocketExecutionBackend)

    @patch('tools.execution.docker_socket.docker.DockerClient')
    def test_prepare_environment(self, mock_docker_class, config, mock_docker_client):
        """Test environment preparation"""
        mock_docker_class.return_value = mock_docker_client
        backend = DockerSocketExecutionBackend(config)

        # Test with custom environment
        env = {"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"}
        result = backend._prepare_environment(env)

        assert "TEST_VAR=test_value" in result
        assert "ANOTHER_VAR=another_value" in result
        assert "DEBIAN_FRONTEND=noninteractive" in result
        assert "LANG=C.UTF-8" in result
        assert "LC_ALL=C.UTF-8" in result

        # Test with None environment
        result = backend._prepare_environment(None)
        assert "DEBIAN_FRONTEND=noninteractive" in result
        assert len([env for env in result if env.startswith("TEST_VAR=")]) == 0

    @patch('tools.execution.docker_socket.docker.DockerClient')
    def test_prepare_volumes(self, mock_docker_class, config, mock_docker_client):
        """Test volume preparation"""
        mock_docker_class.return_value = mock_docker_client
        backend = DockerSocketExecutionBackend(config)

        # Test with working directory
        volumes = backend._prepare_volumes("/host/workspace")

        # Should include configured volume mounts
        assert "/tmp" in volumes
        assert volumes["/tmp"]["bind"] == "/tmp"
        assert volumes["/tmp"]["mode"] == "rw"

        # Should include working directory mount
        assert "/host/workspace" in volumes
        assert volumes["/host/workspace"]["bind"] == "/workspace"
        assert volumes["/host/workspace"]["mode"] == "rw"

        # Test without working directory
        volumes = backend._prepare_volumes(None)
        assert "/tmp" in volumes  # Configured mount should still be there
        assert len([v for v in volumes if volumes[v]["bind"] == "/workspace"]) == 0

    @patch('tools.execution.docker_socket.docker.DockerClient')
    async def test_execute_success(self, mock_docker_class, config, mock_docker_client, mock_container):
        """Test successful command execution"""
        mock_docker_class.return_value = mock_docker_client
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.logs.side_effect = [b"test output\n", b""]  # stdout, stderr

        backend = DockerSocketExecutionBackend(config)

        with patch.object(backend, '_run_container_async', return_value=mock_container):
            result = await backend.execute(ExecutionConfig(command="echo test", timeout=10))

        assert result.success is True
        assert result.stdout == "test output\n"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.execution_time > 0

    @patch('tools.execution.docker_socket.docker.DockerClient')
    async def test_execute_failure(self, mock_docker_class, config, mock_docker_client, mock_container):
        """Test failed command execution"""
        mock_docker_class.return_value = mock_docker_client
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.attrs = {"State": {"ExitCode": 1}}
        mock_container.logs.side_effect = [b"", b"command not found\n"]  # stdout, stderr

        backend = DockerSocketExecutionBackend(config)

        with patch.object(backend, '_run_container_async', return_value=mock_container):
            result = await backend.execute(ExecutionConfig(command="nonexistent-command", timeout=10))

        assert result.success is False
        assert result.stdout == ""
        assert result.stderr == "command not found\n"
        assert result.exit_code == 1

    @patch('tools.execution.docker_socket.docker.DockerClient')
    async def test_execute_timeout(self, mock_docker_class, config, mock_docker_client):
        """Test command execution timeout"""
        mock_docker_class.return_value = mock_docker_client
        backend = DockerSocketExecutionBackend(config)

        # Mock timeout exception
        with patch.object(backend, '_run_container_async', side_effect=asyncio.TimeoutError()):
            result = await backend.execute(ExecutionConfig(command="sleep 60", timeout=1))

        assert result.success is False
        assert "timed out" in result.stderr
        assert result.exit_code == -1

    @patch('tools.execution.docker_socket.docker.DockerClient')
    async def test_execute_blocked_command(self, mock_docker_class, config, mock_docker_client):
        """Test execution of blocked command"""
        mock_docker_class.return_value = mock_docker_client
        backend = DockerSocketExecutionBackend(config)

        # Mock validate_command to return False
        with patch.object(backend, 'validate_command', return_value=False):
            result = await backend.execute(ExecutionConfig(command="rm -rf /", timeout=10))

        assert result.success is False
        assert "blocked by security policy" in result.stderr
        assert result.exit_code == -1

    @patch('tools.execution.docker_socket.docker.DockerClient')
    async def test_execute_with_environment(self, mock_docker_class, config, mock_docker_client, mock_container):
        """Test execution with environment variables"""
        mock_docker_class.return_value = mock_docker_client
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.logs.side_effect = [b"test_value\n", b""]

        backend = DockerSocketExecutionBackend(config)

        with patch.object(backend, '_run_container_async', return_value=mock_container) as mock_run:
            await backend.execute(
                ExecutionConfig(command="echo $TEST_VAR", environment={"TEST_VAR": "test_value"}, timeout=10)
            )

        # Verify container was created with environment
        call_args = mock_run.call_args[0][0]  # First argument (config)
        env_vars = call_args["environment"]
        assert "TEST_VAR=test_value" in env_vars

    @patch('tools.execution.docker_socket.docker.DockerClient')
    async def test_execute_with_working_directory(self, mock_docker_class, config, mock_docker_client, mock_container):
        """Test execution with working directory"""
        mock_docker_class.return_value = mock_docker_client
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.logs.side_effect = [b"/workspace\n", b""]

        backend = DockerSocketExecutionBackend(config)

        with patch.object(backend, '_run_container_async', return_value=mock_container) as mock_run:
            await backend.execute(ExecutionConfig(command="pwd", working_directory="/host/project", timeout=10))

        # Verify container was created with volume mount
        call_args = mock_run.call_args[0][0]  # First argument (config)
        volumes = call_args["volumes"]
        assert "/host/project" in volumes
        assert volumes["/host/project"]["bind"] == "/workspace"

    @patch('tools.execution.docker_socket.docker.DockerClient')
    async def test_health_check_healthy(self, mock_docker_class, config, mock_docker_client):
        """Test health check when Docker is healthy"""
        mock_docker_class.return_value = mock_docker_client
        backend = DockerSocketExecutionBackend(config)

        health = await backend.health_check()

        assert health is True
        # Verify that the Docker client methods were called
        mock_docker_client.ping.assert_called()

    @patch('tools.execution.docker_socket.docker.DockerClient')
    async def test_health_check_unhealthy(self, mock_docker_class, config, mock_docker_client):
        """Test health check when Docker is unhealthy"""
        mock_docker_class.return_value = mock_docker_client

        # First call to ping() during __init__ should succeed
        # Second call to ping() during health_check() should fail
        mock_docker_client.ping.side_effect = [None, Exception("Docker daemon not running")]

        backend = DockerSocketExecutionBackend(config)

        health = await backend.health_check()

        assert health is False

    @patch('tools.execution.docker_socket.docker.DockerClient')
    def test_get_capabilities(self, mock_docker_class, config, mock_docker_client):
        """Test getting backend capabilities"""
        mock_docker_class.return_value = mock_docker_client
        backend = DockerSocketExecutionBackend(config)

        capabilities = backend.get_capabilities()

        assert capabilities["execution_mode"] == "docker_socket"
        assert capabilities["supports_timeout"] is True
        assert capabilities["supports_environment"] is True
        assert capabilities["supports_working_directory"] is True
        assert capabilities["supports_input"] is True
        assert capabilities["isolation_level"] == "container"

        assert "container_isolation" in capabilities["features"]
        assert "resource_limits" in capabilities["features"]
        assert "volume_mounts" in capabilities["features"]

        resource_limits = capabilities["resource_limits"]
        assert resource_limits["memory"] == "512MB"
        assert resource_limits["cpu"] == "0.5"

    @patch('tools.execution.docker_socket.docker.DockerClient')
    async def test_cleanup(self, mock_docker_class, config, mock_docker_client):
        """Test backend cleanup"""
        mock_docker_class.return_value = mock_docker_client

        # Mock containers list
        mock_containers = [Mock(), Mock()]
        for i, container in enumerate(mock_containers):
            container.id = f"container{i}"
            container.remove = Mock()

        mock_docker_client.containers.list.return_value = mock_containers

        backend = DockerSocketExecutionBackend(config)

        await backend.cleanup()

        # Verify containers were cleaned up
        mock_docker_client.containers.list.assert_called_once_with(all=True, filters={"label": "mcp-python-execution"})

        for container in mock_containers:
            container.remove.assert_called_once_with(force=True)

        mock_docker_client.close.assert_called_once()

    @patch('tools.execution.docker_socket.docker.DockerClient')
    async def test_run_container_async(self, mock_docker_class, config, mock_docker_client, mock_container):
        """Test asynchronous container execution"""
        mock_docker_class.return_value = mock_docker_client
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}

        backend = DockerSocketExecutionBackend(config)

        container_config = {"image": "ubuntu:22.04", "command": ["echo", "test"], "detach": False}

        result = await backend._run_container_async(container_config, None, 10)

        assert result == mock_container
        mock_docker_client.containers.create.assert_called_once_with(**container_config)
        mock_container.start.assert_called_once()
        mock_container.wait.assert_called_once_with(timeout=10)

    @patch('tools.execution.docker_socket.docker.DockerClient')
    async def test_run_container_with_input(self, mock_docker_class, config, mock_docker_client, mock_container):
        """Test container execution with input data"""
        mock_docker_class.return_value = mock_docker_client
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}

        # Mock socket for input
        mock_socket = Mock()
        mock_socket._sock.send = Mock()
        mock_socket._sock.close = Mock()
        mock_container.attach_socket.return_value = mock_socket

        backend = DockerSocketExecutionBackend(config)

        container_config = {"image": "ubuntu:22.04", "command": ["cat"], "stdin": True}

        result = await backend._run_container_async(container_config, "test input", 10)

        assert result == mock_container
        mock_container.attach_socket.assert_called_once_with(params={'stdin': 1, 'stream': 1})
        mock_socket._sock.send.assert_called_once_with(b"test input")
        mock_socket._sock.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
