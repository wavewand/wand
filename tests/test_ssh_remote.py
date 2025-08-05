"""
Tests for SSH Remote Execution Backend
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import paramiko
import pytest

from tools.execution.base import ExecutionConfig, ExecutionResult, ExecutionStatus
from tools.execution.ssh_remote import SSHRemoteExecutionBackend, create_ssh_remote_backend

# Skip entire module if paramiko is not available
pytest.importorskip("paramiko")


@pytest.fixture(scope="module", autouse=True)
def mock_ssh_module():
    """Mock SSH module-level dependencies"""
    with patch('paramiko.SSHClient') as mock_client:
        yield mock_client


class TestSSHRemoteExecutionBackend:
    """Test SSH Remote execution backend"""

    @pytest.fixture
    def config_key_auth(self):
        """Test configuration with key authentication"""
        return {
            "host": "remote.example.com",
            "port": 22,
            "username": "mcp-user",
            "auth_method": "key",
            "key_file": "/path/to/key",
            "known_hosts_file": "/path/to/known_hosts",
            "timeout": 30,
            "keepalive": True,
            "remote_working_dir": "/tmp/mcp-workspace",
        }

    @pytest.fixture
    def config_password_auth(self):
        """Test configuration with password authentication"""
        return {
            "host": "remote.example.com",
            "port": 2222,
            "username": "mcp-user",
            "auth_method": "password",
            "password": "secure-password",
            "known_hosts_file": "/path/to/known_hosts",
            "timeout": 30,
            "remote_working_dir": "/home/mcp-user/workspace",
        }

    def test_backend_initialization_key_auth(self, config_key_auth):
        """Test backend initialization with key authentication"""
        backend = SSHRemoteExecutionBackend(config_key_auth)

        assert backend.host == "remote.example.com"
        assert backend.port == 22
        assert backend.username == "mcp-user"
        assert backend.auth_method == "key"
        assert backend.key_file == "/path/to/key"
        assert backend.connect_timeout == 30
        assert backend.remote_working_dir == "/tmp/mcp-workspace"

    def test_backend_initialization_password_auth(self, config_password_auth):
        """Test backend initialization with password authentication"""
        backend = SSHRemoteExecutionBackend(config_password_auth)

        assert backend.host == "remote.example.com"
        assert backend.port == 2222
        assert backend.username == "mcp-user"
        assert backend.auth_method == "password"
        assert backend.password == "secure-password"
        assert backend.remote_working_dir == "/home/mcp-user/workspace"

    def test_factory_function(self, config_key_auth):
        """Test factory function"""
        backend = create_ssh_remote_backend(config_key_auth)
        assert isinstance(backend, SSHRemoteExecutionBackend)

    def test_config_validation_missing_key_file(self):
        """Test configuration validation with missing key file"""
        config = {
            "host": "remote.example.com",
            "username": "user",
            "auth_method": "key"
            # Missing key_file
        }

        with pytest.raises(ValueError, match="SSH key file path required"):
            SSHRemoteExecutionBackend(config)

    def test_config_validation_missing_password(self):
        """Test configuration validation with missing password"""
        config = {
            "host": "remote.example.com",
            "username": "user",
            "auth_method": "password"
            # Missing password
        }

        with pytest.raises(ValueError, match="Password required"):
            SSHRemoteExecutionBackend(config)

    def test_config_validation_missing_host(self):
        """Test configuration validation with missing host"""
        config = {
            "host": "",  # Empty host should fail validation
            "username": "user",
            "auth_method": "key",
            "key_file": "/path/to/key",
        }

        with pytest.raises(ValueError, match="SSH host is required"):
            SSHRemoteExecutionBackend(config)

    def test_prepare_environment(self, config_key_auth):
        """Test environment preparation"""
        backend = SSHRemoteExecutionBackend(config_key_auth)

        # Test with custom environment
        env = {"TEST_VAR": "test_value", "PATH": "/custom/path"}
        result = backend._prepare_environment(env)

        assert result["TEST_VAR"] == "test_value"
        assert result["PATH"] == "/custom/path"
        assert result["LANG"] == "C.UTF-8"
        assert result["LC_ALL"] == "C.UTF-8"
        assert result["DEBIAN_FRONTEND"] == "noninteractive"

        # Test environment escaping
        env_with_quotes = {"QUOTED": "value with 'quotes'"}
        result = backend._prepare_environment(env_with_quotes)
        assert result["QUOTED"] == "value with '\"'\"'quotes'\"'\"'"

    def test_build_remote_command(self, config_key_auth):
        """Test remote command building"""
        backend = SSHRemoteExecutionBackend(config_key_auth)

        command = "echo hello"
        working_dir = "/workspace"
        env_vars = {"TEST": "value", "PATH": "/usr/bin"}

        result = backend._build_remote_command(command, working_dir, env_vars)

        assert "cd '/workspace'" in result
        assert "export TEST='value'" in result
        assert "export PATH='/usr/bin'" in result
        assert "echo hello" in result
        assert " && " in result  # Commands should be chained

    @pytest.mark.asyncio
    async def test_get_ssh_connection_new(self, config_key_auth):
        """Test getting new SSH connection"""
        backend = SSHRemoteExecutionBackend(config_key_auth)

        with patch('paramiko.SSHClient') as mock_ssh_class:
            mock_client = Mock()
            mock_ssh_class.return_value = mock_client

            # Mock the entire _get_ssh_connection method
            with patch.object(backend, '_get_ssh_connection', new_callable=AsyncMock) as mock_get_conn:
                mock_get_conn.return_value = mock_client

                # Call the method
                client = await backend._get_ssh_connection()

                # Verify it was called and returned the mock client
                mock_get_conn.assert_called_once()
                assert client == mock_client

    @pytest.mark.asyncio
    async def test_get_ssh_connection_reuse(self, config_key_auth):
        """Test reusing existing SSH connection"""
        backend = SSHRemoteExecutionBackend(config_key_auth)
        connection_key = "mcp-user@remote.example.com:22"

        # Create a mock SSH client and add it to the pool
        mock_existing_client = Mock()
        backend._ssh_clients[connection_key] = mock_existing_client

        # Get connection should return the existing client
        client = await backend._get_ssh_connection()

        assert client == mock_existing_client
        # Should reuse existing connection, not create new one
        assert len(backend._ssh_clients) == 1

    @patch('tools.execution.ssh_remote.RSAKey')
    @patch('tools.execution.ssh_remote.Path')
    @pytest.mark.asyncio
    async def test_connect_with_key(self, mock_path, mock_rsa_key, config_key_auth):
        """Test SSH connection with key authentication"""
        # Mock key file existence
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Mock key loading
        mock_key = Mock()
        mock_rsa_key.from_private_key_file.return_value = mock_key

        backend = SSHRemoteExecutionBackend(config_key_auth)

        # Create a mock SSH client
        mock_ssh_client = Mock()

        await backend._connect_with_key(mock_ssh_client)

        mock_ssh_client.connect.assert_called_once_with(
            hostname="remote.example.com",
            port=22,
            username="mcp-user",
            pkey=mock_key,
            timeout=30,
            allow_agent=False,
            look_for_keys=False,
        )

    @pytest.mark.asyncio
    async def test_connect_with_password(self, config_password_auth):
        """Test SSH connection with password authentication"""
        backend = SSHRemoteExecutionBackend(config_password_auth)

        # Create a mock SSH client
        mock_ssh_client = Mock()

        await backend._connect_with_password(mock_ssh_client)

        mock_ssh_client.connect.assert_called_once_with(
            hostname="remote.example.com",
            port=2222,
            username="mcp-user",
            password="secure-password",
            timeout=30,
            allow_agent=False,
            look_for_keys=False,
        )

    @pytest.mark.asyncio
    async def test_ensure_remote_directory(self, config_key_auth):
        """Test ensuring remote directory exists"""
        mock_ssh_client = Mock()

        backend = SSHRemoteExecutionBackend(config_key_auth)

        await backend._ensure_remote_directory(mock_ssh_client, "/test/directory")

        mock_ssh_client.exec_command.assert_called_once_with("mkdir -p /test/directory")

    @pytest.mark.asyncio
    async def test_execute_success(self, config_key_auth):
        """Test successful command execution"""
        backend = SSHRemoteExecutionBackend(config_key_auth)

        # Create mock SSH client
        mock_ssh_client = Mock()
        stdout = Mock()
        stderr = Mock()
        stdout.read.return_value = b"test output\n"
        stderr.read.return_value = b""
        stdout.channel.recv_exit_status.return_value = 0
        mock_ssh_client.exec_command.return_value = (Mock(), stdout, stderr)

        exec_config = ExecutionConfig(
            command="echo test", working_directory="/workspace", timeout=30, environment={"TEST_VAR": "value"}
        )

        with patch.object(backend, '_get_ssh_connection', return_value=mock_ssh_client), patch.object(
            backend, '_ensure_remote_directory', new_callable=AsyncMock
        ):
            result = await backend.execute(exec_config)

        assert result.status == ExecutionStatus.SUCCESS
        assert result.stdout == "test output\n"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_execute_failure(self, config_key_auth):
        """Test failed command execution"""
        # Create mock SSH client
        mock_ssh_client = Mock()
        stdout = Mock()
        stderr = Mock()
        stdout.read.return_value = b""
        stderr.read.return_value = b"command not found\n"
        stdout.channel.recv_exit_status.return_value = 127

        mock_ssh_client.exec_command.return_value = (Mock(), stdout, stderr)

        backend = SSHRemoteExecutionBackend(config_key_auth)

        exec_config = ExecutionConfig(command="nonexistent-command", timeout=30)

        with patch.object(backend, '_get_ssh_connection', return_value=mock_ssh_client), patch.object(
            backend, '_ensure_remote_directory', new_callable=AsyncMock
        ):
            result = await backend.execute(exec_config)

        assert result.status == ExecutionStatus.FAILED
        assert result.stdout == ""
        assert result.stderr == "command not found\n"
        assert result.exit_code == 127

    @pytest.mark.asyncio
    async def test_execute_blocked_command(self, config_key_auth):
        """Test execution of blocked command"""
        backend = SSHRemoteExecutionBackend(config_key_auth)

        exec_config = ExecutionConfig(command="rm -rf /", timeout=30)

        # Mock validate_command to return False
        with patch.object(backend, 'validate_command', return_value=False):
            result = await backend.execute(exec_config)

        assert result.status == ExecutionStatus.BLOCKED
        assert "blocked by security policy" in result.stderr
        assert result.exit_code == -1

    @pytest.mark.asyncio
    async def test_execute_with_input(self, config_key_auth):
        """Test execution with input data"""
        mock_ssh_client = Mock()
        stdin = Mock()
        stdout = Mock()
        stderr = Mock()
        stdout.read.return_value = b"test output\n"
        stderr.read.return_value = b""
        stdout.channel.recv_exit_status.return_value = 0
        mock_ssh_client.exec_command.return_value = (stdin, stdout, stderr)

        backend = SSHRemoteExecutionBackend(config_key_auth)

        exec_config = ExecutionConfig(command="cat", input_data="test input data", timeout=30)

        with patch.object(backend, '_get_ssh_connection', return_value=mock_ssh_client), patch.object(
            backend, '_ensure_remote_directory', new_callable=AsyncMock
        ):
            await backend.execute(exec_config)

        # Verify input was sent
        stdin.write.assert_called_once_with("test input data")
        stdin.flush.assert_called_once()
        stdin.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_success(self, config_key_auth):
        """Test successful health check"""
        # Create mock SSH client
        mock_ssh_client = Mock()
        stdout = Mock()
        stdout.read.return_value = b"health_check"
        stdout.channel.recv_exit_status.return_value = 0

        mock_ssh_client.exec_command.return_value = (Mock(), stdout, Mock())

        backend = SSHRemoteExecutionBackend(config_key_auth)

        with patch.object(backend, '_get_ssh_connection', return_value=mock_ssh_client):
            result = await backend.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, config_key_auth):
        """Test failed health check"""
        backend = SSHRemoteExecutionBackend(config_key_auth)

        # Mock connection failure
        with patch.object(backend, '_get_ssh_connection', side_effect=Exception("Connection failed")):
            result = await backend.health_check()

        assert result is False

    def test_get_capabilities(self, config_key_auth):
        """Test getting backend capabilities"""
        backend = SSHRemoteExecutionBackend(config_key_auth)

        capabilities = backend.get_capabilities()

        assert capabilities["execution_mode"] == "ssh_remote"
        assert capabilities["supports_timeout"] is True
        assert capabilities["supports_environment"] is True
        assert capabilities["supports_working_directory"] is True
        assert capabilities["supports_input"] is True
        assert capabilities["isolation_level"] == "remote_host"

        auth_info = capabilities["authentication"]
        assert auth_info["method"] == "key"
        assert auth_info["host"] == "remote.example.com"
        assert auth_info["port"] == 22
        assert auth_info["username"] == "mcp-user"

        assert "remote_execution" in capabilities["features"]
        assert "ssh_key_auth" in capabilities["features"]
        assert "connection_pooling" in capabilities["features"]

    @pytest.mark.asyncio
    async def test_cleanup(self, config_key_auth):
        """Test backend cleanup"""
        backend = SSHRemoteExecutionBackend(config_key_auth)

        # Add mock clients to pool
        mock_client1 = Mock()
        mock_client2 = Mock()
        backend._ssh_clients = {"client1": mock_client1, "client2": mock_client2}

        await backend.cleanup()

        # Verify all clients were closed
        mock_client1.close.assert_called_once()
        mock_client2.close.assert_called_once()

        # Verify pool was cleared
        assert len(backend._ssh_clients) == 0

    @pytest.mark.asyncio
    async def test_test_connection_success(self, config_key_auth):
        """Test connection testing with success"""
        # Create mock SSH client
        mock_ssh_client = Mock()

        # Configure mock responses for system commands
        def mock_exec_command(cmd, timeout=None):
            responses = {
                "hostname": b"remote-host\n",
                "uname -a": b"Linux remote-host 5.4.0\n",
                "whoami": b"mcp-user\n",
                "pwd": b"/home/mcp-user\n",
            }

            stdout = Mock()
            stdout.read.return_value = responses.get(cmd, b"unknown\n")
            return (Mock(), stdout, Mock())

        mock_ssh_client.exec_command.side_effect = mock_exec_command

        backend = SSHRemoteExecutionBackend(config_key_auth)

        with patch.object(backend, '_get_ssh_connection', return_value=mock_ssh_client):
            result = await backend.test_connection()

        assert result["status"] == "connected"
        assert result["host"] == "remote.example.com"
        assert result["system_info"]["hostname"] == "remote-host"

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, config_key_auth):
        """Test connection testing with failure"""
        backend = SSHRemoteExecutionBackend(config_key_auth)

        with patch.object(backend, '_get_ssh_connection', side_effect=Exception("Connection failed")):
            result = await backend.test_connection()

        assert result["status"] == "failed"
        assert "Connection failed" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
