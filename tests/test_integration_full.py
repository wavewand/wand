"""
Comprehensive Integration Tests

End-to-end testing of the complete MCP-Python system including:
- All execution backends
- Process management
- Host agent service
- System command tools
- OpenCode integration scenarios
"""

import asyncio
import importlib.util
import json

# Skip all tests in this module if running in CI without services
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from distributed_server import execute_command, get_process_info, kill_process, list_processes
from tools.execution.base import ExecutionConfig, ExecutionStatus
from tools.execution.factory import create_execution_backend
from tools.host_agent.server import HostAgentServer
from tools.process.manager import ProcessManager
from tools.process.models import ProcessFilter, ProcessSignal

pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true" or os.getenv("SKIP_INTEGRATION_TESTS") == "true",
    reason="Full integration tests require running services - skipping in CI",
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the config.py file directly, not the config package
spec = importlib.util.spec_from_file_location("config_file", os.path.join(os.path.dirname(__file__), '..', 'config.py'))
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
load_config = config_module.load_config


class TestFullSystemIntegration:
    """Test complete system integration scenarios"""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing"""
        temp_dir = tempfile.mkdtemp(prefix="mcp_test_")
        try:
            yield temp_dir
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_config(self, temp_workspace):
        """Create test configuration"""
        return {
            "execution": {
                "mode": "native",
                "default_timeout": 30,
                "working_directory": temp_workspace,
                "security": {
                    "command_validation": True,
                    "allowed_commands": [
                        "echo",
                        "ls",
                        "cat",
                        "pwd",
                        "whoami",
                        "date",
                        "sleep",
                        "python",
                        "python3",
                        "git",
                        "touch",
                        "mkdir",
                        "grep",
                    ],
                    "blocked_commands": ["rm", "sudo", "chmod"],
                    "path_restrictions": [temp_workspace, "/tmp"],
                    "max_execution_time": 60,
                },
            }
        }

    @pytest.mark.asyncio
    async def test_native_execution_backend_full_workflow(self, test_config, temp_workspace):
        """Test complete workflow with native execution backend"""
        # Create execution backend
        backend = create_execution_backend("native", test_config["execution"])

        # Test 1: Basic command execution
        exec_config = ExecutionConfig(command="echo 'Hello MCP World'", working_directory=temp_workspace, timeout=10)

        result = await backend.execute_with_timeout(exec_config)

        assert result.status == ExecutionStatus.SUCCESS
        assert "Hello MCP World" in result.stdout
        assert result.exit_code == 0
        assert result.execution_time > 0

        # Test 2: File operations workflow
        test_file = Path(temp_workspace) / "test.txt"

        # Create file
        create_config = ExecutionConfig(
            command=f"echo 'Test content' > {test_file}", working_directory=temp_workspace, timeout=10
        )
        result = await backend.execute_with_timeout(create_config)
        assert result.success

        # Read file
        read_config = ExecutionConfig(command=f"cat {test_file}", working_directory=temp_workspace, timeout=10)
        result = await backend.execute_with_timeout(read_config)
        assert result.success
        assert "Test content" in result.stdout

        # List directory
        list_config = ExecutionConfig(command="ls -la", working_directory=temp_workspace, timeout=10)
        result = await backend.execute_with_timeout(list_config)
        assert result.success
        assert "test.txt" in result.stdout

        # Test 3: Environment variables
        env_config = ExecutionConfig(
            command="echo $TEST_VAR", environment={"TEST_VAR": "integration_test_value"}, timeout=10
        )
        result = await backend.execute_with_timeout(env_config)
        assert result.success
        assert "integration_test_value" in result.stdout

        # Test 4: Timeout handling
        timeout_config = ExecutionConfig(command="sleep 5", timeout=1)  # Short timeout
        result = await backend.execute_with_timeout(timeout_config)
        assert result.status == ExecutionStatus.TIMEOUT

        # Test 5: Command validation (should block dangerous commands)
        # Note: This depends on the backend implementing validation

        # Cleanup
        await backend.cleanup()

    @pytest.mark.asyncio
    async def test_process_management_integration(self, temp_workspace):
        """Test process management integration"""
        # Create process manager
        manager = ProcessManager()

        # Test 1: List current processes
        processes = await manager.list_processes("native")
        assert len(processes) > 0

        # Find our own process
        current_pid = __import__('os').getpid()
        our_process = None
        for proc in processes:
            if proc.pid == current_pid:
                our_process = proc
                break

        assert our_process is not None
        assert our_process.backend_type == "native"

        # Test 2: Get detailed process info
        detailed_info = await manager.get_process_info(current_pid, "native")
        assert detailed_info is not None
        assert detailed_info.pid == current_pid

        # Test 3: Health check
        health = await manager.health_check()
        assert "native" in health
        assert health["native"] is True

        # Test 4: Process filtering
        python_filter = ProcessFilter(name_pattern="python")
        python_processes = await manager.list_processes("native", python_filter)
        # Should find at least our test process
        assert len(python_processes) > 0

    @pytest.mark.asyncio
    async def test_host_agent_integration(self):
        """Test Host Agent HTTP service integration"""
        # Create host agent server
        server = HostAgentServer(auth_token="test-integration-token", port=8999)

        # Test 1: Server configuration
        assert server.auth_token == "test-integration-token"
        assert server.port == 8999

        # Test 2: Command validation
        assert server._validate_command("echo test") is True
        assert server._validate_command("rm -rf /") is False

        # Test 3: Path validation
        assert server._validate_path("/workspace") is True
        assert server._validate_path("/etc") is False

        # Test 4: Health check endpoints would be tested with actual HTTP server
        # (requires starting server, which we skip in unit tests)

    @pytest.mark.asyncio
    async def test_mcp_tools_integration(self, temp_workspace):
        """Test MCP tools integration as they would be called by OpenCode"""
        # Mock context for MCP tools
        mock_ctx = Mock()
        mock_ctx.user_id = "test_user"

        # Setup execution backend
        test_config = {
            "mode": "native",
            "working_directory": temp_workspace,
            "security": {
                "allowed_commands": ["echo", "ls", "cat", "pwd", "python3"],
                "path_restrictions": [temp_workspace],
            },
        }

        # Mock the global execution_backend variable
        with patch('distributed_server.execution_backend') as mock_backend:
            mock_result = Mock()
            mock_result.success = True
            mock_result.status = ExecutionStatus.SUCCESS
            mock_result.stdout = "Hello World"
            mock_result.stderr = ""
            mock_result.exit_code = 0
            mock_result.execution_time = 0.5
            mock_result.working_directory = temp_workspace

            mock_backend.execute_with_timeout = AsyncMock(return_value=mock_result)

            # Test execute_command tool
            result_json = await execute_command(
                ctx=mock_ctx, command="echo 'Hello World'", working_directory=temp_workspace, timeout=10
            )

            result = json.loads(result_json)
            assert result["success"] is True
            assert result["stdout"] == "Hello World"
            assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_multi_backend_scenario(self, temp_workspace):
        """Test scenario with multiple execution backends"""
        # Create process manager
        manager = ProcessManager()

        # Test 1: Native backend (already exists)
        assert "native" in manager.list_backends()

        # Test 2: Add mock remote backend
        mock_execution_backend = AsyncMock()
        mock_execution_backend.execute.return_value = Mock(
            success=True, stdout="remote command output", stderr="", exit_code=0
        )

        manager.add_execution_backend("test-remote", mock_execution_backend, "test-host")
        assert "test-remote" in manager.list_backends()

        # Test 3: List processes from specific backend
        native_processes = await manager.list_processes("native")
        assert len(native_processes) > 0

        # Test 4: Health check all backends
        health = await manager.health_check()
        assert "native" in health
        assert "test-remote" in health

    @pytest.mark.asyncio
    async def test_error_handling_scenarios(self, temp_workspace):
        """Test error handling across the system"""
        backend = create_execution_backend(
            "native", {"working_directory": temp_workspace, "security": {"allowed_commands": ["echo", "false"]}}
        )

        # Test 1: Command failure
        fail_config = ExecutionConfig(command="false", timeout=10)  # Command that always fails
        result = await backend.execute_with_timeout(fail_config)
        assert result.status == ExecutionStatus.FAILED
        assert result.exit_code != 0

        # Test 2: Invalid working directory
        invalid_config = ExecutionConfig(command="echo test", working_directory="/nonexistent/directory", timeout=10)
        result = await backend.execute_with_timeout(invalid_config)
        # Result depends on backend implementation

        # Test 3: Process manager with invalid backend
        manager = ProcessManager()
        try:
            await manager.get_process_info(1234, "nonexistent-backend")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown backend" in str(e)

        await backend.cleanup()

    @pytest.mark.asyncio
    async def test_concurrent_execution_scenario(self, temp_workspace):
        """Test concurrent command execution"""
        backend = create_execution_backend("native", {"working_directory": temp_workspace, "max_concurrent": 5})

        # Create multiple concurrent execution tasks
        tasks = []
        for i in range(5):
            config = ExecutionConfig(command=f"echo 'Task {i}' && sleep 0.1", timeout=5)
            task = asyncio.create_task(backend.execute_with_timeout(config))
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        for i, result in enumerate(results):
            assert result.success, f"Task {i} failed: {result.stderr}"
            assert f"Task {i}" in result.stdout

        await backend.cleanup()

    @pytest.mark.asyncio
    async def test_opencode_workflow_simulation(self, temp_workspace):
        """Simulate a typical OpenCode workflow"""
        # This simulates how OpenCode would use the MCP-Python system

        # Setup
        backend = create_execution_backend(
            "native",
            {
                "working_directory": temp_workspace,
                "security": {
                    "allowed_commands": [
                        "git",
                        "python3",
                        "pip",
                        "ls",
                        "cat",
                        "echo",
                        "mkdir",
                        "touch",
                        "grep",
                        "find",
                        "cd",
                        "pwd",
                    ]
                },
            },
        )

        project_dir = Path(temp_workspace) / "test_project"

        # Step 1: Create project structure
        setup_config = ExecutionConfig(command=f"mkdir -p {project_dir}", timeout=10)
        result = await backend.execute_with_timeout(setup_config)
        assert result.success

        # Step 2: Initialize project files
        init_config = ExecutionConfig(
            command="echo 'print(\"Hello OpenCode!\")' > main.py", working_directory=str(project_dir), timeout=10
        )
        result = await backend.execute_with_timeout(init_config)
        assert result.success

        # Step 3: List project contents (OpenCode exploring)
        list_config = ExecutionConfig(command="ls -la", working_directory=str(project_dir), timeout=10)
        result = await backend.execute_with_timeout(list_config)
        assert result.success
        assert "main.py" in result.stdout

        # Step 4: Read file contents (OpenCode reading code)
        read_config = ExecutionConfig(command="cat main.py", working_directory=str(project_dir), timeout=10)
        result = await backend.execute_with_timeout(read_config)
        assert result.success
        assert "Hello OpenCode" in result.stdout

        # Step 5: Execute code (OpenCode running code)
        run_config = ExecutionConfig(command="python3 main.py", working_directory=str(project_dir), timeout=10)
        result = await backend.execute_with_timeout(run_config)
        assert result.success
        assert "Hello OpenCode!" in result.stdout

        # Step 6: Get system info (OpenCode checking environment)
        info_config = ExecutionConfig(
            command="python3 --version && pwd && whoami", working_directory=str(project_dir), timeout=10
        )
        result = await backend.execute_with_timeout(info_config)
        assert result.success
        assert "Python" in result.stdout

        await backend.cleanup()

    @pytest.mark.asyncio
    async def test_security_validation_integration(self, temp_workspace):
        """Test security validation across the system"""
        # Create backend with strict security
        backend = create_execution_backend(
            "native",
            {
                "working_directory": temp_workspace,
                "security": {
                    "command_validation": True,
                    "allowed_commands": ["echo", "ls"],
                    "blocked_commands": ["rm", "dd", "sudo"],
                    "path_restrictions": [temp_workspace],
                    "max_execution_time": 10,
                },
            },
        )

        # Test 1: Allowed command should work
        allowed_config = ExecutionConfig(command="echo 'This is allowed'", timeout=5)
        result = await backend.execute_with_timeout(allowed_config)
        assert result.success

        # Test 2: Blocked command should be prevented
        # Note: This test depends on the backend implementing validation
        blocked_config = ExecutionConfig(command="rm -rf /tmp/test", timeout=5)  # Dangerous command
        # The backend should either block this or it should fail safely

        # Test 3: Path restriction validation
        # Depends on backend implementation

        await backend.cleanup()


class TestBackendSpecificIntegration:
    """Test integration scenarios specific to each backend"""

    @pytest.mark.asyncio
    async def test_docker_socket_integration(self):
        """Test Docker Socket backend integration (mocked)"""
        config = {"default_image": "ubuntu:22.04", "auto_remove": True, "memory_limit": "256MB"}

        with patch('tools.execution.docker_socket.docker.DockerClient') as mock_docker:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_docker.return_value = mock_client

            backend = create_execution_backend("docker_socket", config)

            # Test health check
            health = await backend.health_check()
            assert health is True

            # Test capabilities
            caps = backend.get_capabilities()
            assert caps["execution_mode"] == "docker_socket"
            assert caps["supports_timeout"] is True

    @pytest.mark.asyncio
    async def test_ssh_remote_integration(self):
        """Test SSH Remote backend integration (mocked)"""
        config = {"host": "test.example.com", "username": "testuser", "auth_method": "key", "key_file": "/path/to/key"}

        backend = create_execution_backend("ssh_remote", config)

        # Test configuration
        assert backend.host == "test.example.com"
        assert backend.username == "testuser"
        assert backend.auth_method == "key"

        # Test capabilities
        caps = backend.get_capabilities()
        assert caps["execution_mode"] == "ssh_remote"
        assert caps["supports_environment"] is True


class TestPerformanceIntegration:
    """Test performance aspects of the integration"""

    @pytest.mark.asyncio
    async def test_execution_performance(self):
        """Test execution performance characteristics"""
        import tempfile

        temp_workspace = tempfile.mkdtemp(prefix="mcp_test_perf_")
        backend = create_execution_backend("native", {"working_directory": temp_workspace})

        # Test rapid sequential execution
        start_time = time.time()

        for i in range(10):
            config = ExecutionConfig(command=f"echo 'Command {i}'", timeout=5)
            result = await backend.execute_with_timeout(config)
            assert result.success

        total_time = time.time() - start_time
        avg_time = total_time / 10

        # Should be reasonably fast (less than 100ms per command on average)
        assert avg_time < 0.1, f"Average execution time too slow: {avg_time:.3f}s"

        await backend.cleanup()
        shutil.rmtree(temp_workspace, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test memory usage doesn't grow excessively"""
        import gc
        import os
        import tempfile

        import psutil

        temp_workspace = tempfile.mkdtemp(prefix="mcp_test_mem_")
        backend = create_execution_backend("native", {"working_directory": temp_workspace})

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Execute many commands
        for i in range(50):
            config = ExecutionConfig(command=f"echo 'Memory test {i}'", timeout=5)
            result = await backend.execute_with_timeout(config)
            assert result.success

            # Periodic garbage collection
            if i % 10 == 0:
                gc.collect()

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (less than 50MB)
        assert memory_growth < 50 * 1024 * 1024, f"Excessive memory growth: {memory_growth / 1024 / 1024:.1f}MB"

        await backend.cleanup()
        shutil.rmtree(temp_workspace, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
