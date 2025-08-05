"""
OpenCode Integration Tests

Tests that simulate real OpenCode usage patterns and workflows.
These tests verify that the MCP-Python system works correctly with OpenCode.
"""

import asyncio
import json

# Skip all tests in this module if running in CI without services
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tools.execution.base import ExecutionConfig, ExecutionStatus
from tools.execution.factory import create_execution_backend
from tools.host_agent.server import HostAgentServer
from tools.process.manager import ProcessManager

pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true" or os.getenv("SKIP_INTEGRATION_TESTS") == "true",
    reason="OpenCode integration tests require running services - skipping in CI",
)


class TestOpenCodeWorkflows:
    """Test typical OpenCode development workflows"""

    @pytest.fixture
    def workspace(self):
        """Create temporary workspace mimicking OpenCode project structure"""
        temp_dir = tempfile.mkdtemp(prefix="opencode_test_")
        workspace_path = Path(temp_dir)

        # Create typical project structure
        (workspace_path / "src").mkdir()
        (workspace_path / "tests").mkdir()
        (workspace_path / "docs").mkdir()

        # Create sample files
        (workspace_path / "README.md").write_text("# Test Project\n\nThis is a test project.")
        (workspace_path / "requirements.txt").write_text("pytest>=7.0.0\nrequests>=2.28.0\n")
        (workspace_path / "src" / "__init__.py").write_text("")
        (workspace_path / "src" / "main.py").write_text(
            '''
def hello_world():
    """Return a greeting message."""
    return "Hello, World!"

if __name__ == "__main__":
    print(hello_world())
'''
        )
        (workspace_path / "tests" / "test_main.py").write_text(
            '''
import sys
sys.path.append('../src')
from main import hello_world

def test_hello_world():
    assert hello_world() == "Hello, World!"
'''
        )

        yield str(workspace_path)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def execution_backend(self, workspace):
        """Create execution backend configured for OpenCode"""
        config = {
            "working_directory": workspace,
            "default_timeout": 60,
            "security": {
                "command_validation": True,
                "allowed_commands": [
                    # Development tools
                    "python",
                    "python3",
                    "pip",
                    "pip3",
                    "poetry",
                    "pipenv",
                    "npm",
                    "yarn",
                    "node",
                    "go",
                    "cargo",
                    "rustc",
                    # Version control
                    "git",
                    "svn",
                    "hg",
                    # File operations
                    "ls",
                    "cat",
                    "head",
                    "tail",
                    "grep",
                    "find",
                    "wc",
                    "sort",
                    "echo",
                    "touch",
                    "mkdir",
                    "cp",
                    "mv",
                    # System info
                    "pwd",
                    "whoami",
                    "which",
                    "whereis",
                    "uname",
                    "date",
                    "ps",
                    "top",
                    "df",
                    "free",
                    "uptime",
                    # Build tools
                    "make",
                    "cmake",
                    "gcc",
                    "g++",
                    "clang",
                    # Testing
                    "pytest",
                    "jest",
                    "mocha",
                    "phpunit",
                    # Package managers
                    "apt",
                    "yum",
                    "dnf",
                    "brew",
                    "pacman",
                ],
                "blocked_commands": [
                    "rm",
                    "rmdir",
                    "dd",
                    "mkfs",
                    "fdisk",
                    "mount",
                    "umount",
                    "su",
                    "sudo",
                    "passwd",
                    "useradd",
                    "userdel",
                    "usermod",
                    "systemctl",
                    "service",
                    "kill",
                    "killall",
                    "reboot",
                    "shutdown",
                    "iptables",
                    "netfilter",
                    "crontab",
                ],
                "path_restrictions": [workspace, "/tmp", "/var/tmp"],
                "max_execution_time": 300,
                "max_memory": "2GB",
            },
        }

        return create_execution_backend("native", config)

    @pytest.mark.asyncio
    async def test_project_exploration_workflow(self, execution_backend, workspace):
        """Test OpenCode exploring a new project"""

        # Step 1: Get current directory (OpenCode orientation)
        pwd_config = ExecutionConfig(command="pwd", timeout=10)
        result = await execution_backend.execute_with_timeout(pwd_config)
        assert result.success
        assert workspace in result.stdout

        # Step 2: List project structure
        ls_config = ExecutionConfig(command="ls -la", timeout=10)
        result = await execution_backend.execute_with_timeout(ls_config)
        assert result.success
        assert "src" in result.stdout
        assert "tests" in result.stdout
        assert "README.md" in result.stdout

        # Step 3: Read README (understanding project)
        readme_config = ExecutionConfig(command="cat README.md", timeout=10)
        result = await execution_backend.execute_with_timeout(readme_config)
        assert result.success
        assert "Test Project" in result.stdout

        # Step 4: Explore source structure
        tree_config = ExecutionConfig(command="find . -name '*.py' | head -10", timeout=10)
        result = await execution_backend.execute_with_timeout(tree_config)
        assert result.success
        assert "src/main.py" in result.stdout
        assert "tests/test_main.py" in result.stdout

        # Step 5: Check requirements
        req_config = ExecutionConfig(command="cat requirements.txt", timeout=10)
        result = await execution_backend.execute_with_timeout(req_config)
        assert result.success
        assert "pytest" in result.stdout
        assert "requests" in result.stdout

    @pytest.mark.asyncio
    async def test_code_analysis_workflow(self, execution_backend, workspace):
        """Test OpenCode analyzing code"""

        # Step 1: Read main source file
        main_config = ExecutionConfig(command="cat src/main.py", timeout=10)
        result = await execution_backend.execute_with_timeout(main_config)
        assert result.success
        assert "hello_world" in result.stdout
        assert "def " in result.stdout

        # Step 2: Count lines of code
        loc_config = ExecutionConfig(command="find . -name '*.py' -exec wc -l {} + | tail -1", timeout=10)
        result = await execution_backend.execute_with_timeout(loc_config)
        assert result.success

        # Step 3: Search for TODO comments
        todo_config = ExecutionConfig(command="grep -r 'TODO\\|FIXME\\|XXX' . || echo 'No TODOs found'", timeout=10)
        result = await execution_backend.execute_with_timeout(todo_config)
        assert result.success

        # Step 4: Check for common patterns
        imports_config = ExecutionConfig(command="grep -r '^import\\|^from' . | head -5", timeout=10)
        result = await execution_backend.execute_with_timeout(imports_config)
        assert result.success

        # Step 5: Find function definitions
        funcs_config = ExecutionConfig(command="grep -n 'def ' src/*.py", timeout=10)
        result = await execution_backend.execute_with_timeout(funcs_config)
        assert result.success
        assert "hello_world" in result.stdout

    @pytest.mark.asyncio
    async def test_development_workflow(self, execution_backend, workspace):
        """Test OpenCode development workflow"""

        # Step 1: Check Python version
        python_config = ExecutionConfig(command="python3 --version", timeout=10)
        result = await execution_backend.execute_with_timeout(python_config)
        assert result.success
        assert "Python" in result.stdout

        # Step 2: Run the main script
        run_config = ExecutionConfig(command="python3 src/main.py", working_directory=workspace, timeout=30)
        result = await execution_backend.execute_with_timeout(run_config)
        assert result.success
        assert "Hello, World!" in result.stdout

        # Step 3: Run tests (if pytest available)
        test_config = ExecutionConfig(
            command="python3 -m pytest tests/ -v || python3 tests/test_main.py", working_directory=workspace, timeout=60
        )
        result = await execution_backend.execute_with_timeout(test_config)
        # Tests might pass or fail depending on environment, but shouldn't crash

        # Step 4: Check syntax of Python files
        syntax_config = ExecutionConfig(command="python3 -m py_compile src/main.py", timeout=30)
        result = await execution_backend.execute_with_timeout(syntax_config)
        assert result.success or "No module named py_compile" in result.stderr

    @pytest.mark.asyncio
    async def test_git_workflow(self, execution_backend, workspace):
        """Test OpenCode Git operations"""

        # Step 1: Initialize git repo
        init_config = ExecutionConfig(command="git init", timeout=10)
        result = await execution_backend.execute_with_timeout(init_config)
        assert result.success or "already exists" in result.stderr

        # Step 2: Check git status
        status_config = ExecutionConfig(command="git status || echo 'Git not available'", timeout=10)
        result = await execution_backend.execute_with_timeout(status_config)
        assert result.success

        # Step 3: Add files (if git available)
        add_config = ExecutionConfig(command="git add . || echo 'Git add failed'", timeout=10)
        result = await execution_backend.execute_with_timeout(add_config)
        assert result.success

        # Step 4: Check what would be committed
        diff_config = ExecutionConfig(command="git diff --cached --name-only || echo 'No git diff'", timeout=10)
        result = await execution_backend.execute_with_timeout(diff_config)
        assert result.success

        # Step 5: Show git log (might be empty)
        log_config = ExecutionConfig(command="git log --oneline -5 || echo 'No commits yet'", timeout=10)
        result = await execution_backend.execute_with_timeout(log_config)
        assert result.success

    @pytest.mark.asyncio
    async def test_debugging_workflow(self, execution_backend, workspace):
        """Test OpenCode debugging workflow"""

        # Step 1: Create a file with error for debugging
        error_file = Path(workspace) / "debug_test.py"
        error_file.write_text(
            '''
def buggy_function():
    x = 10
    y = 0
    return x / y  # Division by zero error

if __name__ == "__main__":
    result = buggy_function()
    print(f"Result: {result}")
'''
        )

        # Step 2: Try to run the buggy code
        debug_config = ExecutionConfig(command="python3 debug_test.py", working_directory=workspace, timeout=10)
        result = await execution_backend.execute_with_timeout(debug_config)
        assert not result.success  # Should fail
        assert "ZeroDivisionError" in result.stderr

        # Step 3: Use Python to check syntax
        syntax_config = ExecutionConfig(command="python3 -m py_compile debug_test.py", timeout=10)
        result = await execution_backend.execute_with_timeout(syntax_config)
        # Syntax should be valid even if runtime error exists

        # Step 4: Search for error patterns
        error_search_config = ExecutionConfig(command="grep -n 'y = 0' debug_test.py", timeout=10)
        result = await execution_backend.execute_with_timeout(error_search_config)
        assert result.success
        assert "y = 0" in result.stdout

    @pytest.mark.asyncio
    async def test_system_information_workflow(self, execution_backend, workspace):
        """Test OpenCode gathering system information"""

        # Step 1: Get system info
        uname_config = ExecutionConfig(command="uname -a", timeout=10)
        result = await execution_backend.execute_with_timeout(uname_config)
        assert result.success

        # Step 2: Check available disk space
        df_config = ExecutionConfig(command="df -h . | tail -1", timeout=10)
        result = await execution_backend.execute_with_timeout(df_config)
        assert result.success

        # Step 3: Check memory info
        free_config = ExecutionConfig(command="free -h || echo 'free command not available'", timeout=10)
        result = await execution_backend.execute_with_timeout(free_config)
        assert result.success

        # Step 4: List running processes (limited)
        ps_config = ExecutionConfig(command="ps aux | head -5", timeout=10)
        result = await execution_backend.execute_with_timeout(ps_config)
        assert result.success

        # Step 5: Check current user
        whoami_config = ExecutionConfig(command="whoami", timeout=10)
        result = await execution_backend.execute_with_timeout(whoami_config)
        assert result.success
        assert len(result.stdout.strip()) > 0

    @pytest.mark.asyncio
    async def test_package_management_workflow(self, execution_backend, workspace):
        """Test OpenCode package management operations"""

        # Step 1: Check pip version
        pip_config = ExecutionConfig(command="pip3 --version || pip --version", timeout=10)
        result = await execution_backend.execute_with_timeout(pip_config)
        # pip might not be available in all environments

        # Step 2: List installed packages (sample)
        list_config = ExecutionConfig(command="pip3 list | head -5 || echo 'pip not available'", timeout=30)
        result = await execution_backend.execute_with_timeout(list_config)
        assert result.success

        # Step 3: Check if specific package is installed
        check_config = ExecutionConfig(command="python3 -c 'import sys; print(sys.version)' ", timeout=10)
        result = await execution_backend.execute_with_timeout(check_config)
        assert result.success
        assert "." in result.stdout  # Should contain version with dots

        # Step 4: Show Python path
        path_config = ExecutionConfig(command="python3 -c 'import sys; print(\"\\n\".join(sys.path[:5]))'", timeout=10)
        result = await execution_backend.execute_with_timeout(path_config)
        assert result.success

    @pytest.mark.asyncio
    async def test_error_handling_scenarios(self, execution_backend, workspace):
        """Test OpenCode error handling scenarios"""

        # Test 1: Command not found
        missing_config = ExecutionConfig(command="nonexistent_command_12345", timeout=5)
        result = await execution_backend.execute_with_timeout(missing_config)
        assert not result.success
        assert result.exit_code != 0

        # Test 2: Permission denied (try to access restricted area)
        perm_config = ExecutionConfig(command="ls /root 2>&1 || echo 'Permission denied as expected'", timeout=5)
        result = await execution_backend.execute_with_timeout(perm_config)
        assert result.success  # The || echo part should make it succeed

        # Test 3: Timeout scenario
        timeout_config = ExecutionConfig(command="sleep 10", timeout=1)
        result = await execution_backend.execute_with_timeout(timeout_config)
        assert result.status == ExecutionStatus.TIMEOUT

        # Test 4: Invalid Python syntax
        syntax_error_config = ExecutionConfig(command="python3 -c 'print(\"missing quote'", timeout=5)
        result = await execution_backend.execute_with_timeout(syntax_error_config)
        assert not result.success
        assert "SyntaxError" in result.stderr

    @pytest.mark.asyncio
    async def test_multi_step_workflow(self, execution_backend, workspace):
        """Test complex multi-step OpenCode workflow"""

        # Step 1: Create a new Python module
        create_config = ExecutionConfig(
            command="echo 'def add(a, b):\\n    return a + b\\n\\ndef multiply(a, b):\\n    return a * b' > src/math_utils.py",
            working_directory=workspace,
            timeout=10,
        )
        result = await execution_backend.execute_with_timeout(create_config)
        assert result.success

        # Step 2: Verify the file was created
        check_config = ExecutionConfig(command="ls -la src/", timeout=10)
        result = await execution_backend.execute_with_timeout(check_config)
        assert result.success
        assert "math_utils.py" in result.stdout

        # Step 3: Test the new module
        test_module_config = ExecutionConfig(
            command="python3 -c 'import sys; sys.path.append(\"src\"); from math_utils import add, multiply; print(f\"add(2,3)={add(2,3)}, multiply(4,5)={multiply(4,5)}\")'",
            working_directory=workspace,
            timeout=10,
        )
        result = await execution_backend.execute_with_timeout(test_module_config)
        assert result.success
        assert "add(2,3)=5" in result.stdout
        assert "multiply(4,5)=20" in result.stdout

        # Step 4: Create a test for the new module
        test_create_config = ExecutionConfig(
            command="echo 'import sys\\nimport os\\nsys.path.append(os.path.join(os.path.dirname(__file__), \"../src\"))\\nfrom math_utils import add, multiply\\n\\ndef test_add():\\n    assert add(2, 3) == 5\\n\\ndef test_multiply():\\n    assert multiply(4, 5) == 20\\n\\nif __name__ == \"__main__\":\\n    test_add()\\n    test_multiply()\\n    print(\"All tests passed!\")' > tests/test_math_utils.py",
            working_directory=workspace,
            timeout=10,
        )
        result = await execution_backend.execute_with_timeout(test_create_config)
        assert result.success

        # Step 5: Run the new test
        run_test_config = ExecutionConfig(
            command="python3 tests/test_math_utils.py", working_directory=workspace, timeout=10
        )
        result = await execution_backend.execute_with_timeout(run_test_config)
        assert result.success
        assert "All tests passed!" in result.stdout


class TestOpenCodeMCPIntegration:
    """Test MCP protocol integration as OpenCode would use it"""

    @pytest.mark.asyncio
    async def test_mcp_tool_responses(self):
        """Test that MCP tools return properly formatted responses for OpenCode"""
        import tempfile

        workspace = tempfile.mkdtemp(prefix="opencode_mcp_test_")

        # Mock the global execution backend
        with patch('distributed_server.execution_backend') as mock_backend:
            mock_result = Mock()
            mock_result.success = True
            mock_result.status = ExecutionStatus.SUCCESS
            mock_result.stdout = "Hello OpenCode"
            mock_result.stderr = ""
            mock_result.exit_code = 0
            mock_result.execution_time = 0.5
            mock_result.working_directory = workspace

            mock_backend.execute_with_timeout = AsyncMock(return_value=mock_result)

            # Import the MCP tool
            from distributed_server import execute_command

            # Create mock context
            mock_ctx = Mock()
            mock_ctx.user_id = "opencode_user"

            # Test execute_command tool
            result_json = await execute_command(
                ctx=mock_ctx,
                command="echo 'Hello OpenCode'",
                working_directory=workspace,
                timeout=30,
                env_vars={"OPENCODE_SESSION": "test"},
            )

            # Parse and validate response
            result = json.loads(result_json)

            # OpenCode expects these fields
            assert "success" in result
            assert "status" in result
            assert "stdout" in result
            assert "stderr" in result
            assert "exit_code" in result
            assert "execution_time" in result
            assert "working_directory" in result

            # Validate values
            assert result["success"] is True
            assert result["status"] == "success"
            assert result["stdout"] == "Hello OpenCode"
            assert result["exit_code"] == 0
            assert isinstance(result["execution_time"], (int, float))

            # Cleanup
            import shutil

            shutil.rmtree(workspace, ignore_errors=True)

    @pytest.mark.skip(reason="MCP tool signature mismatch - needs refactoring")
    @pytest.mark.asyncio
    async def test_process_management_mcp_tools(self):
        """Test process management MCP tools for OpenCode"""

        # This test needs to be refactored to match the actual MCP tool implementation
        # The distributed_server.list_processes function doesn't have a ctx parameter
        # and returns different data format than expected by this test
        pass

    @pytest.mark.asyncio
    async def test_error_responses_for_opencode(self):
        """Test that error responses are properly formatted for OpenCode"""

        with patch('distributed_server.execution_backend') as mock_backend:
            # Mock a failed execution
            mock_result = Mock()
            mock_result.success = False
            mock_result.status = ExecutionStatus.FAILED
            mock_result.stdout = ""
            mock_result.stderr = "Command not found"
            mock_result.exit_code = 127
            mock_result.execution_time = 0.1
            mock_result.working_directory = "/workspace"

            mock_backend.execute_with_timeout = AsyncMock(return_value=mock_result)

            from distributed_server import execute_command

            mock_ctx = Mock()

            result_json = await execute_command(ctx=mock_ctx, command="nonexistent_command", timeout=10)

            result = json.loads(result_json)

            # OpenCode should be able to handle failures gracefully
            assert result["success"] is False
            assert result["status"] == "failed"
            assert result["stderr"] == "Command not found"
            assert result["exit_code"] == 127


class TestOpenCodePerformance:
    """Test performance characteristics important to OpenCode"""

    @pytest.mark.asyncio
    async def test_rapid_command_execution(self):
        """Test rapid succession of commands as OpenCode might do"""
        import tempfile

        workspace = tempfile.mkdtemp(prefix="opencode_perf_test_")

        backend = create_execution_backend(
            "native", {"working_directory": workspace, "security": {"allowed_commands": ["echo", "date", "pwd"]}}
        )

        # Simulate OpenCode rapidly executing commands
        commands = ["echo 'test1'", "pwd", "date", "echo 'test2'", "echo 'test3'"]

        start_time = time.time()

        for cmd in commands:
            config = ExecutionConfig(command=cmd, timeout=5)
            result = await backend.execute_with_timeout(config)
            assert result.success, f"Command failed: {cmd}"

        total_time = time.time() - start_time
        avg_time = total_time / len(commands)

        # Should be responsive for OpenCode (< 200ms per command)
        assert avg_time < 0.2, f"Commands too slow: {avg_time:.3f}s average"

        await backend.cleanup()

        # Cleanup
        import shutil

        shutil.rmtree(workspace, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_concurrent_opencode_sessions(self):
        """Test multiple concurrent OpenCode sessions"""
        import tempfile

        workspace = tempfile.mkdtemp(prefix="opencode_concurrent_test_")

        backend = create_execution_backend("native", {"working_directory": workspace, "max_concurrent": 10})

        # Simulate multiple OpenCode sessions
        async def simulate_session(session_id):
            commands = [f"echo 'Session {session_id} - Command 1'", f"echo 'Session {session_id} - Command 2'", "pwd"]

            for cmd in commands:
                config = ExecutionConfig(command=cmd, timeout=10)
                result = await backend.execute_with_timeout(config)
                assert result.success
                assert f"Session {session_id}" in result.stdout or "pwd" in cmd

        # Run 5 concurrent sessions
        tasks = [simulate_session(i) for i in range(5)]
        await asyncio.gather(*tasks)

        await backend.cleanup()

        # Cleanup
        import shutil

        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
