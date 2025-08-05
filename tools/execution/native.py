"""
Native execution backend - direct command execution on the host system
"""

import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from .base import ExecutionBackend, ExecutionConfig, ExecutionResult, ExecutionStatus


class NativeExecutionBackend(ExecutionBackend):
    """Execute commands directly on the host system"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.working_directory = config.get('working_directory', '/workspace')
        self.allowed_commands = config.get('allowed_commands', [])
        self.blocked_commands = config.get('blocked_commands', [])
        self.path_restrictions = config.get('path_restrictions', [])

    async def execute(self, exec_config: ExecutionConfig) -> ExecutionResult:
        """Execute command directly using subprocess"""
        start_time = time.time()

        # Validate command
        if not self.validate_command(exec_config.command, self.allowed_commands, self.blocked_commands):
            return ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                stderr="Command blocked by security policy",
                exit_code=1,
                command=exec_config.command,
                working_directory=exec_config.working_directory or "",
                execution_time=time.time() - start_time,
            )

        # Determine working directory
        work_dir = exec_config.working_directory or self.working_directory

        # Validate path restrictions
        if self.path_restrictions and not self._is_path_allowed(work_dir):
            return ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                stderr=f"Working directory not allowed: {work_dir}",
                exit_code=1,
                command=exec_config.command,
                working_directory=work_dir,
                execution_time=time.time() - start_time,
            )

        # Prepare environment
        env = os.environ.copy()
        if exec_config.environment:
            sanitized_env = self.sanitize_environment(exec_config.environment)
            env.update(sanitized_env)

        try:
            # Create the subprocess
            process = await asyncio.create_subprocess_shell(
                exec_config.command,
                cwd=work_dir,
                env=env,
                stdout=subprocess.PIPE if exec_config.capture_output else None,
                stderr=subprocess.PIPE if exec_config.capture_output else None,
                stdin=subprocess.PIPE if exec_config.input_data else None,
            )

            # Communicate with the process
            stdout, stderr = await process.communicate(
                input=exec_config.input_data.encode() if exec_config.input_data else None
            )

            execution_time = time.time() - start_time

            # Decode output
            stdout_str = stdout.decode('utf-8', errors='replace') if stdout else ""
            stderr_str = stderr.decode('utf-8', errors='replace') if stderr else ""

            # Determine status
            status = ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILED

            result = ExecutionResult(
                status=status,
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=process.returncode,
                command=exec_config.command,
                working_directory=work_dir,
                execution_time=execution_time,
                metadata={'backend': 'native', 'pid': process.pid},
            )

            # Audit log
            await self.audit_log(exec_config, result)

            return result

        except FileNotFoundError as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                stderr=f"Command not found: {str(e)}",
                exit_code=127,
                command=exec_config.command,
                working_directory=work_dir,
                execution_time=execution_time,
            )

        except PermissionError as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                stderr=f"Permission denied: {str(e)}",
                exit_code=126,
                command=exec_config.command,
                working_directory=work_dir,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Unexpected error executing command: {e}", exc_info=True)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                stderr=f"Execution error: {str(e)}",
                exit_code=1,
                command=exec_config.command,
                working_directory=work_dir,
                execution_time=execution_time,
            )

    async def health_check(self) -> bool:
        """Check if native execution is available"""
        try:
            # Test basic command execution
            result = await self.execute(ExecutionConfig(command="echo 'health_check'", timeout=5))
            return result.success and "health_check" in result.stdout
        except Exception as e:
            self.logger.error(f"Native execution health check failed: {e}")
            return False

    async def cleanup(self):
        """Clean up native execution resources"""
        # Cancel any active executions
        for execution_id in list(self._active_executions.keys()):
            await self.cancel_execution(execution_id)

        self.logger.info("Native execution backend cleaned up")

    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is within allowed restrictions"""
        if not self.path_restrictions:
            return True

        try:
            resolved_path = Path(path).resolve()
            for allowed_path in self.path_restrictions:
                allowed_resolved = Path(allowed_path).resolve()
                if resolved_path == allowed_resolved or allowed_resolved in resolved_path.parents:
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"Path validation error for {path}: {e}")
            return False

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for native execution"""
        import platform

        import psutil

        return {
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "disk_usage": psutil.disk_usage('/').total,
            "working_directory": self.working_directory,
            "path_restrictions": self.path_restrictions,
        }
