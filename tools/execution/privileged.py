"""
Privileged Execution Backend

Executes commands with full system privileges in containerized environments.
WARNING: This is dangerous and should only be used in trusted environments.
"""

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ExecutionBackend, ExecutionConfig, ExecutionResult, ExecutionStatus

logger = logging.getLogger(__name__)


class PrivilegedExecutionBackend(ExecutionBackend):
    """Execute commands with full system privileges"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host_root = config.get("host_root", "/host")
        self.namespace_isolation = config.get("namespace_isolation", False)
        self.capabilities = config.get("capabilities", ["ALL"])
        self.allow_dangerous_commands = config.get("allow_dangerous_commands", True)
        self.mount_host_filesystem = config.get("mount_host_filesystem", True)

        logger.warning("⚠️  PRIVILEGED EXECUTION BACKEND INITIALIZED - USE WITH EXTREME CAUTION")

        # Initialize host filesystem access if enabled
        if self.mount_host_filesystem and os.path.exists(self.host_root):
            logger.info(f"Host filesystem accessible at: {self.host_root}")

    async def execute(self, exec_config: ExecutionConfig) -> ExecutionResult:
        """Execute command with full privileges"""

        start_time = time.time()

        try:
            logger.info(f"Executing privileged command: {exec_config.command[:100]}...")

            # Prepare environment
            env = os.environ.copy()
            if exec_config.environment:
                env.update(exec_config.environment)

            # Set working directory (use host paths if available)
            working_dir = exec_config.working_directory
            if working_dir and self.mount_host_filesystem:
                # Check if we should use host filesystem
                if not working_dir.startswith('/host') and os.path.exists(f"{self.host_root}{working_dir}"):
                    working_dir = f"{self.host_root}{working_dir}"

            if not working_dir or not os.path.exists(working_dir):
                working_dir = "/workspace"

            # Create subprocess with full privileges
            process = await asyncio.create_subprocess_shell(
                exec_config.command,
                cwd=working_dir,
                env=env,
                stdin=subprocess.PIPE if exec_config.input_data else None,
                stdout=subprocess.PIPE if exec_config.capture_output else None,
                stderr=subprocess.PIPE if exec_config.capture_output else None,
                shell=True,
            )

            # Execute with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=exec_config.input_data.encode() if exec_config.input_data else None),
                    timeout=exec_config.timeout,
                )

                execution_time = time.time() - start_time

                status = ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILED

                logger.info(
                    f"Privileged command completed. Exit code: {process.returncode}, Time: {execution_time:.2f}s"
                )

                return ExecutionResult(
                    status=status,
                    stdout=stdout.decode('utf-8', errors='replace') if stdout else "",
                    stderr=stderr.decode('utf-8', errors='replace') if stderr else "",
                    exit_code=process.returncode or 0,
                    execution_time=execution_time,
                    command=exec_config.command,
                    working_directory=working_dir,
                )

            except asyncio.TimeoutError:
                # Kill the process if it times out
                try:
                    process.kill()
                    await process.wait()
                except BaseException:
                    pass

                logger.warning(f"Privileged command timed out after {exec_config.timeout}s")

                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    stdout="",
                    stderr=f"Command timed out after {exec_config.timeout} seconds",
                    exit_code=124,
                    execution_time=exec_config.timeout,
                    command=exec_config.command,
                    working_directory=working_dir,
                )

        except Exception as e:
            logger.error(f"Privileged execution failed: {e}")
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                stdout="",
                stderr=f"Privileged execution error: {str(e)}",
                exit_code=-1,
                execution_time=time.time() - start_time,
                command=exec_config.command,
                working_directory=exec_config.working_directory or "/workspace",
            )

    async def health_check(self) -> bool:
        """Check privileged backend health"""
        try:
            # Test basic command execution
            test_config = ExecutionConfig(command="echo 'privileged_health_check'", timeout=5)

            result = await self.execute(test_config)
            return result.success and "privileged_health_check" in result.stdout

        except Exception as e:
            logger.error(f"Privileged health check failed: {e}")
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Get privileged backend capabilities"""
        return {
            "execution_mode": "privileged",
            "supports_timeout": True,
            "supports_environment": True,
            "supports_working_directory": True,
            "supports_input": True,
            "isolation_level": "none",
            "danger_level": "maximum",
            "features": [
                "full_system_access",
                "host_filesystem_access",
                "all_capabilities",
                "dangerous_commands_allowed",
                "no_restrictions",
            ],
            "warnings": [
                "⚠️  EXTREMELY DANGEROUS - Full system access",
                "⚠️  No security restrictions applied",
                "⚠️  Can execute any command with root privileges",
                "⚠️  Can access host filesystem",
                "⚠️  Use only in trusted environments",
            ],
        }

    async def cleanup(self):
        """Cleanup privileged backend resources"""
        try:
            logger.info("Privileged backend cleanup completed")
        except Exception as e:
            logger.error(f"Error during privileged cleanup: {e}")


def create_privileged_backend(config: Dict[str, Any]) -> PrivilegedExecutionBackend:
    """Factory function to create Privileged execution backend"""
    logger.warning("⚠️  CREATING PRIVILEGED EXECUTION BACKEND - EXTREMELY DANGEROUS")
    return PrivilegedExecutionBackend(config)
