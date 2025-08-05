"""
Docker Socket Execution Backend

Executes commands in Docker containers using Docker Socket API.
Suitable for CI/CD environments and containerized execution.
"""

import asyncio
import json
import logging
import tempfile
import time
from typing import Any, Dict, List, Optional

import docker
from docker.models.containers import Container

from .base import ExecutionBackend, ExecutionConfig, ExecutionResult, ExecutionStatus

logger = logging.getLogger(__name__)


class DockerSocketExecutionBackend(ExecutionBackend):
    """Execute commands in Docker containers via Docker socket"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.docker_client = None
        self.default_image = config.get("default_image", "ubuntu:22.04")
        self.network = config.get("network", "bridge")
        self.auto_remove = config.get("auto_remove", True)
        self.memory_limit = config.get("memory_limit", "1GB")
        self.cpu_limit = config.get("cpu_limit", "1.0")
        self.disk_limit = config.get("disk_limit", "2GB")
        self.container_working_dir = config.get("working_dir", "/workspace")

        # Volume mounts for workspace access
        self.volume_mounts = config.get("volume_mounts", {})

        self._initialize_client()

    def _initialize_client(self):
        """Initialize Docker client"""
        try:
            socket_path = self.config.get("socket_path", "/var/run/docker.sock")
            self.docker_client = docker.DockerClient(base_url=f"unix://{socket_path}")

            # Test connection
            self.docker_client.ping()
            logger.info(f"Connected to Docker socket at {socket_path}")

        except Exception as e:
            logger.error(f"Failed to connect to Docker: {e}")
            raise

    async def execute(self, exec_config: ExecutionConfig) -> ExecutionResult:
        """Execute command in Docker container"""

        start_time = time.time()
        container = None

        try:
            # Validate command
            if not self.validate_command(exec_config.command, [], []):
                return ExecutionResult(
                    status=ExecutionStatus.BLOCKED,
                    stdout="",
                    stderr="Command blocked by security policy",
                    exit_code=-1,
                    execution_time=0,
                    command=exec_config.command,
                    working_directory=exec_config.working_directory or "",
                )

            # Prepare environment
            container_env = self._prepare_environment(exec_config.environment)

            # Prepare volumes
            volumes = self._prepare_volumes(exec_config.working_directory)

            # Create container configuration
            container_config = {
                "image": self.default_image,
                "command": ["sh", "-c", exec_config.command],
                "working_dir": self.container_working_dir,
                "environment": container_env,
                "volumes": volumes,
                "network": self.network,
                "auto_remove": self.auto_remove,
                "mem_limit": self.memory_limit,
                "cpu_period": 100000,  # 100ms
                "cpu_quota": int(float(self.cpu_limit) * 100000),  # CPU limit
                "detach": False,
                "stdout": True,
                "stderr": True,
                "stdin": bool(exec_config.input_data),
            }

            logger.info(f"Creating container for command: {exec_config.command[:50]}...")

            # Create and start container
            container = await self._run_container_async(container_config, exec_config.input_data, exec_config.timeout)

            # Get results
            exit_code = container.attrs["State"]["ExitCode"]

            # Get logs
            logs = container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace')
            error_logs = container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace')

            execution_time = time.time() - start_time

            logger.info(f"Container execution completed. Exit code: {exit_code}, Time: {execution_time:.2f}s")

            status = ExecutionStatus.SUCCESS if exit_code == 0 else ExecutionStatus.FAILED

            return ExecutionResult(
                status=status,
                stdout=logs,
                stderr=error_logs,
                exit_code=exit_code,
                execution_time=execution_time,
                command=exec_config.command,
                working_directory=exec_config.working_directory or "",
            )

        except asyncio.TimeoutError:
            logger.warning(f"Container execution timed out after {exec_config.timeout}s")
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                stdout="",
                stderr=f"Command timed out after {exec_config.timeout} seconds",
                exit_code=-1,
                execution_time=exec_config.timeout,
                command=exec_config.command,
                working_directory=exec_config.working_directory or "",
            )

        except Exception as e:
            logger.error(f"Container execution failed: {e}")
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                exit_code=-1,
                execution_time=time.time() - start_time,
                command=exec_config.command,
                working_directory=exec_config.working_directory or "",
            )

        finally:
            # Cleanup container if not auto-removed
            if container and not self.auto_remove:
                try:
                    container.remove(force=True)
                except Exception as e:
                    logger.warning(f"Failed to remove container: {e}")

    async def _run_container_async(self, config: Dict[str, Any], input_data: Optional[str], timeout: int) -> Container:
        """Run container asynchronously with timeout"""

        loop = asyncio.get_event_loop()

        def _run_container():
            """Run container in thread pool"""
            try:
                # Create container
                container = self.docker_client.containers.create(**config)

                # Start container
                container.start()

                # Send input data if provided
                if input_data:
                    try:
                        # For interactive input, we need to attach to stdin
                        socket = container.attach_socket(params={'stdin': 1, 'stream': 1})
                        socket._sock.send(input_data.encode())
                        socket._sock.close()
                    except Exception as e:
                        logger.warning(f"Failed to send input data: {e}")

                # Wait for completion
                result = container.wait(timeout=timeout)

                return container

            except Exception as e:
                # Clean up container on error
                try:
                    if 'container' in locals():
                        container.remove(force=True)
                except BaseException:
                    pass
                raise e

        # Run in thread pool with timeout
        container = await asyncio.wait_for(loop.run_in_executor(None, _run_container), timeout=timeout)

        return container

    def _prepare_environment(self, environment: Optional[Dict[str, str]]) -> List[str]:
        """Prepare environment variables for container"""
        env_list = []

        # Add provided environment variables
        if environment:
            for key, value in environment.items():
                if isinstance(key, str) and isinstance(value, str):
                    env_list.append(f"{key}={value}")

        # Add default environment variables
        env_list.extend(["DEBIAN_FRONTEND=noninteractive", "LANG=C.UTF-8", "LC_ALL=C.UTF-8"])

        return env_list

    def _prepare_volumes(self, working_directory: Optional[str]) -> Dict[str, Dict[str, str]]:
        """Prepare volume mounts for container"""
        volumes = {}

        # Add configured volume mounts
        for host_path, mount_config in self.volume_mounts.items():
            container_path = mount_config.get("container_path", host_path)
            mode = mount_config.get("mode", "rw")
            volumes[host_path] = {"bind": container_path, "mode": mode}

        # Add working directory mount if specified
        if working_directory:
            volumes[working_directory] = {"bind": self.container_working_dir, "mode": "rw"}

        return volumes

    async def health_check(self) -> bool:
        """Check Docker socket backend health"""
        try:
            if not self.docker_client:
                return False

            # Test Docker connection
            self.docker_client.ping()
            return True

        except Exception as e:
            logger.error(f"Docker health check failed: {e}")
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Get Docker socket backend capabilities"""
        return {
            "execution_mode": "docker_socket",
            "supports_timeout": True,
            "supports_environment": True,
            "supports_working_directory": True,
            "supports_input": True,
            "isolation_level": "container",
            "resource_limits": {"memory": self.memory_limit, "cpu": self.cpu_limit, "disk": self.disk_limit},
            "features": [
                "container_isolation",
                "resource_limits",
                "volume_mounts",
                "network_isolation",
                "auto_cleanup",
            ],
        }

    async def cleanup(self):
        """Cleanup Docker resources"""
        try:
            if self.docker_client:
                # Clean up any dangling containers
                containers = self.docker_client.containers.list(all=True, filters={"label": "mcp-python-execution"})

                for container in containers:
                    try:
                        container.remove(force=True)
                        logger.info(f"Cleaned up container {container.id[:12]}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup container {container.id[:12]}: {e}")

                self.docker_client.close()
                logger.info("Docker client closed")

        except Exception as e:
            logger.error(f"Error during Docker cleanup: {e}")


def create_docker_socket_backend(config: Dict[str, Any]) -> DockerSocketExecutionBackend:
    """Factory function to create Docker Socket execution backend"""
    return DockerSocketExecutionBackend(config)
