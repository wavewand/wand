"""
SSH Remote Execution Backend

Executes commands on remote systems via SSH.
Suitable for high-security environments and remote execution scenarios.
"""

import asyncio
import io
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import paramiko
from paramiko import AutoAddPolicy, Ed25519Key, RSAKey, SSHClient

from .base import ExecutionBackend, ExecutionConfig, ExecutionResult, ExecutionStatus

logger = logging.getLogger(__name__)


class SSHRemoteExecutionBackend(ExecutionBackend):
    """Execute commands on remote systems via SSH"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 22)
        self.username = config.get("username", "mcp-agent")
        self.auth_method = config.get("auth_method", "key")  # 'key' or 'password'
        self.key_file = config.get("key_file")  # No default, will validate
        self.password = config.get("password")
        self.known_hosts_file = config.get("known_hosts_file", "/app/config/known_hosts")
        self.connect_timeout = config.get("timeout", 30)
        self.keepalive = config.get("keepalive", True)
        self.remote_working_dir = config.get("remote_working_dir", "/tmp/mcp-workspace")

        # SSH client connection pool
        self._ssh_clients = {}
        self._connection_lock = asyncio.Lock()

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate SSH configuration"""
        if self.auth_method == "key" and not self.key_file:
            raise ValueError("SSH key file path required for key authentication")

        if self.auth_method == "password" and not self.password:
            raise ValueError("Password required for password authentication")

        if not self.host:
            raise ValueError("SSH host is required")

    async def execute(self, exec_config: ExecutionConfig) -> ExecutionResult:
        """Execute command on remote system via SSH"""

        start_time = time.time()
        ssh_client = None

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

            # Get SSH connection
            ssh_client = await self._get_ssh_connection()

            # Prepare working directory
            working_dir = exec_config.working_directory or self.remote_working_dir
            await self._ensure_remote_directory(ssh_client, working_dir)

            # Prepare environment
            env_vars = self._prepare_environment(exec_config.environment)

            # Build command with environment and working directory
            full_command = self._build_remote_command(exec_config.command, working_dir, env_vars)

            logger.info(f"Executing SSH command: {full_command[:100]}...")

            # Execute command
            stdin, stdout, stderr = ssh_client.exec_command(full_command, timeout=exec_config.timeout, get_pty=False)

            # Send input data if provided
            if exec_config.input_data:
                stdin.write(exec_config.input_data)
                stdin.flush()
            stdin.close()

            # Wait for completion with timeout
            stdout.channel.settimeout(exec_config.timeout)
            stderr.channel.settimeout(exec_config.timeout)

            # Read output
            stdout_data = stdout.read().decode('utf-8', errors='replace')
            stderr_data = stderr.read().decode('utf-8', errors='replace')

            # Get exit code
            exit_code = stdout.channel.recv_exit_status()

            execution_time = time.time() - start_time

            logger.info(f"SSH execution completed. Exit code: {exit_code}, Time: {execution_time:.2f}s")

            status = ExecutionStatus.SUCCESS if exit_code == 0 else ExecutionStatus.FAILED

            return ExecutionResult(
                status=status,
                stdout=stdout_data,
                stderr=stderr_data,
                exit_code=exit_code,
                execution_time=execution_time,
                command=exec_config.command,
                working_directory=working_dir,
            )

        except asyncio.TimeoutError:
            logger.warning(f"SSH execution timed out after {exec_config.timeout}s")
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                stdout="",
                stderr=f"Command timed out after {exec_config.timeout} seconds",
                exit_code=124,
                execution_time=exec_config.timeout,
                command=exec_config.command,
                working_directory=exec_config.working_directory or "",
            )

        except Exception as e:
            logger.error(f"SSH execution failed: {e}")
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                stdout="",
                stderr=f"SSH execution error: {str(e)}",
                exit_code=-1,
                execution_time=time.time() - start_time,
                command=exec_config.command,
                working_directory=exec_config.working_directory or "",
            )

        finally:
            # Return SSH client to pool (don't close it)
            pass

    async def _get_ssh_connection(self) -> SSHClient:
        """Get or create SSH connection"""
        async with self._connection_lock:
            connection_key = f"{self.username}@{self.host}:{self.port}"

            # Reuse existing connection if available and active
            if connection_key in self._ssh_clients:
                client = self._ssh_clients[connection_key]
                try:
                    # Test connection with a simple command
                    transport = client.get_transport()
                    if transport and transport.is_active():
                        return client
                    else:
                        # Connection is dead, remove it
                        del self._ssh_clients[connection_key]
                except BaseException:
                    # Connection is bad, remove it
                    if connection_key in self._ssh_clients:
                        del self._ssh_clients[connection_key]

            # Create new connection
            client = SSHClient()

            # Configure known hosts
            if Path(self.known_hosts_file).exists():
                client.load_host_keys(self.known_hosts_file)
            else:
                logger.warning(f"Known hosts file not found: {self.known_hosts_file}")
                client.set_missing_host_key_policy(AutoAddPolicy())

            # Connect based on authentication method
            if self.auth_method == "key":
                await self._connect_with_key(client)
            else:
                await self._connect_with_password(client)

            # Store in connection pool
            self._ssh_clients[connection_key] = client

            logger.info(f"Established SSH connection to {connection_key}")

            return client

    async def _connect_with_key(self, client: SSHClient):
        """Connect using SSH key authentication"""
        loop = asyncio.get_event_loop()

        def _connect():
            # Load private key
            key_path = Path(self.key_file)
            if not key_path.exists():
                raise FileNotFoundError(f"SSH private key not found: {self.key_file}")

            # Try different key types
            private_key = None
            for key_class in [RSAKey, Ed25519Key]:
                try:
                    private_key = key_class.from_private_key_file(str(key_path))
                    break
                except Exception:
                    continue

            if not private_key:
                raise ValueError(f"Could not load SSH private key from {self.key_file}")

            # Connect
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                pkey=private_key,
                timeout=self.connect_timeout,
                allow_agent=False,
                look_for_keys=False,
            )

        await loop.run_in_executor(None, _connect)

    async def _connect_with_password(self, client: SSHClient):
        """Connect using password authentication"""
        loop = asyncio.get_event_loop()

        def _connect():
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                timeout=self.connect_timeout,
                allow_agent=False,
                look_for_keys=False,
            )

        await loop.run_in_executor(None, _connect)

    async def _ensure_remote_directory(self, client: SSHClient, directory: str):
        """Ensure remote working directory exists"""
        try:
            stdin, stdout, stderr = client.exec_command(f"mkdir -p {directory}")
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                error = stderr.read().decode('utf-8', errors='replace')
                logger.warning(f"Failed to create remote directory {directory}: {error}")
        except Exception as e:
            logger.warning(f"Could not ensure remote directory {directory}: {e}")

    def _prepare_environment(self, environment: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Prepare environment variables for remote execution"""
        env_vars = {}

        # Add provided environment variables
        if environment:
            for key, value in environment.items():
                if isinstance(key, str) and isinstance(value, str):
                    # Escape shell special characters
                    safe_value = value.replace("'", "'\"'\"'")
                    env_vars[key] = safe_value

        # Add default environment variables
        env_vars.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "DEBIAN_FRONTEND": "noninteractive"})

        return env_vars

    def _build_remote_command(self, command: str, working_dir: str, env_vars: Dict[str, str]) -> str:
        """Build complete remote command with environment and working directory"""

        # Build environment variable assignments
        env_assignments = []
        for key, value in env_vars.items():
            env_assignments.append(f"export {key}='{value}'")

        # Build complete command
        parts = []

        # Change to working directory
        parts.append(f"cd '{working_dir}'")

        # Set environment variables
        if env_assignments:
            parts.extend(env_assignments)

        # Execute the actual command
        parts.append(command)

        # Combine with && to ensure all steps succeed
        return " && ".join(parts)

    async def health_check(self) -> bool:
        """Check SSH remote backend health"""
        try:
            # Test SSH connection
            client = await self._get_ssh_connection()

            # Execute a simple test command
            stdin, stdout, stderr = client.exec_command("echo 'health_check'", timeout=10)
            output = stdout.read().decode('utf-8', errors='replace').strip()
            exit_code = stdout.channel.recv_exit_status()

            return exit_code == 0 and output == "health_check"

        except Exception as e:
            logger.error(f"SSH health check failed: {e}")
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Get SSH remote backend capabilities"""
        return {
            "execution_mode": "ssh_remote",
            "supports_timeout": True,
            "supports_environment": True,
            "supports_working_directory": True,
            "supports_input": True,
            "isolation_level": "remote_host",
            "authentication": {
                "method": self.auth_method,
                "host": self.host,
                "port": self.port,
                "username": self.username,
            },
            "features": [
                "remote_execution",
                "ssh_key_auth",
                "password_auth",
                "connection_pooling",
                "persistent_connections",
                "working_directory_management",
            ],
        }

    async def cleanup(self):
        """Cleanup SSH connections"""
        try:
            async with self._connection_lock:
                for connection_key, client in self._ssh_clients.items():
                    try:
                        client.close()
                        logger.info(f"Closed SSH connection to {connection_key}")
                    except Exception as e:
                        logger.warning(f"Error closing SSH connection {connection_key}: {e}")

                self._ssh_clients.clear()
                logger.info("All SSH connections closed")

        except Exception as e:
            logger.error(f"Error during SSH cleanup: {e}")

    async def test_connection(self) -> Dict[str, Any]:
        """Test SSH connection and return connection details"""
        try:
            client = await self._get_ssh_connection()

            # Get remote system information
            commands = {
                "hostname": "hostname",
                "uname": "uname -a",
                "whoami": "whoami",
                "pwd": "pwd",
                "date": "date",
                "uptime": "uptime",
            }

            results = {}
            for name, cmd in commands.items():
                try:
                    stdin, stdout, stderr = client.exec_command(cmd, timeout=5)
                    output = stdout.read().decode('utf-8', errors='replace').strip()
                    results[name] = output
                except Exception as e:
                    results[name] = f"Error: {str(e)}"

            return {
                "status": "connected",
                "host": self.host,
                "port": self.port,
                "username": self.username,
                "auth_method": self.auth_method,
                "system_info": results,
            }

        except Exception as e:
            return {
                "status": "failed",
                "host": self.host,
                "port": self.port,
                "username": self.username,
                "error": str(e),
            }


def create_ssh_remote_backend(config: Dict[str, Any]) -> SSHRemoteExecutionBackend:
    """Factory function to create SSH Remote execution backend"""
    return SSHRemoteExecutionBackend(config)
