# Execution Backends for MCP-Python
# Supports multiple deployment modes for host command execution

from .base import ExecutionBackend, ExecutionConfig, ExecutionResult
from .docker_socket import DockerSocketExecutionBackend
from .factory import create_execution_backend
from .host_agent import HostAgentExecutionBackend
from .native import NativeExecutionBackend
from .privileged import PrivilegedExecutionBackend
from .ssh_remote import SSHRemoteExecutionBackend

__all__ = [
    "ExecutionBackend",
    "ExecutionResult",
    "ExecutionConfig",
    "NativeExecutionBackend",
    "HostAgentExecutionBackend",
    "DockerSocketExecutionBackend",
    "SSHRemoteExecutionBackend",
    "PrivilegedExecutionBackend",
    "create_execution_backend",
]

# Supported execution modes
EXECUTION_MODES = {
    "native": NativeExecutionBackend,
    "host_agent": HostAgentExecutionBackend,
    "docker_socket": DockerSocketExecutionBackend,
    "ssh_remote": SSHRemoteExecutionBackend,
    "privileged": PrivilegedExecutionBackend,
}
