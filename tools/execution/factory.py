"""
Factory for creating execution backends based on configuration
"""

import logging
from typing import Any, Dict

from .base import ExecutionBackend
from .docker_socket import DockerSocketExecutionBackend
from .host_agent import HostAgentExecutionBackend
from .native import NativeExecutionBackend
from .privileged import PrivilegedExecutionBackend
from .ssh_remote import SSHRemoteExecutionBackend

logger = logging.getLogger(__name__)

# Registry of available execution backends
EXECUTION_BACKENDS = {
    'native': NativeExecutionBackend,
    'host_agent': HostAgentExecutionBackend,
    'docker_socket': DockerSocketExecutionBackend,
    'ssh_remote': SSHRemoteExecutionBackend,
    'privileged': PrivilegedExecutionBackend,
    # Additional backends will be added here:
    # 'volume_mount': VolumeMountExecutionBackend,
}


def create_execution_backend(mode: str, config: Dict[str, Any]) -> ExecutionBackend:
    """
    Create an execution backend based on the specified mode and configuration

    Args:
        mode: Execution mode (native, host_agent, etc.)
        config: Configuration dictionary

    Returns:
        ExecutionBackend instance

    Raises:
        ValueError: If the mode is not supported
    """
    if mode not in EXECUTION_BACKENDS:
        available_modes = ', '.join(EXECUTION_BACKENDS.keys())
        raise ValueError(f"Unknown execution mode '{mode}'. Available modes: {available_modes}")

    backend_class = EXECUTION_BACKENDS[mode]

    try:
        backend = backend_class(config)
        logger.info(f"Created execution backend: {mode}")
        return backend

    except Exception as e:
        logger.error(f"Failed to create execution backend '{mode}': {e}")
        raise


def list_available_backends() -> Dict[str, str]:
    """
    List all available execution backends

    Returns:
        Dictionary mapping backend names to their descriptions
    """
    descriptions = {
        'native': 'Direct command execution on host system',
        'host_agent': 'Execute commands via separate host agent service',
        'docker_socket': 'Execute commands in Docker containers',
        'ssh_remote': 'Execute commands via SSH connection',
        'privileged': 'Privileged container execution (DANGEROUS)',
        'volume_mount': 'Execute with mounted host binaries (coming soon)',
    }

    available = {}
    for backend_name in EXECUTION_BACKENDS.keys():
        available[backend_name] = descriptions.get(backend_name, 'No description available')

    return available


def validate_backend_config(mode: str, config: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate configuration for a specific backend mode

    Args:
        mode: Execution mode
        config: Configuration to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if mode not in EXECUTION_BACKENDS:
        errors.append(f"Unknown execution mode: {mode}")
        return False, errors

    # Mode-specific validation
    if mode == 'host_agent':
        if not config.get('url'):
            errors.append("Host agent URL is required")
        if not config.get('auth_token'):
            errors.append("Host agent auth token is required")

    elif mode == 'ssh_remote':
        if not config.get('host'):
            errors.append("SSH host is required")
        if not config.get('username'):
            errors.append("SSH username is required")
        if config.get('auth_method') == 'key' and not config.get('key_file'):
            errors.append("SSH key file is required when using key authentication")

    elif mode == 'docker_socket':
        if not config.get('socket_path'):
            errors.append("Docker socket path is required")

    # Common validation
    if config.get('timeout', 0) <= 0:
        errors.append("Timeout must be greater than 0")

    if config.get('max_concurrent', 0) <= 0:
        errors.append("Max concurrent executions must be greater than 0")

    return len(errors) == 0, errors
