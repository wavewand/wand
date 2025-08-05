"""
Base classes for execution backends in MCP-Python
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution status enumeration"""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


@dataclass
class ExecutionResult:
    """Result of command execution"""

    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time: float = 0.0
    command: str = ""
    working_directory: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def success(self) -> bool:
        """Check if execution was successful"""
        return self.status == ExecutionStatus.SUCCESS and self.exit_code == 0

    @property
    def output(self) -> str:
        """Get combined output"""
        if self.stdout and self.stderr:
            return f"{self.stdout}\n{self.stderr}"
        return self.stdout or self.stderr or ""


@dataclass
class ExecutionConfig:
    """Configuration for command execution"""

    command: str
    working_directory: Optional[str] = None
    timeout: int = 30
    environment: Optional[Dict[str, str]] = None
    input_data: Optional[str] = None
    capture_output: bool = True
    shell: bool = False
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.environment is None:
            self.environment = {}
        if self.metadata is None:
            self.metadata = {}


class ExecutionBackend(ABC):
    """Abstract base class for execution backends"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the execution backend with configuration"""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._active_executions: Dict[str, asyncio.Task] = {}

    @abstractmethod
    async def execute(self, exec_config: ExecutionConfig) -> ExecutionResult:
        """Execute a command and return the result"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the execution backend is healthy"""
        pass

    @abstractmethod
    async def cleanup(self):
        """Clean up resources"""
        pass

    async def execute_with_timeout(self, exec_config: ExecutionConfig) -> ExecutionResult:
        """Execute command with timeout handling"""
        execution_id = f"{time.time()}_{hash(exec_config.command)}"

        try:
            # Create execution task
            task = asyncio.create_task(self.execute(exec_config))
            self._active_executions[execution_id] = task

            # Wait with timeout
            result = await asyncio.wait_for(task, timeout=exec_config.timeout)
            return result

        except asyncio.TimeoutError:
            self.logger.warning(f"Command execution timed out after {exec_config.timeout}s: {exec_config.command}")

            # Cancel the task
            if execution_id in self._active_executions:
                self._active_executions[execution_id].cancel()
                del self._active_executions[execution_id]

            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                stderr=f"Command timed out after {exec_config.timeout} seconds",
                exit_code=124,  # Standard timeout exit code
                command=exec_config.command,
                working_directory=exec_config.working_directory or "",
                execution_time=exec_config.timeout,
            )

        except Exception as e:
            self.logger.error(f"Command execution failed: {e}", exc_info=True)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                stderr=str(e),
                exit_code=1,
                command=exec_config.command,
                working_directory=exec_config.working_directory or "",
                execution_time=0.0,
            )

        finally:
            # Clean up execution tracking
            if execution_id in self._active_executions:
                del self._active_executions[execution_id]

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        if execution_id in self._active_executions:
            task = self._active_executions[execution_id]
            task.cancel()
            del self._active_executions[execution_id]
            return True
        return False

    async def list_active_executions(self) -> List[str]:
        """List active execution IDs"""
        return list(self._active_executions.keys())

    def validate_command(self, command: str, allowed_commands: List[str], blocked_commands: List[str]) -> bool:
        """Validate if command is allowed to execute"""
        if not command.strip():
            return False

        # Extract the base command (first part)
        base_command = command.strip().split()[0]

        # Check blocked commands first
        for blocked in blocked_commands:
            if blocked in command:
                self.logger.warning(f"Command blocked by security policy: {command}")
                return False

        # Check allowed commands
        if allowed_commands:
            command_allowed = any(allowed in command for allowed in allowed_commands)
            if not command_allowed:
                self.logger.warning(f"Command not in allowed list: {command}")
                return False

        return True

    def sanitize_environment(self, env: Dict[str, str]) -> Dict[str, str]:
        """Sanitize environment variables"""
        sanitized = {}

        # Block potentially dangerous environment variables
        blocked_env_vars = [
            'LD_PRELOAD',
            'LD_LIBRARY_PATH',
            'DYLD_INSERT_LIBRARIES',
            'PYTHONPATH',
            'NODE_PATH',
            'GEM_PATH',
            'GOPATH',
        ]

        for key, value in env.items():
            if key not in blocked_env_vars:
                # Basic sanitization - remove control characters
                clean_value = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')
                sanitized[key] = clean_value
            else:
                self.logger.warning(f"Blocked environment variable: {key}")

        return sanitized

    async def audit_log(self, exec_config: ExecutionConfig, result: ExecutionResult):
        """Log execution for audit purposes"""
        audit_data = {
            "timestamp": time.time(),
            "backend": self.__class__.__name__,
            "command": exec_config.command,
            "working_directory": exec_config.working_directory,
            "user_id": exec_config.user_id,
            "status": result.status.value,
            "exit_code": result.exit_code,
            "execution_time": result.execution_time,
            "success": result.success,
        }

        self.logger.info("COMMAND_AUDIT", extra=audit_data)

        # Store in audit database if configured
        # This would integrate with the existing database layer
        # await self._store_audit_record(audit_data)

    def __str__(self):
        return f"{self.__class__.__name__}({self.config})"

    def __repr__(self):
        return self.__str__()
