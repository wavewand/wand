"""
Host Agent HTTP Server

FastAPI-based HTTP service for secure command execution on host system.
This service runs as a separate privileged service to handle command execution
for containerized MCP servers.
"""

import asyncio
import logging
import os
import platform
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .models import (
    CapabilitiesResponse,
    ExecutionRequest,
    ExecutionResponse,
    ExecutionStatus,
    HealthResponse,
    InfoResponse,
)

logger = logging.getLogger(__name__)


class HostAgentServer:
    """Host Agent HTTP Server for secure command execution"""

    def __init__(self, auth_token: str = None, port: int = 8001, host: str = "0.0.0.0"):
        self.auth_token = auth_token or os.getenv("HOST_AGENT_TOKEN", "default-token")
        self.port = port
        self.host = host
        self.start_time = time.time()
        self.active_executions = 0
        self.version = "1.0.0"

        # Security configuration
        self.allowed_commands = {
            "git",
            "npm",
            "yarn",
            "python",
            "python3",
            "pip",
            "pip3",
            "node",
            "docker",
            "ls",
            "cat",
            "grep",
            "find",
            "head",
            "tail",
            "echo",
            "which",
            "whoami",
            "pwd",
            "mkdir",
            "touch",
            "cp",
            "mv",
            "curl",
            "wget",
            "make",
            "cmake",
            "go",
            "cargo",
            "rustc",
        }

        self.blocked_commands = {
            "rm",
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
        }

        self.allowed_paths = {"/workspace", "/tmp", "/var/tmp"}
        self.max_execution_time = 300
        self.max_concurrent_executions = 10

        # Initialize FastAPI app
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        app = FastAPI(
            title="MCP Host Agent",
            description="Host Agent HTTP service for secure command execution",
            version=self.version,
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add routes
        self._add_routes(app)

        return app

    def _add_routes(self, app: FastAPI):
        """Add API routes to FastAPI app"""
        security = HTTPBearer()

        def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
            """Verify authentication token"""
            if credentials.credentials != self.auth_token:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")
            return credentials.credentials

        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy",
                version=self.version,
                uptime=time.time() - self.start_time,
                active_executions=self.active_executions,
                system_info={
                    "platform": platform.system(),
                    "platform_version": platform.version(),
                    "architecture": platform.machine(),
                    "hostname": platform.node(),
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": psutil.virtual_memory().total,
                    "memory_available": psutil.virtual_memory().available,
                    "disk_usage": psutil.disk_usage('/')._asdict(),
                },
            )

        @app.get("/info", response_model=InfoResponse)
        async def service_info():
            """Service information endpoint"""
            return InfoResponse(
                name="MCP Host Agent",
                version=self.version,
                host_info={
                    "platform": platform.system(),
                    "platform_version": platform.version(),
                    "architecture": platform.machine(),
                    "python_version": platform.python_version(),
                    "hostname": platform.node(),
                },
                capabilities={
                    "command_execution": True,
                    "file_operations": True,
                    "system_info": True,
                    "security_validation": True,
                    "audit_logging": True,
                },
            )

        @app.get("/capabilities", response_model=CapabilitiesResponse)
        async def capabilities():
            """Capabilities endpoint"""
            return CapabilitiesResponse(
                available=True,
                execution_modes=["subprocess", "shell"],
                security_features={
                    "command_validation": True,
                    "path_restrictions": True,
                    "resource_limits": True,
                    "audit_logging": True,
                    "user_isolation": False,  # Not implemented yet
                },
                resource_limits={
                    "max_execution_time": self.max_execution_time,
                    "max_concurrent_executions": self.max_concurrent_executions,
                    "allowed_paths": list(self.allowed_paths),
                    "allowed_commands": list(self.allowed_commands),
                    "blocked_commands": list(self.blocked_commands),
                },
            )

        @app.post("/execute", response_model=ExecutionResponse)
        async def execute_command(request: ExecutionRequest, token: str = Depends(verify_token)):
            """Execute command endpoint"""

            # Check concurrent execution limit
            if self.active_executions >= self.max_concurrent_executions:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Maximum concurrent executions reached"
                )

            # Validate command
            if not self._validate_command(request.command):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Command not allowed by security policy"
                )

            # Validate working directory
            if request.working_directory and not self._validate_path(request.working_directory):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Working directory not allowed by security policy"
                )

            # Execute command
            try:
                self.active_executions += 1
                start_time = time.time()

                result = await self._execute_command_async(request)

                execution_time = time.time() - start_time

                # Log execution for audit
                logger.info(
                    f"Command executed: {request.command[:100]}... "
                    f"Exit code: {result.exit_code} "
                    f"Execution time: {execution_time:.2f}s "
                    f"User: {request.user_id or 'unknown'}"
                )

                result.execution_time = execution_time
                return result

            except Exception as e:
                logger.error(f"Command execution failed: {str(e)}")
                return ExecutionResponse(
                    status=ExecutionStatus.FAILED, stderr=str(e), exit_code=-1, execution_time=time.time() - start_time
                )
            finally:
                self.active_executions -= 1

    def _validate_command(self, command: str) -> bool:
        """Validate command against security policy"""
        if not command or not command.strip():
            return False

        # Extract the base command (first word)
        base_command = command.strip().split()[0]

        # Remove path if present
        if '/' in base_command:
            base_command = os.path.basename(base_command)

        # Check against blocked commands
        if base_command in self.blocked_commands:
            return False

        # Check against allowed commands (if allowlist is non-empty)
        if self.allowed_commands and base_command not in self.allowed_commands:
            return False

        # Additional security checks
        dangerous_patterns = ['|', '&', ';', '`', '$', '>', '<', '||', '&&']
        if any(pattern in command for pattern in dangerous_patterns):
            return False

        return True

    def _validate_path(self, path: str) -> bool:
        """Validate path against security policy"""
        if not path:
            return True

        # Resolve absolute path
        try:
            abs_path = os.path.abspath(path)
        except Exception:
            return False

        # Check against allowed paths
        if self.allowed_paths:
            return any(abs_path.startswith(allowed) for allowed in self.allowed_paths)

        return True

    async def _execute_command_async(self, request: ExecutionRequest) -> ExecutionResponse:
        """Execute command asynchronously"""

        # Prepare environment
        env = os.environ.copy()
        if request.environment:
            # Sanitize environment variables
            for key, value in request.environment.items():
                if isinstance(key, str) and isinstance(value, str):
                    env[key] = value

        # Prepare working directory
        working_dir = request.working_directory or os.getcwd()
        if not os.path.exists(working_dir):
            return ExecutionResponse(
                status=ExecutionStatus.FAILED,
                stderr=f"Working directory does not exist: {working_dir}",
                exit_code=-1,
                working_directory=working_dir,
            )

        try:
            # Create subprocess
            process = await asyncio.create_subprocess_shell(
                request.command,
                cwd=working_dir,
                env=env,
                stdin=subprocess.PIPE if request.input_data else None,
                stdout=subprocess.PIPE if request.capture_output else None,
                stderr=subprocess.PIPE if request.capture_output else None,
                shell=request.shell,
            )

            # Execute with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=request.input_data.encode() if request.input_data else None),
                    timeout=request.timeout,
                )

                return ExecutionResponse(
                    status=ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILED,
                    stdout=stdout.decode() if stdout else "",
                    stderr=stderr.decode() if stderr else "",
                    exit_code=process.returncode,
                    working_directory=working_dir,
                )

            except asyncio.TimeoutError:
                # Kill the process if it times out
                try:
                    process.kill()
                    await process.wait()
                except BaseException:
                    pass

                return ExecutionResponse(
                    status=ExecutionStatus.TIMEOUT,
                    stderr=f"Command timed out after {request.timeout} seconds",
                    exit_code=-1,
                    working_directory=working_dir,
                )

        except Exception as e:
            return ExecutionResponse(
                status=ExecutionStatus.FAILED,
                stderr=f"Execution error: {str(e)}",
                exit_code=-1,
                working_directory=working_dir,
            )

    def run(self, **kwargs):
        """Run the server"""
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info", **kwargs)
        server = uvicorn.Server(config)

        logger.info(f"Starting Host Agent server on {self.host}:{self.port}")
        logger.info(f"Authentication token: {self.auth_token[:8]}...")

        asyncio.run(server.serve())


def create_server(auth_token: str = None, port: int = 8001, host: str = "0.0.0.0") -> HostAgentServer:
    """Create a new Host Agent server instance"""
    return HostAgentServer(auth_token=auth_token, port=port, host=host)


if __name__ == "__main__":
    # CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="MCP Host Agent Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--token", help="Authentication token")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run server
    server = create_server(auth_token=args.token, port=args.port, host=args.host)

    server.run()
