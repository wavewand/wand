"""
Data models for Host Agent HTTP API
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class ExecutionRequest(BaseModel):
    """Request model for command execution"""

    command: str = Field(..., description="Command to execute")
    working_directory: Optional[str] = Field(None, description="Working directory")
    timeout: int = Field(30, description="Timeout in seconds", ge=1, le=300)
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    input_data: Optional[str] = Field(None, description="Input data for command")
    capture_output: bool = Field(True, description="Whether to capture stdout/stderr")
    shell: bool = Field(False, description="Whether to run in shell")
    user_id: Optional[str] = Field(None, description="User ID for audit logging")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ExecutionResponse(BaseModel):
    """Response model for command execution"""

    status: ExecutionStatus = Field(..., description="Execution status")
    stdout: str = Field("", description="Standard output")
    stderr: str = Field("", description="Standard error")
    exit_code: int = Field(0, description="Process exit code")
    execution_time: float = Field(0.0, description="Execution time in seconds")
    working_directory: str = Field("", description="Working directory used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Uptime in seconds")
    active_executions: int = Field(0, description="Number of active executions")
    system_info: Dict[str, Any] = Field(default_factory=dict, description="System information")


class InfoResponse(BaseModel):
    """Service information response"""

    name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    host_info: Dict[str, Any] = Field(default_factory=dict, description="Host system information")
    capabilities: Dict[str, Any] = Field(default_factory=dict, description="Service capabilities")


class CapabilitiesResponse(BaseModel):
    """Capabilities response"""

    available: bool = Field(True, description="Whether service is available")
    execution_modes: list = Field(default_factory=list, description="Supported execution modes")
    security_features: Dict[str, bool] = Field(default_factory=dict, description="Security features")
    resource_limits: Dict[str, Any] = Field(default_factory=dict, description="Resource limits")
