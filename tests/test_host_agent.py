"""
Tests for Host Agent HTTP Server
"""

import asyncio
import json
import subprocess
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from tools.host_agent.models import (
    CapabilitiesResponse,
    ExecutionRequest,
    ExecutionResponse,
    ExecutionStatus,
    HealthResponse,
    InfoResponse,
)
from tools.host_agent.server import HostAgentServer, create_server


class TestHostAgentServer:
    """Test Host Agent Server functionality"""

    @pytest.fixture
    def server(self):
        """Create test server instance"""
        return HostAgentServer(auth_token="test-token-123", port=8001)

    @pytest.fixture
    def client(self, server):
        """Create test client"""
        return TestClient(server.app)

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers for test requests"""
        return {"Authorization": "Bearer test-token-123"}

    def test_server_creation(self):
        """Test server creation with default and custom parameters"""
        # Default creation
        server1 = HostAgentServer()
        assert server1.auth_token == "default-token"
        assert server1.port == 8001
        assert server1.host == "0.0.0.0"

        # Custom creation
        server2 = HostAgentServer(auth_token="custom-token", port=9001, host="127.0.0.1")
        assert server2.auth_token == "custom-token"
        assert server2.port == 9001
        assert server2.host == "127.0.0.1"

    def test_factory_function(self):
        """Test server factory function"""
        server = create_server(auth_token="factory-token", port=9002)
        assert server.auth_token == "factory-token"
        assert server.port == 9002

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime" in data
        assert "system_info" in data
        assert data["active_executions"] == 0

        # Validate health response model
        health_response = HealthResponse(**data)
        assert health_response.status == "healthy"

    def test_info_endpoint(self, client):
        """Test service info endpoint"""
        response = client.get("/info")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "MCP Host Agent"
        assert "version" in data
        assert "host_info" in data
        assert "capabilities" in data

        # Validate info response model
        info_response = InfoResponse(**data)
        assert info_response.name == "MCP Host Agent"

    def test_capabilities_endpoint(self, client):
        """Test capabilities endpoint"""
        response = client.get("/capabilities")
        assert response.status_code == 200

        data = response.json()
        assert data["available"] is True
        assert "execution_modes" in data
        assert "security_features" in data
        assert "resource_limits" in data

        # Validate capabilities response model
        caps_response = CapabilitiesResponse(**data)
        assert caps_response.available is True

    def test_execute_without_auth(self, client):
        """Test execute endpoint without authentication"""
        request_data = {"command": "echo test", "timeout": 10}

        response = client.post("/execute", json=request_data)
        assert response.status_code == 403  # Forbidden without auth

    def test_execute_with_invalid_auth(self, client):
        """Test execute endpoint with invalid authentication"""
        request_data = {"command": "echo test", "timeout": 10}

        headers = {"Authorization": "Bearer invalid-token"}
        response = client.post("/execute", json=request_data, headers=headers)
        assert response.status_code == 401  # Unauthorized

    def test_command_validation_allowed(self, server):
        """Test command validation for allowed commands"""
        # Test allowed commands
        assert server._validate_command("git status") is True
        assert server._validate_command("python script.py") is True
        assert server._validate_command("ls -la") is True
        assert server._validate_command("echo hello") is True

    def test_command_validation_blocked(self, server):
        """Test command validation for blocked commands"""
        # Test blocked commands
        assert server._validate_command("rm -rf /") is False
        assert server._validate_command("sudo systemctl stop nginx") is False
        assert server._validate_command("dd if=/dev/zero") is False
        assert server._validate_command("mount /dev/sda1") is False

    def test_command_validation_dangerous_patterns(self, server):
        """Test command validation for dangerous patterns"""
        # Test dangerous shell patterns
        assert server._validate_command("echo hello | rm -rf /") is False
        assert server._validate_command("ls && rm file") is False
        assert server._validate_command("echo `rm file`") is False
        assert server._validate_command("echo $(rm file)") is False
        assert server._validate_command("echo > /etc/passwd") is False

    def test_path_validation_allowed(self, server):
        """Test path validation for allowed paths"""
        # Test allowed paths
        assert server._validate_path("/workspace") is True
        assert server._validate_path("/workspace/project") is True
        assert server._validate_path("/tmp/temp") is True
        assert server._validate_path("/var/tmp/cache") is True

    def test_path_validation_blocked(self, server):
        """Test path validation for blocked paths"""
        # Test blocked paths (outside allowed paths)
        assert server._validate_path("/etc") is False
        assert server._validate_path("/root") is False
        assert server._validate_path("/usr/bin") is False
        assert server._validate_path("/home/user") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
