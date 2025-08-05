# Wand API Documentation

## Overview

**Wand** provides a comprehensive Model Context Protocol (MCP) implementation supporting both Claude Code and OpenCode integrations. The system offers HTTP and stdio transport mechanisms with a 69-tool orchestrator backend.

## Current System Status

- **Protocol Version**: MCP 2025-06-18 (backward compatible with 2024-11-05)
- **Transport Methods**: HTTP API and stdio
- **Active Configuration**: HTTP at `http://localhost:8001/mcp`
- **Available Tools**: 69 orchestrator tools via agent system

## HTTP API Endpoints

### Core MCP Endpoint: `/mcp`

#### POST `/mcp` - JSON-RPC Communication
Primary endpoint for client-server MCP communication.

**Required Headers:**
- `Content-Type: application/json`
- `MCP-Protocol-Version: 2025-06-18` (optional, defaults to 2024-11-05)
- `Mcp-Session-Id: <session-id>` (optional, auto-generated if not provided)

**Request Format:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "method_name",
  "params": {}
}
```

**Core Methods:**

1. **initialize** - Protocol handshake
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": {},
    "clientInfo": {"name": "client", "version": "1.0.0"}
  }
}
```

2. **tools/list** - List available tools
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

3. **tools/call** - Execute tool
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "execute_command",
    "arguments": {"command": "ls -la"}
  }
}
```

#### GET `/mcp` - Server-Sent Events
Opens SSE stream for server-to-client communication.

**Headers:**
- `Accept: text/event-stream`
- `MCP-Protocol-Version: 2025-06-18`
- `Last-Event-ID: <event-id>` (for resumption)

#### DELETE `/mcp` - Session Cleanup
Terminates MCP session and cleans up resources.

## Available Tools (69 Total)

### System Operations
- `execute_command` - Execute shell commands with security controls
- `get_system_info` - Comprehensive system information
- `get_system_status` - System health monitoring
- `check_command_exists` - Command availability verification

### File Operations
- `read_file` - Read file contents
- `write_file` - Write content to files
- `list_directory` - Directory listing with filtering
- `search_files` - Pattern-based file search

### Project Management
- `create_project` - Multi-component project creation
- `distribute_task` - Task distribution to agents
- `get_project_status` - Project status monitoring
- `create_task` - Task creation and management
- `list_agents` - Available agent listing

### External Integrations
- `slack_notify` - Slack communication
- `git_operation` - Version control operations
- `jenkins_trigger` - CI/CD pipeline management
- `youtrack_issue` - Issue tracking
- `postgres_query` - Database operations
- `aws_operation` - Cloud service management (EC2, S3, Lambda, RDS)
- `bambu_print` - 3D printing via Bambu Lab
- `web_search` - Web search functionality
- `api_request` - Generic HTTP API requests

## Tool Schemas

Each tool includes JSON Schema validation for parameters:

```json
{
  "name": "execute_command",
  "description": "Execute shell command with security controls",
  "inputSchema": {
    "type": "object",
    "properties": {
      "command": {
        "type": "string",
        "description": "Command to execute"
      }
    },
    "required": ["command"]
  }
}
```

## Error Handling

### JSON-RPC Errors
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "Method not found",
    "data": {"method": "invalid/method"}
  }
}
```

### HTTP Status Codes
- `200 OK` - Successful request
- `202 Accepted` - Notification accepted
- `400 Bad Request` - Invalid JSON-RPC
- `401 Unauthorized` - Authentication required
- `404 Not Found` - Method/resource not found
- `500 Internal Server Error` - Server error

## Authentication & Security

### Supported Methods
- OAuth 2.1 with Resource Indicators (RFC 8707)
- API Key authentication (`Authorization: Bearer <token>`)
- Environment variable-based credentials

### Security Features
- Origin validation for browser requests
- Session management with unique identifiers
- Command validation and sandboxing
- Resource limits and audit logging

## Client Integration

### Claude Code Integration
```bash
# Configure HTTP transport
claude mcp add wand-http --transport http http://localhost:8001/mcp

# Verify connection
claude mcp list
```

### Manual HTTP Testing
```bash
# Initialize connection
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {...}}'

# List available tools
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}'
```

## Architecture

### Multi-Agent Orchestrator
- **Agents**: 3 internal agents with distributed task management
- **Tools per Agent**: 23 tools (22 orchestrator + Claude API)
- **Execution Backends**: Native, Docker, SSH, Host Agent modes
- **Health Monitoring**: Automatic agent restart and failover

### Transport Layer
- **HTTP**: FastAPI-based with CORS support
- **stdio**: Direct process communication for local clients
- **SSE**: Server-Sent Events for real-time updates

### Configuration
Server configuration supports multiple execution modes:
- `native` - Direct host execution (development)
- `host_agent` - Separate privileged service (production)
- `docker_socket` - Container-based execution
- `ssh_remote` - Distributed execution
- `volume_mount` - Mounted host binaries
- `privileged` - Full system access (development only)

## Performance Characteristics

- **Response Time**: Sub-200ms average for tool execution
- **Concurrency**: 10+ simultaneous client sessions
- **Memory Usage**: <50MB per execution backend
- **Scalability**: Horizontal scaling support via agent distribution

## OpenAPI Specification

The complete OpenAPI 3.0.3 specification is available at `/openapi.yaml` with:
- Full endpoint documentation
- Request/response schemas
- Authentication flows
- Error specifications
- Example requests and responses

## Starting the Server

```bash
# HTTP mode (recommended for integrations)
python wand.py http

# stdio mode (for direct process communication)
python wand.py stdio
```

The HTTP server runs on `http://localhost:8001` by default with the MCP endpoint at `/mcp`.
