# MCP-Python Architecture

## System Overview

MCP-Python is a comprehensive Model Context Protocol implementation providing a bridge between AI development tools (Claude Code, OpenCode) and system execution capabilities through a multi-agent orchestrator architecture.

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Claude Code   │    │    OpenCode     │    │  Other Clients  │
│                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │       MCP Protocol         │
                    │    (JSON-RPC 2.0 over     │
                    │    HTTP/stdio/SSE)         │
                    └─────────────┬──────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │    Transport Layer         │
                    │  ┌─────────────────────┐   │
                    │  │  HTTP Transport     │   │
                    │  │  (FastAPI)          │   │
                    │  └─────────────────────┘   │
                    │  ┌─────────────────────┐   │
                    │  │  stdio Transport    │   │
                    │  │  (FastMCP)          │   │
                    │  └─────────────────────┘   │
                    │  ┌─────────────────────┐   │
                    │  │  SSE Transport      │   │
                    │  │  (Server-Sent Events)  │
                    │  └─────────────────────┘   │
                    └─────────────┬──────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │   Agent Orchestrator       │
                    │                            │
                    │  ┌─────────────────────┐   │
                    │  │    Agent Pool       │   │
                    │  │  ┌─────┐ ┌─────┐    │   │
                    │  │  │Agent│ │Agent│    │   │
                    │  │  │  1  │ │  2  │ ..│   │
                    │  │  │23tls│ │23tls│    │   │
                    │  │  └─────┘ └─────┘    │   │
                    │  └─────────────────────┘   │
                    │                            │
                    │  ┌─────────────────────┐   │
                    │  │   Task Manager      │   │
                    │  │   Load Balancing    │   │
                    │  │   Health Monitoring │   │
                    │  └─────────────────────┘   │
                    └─────────────┬──────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │    Execution Layer         │
                    │                            │
                    │  ┌─────────────────────┐   │
                    │  │  Native Backend     │   │
                    │  │  (Direct Execution) │   │
                    │  └─────────────────────┘   │
                    │  ┌─────────────────────┐   │
                    │  │  Docker Backend     │   │
                    │  │  (Container Exec)   │   │
                    │  └─────────────────────┘   │
                    │  ┌─────────────────────┐   │
                    │  │  SSH Backend        │   │
                    │  │  (Remote Execution) │   │
                    │  └─────────────────────┘   │
                    │  ┌─────────────────────┐   │
                    │  │  Host Agent Backend │   │
                    │  │  (Privileged Service)  │
                    │  └─────────────────────┘   │
                    └─────────────┬──────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │      System Layer          │
                    │                            │
                    │  File System • Network     │
                    │  Processes • Services      │
                    │  Databases • APIs          │
                    │  Cloud Resources           │
                    └────────────────────────────┘
```

## Core Components

### 1. Transport Layer

#### HTTP Transport (`transport/mcp_http.py`)
- **Framework**: FastAPI with CORS support
- **Protocol**: MCP 2025-06-18 via JSON-RPC 2.0
- **Endpoints**:
  - `POST /mcp` - Primary communication
  - `GET /mcp` - Server-Sent Events
  - `DELETE /mcp` - Session cleanup
- **Features**: Session management, protocol negotiation, error handling

#### stdio Transport
- **Framework**: FastMCP
- **Usage**: Direct process communication (Claude Code)
- **Benefits**: Low latency, simple integration

#### SSE Transport
- **Purpose**: Real-time server-to-client updates
- **Use Cases**: Progress notifications, status updates
- **Implementation**: HTTP streaming with event resumption

### 2. Agent Orchestrator (`orchestrator/agent_orchestrator.py`)

#### Multi-Agent System
- **Agent Count**: 3 internal agents (configurable)
- **Tools per Agent**: 23 tools (22 orchestrator + Claude API)
- **Distribution**: Round-robin task assignment with health monitoring
- **Failover**: Automatic agent restart on failure

#### Task Management
- **Queue**: Distributed task queue with UUID tracking
- **Load Balancing**: Agent selection based on availability
- **Monitoring**: Health checks every 10 seconds
- **Scaling**: Dynamic agent creation/destruction

### 3. Tool System

#### Available Tools (22 total)

**System Operations:**
- `execute_command` - Shell command execution
- `get_system_info` - System information gathering
- `get_system_status` - Health monitoring
- `check_command_exists` - Command availability

**File Operations:**
- `read_file`, `write_file` - File I/O
- `list_directory`, `search_files` - Directory operations

**Project Management:**
- `create_project`, `distribute_task` - Project lifecycle
- `get_project_status`, `list_agents` - Status monitoring

**External Integrations:**
- `slack_notify` - Communication
- `git_operation` - Version control
- `jenkins_trigger` - CI/CD
- `postgres_query` - Database
- `aws_operation` - Cloud services
- `web_search`, `api_request` - External APIs

#### Tool Architecture
```python
class ToolInterface:
    async def execute(self, parameters: dict) -> dict:
        """Execute tool with validated parameters"""
        pass

    def get_schema(self) -> dict:
        """Return JSON Schema for parameter validation"""
        pass
```

### 4. Execution Backends

#### Native Backend (`tools/execution/base.py`)
- **Direct Execution**: Commands run on host system
- **Security**: Command validation, path restrictions
- **Performance**: Fastest execution, lowest overhead
- **Use Case**: Development environments

#### Docker Backend
- **Container Execution**: Commands run in isolated containers
- **Security**: Full process isolation
- **Configuration**: Custom images, resource limits
- **Use Case**: Secure execution environments

#### SSH Backend
- **Remote Execution**: Commands run on remote systems
- **Distribution**: Load balancing across multiple hosts
- **Authentication**: SSH key-based authentication
- **Use Case**: Distributed computing, specialized environments

#### Host Agent Backend
- **Privileged Service**: Separate service with elevated permissions
- **Security**: Request validation, audit logging
- **Production**: Recommended for production deployments
- **Communication**: HTTP API between MCP server and host agent

### 5. Security Layer

#### Command Validation
```python
security_config = {
    "command_validation": True,
    "allowed_commands": ["git", "npm", "python", "ls"],
    "blocked_commands": ["rm", "dd", "sudo"],
    "path_restrictions": ["/workspace", "/tmp"],
    "resource_limits": {
        "max_memory": "2GB",
        "max_cpu": "2.0",
        "max_execution_time": 300
    }
}
```

#### Authentication & Authorization
- **OAuth 2.1**: Complete implementation with resource indicators
- **API Keys**: Bearer token authentication
- **Session Management**: Secure session handling with UUIDs
- **Origin Validation**: CORS protection for browser clients

#### Audit & Logging
- **Command Auditing**: All executions logged with context
- **Security Events**: Failed authentication, blocked commands
- **Performance Metrics**: Execution times, resource usage
- **Error Tracking**: Detailed error reporting and tracking

## Data Flow

### Request Processing
1. **Client Request** → Transport layer receives MCP message
2. **Protocol Validation** → JSON-RPC 2.0 format verification
3. **Authentication** → Session validation, permission checks
4. **Method Routing** → Route to appropriate handler (tools/list, tools/call)
5. **Agent Selection** → Orchestrator selects available agent
6. **Tool Execution** → Agent executes tool via backend
7. **Response Formatting** → Results formatted as MCP response
8. **Client Response** → JSON-RPC response sent to client

### Tool Execution Flow
```
Tool Call → Parameter Validation → Backend Selection →
Security Check → Command Execution → Result Processing →
Audit Logging → Response Return
```

## Configuration Management

### Environment-Based Configuration
```python
config = {
    "server": {
        "version": "2025-06-18",
        "mcp_http_port": 8001
    },
    "orchestrator": {
        "agent_count": 3,
        "max_concurrent_tasks": 10,
        "health_check_interval": 10
    },
    "execution": {
        "mode": "native",  # native|docker|ssh|host_agent
        "timeout": 300,
        "security": {...}
    }
}
```

### Runtime Configuration
- **Dynamic Agent Scaling**: Add/remove agents based on load
- **Backend Switching**: Change execution backend without restart
- **Security Updates**: Modify allowed commands, paths dynamically
- **Feature Flags**: Enable/disable specific tools or features

## Performance Characteristics

### Metrics
- **Response Time**: <200ms average for tool execution
- **Throughput**: 10+ concurrent client sessions
- **Memory Usage**: <50MB per execution backend
- **CPU Usage**: <10% during normal operation

### Optimization Strategies
- **Connection Pooling**: Reuse HTTP connections
- **Agent Caching**: Keep agents warm for faster response
- **Tool Result Caching**: Cache idempotent operations
- **Lazy Loading**: Load backends and tools on demand

## Scalability

### Horizontal Scaling
- **Multiple Instances**: Deploy behind load balancer
- **Shared State**: Redis for session management
- **Message Queue**: RabbitMQ for task distribution
- **Service Discovery**: Consul/etcd for instance coordination

### Vertical Scaling
- **More Agents**: Increase agent count per instance
- **Resource Limits**: Adjust memory/CPU per backend
- **Connection Limits**: Tune FastAPI worker processes
- **Database Connections**: Pool management for external services

## Monitoring & Observability

### Health Checks
- **Endpoint**: `GET /health` - Basic server health
- **Agent Health**: Individual agent status monitoring
- **Backend Health**: Execution backend connectivity
- **External Dependencies**: Database, API connectivity

### Metrics Collection
- **Prometheus**: Metrics export for monitoring
- **OpenTelemetry**: Distributed tracing support
- **Custom Metrics**: Tool execution counts, latencies
- **Resource Metrics**: Memory, CPU, disk usage

### Logging Strategy
- **Structured Logging**: JSON format for log aggregation
- **Log Levels**: DEBUG, INFO, WARN, ERROR, CRITICAL
- **Context Propagation**: Request IDs across components
- **Log Rotation**: Size-based and time-based rotation

## Extensibility

### Adding New Tools
```python
@tool_registry.register("new_tool")
class NewTool:
    def get_schema(self):
        return {"type": "object", "properties": {...}}

    async def execute(self, params):
        # Tool implementation
        return {"result": "success"}
```

### Custom Execution Backends
```python
class CustomBackend(ExecutionBackend):
    async def execute(self, command, context):
        # Custom execution logic
        return ExecutionResult(...)
```

### Transport Extensions
- **WebSocket**: Real-time bidirectional communication
- **gRPC**: High-performance binary protocol
- **GraphQL**: Query-based API interface
- **Custom Protocols**: Domain-specific communication
