# Wand Architecture

## System Overview

Wand is a Model Context Protocol (MCP) server that provides a unified interface for 55+ integrations, enabling AI agents to interact with various services and tools. Built on a comprehensive MCP implementation, Wand provides a bridge between AI development tools (Claude Code, OpenCode) and system execution capabilities through a multi-agent orchestrator architecture.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                CLIENT LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Claude Desktop    Claude Code    Other MCP Clients    Web Dashboard        │
│       ↓                ↓                ↓                    ↓              │
└───────┼────────────────┼────────────────┼────────────────────┼──────────────┘
        ↓                ↓                ↓                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRANSPORT LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│     stdio           stdio         HTTP/SSE            WebSocket              │
│   (process)       (process)        (REST)             (realtime)            │
└───────┼────────────────┼────────────────┼────────────────────┼──────────────┘
        ↓                ↓                ↓                    ↓
        └────────────────┴────────────────┴────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MCP SERVER CORE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────────┐     ┌────────────────┐        │
│   │  MCP Main    │────▶│   Orchestrator   │────▶│  Task Manager  │        │
│   │   Server     │     │     Manager      │     │                │        │
│   └──────────────┘     └──────────────────┘     └────────────────┘        │
│          │                      │                        │                 │
│          │                      ↓                        ↓                 │
│          │              ┌──────────────┐      ┌──────────────────┐        │
│          │              │Agent Manager │      │Execution Backends│        │
│          │              └──────────────┘      └──────────────────┘        │
└──────────┼──────────────────────────────────────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INTEGRATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐  │
│  │      AI/ML          │  │      DevOps         │  │  Communication   │  │
│  ├─────────────────────┤  ├─────────────────────┤  ├──────────────────┤  │
│  │ • Ollama            │  │ • Docker            │  │ • Slack          │  │
│  │ • OpenAI           │  │ • Kubernetes        │  │ • Telegram       │  │
│  │ • Anthropic        │  │ • Git/GitHub        │  │ • Email          │  │
│  │ • Cohere           │  │ • Jenkins           │  │ • Discord        │  │
│  │ • HuggingFace      │  │ • Terraform         │  │ • Teams          │  │
│  └─────────────────────┘  └─────────────────────┘  └──────────────────┘  │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐  │
│  │   Cloud Storage     │  │     Business        │  │    Security      │  │
│  ├─────────────────────┤  ├─────────────────────┤  ├──────────────────┤  │
│  │ • AWS S3            │  │ • CRM Systems       │  │ • Auth providers │  │
│  │ • Google Drive      │  │ • Payment APIs      │  │ • Vault          │  │
│  │ • Dropbox           │  │ • Project Mgmt      │  │ • Security scan  │  │
│  │ • OneDrive          │  │ • HR Tools          │  │ • ServiceNow     │  │
│  │                     │  │                     │  │ • SailPoint      │  │
│  │                     │  │                     │  │ • Entra          │  │
│  │                     │  │                     │  │ • Britive        │  │
│  └─────────────────────┘  └─────────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Native     │  │Docker Socket │  │ SSH Remote   │  │ Host Agent   │  │
│  │  Execution   │  │  Container   │  │   Server     │  │  Privileged  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  PostgreSQL  │  │  File System │  │Vector Store  │  │    Cache     │  │
│  │   Database   │  │   Storage    │  │ (Embeddings) │  │   (Redis)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### Client Layer
- **Claude Desktop/Code**: Primary MCP clients using stdio transport
- **MCP Clients**: Third-party clients using HTTP/WebSocket

### Transport Layer

#### 1. HTTP Transport (`transport/mcp_http.py`)
- **Framework**: FastAPI with CORS support
- **Protocol**: MCP 2025-06-18 via JSON-RPC 2.0
- **Endpoints**:
  - `POST /mcp` - Primary communication
  - `GET /mcp` - Server-Sent Events
  - `DELETE /mcp` - Session cleanup
- **Features**: Session management, protocol negotiation, error handling

#### 2. stdio Transport
- **Framework**: FastMCP
- **Usage**: Direct process communication (Claude Code)
- **Benefits**: Low latency, simple integration

#### 3. SSE Transport
- **Purpose**: Real-time server-to-client updates
- **Use Cases**: Progress notifications, status updates
- **Implementation**: HTTP streaming with event resumption

#### 4. WebSocket Transport
- **Purpose**: Real-time bidirectional communication
- **Use Cases**: Interactive sessions, live updates

### Wand Server Core

#### Wand Server (`wand.py`)
- Handles protocol communication
- Routes requests to appropriate handlers
- Manages sessions and authentication

#### Agent Orchestrator (`orchestrator/agent_orchestrator.py`)

##### Multi-Agent System
- **Agent Count**: 3 internal agents (configurable)
- **Tools per Agent**: 23 tools (22 orchestrator + Claude API)
- **Distribution**: Round-robin task assignment with health monitoring
- **Failover**: Automatic agent restart on failure

##### Task Management
- **Queue**: Distributed task queue with UUID tracking
- **Load Balancing**: Agent selection based on availability
- **Monitoring**: Health checks every 10 seconds
- **Scaling**: Dynamic agent creation/destruction

#### Task Manager
- Tracks task lifecycle
- Manages execution backends
- Handles retries and timeouts

#### Agent Manager
- Registers and monitors agents
- Capability-based routing
- Health checking

### Integration Layer

55+ integrations organized by category:

#### AI/ML Integrations
- **Ollama** - Local language model management
- **OpenAI** - GPT models and API integration
- **Anthropic** - Claude model integration
- **Cohere** - Language model services
- **HuggingFace** - Model hub and transformers
- **Replicate** - Cloud AI model hosting

#### DevOps Integrations
- **Docker** - Container management and orchestration
- **Kubernetes** - Container orchestration platform
- **Git/GitHub** - Version control and collaboration
- **Jenkins** - CI/CD automation
- **Terraform** - Infrastructure as code

#### Communication Integrations
- **Slack** - Team messaging and notifications
- **Discord** - Community and bot integration
- **Telegram** - Bot and messaging automation
- **Microsoft Teams** - Enterprise communication and webhooks
- **Email** - SMTP/IMAP email management

#### Enterprise Identity & Security
- **ServiceNow** - IT Service Management and ITSM workflows
- **SailPoint** - Identity Security Cloud and governance
- **Microsoft Entra** - Azure AD identity management
- **Britive** - Privileged access management (PAM)
- **Vault** - Secret management and security

#### Cloud Storage
- **AWS S3** - Object storage and file management
- **Google Drive** - Cloud file storage and sharing
- **Dropbox** - File synchronization and sharing
- **OneDrive** - Microsoft cloud storage

### Tool System

#### Available Tools (22 core tools)

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

### Execution Backends

#### 1. Native Backend (`tools/execution/base.py`)
- **Direct Execution**: Commands run on host system
- **Security**: Command validation, path restrictions
- **Performance**: Fastest execution, lowest overhead
- **Use Case**: Development environments

#### 2. Docker Backend
- **Container Execution**: Commands run in isolated containers
- **Security**: Full process isolation
- **Configuration**: Custom images, resource limits
- **Use Case**: Secure execution environments

#### 3. SSH Backend
- **Remote Execution**: Commands run on remote systems
- **Distribution**: Load balancing across multiple hosts
- **Authentication**: SSH key-based authentication
- **Use Case**: Distributed computing, specialized environments

#### 4. Host Agent Backend
- **Privileged Service**: Separate service with elevated permissions
- **Security**: Request validation, audit logging
- **Production**: Recommended for production deployments
- **Communication**: HTTP API between MCP server and host agent

### Security Layer

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

### Enterprise Integration Flow
1. **Authentication** → Enterprise credentials validation
2. **Operation Routing** → Route to specific enterprise service
3. **API Call** → Execute operation against enterprise system
4. **Response Processing** → Format and validate response
5. **Audit Logging** → Log enterprise operations for compliance

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
    },
    "enterprise": {
        "servicenow": {
            "instance_url": "https://company.service-now.com",
            "username": "api_user",
            "password": "secure_password"
        },
        "sailpoint": {
            "base_url": "https://company.api.identitynow.com",
            "client_id": "client_id",
            "client_secret": "client_secret"
        }
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

## Security Features

- **JWT-based authentication** for secure API access
- **Rate limiting per integration** to prevent abuse
- **Command validation and sanitization** for safe execution
- **Comprehensive audit logging** for compliance
- **Sandboxed execution environments** for isolation
- **Enterprise SSO integration** via OAuth 2.1

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

### Enterprise Integration Extensions
```python
class CustomEnterpriseIntegration(BaseIntegration):
    REQUIRED_CONFIG_KEYS = ["api_url", "token"]

    async def initialize(self):
        # Initialize enterprise connection
        pass

    async def _execute_operation_impl(self, operation: str, **kwargs):
        # Execute enterprise-specific operations
        return {"result": "success"}
```

## Production Deployment

### Container Support
- **Docker**: Full containerization support
- **Kubernetes**: Orchestration and scaling
- **Health Checks**: Readiness and liveness probes
- **Resource Management**: CPU and memory limits

### Load Balancing
- **Multiple Instances**: Deploy behind load balancer
- **Session Affinity**: Sticky sessions when needed
- **Health Monitoring**: Automatic failover
- **Geographic Distribution**: Multi-region deployment

### Enterprise Integration
- **Single Sign-On**: SAML/OAuth integration
- **Network Security**: VPN and firewall support
- **Compliance**: Audit trails and data governance
- **High Availability**: Redundancy and disaster recovery
