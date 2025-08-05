# OpenCode Agent Requirements Document

## Executive Summary

This document outlines the comprehensive requirements for developing OpenCode agents that integrate with the Model Context Protocol (MCP) system. OpenCode agents are AI-powered assistants that can interact with various tools, services, and frameworks through the MCP infrastructure.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Requirements](#core-requirements)
4. [Technical Specifications](#technical-specifications)
5. [Integration Requirements](#integration-requirements)
6. [Security Requirements](#security-requirements)
7. [Performance Requirements](#performance-requirements)
8. [API Specifications](#api-specifications)
9. [Development Guidelines](#development-guidelines)
10. [Testing Requirements](#testing-requirements)
11. [Deployment Requirements](#deployment-requirements)
12. [Monitoring and Observability](#monitoring-and-observability)
13. [Examples and Use Cases](#examples-and-use-cases)

## 1. Overview

### 1.1 Purpose
OpenCode agents serve as intelligent interfaces between users and the MCP ecosystem, providing:
- Natural language understanding and processing
- Tool execution and orchestration
- Context management and state persistence
- Multi-framework integration capabilities

### 1.2 Scope
This document covers:
- Agent architecture and design patterns
- Integration with MCP servers and tools
- Communication protocols and data formats
- Security and authentication mechanisms
- Performance and scalability considerations

## 2. System Architecture

### 2.1 High-Level Architecture
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   OpenCode UI   │────▶│  OpenCode Agent  │────▶│   MCP Server    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────┐           ┌──────────────┐
                        │ LLM Provider │           │ Tool Servers │
                        └──────────────┘           └──────────────┘
```

### 2.2 Component Interactions
- **OpenCode UI**: User interface for interaction
- **OpenCode Agent**: Core processing unit
- **MCP Server**: Protocol handler and router
- **LLM Provider**: Language model integration
- **Tool Servers**: Specialized capability providers

## 3. Core Requirements

### 3.1 Functional Requirements

#### 3.1.1 Language Processing
- Support for natural language input/output
- Context-aware conversation management
- Multi-turn dialogue handling
- Intent recognition and routing

#### 3.1.2 Tool Integration
- Dynamic tool discovery and registration
- Tool capability negotiation
- Parameter validation and type checking
- Error handling and recovery

#### 3.1.3 State Management
- Conversation history persistence
- Context window management
- Session state handling
- User preference storage

### 3.2 Non-Functional Requirements

#### 3.2.1 Scalability
- Support for concurrent user sessions
- Horizontal scaling capabilities
- Load balancing support
- Resource optimization

#### 3.2.2 Reliability
- 99.9% uptime SLA
- Graceful degradation
- Automatic recovery mechanisms
- Data consistency guarantees

## 4. Technical Specifications

### 4.1 Communication Protocols

#### 4.1.1 MCP Protocol
```python
class MCPMessage:
    """Base MCP message structure"""

    def __init__(self):
        self.jsonrpc = "2.0"
        self.id = None
        self.method = None
        self.params = {}

class MCPRequest(MCPMessage):
    """MCP request message"""

    def __init__(self, method: str, params: dict):
        super().__init__()
        self.method = method
        self.params = params
        self.id = generate_id()

class MCPResponse(MCPMessage):
    """MCP response message"""

    def __init__(self, result: Any, error: Optional[dict] = None):
        super().__init__()
        self.result = result
        self.error = error
```

#### 4.1.2 WebSocket Support
- Real-time bidirectional communication
- Connection pooling and management
- Automatic reconnection logic
- Message queuing and buffering

### 4.2 Data Formats

#### 4.2.1 Agent Configuration
```json
{
  "agent": {
    "id": "opencode-agent-001",
    "name": "OpenCode Assistant",
    "version": "1.0.0",
    "capabilities": {
      "tools": ["file_access", "code_execution", "web_search"],
      "languages": ["python", "javascript", "typescript"],
      "frameworks": ["langchain", "llamaindex", "haystack"]
    },
    "model": {
      "provider": "anthropic",
      "name": "claude-3-opus",
      "temperature": 0.7,
      "max_tokens": 4096
    },
    "mcp": {
      "servers": [
        {
          "name": "filesystem",
          "url": "ws://localhost:3000/filesystem"
        },
        {
          "name": "database",
          "url": "ws://localhost:3001/database"
        }
      ]
    }
  }
}
```

#### 4.2.2 Tool Schema
```json
{
  "tool": {
    "name": "read_file",
    "description": "Read contents of a file",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "description": "File path to read"
        },
        "encoding": {
          "type": "string",
          "enum": ["utf-8", "ascii", "latin-1"],
          "default": "utf-8"
        }
      },
      "required": ["path"]
    }
  }
}
```

## 5. Integration Requirements

### 5.1 MCP Server Integration

#### 5.1.1 Server Discovery
```python
class MCPServerDiscovery:
    """Discover and connect to MCP servers"""

    async def discover_servers(self) -> List[MCPServer]:
        """Discover available MCP servers"""
        servers = []

        # Local discovery
        local_servers = await self._discover_local()
        servers.extend(local_servers)

        # Network discovery
        network_servers = await self._discover_network()
        servers.extend(network_servers)

        # Registry discovery
        registry_servers = await self._discover_registry()
        servers.extend(registry_servers)

        return servers
```

#### 5.1.2 Capability Negotiation
```python
class CapabilityNegotiator:
    """Negotiate capabilities with MCP servers"""

    async def negotiate(self, server: MCPServer) -> ServerCapabilities:
        """Negotiate server capabilities"""

        # Send initialize request
        response = await server.request("initialize", {
            "clientInfo": {
                "name": "opencode-agent",
                "version": "1.0.0"
            },
            "capabilities": {
                "tools": True,
                "prompts": True,
                "resources": True
            }
        })

        return ServerCapabilities(response.result)
```

### 5.2 LLM Provider Integration

#### 5.2.1 Provider Abstraction
```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion for prompt"""
        pass

    @abstractmethod
    async def stream_complete(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream completion for prompt"""
        pass

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation"""

    async def complete(self, prompt: str, **kwargs) -> str:
        # Implementation
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider implementation"""

    async def complete(self, prompt: str, **kwargs) -> str:
        # Implementation
        pass
```

### 5.3 Tool Server Integration

#### 5.3.1 Tool Registration
```python
class ToolRegistry:
    """Registry for available tools"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.servers: Dict[str, MCPServer] = {}

    async def register_server_tools(self, server: MCPServer):
        """Register all tools from an MCP server"""

        # Get available tools
        response = await server.request("tools/list", {})

        for tool_info in response.result["tools"]:
            tool = Tool(
                name=tool_info["name"],
                description=tool_info["description"],
                parameters=tool_info["inputSchema"],
                server=server
            )
            self.tools[tool.name] = tool
```

## 6. Security Requirements

### 6.1 Authentication

#### 6.1.1 User Authentication
- OAuth 2.0 support
- API key authentication
- JWT token validation
- Session management

#### 6.1.2 Server Authentication
```python
class MCPAuthenticator:
    """Handle MCP server authentication"""

    async def authenticate(self, server: MCPServer, credentials: dict):
        """Authenticate with MCP server"""

        if credentials.get("type") == "bearer":
            server.headers["Authorization"] = f"Bearer {credentials['token']}"
        elif credentials.get("type") == "api_key":
            server.headers["X-API-Key"] = credentials["key"]
        elif credentials.get("type") == "oauth2":
            token = await self._get_oauth_token(credentials)
            server.headers["Authorization"] = f"Bearer {token}"
```

### 6.2 Authorization

#### 6.2.1 Permission Model
```python
class Permission:
    """Permission definition"""

    def __init__(self, resource: str, action: str):
        self.resource = resource
        self.action = action

class Role:
    """Role with permissions"""

    def __init__(self, name: str, permissions: List[Permission]):
        self.name = name
        self.permissions = permissions

class AuthorizationManager:
    """Manage user authorization"""

    def check_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission"""

        for role in user.roles:
            for permission in role.permissions:
                if (permission.resource == resource and
                    permission.action == action):
                    return True
        return False
```

### 6.3 Data Security

#### 6.3.1 Encryption
- TLS 1.3 for transport security
- End-to-end encryption for sensitive data
- Key rotation and management
- Secure credential storage

#### 6.3.2 Data Privacy
- PII detection and masking
- Data retention policies
- User consent management
- Audit logging

## 7. Performance Requirements

### 7.1 Response Time
- Average response time < 200ms
- 95th percentile < 500ms
- 99th percentile < 1000ms
- Streaming response initiation < 100ms

### 7.2 Throughput
- Support 1000+ concurrent users
- Handle 10,000+ requests/minute
- Process 1GB+ context windows
- Stream at 50+ tokens/second

### 7.3 Resource Utilization
```python
class ResourceMonitor:
    """Monitor and manage resource usage"""

    def __init__(self):
        self.cpu_limit = 80  # percentage
        self.memory_limit = 4096  # MB
        self.connection_limit = 1000

    async def check_resources(self) -> ResourceStatus:
        """Check current resource usage"""

        return ResourceStatus(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            active_connections=len(self.connections),
            can_accept_request=self._can_accept_request()
        )
```

## 8. API Specifications

### 8.1 REST API Endpoints

#### 8.1.1 Agent Management
```yaml
openapi: 3.0.0
paths:
  /agents:
    post:
      summary: Create new agent
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/AgentConfig'
      responses:
        201:
          description: Agent created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Agent'

  /agents/{agentId}:
    get:
      summary: Get agent details
      parameters:
        - name: agentId
          in: path
          required: true
          schema:
            type: string
      responses:
        200:
          description: Agent details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Agent'
```

#### 8.1.2 Conversation Management
```yaml
  /conversations:
    post:
      summary: Start new conversation
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                agentId:
                  type: string
                message:
                  type: string
      responses:
        201:
          description: Conversation started
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Conversation'

  /conversations/{conversationId}/messages:
    post:
      summary: Send message
      parameters:
        - name: conversationId
          in: path
          required: true
          schema:
            type: string
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
      responses:
        200:
          description: Message response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Message'
```

### 8.2 WebSocket API

#### 8.2.1 Connection Protocol
```typescript
interface WebSocketMessage {
  type: 'request' | 'response' | 'event';
  id?: string;
  method?: string;
  params?: any;
  result?: any;
  error?: {
    code: number;
    message: string;
    data?: any;
  };
}

// Connection establishment
ws.on('open', () => {
  ws.send(JSON.stringify({
    type: 'request',
    id: '1',
    method: 'initialize',
    params: {
      clientInfo: {
        name: 'opencode-agent',
        version: '1.0.0'
      }
    }
  }));
});
```

## 9. Development Guidelines

### 9.1 Code Structure

#### 9.1.1 Project Layout
```
opencode-agent/
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── conversation.py
│   │   └── state.py
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── server.py
│   │   └── protocol.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   └── executor.py
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── anthropic.py
│   │   └── openai.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── metrics.py
├── tests/
├── docs/
├── examples/
└── requirements.txt
```

#### 9.1.2 Coding Standards
```python
# Example: Agent implementation following standards

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for OpenCode agent"""

    id: str
    name: str
    model_provider: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 4096

class OpenCodeAgent:
    """Main agent implementation"""

    def __init__(self, config: AgentConfig):
        """Initialize agent with configuration

        Args:
            config: Agent configuration
        """
        self.config = config
        self.conversation_manager = ConversationManager()
        self.tool_registry = ToolRegistry()
        self.mcp_client = MCPClient()

    async def initialize(self) -> None:
        """Initialize agent components"""

        logger.info(f"Initializing agent {self.config.id}")

        # Initialize MCP connections
        await self.mcp_client.connect()

        # Discover and register tools
        await self._discover_tools()

        # Initialize LLM provider
        await self._initialize_llm()

    async def process_message(
        self,
        message: str,
        conversation_id: Optional[str] = None
    ) -> str:
        """Process user message and return response

        Args:
            message: User message
            conversation_id: Optional conversation ID

        Returns:
            Agent response
        """

        # Implementation follows...
```

### 9.2 Best Practices

#### 9.2.1 Error Handling
```python
class AgentError(Exception):
    """Base exception for agent errors"""
    pass

class ToolExecutionError(AgentError):
    """Error during tool execution"""
    pass

class MCPConnectionError(AgentError):
    """Error with MCP connection"""
    pass

async def execute_tool_safe(tool: Tool, params: dict) -> Any:
    """Execute tool with proper error handling"""

    try:
        result = await tool.execute(params)
        return result
    except ValidationError as e:
        logger.error(f"Invalid parameters for tool {tool.name}: {e}")
        raise ToolExecutionError(f"Invalid parameters: {e}")
    except TimeoutError as e:
        logger.error(f"Tool {tool.name} timed out: {e}")
        raise ToolExecutionError(f"Tool execution timed out")
    except Exception as e:
        logger.error(f"Unexpected error in tool {tool.name}: {e}")
        raise ToolExecutionError(f"Tool execution failed: {e}")
```

#### 9.2.2 Logging and Monitoring
```python
import structlog
from prometheus_client import Counter, Histogram, Gauge

# Structured logging
logger = structlog.get_logger()

# Metrics
message_counter = Counter(
    'agent_messages_total',
    'Total messages processed',
    ['agent_id', 'status']
)

response_time = Histogram(
    'agent_response_duration_seconds',
    'Response time in seconds',
    ['agent_id', 'tool']
)

active_conversations = Gauge(
    'agent_active_conversations',
    'Number of active conversations',
    ['agent_id']
)

class MetricsMiddleware:
    """Middleware for collecting metrics"""

    async def process_message(self, message: str, agent_id: str):
        """Process message with metrics"""

        with response_time.labels(
            agent_id=agent_id,
            tool='process_message'
        ).time():
            try:
                result = await self._process(message)
                message_counter.labels(
                    agent_id=agent_id,
                    status='success'
                ).inc()
                return result
            except Exception as e:
                message_counter.labels(
                    agent_id=agent_id,
                    status='error'
                ).inc()
                raise
```

## 10. Testing Requirements

### 10.1 Unit Testing

#### 10.1.1 Test Coverage
- Minimum 80% code coverage
- 100% coverage for critical paths
- Edge case testing
- Error condition testing

#### 10.1.2 Example Tests
```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from opencode_agent import OpenCodeAgent, AgentConfig

class TestOpenCodeAgent:
    """Test OpenCode agent functionality"""

    @pytest.fixture
    async def agent(self):
        """Create test agent"""
        config = AgentConfig(
            id="test-agent",
            name="Test Agent",
            model_provider="mock",
            model_name="mock-model"
        )
        agent = OpenCodeAgent(config)

        # Mock dependencies
        agent.mcp_client = AsyncMock()
        agent.tool_registry = MagicMock()

        return agent

    @pytest.mark.asyncio
    async def test_initialize(self, agent):
        """Test agent initialization"""

        # Setup mocks
        agent.mcp_client.connect.return_value = None

        # Initialize
        await agent.initialize()

        # Verify
        agent.mcp_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message(self, agent):
        """Test message processing"""

        # Setup
        message = "What files are in the current directory?"
        expected_response = "Here are the files..."

        agent._generate_response = AsyncMock(
            return_value=expected_response
        )

        # Execute
        response = await agent.process_message(message)

        # Verify
        assert response == expected_response
```

### 10.2 Integration Testing

#### 10.2.1 MCP Server Integration
```python
class TestMCPIntegration:
    """Test MCP server integration"""

    @pytest.fixture
    async def mcp_server(self):
        """Start test MCP server"""
        server = TestMCPServer()
        await server.start()
        yield server
        await server.stop()

    @pytest.mark.asyncio
    async def test_tool_discovery(self, agent, mcp_server):
        """Test tool discovery from MCP server"""

        # Connect to test server
        await agent.mcp_client.connect(mcp_server.url)

        # Discover tools
        tools = await agent.discover_tools()

        # Verify
        assert len(tools) > 0
        assert "read_file" in [t.name for t in tools]
```

### 10.3 End-to-End Testing

#### 10.3.1 Conversation Flow Testing
```python
class TestE2EConversation:
    """End-to-end conversation testing"""

    @pytest.mark.asyncio
    async def test_full_conversation(self, test_client):
        """Test complete conversation flow"""

        # Start conversation
        response = await test_client.post("/conversations", json={
            "agentId": "test-agent",
            "message": "Hello, can you help me?"
        })

        assert response.status_code == 201
        conversation_id = response.json()["id"]

        # Send follow-up message
        response = await test_client.post(
            f"/conversations/{conversation_id}/messages",
            json={"message": "List files in /tmp"}
        )

        assert response.status_code == 200
        assert "files" in response.json()["content"].lower()
```

## 11. Deployment Requirements

### 11.1 Container Deployment

#### 11.1.1 Dockerfile
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app

# Switch to non-root user
USER agent

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start agent
CMD ["python", "-m", "opencode_agent.server"]
```

#### 11.1.2 Docker Compose
```yaml
version: '3.8'

services:
  agent:
    build: .
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - AGENT_ID=opencode-agent-001
      - LOG_LEVEL=INFO
      - MCP_SERVERS=filesystem:ws://mcp-filesystem:3000,database:ws://mcp-database:3001
    depends_on:
      - redis
      - postgres
    volumes:
      - ./config:/app/config
      - agent-data:/app/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=opencode
      - POSTGRES_USER=agent
      - POSTGRES_PASSWORD=secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

  mcp-filesystem:
    image: mcp/filesystem-server:latest
    ports:
      - "3000:3000"
    volumes:
      - /workspace:/workspace:ro

volumes:
  agent-data:
  redis-data:
  postgres-data:
```

### 11.2 Kubernetes Deployment

#### 11.2.1 Deployment Manifest
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opencode-agent
  namespace: opencode
spec:
  replicas: 3
  selector:
    matchLabels:
      app: opencode-agent
  template:
    metadata:
      labels:
        app: opencode-agent
    spec:
      containers:
      - name: agent
        image: opencode/agent:1.0.0
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8001
          name: grpc
        env:
        - name: AGENT_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /app/config
      volumes:
      - name: config
        configMap:
          name: opencode-agent-config
```

## 12. Monitoring and Observability

### 12.1 Logging

#### 12.1.1 Log Format
```json
{
  "timestamp": "2024-08-04T10:30:45.123Z",
  "level": "INFO",
  "agent_id": "opencode-agent-001",
  "conversation_id": "conv-123",
  "message_id": "msg-456",
  "user_id": "user-789",
  "event": "message_processed",
  "duration_ms": 245,
  "tool_calls": ["read_file", "search_web"],
  "tokens_used": {
    "prompt": 1234,
    "completion": 567
  },
  "metadata": {
    "model": "claude-3-opus",
    "temperature": 0.7
  }
}
```

#### 12.1.2 Log Aggregation
```python
class LogAggregator:
    """Aggregate and forward logs"""

    def __init__(self, config: dict):
        self.elasticsearch = Elasticsearch(config['elasticsearch_url'])
        self.index_pattern = "opencode-agent-{date}"

    async def log_event(self, event: dict):
        """Log event to Elasticsearch"""

        index = self.index_pattern.format(
            date=datetime.now().strftime("%Y.%m.%d")
        )

        await self.elasticsearch.index(
            index=index,
            body=event
        )
```

### 12.2 Metrics

#### 12.2.1 Key Metrics
```python
# Prometheus metrics configuration

from prometheus_client import Counter, Histogram, Gauge, Summary

# Request metrics
request_count = Counter(
    'opencode_agent_requests_total',
    'Total number of requests',
    ['method', 'status']
)

request_duration = Histogram(
    'opencode_agent_request_duration_seconds',
    'Request duration in seconds',
    ['method'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Agent metrics
active_agents = Gauge(
    'opencode_agent_active_count',
    'Number of active agents'
)

conversation_duration = Summary(
    'opencode_agent_conversation_duration_seconds',
    'Conversation duration in seconds'
)

# Tool metrics
tool_usage = Counter(
    'opencode_agent_tool_usage_total',
    'Tool usage count',
    ['tool_name', 'status']
)

tool_duration = Histogram(
    'opencode_agent_tool_duration_seconds',
    'Tool execution duration',
    ['tool_name']
)

# Model metrics
tokens_used = Counter(
    'opencode_agent_tokens_total',
    'Total tokens used',
    ['model', 'type']
)

model_latency = Histogram(
    'opencode_agent_model_latency_seconds',
    'Model inference latency',
    ['model', 'provider']
)
```

### 12.3 Tracing

#### 12.3.1 Distributed Tracing
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

class TracedAgent:
    """Agent with distributed tracing"""

    async def process_message(self, message: str, conversation_id: str):
        """Process message with tracing"""

        with tracer.start_as_current_span(
            "process_message",
            attributes={
                "conversation_id": conversation_id,
                "message_length": len(message)
            }
        ) as span:
            try:
                # Process message
                result = await self._process(message)
                span.set_attribute("success", True)
                return result
            except Exception as e:
                span.set_attribute("success", False)
                span.set_attribute("error", str(e))
                raise
```

## 13. Examples and Use Cases

### 13.1 Basic Agent Implementation

```python
"""
Basic OpenCode agent implementation example
"""

import asyncio
from typing import Optional
from opencode_agent import OpenCodeAgent, AgentConfig
from opencode_agent.mcp import MCPClient
from opencode_agent.tools import ToolRegistry

async def main():
    # Configure agent
    config = AgentConfig(
        id="example-agent",
        name="Example OpenCode Agent",
        model_provider="anthropic",
        model_name="claude-3-opus",
        temperature=0.7
    )

    # Create agent
    agent = OpenCodeAgent(config)

    # Initialize
    await agent.initialize()

    # Start conversation
    conversation_id = await agent.start_conversation()

    # Process messages
    response = await agent.process_message(
        "What files are in the /tmp directory?",
        conversation_id
    )

    print(f"Agent response: {response}")

    # Continue conversation
    response = await agent.process_message(
        "Can you read the first file you found?",
        conversation_id
    )

    print(f"Agent response: {response}")

    # Cleanup
    await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### 13.2 Custom Tool Implementation

```python
"""
Custom tool implementation for OpenCode agent
"""

from opencode_agent.tools import Tool, ToolSchema
from typing import Dict, Any
import aiohttp

class WebSearchTool(Tool):
    """Web search tool implementation"""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information",
            schema=ToolSchema({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            })
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search"""

        query = params["query"]
        num_results = params.get("num_results", 5)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.search.example.com/search",
                params={"q": query, "limit": num_results}
            ) as response:
                results = await response.json()

        return {
            "results": results["items"],
            "total": results["total"]
        }

# Register custom tool
async def register_custom_tools(agent: OpenCodeAgent):
    """Register custom tools with agent"""

    web_search = WebSearchTool()
    agent.tool_registry.register(web_search)
```

### 13.3 Multi-Agent Orchestration

```python
"""
Multi-agent orchestration example
"""

from opencode_agent.orchestrator import AgentOrchestrator
from opencode_agent import OpenCodeAgent, AgentConfig

class ResearchOrchestrator(AgentOrchestrator):
    """Orchestrate multiple agents for research tasks"""

    def __init__(self):
        super().__init__()

        # Create specialized agents
        self.research_agent = self._create_agent(
            "research-agent",
            "Research Specialist",
            specialized_tools=["web_search", "arxiv_search"]
        )

        self.analysis_agent = self._create_agent(
            "analysis-agent",
            "Data Analyst",
            specialized_tools=["data_analysis", "visualization"]
        )

        self.writer_agent = self._create_agent(
            "writer-agent",
            "Technical Writer",
            specialized_tools=["document_generation", "formatting"]
        )

    async def research_topic(self, topic: str) -> str:
        """Research a topic using multiple agents"""

        # Step 1: Research
        research_data = await self.research_agent.process_message(
            f"Research the topic: {topic}. Gather relevant papers, articles, and data."
        )

        # Step 2: Analysis
        analysis = await self.analysis_agent.process_message(
            f"Analyze the following research data and provide insights: {research_data}"
        )

        # Step 3: Report Generation
        report = await self.writer_agent.process_message(
            f"Write a comprehensive report based on this analysis: {analysis}"
        )

        return report

    def _create_agent(
        self,
        agent_id: str,
        name: str,
        specialized_tools: list
    ) -> OpenCodeAgent:
        """Create specialized agent"""

        config = AgentConfig(
            id=agent_id,
            name=name,
            model_provider="anthropic",
            model_name="claude-3-opus",
            specialized_tools=specialized_tools
        )

        return OpenCodeAgent(config)
```

### 13.4 Error Handling and Recovery

```python
"""
Comprehensive error handling example
"""

from opencode_agent import OpenCodeAgent
from opencode_agent.errors import (
    ToolExecutionError,
    MCPConnectionError,
    ModelOverloadError
)
import logging
import asyncio

logger = logging.getLogger(__name__)

class ResilientAgent(OpenCodeAgent):
    """Agent with enhanced error handling and recovery"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry_config = {
            "max_retries": 3,
            "backoff_factor": 2,
            "max_backoff": 60
        }

    async def process_message_safe(
        self,
        message: str,
        conversation_id: str
    ) -> str:
        """Process message with comprehensive error handling"""

        retries = 0
        backoff = 1

        while retries < self.retry_config["max_retries"]:
            try:
                return await self.process_message(message, conversation_id)

            except ToolExecutionError as e:
                logger.error(f"Tool execution failed: {e}")
                # Try alternative approach
                return await self._fallback_response(message, e)

            except MCPConnectionError as e:
                logger.error(f"MCP connection error: {e}")
                retries += 1
                if retries >= self.retry_config["max_retries"]:
                    return "I'm having trouble connecting to my tools. Please try again later."

                # Exponential backoff
                await asyncio.sleep(backoff)
                backoff = min(
                    backoff * self.retry_config["backoff_factor"],
                    self.retry_config["max_backoff"]
                )

                # Try to reconnect
                await self._reconnect_mcp()

            except ModelOverloadError as e:
                logger.error(f"Model overloaded: {e}")
                # Switch to smaller model
                return await self._process_with_fallback_model(
                    message,
                    conversation_id
                )

            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                return "I encountered an unexpected error. Please try rephrasing your request."

    async def _fallback_response(
        self,
        message: str,
        error: Exception
    ) -> str:
        """Generate fallback response without tools"""

        prompt = f"""
        The user asked: {message}

        I encountered an error with my tools: {error}

        Please provide a helpful response without using tools.
        """

        return await self.llm_provider.complete(prompt)

    async def _reconnect_mcp(self):
        """Attempt to reconnect to MCP servers"""

        logger.info("Attempting to reconnect to MCP servers")

        for server_name, server in self.mcp_client.servers.items():
            try:
                await server.reconnect()
                logger.info(f"Reconnected to {server_name}")
            except Exception as e:
                logger.error(f"Failed to reconnect to {server_name}: {e}")
```

## Conclusion

This comprehensive requirements document provides the foundation for developing robust OpenCode agents that integrate seamlessly with the MCP ecosystem. By following these specifications and guidelines, developers can create agents that are:

- **Scalable**: Handle thousands of concurrent users
- **Reliable**: Provide consistent service with proper error handling
- **Secure**: Implement proper authentication and authorization
- **Extensible**: Support custom tools and integrations
- **Observable**: Provide comprehensive monitoring and debugging

For the latest updates and additional resources, please refer to:
- [MCP Python Repository](https://github.com/djbchepe/mcp-python)
- [OpenCode Documentation](https://docs.opencode.ai)
- [MCP Protocol Specification](https://modelcontextprotocol.io)

---

*Document Version: 1.0.0*
*Last Updated: August 2024*
*Maintained by: OpenCode Development Team*
