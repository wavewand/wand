# MCP Platform Configuration Guide

## Overview

This guide provides comprehensive configuration information for the MCP Python platform, including environment variables, agent configuration, Claude Code integration, and deployment settings.

## Table of Contents

1. [Environment Configuration](#environment-configuration)
2. [Agent Configuration](#agent-configuration)
3. [Claude Code Configuration](#claude-code-configuration)
4. [Docker Configuration](#docker-configuration)
5. [Database Configuration](#database-configuration)
6. [Security Configuration](#security-configuration)
7. [Monitoring Configuration](#monitoring-configuration)

## Environment Configuration

### Main Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
WORKERS=4

# Database Configuration
DATABASE_URL=postgresql://mcp_user:mcp_password@localhost:5432/mcp_platform
# For in-memory database (development only)
# DATABASE_URL=memory://

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=redis_password

# Authentication
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# AI Framework API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
COHERE_API_KEY=your_cohere_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key

# Framework Configuration
HAYSTACK_API_KEY=your_haystack_api_key
LLAMAINDEX_API_KEY=your_llamaindex_api_key

# Caching Configuration
CACHE_MAX_SIZE=1000
CACHE_DEFAULT_TTL=3600
ENABLE_REDIS_CACHE=true

# Monitoring Configuration
ENABLE_MONITORING=true
ENABLE_METRICS=true
METRICS_PORT=9090
METRICS_EXPORT_FORMAT=json

# WebSocket Configuration
ENABLE_WEBSOCKETS=true
WEBSOCKET_MAX_CONNECTIONS=1000

# Batch Processing Configuration
BATCH_MAX_SIZE=100
BATCH_TIMEOUT_SECONDS=30

# Security Configuration
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
RATE_LIMIT_REQUESTS_PER_MINUTE=100
ENABLE_API_KEY_AUTH=false

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE_PATH=./logs/mcp-platform.log
```

### Docker Environment Variables

For Docker deployments, these variables are set in `docker-compose.yml`:

```yaml
environment:
  # Database configuration
  DATABASE_URL: postgresql://mcp_user:mcp_password@postgres:5432/mcp_platform
  REDIS_URL: redis://redis:6379/0

  # API configuration
  API_HOST: 0.0.0.0
  API_PORT: 8000
  WORKERS: 4

  # Framework configuration
  OPENAI_API_KEY: ${OPENAI_API_KEY:-}
  ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}

  # Feature flags
  ENABLE_MONITORING: ${ENABLE_MONITORING:-true}
  ENABLE_METRICS: ${ENABLE_METRICS:-true}
  METRICS_PORT: 9090
```

## Agent Configuration

### Agent Types and Capabilities

The platform supports various agent types with specific capabilities:

#### 1. Processing Agents
```json
{
  "name": "DataProcessorAgent",
  "type": "processing",
  "capabilities": [
    "data_processing",
    "file_handling",
    "batch_operations",
    "data_validation"
  ],
  "status": "active",
  "config": {
    "max_concurrent_tasks": 5,
    "supported_formats": ["json", "csv", "xml"],
    "memory_limit_mb": 1024
  },
  "metadata": {
    "version": "1.2.0",
    "created_by": "system",
    "last_updated": "2024-01-15T10:30:00Z"
  }
}
```

#### 2. Automation Agents
```json
{
  "name": "WorkflowAutomationAgent",
  "type": "automation",
  "capabilities": [
    "workflow_execution",
    "task_scheduling",
    "event_handling",
    "integration_management"
  ],
  "status": "active",
  "config": {
    "max_workflow_depth": 10,
    "timeout_seconds": 300,
    "retry_attempts": 3
  },
  "metadata": {
    "department": "operations",
    "cost_center": "IT-001"
  }
}
```

#### 3. AI Framework Agents
```json
{
  "name": "LlamaIndexAgent",
  "type": "ai_framework",
  "capabilities": [
    "document_indexing",
    "similarity_search",
    "rag_queries",
    "embeddings_generation"
  ],
  "status": "active",
  "config": {
    "framework": "llamaindex",
    "model": "gpt-3.5-turbo",
    "max_tokens": 4096,
    "temperature": 0.7
  }
}
```

### Agent Configuration Script

Here's a script to set up common agents:

```bash
#!/bin/bash
# setup_agents.sh - Configure standard agents for the MCP platform

BASE_URL="http://localhost:8000"

echo "🤖 Setting up MCP Platform Agents"
echo "================================="

# Function to create an agent
create_agent() {
    local name="$1"
    local type="$2"
    local capabilities="$3"
    local config="$4"

    echo ""
    echo "Creating agent: $name"

    curl -s -X POST "$BASE_URL/api/v1/agents" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"$name\",
            \"type\": \"$type\",
            \"capabilities\": $capabilities,
            \"status\": \"active\",
            \"config\": $config,
            \"metadata\": {
                \"created_by\": \"setup_script\",
                \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
            }
        }" | jq '.'
}

# 1. Data Processing Agent
create_agent "DataProcessorAgent" "processing" \
    '["data_processing", "file_handling", "batch_operations"]' \
    '{"max_concurrent_tasks": 5, "supported_formats": ["json", "csv", "xml"]}'

# 2. Workflow Automation Agent
create_agent "WorkflowAgent" "automation" \
    '["workflow_execution", "task_scheduling", "event_handling"]' \
    '{"max_workflow_depth": 10, "timeout_seconds": 300}'

# 3. Document Analysis Agent
create_agent "DocumentAnalyzer" "analysis" \
    '["document_parsing", "content_extraction", "classification"]' \
    '{"supported_types": ["pdf", "docx", "txt"], "max_file_size_mb": 50}'

# 4. Integration Agent
create_agent "IntegrationAgent" "integration" \
    '["api_integration", "data_sync", "webhook_handling"]' \
    '{"max_connections": 20, "retry_attempts": 3}'

echo ""
echo "✅ Agent setup completed!"
echo ""
echo "📋 Verify agents were created:"
echo "curl -X GET $BASE_URL/api/v1/agents | jq '.'"
```

### Agent Management via MCP

You can also manage agents through Claude Code using MCP tools:

```bash
# List all agents
claude --print "List all agents in the system" \
    --allowedTools "mcp__mcp-automation-api__list_agents"

# Create agent via Claude Code (will use create_task tool to request agent creation)
claude --print "Create a task to set up a new data processing agent named 'CSVProcessor' with capabilities for CSV file processing" \
    --allowedTools "mcp__mcp-automation-api__create_task"
```

## Claude Code Configuration

### Configuration File Locations

Claude Code configuration can be placed in different locations based on scope:

#### 1. User-Level Configuration
**Location**: `~/.config/claude/claude_code_config.json`

```json
{
  "mcpServers": {
    "mcp-automation-api": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"],
      "env": {
        "MCP_FETCH_BASE_URL": "http://localhost:8000/mcp"
      }
    }
  },
  "defaultSettings": {
    "allowedTools": ["mcp__mcp-automation-api__*"],
    "permissionMode": "prompt"
  }
}
```

#### 2. Project-Level Configuration
**Location**: `./.claude.json` (in project directory)

```json
{
  "mcpServers": {
    "mcp-automation-api": {
      "transport": "http",
      "url": "http://localhost:8000/mcp",
      "headers": {
        "X-Project": "mcp-automation"
      }
    }
  },
  "allowedTools": [
    "mcp__mcp-automation-api__get_system_status",
    "mcp__mcp-automation-api__list_agents",
    "mcp__mcp-automation-api__create_task"
  ],
  "permissionMode": "acceptEdits"
}
```

#### 3. Session-Specific Settings
**Location**: `~/.config/claude/mcp_session_settings.json`

```json
{
  "allowedTools": ["mcp__mcp-automation-api__*"],
  "disallowedTools": [],
  "permissionMode": "acceptEdits",
  "autoApprove": {
    "tools": ["mcp__mcp-automation-api__get_system_status"],
    "maxUsagePerSession": 50
  },
  "logging": {
    "enabled": true,
    "logLevel": "INFO",
    "logFile": "~/.claude/mcp_session.log"
  }
}
```

### MCP Server Configuration Options

#### HTTP Transport Configuration
```json
{
  "mcpServers": {
    "mcp-automation-api": {
      "transport": "http",
      "url": "http://localhost:8000/mcp",
      "timeout": 30000,
      "retries": 3,
      "headers": {
        "User-Agent": "Claude-Code-MCP/1.0",
        "X-API-Version": "v1"
      },
      "auth": {
        "type": "bearer",
        "token": "${MCP_API_TOKEN}"
      }
    }
  }
}
```

#### Stdio Transport Configuration
```json
{
  "mcpServers": {
    "mcp-automation-local": {
      "transport": "stdio",
      "command": "python",
      "args": ["/path/to/mcp-python/server.py"],
      "env": {
        "PYTHONPATH": "/path/to/mcp-python",
        "DATABASE_URL": "memory://",
        "LOG_LEVEL": "DEBUG"
      },
      "cwd": "/path/to/mcp-python"
    }
  }
}
```

### Permission Management Configuration

Create different permission profiles for different use cases:

#### Development Profile
```json
{
  "allowedTools": ["*"],
  "permissionMode": "acceptEdits",
  "dangerouslySkipPermissions": true,
  "logging": {
    "enabled": true,
    "logLevel": "DEBUG"
  }
}
```

#### Production Profile
```json
{
  "allowedTools": [
    "mcp__mcp-automation-api__get_system_status",
    "mcp__mcp-automation-api__list_agents"
  ],
  "disallowedTools": [
    "mcp__mcp-automation-api__create_task"
  ],
  "permissionMode": "prompt",
  "requireConfirmation": true,
  "maxUsagePerHour": 100
}
```

## Docker Configuration

### Docker Compose Configuration

The main `docker-compose.yml` configuration:

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15
    container_name: mcp-postgres
    environment:
      POSTGRES_DB: mcp_platform
      POSTGRES_USER: mcp_user
      POSTGRES_PASSWORD: mcp_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mcp_user -d mcp_platform"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - mcp-network

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: mcp-redis
    command: redis-server --appendonly yes --requirepass redis_password
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - mcp-network

  # MCP Python Backend API
  mcp-python:
    build:
      context: ../mcp-python
      dockerfile: Dockerfile
      target: production
    container_name: mcp-python
    command: ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
    environment:
      # Database configuration
      DATABASE_URL: postgresql://mcp_user:mcp_password@postgres:5432/mcp_platform
      REDIS_URL: redis://redis:6379/0

      # API configuration
      API_HOST: 0.0.0.0
      API_PORT: 8000

      # Framework configuration
      OPENAI_API_KEY: ${OPENAI_API_KEY:-}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}

      # Feature toggles
      ENABLE_MONITORING: ${ENABLE_MONITORING:-true}
      ENABLE_METRICS: ${ENABLE_METRICS:-true}
      METRICS_PORT: 9090
    ports:
      - "8000:8000"
      - "9092:9090"  # Metrics port
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - mcp-network

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local

networks:
  mcp-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.23.0.0/16
```

### Dockerfile Configuration

Multi-stage Dockerfile for optimal production builds:

```dockerfile
# Base stage
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mcp && useradd -r -g mcp mcp

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy source code
COPY --chown=mcp:mcp . .

USER mcp

CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --chown=mcp:mcp config ./config
COPY --chown=mcp:mcp database ./database
COPY --chown=mcp:mcp security ./security
COPY --chown=mcp:mcp utils ./utils
COPY --chown=mcp:mcp observability ./observability
COPY --chown=mcp:mcp frameworks ./frameworks
COPY --chown=mcp:mcp api ./api
COPY --chown=mcp:mcp main.py .

# Create directories and set permissions
RUN mkdir -p /app/logs /app/data && \
    chown -R mcp:mcp /app

USER mcp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

## Database Configuration

### PostgreSQL Configuration

#### Connection Settings
```python
# In api/server.py
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://mcp_user:mcp_password@localhost:5432/mcp_platform"
)

# Connection pool settings
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True,
    pool_recycle=300
)
```

#### Database Schema
```sql
-- agents table
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    capabilities TEXT[],
    status VARCHAR(50) DEFAULT 'inactive',
    config JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- tasks table
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    priority VARCHAR(20) DEFAULT 'medium',
    assigned_to UUID REFERENCES agents(id),
    project_id UUID REFERENCES projects(id),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- projects table
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### In-Memory Database Configuration

For development and testing:

```python
# In api/server.py
if DATABASE_URL == "memory://":
    class InMemoryDB:
        def __init__(self):
            self.agents: Dict[str, Agent] = {}
            self.tasks: Dict[str, Task] = {}
            self.projects: Dict[str, Project] = {}
            self.start_time = datetime.now(timezone.utc)

        def generate_id(self) -> str:
            return str(uuid.uuid4())

        def update_project_progress(self, project_id: str):
            if project_id in self.projects:
                project = self.projects[project_id]
                if project.tasks_total > 0:
                    project.progress = int((project.tasks_completed / project.tasks_total) * 100)
                else:
                    project.progress = 0

    db = InMemoryDB()
    use_postgres = False
```

## Security Configuration

### JWT Authentication
```python
# Security settings
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-this-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Password hashing
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
```

### CORS Configuration
```python
# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", ["http://localhost:3000"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Rate Limiting
```python
# Rate limiting configuration
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/v1/system/status")
@limiter.limit("100/minute")
async def get_system_status(request: Request):
    # Implementation
    pass
```

## Monitoring Configuration

### Prometheus Metrics
```python
# Metrics configuration
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
REQUEST_COUNT = Counter(
    'mcp_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'mcp_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    return response

@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Logging Configuration
```python
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.environ.get("LOG_FILE_PATH", "./logs/mcp-platform.log"))
    ]
)

logger = logging.getLogger("mcp-platform")
```

---

This configuration guide provides comprehensive coverage of all aspects of the MCP platform configuration, from basic environment setup to advanced production deployment settings.
