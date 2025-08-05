# Wand Architecture

## System Overview

Wand is a Model Context Protocol (MCP) server that provides a unified interface for 50+ integrations, enabling AI agents to interact with various services and tools.

## Architecture Diagram

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
│  │ • HuggingFace      │  │ • Terraform         │  │                  │  │
│  └─────────────────────┘  └─────────────────────┘  └──────────────────┘  │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐  │
│  │   Cloud Storage     │  │     Business        │  │    Security      │  │
│  ├─────────────────────┤  ├─────────────────────┤  ├──────────────────┤  │
│  │ • AWS S3            │  │ • CRM Systems       │  │ • Auth providers │  │
│  │ • Google Drive      │  │ • Payment APIs      │  │ • Vault          │  │
│  │ • Dropbox           │  │ • Project Mgmt      │  │ • Security scan  │  │
│  │ • OneDrive          │  │ • HR Tools          │  │                  │  │
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

## Component Description

### Client Layer
- **Claude Desktop/Code**: Primary MCP clients using stdio transport
- **MCP Clients**: Third-party clients using HTTP/WebSocket

### Transport Layer
- **stdio**: Direct process communication for local clients
- **HTTP/SSE**: RESTful API with Server-Sent Events for remote clients
- **WebSocket**: Real-time bidirectional communication

### Core Components

#### Wand Server (`wand.py`)
- Handles protocol communication
- Routes requests to appropriate handlers
- Manages sessions and authentication

#### Orchestrator Manager
- Distributes tasks across agents
- Load balancing and fault tolerance
- Task prioritization and queuing

#### Task Manager
- Tracks task lifecycle
- Manages execution backends
- Handles retries and timeouts

#### Agent Manager
- Registers and monitors agents
- Capability-based routing
- Health checking

### Integration Layer
50+ integrations organized by category:
- **AI/ML**: Language models and ML services
- **DevOps**: Development and deployment tools
- **Communication**: Messaging and collaboration
- **Cloud Storage**: File storage and management

### Execution Backends
- **Native**: Direct process execution
- **Docker Socket**: Container-based isolation
- **SSH Remote**: Remote server execution
- **Host Agent**: Privileged host operations

### Data Layer
- **PostgreSQL**: Persistent storage for configurations and state
- **File System**: Local file operations
- **Vector Store**: Embeddings for RAG applications

## Request Flow

1. Client sends MCP request via transport
2. MCP Server validates and authenticates
3. Request routed to appropriate integration
4. Integration executes with selected backend
5. Response returned through transport layer

## Security Features

- JWT-based authentication
- Rate limiting per integration
- Command validation and sanitization
- Audit logging
- Sandboxed execution environments

## Scalability

- Horizontal scaling via distributed agents
- Task queue for async processing
- Connection pooling for database
- Caching layer for frequent requests

## Monitoring

- Prometheus metrics endpoint
- Health check endpoints
- Structured logging
- Performance tracking
