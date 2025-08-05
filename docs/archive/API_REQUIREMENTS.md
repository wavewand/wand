# MCP HTTP API Requirements for Claude Code and OpenCode Integration

## Overview

This document specifies the complete API requirements for implementing a Model Context Protocol (MCP) server over HTTP to support both Claude Code and OpenCode integrations. The implementation must conform to the MCP specification version 2025-06-18 and support Streamable HTTP transport with optional Server-Sent Events (SSE).

## Protocol Foundation

### Base Protocol
- **Message Format**: JSON-RPC 2.0
- **Character Encoding**: UTF-8
- **Transport**: Streamable HTTP with optional Server-Sent Events (SSE)
- **Protocol Version**: 2025-06-18 (with backwards compatibility to 2024-11-05)

### Message Types
1. **Requests**: Client-server communication to initiate operations
2. **Responses**: Server replies to requests (success/error)
3. **Notifications**: One-way messages that don't require responses

## HTTP Transport Specification

### Endpoint Requirements

#### Single MCP Endpoint
- **Path**: Single HTTP endpoint (e.g., `/mcp`)
- **Methods**: Both POST and GET must be supported
- **POST**: Send JSON-RPC messages from client to server
- **GET**: Open Server-Sent Events (SSE) stream for server-to-client communication

### HTTP Headers

#### Required Headers
- **MCP-Protocol-Version**: Protocol version (defaults to previous version if not specified)
- **Accept**: Must include both `application/json` and `text/event-stream`
- **Content-Type**: `application/json` for POST requests
- **Origin**: Must be validated for security

#### Optional Headers
- **Mcp-Session-Id**: Session identifier for session management
- **Last-Event-ID**: For SSE stream resumability
- **Authorization**: For OAuth 2.0/2.1 authentication

### Request/Response Flow

#### Client-to-Server (POST)
```http
POST /mcp HTTP/1.1
Host: server.example.com
Content-Type: application/json
Accept: application/json, text/event-stream
MCP-Protocol-Version: 2025-06-18
Mcp-Session-Id: session-123

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

#### Server Response Options
1. **JSON Response** (202 Accepted for responses/notifications)
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [...]
  }
}
```

2. **SSE Stream Response**
```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"jsonrpc": "2.0", "id": 1, "result": {...}}

```

#### Server-to-Client (SSE GET)
```http
GET /mcp HTTP/1.1
Host: server.example.com
Accept: text/event-stream
MCP-Protocol-Version: 2025-06-18
Mcp-Session-Id: session-123
Last-Event-ID: event-456
```

## Core MCP Capabilities

### 1. Tools (Function Calling)
Functions that LLMs can invoke to perform specific actions.

#### Tool Discovery
- **Method**: `tools/list`
- **Description**: List available tools
- **Response**: Array of tool definitions with name, description, and input schema

#### Tool Invocation
- **Method**: `tools/call`
- **Parameters**:
  - `name`: Tool name
  - `arguments`: Tool-specific parameters
- **Response**: Tool execution result

### 2. Resources (Data Access)
Data sources that provide context without side effects.

#### Resource Discovery
- **Method**: `resources/list`
- **Description**: List available resources
- **Response**: Array of resource definitions

#### Resource Access
- **Method**: `resources/read`
- **Parameters**:
  - `uri`: Resource identifier
- **Response**: Resource content

### 3. Prompts (Templates)
Pre-defined templates for optimal tool/resource usage.

#### Prompt Discovery
- **Method**: `prompts/list`
- **Description**: List available prompts
- **Response**: Array of prompt definitions

#### Prompt Access
- **Method**: `prompts/get`
- **Parameters**:
  - `name`: Prompt name
  - `arguments`: Template variables
- **Response**: Rendered prompt content

## Authentication and Security

### OAuth 2.0/2.1 Support
- **Framework**: OAuth 2.1 mandatory for remote HTTP servers
- **Resource Indicators**: RFC 8707 compliance required
- **Token Management**: Automatic refresh and secure storage
- **Scopes**: Configurable access levels (local, project, user)

### API Key Authentication
- **Header**: `Authorization: Bearer <api-key>`
- **Environment Variables**: Support for credential injection
- **Validation**: Server-side key verification

### Security Requirements
- **Origin Validation**: Mandatory Origin header validation
- **Local Binding**: Prevent network attacks for local servers
- **TLS/SSL**: HTTPS required for remote servers
- **Session Management**: Secure session handling with unique identifiers

## Session Management

### Session Lifecycle
1. **Initialization**: Handshake with capability exchange
2. **Active Session**: Ongoing client-server communication
3. **Termination**: Explicit session cleanup

### Session Operations
- **Create**: Automatic on first request or explicit initialization
- **Maintain**: Session ID tracking across requests
- **Terminate**: HTTP DELETE for explicit cleanup
- **Resume**: SSE stream resumability with event IDs

## Claude Code Integration Requirements

### Configuration Support
- **Command Structure**: `claude mcp add [options] <server-name> <server-url>`
- **Transport Options**: `--transport http` or `--transport sse`
- **Scope Configuration**: `--scope local|project|user`
- **Environment Variables**: `--env KEY=VALUE`
- **Headers**: `--header KEY=VALUE`

### Authentication Methods
- OAuth 2.0 flows with automatic token refresh
- API key authentication via headers or environment variables
- Environment variable-based credential management

### Security Considerations
- Third-party server risk acknowledgment
- Prompt injection protection
- Server trustworthiness verification

## OpenCode Integration Requirements

### Client-Server Architecture
- **MCP Hosts**: AI applications (OpenCode)
- **MCP Clients**: Protocol clients maintaining 1:1 server connections
- **MCP Servers**: Lightweight capability providers

### Protocol Flow
1. **Initialization**: Capability and version handshake
2. **Discovery**: Server capability enumeration
3. **Context Provision**: Resource and prompt availability
4. **Invocation**: Tool execution requests
5. **Execution**: Server-side logic execution and response

### Scalability Requirements
- Multiple concurrent client connections
- Efficient message handling
- Resource pooling and management
- Load balancing capability

## Error Handling

### JSON-RPC Error Responses
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "Method not found",
    "data": {
      "method": "invalid/method"
    }
  }
}
```

### HTTP Status Codes
- **200 OK**: Successful request
- **202 Accepted**: Notification/response accepted
- **400 Bad Request**: Invalid JSON-RPC or malformed request
- **401 Unauthorized**: Authentication required
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Resource or method not found
- **500 Internal Server Error**: Server-side error

### Error Recovery
- Connection retry mechanisms
- Graceful degradation for unsupported features
- Session recovery and resumption

## Performance Requirements

### Response Times
- **Tool Calls**: < 5 seconds for most operations
- **Resource Access**: < 2 seconds for data retrieval
- **Discovery Operations**: < 1 second

### Throughput
- Support for concurrent client connections
- Efficient JSON-RPC message processing
- SSE stream management without resource leaks

### Caching
- Resource content caching where appropriate
- Tool result caching for idempotent operations
- Session state caching

## Backwards Compatibility

### Version Support
- Support for protocol versions 2024-11-05 and 2025-06-18
- Graceful fallback for older clients
- Feature detection and negotiation

### Transport Compatibility
- Support for both SSE and Streamable HTTP transports
- Automatic transport selection based on client capabilities
- Migration path from deprecated transports

## Implementation Checklist

### Core Protocol
- [ ] JSON-RPC 2.0 message handling
- [ ] UTF-8 encoding support
- [ ] Protocol version negotiation
- [ ] Error handling and reporting

### HTTP Transport
- [ ] Single endpoint supporting POST/GET
- [ ] Required header validation
- [ ] Session management
- [ ] SSE stream implementation

### MCP Capabilities
- [ ] Tools discovery and invocation
- [ ] Resources listing and access
- [ ] Prompts discovery and rendering

### Security
- [ ] OAuth 2.0/2.1 implementation
- [ ] API key authentication
- [ ] Origin validation
- [ ] TLS/SSL support

### Client Integration
- [ ] Claude Code configuration support
- [ ] OpenCode protocol compliance
- [ ] Multi-client connection handling

### Performance
- [ ] Response time optimization
- [ ] Connection pooling
- [ ] Caching implementation
- [ ] Resource management

## Testing Requirements

### Unit Tests
- JSON-RPC message parsing and generation
- Authentication mechanisms
- Session management
- Error handling

### Integration Tests
- Claude Code client integration
- OpenCode client integration
- Multi-client scenarios
- Authentication flows

### Performance Tests
- Concurrent connection handling
- Response time benchmarks
- Memory usage profiling
- SSE stream performance

## Documentation Requirements

### API Documentation
- Complete endpoint documentation
- Authentication setup guides
- Configuration examples
- Error code reference

### Integration Guides
- Claude Code setup instructions
- OpenCode integration examples
- Troubleshooting guides
- Best practices documentation

This specification provides the complete foundation for implementing an MCP HTTP server that supports both Claude Code and OpenCode integrations while maintaining compliance with the latest MCP protocol standards.
