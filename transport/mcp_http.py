#!/usr/bin/env python3
"""
MCP HTTP Transport Layer

Provides HTTP transport for MCP protocol compliant with specification version 2025-06-18.
Supports both Claude Code and OpenCode integrations with Streamable HTTP transport.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from observability.enhanced_logging import LogCategory, correlation_id_var, get_enhanced_logger, get_mcp_logger

logger = get_enhanced_logger(LogCategory.MCP_PROTOCOL)
system_logger = get_enhanced_logger(LogCategory.SYSTEM)
mcp_logger = get_mcp_logger()


class MCPRequest(BaseModel):
    """MCP protocol request model"""

    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    method: str
    params: dict = {}


class MCPResponse(BaseModel):
    """MCP protocol response model"""

    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    result: dict = None
    error: dict = None


class MCPHttpTransport:
    """HTTP transport for MCP protocol compliant with 2025-06-18 specification"""

    def __init__(self, orchestrator, server_version: str = "2025-06-18"):
        self.app = FastAPI(
            title="Model Context Protocol (MCP) HTTP API",
            version=server_version,
            description="HTTP API for Model Context Protocol supporting Claude Code and OpenCode integrations",
        )
        self.orchestrator = orchestrator
        self.sessions = {}  # Session management
        self.protocol_version = "2025-06-18"

        # Add CORS for Claude Code and OpenCode compatibility
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=[
                "Content-Type",
                "Accept",
                "Authorization",
                "Origin",
                "MCP-Protocol-Version",
                "Mcp-Session-Id",
                "Last-Event-ID",
            ],
        )

        self._setup_routes()

    def _setup_routes(self):
        """Setup MCP HTTP routes compliant with 2025-06-18 specification"""

        @self.app.post("/mcp")
        async def mcp_http_endpoint(
            request: MCPRequest,
            mcp_protocol_version: Optional[str] = Header(None, alias="MCP-Protocol-Version"),
            mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
            origin: Optional[str] = Header(None, alias="Origin"),
        ):
            """Primary MCP endpoint for JSON-RPC communication"""
            # Generate correlation ID for request tracking
            correlation_id = str(uuid.uuid4())
            correlation_id_var.set(correlation_id)
            start_time = time.time()

            # Log the incoming MCP request
            mcp_logger.log_mcp_request(
                method=request.method, params=request.params, correlation_id=correlation_id, stdio_mode=False
            )

            try:
                # Handle protocol version negotiation
                protocol_version = mcp_protocol_version or "2024-11-05"  # Default to previous version
                if protocol_version not in ["2024-11-05", "2025-03-26", "2025-06-18"]:
                    protocol_version = "2024-11-05"

                # Session management
                session_id = mcp_session_id or str(uuid.uuid4())
                if session_id not in self.sessions:
                    self.sessions[session_id] = {"created_at": datetime.now(), "protocol_version": protocol_version}

                    system_logger.info(
                        f"New MCP session created: {session_id}",
                        extra={
                            "category": LogCategory.SYSTEM,
                            "session_data": {
                                "session_id": session_id,
                                "protocol_version": protocol_version,
                                "origin": origin,
                            },
                        },
                    )

                if request.method == "initialize":
                    return JSONResponse(
                        content={
                            "jsonrpc": "2.0",
                            "id": request.id,
                            "result": {
                                "protocolVersion": protocol_version,
                                "capabilities": {
                                    "tools": {"listChanged": True},
                                    "resources": {"subscribe": True, "listChanged": True},
                                    "prompts": {"listChanged": True},
                                    "logging": {},
                                },
                                "serverInfo": {
                                    "name": "mcp-python-orchestrator",
                                    "version": self.protocol_version,
                                    "description": "MCP server with 22-tool orchestrator backend",
                                },
                            },
                        }
                    )

                elif request.method == "notifications/initialized":
                    # No response for notifications
                    return JSONResponse(content="", status_code=204)

                elif request.method == "tools/list":
                    # Return all 22 tools available through orchestrator
                    tools = self._get_available_tools()
                    return JSONResponse(content={"jsonrpc": "2.0", "id": request.id, "result": {"tools": tools}})

                elif request.method == "tools/call":
                    # Execute tool via orchestrator
                    tool_name = request.params.get("name")
                    arguments = request.params.get("arguments", {})

                    # Log tool call
                    mcp_logger.log_tool_call(tool_name, arguments, correlation_id)

                    tool_start_time = time.time()
                    try:
                        # Lazy import to avoid circular imports
                        from orchestrator.agent_orchestrator import execute_tool_via_orchestrator

                        result = await execute_tool_via_orchestrator(tool_name, arguments)

                        tool_execution_time = (time.time() - tool_start_time) * 1000

                        # Log successful tool result
                        mcp_logger.log_tool_result(
                            tool_name=tool_name,
                            result=result,
                            error=None,
                            execution_time_ms=tool_execution_time,
                            correlation_id=correlation_id,
                        )

                        response_content = {
                            "jsonrpc": "2.0",
                            "id": request.id,
                            "result": {
                                "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]
                            },
                        }

                        # Log successful MCP response
                        execution_time = (time.time() - start_time) * 1000
                        mcp_logger.log_mcp_response(
                            method=request.method,
                            result=response_content["result"],
                            error=None,
                            correlation_id=correlation_id,
                            execution_time_ms=execution_time,
                        )

                        return JSONResponse(content=response_content)

                    except Exception as e:
                        tool_execution_time = (time.time() - tool_start_time) * 1000
                        execution_time = (time.time() - start_time) * 1000

                        # Log tool error
                        mcp_logger.log_tool_result(
                            tool_name=tool_name,
                            result=None,
                            error=str(e),
                            execution_time_ms=tool_execution_time,
                            correlation_id=correlation_id,
                        )

                        logger.error(
                            f"Tool execution error: {e}",
                            extra={
                                "category": LogCategory.ERROR,
                                "error_data": {
                                    "tool_name": tool_name,
                                    "arguments": arguments,
                                    "correlation_id": correlation_id,
                                    "execution_time_ms": tool_execution_time,
                                },
                            },
                        )

                        error_response = {
                            "jsonrpc": "2.0",
                            "id": request.id,
                            "error": {"code": -32603, "message": f"Tool execution failed: {str(e)}"},
                        }

                        # Log error MCP response
                        mcp_logger.log_mcp_response(
                            method=request.method,
                            result=None,
                            error=error_response["error"],
                            correlation_id=correlation_id,
                            execution_time_ms=execution_time,
                        )

                        return JSONResponse(content=error_response)

                elif request.method == "ping":
                    # Health check endpoint
                    try:
                        from orchestrator.agent_orchestrator import get_orchestrator_status

                        status = await get_orchestrator_status()
                        return JSONResponse(
                            content={
                                "jsonrpc": "2.0",
                                "id": request.id,
                                "result": {"status": "healthy", "orchestrator": status},
                            }
                        )
                    except Exception as e:
                        return JSONResponse(
                            content={
                                "jsonrpc": "2.0",
                                "id": request.id,
                                "error": {"code": -32603, "message": f"Health check failed: {str(e)}"},
                            }
                        )

                else:
                    # Unknown method
                    return JSONResponse(
                        content={
                            "jsonrpc": "2.0",
                            "id": request.id,
                            "error": {"code": -32601, "message": f"Unknown method: {request.method}"},
                        }
                    )

            except Exception as e:
                logger.error(f"MCP HTTP transport error: {e}")
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": request.id,
                        "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                    }
                )

        @self.app.get("/mcp")
        async def mcp_sse_endpoint(
            mcp_protocol_version: Optional[str] = Header(None, alias="MCP-Protocol-Version"),
            mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
            last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
            origin: Optional[str] = Header(None, alias="Origin"),
        ):
            """Server-Sent Events endpoint for server-to-client communication"""

            # For now, return a simple SSE stream
            # In a full implementation, this would maintain persistent connections
            def generate_sse():
                yield f"data: {{\"jsonrpc\": \"2.0\", \"method\": \"notifications/initialized\", \"params\": {{}}}}\n\n"

            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        @self.app.delete("/mcp")
        async def mcp_session_cleanup(mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id")):
            """Terminate MCP session and cleanup resources"""
            if mcp_session_id and mcp_session_id in self.sessions:
                del self.sessions[mcp_session_id]
                return JSONResponse(content={"status": "session terminated"})
            return JSONResponse(content={"error": "Session not found"}, status_code=404)

        @self.app.get("/health")
        async def health_check():
            """Simple health check endpoint"""
            try:
                from orchestrator.agent_orchestrator import get_orchestrator_status

                status = await get_orchestrator_status()
                return {"status": "healthy", "orchestrator": status}
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}

    def _get_available_tools(self):
        """Get list of all available tools with proper JSON Schema"""
        # Get tools from orchestrator
        if hasattr(self.orchestrator, 'get_available_tools'):
            return self.orchestrator.get_available_tools()

        # Fallback to hardcoded list
        return [
            {
                "name": "execute_command",
                "description": "Execute shell command",
                "inputSchema": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "Shell command to execute"}},
                    "required": ["command"],
                },
            },
            {
                "name": "read_file",
                "description": "Read file contents",
                "inputSchema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string", "description": "Path to file to read"}},
                    "required": ["file_path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write content to file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to file to write"},
                        "content": {"type": "string", "description": "Content to write to file"},
                    },
                    "required": ["file_path", "content"],
                },
            },
            {
                "name": "list_directory",
                "description": "List directory contents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "Directory path to list", "default": "."}
                    },
                    "required": [],
                },
            },
            {
                "name": "search_files",
                "description": "Search for files",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Search pattern"},
                        "directory": {"type": "string", "description": "Directory to search in", "default": "."},
                    },
                    "required": ["pattern"],
                },
            },
            {
                "name": "get_system_info",
                "description": "Get system information",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_command_exists",
                "description": "Check if command exists",
                "inputSchema": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "Command to check"}},
                    "required": ["command"],
                },
            },
            {
                "name": "create_project",
                "description": "Create multi-component project",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Project name"},
                        "description": {"type": "string", "description": "Project description"},
                        "components": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Project components",
                        },
                    },
                    "required": ["name", "description", "components"],
                },
            },
            {
                "name": "distribute_task",
                "description": "Distribute task to agents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Task title"},
                        "description": {"type": "string", "description": "Task description"},
                        "task_type": {"type": "string", "description": "Task type"},
                    },
                    "required": ["title", "description", "task_type"],
                },
            },
            {
                "name": "get_project_status",
                "description": "Get project and agent status",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "slack_notify",
                "description": "Send Slack notification",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string", "description": "Slack channel"},
                        "message": {"type": "string", "description": "Message to send"},
                    },
                    "required": ["channel", "message"],
                },
            },
            {
                "name": "git_operation",
                "description": "Perform Git operations",
                "inputSchema": {
                    "type": "object",
                    "properties": {"operation": {"type": "string", "description": "Git operation to perform"}},
                    "required": ["operation"],
                },
            },
            {
                "name": "jenkins_trigger",
                "description": "Trigger Jenkins build",
                "inputSchema": {
                    "type": "object",
                    "properties": {"job_name": {"type": "string", "description": "Jenkins job name"}},
                    "required": ["job_name"],
                },
            },
            {
                "name": "youtrack_issue",
                "description": "Create YouTrack issue",
                "inputSchema": {
                    "type": "object",
                    "properties": {"operation": {"type": "string", "description": "YouTrack operation"}},
                    "required": ["operation"],
                },
            },
            {
                "name": "postgres_query",
                "description": "Execute PostgreSQL query",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database": {"type": "string", "description": "Database name"},
                        "query": {"type": "string", "description": "SQL query"},
                    },
                    "required": ["database", "query"],
                },
            },
            {
                "name": "aws_operation",
                "description": "Execute AWS operation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string", "description": "AWS service"},
                        "operation": {"type": "string", "description": "AWS operation"},
                    },
                    "required": ["service", "operation"],
                },
            },
            {
                "name": "bambu_print",
                "description": "Start Bambu Lab 3D print",
                "inputSchema": {
                    "type": "object",
                    "properties": {"printer_id": {"type": "string", "description": "Printer ID"}},
                    "required": ["printer_id"],
                },
            },
            {
                "name": "web_search",
                "description": "Search the web",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
            },
            {
                "name": "api_request",
                "description": "Make HTTP API request",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "API URL"},
                        "method": {"type": "string", "description": "HTTP method", "default": "GET"},
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "get_system_status",
                "description": "Get system status",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "list_agents",
                "description": "List available agents",
                "inputSchema": {
                    "type": "object",
                    "properties": {"status": {"type": "string", "description": "Filter by status", "default": "all"}},
                    "required": [],
                },
            },
            {
                "name": "create_task",
                "description": "Create new task",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Task title"},
                        "description": {"type": "string", "description": "Task description"},
                        "task_type": {"type": "string", "description": "Task type"},
                    },
                    "required": ["title", "description", "task_type"],
                },
            },
            {
                "name": "ollama",
                "description": "Execute local Ollama AI operations (generate, chat, embed, list_models, pull_model, show_model)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "description": "Ollama operation to perform",
                            "enum": ["generate", "chat", "embed", "list_models", "pull_model", "show_model"],
                        },
                        "prompt": {"type": "string", "description": "Text prompt (for generate operation)"},
                        "messages": {
                            "type": "array",
                            "description": "Chat messages array (for chat operation)",
                            "items": {
                                "type": "object",
                                "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
                                "required": ["role", "content"],
                            },
                        },
                        "input_text": {"type": "string", "description": "Text to embed (for embed operation)"},
                        "model_name": {
                            "type": "string",
                            "description": "Model name (for pull_model and show_model operations)",
                        },
                        "model": {"type": "string", "description": "Model to use for operation"},
                        "temperature": {"type": "number", "description": "Temperature for text generation"},
                        "max_tokens": {"type": "integer", "description": "Maximum tokens to generate"},
                    },
                    "required": ["operation"],
                },
            },
        ]
