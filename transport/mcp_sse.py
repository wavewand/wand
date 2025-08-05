#!/usr/bin/env python3
"""
MCP SSE (Server-Sent Events) Transport Layer

Provides SSE transport for MCP protocol to support Claude integration.
Based on the MCP SSE specification and Claude's requirements.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SSEMessage(BaseModel):
    """SSE message format"""

    id: Optional[str] = None
    event: Optional[str] = None
    data: str
    retry: Optional[int] = None


class MCPSSETransport:
    """SSE transport for MCP protocol compatible with Claude"""

    def __init__(self, orchestrator, server_version: str = "1.0.0"):
        self.orchestrator = orchestrator
        self.server_version = server_version
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def setup_routes(self, app):
        """Setup SSE routes on the FastAPI app"""

        @app.get("/sse")
        async def mcp_sse_endpoint(request: Request):
            """SSE endpoint for MCP protocol - Claude compatible"""
            session_id = str(uuid.uuid4())

            logger.info(f"New SSE connection established: {session_id}")

            # Create session
            self.sessions[session_id] = {"created_at": time.time(), "last_activity": time.time(), "initialized": False}

            async def event_generator() -> AsyncGenerator[str, None]:
                """Generate SSE events for Claude"""
                try:
                    # Send initial connection event
                    yield self._format_sse_message(
                        {"jsonrpc": "2.0", "method": "connection/established", "params": {"sessionId": session_id}}
                    )

                    # Wait for initialize request via separate POST endpoint
                    # For now, auto-initialize for Claude
                    await asyncio.sleep(0.1)

                    # Send capabilities
                    yield self._format_sse_message(
                        {
                            "jsonrpc": "2.0",
                            "id": "init-1",
                            "result": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {"tools": {}, "resources": {}},
                                "serverInfo": {
                                    "name": "distributed-mcp-server-orchestrated",
                                    "version": self.server_version,
                                },
                            },
                        }
                    )

                    # Keep connection alive
                    while session_id in self.sessions:
                        # Send heartbeat every 30 seconds
                        await asyncio.sleep(30)
                        yield ":heartbeat\n\n"

                except asyncio.CancelledError:
                    logger.info(f"SSE connection cancelled: {session_id}")
                except Exception as e:
                    logger.error(f"Error in SSE stream for {session_id}: {e}")
                finally:
                    # Clean up session
                    if session_id in self.sessions:
                        del self.sessions[session_id]
                    logger.info(f"SSE connection closed: {session_id}")

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable Nginx buffering
                    "Access-Control-Allow-Origin": "*",  # CORS for Claude
                },
            )

        @app.post("/sse")
        async def mcp_sse_rpc(request: Request):
            """Handle JSON-RPC requests over POST for SSE transport"""
            try:
                body = await request.json()

                # Handle different MCP methods
                if body.get("method") == "initialize":
                    return {
                        "jsonrpc": "2.0",
                        "id": body.get("id"),
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}, "resources": {}},
                            "serverInfo": {
                                "name": "distributed-mcp-server-orchestrated",
                                "version": self.server_version,
                            },
                        },
                    }

                elif body.get("method") == "tools/list":
                    # Get tools from orchestrator
                    tools = self.orchestrator.get_available_tools()
                    return {"jsonrpc": "2.0", "id": body.get("id"), "result": {"tools": tools}}

                elif body.get("method") == "tools/call":
                    # Execute tool via orchestrator
                    tool_name = body.get("params", {}).get("name")
                    arguments = body.get("params", {}).get("arguments", {})

                    try:
                        from orchestrator.agent_orchestrator import execute_tool_via_orchestrator

                        result = await execute_tool_via_orchestrator(tool_name, arguments)

                        return {
                            "jsonrpc": "2.0",
                            "id": body.get("id"),
                            "result": {"content": [{"type": "text", "text": str(result)}]},
                        }
                    except Exception as e:
                        return {"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32603, "message": str(e)}}

                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": body.get("id"),
                        "error": {"code": -32601, "message": f"Method not found: {body.get('method')}"},
                    }

            except Exception as e:
                logger.error(f"Error handling SSE RPC: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id") if "body" in locals() else None,
                    "error": {"code": -32603, "message": str(e)},
                }

    def _format_sse_message(self, data: Dict[str, Any], event: Optional[str] = None) -> str:
        """Format a message for SSE"""
        message = ""
        if event:
            message += f"event: {event}\n"
        message += f"data: {json.dumps(data)}\n\n"
        return message
