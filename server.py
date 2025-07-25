#!/usr/bin/env python3

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from mcp.types import TextContent, PromptMessage

# Initialize the MCP server
mcp = FastMCP("example-mcp-server", "1.0.0")

# Server state
server_start_time = datetime.now(timezone.utc)


# Tools
@mcp.tool()
async def get_time(ctx: Context) -> str:
    """Get the current server time"""
    current_time = datetime.now(timezone.utc)
    return f"Current time: {current_time.isoformat()}"


@mcp.tool()
async def get_uptime(ctx: Context) -> str:
    """Get the server uptime"""
    uptime = datetime.now(timezone.utc) - server_start_time
    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"Server uptime: {hours}h {minutes}m {seconds}s"


@mcp.tool()
async def calculate(ctx: Context, expression: str) -> str:
    """
    Perform a calculation
    
    Args:
        expression: Mathematical expression to calculate
    """
    try:
        # In production, use a proper expression parser for safety
        # This is a simple example that only allows basic math operations
        allowed_chars = "0123456789+-*/()., "
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"Result: {expression} = {result}"
        else:
            return f"Error: Invalid characters in expression '{expression}'"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@mcp.tool()
async def echo(ctx: Context, message: str, uppercase: bool = False) -> str:
    """
    Echo a message back
    
    Args:
        message: The message to echo
        uppercase: Whether to convert to uppercase (default: false)
    """
    result = message.upper() if uppercase else message
    return f"Echo: {result}"


# Resources
@mcp.resource("system://info")
async def get_system_info() -> str:
    """System information"""
    uptime = datetime.now(timezone.utc) - server_start_time
    return f"""System Information:
OS: {os.name}
Python Version: {os.sys.version.split()[0]}
Server Uptime: {uptime}
Process ID: {os.getpid()}
Working Directory: {os.getcwd()}"""


@mcp.resource("logs://recent")
async def get_recent_logs() -> str:
    """Recent server logs"""
    current_time = datetime.now(timezone.utc)
    logs = f"""Recent logs as of {current_time.isoformat()}:
[{server_start_time.isoformat()}] Server started
[{server_start_time.isoformat()}] Initialized MCP protocol
[{current_time.isoformat()}] Processing log request
[{current_time.isoformat()}] All systems operational"""
    return logs


@mcp.resource("config://current")
async def get_config() -> str:
    """Current server configuration"""
    config = {
        "server_name": "example-mcp-server",
        "version": "1.0.0",
        "features": {
            "tools": ["get_time", "get_uptime", "calculate", "echo"],
            "resources": ["system://info", "logs://recent", "config://current"],
            "prompts": ["greeting", "summary", "code_review"]
        },
        "limits": {
            "max_request_size": "10MB",
            "timeout": "30s"
        }
    }
    return json.dumps(config, indent=2)


# Prompts
@mcp.prompt()
async def greeting(name: str, formal: bool = False) -> List[PromptMessage]:
    """
    Generate a friendly greeting
    
    Args:
        name: Name of the person to greet
        formal: Whether to use formal language (default: false)
    """
    if formal:
        template = f"Good day, {name}. I hope this message finds you well."
    else:
        template = f"Hey {name}! How's it going?"
    
    return [
        PromptMessage(
            role="user",
            content=TextContent(type="text", text=template)
        )
    ]


@mcp.prompt()
async def summary(text: str, max_length: int = 100) -> List[PromptMessage]:
    """
    Generate a summary of the given text
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary (default: 100)
    """
    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"Please summarize the following text in no more than {max_length} characters:\n\n{text}"
            )
        )
    ]


@mcp.prompt()
async def code_review(code: str, language: str = "python", focus: str = "general") -> List[PromptMessage]:
    """
    Request a code review
    
    Args:
        code: Code to review
        language: Programming language (default: python)
        focus: Review focus area - general, security, performance, style (default: general)
    """
    focus_prompts = {
        "general": "Please review this code for correctness, readability, and best practices.",
        "security": "Please review this code with a focus on security vulnerabilities and potential risks.",
        "performance": "Please review this code with a focus on performance optimizations and efficiency.",
        "style": "Please review this code with a focus on code style, naming conventions, and organization."
    }
    
    prompt_text = focus_prompts.get(focus, focus_prompts["general"])
    
    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"{prompt_text}\n\nLanguage: {language}\n\n```{language}\n{code}\n```"
            )
        )
    ]


# Advanced tool with complex input validation
@mcp.tool()
async def process_data(
    ctx: Context,
    data: List[Dict[str, Any]],
    operation: str,
    options: Dict[str, Any] = None
) -> str:
    """
    Process a list of data items with specified operation
    
    Args:
        data: List of data items to process
        operation: Operation to perform (filter, transform, aggregate)
        options: Additional options for the operation
    """
    if options is None:
        options = {}
    
    try:
        if operation == "filter":
            key = options.get("key", "")
            value = options.get("value", "")
            filtered = [item for item in data if item.get(key) == value]
            return f"Filtered {len(filtered)} items out of {len(data)}"
        
        elif operation == "transform":
            transform_key = options.get("key", "")
            transform_type = options.get("type", "uppercase")
            
            for item in data:
                if transform_key in item and isinstance(item[transform_key], str):
                    if transform_type == "uppercase":
                        item[transform_key] = item[transform_key].upper()
                    elif transform_type == "lowercase":
                        item[transform_key] = item[transform_key].lower()
            
            return f"Transformed {len(data)} items"
        
        elif operation == "aggregate":
            agg_key = options.get("key", "")
            agg_type = options.get("type", "count")
            
            if agg_type == "count":
                result = len(data)
            elif agg_type == "sum" and agg_key:
                result = sum(item.get(agg_key, 0) for item in data if isinstance(item.get(agg_key), (int, float)))
            else:
                result = "Unknown aggregation type"
            
            return f"Aggregation result: {result}"
        
        else:
            return f"Unknown operation: {operation}"
            
    except Exception as e:
        return f"Error processing data: {str(e)}"


def main():
    """Run the MCP server"""
    import sys
    
    # Run the server using stdio transport
    asyncio.run(mcp.run(transport="stdio"))


if __name__ == "__main__":
    main()