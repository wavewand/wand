#!/usr/bin/env python3
"""
Simple MCP server for OpenCode testing
"""

import asyncio
import subprocess

from mcp.server import FastMCP

# Create a simple MCP server
mcp = FastMCP("simple-mcp-test")


@mcp.tool()
def execute_command(command: str) -> str:
    """Execute a shell command and return the output"""
    import sys

    print(f"DEBUG: Executing command: {command}", file=sys.stderr)

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)

        print(f"DEBUG: Command result - return code: {result.returncode}", file=sys.stderr)
        print(f"DEBUG: stdout length: {len(result.stdout)}", file=sys.stderr)

        if result.returncode == 0:
            output = f"Command succeeded:\n{result.stdout}"
            print(f"DEBUG: Returning output length: {len(output)}", file=sys.stderr)
            return output
        else:
            error_output = f"Command failed (exit code {result.returncode}):\n{result.stderr}"
            print(f"DEBUG: Returning error: {error_output}", file=sys.stderr)
            return error_output

    except subprocess.TimeoutExpired:
        print("DEBUG: Command timed out", file=sys.stderr)
        return "Command timed out after 30 seconds"
    except Exception as e:
        error_msg = f"Error executing command: {str(e)}"
        print(f"DEBUG: Exception: {error_msg}", file=sys.stderr)
        return error_msg


@mcp.tool()
def list_directory(path: str = ".") -> str:
    """List contents of a directory"""
    try:
        result = subprocess.run(["ls", "-la", path], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            return f"Directory listing for {path}:\n{result.stdout}"
        else:
            return f"Failed to list directory {path}: {result.stderr}"

    except Exception as e:
        return f"Error listing directory: {str(e)}"


@mcp.tool()
def get_current_directory() -> str:
    """Get the current working directory"""
    try:
        result = subprocess.run(["pwd"], capture_output=True, text=True, timeout=5)
        return f"Current directory: {result.stdout.strip()}"
    except Exception as e:
        return f"Error getting current directory: {str(e)}"


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="stdio")
