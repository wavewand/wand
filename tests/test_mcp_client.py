#!/usr/bin/env python3
"""
Test MCP server using proper MCP client
"""
import asyncio
import subprocess
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """Test MCP server using proper MCP client"""
    print("=== Testing MCP Server with MCP Client ===\n")

    # Server parameters
    server_params = StdioServerParameters(command=sys.executable, args=["distributed_server.py"], env=None)

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print("✓ Connected to MCP server")

                # Initialize the session
                await session.initialize()
                print("✓ Session initialized")

                # List available tools
                tools = await session.list_tools()
                print(f"✓ Found {len(tools.tools)} tools:")
                for tool in tools.tools:
                    print(f"  - {tool.name}: {tool.description}")

                # Test execute_command tool if available
                if any(tool.name == "execute_command" for tool in tools.tools):
                    print("\n• Testing execute_command tool...")
                    result = await session.call_tool("execute_command", arguments={"command": "ls -la ."})
                    print("✓ Command executed successfully:")
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print(f"  Output: {content.text[:200]}...")

                return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1)
