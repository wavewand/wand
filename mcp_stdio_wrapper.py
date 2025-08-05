#!/usr/bin/env python3
"""
MCP STDIO Wrapper for HTTP Server
Provides a stdio interface that proxies to the HTTP MCP server
"""

import json
import logging
import sys

import requests

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='mcp_stdio_wrapper.log'
)
logger = logging.getLogger(__name__)

MCP_SERVER_URL = "http://localhost:8001/mcp"


def handle_request(request):
    """Forward stdio request to HTTP server"""
    try:
        logger.debug(f"Received request: {request}")

        # Forward to HTTP server
        response = requests.post(MCP_SERVER_URL, json=request)
        response.raise_for_status()

        result = response.json()
        logger.debug(f"Got response: {result}")
        return result

    except Exception as e:
        logger.error(f"Error handling request: {e}")
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
        }


def main():
    """Main stdio loop"""
    logger.info("Starting MCP STDIO wrapper")

    while True:
        try:
            # Read line from stdin
            line = sys.stdin.readline()
            if not line:
                break

            # Parse JSON request
            request = json.loads(line.strip())

            # Handle request
            response = handle_request(request)

            # Write response to stdout
            sys.stdout.write(json.dumps(response) + '\n')
            sys.stdout.flush()

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
            }
            sys.stdout.write(json.dumps(error_response) + '\n')
            sys.stdout.flush()


if __name__ == "__main__":
    main()
