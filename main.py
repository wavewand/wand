#!/usr/bin/env python3
"""Main entry point for running the Python MCP server with API."""

import asyncio
import argparse
import threading
import signal
import sys
import logging
from typing import Optional

from api.server import run_server as run_api_server


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_mcp_server():
    """Run the MCP server in a separate thread."""
    import subprocess
    try:
        # Run the MCP server as a subprocess
        proc = subprocess.Popen(
            [sys.executable, "server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info("MCP server started")
        proc.wait()
    except Exception as e:
        logger.error(f"MCP server error: {e}")


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    logger.info("Shutting down...")
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Python MCP Server with API")
    parser.add_argument("--api-host", default="0.0.0.0", help="API server host")
    parser.add_argument("--api-port", type=int, default=8000, help="API server port")
    parser.add_argument("--mcp-only", action="store_true", help="Run only MCP server")
    parser.add_argument("--api-only", action="store_true", help="Run only API server")
    
    args = parser.parse_args()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.api_only:
            # Run only API server
            logger.info(f"Starting API server on {args.api_host}:{args.api_port}")
            run_api_server(host=args.api_host, port=args.api_port)
        elif args.mcp_only:
            # Run only MCP server
            logger.info("Starting MCP server...")
            run_mcp_server()
        else:
            # Run both servers
            # Start MCP server in a thread
            mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
            mcp_thread.start()
            
            # Run API server in main thread
            logger.info(f"Starting API server on {args.api_host}:{args.api_port}")
            run_api_server(host=args.api_host, port=args.api_port)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()