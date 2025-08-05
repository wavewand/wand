#!/usr/bin/env python3
"""
MCP-Python Server Startup Script

Starts the MCP-Python server with the specified configuration.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('/app/logs/mcp-python.log', mode='a')],
    )


def main():
    parser = argparse.ArgumentParser(description='MCP-Python Server')
    parser.add_argument('--config', default='config.privileged.json', help='Configuration file')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--log-level', default='INFO', help='Logging level')

    args = parser.parse_args()

    # Setup logging
    os.makedirs('/app/logs', exist_ok=True)
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting MCP-Python server with config: {args.config}")

    # Set environment variables
    os.environ['HOST'] = args.host
    os.environ['PORT'] = str(args.port)
    os.environ['CONFIG_FILE'] = args.config

    # Import and run the distributed server
    try:
        from distributed_server import main as server_main

        server_main()
    except ImportError:
        logger.error("Failed to import distributed_server. Starting basic MCP server.")
        # Fallback to basic server
        import fastapi
        import uvicorn

        app = fastapi.FastAPI(title="MCP-Python Server", version="1.0.0")

        @app.get("/health")
        def health():
            return {"status": "healthy", "mode": "privileged"}

        @app.get("/")
        def root():
            return {"message": "MCP-Python Server Running", "mode": "privileged"}

        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
