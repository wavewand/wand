#!/usr/bin/env python3
"""
MCP Distributed System - Main Entry Point

Start the complete distributed MCP system with all services.
"""

import argparse
import logging
import sys
from pathlib import Path

from orchestrator.orchestrator import MCPSystemOrchestrator


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "mcp_system.log", mode='a')],
    )

    # Reduce noise from some libraries
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('grpc').setLevel(logging.WARNING)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MCP Distributed System - Multi-Agent Task Management Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Start with default settings
  %(prog)s --log-level DEBUG  # Start with debug logging
  %(prog)s --status          # Show system status and exit

Services Started:
  - Coordinator (gRPC): Task distribution and management
  - Integration Service (gRPC): External service integrations
  - Agent Services (gRPC): Specialized worker agents
  - REST API Gateway (HTTP): External API interface

Visit http://localhost:8000/docs for API documentation.
        """,
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)',
    )

    parser.add_argument(
        '--status', action='store_true', help='Show system status and exit (requires system to be running)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger("main")

    if args.status:
        # Show status and exit
        try:
            orchestrator = MCPSystemOrchestrator()
            orchestrator.print_system_status()
        except Exception as e:
            print(f"Error getting system status: {e}")
            sys.exit(1)
        return

    # Print banner
    print_banner()

    # Create and start orchestrator
    orchestrator = MCPSystemOrchestrator()

    try:
        logger.info("Starting MCP Distributed System...")
        orchestrator.start_system()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        sys.exit(1)


def print_banner():
    """Print startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                    MCP DISTRIBUTED SYSTEM v3.0.0                            ║
║                                                                               ║
║                Multi-Agent Context Protocol Platform                         ║
║                     with gRPC + REST Architecture                           ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Features:
• 7 Specialized Agent Types (Manager, Frontend, Backend, Database, DevOps, Integration, QA)
• gRPC-based Inter-Service Communication
• REST API Gateway for External Access
• Comprehensive Integration Suite (Slack, Git, AWS, Jenkins, YouTrack)
• Real-time Task Distribution and Monitoring
• Process Orchestration and Health Monitoring

"""
    print(banner)


if __name__ == "__main__":
    main()
