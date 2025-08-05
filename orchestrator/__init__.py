"""
Process Orchestration System for the distributed MCP system.

Manages the lifecycle of all system processes including agents, coordinator,
integration services, and REST API gateway.
"""

from .orchestrator import MCPSystemOrchestrator

__all__ = [
    "MCPSystemOrchestrator",
]
