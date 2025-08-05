"""
Distributed MCP System - Core Components

This module provides the foundational components for the distributed MCP system:
- Agent types and management
- Task distribution and prioritization
- Inter-agent communication
- Persistence layer integration
"""

from .types import (
    Agent,
    AgentStatus,
    AgentType,
    DistributedAgent,
    Task,
    TaskPriority,
    TaskStatus,
    get_default_capabilities,
)

__all__ = [
    # Types
    "AgentType",
    "TaskStatus",
    "TaskPriority",
    "AgentStatus",
    "Task",
    "Agent",
    "DistributedAgent",
    "get_default_capabilities",
]
