"""
Observability Package

Provides enhanced logging infrastructure with system-level directories,
granular MCP request logging, and comprehensive log management.
"""

# Import original logging for backward compatibility
try:
    from .logging import CorrelationIdMiddleware, LogConfig, LogFormat
    from .logging import LogLevel as OriginalLogLevel
    from .logging import StructuredLogger
    from .logging import get_logger as get_original_logger
    from .logging import log_execution_time, log_function_call, setup_logging
except ImportError:
    # If original logging not available, provide minimal compatibility
    pass

# Import enhanced logging system
from .enhanced_logging import (
    EnhancedLoggingSystem,
    LogCategory,
    LogLevel,
    MCPProtocolLogger,
    SystemLogConfig,
    get_enhanced_logger,
    get_mcp_logger,
    initialize_enhanced_logging,
    log_mcp_call,
    log_tool_execution,
)
from .log_management import (
    CompressionFormat,
    LogHealthMonitor,
    LogManagementConfig,
    LogManagementSystem,
    LogRetentionManager,
    RetentionPolicy,
)

__all__ = [
    # Enhanced logging system
    'initialize_enhanced_logging',
    'SystemLogConfig',
    'LogLevel',
    'LogCategory',
    'get_enhanced_logger',
    'get_mcp_logger',
    'log_mcp_call',
    'log_tool_execution',
    'EnhancedLoggingSystem',
    'MCPProtocolLogger',
    # Log management
    'LogManagementSystem',
    'LogManagementConfig',
    'LogHealthMonitor',
    'LogRetentionManager',
    'RetentionPolicy',
    'CompressionFormat',
]
