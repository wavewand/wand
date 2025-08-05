"""
Enhanced System-Level Logging Infrastructure for MCP Wand

Provides comprehensive system-level logging with:
- Automatic system log directory detection with fallback
- Granular MCP request/response logging
- Tool call tracking with inputs/outputs
- Performance metrics and audit trails
- Safe stdio integration for MCP protocol
"""

import contextvars
import functools
import hashlib
import json
import logging
import logging.handlers
import os
import platform
import stat
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Context variable for correlation ID tracking
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default='')


class LogLevel(str, Enum):
    """Enhanced log levels for MCP operations."""

    TRACE = "TRACE"  # Ultra-detailed tracing
    DEBUG = "DEBUG"  # Debug information
    INFO = "INFO"  # General information
    AUDIT = "AUDIT"  # Audit trail events
    WARNING = "WARNING"  # Warning messages
    ERROR = "ERROR"  # Error conditions
    CRITICAL = "CRITICAL"  # Critical failures
    SECURITY = "SECURITY"  # Security events


class LogCategory(str, Enum):
    """Log categories for enhanced filtering and routing."""

    MCP_PROTOCOL = "mcp_protocol"  # MCP request/response
    TOOL_EXECUTION = "tool_execution"  # Tool calls and results
    AGENT_LIFECYCLE = "agent_lifecycle"  # Agent management
    SECURITY = "security"  # Security events
    PERFORMANCE = "performance"  # Performance metrics
    AUDIT = "audit"  # Audit trail
    SYSTEM = "system"  # System events
    INTEGRATION = "integration"  # External integrations
    ERROR = "error"  # Error conditions


@dataclass
class SystemLogConfig:
    """Enhanced logging configuration with system-level directories."""

    # Base configuration
    level: LogLevel = LogLevel.INFO
    enable_stdio_safety: bool = True
    correlation_tracking: bool = True

    # System directory configuration
    use_system_logs: bool = True
    system_log_base: Optional[str] = None  # Will be auto-detected

    # Log file organization
    use_single_log_file: bool = True  # True = single wand.log, False = categorized files

    # Log file configuration
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    backup_count: int = 10
    compress_backups: bool = True

    # Granular logging features
    log_mcp_requests: bool = True
    log_tool_calls: bool = True
    log_tool_inputs: bool = True
    log_tool_outputs: bool = True
    log_performance_metrics: bool = True

    # Security and audit
    log_security_events: bool = True
    log_audit_trail: bool = True
    mask_sensitive_data: bool = True

    # Output format
    use_json_format: bool = True
    include_caller_info: bool = True
    include_process_info: bool = True


class SystemLogDirectoryManager:
    """Manages system-level log directory detection and creation."""

    SYSTEM_LOG_CANDIDATES = ["/var/log/wand", "/usr/local/var/log/wand"]  # Linux/Unix standard  # MacOS Homebrew style

    def __init__(self, config: SystemLogConfig):
        self.config = config
        self.detected_path: Optional[Path] = None
        self.fallback_path: Optional[Path] = None

    def detect_system_log_directory(self) -> Path:
        """
        Detect the appropriate system log directory with fallback.

        Returns:
            Path: The logging directory to use
        """
        if self.config.system_log_base:
            # Use explicitly configured path
            custom_path = Path(self.config.system_log_base)
            if self._test_directory_writable(custom_path):
                self.detected_path = custom_path
                return custom_path

        # Try system-level candidates
        for candidate in self.SYSTEM_LOG_CANDIDATES:
            candidate_path = Path(candidate)
            if self._test_directory_writable(candidate_path):
                self.detected_path = candidate_path
                return candidate_path

        # Fallback to tmp directory with warning
        tmp_logs = Path("/tmp/wand")
        try:
            tmp_logs.mkdir(exist_ok=True)
            self.fallback_path = tmp_logs

            # Create a basic logger to warn about fallback (before full system is initialized)
            fallback_logger = logging.getLogger("wand.system.fallback")
            fallback_logger.setLevel(logging.WARNING)
            if not fallback_logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                fallback_logger.addHandler(handler)

            fallback_logger.warning(
                f"SYSTEM LOGGING FALLBACK: No writable system log directory found. "
                f"Falling back to temporary directory: {tmp_logs}. "
                f"For production use, ensure system directories are available: "
                f"{', '.join(self.SYSTEM_LOG_CANDIDATES)}"
            )

            return tmp_logs

        except Exception as e:
            raise PermissionError(
                f"No writable system log directory found and tmp fallback failed: {e}. "
                f"Attempted directories: {', '.join(self.SYSTEM_LOG_CANDIDATES)}, /tmp/wand"
            )

    def _test_directory_writable(self, path: Path) -> bool:
        """Test if a directory exists or can be created and is writable."""
        try:
            # Create directory if it doesn't exist
            path.mkdir(parents=True, exist_ok=True)

            # Test write permissions
            test_file = path / ".write_test"
            test_file.write_text("test")
            test_file.unlink()

            return True
        except (OSError, PermissionError):
            return False


class MCPProtocolLogger:
    """Specialized logger for MCP protocol messages."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.sensitive_fields = {'password', 'token', 'key', 'secret', 'auth', 'credential'}

    def log_mcp_request(self, method: str, params: Dict[str, Any], correlation_id: str, stdio_mode: bool = False):
        """Log an incoming MCP request."""
        masked_params = self._mask_sensitive_data(params)

        log_entry = {
            "event": "mcp_request",
            "method": method,
            "params": masked_params,
            "correlation_id": correlation_id,
            "stdio_mode": stdio_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "inbound",
        }

        self.logger.info(f"MCP_REQUEST: {method}", extra={"category": LogCategory.MCP_PROTOCOL, "mcp_data": log_entry})

    def log_mcp_response(
        self,
        method: str,
        result: Any,
        error: Optional[Dict] = None,
        correlation_id: str = None,
        execution_time_ms: float = None,
    ):
        """Log an MCP response."""
        masked_result = self._mask_sensitive_data(result) if result else None

        log_entry = {
            "event": "mcp_response",
            "method": method,
            "result": masked_result,
            "error": error,
            "correlation_id": correlation_id,
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "outbound",
            "success": error is None,
        }

        level = logging.ERROR if error else logging.INFO
        message = f"MCP_RESPONSE: {method} ({'ERROR' if error else 'SUCCESS'})"

        self.logger.log(level, message, extra={"category": LogCategory.MCP_PROTOCOL, "mcp_data": log_entry})

    def log_tool_call(self, tool_name: str, arguments: Dict[str, Any], correlation_id: str = None):
        """Log a tool execution call."""
        masked_args = self._mask_sensitive_data(arguments)

        log_entry = {
            "event": "tool_call",
            "tool_name": tool_name,
            "arguments": masked_args,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.logger.info(
            f"TOOL_CALL: {tool_name}", extra={"category": LogCategory.TOOL_EXECUTION, "tool_data": log_entry}
        )

    def log_tool_result(
        self,
        tool_name: str,
        result: Any,
        error: Optional[str] = None,
        execution_time_ms: float = None,
        correlation_id: str = None,
    ):
        """Log a tool execution result."""
        masked_result = self._mask_sensitive_data(result) if result else None

        log_entry = {
            "event": "tool_result",
            "tool_name": tool_name,
            "result": masked_result,
            "error": error,
            "execution_time_ms": execution_time_ms,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": error is None,
        }

        level = logging.ERROR if error else logging.INFO
        message = f"TOOL_RESULT: {tool_name} ({'ERROR' if error else 'SUCCESS'})"

        self.logger.log(level, message, extra={"category": LogCategory.TOOL_EXECUTION, "tool_data": log_entry})

    def _mask_sensitive_data(self, data: Any) -> Any:
        """Mask sensitive information in log data."""
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                    masked[key] = "*" * 8
                else:
                    masked[key] = self._mask_sensitive_data(value)
            return masked
        elif isinstance(data, list):
            return [self._mask_sensitive_data(item) for item in data]
        else:
            return data


class EnhancedJSONFormatter(logging.Formatter):
    """Enhanced JSON formatter with MCP-specific fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with enhanced fields."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "process_id": os.getpid(),
            "thread_id": threading.current_thread().ident,
        }

        # Add correlation ID if available
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        # Add category if specified
        if hasattr(record, 'category'):
            log_entry["category"] = record.category

        # Add MCP-specific data
        if hasattr(record, 'mcp_data'):
            log_entry["mcp_data"] = record.mcp_data

        if hasattr(record, 'tool_data'):
            log_entry["tool_data"] = record.tool_data

        # Add caller information
        if hasattr(record, 'pathname'):
            log_entry["source"] = {"file": record.pathname, "line": record.lineno, "function": record.funcName}

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_entry, default=str, separators=(',', ':'))


class EnhancedLoggingSystem:
    """Main enhanced logging system with system-level directory support."""

    def __init__(self, config: SystemLogConfig = None):
        self.config = config or SystemLogConfig()
        self.directory_manager = SystemLogDirectoryManager(self.config)
        self.log_directory: Optional[Path] = None
        self.loggers: Dict[str, logging.Logger] = {}
        self.mcp_logger: Optional[MCPProtocolLogger] = None
        self._lock = threading.Lock()

        # Detect if running in stdio mode for MCP safety
        self.stdio_mode = len(sys.argv) > 1 and sys.argv[1] == "stdio"

        self._setup_logging_system()

    def _setup_logging_system(self):
        """Setup the complete logging system."""
        # Detect log directory
        self.log_directory = self.directory_manager.detect_system_log_directory()

        # Setup different log files for different categories
        self._setup_category_loggers()

        # Setup MCP protocol logger
        self._setup_mcp_logger()

        # Configure root logger
        self._configure_root_logger()

    def _setup_category_loggers(self):
        """Setup loggers based on configuration: single file or categorized files."""
        if self.config.use_single_log_file:
            # Single log file mode
            single_log_file = self.log_directory / "wand.log"

            main_logger = self._create_logger(name="wand.main", log_file=single_log_file)

            # Map all categories to the same logger
            for category in LogCategory:
                self.loggers[category] = main_logger
        else:
            # Categorized log files mode
            log_files = {
                LogCategory.MCP_PROTOCOL: "mcp_protocol.log",
                LogCategory.TOOL_EXECUTION: "tool_execution.log",
                LogCategory.SECURITY: "security.log",
                LogCategory.AUDIT: "audit.log",
                LogCategory.PERFORMANCE: "performance.log",
                LogCategory.SYSTEM: "system.log",
                LogCategory.INTEGRATION: "integration.log",
                LogCategory.ERROR: "errors.log",
            }

            for category, filename in log_files.items():
                logger = self._create_logger(name=f"wand.{category}", log_file=self.log_directory / filename)
                self.loggers[category] = logger

    def _setup_mcp_logger(self):
        """Setup the specialized MCP protocol logger."""
        mcp_logger = self.loggers[LogCategory.MCP_PROTOCOL]
        self.mcp_logger = MCPProtocolLogger(mcp_logger)

    def _configure_root_logger(self):
        """Configure the root logger safely for stdio mode."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.level.value))

        # Choose log file based on single vs categorized mode
        if self.config.use_single_log_file:
            main_log_file = self.log_directory / "wand.log"
        else:
            main_log_file = self.log_directory / "wand_main.log"

        if self.stdio_mode and self.config.enable_stdio_safety:
            # In stdio mode, only log to files to avoid interfering with MCP protocol
            handler = self._create_rotating_handler(main_log_file)
            root_logger.addHandler(handler)
        else:
            # Normal mode - can use console logging
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(EnhancedJSONFormatter())
            root_logger.addHandler(console_handler)

            # Also add file handler
            file_handler = self._create_rotating_handler(main_log_file)
            root_logger.addHandler(file_handler)

    def _create_logger(self, name: str, log_file: Path) -> logging.Logger:
        """Create a logger with rotating file handler."""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, self.config.level.value))

        # Add rotating file handler
        handler = self._create_rotating_handler(log_file)
        logger.addHandler(handler)

        # Prevent propagation to root logger to avoid duplication
        logger.propagate = False

        return logger

    def _create_rotating_handler(self, log_file: Path) -> logging.handlers.RotatingFileHandler:
        """Create a rotating file handler with enhanced configuration."""
        handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file), maxBytes=self.config.max_file_size, backupCount=self.config.backup_count
        )

        handler.setFormatter(EnhancedJSONFormatter())
        return handler

    def get_logger(self, category: LogCategory) -> logging.Logger:
        """Get a logger for a specific category."""
        return self.loggers.get(category, logging.getLogger())

    def get_mcp_logger(self) -> MCPProtocolLogger:
        """Get the MCP protocol logger."""
        return self.mcp_logger

    def log_system_startup(self):
        """Log system startup information."""
        system_logger = self.get_logger(LogCategory.SYSTEM)

        startup_info = {
            "event": "system_startup",
            "platform": platform.platform(),
            "python_version": sys.version,
            "log_directory": str(self.log_directory),
            "stdio_mode": self.stdio_mode,
            "config": asdict(self.config),
        }

        system_logger.info(
            "Wand MCP system starting up", extra={"category": LogCategory.SYSTEM, "system_data": startup_info}
        )


# Decorators for enhanced logging
def log_mcp_call(func):
    """Decorator to automatically log MCP method calls."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)

        # Extract method name and params
        method = kwargs.get('method', func.__name__)
        params = kwargs.get('params', {})

        # Get MCP logger from global system (would need to be initialized)
        if hasattr(logging, '_enhanced_system'):
            mcp_logger = logging._enhanced_system.get_mcp_logger()
            mcp_logger.log_mcp_request(method, params, correlation_id)

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000

            if hasattr(logging, '_enhanced_system'):
                mcp_logger.log_mcp_response(method, result, None, correlation_id, execution_time)

            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error = {"type": type(e).__name__, "message": str(e)}

            if hasattr(logging, '_enhanced_system'):
                mcp_logger.log_mcp_response(method, None, error, correlation_id, execution_time)

            raise

    return wrapper


def log_tool_execution(func):
    """Decorator to automatically log tool executions."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = kwargs.get('tool_name', func.__name__)
        arguments = kwargs.get('arguments', {})
        correlation_id = correlation_id_var.get() or str(uuid.uuid4())

        # Log tool call
        if hasattr(logging, '_enhanced_system'):
            mcp_logger = logging._enhanced_system.get_mcp_logger()
            mcp_logger.log_tool_call(tool_name, arguments, correlation_id)

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000

            if hasattr(logging, '_enhanced_system'):
                mcp_logger.log_tool_result(tool_name, result, None, execution_time, correlation_id)

            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            if hasattr(logging, '_enhanced_system'):
                mcp_logger.log_tool_result(tool_name, None, str(e), execution_time, correlation_id)

            raise

    return wrapper


# Global system instance
_enhanced_system: Optional[EnhancedLoggingSystem] = None


def initialize_enhanced_logging(config: SystemLogConfig = None) -> EnhancedLoggingSystem:
    """Initialize the enhanced logging system globally."""
    global _enhanced_system

    if _enhanced_system is None:
        _enhanced_system = EnhancedLoggingSystem(config)
        # Attach to logging module for decorator access
        logging._enhanced_system = _enhanced_system

        # Log system startup
        _enhanced_system.log_system_startup()

    return _enhanced_system


def get_enhanced_logger(category: LogCategory) -> logging.Logger:
    """Get an enhanced logger for a specific category."""
    if _enhanced_system is None:
        initialize_enhanced_logging()

    return _enhanced_system.get_logger(category)


def get_mcp_logger() -> MCPProtocolLogger:
    """Get the MCP protocol logger."""
    if _enhanced_system is None:
        initialize_enhanced_logging()

    return _enhanced_system.get_mcp_logger()
