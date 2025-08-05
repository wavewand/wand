"""
Structured Logging System

Provides comprehensive structured logging with correlation IDs, contextual information,
and multiple output formats for effective debugging and monitoring.
"""

import asyncio
import contextvars
import functools
import json
import logging
import logging.config
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from utils.error_handling import ErrorCategory, MCPError


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output formats."""

    JSON = "json"
    TEXT = "text"
    COLORED = "colored"


@dataclass
class LogConfig:
    """Logging configuration."""

    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.JSON
    output_file: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_correlation_id: bool = True
    include_caller_info: bool = True
    include_process_info: bool = True
    custom_fields: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_fields is None:
            self.custom_fields = {}


# Context variable for correlation ID
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default=None)


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records."""

    def filter(self, record):
        correlation_id = correlation_id_var.get()
        if correlation_id:
            record.correlation_id = correlation_id
        else:
            record.correlation_id = str(uuid.uuid4())
            correlation_id_var.set(record.correlation_id)
        return True


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured output."""

    def __init__(self, include_caller_info=True, include_process_info=True, custom_fields=None):
        super().__init__()
        self.include_caller_info = include_caller_info
        self.include_process_info = include_process_info
        self.custom_fields = custom_fields or {}

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, 'correlation_id', None),
        }

        # Add process information
        if self.include_process_info:
            log_entry.update(
                {
                    "process_id": record.process,
                    "thread_id": record.thread,
                    "thread_name": record.threadName,
                }
            )

        # Add caller information
        if self.include_caller_info:
            log_entry.update(
                {
                    "filename": record.filename,
                    "function": record.funcName,
                    "line_number": record.lineno,
                    "module": record.module,
                    "pathname": record.pathname,
                }
            )

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields from the log record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in [
                'name',
                'msg',
                'args',
                'levelname',
                'levelno',
                'pathname',
                'filename',
                'module',
                'lineno',
                'funcName',
                'created',
                'msecs',
                'relativeCreated',
                'thread',
                'threadName',
                'processName',
                'process',
                'message',
                'exc_info',
                'exc_text',
                'stack_info',
                'correlation_id',
            ] and not key.startswith('_'):
                extra_fields[key] = value

        if extra_fields:
            log_entry["extra"] = extra_fields

        # Add custom fields
        if self.custom_fields:
            log_entry.update(self.custom_fields)

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development."""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
        'ENDC': '\033[0m',  # End color
        'BOLD': '\033[1m',  # Bold
    }

    def __init__(self):
        super().__init__()
        self.format_string = (
            "{color}{bold}[{levelname:8}]{endc} "
            "{color}{timestamp}{endc} "
            "{bold}{logger}{endc}:{function}:{line} "
            "- {message}"
        )

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')

        formatted_message = self.format_string.format(
            color=color,
            bold=self.COLORS['BOLD'],
            endc=self.COLORS['ENDC'],
            levelname=record.levelname,
            timestamp=datetime.fromtimestamp(record.created).strftime('%H:%M:%S'),
            logger=record.name,
            function=record.funcName,
            line=record.lineno,
            message=record.getMessage(),
        )

        # Add correlation ID if available
        correlation_id = getattr(record, 'correlation_id', None)
        if correlation_id:
            formatted_message += f" {self.COLORS['BOLD']}(ID: {correlation_id[:8]}){self.COLORS['ENDC']}"

        # Add exception information
        if record.exc_info:
            formatted_message += "\n" + self.formatException(record.exc_info)

        return formatted_message


class StructuredLogger:
    """Enhanced structured logger with contextual information."""

    def __init__(self, name: str, config: LogConfig = None):
        self.name = name
        self.config = config or LogConfig()
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        """Setup logger with configuration."""
        self.logger.setLevel(getattr(logging, self.config.level.value))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Add correlation ID filter
        if self.config.enable_correlation_id:
            correlation_filter = CorrelationIdFilter()
            self.logger.addFilter(correlation_filter)

        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.level.value))

        if self.config.format == LogFormat.JSON:
            console_formatter = JSONFormatter(
                include_caller_info=self.config.include_caller_info,
                include_process_info=self.config.include_process_info,
                custom_fields=self.config.custom_fields,
            )
        elif self.config.format == LogFormat.COLORED:
            console_formatter = ColoredFormatter()
        else:  # TEXT
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Setup file handler if specified
        if self.config.output_file:
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                self.config.output_file, maxBytes=self.config.max_file_size, backupCount=self.config.backup_count
            )
            file_handler.setLevel(getattr(logging, self.config.level.value))

            # Always use JSON format for file output
            file_formatter = JSONFormatter(
                include_caller_info=self.config.include_caller_info,
                include_process_info=self.config.include_process_info,
                custom_fields=self.config.custom_fields,
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with additional context."""
        # Extract standard logging kwargs
        exc_info = kwargs.pop('exc_info', None)
        stack_info = kwargs.pop('stack_info', None)
        stacklevel = kwargs.pop('stacklevel', 1)

        # Add remaining kwargs as extra fields
        extra = kwargs if kwargs else None

        self.logger.log(
            level, message, exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel + 1, extra=extra
        )

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        kwargs['exc_info'] = True
        self.error(message, **kwargs)

    def log_request(self, method: str, path: str, status_code: int, response_time_ms: float, **kwargs):
        """Log HTTP request."""
        self.info(
            f"{method} {path} - {status_code}",
            method=method,
            path=path,
            status_code=status_code,
            response_time_ms=response_time_ms,
            request_type="http",
            **kwargs,
        )

    def log_database_query(self, query: str, duration_ms: float, affected_rows: int = None, **kwargs):
        """Log database query."""
        self.debug(
            f"Database query executed in {duration_ms:.2f}ms",
            query=query[:200] + "..." if len(query) > 200 else query,
            duration_ms=duration_ms,
            affected_rows=affected_rows,
            operation_type="database",
            **kwargs,
        )

    def log_framework_operation(self, framework: str, operation: str, success: bool, duration_ms: float, **kwargs):
        """Log AI framework operation."""
        level = logging.INFO if success else logging.ERROR
        status = "succeeded" if success else "failed"

        self._log_with_context(
            level,
            f"{framework} {operation} {status} in {duration_ms:.2f}ms",
            framework=framework,
            operation=operation,
            success=success,
            duration_ms=duration_ms,
            operation_type="framework",
            **kwargs,
        )

    def log_performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metrics."""
        self.info(
            f"Performance: {operation} completed in {duration_ms:.2f}ms",
            operation=operation,
            duration_ms=duration_ms,
            operation_type="performance",
            **kwargs,
        )

    def log_security_event(self, event_type: str, user_id: str = None, ip_address: str = None, **kwargs):
        """Log security-related events."""
        self.warning(
            f"Security event: {event_type}",
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            operation_type="security",
            **kwargs,
        )

    def log_business_event(self, event_type: str, entity_type: str, entity_id: str, **kwargs):
        """Log business logic events."""
        self.info(
            f"Business event: {event_type} for {entity_type} {entity_id}",
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            operation_type="business",
            **kwargs,
        )

    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current context."""
        correlation_id_var.set(correlation_id)

    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID."""
        return correlation_id_var.get()

    def with_fields(self, **fields) -> 'ContextualLogger':
        """Create a logger with additional fields."""
        return ContextualLogger(self, fields)


class ContextualLogger:
    """Logger wrapper that adds contextual fields to all log messages."""

    def __init__(self, logger: StructuredLogger, fields: Dict[str, Any]):
        self.logger = logger
        self.fields = fields

    def _merge_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge contextual fields with kwargs."""
        merged = self.fields.copy()
        merged.update(kwargs)
        return merged

    def debug(self, message: str, **kwargs):
        self.logger.debug(message, **self._merge_kwargs(kwargs))

    def info(self, message: str, **kwargs):
        self.logger.info(message, **self._merge_kwargs(kwargs))

    def warning(self, message: str, **kwargs):
        self.logger.warning(message, **self._merge_kwargs(kwargs))

    def error(self, message: str, **kwargs):
        self.logger.error(message, **self._merge_kwargs(kwargs))

    def critical(self, message: str, **kwargs):
        self.logger.critical(message, **self._merge_kwargs(kwargs))

    def exception(self, message: str, **kwargs):
        self.logger.exception(message, **self._merge_kwargs(kwargs))


class CorrelationIdMiddleware:
    """Middleware to handle correlation IDs in web requests."""

    def __init__(self, header_name: str = "X-Correlation-ID"):
        self.header_name = header_name

    async def __call__(self, request, call_next):
        """Process request with correlation ID."""
        # Get correlation ID from header or generate new one
        correlation_id = request.headers.get(self.header_name) or str(uuid.uuid4())

        # Set correlation ID in context
        correlation_id_var.set(correlation_id)

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers[self.header_name] = correlation_id

        return response


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}
_default_config: Optional[LogConfig] = None


def setup_logging(config: LogConfig):
    """Setup global logging configuration."""
    global _default_config
    _default_config = config

    # Configure root logger
    root_logger = StructuredLogger("root", config)
    _loggers["root"] = root_logger


def get_logger(name: str, config: LogConfig = None) -> StructuredLogger:
    """Get or create logger with given name."""
    if name not in _loggers:
        logger_config = config or _default_config or LogConfig()
        _loggers[name] = StructuredLogger(name, logger_config)

    return _loggers[name]


def log_execution_time(operation_name: str = None, logger_name: str = None):
    """Decorator to log function execution time."""

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(op_name, duration_ms, success=True)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(op_name, duration_ms, success=False, error=str(e))
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(op_name, duration_ms, success=True)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(op_name, duration_ms, success=False, error=str(e))
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def log_function_call(include_args: bool = False, include_result: bool = False, logger_name: str = None):
    """Decorator to log function calls."""

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)

            log_data = {"function": f"{func.__module__}.{func.__name__}", "operation_type": "function_call"}

            if include_args:
                log_data.update({"args": [str(arg) for arg in args], "kwargs": {k: str(v) for k, v in kwargs.items()}})

            logger.debug(f"Calling {func.__name__}", **log_data)

            try:
                result = await func(*args, **kwargs)

                if include_result:
                    log_data["result"] = str(result)[:200]  # Truncate long results

                logger.debug(f"Completed {func.__name__}", **log_data)
                return result

            except Exception as e:
                log_data["error"] = str(e)
                logger.error(f"Failed {func.__name__}", **log_data)
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)

            log_data = {"function": f"{func.__module__}.{func.__name__}", "operation_type": "function_call"}

            if include_args:
                log_data.update({"args": [str(arg) for arg in args], "kwargs": {k: str(v) for k, v in kwargs.items()}})

            logger.debug(f"Calling {func.__name__}", **log_data)

            try:
                result = func(*args, **kwargs)

                if include_result:
                    log_data["result"] = str(result)[:200]  # Truncate long results

                logger.debug(f"Completed {func.__name__}", **log_data)
                return result

            except Exception as e:
                log_data["error"] = str(e)
                logger.error(f"Failed {func.__name__}", **log_data)
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
