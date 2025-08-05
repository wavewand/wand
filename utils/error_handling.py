"""
Error Handling and Resilience Patterns

Provides comprehensive error handling, retry mechanisms, circuit breakers,
and graceful degradation for the multi-framework AI platform.
"""

import asyncio
import functools
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification."""

    FRAMEWORK_ERROR = "framework_error"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    CONFIGURATION_ERROR = "configuration_error"
    DATABASE_ERROR = "database_error"
    CACHE_ERROR = "cache_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ErrorContext:
    """Comprehensive error context information."""

    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    framework: Optional[str] = None
    operation: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    stack_trace: Optional[str] = None
    retry_count: int = 0
    recovery_suggestions: List[str] = field(default_factory=list)


class MCPError(Exception):
    """Base exception class for MCP platform errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Dict[str, Any] = None,
        framework: str = None,
        operation: str = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.framework = framework
        self.operation = operation
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        self.error_id = f"{category.value}_{int(time.time() * 1000)}"


class FrameworkError(MCPError):
    """Framework-specific errors."""

    def __init__(self, message: str, framework: str, **kwargs):
        super().__init__(message, category=ErrorCategory.FRAMEWORK_ERROR, framework=framework, **kwargs)


class APIError(MCPError):
    """API-related errors."""

    def __init__(self, message: str, status_code: int = 500, **kwargs):
        super().__init__(message, category=ErrorCategory.API_ERROR, details={"status_code": status_code}, **kwargs)
        self.status_code = status_code


class AuthenticationError(MCPError):
    """Authentication and authorization errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION_ERROR,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs,
        )


class RateLimitError(MCPError):
    """Rate limiting errors."""

    def __init__(self, message: str, retry_after: int = None, **kwargs):
        details = kwargs.get("details", {})
        if retry_after:
            details["retry_after"] = retry_after

        super().__init__(message, category=ErrorCategory.RATE_LIMIT_ERROR, details=details, **kwargs)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    expected_exception: tuple = (Exception,)
    name: str = "default"


class CircuitBreaker:
    """Circuit breaker implementation for resilience."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self.logger = logging.getLogger(__name__)

    def __call__(self, func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                if (datetime.now() - self.last_failure_time).seconds > self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info(f"Circuit breaker {self.config.name} moved to HALF_OPEN")
                else:
                    raise MCPError(
                        f"Circuit breaker {self.config.name} is OPEN",
                        category=ErrorCategory.FRAMEWORK_ERROR,
                        severity=ErrorSeverity.HIGH,
                        details={"circuit_state": self.state.value},
                    )

            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.config.expected_exception as e:
                self._on_failure()
                raise

        return wrapper

    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Require 3 successes to close
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.logger.info(f"Circuit breaker {self.config.name} moved to CLOSED")
        else:
            self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.success_count = 0
            self.logger.error(f"Circuit breaker {self.config.name} moved to OPEN")


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)
    non_retryable_exceptions: tuple = (AuthenticationError,)


class RetryMechanism:
    """Exponential backoff retry mechanism."""

    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def __call__(self, func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(self.config.max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        self.logger.info(f"Operation succeeded on attempt {attempt + 1}")
                    return result

                except self.config.non_retryable_exceptions as e:
                    self.logger.error(f"Non-retryable error: {e}")
                    raise

                except self.config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == self.config.max_attempts - 1:
                        self.logger.error(f"All retry attempts failed: {e}")
                        if isinstance(e, MCPError):
                            e.retry_count = attempt + 1
                        raise

                    delay = self._calculate_delay(attempt)
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)

            raise last_exception

        return wrapper

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = min(self.config.base_delay * (self.config.exponential_base**attempt), self.config.max_delay)

        if self.config.jitter:
            import random

            delay *= 0.5 + random.random() * 0.5  # Add 0-50% jitter

        return delay


class ErrorHandler:
    """Centralized error handling and reporting."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history: List[ErrorContext] = []
        self.max_history_size = 1000

        # Error statistics
        self.error_counts: Dict[ErrorCategory, int] = {}
        self.framework_errors: Dict[str, int] = {}

    def handle_error(self, error: Exception, context: Dict[str, Any] = None, request_id: str = None) -> ErrorContext:
        """Handle and log an error with full context."""

        # Create error context
        if isinstance(error, MCPError):
            # Merge provided context with error details
            merged_details = {**(error.details or {}), **(context or {})}
            error_context = ErrorContext(
                error_id=error.error_id,
                timestamp=error.timestamp,
                category=error.category,
                severity=error.severity,
                message=error.message,
                details=merged_details,
                framework=error.framework,
                operation=error.operation,
                request_id=request_id,
                stack_trace=traceback.format_exc(),
                retry_count=getattr(error, 'retry_count', 0),
            )
        else:
            error_context = ErrorContext(
                error_id=f"error_{int(time.time() * 1000)}",
                timestamp=datetime.now(),
                category=ErrorCategory.UNKNOWN_ERROR,
                severity=ErrorSeverity.MEDIUM,
                message=str(error),
                details=context or {},
                request_id=request_id,
                stack_trace=traceback.format_exc(),
            )

        # Add recovery suggestions
        error_context.recovery_suggestions = self._get_recovery_suggestions(error_context)

        # Log the error
        self._log_error(error_context)

        # Update statistics
        self._update_statistics(error_context)

        # Store in history
        self._store_error(error_context)

        # Send alerts for critical errors
        if error_context.severity == ErrorSeverity.CRITICAL:
            self._send_alert(error_context)

        return error_context

    def _get_recovery_suggestions(self, error_context: ErrorContext) -> List[str]:
        """Generate recovery suggestions based on error type."""
        suggestions = []

        if error_context.category == ErrorCategory.FRAMEWORK_ERROR:
            suggestions.extend(
                [
                    "Check framework availability and configuration",
                    "Verify API keys and authentication",
                    "Try switching to alternative framework",
                    "Check network connectivity",
                ]
            )
        elif error_context.category == ErrorCategory.RATE_LIMIT_ERROR:
            suggestions.extend(
                [
                    "Reduce request frequency",
                    "Implement request queuing",
                    "Use batch processing where possible",
                    "Check rate limit configuration",
                ]
            )
        elif error_context.category == ErrorCategory.AUTHENTICATION_ERROR:
            suggestions.extend(
                [
                    "Verify API credentials",
                    "Check token expiration",
                    "Refresh authentication tokens",
                    "Contact administrator for access",
                ]
            )
        elif error_context.category == ErrorCategory.TIMEOUT_ERROR:
            suggestions.extend(
                [
                    "Increase timeout values",
                    "Check network connectivity",
                    "Optimize query complexity",
                    "Use async processing for long operations",
                ]
            )

        return suggestions

    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level."""
        log_data = {
            "error_id": error_context.error_id,
            "category": error_context.category.value,
            "severity": error_context.severity.value,
            "framework": error_context.framework,
            "operation": error_context.operation,
            "request_id": error_context.request_id,
            "details": error_context.details,
        }

        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {error_context.message}", extra=log_data)
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY: {error_context.message}", extra=log_data)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY: {error_context.message}", extra=log_data)
        else:
            self.logger.info(f"LOW SEVERITY: {error_context.message}", extra=log_data)

    def _update_statistics(self, error_context: ErrorContext):
        """Update error statistics."""
        # Category statistics
        self.error_counts[error_context.category] = self.error_counts.get(error_context.category, 0) + 1

        # Framework statistics
        if error_context.framework:
            self.framework_errors[error_context.framework] = self.framework_errors.get(error_context.framework, 0) + 1

    def _store_error(self, error_context: ErrorContext):
        """Store error in history."""
        self.error_history.append(error_context)

        # Maintain history size
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size :]

    def _send_alert(self, error_context: ErrorContext):
        """Send alert for critical errors."""
        # This would integrate with alerting systems (Slack, PagerDuty, etc.)
        self.logger.critical(f"ALERT: Critical error occurred - {error_context.error_id}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        total_errors = len(self.error_history)
        recent_errors = [e for e in self.error_history if e.timestamp > datetime.now() - timedelta(hours=24)]

        return {
            "total_errors": total_errors,
            "recent_errors_24h": len(recent_errors),
            "error_rate_24h": len(recent_errors) / 24 if recent_errors else 0,
            "errors_by_category": dict(self.error_counts),
            "errors_by_framework": dict(self.framework_errors),
            "most_common_category": max(self.error_counts, key=self.error_counts.get) if self.error_counts else None,
            "error_severity_distribution": self._get_severity_distribution(),
        }

    def _get_severity_distribution(self) -> Dict[str, int]:
        """Get distribution of error severities."""
        distribution = {severity.value: 0 for severity in ErrorSeverity}

        for error in self.error_history:
            distribution[error.severity.value] += 1

        return distribution

    def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent errors for monitoring dashboard."""
        recent_errors = sorted(self.error_history[-limit:], key=lambda e: e.timestamp, reverse=True)

        return [
            {
                "error_id": error.error_id,
                "timestamp": error.timestamp.isoformat(),
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "framework": error.framework,
                "operation": error.operation,
                "request_id": error.request_id,
                "retry_count": error.retry_count,
                "recovery_suggestions": error.recovery_suggestions,
            }
            for error in recent_errors
        ]


# Global error handler instance
error_handler = ErrorHandler()


# Decorator functions for easy use
def with_circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 60, name: str = "default"):
    """Decorator to add circuit breaker protection."""
    config = CircuitBreakerConfig(failure_threshold=failure_threshold, recovery_timeout=recovery_timeout, name=name)
    return CircuitBreaker(config)


def with_retry(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator to add retry mechanism."""
    config = RetryConfig(max_attempts=max_attempts, base_delay=base_delay, max_delay=max_delay)
    return RetryMechanism(config)


def handle_errors(func):
    """Decorator to automatically handle and log errors."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_context = error_handler.handle_error(e)

            # Re-raise as MCPError if it's not already
            if not isinstance(e, MCPError):
                raise MCPError(
                    str(e), category=ErrorCategory.UNKNOWN_ERROR, details={"original_error": type(e).__name__}
                ) from e
            raise

    return wrapper


# Graceful degradation helpers
class GracefulDegradation:
    """Utilities for graceful degradation."""

    @staticmethod
    async def fallback_to_cache(primary_func, cache_func, *args, **kwargs):
        """Try primary function, fallback to cache on failure."""
        try:
            return await primary_func(*args, **kwargs)
        except Exception as e:
            error_handler.handle_error(e, context={"fallback": "cache"})
            return await cache_func(*args, **kwargs)

    @staticmethod
    async def fallback_to_alternative_framework(primary_framework_func, alternative_framework_func, *args, **kwargs):
        """Try primary framework, fallback to alternative on failure."""
        try:
            return await primary_framework_func(*args, **kwargs)
        except Exception as e:
            error_handler.handle_error(e, context={"fallback": "alternative_framework"})
            return await alternative_framework_func(*args, **kwargs)

    @staticmethod
    def default_response(operation: str, error: Exception) -> Dict[str, Any]:
        """Generate a default response when all else fails."""
        return {
            "success": False,
            "error": "Service temporarily unavailable",
            "operation": operation,
            "error_id": getattr(error, 'error_id', 'unknown'),
            "message": "Please try again later or contact support",
            "fallback": True,
        }
