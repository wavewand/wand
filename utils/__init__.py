"""
Utilities Package

Provides common utilities for error handling, security, logging, and configuration management.
"""

from .error_handling import (
    APIError,
    AuthenticationError,
    CircuitBreaker,
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ErrorSeverity,
    FrameworkError,
    GracefulDegradation,
    MCPError,
    RateLimitError,
    RetryMechanism,
    error_handler,
    handle_errors,
    with_circuit_breaker,
    with_retry,
)

__all__ = [
    'MCPError',
    'FrameworkError',
    'APIError',
    'AuthenticationError',
    'RateLimitError',
    'ErrorCategory',
    'ErrorSeverity',
    'ErrorContext',
    'CircuitBreaker',
    'RetryMechanism',
    'ErrorHandler',
    'error_handler',
    'with_circuit_breaker',
    'with_retry',
    'handle_errors',
    'GracefulDegradation',
]
