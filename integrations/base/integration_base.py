"""
Base integration class providing common functionality for all Wand integrations
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from .auth_manager import AuthManager
from .cache_manager import CacheManager
from .error_handler import ErrorHandler
from .error_types import (
    EnhancedError,
    ErrorCategory,
    ErrorSeverity,
    create_enhanced_error_response,
    create_integration_error,
)
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class BaseIntegration(ABC):
    """
    Base class for all Wand integrations providing:
    - Authentication management
    - Rate limiting
    - Response caching
    - Error handling and retry logic
    - Consistent logging
    - Performance metrics
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.enabled = True
        self.initialization_error: Optional[EnhancedError] = None

        # Initialize base services
        self.auth_manager = AuthManager(self.config.get("auth", {}))
        self.rate_limiter = RateLimiter(self.config.get("rate_limit", {}))
        self.cache_manager = CacheManager(self.config.get("cache", {}))
        self.error_handler = ErrorHandler(self.config.get("error_handling", {}))

        # Metrics tracking
        self.metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "total_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "rate_limit_hits": 0,
            "last_used": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Validate configuration during initialization - MUST propagate errors
        try:
            self._validate_configuration()
        except Exception as e:
            self.enabled = False
            missing_keys = getattr(e, 'missing_keys', None)
            details = {"config_issue": str(e)}
            if missing_keys:
                details["missing_keys"] = missing_keys

            self.initialization_error = create_integration_error(
                message=f"Configuration error: {str(e)}",
                code="CONFIG_ERROR",
                severity=ErrorSeverity.ERROR,
                troubleshooting="Check integration configuration, API keys, and required settings",
                details=details,
            )
            logger.error(f"❌ {self.name} integration disabled due to initialization failure: {e}")

        # Check for system logging fallback warnings and report them
        self._check_system_logging()

        logger.info(f"🪄 {self.name} integration initialized")

    def _check_system_logging(self):
        """Check system logging configuration and report issues"""
        import os
        import tempfile

        # Check if we're using fallback logging directory
        system_log_dirs = ["/var/log/wand", "/usr/local/var/log/wand"]
        writable_system_dir = None

        for log_dir in system_log_dirs:
            if os.path.exists(log_dir) and os.access(log_dir, os.W_OK):
                writable_system_dir = log_dir
                break

        if not writable_system_dir:
            fallback_dir = os.path.join(tempfile.gettempdir(), "wand")
            warning_msg = (
                f"SYSTEM LOGGING FALLBACK: No writable system log directory found. "
                f"Falling back to temporary directory: {fallback_dir}. "
                f"For production use, ensure system directories are available: {', '.join(system_log_dirs)}"
            )

            # Log the system warning with structured context
            logger.warning(
                f"⚠️  SYSTEM LOGGING FALLBACK: No writable system log directory found. "
                f"Falling back to temporary directory: {fallback_dir}. "
                f"For production use, ensure system directories are available: {', '.join(system_log_dirs)}",
                extra={
                    "integration": self.name,
                    "attempted_directories": system_log_dirs,
                    "fallback_directory": fallback_dir,
                    "current_user": os.getenv("USER", "unknown"),
                    "warning_type": "SYSTEM_LOGGING_FALLBACK",
                    "troubleshooting": "Create writable system log directories or configure proper logging paths",
                },
            )

    def _validate_configuration(self):
        """Validate required configuration - override in subclasses"""
        required_keys = getattr(self, 'REQUIRED_CONFIG_KEYS', [])
        missing_keys = [key for key in required_keys if not self.config.get(key)]

        if missing_keys:
            exc = ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")
            exc.missing_keys = missing_keys
            raise exc

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

    @abstractmethod
    async def initialize(self):
        """Initialize the integration (connections, auth, etc.)"""
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup resources (close connections, etc.)"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check if the integration is healthy and accessible"""
        health_data = {
            "status": "healthy" if self.enabled else "disabled",
            "enabled": self.enabled,
            "metrics": self.get_metrics(),
            "last_error": str(self.initialization_error) if self.initialization_error else None,
            "config_summary": self._get_config_summary(),
        }

        # Report to Go monitoring service if in hosted mode
        hosted_mode = self.config.get("hosted_mode", False)
        if hosted_mode:
            await self._report_health_to_monitoring_service(health_data)

        return health_data

    async def execute_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute operation with enhanced error handling"""

        # Check if integration is enabled
        if not self.enabled:
            return create_enhanced_error_response(
                integration_name=self.name,
                operation=operation,
                error=self.initialization_error
                or create_integration_error(
                    message=f"{self.name} integration is disabled", code="INTEGRATION_DISABLED"
                ),
            )

        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        operation_key = f"{self.name}:{operation}:{hash(str(sorted(kwargs.items())))}"

        try:
            # Update metrics
            self.metrics["requests_total"] += 1
            self.metrics["last_used"] = datetime.now(timezone.utc).isoformat()

            # Rate limiting check with enhanced error (can be disabled via config)
            rate_limiting_enabled = self.config.get("rate_limiting_enabled", False)
            if rate_limiting_enabled and not await self.rate_limiter.check_rate_limit(self.name):
                retry_after = self.rate_limiter.get_retry_after(self.name)
                current_usage = (
                    self.rate_limiter.get_current_usage(self.name)
                    if hasattr(self.rate_limiter, 'get_current_usage')
                    else None
                )

                details = {"limit_type": f"{self.name} requests"}
                if current_usage is not None:
                    details["current_usage"] = current_usage

                error = create_integration_error(
                    message=f"Rate limit exceeded: {self.name} requests"
                    + (f", retry after {retry_after} seconds" if retry_after else ""),
                    code="RATE_LIMIT_EXCEEDED",
                    severity=ErrorSeverity.WARNING,
                    retry_suggested=True,
                    retry_after_seconds=retry_after or 300,
                    troubleshooting="Wait for rate limit window to reset or upgrade service plan",
                    details=details,
                )

                self.metrics["rate_limit_hits"] += 1
                return create_enhanced_error_response(
                    integration_name=self.name,
                    operation=operation,
                    error=error,
                    response_time_ms=(time.time() - start_time) * 1000,
                    correlation_id=correlation_id,
                )

            # Check cache
            cached_result = await self.cache_manager.get(operation_key)
            if cached_result:
                self.metrics["cache_hits"] += 1
                logger.debug(f"🎯 Cache hit for {self.name}:{operation}")
                return {**cached_result, "cached": True, "integration": self.name, "correlation_id": correlation_id}

            self.metrics["cache_misses"] += 1

            # Execute operation with retry logic
            result = await self.error_handler.execute_with_retry(self._execute_operation_impl, operation, **kwargs)

            response_time_ms = (time.time() - start_time) * 1000

            if result.get("success", False):
                self.metrics["requests_successful"] += 1
                # Add enhanced metadata to successful responses
                result.update(
                    {
                        "integration": self.name,
                        "operation": operation,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "correlation_id": correlation_id,
                        "response_time_ms": response_time_ms,
                    }
                )

                # Cache successful results
                await self.cache_manager.set(operation_key, result)
            else:
                self.metrics["requests_failed"] += 1
                # If implementation returns basic error, enhance it
                if "error" in result and isinstance(result["error"], str):
                    enhanced_error = create_integration_error(
                        message=result["error"], code="OPERATION_FAILED", retry_suggested=True
                    )
                    result = create_enhanced_error_response(
                        integration_name=self.name,
                        operation=operation,
                        error=enhanced_error,
                        response_time_ms=response_time_ms,
                        correlation_id=correlation_id,
                    )

            # Update metrics
            self.metrics["total_response_time"] += response_time_ms / 1000

            logger.info(f"✅ {self.name}:{operation} completed in {response_time_ms:.2f}ms")
            return result

        except Exception as e:
            self.metrics["requests_failed"] += 1
            response_time_ms = (time.time() - start_time) * 1000

            # Create enhanced error based on exception type
            error = self._categorize_exception(e, operation, **kwargs)

            logger.error(f"❌ {self.name}:{operation} failed: {e}")

            return create_enhanced_error_response(
                integration_name=self.name,
                operation=operation,
                error=error,
                response_time_ms=response_time_ms,
                correlation_id=correlation_id,
            )

    def _categorize_exception(self, exception: Exception, operation: str, **kwargs) -> EnhancedError:
        """Categorize exception into enhanced error"""
        error_msg = str(exception)
        error_type = type(exception).__name__

        details = {
            "exception_type": error_type,
            "operation": operation,
            "parameters": {k: str(v)[:100] for k, v in kwargs.items()},
        }

        # Timeout errors
        if "timeout" in error_msg.lower() or "TimeoutError" in error_type:
            timeout_duration = getattr(exception, 'timeout', 30)
            details.update({"timeout_duration": timeout_duration, "attempt_number": getattr(exception, 'attempt', 1)})
            if kwargs.get('url') or kwargs.get('endpoint'):
                details["endpoint"] = kwargs.get('url') or kwargs.get('endpoint')

            return create_integration_error(
                message=f"Operation timed out after {timeout_duration} seconds",
                code="TIMEOUT_ERROR",
                severity=ErrorSeverity.WARNING,
                retry_suggested=True,
                retry_after_seconds=min(int(timeout_duration * 1.5), 120),
                troubleshooting="Check network connectivity and service response times",
                details=details,
                native_error=exception,
            )

        # Authentication errors
        elif any(auth_term in error_msg.lower() for auth_term in ['auth', 'unauthorized', '401', 'forbidden', '403']):
            details.update({"auth_type": self._detect_auth_type(), "auth_message": error_msg})
            return create_integration_error(
                message=f"Authentication failed: {error_msg}",
                code="AUTH_ERROR",
                retry_suggested=False,
                troubleshooting="Verify API credentials, tokens, or authentication configuration",
                details=details,
                native_error=exception,
            )

        # Connection errors
        elif any(conn_term in error_msg.lower() for conn_term in ['connection', 'network', 'unreachable', 'refused']):
            status_code = getattr(exception, 'status_code', None)
            details["connection_message"] = error_msg
            if kwargs.get('url') or kwargs.get('endpoint'):
                details["endpoint"] = kwargs.get('url') or kwargs.get('endpoint')
            if status_code:
                details["status_code"] = status_code

            return create_integration_error(
                message=f"Connection failed: {error_msg}",
                code="CONNECTION_ERROR",
                retry_suggested=True,
                retry_after_seconds=60,
                troubleshooting="Check network connectivity, firewall settings, and service availability",
                details=details,
                native_error=exception,
            )

        # Generic service error
        else:
            return create_integration_error(
                message=f"{error_type}: {error_msg}",
                code=f"{error_type.upper()}",
                retry_suggested=True,
                troubleshooting=f"Check {self.name} integration logs and service status",
                details=details,
                native_error=exception,
            )

    def _detect_auth_type(self) -> str:
        """Detect authentication type from config"""
        if 'api_key' in self.config:
            return "api_key"
        elif 'token' in self.config:
            return "bearer_token"
        elif 'oauth' in self.config:
            return "oauth"
        else:
            return "unknown"

    @abstractmethod
    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Implementation-specific operation execution
        Must be implemented by each integration
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get integration performance metrics"""
        metrics = self.metrics.copy()

        # Calculate derived metrics
        if metrics["requests_total"] > 0:
            metrics["success_rate"] = metrics["requests_successful"] / metrics["requests_total"]
            metrics["average_response_time"] = metrics["total_response_time"] / metrics["requests_total"]
            metrics["cache_hit_rate"] = metrics["cache_hits"] / (metrics["cache_hits"] + metrics["cache_misses"])
        else:
            metrics["success_rate"] = 0.0
            metrics["average_response_time"] = 0.0
            metrics["cache_hit_rate"] = 0.0

        return metrics

    def get_status(self) -> Dict[str, Any]:
        """Get integration status summary"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "healthy": True,  # Will be updated by health_check
            "metrics": self.get_metrics(),
            "config_summary": self._get_config_summary(),
        }

    def _get_config_summary(self) -> Dict[str, Any]:
        """Get safe configuration summary (no sensitive data)"""
        safe_config = {}

        # Add non-sensitive config items
        for key, value in self.config.items():
            if key not in ["auth", "api_key", "token", "password", "secret"]:
                safe_config[key] = value
            else:
                safe_config[key] = "***configured***" if value else "***not configured***"

        return safe_config

    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to the external service"""
        try:
            health_result = await self.health_check()
            return {
                "success": True,
                "integration": self.name,
                "health": health_result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "integration": self.name,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def reset_metrics(self):
        """Reset all metrics counters"""
        self.metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "total_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "rate_limit_hits": 0,
            "last_used": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"🔄 {self.name} metrics reset")

    async def _report_health_to_monitoring_service(self, health_data: Dict[str, Any]):
        """Report health data to Go monitoring service (if enabled)"""
        try:
            import os

            import httpx

            # Get monitoring service URL and token from config or environment
            monitoring_url = self.config.get(
                "monitoring_service_url", os.getenv("MONITORING_SERVICE_URL", "http://localhost:8083")
            )
            service_token = self.config.get("service_token", os.getenv("WAND_SERVICE_TOKEN"))

            if not service_token:
                logger.debug("No service token configured, skipping health report")
                return

            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{monitoring_url}/api/v1/monitoring/health/{self.name}",
                    json=health_data,
                    headers={"Authorization": f"Bearer {service_token}", "X-Service-Name": "wand-backend"},
                )
                logger.debug(f"Health data reported for {self.name}")

        except Exception as e:
            # Don't fail health check if monitoring report fails
            logger.debug(f"Failed to report health for {self.name}: {e}")

    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})>"
