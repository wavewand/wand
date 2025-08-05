"""
Error handling and retry logic for Wand integrations
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetryStrategy(Enum):
    """Retry strategies"""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


class ErrorHandler:
    """
    Comprehensive error handling with:
    - Multiple retry strategies
    - Circuit breaker pattern
    - Error classification
    - Detailed logging
    - Fallback mechanisms
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)

        # Retry configuration
        self.max_retries = config.get("max_retries", 3)
        self.retry_strategy = RetryStrategy(config.get("retry_strategy", "exponential_backoff"))
        self.base_delay = config.get("base_delay", 1.0)
        self.max_delay = config.get("max_delay", 60.0)
        self.jitter = config.get("jitter", True)

        # Circuit breaker configuration
        self.circuit_breaker_enabled = config.get("circuit_breaker_enabled", True)
        self.failure_threshold = config.get("failure_threshold", 5)
        self.recovery_timeout = config.get("recovery_timeout", 43200)  # 12 hours
        self.success_threshold = config.get("success_threshold", 2)

        # Circuit breaker state
        self.circuit_states = {}  # service -> state
        self.failure_counts = {}  # service -> count
        self.last_failure_times = {}  # service -> timestamp
        self.success_counts = {}  # service -> count in half-open state

        # Error tracking
        self.error_history = []
        self.error_counts = {}

        # Retryable errors (HTTP status codes and exception types)
        self.retryable_status_codes = set(
            config.get(
                "retryable_status_codes",
                [
                    408,  # Request Timeout
                    429,  # Too Many Requests
                    500,  # Internal Server Error
                    502,  # Bad Gateway
                    503,  # Service Unavailable
                    504,  # Gateway Timeout
                ],
            )
        )

        self.retryable_exceptions = config.get(
            "retryable_exceptions",
            [
                "aiohttp.ClientTimeout",
                "aiohttp.ClientConnectorError",
                "aiohttp.ServerDisconnectedError",
                "asyncio.TimeoutError",
                "ConnectionError",
                "ConnectionResetError",
            ],
        )

    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic and circuit breaker

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries exhausted or circuit breaker open
        """
        if not self.enabled:
            return await func(*args, **kwargs)

        service_name = kwargs.get("service_name", func.__name__)

        # Check circuit breaker
        if self.circuit_breaker_enabled:
            circuit_state = self._get_circuit_state(service_name)

            if circuit_state == "open":
                raise Exception(f"Circuit breaker is OPEN for {service_name}")

            elif circuit_state == "half_open":
                # Allow limited requests in half-open state
                pass

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Success - update circuit breaker
                await self._record_success(service_name, execution_time)

                if attempt > 0:
                    logger.info(f"✅ {service_name} succeeded after {attempt} retries")

                return result

            except Exception as e:
                last_exception = e
                execution_time = time.time() - start_time

                # Record the error
                await self._record_error(service_name, e, attempt, execution_time)

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    logger.error(f"❌ {service_name} failed with non-retryable error: {e}")
                    break

                # Don't retry on last attempt
                if attempt >= self.max_retries:
                    break

                # Calculate delay for next retry
                delay = self._calculate_delay(attempt)
                logger.warning(f"🔄 {service_name} attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")

                await asyncio.sleep(delay)

        # All retries exhausted
        await self._record_failure(service_name)
        raise last_exception

    def _get_circuit_state(self, service: str) -> str:
        """Get current circuit breaker state for service"""
        if service not in self.circuit_states:
            self.circuit_states[service] = "closed"
            self.failure_counts[service] = 0
            self.success_counts[service] = 0

        current_time = time.time()
        state = self.circuit_states[service]

        if state == "open":
            # Check if recovery timeout has passed
            last_failure = self.last_failure_times.get(service, 0)
            if current_time - last_failure >= self.recovery_timeout:
                self.circuit_states[service] = "half_open"
                self.success_counts[service] = 0
                logger.info(f"🔄 Circuit breaker for {service} moved to HALF_OPEN")
                return "half_open"

        return state

    async def _record_success(self, service: str, execution_time: float):
        """Record successful execution"""
        state = self.circuit_states.get(service, "closed")

        if state == "half_open":
            self.success_counts[service] = self.success_counts.get(service, 0) + 1

            if self.success_counts[service] >= self.success_threshold:
                # Move to closed state
                self.circuit_states[service] = "closed"
                self.failure_counts[service] = 0
                self.success_counts[service] = 0
                logger.info(f"✅ Circuit breaker for {service} moved to CLOSED")

        elif state == "closed":
            # Reset failure count on success
            self.failure_counts[service] = 0

    async def _record_error(self, service: str, error: Exception, attempt: int, execution_time: float):
        """Record error occurrence"""
        error_info = {
            "service": service,
            "error": str(error),
            "error_type": type(error).__name__,
            "attempt": attempt,
            "execution_time": execution_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": self._classify_error(error),
        }

        self.error_history.append(error_info)

        # Limit error history size
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]

        # Update error counts
        error_key = f"{service}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

    async def _record_failure(self, service: str):
        """Record service failure for circuit breaker"""
        if not self.circuit_breaker_enabled:
            return

        self.failure_counts[service] = self.failure_counts.get(service, 0) + 1
        self.last_failure_times[service] = time.time()

        # Check if we should open the circuit
        if self.failure_counts[service] >= self.failure_threshold:
            self.circuit_states[service] = "open"
            logger.warning(
                f"🚨 Circuit breaker for {service} moved to OPEN after {self.failure_counts[service]} failures"
            )

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable"""
        # Check by exception type
        error_type = f"{type(error).__module__}.{type(error).__name__}"
        if error_type in self.retryable_exceptions:
            return True

        # Check by HTTP status code (if available)
        if hasattr(error, 'status') and error.status in self.retryable_status_codes:
            return True

        # Check by error message patterns
        error_message = str(error).lower()
        retryable_patterns = ["timeout", "connection", "network", "temporary", "rate limit", "throttle"]

        for pattern in retryable_patterns:
            if pattern in error_message:
                return True

        return False

    def _classify_error(self, error: Exception) -> ErrorSeverity:
        """Classify error severity"""
        error_message = str(error).lower()

        # Critical errors
        if any(keyword in error_message for keyword in ["authentication", "unauthorized", "forbidden"]):
            return ErrorSeverity.CRITICAL

        # High severity
        if any(keyword in error_message for keyword in ["internal server error", "database", "critical"]):
            return ErrorSeverity.HIGH

        # Medium severity
        if any(keyword in error_message for keyword in ["timeout", "connection", "network"]):
            return ErrorSeverity.MEDIUM

        # Default to low
        return ErrorSeverity.LOW

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.retry_strategy == RetryStrategy.FIXED_DELAY:
            delay = self.base_delay

        elif self.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt + 1)

        elif self.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (2**attempt)

        elif self.retry_strategy == RetryStrategy.FIBONACCI_BACKOFF:
            fib_sequence = [1, 1]
            for i in range(2, attempt + 2):
                fib_sequence.append(fib_sequence[i - 1] + fib_sequence[i - 2])
            delay = self.base_delay * fib_sequence[attempt]

        else:
            delay = self.base_delay

        # Apply maximum delay limit
        delay = min(delay, self.max_delay)

        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)

        return max(delay, 0.1)  # Minimum 0.1 second delay

    def get_error_stats(self, service: str = None) -> Dict[str, Any]:
        """Get error statistics"""
        current_time = time.time()

        if service:
            # Service-specific stats
            service_errors = [e for e in self.error_history if e["service"] == service]
            circuit_state = self.circuit_states.get(service, "closed")
            failure_count = self.failure_counts.get(service, 0)

            return {
                "service": service,
                "circuit_state": circuit_state,
                "failure_count": failure_count,
                "recent_errors": len(
                    [
                        e
                        for e in service_errors
                        if (current_time - time.mktime(time.strptime(e["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z"))) < 3600
                    ]
                ),
                "total_errors": len(service_errors),
                "error_types": list(set(e["error_type"] for e in service_errors)),
            }

        else:
            # Overall stats
            recent_errors = [
                e
                for e in self.error_history
                if (current_time - time.mktime(time.strptime(e["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z"))) < 3600
            ]

            return {
                "total_errors": len(self.error_history),
                "recent_errors": len(recent_errors),
                "services_with_errors": len(set(e["service"] for e in self.error_history)),
                "circuit_breaker_open": sum(1 for state in self.circuit_states.values() if state == "open"),
                "most_common_errors": self._get_most_common_errors(),
            }

    def _get_most_common_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most common error types"""
        sorted_errors = sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"error": error, "count": count} for error, count in sorted_errors[:limit]]

    def reset_circuit_breaker(self, service: str):
        """Reset circuit breaker for a service"""
        if service in self.circuit_states:
            self.circuit_states[service] = "closed"
            self.failure_counts[service] = 0
            self.success_counts[service] = 0
            logger.info(f"🔄 Circuit breaker for {service} manually reset")

    def reset_all_circuit_breakers(self):
        """Reset all circuit breakers"""
        for service in self.circuit_states.keys():
            self.reset_circuit_breaker(service)

    def clear_error_history(self):
        """Clear error history"""
        self.error_history.clear()
        self.error_counts.clear()
        logger.info("🧹 Error history cleared")
