"""
Unit Tests for Error Handling System

Tests error handling, circuit breakers, retry mechanisms, and graceful degradation.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from utils.error_handling import (
    APIError,
    AuthenticationError,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ErrorSeverity,
    FrameworkError,
    GracefulDegradation,
    MCPError,
    RateLimitError,
    RetryConfig,
    RetryMechanism,
    handle_errors,
    with_circuit_breaker,
    with_retry,
)


class TestMCPError:
    """Test MCPError and its subclasses."""

    def test_mcp_error_creation(self):
        """Test basic MCPError creation."""
        error = MCPError(
            "Test error", category=ErrorCategory.FRAMEWORK_ERROR, severity=ErrorSeverity.HIGH, details={"test": "data"}
        )

        assert error.message == "Test error"
        assert error.category == ErrorCategory.FRAMEWORK_ERROR
        assert error.severity == ErrorSeverity.HIGH
        assert error.details == {"test": "data"}
        assert error.recoverable is True
        assert isinstance(error.timestamp, datetime)
        assert error.error_id.startswith("framework_error_")

    def test_framework_error(self):
        """Test FrameworkError creation."""
        error = FrameworkError("Framework failed", "haystack")

        assert error.framework == "haystack"
        assert error.category == ErrorCategory.FRAMEWORK_ERROR
        assert isinstance(error, MCPError)

    def test_api_error(self):
        """Test APIError creation."""
        error = APIError("Bad request", status_code=400)

        assert error.status_code == 400
        assert error.details["status_code"] == 400
        assert error.category == ErrorCategory.API_ERROR

    def test_authentication_error(self):
        """Test AuthenticationError creation."""
        error = AuthenticationError("Invalid token")

        assert error.category == ErrorCategory.AUTHENTICATION_ERROR
        assert error.severity == ErrorSeverity.HIGH
        assert error.recoverable is False

    def test_rate_limit_error(self):
        """Test RateLimitError creation."""
        error = RateLimitError("Rate limit exceeded", retry_after=60)

        assert error.category == ErrorCategory.RATE_LIMIT_ERROR
        assert error.details["retry_after"] == 60


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    @pytest.fixture
    def circuit_config(self):
        """Create circuit breaker config for testing."""
        return CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout=1, name="test_circuit"  # 1 second for testing
        )

    @pytest.fixture
    def circuit_breaker(self, circuit_config):
        """Create circuit breaker for testing."""
        return CircuitBreaker(circuit_config)

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self, circuit_breaker):
        """Test circuit breaker in closed state."""

        @circuit_breaker
        async def test_function():
            return "success"

        result = await test_function()
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, circuit_breaker):
        """Test circuit breaker opens after threshold failures."""
        failure_count = 0

        @circuit_breaker
        async def failing_function():
            nonlocal failure_count
            failure_count += 1
            raise Exception(f"Failure {failure_count}")

        # Trigger failures to open circuit
        for i in range(3):
            with pytest.raises(Exception):
                await failing_function()

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.failure_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_when_open(self, circuit_breaker):
        """Test circuit breaker blocks calls when open."""
        # Force circuit to open
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failure_count = 5
        circuit_breaker.last_failure_time = datetime.now()

        @circuit_breaker
        async def test_function():
            return "should not execute"

        with pytest.raises(MCPError) as exc_info:
            await test_function()

        assert "Circuit breaker test_circuit is OPEN" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self, circuit_breaker):
        """Test circuit breaker recovery through half-open state."""
        # Force circuit to open and set old failure time
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failure_count = 5
        circuit_breaker.last_failure_time = datetime.now() - timedelta(seconds=2)

        success_count = 0

        @circuit_breaker
        async def recovering_function():
            nonlocal success_count
            success_count += 1
            return f"success {success_count}"

        # First call should move to half-open
        result1 = await recovering_function()
        assert result1 == "success 1"
        assert circuit_breaker.state == CircuitState.HALF_OPEN

        # More successful calls should close the circuit
        for i in range(2, 5):
            result = await recovering_function()
            assert result == f"success {i}"

        assert circuit_breaker.state == CircuitState.CLOSED


class TestRetryMechanism:
    """Test RetryMechanism functionality."""

    @pytest.fixture
    def retry_config(self):
        """Create retry config for testing."""
        return RetryConfig(
            max_attempts=3, base_delay=0.1, max_delay=1.0, exponential_base=2.0  # Small delay for testing
        )

    @pytest.fixture
    def retry_mechanism(self, retry_config):
        """Create retry mechanism for testing."""
        return RetryMechanism(retry_config)

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self, retry_mechanism):
        """Test retry mechanism with successful first attempt."""
        call_count = 0

        @retry_mechanism
        async def test_function():
            nonlocal call_count
            call_count += 1
            return f"success on attempt {call_count}"

        result = await test_function()
        assert result == "success on attempt 1"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self, retry_mechanism):
        """Test retry mechanism with success after failures."""
        call_count = 0

        @retry_mechanism
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Failure on attempt {call_count}")
            return f"success on attempt {call_count}"

        result = await test_function()
        assert result == "success on attempt 3"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausts_attempts(self, retry_mechanism):
        """Test retry mechanism exhausts all attempts."""
        call_count = 0

        @retry_mechanism
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise Exception(f"Failure on attempt {call_count}")

        with pytest.raises(Exception) as exc_info:
            await failing_function()

        assert call_count == 3  # max_attempts
        assert "Failure on attempt 3" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retry_non_retryable_exception(self, retry_mechanism):
        """Test retry mechanism with non-retryable exception."""
        call_count = 0

        @retry_mechanism
        async def test_function():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Invalid credentials")

        with pytest.raises(AuthenticationError):
            await test_function()

        assert call_count == 1  # No retries for non-retryable exception


class TestErrorHandler:
    """Test ErrorHandler functionality."""

    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing."""
        return ErrorHandler()

    def test_handle_mcp_error(self, error_handler):
        """Test handling of MCPError."""
        error = FrameworkError("Test framework error", "haystack")
        context = error_handler.handle_error(error, {"test": "context"})

        assert context.error_id == error.error_id
        assert context.category == ErrorCategory.FRAMEWORK_ERROR
        assert context.framework == "haystack"
        assert context.details["test"] == "context"
        assert len(context.recovery_suggestions) > 0

    def test_handle_generic_exception(self, error_handler):
        """Test handling of generic exception."""
        error = ValueError("Invalid value")
        context = error_handler.handle_error(error, {"field": "test_field"})

        assert context.category == ErrorCategory.UNKNOWN_ERROR
        assert context.message == "Invalid value"
        assert context.details["field"] == "test_field"
        assert context.stack_trace is not None

    def test_error_statistics(self, error_handler):
        """Test error statistics collection."""
        # Generate some errors
        errors = [
            FrameworkError("Error 1", "haystack"),
            FrameworkError("Error 2", "llamaindex"),
            APIError("Error 3", 500),
            AuthenticationError("Error 4"),
        ]

        for error in errors:
            error_handler.handle_error(error)

        stats = error_handler.get_error_statistics()

        assert stats["total_errors"] == 4
        assert stats["errors_by_category"][ErrorCategory.FRAMEWORK_ERROR.value] == 2
        assert stats["errors_by_category"][ErrorCategory.API_ERROR.value] == 1
        assert stats["errors_by_category"][ErrorCategory.AUTHENTICATION_ERROR.value] == 1
        assert stats["errors_by_framework"]["haystack"] == 1
        assert stats["errors_by_framework"]["llamaindex"] == 1

    def test_recent_errors(self, error_handler):
        """Test recent errors retrieval."""
        # Generate some errors
        for i in range(5):
            error = FrameworkError(f"Error {i}", "test_framework")
            error_handler.handle_error(error)

        recent = error_handler.get_recent_errors(limit=3)

        assert len(recent) == 3
        # Should be sorted by timestamp (most recent first)
        assert recent[0]["message"] == "Error 4"
        assert recent[1]["message"] == "Error 3"
        assert recent[2]["message"] == "Error 2"


class TestGracefulDegradation:
    """Test GracefulDegradation utilities."""

    @pytest.mark.asyncio
    async def test_fallback_to_cache(self):
        """Test fallback to cache functionality."""

        async def primary_function(data):
            raise Exception("Primary function failed")

        async def cache_function(data):
            return {"cached": True, "data": data}

        result = await GracefulDegradation.fallback_to_cache(primary_function, cache_function, "test_data")

        assert result["cached"] is True
        assert result["data"] == "test_data"

    @pytest.mark.asyncio
    async def test_fallback_to_alternative_framework(self):
        """Test fallback to alternative framework."""

        async def primary_framework(query):
            raise Exception("Primary framework failed")

        async def alternative_framework(query):
            return {"framework": "alternative", "answer": f"Answer for {query}"}

        result = await GracefulDegradation.fallback_to_alternative_framework(
            primary_framework, alternative_framework, "test query"
        )

        assert result["framework"] == "alternative"
        assert result["answer"] == "Answer for test query"

    def test_default_response(self):
        """Test default response generation."""
        error = Exception("Something went wrong")
        response = GracefulDegradation.default_response("rag_query", error)

        assert response["success"] is False
        assert response["error"] == "Service temporarily unavailable"
        assert response["operation"] == "rag_query"
        assert response["fallback"] is True


class TestDecorators:
    """Test error handling decorators."""

    @pytest.mark.asyncio
    async def test_handle_errors_decorator(self):
        """Test handle_errors decorator."""

        @handle_errors
        async def test_function():
            raise ValueError("Test error")

        with pytest.raises(MCPError) as exc_info:
            await test_function()

        assert exc_info.value.category == ErrorCategory.UNKNOWN_ERROR
        assert "Test error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_with_circuit_breaker_decorator(self):
        """Test with_circuit_breaker decorator."""

        @with_circuit_breaker(failure_threshold=2, recovery_timeout=1, name="test")
        async def test_function(should_fail=False):
            if should_fail:
                raise Exception("Test failure")
            return "success"

        # Test successful call
        result = await test_function()
        assert result == "success"

        # Test failures to open circuit
        with pytest.raises(Exception):
            await test_function(should_fail=True)

        with pytest.raises(Exception):
            await test_function(should_fail=True)

        # Next call should be blocked by open circuit
        with pytest.raises(MCPError) as exc_info:
            await test_function()

        assert "Circuit breaker test is OPEN" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_with_retry_decorator(self):
        """Test with_retry decorator."""
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01)
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Attempt {call_count} failed")
            return f"success on attempt {call_count}"

        result = await test_function()
        assert result == "success on attempt 3"
        assert call_count == 3
