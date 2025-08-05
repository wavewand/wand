"""
Rate Limiting System

Provides advanced rate limiting capabilities with multiple algorithms,
user-based limits, and integration with the authentication system.
"""

import asyncio
import functools
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from utils.error_handling import ErrorCategory, MCPError, RateLimitError


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10  # For token bucket
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    grace_period: int = 5  # seconds to allow slight overages


class TokenBucket:
    """Token bucket rate limiter implementation."""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket."""
        async with self.lock:
            now = time.time()

            # Add tokens based on elapsed time
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait before tokens are available."""
        if self.tokens >= tokens:
            return 0.0

        needed_tokens = tokens - self.tokens
        return needed_tokens / self.refill_rate


class SlidingWindowCounter:
    """Sliding window rate limiter implementation."""

    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size  # in seconds
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = asyncio.Lock()

    async def is_allowed(self) -> bool:
        """Check if request is allowed under sliding window."""
        async with self.lock:
            now = time.time()

            # Remove old requests outside the window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()

            # Check if we're under the limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True

            return False

    def get_reset_time(self) -> float:
        """Get time when the oldest request will expire."""
        if not self.requests:
            return 0.0

        oldest_request = self.requests[0]
        return max(0.0, (oldest_request + self.window_size) - time.time())


class FixedWindowCounter:
    """Fixed window rate limiter implementation."""

    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size  # in seconds
        self.max_requests = max_requests
        self.count = 0
        self.window_start = time.time()
        self.lock = asyncio.Lock()

    async def is_allowed(self) -> bool:
        """Check if request is allowed under fixed window."""
        async with self.lock:
            now = time.time()

            # Reset window if needed
            if now >= self.window_start + self.window_size:
                self.count = 0
                self.window_start = now

            # Check if we're under the limit
            if self.count < self.max_requests:
                self.count += 1
                return True

            return False

    def get_reset_time(self) -> float:
        """Get time when the window will reset."""
        return max(0.0, (self.window_start + self.window_size) - time.time())


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[int] = None


class RateLimiter:
    """Advanced rate limiter with multiple algorithms and user-based limits."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Storage for different rate limiters by user/key
        self.user_limiters: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.global_limiters: Dict[str, Any] = {}

        # Default configurations
        self.default_config = RateLimitConfig()
        self.user_configs: Dict[str, RateLimitConfig] = {}

        # Statistics
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "user_stats": defaultdict(lambda: {"requests": 0, "blocked": 0}),
        }

    def set_user_config(self, user_id: str, config: RateLimitConfig):
        """Set rate limit configuration for specific user."""
        self.user_configs[user_id] = config
        self.logger.info(f"Set rate limit config for user {user_id}: {config}")

    def get_user_config(self, user_id: str) -> RateLimitConfig:
        """Get rate limit configuration for user."""
        return self.user_configs.get(user_id, self.default_config)

    async def check_rate_limit(self, user_id: str, endpoint: str = "default", tokens: int = 1) -> RateLimitResult:
        """Check if request is allowed under rate limits."""

        config = self.get_user_config(user_id)
        self.stats["total_requests"] += 1
        self.stats["user_stats"][user_id]["requests"] += 1

        # Create limiters if they don't exist
        limiter_key = f"{user_id}:{endpoint}"
        if limiter_key not in self.user_limiters[user_id]:
            await self._create_limiters(user_id, endpoint, config)

        limiters = self.user_limiters[user_id][limiter_key]

        # Check all time windows
        results = []

        # Check per-minute limit
        if "minute" in limiters:
            allowed = await self._check_limiter(limiters["minute"], tokens, config)
            results.append(("minute", allowed))

        # Check per-hour limit
        if "hour" in limiters:
            allowed = await self._check_limiter(limiters["hour"], tokens, config)
            results.append(("hour", allowed))

        # Check per-day limit
        if "day" in limiters:
            allowed = await self._check_limiter(limiters["day"], tokens, config)
            results.append(("day", allowed))

        # Determine overall result
        blocked_by = [window for window, allowed in results if not allowed]

        if blocked_by:
            self.stats["blocked_requests"] += 1
            self.stats["user_stats"][user_id]["blocked"] += 1

            # Calculate retry after time
            retry_after = await self._calculate_retry_after(limiters, blocked_by)

            self.logger.warning(f"Rate limit exceeded for user {user_id} on {endpoint} (blocked by: {blocked_by})")

            return RateLimitResult(allowed=False, remaining=0, reset_time=retry_after, retry_after=int(retry_after))

        # Calculate remaining requests (use most restrictive)
        remaining = await self._calculate_remaining(limiters, config)
        reset_time = await self._calculate_next_reset(limiters)

        return RateLimitResult(allowed=True, remaining=remaining, reset_time=reset_time)

    async def _create_limiters(self, user_id: str, endpoint: str, config: RateLimitConfig):
        """Create rate limiters for user/endpoint combination."""
        limiter_key = f"{user_id}:{endpoint}"
        limiters = {}

        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            # Per-minute token bucket
            limiters["minute"] = TokenBucket(capacity=config.burst_size, refill_rate=config.requests_per_minute / 60.0)

            # Per-hour token bucket
            limiters["hour"] = TokenBucket(
                capacity=config.requests_per_hour // 10,  # Allow some burst
                refill_rate=config.requests_per_hour / 3600.0,
            )

            # Per-day token bucket
            limiters["day"] = TokenBucket(
                capacity=config.requests_per_day // 24,  # Allow hourly burst
                refill_rate=config.requests_per_day / 86400.0,
            )

        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            limiters["minute"] = SlidingWindowCounter(60, config.requests_per_minute)
            limiters["hour"] = SlidingWindowCounter(3600, config.requests_per_hour)
            limiters["day"] = SlidingWindowCounter(86400, config.requests_per_day)

        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            limiters["minute"] = FixedWindowCounter(60, config.requests_per_minute)
            limiters["hour"] = FixedWindowCounter(3600, config.requests_per_hour)
            limiters["day"] = FixedWindowCounter(86400, config.requests_per_day)

        self.user_limiters[user_id][limiter_key] = limiters

    async def _check_limiter(self, limiter, tokens: int, config: RateLimitConfig) -> bool:
        """Check if request is allowed by specific limiter."""
        if isinstance(limiter, TokenBucket):
            return await limiter.consume(tokens)
        elif isinstance(limiter, (SlidingWindowCounter, FixedWindowCounter)):
            return await limiter.is_allowed()

        return True

    async def _calculate_retry_after(self, limiters: Dict[str, Any], blocked_by: List[str]) -> float:
        """Calculate retry after time."""
        retry_times = []

        for window in blocked_by:
            if window in limiters:
                limiter = limiters[window]
                if isinstance(limiter, TokenBucket):
                    retry_times.append(limiter.get_wait_time())
                elif isinstance(limiter, (SlidingWindowCounter, FixedWindowCounter)):
                    retry_times.append(limiter.get_reset_time())

        return max(retry_times) if retry_times else 60.0

    async def _calculate_remaining(self, limiters: Dict[str, Any], config: RateLimitConfig) -> int:
        """Calculate remaining requests (most restrictive limit)."""
        # This is a simplified calculation
        # In practice, you'd need to check each limiter's current state
        return (
            min(config.requests_per_minute, config.requests_per_hour, config.requests_per_day) // 10
        )  # Conservative estimate

    async def _calculate_next_reset(self, limiters: Dict[str, Any]) -> float:
        """Calculate next reset time."""
        reset_times = []

        for limiter in limiters.values():
            if isinstance(limiter, (SlidingWindowCounter, FixedWindowCounter)):
                reset_times.append(limiter.get_reset_time())

        return min(reset_times) if reset_times else 60.0

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get rate limiting statistics for user."""
        user_stats = self.stats["user_stats"][user_id]

        return {
            "user_id": user_id,
            "total_requests": user_stats["requests"],
            "blocked_requests": user_stats["blocked"],
            "block_rate": user_stats["blocked"] / max(user_stats["requests"], 1),
            "config": self.get_user_config(user_id).__dict__,
        }

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics."""
        return {
            "total_requests": self.stats["total_requests"],
            "blocked_requests": self.stats["blocked_requests"],
            "block_rate": self.stats["blocked_requests"] / max(self.stats["total_requests"], 1),
            "active_users": len(self.stats["user_stats"]),
            "top_users": sorted(
                [{"user_id": uid, **stats} for uid, stats in self.stats["user_stats"].items()],
                key=lambda x: x["requests"],
                reverse=True,
            )[:10],
        }

    def reset_user_limits(self, user_id: str):
        """Reset rate limits for specific user."""
        if user_id in self.user_limiters:
            del self.user_limiters[user_id]

        if user_id in self.stats["user_stats"]:
            del self.stats["user_stats"][user_id]

        self.logger.info(f"Reset rate limits for user {user_id}")


# Global rate limiter instance
rate_limiter = RateLimiter()


class RateLimitExceeded(RateLimitError):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, result: RateLimitResult, user_id: str, endpoint: str):
        super().__init__(f"Rate limit exceeded for user {user_id} on {endpoint}", retry_after=result.retry_after)
        self.result = result
        self.user_id = user_id
        self.endpoint = endpoint


def rate_limit(endpoint: str = "default", tokens: int = 1):
    """Decorator to apply rate limiting to functions."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user_id from kwargs or context
            # This would be integrated with the auth system
            user_id = kwargs.get("user_id", "anonymous")

            # Check rate limit
            result = await rate_limiter.check_rate_limit(user_id, endpoint, tokens)

            if not result.allowed:
                raise RateLimitExceeded(result, user_id, endpoint)

            # Add rate limit info to response headers (if applicable)
            response = await func(*args, **kwargs)

            # If response is a dict, add rate limit info
            if isinstance(response, dict):
                response["_rate_limit"] = {"remaining": result.remaining, "reset_time": result.reset_time}

            return response

        return wrapper

    return decorator
