"""
Rate limiting for Wand integrations
"""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class RateLimiter:
    """
    Token bucket rate limiter with per-service limits
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)

        # Default rate limits (requests per minute)
        self.default_limits = {
            "requests_per_minute": config.get("default_rpm", 60),
            "requests_per_second": config.get("default_rps", 10),
            "burst_size": config.get("default_burst", 20),
        }

        # Service-specific limits
        self.service_limits = config.get("service_limits", {})

        # Token buckets for each service
        self.buckets = defaultdict(
            lambda: {"tokens": 0, "last_refill": time.time(), "requests_history": deque()}  # For sliding window
        )

        # Track rate limit violations
        self.violations = defaultdict(int)
        self.violation_timestamps = defaultdict(list)

    async def check_rate_limit(self, service: str) -> bool:
        """
        Check if request is allowed for service

        Args:
            service: Service name to check

        Returns:
            True if request is allowed, False if rate limited
        """
        if not self.enabled:
            return True

        current_time = time.time()
        bucket = self.buckets[service]
        limits = self._get_service_limits(service)

        # Refill tokens based on time elapsed
        time_elapsed = current_time - bucket["last_refill"]
        tokens_to_add = time_elapsed * (limits["requests_per_minute"] / 60.0)

        bucket["tokens"] = min(limits["burst_size"], bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = current_time

        # Check if we have tokens available
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1

            # Update request history for sliding window
            bucket["requests_history"].append(current_time)

            # Clean old requests (older than 1 minute)
            cutoff_time = current_time - 60
            while bucket["requests_history"] and bucket["requests_history"][0] < cutoff_time:
                bucket["requests_history"].popleft()

            return True

        # Rate limited
        self.violations[service] += 1
        self.violation_timestamps[service].append(current_time)

        # Clean old violations (older than 1 hour)
        cutoff_time = current_time - 3600
        self.violation_timestamps[service] = [ts for ts in self.violation_timestamps[service] if ts > cutoff_time]

        return False

    def _get_service_limits(self, service: str) -> Dict[str, int]:
        """Get rate limits for a specific service"""
        service_config = self.service_limits.get(service, {})

        return {
            "requests_per_minute": service_config.get(
                "requests_per_minute", self.default_limits["requests_per_minute"]
            ),
            "requests_per_second": service_config.get(
                "requests_per_second", self.default_limits["requests_per_second"]
            ),
            "burst_size": service_config.get("burst_size", self.default_limits["burst_size"]),
        }

    def get_retry_after(self, service: str) -> float:
        """
        Get recommended retry delay in seconds

        Args:
            service: Service name

        Returns:
            Seconds to wait before retrying
        """
        if not self.enabled:
            return 0

        bucket = self.buckets[service]
        limits = self._get_service_limits(service)

        # Calculate time until next token is available
        tokens_needed = 1 - bucket["tokens"]
        if tokens_needed <= 0:
            return 0

        # Time to generate one token
        token_generation_rate = limits["requests_per_minute"] / 60.0
        return tokens_needed / token_generation_rate

    def get_remaining_requests(self, service: str) -> Dict[str, Any]:
        """
        Get remaining requests for service in current window

        Args:
            service: Service name

        Returns:
            Dict with remaining request info
        """
        if not self.enabled:
            return {
                "tokens_available": float("inf"),
                "requests_in_window": 0,
                "window_reset_time": None,
                "rate_limited": False,
            }

        current_time = time.time()
        bucket = self.buckets[service]
        limits = self._get_service_limits(service)

        # Update bucket tokens
        time_elapsed = current_time - bucket["last_refill"]
        tokens_to_add = time_elapsed * (limits["requests_per_minute"] / 60.0)
        current_tokens = min(limits["burst_size"], bucket["tokens"] + tokens_to_add)

        # Count requests in current minute window
        cutoff_time = current_time - 60
        requests_in_window = sum(1 for ts in bucket["requests_history"] if ts > cutoff_time)

        # Calculate when window resets (oldest request + 60 seconds)
        window_reset_time = None
        if bucket["requests_history"]:
            oldest_request = bucket["requests_history"][0]
            window_reset_time = oldest_request + 60

        return {
            "tokens_available": int(current_tokens),
            "requests_in_window": requests_in_window,
            "window_reset_time": window_reset_time,
            "rate_limited": current_tokens < 1,
            "retry_after": self.get_retry_after(service) if current_tokens < 1 else 0,
        }

    def get_service_stats(self, service: str) -> Dict[str, Any]:
        """Get rate limiting statistics for a service"""
        bucket = self.buckets[service]
        limits = self._get_service_limits(service)
        current_time = time.time()

        # Count recent violations (last hour)
        recent_violations = len([ts for ts in self.violation_timestamps[service] if ts > current_time - 3600])

        return {
            "service": service,
            "limits": limits,
            "total_violations": self.violations[service],
            "recent_violations": recent_violations,
            "current_status": self.get_remaining_requests(service),
            "enabled": self.enabled,
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics for all services"""
        stats = {"enabled": self.enabled, "default_limits": self.default_limits, "services": {}}

        for service in self.buckets.keys():
            stats["services"][service] = self.get_service_stats(service)

        return stats

    def reset_service(self, service: str):
        """Reset rate limiting for a service"""
        if service in self.buckets:
            limits = self._get_service_limits(service)
            self.buckets[service] = {
                "tokens": limits["burst_size"],
                "last_refill": time.time(),
                "requests_history": deque(),
            }
            self.violations[service] = 0
            self.violation_timestamps[service] = []

    def reset_all(self):
        """Reset rate limiting for all services"""
        for service in list(self.buckets.keys()):
            self.reset_service(service)

    async def wait_for_capacity(self, service: str, max_wait: float = 60.0) -> bool:
        """
        Wait until service has capacity for a request

        Args:
            service: Service name
            max_wait: Maximum time to wait in seconds

        Returns:
            True if capacity is available, False if max_wait exceeded
        """
        if not self.enabled:
            return True

        start_time = time.time()

        while time.time() - start_time < max_wait:
            if await self.check_rate_limit(service):
                return True

            # Wait for recommended retry delay
            retry_after = self.get_retry_after(service)
            await asyncio.sleep(min(retry_after, 1.0))  # Wait at most 1 second at a time

        return False

    def update_service_limits(self, service: str, limits: Dict[str, int]):
        """
        Update rate limits for a service

        Args:
            service: Service name
            limits: New limits dict with requests_per_minute, requests_per_second, burst_size
        """
        self.service_limits[service] = limits

        # Reset the service to apply new limits
        self.reset_service(service)

    def enable(self):
        """Enable rate limiting"""
        self.enabled = True

    def disable(self):
        """Disable rate limiting"""
        self.enabled = False
