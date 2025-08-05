"""
Framework Performance Monitoring

Provides comprehensive monitoring and metrics collection for AI frameworks
including performance tracking, error rates, and usage analytics.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class MetricType(Enum):
    """Types of metrics that can be collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class FrameworkMetric:
    """Individual framework metric."""

    framework: str
    operation: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['metric_type'] = self.metric_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class FrameworkPerformanceMonitor:
    """Monitors and collects performance metrics for AI frameworks."""

    def __init__(self, max_history: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.max_history = max_history

        # Metric storage
        self.metrics: List[FrameworkMetric] = []
        self.framework_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_execution_time": 0.0,
                "avg_execution_time": 0.0,
                "min_execution_time": float('inf'),
                "max_execution_time": 0.0,
                "last_request": None,
                "error_rate": 0.0,
                "requests_per_minute": deque(maxlen=60),  # Last 60 minutes
                "operation_counts": defaultdict(int),
                "recent_errors": deque(maxlen=10),
            }
        )

        # Active request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}

        self.logger.info("Framework performance monitor initialized")

    def start_request_tracking(self, framework: str, operation: str, request_id: str = None) -> str:
        """Start tracking a request."""
        if not request_id:
            request_id = f"{framework}_{operation}_{int(time.time() * 1000)}"

        self.active_requests[request_id] = {
            "framework": framework,
            "operation": operation,
            "start_time": time.time(),
            "timestamp": datetime.now(),
        }

        return request_id

    def end_request_tracking(
        self, request_id: str, success: bool = True, error: str = None, metadata: Dict[str, Any] = None
    ):
        """End tracking a request and record metrics."""
        if request_id not in self.active_requests:
            self.logger.warning(f"Request ID {request_id} not found in active requests")
            return

        request_info = self.active_requests.pop(request_id)
        framework = request_info["framework"]
        operation = request_info["operation"]
        execution_time = time.time() - request_info["start_time"]

        # Update framework stats
        stats = self.framework_stats[framework]
        stats["total_requests"] += 1
        stats["last_request"] = datetime.now()
        stats["operation_counts"][operation] += 1

        if success:
            stats["successful_requests"] += 1
        else:
            stats["failed_requests"] += 1
            if error:
                stats["recent_errors"].append(
                    {"operation": operation, "error": error, "timestamp": datetime.now().isoformat()}
                )

        # Update execution time stats
        stats["total_execution_time"] += execution_time
        stats["avg_execution_time"] = stats["total_execution_time"] / stats["total_requests"]
        stats["min_execution_time"] = min(stats["min_execution_time"], execution_time)
        stats["max_execution_time"] = max(stats["max_execution_time"], execution_time)

        # Update error rate
        stats["error_rate"] = (stats["failed_requests"] / stats["total_requests"]) * 100

        # Add to requests per minute tracking
        current_minute = datetime.now().replace(second=0, microsecond=0)
        stats["requests_per_minute"].append({"minute": current_minute, "count": 1})

        # Record individual metric
        metric = FrameworkMetric(
            framework=framework,
            operation=operation,
            metric_type=MetricType.TIMER,
            value=execution_time * 1000,  # Convert to milliseconds
            timestamp=datetime.now(),
            metadata={"success": success, "error": error, **(metadata or {})},
        )

        self._add_metric(metric)

        self.logger.debug(f"Recorded {operation} request for {framework}: {execution_time:.3f}s, success={success}")

    def record_counter(self, framework: str, operation: str, value: int = 1, metadata: Dict[str, Any] = None):
        """Record a counter metric."""
        metric = FrameworkMetric(
            framework=framework,
            operation=operation,
            metric_type=MetricType.COUNTER,
            value=value,
            timestamp=datetime.now(),
            metadata=metadata,
        )
        self._add_metric(metric)

    def record_gauge(self, framework: str, operation: str, value: float, metadata: Dict[str, Any] = None):
        """Record a gauge metric."""
        metric = FrameworkMetric(
            framework=framework,
            operation=operation,
            metric_type=MetricType.GAUGE,
            value=value,
            timestamp=datetime.now(),
            metadata=metadata,
        )
        self._add_metric(metric)

    def _add_metric(self, metric: FrameworkMetric):
        """Add a metric to storage."""
        self.metrics.append(metric)

        # Keep only recent metrics
        if len(self.metrics) > self.max_history:
            self.metrics = self.metrics[-self.max_history :]

    def get_framework_stats(self, framework: str = None) -> Dict[str, Any]:
        """Get comprehensive stats for a framework or all frameworks."""
        if framework:
            if framework not in self.framework_stats:
                return {}

            stats = dict(self.framework_stats[framework])
            # Convert deque to list for serialization
            stats["requests_per_minute"] = list(stats["requests_per_minute"])
            stats["recent_errors"] = list(stats["recent_errors"])
            stats["operation_counts"] = dict(stats["operation_counts"])

            # Fix infinite values for serialization
            if stats["min_execution_time"] == float('inf'):
                stats["min_execution_time"] = 0.0

            return stats
        else:
            # Return stats for all frameworks
            all_stats = {}
            for fw_name in self.framework_stats:
                all_stats[fw_name] = self.get_framework_stats(fw_name)
            return all_stats

    def get_metrics(
        self, framework: str = None, operation: str = None, since: datetime = None, limit: int = None
    ) -> List[Dict[str, Any]]:
        """Get metrics with optional filtering."""
        filtered_metrics = self.metrics

        # Apply filters
        if framework:
            filtered_metrics = [m for m in filtered_metrics if m.framework == framework]

        if operation:
            filtered_metrics = [m for m in filtered_metrics if m.operation == operation]

        if since:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= since]

        # Apply limit
        if limit:
            filtered_metrics = filtered_metrics[-limit:]

        return [metric.to_dict() for metric in filtered_metrics]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary."""
        total_requests = sum(stats["total_requests"] for stats in self.framework_stats.values())
        total_errors = sum(stats["failed_requests"] for stats in self.framework_stats.values())

        # Calculate average response times per framework
        framework_performance = {}
        for framework, stats in self.framework_stats.items():
            if stats["total_requests"] > 0:
                framework_performance[framework] = {
                    "avg_response_time_ms": stats["avg_execution_time"] * 1000,
                    "success_rate": ((stats["successful_requests"] / stats["total_requests"]) * 100),
                    "total_requests": stats["total_requests"],
                    "requests_per_hour": self._calculate_requests_per_hour(framework),
                }

        return {
            "timestamp": datetime.now().isoformat(),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "overall_error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0,
            "active_requests": len(self.active_requests),
            "frameworks": framework_performance,
            "top_operations": self._get_top_operations(),
            "recent_error_count": sum(len(stats["recent_errors"]) for stats in self.framework_stats.values()),
        }

    def _calculate_requests_per_hour(self, framework: str) -> float:
        """Calculate requests per hour for a framework."""
        stats = self.framework_stats[framework]
        recent_requests = stats["requests_per_minute"]

        if not recent_requests:
            return 0.0

        # Count requests in the last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_count = sum(req["count"] for req in recent_requests if req["minute"] >= one_hour_ago)

        return recent_count

    def _get_top_operations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most frequently used operations across all frameworks."""
        operation_totals = defaultdict(int)

        for framework_stats in self.framework_stats.values():
            for operation, count in framework_stats["operation_counts"].items():
                operation_totals[operation] += count

        # Sort by count and return top operations
        sorted_operations = sorted(operation_totals.items(), key=lambda x: x[1], reverse=True)

        return [{"operation": op, "total_requests": count} for op, count in sorted_operations[:limit]]

    def reset_metrics(self, framework: str = None):
        """Reset metrics for a specific framework or all frameworks."""
        if framework:
            if framework in self.framework_stats:
                del self.framework_stats[framework]
            self.metrics = [m for m in self.metrics if m.framework != framework]
            self.logger.info(f"Reset metrics for framework: {framework}")
        else:
            self.framework_stats.clear()
            self.metrics.clear()
            self.active_requests.clear()
            self.logger.info("Reset all framework metrics")

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "summary": self.get_performance_summary(),
            "framework_stats": self.get_framework_stats(),
            "recent_metrics": self.get_metrics(limit=100),
        }

        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global monitor instance
framework_monitor = FrameworkPerformanceMonitor()


class FrameworkMonitoringMiddleware:
    """Middleware for automatic framework performance monitoring."""

    def __init__(self, monitor: FrameworkPerformanceMonitor = None):
        self.monitor = monitor or framework_monitor

    async def __call__(self, framework: str, operation: str, func, *args, **kwargs):
        """Execute function with monitoring."""
        request_id = self.monitor.start_request_tracking(framework, operation)

        try:
            result = await func(*args, **kwargs)

            # Check if result indicates success
            success = True
            error = None

            if isinstance(result, dict):
                success = result.get("success", True)
                error = result.get("error") if not success else None

            self.monitor.end_request_tracking(request_id, success=success, error=error)
            return result

        except Exception as e:
            self.monitor.end_request_tracking(request_id, success=False, error=str(e))
            raise


# Decorator for easy monitoring
def monitor_framework_operation(framework: str, operation: str):
    """Decorator to automatically monitor framework operations."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            middleware = FrameworkMonitoringMiddleware()
            return await middleware(framework, operation, func, *args, **kwargs)

        return wrapper

    return decorator
