"""
Analytics Data Collectors

Collects metrics, performance data, and usage statistics from various
sources across the MCP platform for analysis and reporting.
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from database.connection import get_session
from database.repositories import (
    APIKeyRepository,
    CacheRepository,
    ErrorLogRepository,
    PerformanceRepository,
    QueryRepository,
    UserRepository,
)
from observability.logging import get_logger


class MetricType(str, Enum):
    """Types of metrics collected."""

    PERFORMANCE = "performance"
    USAGE = "usage"
    ERROR = "error"
    BUSINESS = "business"
    SYSTEM = "system"


@dataclass
class MetricPoint:
    """Individual metric data point."""

    timestamp: datetime
    metric_name: str
    metric_type: MetricType
    value: float
    unit: str
    tags: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class QueryMetrics:
    """Query execution metrics."""

    total_queries: int
    successful_queries: int
    failed_queries: int
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    queries_per_framework: Dict[str, int]
    queries_per_type: Dict[str, int]
    error_rate: float
    success_rate: float


@dataclass
class UserMetrics:
    """User activity metrics."""

    total_users: int
    active_users: int
    new_users: int
    queries_per_user: Dict[str, int]
    api_key_usage: Dict[str, int]
    most_active_users: List[Tuple[str, int]]


@dataclass
class SystemMetrics:
    """System performance metrics."""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    cache_hit_rate: float
    database_connections: int
    response_time_percentiles: Dict[str, float]


class MetricsCollector:
    """Central metrics collection service."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.collection_interval = 60  # seconds
        self._running = False

    def record_metric(self, metric: MetricPoint):
        """Record a single metric point."""
        self.metrics_buffer.append(metric)
        self.logger.debug(f"Recorded metric: {metric.metric_name} = {metric.value}")

    def record_query_metric(self, framework: str, query_type: str, response_time: float, success: bool):
        """Record query execution metric."""
        metric = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            metric_name="query_execution",
            metric_type=MetricType.PERFORMANCE,
            value=response_time,
            unit="ms",
            tags={"framework": framework, "query_type": query_type, "success": success},
        )
        self.record_metric(metric)

    def record_user_activity(self, user_id: str, action: str, metadata: Dict[str, Any] = None):
        """Record user activity metric."""
        metric = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            metric_name="user_activity",
            metric_type=MetricType.USAGE,
            value=1,
            unit="count",
            tags={"user_id": user_id, "action": action, **(metadata or {})},
        )
        self.record_metric(metric)

    def record_error_metric(self, error_type: str, severity: str, framework: str = None):
        """Record error occurrence metric."""
        metric = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            metric_name="error_occurrence",
            metric_type=MetricType.ERROR,
            value=1,
            unit="count",
            tags={"error_type": error_type, "severity": severity, "framework": framework},
        )
        self.record_metric(metric)

    def get_recent_metrics(self, hours: int = 1) -> List[MetricPoint]:
        """Get recent metrics from buffer."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [m for m in self.metrics_buffer if m.timestamp >= cutoff]

    def start_collection(self):
        """Start background metrics collection."""
        self._running = True
        asyncio.create_task(self._collection_loop())
        self.logger.info("Metrics collection started")

    def stop_collection(self):
        """Stop background metrics collection."""
        self._running = False
        self.logger.info("Metrics collection stopped")

    async def _collection_loop(self):
        """Background collection loop."""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(10)

    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric(
                MetricPoint(
                    timestamp=datetime.now(timezone.utc),
                    metric_name="cpu_usage",
                    metric_type=MetricType.SYSTEM,
                    value=cpu_percent,
                    unit="percent",
                )
            )

            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric(
                MetricPoint(
                    timestamp=datetime.now(timezone.utc),
                    metric_name="memory_usage",
                    metric_type=MetricType.SYSTEM,
                    value=memory.percent,
                    unit="percent",
                )
            )

            # Disk usage
            disk = psutil.disk_usage('/')
            self.record_metric(
                MetricPoint(
                    timestamp=datetime.now(timezone.utc),
                    metric_name="disk_usage",
                    metric_type=MetricType.SYSTEM,
                    value=(disk.used / disk.total) * 100,
                    unit="percent",
                )
            )

        except ImportError:
            self.logger.warning("psutil not available for system metrics collection")
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")


class QueryAnalyzer:
    """Analyzes query patterns and performance."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.query_repo = QueryRepository()

    def get_query_metrics(self, hours: int = 24, framework_id: str = None) -> QueryMetrics:
        """Get comprehensive query metrics."""
        try:
            with get_session() as session:
                # Get query performance stats
                stats = self.query_repo.get_performance_stats(session, framework_id, hours)

                # Get recent queries for detailed analysis
                since = datetime.now(timezone.utc) - timedelta(hours=hours)

                if framework_id:
                    queries = self.query_repo.get_by_framework(session, framework_id, limit=1000)
                else:
                    # Get recent queries across all frameworks
                    queries = (
                        session.query(self.query_repo.model_class)
                        .filter(self.query_repo.model_class.created_at >= since)
                        .all()
                    )

                # Calculate detailed metrics
                response_times = [q.execution_time_ms for q in queries]

                # Group by framework
                queries_per_framework = defaultdict(int)
                for query in queries:
                    framework_name = getattr(query.framework, 'name', 'unknown') if query.framework else 'unknown'
                    queries_per_framework[framework_name] += 1

                # Group by query type
                queries_per_type = defaultdict(int)
                for query in queries:
                    queries_per_type[query.query_type] += 1

                return QueryMetrics(
                    total_queries=stats['total_queries'],
                    successful_queries=stats['successful_queries'],
                    failed_queries=stats['failed_queries'],
                    average_response_time=stats['avg_response_time_ms'],
                    median_response_time=statistics.median(response_times) if response_times else 0,
                    p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else 0,
                    queries_per_framework=dict(queries_per_framework),
                    queries_per_type=dict(queries_per_type),
                    error_rate=(100 - stats['success_rate']),
                    success_rate=stats['success_rate'],
                )

        except Exception as e:
            self.logger.error(f"Error getting query metrics: {e}")
            return QueryMetrics(
                total_queries=0,
                successful_queries=0,
                failed_queries=0,
                average_response_time=0,
                median_response_time=0,
                p95_response_time=0,
                queries_per_framework={},
                queries_per_type={},
                error_rate=0,
                success_rate=0,
            )

    def analyze_query_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze query patterns and trends."""
        try:
            with get_session() as session:
                since = datetime.now(timezone.utc) - timedelta(hours=hours)

                queries = (
                    session.query(self.query_repo.model_class)
                    .filter(self.query_repo.model_class.created_at >= since)
                    .all()
                )

                # Time-based patterns
                hourly_counts = defaultdict(int)
                for query in queries:
                    hour = query.created_at.hour
                    hourly_counts[hour] += 1

                # Most common query types
                query_types = defaultdict(int)
                for query in queries:
                    query_types[query.query_type] += 1

                # Performance by framework
                framework_performance = defaultdict(list)
                for query in queries:
                    framework_name = getattr(query.framework, 'name', 'unknown') if query.framework else 'unknown'
                    framework_performance[framework_name].append(query.execution_time_ms)

                # Calculate average performance per framework
                avg_performance = {}
                for framework, times in framework_performance.items():
                    avg_performance[framework] = sum(times) / len(times) if times else 0

                return {
                    'hourly_distribution': dict(hourly_counts),
                    'query_type_distribution': dict(query_types),
                    'framework_performance': avg_performance,
                    'peak_hour': max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else None,
                    'total_unique_users': len(set(q.user_id for q in queries if q.user_id)),
                    'analysis_period_hours': hours,
                }

        except Exception as e:
            self.logger.error(f"Error analyzing query patterns: {e}")
            return {}

    def get_slow_queries(self, threshold_ms: float = 5000, limit: int = 50) -> List[Dict[str, Any]]:
        """Get slow queries above threshold."""
        try:
            with get_session() as session:
                queries = (
                    session.query(self.query_repo.model_class)
                    .filter(self.query_repo.model_class.execution_time_ms > threshold_ms)
                    .order_by(self.query_repo.model_class.execution_time_ms.desc())
                    .limit(limit)
                    .all()
                )

                return [
                    {
                        'id': query.id,
                        'query_text': query.query_text[:200] + '...'
                        if len(query.query_text) > 200
                        else query.query_text,
                        'execution_time_ms': query.execution_time_ms,
                        'framework': getattr(query.framework, 'name', 'unknown') if query.framework else 'unknown',
                        'query_type': query.query_type,
                        'created_at': query.created_at.isoformat(),
                        'success': query.success,
                        'user_id': query.user_id,
                    }
                    for query in queries
                ]

        except Exception as e:
            self.logger.error(f"Error getting slow queries: {e}")
            return []


class PerformanceAnalyzer:
    """Analyzes system and application performance."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.perf_repo = PerformanceRepository()
        self.error_repo = ErrorLogRepository()
        self.cache_repo = CacheRepository()

    def get_system_performance(self, hours: int = 24) -> SystemMetrics:
        """Get comprehensive system performance metrics."""
        try:
            with get_session() as session:
                # Get performance metrics
                metrics = self.perf_repo.get_operation_metrics(session, "system", hours)

                # Calculate basic stats
                cpu_usage = 0
                memory_usage = 0
                response_times = []

                for metric in metrics:
                    if metric.operation == "cpu_usage":
                        cpu_usage = metric.value
                    elif metric.operation == "memory_usage":
                        memory_usage = metric.value
                    elif "response_time" in metric.operation:
                        response_times.append(metric.value)

                # Calculate percentiles
                percentiles = {}
                if response_times:
                    sorted_times = sorted(response_times)
                    percentiles = {
                        'p50': statistics.median(sorted_times),
                        'p95': statistics.quantiles(sorted_times, n=20)[18]
                        if len(sorted_times) > 20
                        else max(sorted_times),
                        'p99': statistics.quantiles(sorted_times, n=100)[98]
                        if len(sorted_times) > 100
                        else max(sorted_times),
                    }

                # Get cache statistics
                cache_stats = self.cache_repo.get_cache_stats(session)
                cache_hit_rate = cache_stats.get('hit_rate_estimate', 0)

                return SystemMetrics(
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    disk_usage=0,  # Would need system integration
                    network_io=0,  # Would need system integration
                    cache_hit_rate=cache_hit_rate,
                    database_connections=0,  # Would need database monitoring
                    response_time_percentiles=percentiles,
                )

        except Exception as e:
            self.logger.error(f"Error getting system performance: {e}")
            return SystemMetrics(
                cpu_usage=0,
                memory_usage=0,
                disk_usage=0,
                network_io=0,
                cache_hit_rate=0,
                database_connections=0,
                response_time_percentiles={},
            )

    def analyze_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        try:
            with get_session() as session:
                since = datetime.now(timezone.utc) - timedelta(days=days)

                # Get performance metrics over time
                metrics = (
                    session.query(self.perf_repo.model_class)
                    .filter(self.perf_repo.model_class.time_bucket >= since)
                    .order_by(self.perf_repo.model_class.time_bucket)
                    .all()
                )

                # Group by day and operation
                daily_metrics = defaultdict(lambda: defaultdict(list))

                for metric in metrics:
                    day = metric.time_bucket.date()
                    daily_metrics[day][metric.operation].append(metric.value)

                # Calculate daily averages
                daily_averages = {}
                for day, operations in daily_metrics.items():
                    daily_averages[day.isoformat()] = {
                        op: sum(values) / len(values) for op, values in operations.items()
                    }

                # Identify trends
                trends = {}
                for operation in ['response_time', 'throughput', 'error_rate']:
                    values = []
                    for day_data in daily_averages.values():
                        if operation in day_data:
                            values.append(day_data[operation])

                    if len(values) > 1:
                        # Simple trend calculation (positive = improving, negative = degrading)
                        trend = (values[-1] - values[0]) / len(values)
                        trends[operation] = {
                            'trend': trend,
                            'direction': 'improving' if trend < 0 else 'degrading',
                            'current_value': values[-1],
                            'previous_value': values[0],
                        }

                return {
                    'daily_averages': daily_averages,
                    'trends': trends,
                    'analysis_period_days': days,
                    'total_data_points': len(metrics),
                }

        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {e}")
            return {}

    def get_error_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze error patterns and frequencies."""
        try:
            with get_session() as session:
                errors = self.error_repo.get_recent_errors(session, hours, limit=1000)

                # Error distribution by category
                by_category = defaultdict(int)
                by_severity = defaultdict(int)
                by_framework = defaultdict(int)

                for error in errors:
                    by_category[error.category] += 1
                    by_severity[error.severity] += 1
                    if error.framework_name:
                        by_framework[error.framework_name] += 1

                # Most common errors
                error_messages = defaultdict(int)
                for error in errors:
                    # Group similar errors by first 100 chars of message
                    message_key = error.message[:100] + "..." if len(error.message) > 100 else error.message
                    error_messages[message_key] += 1

                # Sort by frequency
                top_errors = sorted(error_messages.items(), key=lambda x: x[1], reverse=True)[:10]

                return {
                    'total_errors': len(errors),
                    'by_category': dict(by_category),
                    'by_severity': dict(by_severity),
                    'by_framework': dict(by_framework),
                    'top_errors': top_errors,
                    'error_rate_per_hour': len(errors) / hours if hours > 0 else 0,
                }

        except Exception as e:
            self.logger.error(f"Error analyzing error patterns: {e}")
            return {}
