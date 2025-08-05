"""
Analytics Data Aggregators

Aggregates and processes raw metrics data into meaningful insights,
time-series data, and statistical summaries for reporting and analysis.
"""

import logging
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from database.connection import get_session
from database.repositories import ErrorLogRepository, PerformanceRepository, QueryRepository
from observability.logging import get_logger

from .collectors import MetricPoint, MetricType


class AggregationPeriod(str, Enum):
    """Time periods for data aggregation."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class AggregationFunction(str, Enum):
    """Aggregation functions."""

    SUM = "sum"
    AVERAGE = "avg"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    STANDARD_DEVIATION = "stddev"


@dataclass
class TimeSeriesPoint:
    """Single point in time series data."""

    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AggregatedMetric:
    """Aggregated metric result."""

    metric_name: str
    aggregation_function: AggregationFunction
    period: AggregationPeriod
    value: float
    timestamp: datetime
    data_points: int
    tags: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class TimeSeriesData:
    """Complete time series dataset."""

    metric_name: str
    period: AggregationPeriod
    start_time: datetime
    end_time: datetime
    data_points: List[TimeSeriesPoint]
    aggregation_function: AggregationFunction
    total_points: int


class DataAggregator:
    """Base data aggregation functionality."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def aggregate_by_period(
        self,
        data: List[MetricPoint],
        period: AggregationPeriod,
        function: AggregationFunction = AggregationFunction.AVERAGE,
    ) -> List[AggregatedMetric]:
        """Aggregate metrics by time period."""
        try:
            if not data:
                return []

            # Group data by time period
            grouped_data = self._group_by_period(data, period)

            aggregated = []
            for period_key, points in grouped_data.items():
                if not points:
                    continue

                # Calculate aggregated value
                values = [p.value for p in points]
                aggregated_value = self._apply_aggregation_function(values, function)

                # Create aggregated metric
                metric = AggregatedMetric(
                    metric_name=points[0].metric_name,
                    aggregation_function=function,
                    period=period,
                    value=aggregated_value,
                    timestamp=period_key,
                    data_points=len(points),
                    tags=points[0].tags,
                )
                aggregated.append(metric)

            return sorted(aggregated, key=lambda x: x.timestamp)

        except Exception as e:
            self.logger.error(f"Error aggregating data by period: {e}")
            return []

    def _group_by_period(self, data: List[MetricPoint], period: AggregationPeriod) -> Dict[datetime, List[MetricPoint]]:
        """Group data points by time period."""
        grouped = defaultdict(list)

        for point in data:
            period_key = self._get_period_key(point.timestamp, period)
            grouped[period_key].append(point)

        return dict(grouped)

    def _get_period_key(self, timestamp: datetime, period: AggregationPeriod) -> datetime:
        """Get period key for timestamp."""
        if period == AggregationPeriod.MINUTE:
            return timestamp.replace(second=0, microsecond=0)
        elif period == AggregationPeriod.HOUR:
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif period == AggregationPeriod.DAY:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == AggregationPeriod.WEEK:
            # Start of week (Monday)
            days_since_monday = timestamp.weekday()
            week_start = timestamp - timedelta(days=days_since_monday)
            return week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == AggregationPeriod.MONTH:
            return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp

    def _apply_aggregation_function(self, values: List[float], function: AggregationFunction) -> float:
        """Apply aggregation function to values."""
        if not values:
            return 0.0

        if function == AggregationFunction.SUM:
            return sum(values)
        elif function == AggregationFunction.AVERAGE:
            return sum(values) / len(values)
        elif function == AggregationFunction.COUNT:
            return len(values)
        elif function == AggregationFunction.MIN:
            return min(values)
        elif function == AggregationFunction.MAX:
            return max(values)
        elif function == AggregationFunction.MEDIAN:
            return statistics.median(values)
        elif function == AggregationFunction.PERCENTILE_95:
            if len(values) >= 20:
                return statistics.quantiles(values, n=20)[18]
            return max(values)
        elif function == AggregationFunction.PERCENTILE_99:
            if len(values) >= 100:
                return statistics.quantiles(values, n=100)[98]
            return max(values)
        elif function == AggregationFunction.STANDARD_DEVIATION:
            if len(values) > 1:
                return statistics.stdev(values)
            return 0.0
        else:
            return sum(values) / len(values)  # Default to average


class TimeSeriesAggregator(DataAggregator):
    """Specialized aggregator for time series data."""

    def create_time_series(
        self,
        data: List[MetricPoint],
        period: AggregationPeriod,
        function: AggregationFunction = AggregationFunction.AVERAGE,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> TimeSeriesData:
        """Create time series from metric points."""
        try:
            if not data:
                return TimeSeriesData(
                    metric_name="unknown",
                    period=period,
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc),
                    data_points=[],
                    aggregation_function=function,
                    total_points=0,
                )

            # Filter by time range if specified
            filtered_data = data
            if start_time or end_time:
                filtered_data = [
                    p
                    for p in data
                    if (not start_time or p.timestamp >= start_time) and (not end_time or p.timestamp <= end_time)
                ]

            # Aggregate data
            aggregated = self.aggregate_by_period(filtered_data, period, function)

            # Convert to time series points
            time_points = [
                TimeSeriesPoint(timestamp=agg.timestamp, value=agg.value, metadata={'data_points': agg.data_points})
                for agg in aggregated
            ]

            # Determine time range
            if filtered_data:
                actual_start = min(p.timestamp for p in filtered_data)
                actual_end = max(p.timestamp for p in filtered_data)
            else:
                actual_start = actual_end = datetime.now(timezone.utc)

            return TimeSeriesData(
                metric_name=data[0].metric_name,
                period=period,
                start_time=start_time or actual_start,
                end_time=end_time or actual_end,
                data_points=time_points,
                aggregation_function=function,
                total_points=len(filtered_data),
            )

        except Exception as e:
            self.logger.error(f"Error creating time series: {e}")
            raise

    def resample_time_series(
        self, time_series: TimeSeriesData, new_period: AggregationPeriod, new_function: AggregationFunction = None
    ) -> TimeSeriesData:
        """Resample existing time series to different period/function."""
        try:
            # Convert time series points back to metric points
            metric_points = []
            for point in time_series.data_points:
                # Create multiple points based on data_points metadata
                data_points_count = point.metadata.get('data_points', 1)

                # Distribute the value across the original data points
                point_value = point.value / data_points_count if data_points_count > 0 else point.value

                for _ in range(data_points_count):
                    metric_points.append(
                        MetricPoint(
                            timestamp=point.timestamp,
                            metric_name=time_series.metric_name,
                            metric_type=MetricType.PERFORMANCE,  # Default
                            value=point_value,
                            unit="resampled",
                        )
                    )

            # Create new time series with different aggregation
            function = new_function or time_series.aggregation_function
            return self.create_time_series(
                metric_points, new_period, function, time_series.start_time, time_series.end_time
            )

        except Exception as e:
            self.logger.error(f"Error resampling time series: {e}")
            raise

    def fill_gaps(self, time_series: TimeSeriesData, fill_method: str = "interpolate") -> TimeSeriesData:
        """Fill gaps in time series data."""
        try:
            if not time_series.data_points:
                return time_series

            # Sort points by timestamp
            sorted_points = sorted(time_series.data_points, key=lambda x: x.timestamp)

            # Generate expected timestamps based on period
            expected_timestamps = self._generate_expected_timestamps(
                time_series.start_time, time_series.end_time, time_series.period
            )

            # Create filled time series
            filled_points = []
            point_index = 0

            for expected_ts in expected_timestamps:
                # Check if we have a point for this timestamp
                if point_index < len(sorted_points) and sorted_points[point_index].timestamp == expected_ts:
                    # Use existing point
                    filled_points.append(sorted_points[point_index])
                    point_index += 1
                else:
                    # Fill gap
                    filled_value = self._fill_gap_value(expected_ts, sorted_points, point_index, fill_method)

                    filled_points.append(
                        TimeSeriesPoint(
                            timestamp=expected_ts, value=filled_value, metadata={'filled': True, 'method': fill_method}
                        )
                    )

            return TimeSeriesData(
                metric_name=time_series.metric_name,
                period=time_series.period,
                start_time=time_series.start_time,
                end_time=time_series.end_time,
                data_points=filled_points,
                aggregation_function=time_series.aggregation_function,
                total_points=time_series.total_points,
            )

        except Exception as e:
            self.logger.error(f"Error filling gaps: {e}")
            return time_series

    def _generate_expected_timestamps(
        self, start_time: datetime, end_time: datetime, period: AggregationPeriod
    ) -> List[datetime]:
        """Generate expected timestamps for period."""
        timestamps = []
        current = self._get_period_key(start_time, period)

        # Determine increment
        if period == AggregationPeriod.MINUTE:
            increment = timedelta(minutes=1)
        elif period == AggregationPeriod.HOUR:
            increment = timedelta(hours=1)
        elif period == AggregationPeriod.DAY:
            increment = timedelta(days=1)
        elif period == AggregationPeriod.WEEK:
            increment = timedelta(weeks=1)
        elif period == AggregationPeriod.MONTH:
            increment = timedelta(days=30)  # Approximate
        else:
            increment = timedelta(hours=1)  # Default

        while current <= end_time:
            timestamps.append(current)
            current += increment

        return timestamps

    def _fill_gap_value(
        self, timestamp: datetime, points: List[TimeSeriesPoint], current_index: int, fill_method: str
    ) -> float:
        """Calculate value for gap filling."""
        if fill_method == "zero":
            return 0.0

        elif fill_method == "forward_fill":
            # Use last known value
            if current_index > 0:
                return points[current_index - 1].value
            return 0.0

        elif fill_method == "backward_fill":
            # Use next known value
            if current_index < len(points):
                return points[current_index].value
            return 0.0

        elif fill_method == "interpolate":
            # Linear interpolation
            if current_index > 0 and current_index < len(points):
                prev_point = points[current_index - 1]
                next_point = points[current_index]

                # Calculate interpolated value
                time_diff = (next_point.timestamp - prev_point.timestamp).total_seconds()
                target_diff = (timestamp - prev_point.timestamp).total_seconds()

                if time_diff > 0:
                    ratio = target_diff / time_diff
                    value_diff = next_point.value - prev_point.value
                    return prev_point.value + (ratio * value_diff)

            # Fallback to forward fill
            return self._fill_gap_value(timestamp, points, current_index, "forward_fill")

        else:
            return 0.0


class MetricsAggregator:
    """Aggregates platform metrics from database."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.query_repo = QueryRepository()
        self.perf_repo = PerformanceRepository()
        self.error_repo = ErrorLogRepository()
        self.time_series_aggregator = TimeSeriesAggregator()

    def get_query_metrics_time_series(
        self, hours: int = 24, period: AggregationPeriod = AggregationPeriod.HOUR, framework_id: str = None
    ) -> Dict[str, TimeSeriesData]:
        """Get query metrics as time series."""
        try:
            with get_session() as session:
                since = datetime.now(timezone.utc) - timedelta(hours=hours)

                # Get queries
                if framework_id:
                    queries = self.query_repo.get_by_framework(session, framework_id, limit=10000)
                else:
                    queries = (
                        session.query(self.query_repo.model_class)
                        .filter(self.query_repo.model_class.created_at >= since)
                        .all()
                    )

                # Convert to metric points
                response_time_points = []
                throughput_points = []
                error_rate_points = []

                # Group by time period for throughput calculation
                period_groups = defaultdict(list)

                for query in queries:
                    period_key = self.time_series_aggregator._get_period_key(query.created_at, period)
                    period_groups[period_key].append(query)

                    # Response time metric
                    response_time_points.append(
                        MetricPoint(
                            timestamp=query.created_at,
                            metric_name="response_time",
                            metric_type=MetricType.PERFORMANCE,
                            value=query.execution_time_ms,
                            unit="ms",
                        )
                    )

                # Calculate throughput and error rate per period
                for period_key, period_queries in period_groups.items():
                    # Throughput (queries per period)
                    throughput_points.append(
                        MetricPoint(
                            timestamp=period_key,
                            metric_name="throughput",
                            metric_type=MetricType.PERFORMANCE,
                            value=len(period_queries),
                            unit="queries",
                        )
                    )

                    # Error rate
                    failed_queries = sum(1 for q in period_queries if not q.success)
                    error_rate = (failed_queries / len(period_queries)) * 100 if period_queries else 0

                    error_rate_points.append(
                        MetricPoint(
                            timestamp=period_key,
                            metric_name="error_rate",
                            metric_type=MetricType.ERROR,
                            value=error_rate,
                            unit="percent",
                        )
                    )

                # Create time series
                return {
                    'response_time': self.time_series_aggregator.create_time_series(
                        response_time_points, period, AggregationFunction.AVERAGE
                    ),
                    'throughput': self.time_series_aggregator.create_time_series(
                        throughput_points, period, AggregationFunction.SUM
                    ),
                    'error_rate': self.time_series_aggregator.create_time_series(
                        error_rate_points, period, AggregationFunction.AVERAGE
                    ),
                }

        except Exception as e:
            self.logger.error(f"Error getting query metrics time series: {e}")
            return {}

    def get_system_metrics_aggregated(
        self, hours: int = 24, function: AggregationFunction = AggregationFunction.AVERAGE
    ) -> Dict[str, float]:
        """Get aggregated system metrics."""
        try:
            with get_session() as session:
                since = datetime.now(timezone.utc) - timedelta(hours=hours)

                metrics = (
                    session.query(self.perf_repo.model_class)
                    .filter(self.perf_repo.model_class.time_bucket >= since)
                    .all()
                )

                # Group by operation
                grouped_metrics = defaultdict(list)
                for metric in metrics:
                    grouped_metrics[metric.operation].append(metric.value)

                # Apply aggregation function
                aggregated = {}
                data_aggregator = DataAggregator()

                for operation, values in grouped_metrics.items():
                    aggregated[operation] = data_aggregator._apply_aggregation_function(values, function)

                return aggregated

        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {}

    def get_error_trends(self, days: int = 7, period: AggregationPeriod = AggregationPeriod.DAY) -> TimeSeriesData:
        """Get error trends over time."""
        try:
            with get_session() as session:
                since = datetime.now(timezone.utc) - timedelta(days=days)

                errors = self.error_repo.get_recent_errors(session, days * 24, limit=10000)

                # Convert to metric points
                error_points = []
                for error in errors:
                    error_points.append(
                        MetricPoint(
                            timestamp=error.created_at,
                            metric_name="error_count",
                            metric_type=MetricType.ERROR,
                            value=1,
                            unit="count",
                        )
                    )

                return self.time_series_aggregator.create_time_series(error_points, period, AggregationFunction.SUM)

        except Exception as e:
            self.logger.error(f"Error getting error trends: {e}")
            return TimeSeriesData(
                metric_name="error_count",
                period=period,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                data_points=[],
                aggregation_function=AggregationFunction.SUM,
                total_points=0,
            )
