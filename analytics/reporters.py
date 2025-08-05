"""
Analytics Reporters

Generates reports, dashboards, and exports analytics data in various formats
for business intelligence and operational monitoring.
"""

import asyncio
import io
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from observability.logging import get_logger

from .collectors import MetricsCollector, PerformanceAnalyzer, QueryAnalyzer


class ReportFormat(str, Enum):
    """Report output formats."""

    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    HTML = "html"
    EXCEL = "xlsx"


class ReportType(str, Enum):
    """Types of reports available."""

    PERFORMANCE = "performance"
    USAGE = "usage"
    ERROR_ANALYSIS = "error_analysis"
    USER_ACTIVITY = "user_activity"
    SYSTEM_HEALTH = "system_health"
    CUSTOM = "custom"


@dataclass
class ReportConfig:
    """Report generation configuration."""

    report_type: ReportType
    format: ReportFormat
    time_range_hours: int = 24
    include_charts: bool = True
    include_raw_data: bool = False
    filters: Dict[str, Any] = None

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


@dataclass
class DashboardWidget:
    """Dashboard widget definition."""

    widget_id: str
    title: str
    widget_type: str  # chart, metric, table, gauge
    data_source: str
    config: Dict[str, Any] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}


@dataclass
class DashboardData:
    """Complete dashboard data structure."""

    dashboard_id: str
    title: str
    description: str
    widgets: List[DashboardWidget]
    last_updated: datetime
    auto_refresh_seconds: int = 30


class ReportGenerator:
    """Generates various types of analytics reports."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics_collector = MetricsCollector()
        self.query_analyzer = QueryAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()

    async def generate_report(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate report based on configuration."""
        try:
            self.logger.info(f"Generating {config.report_type} report in {config.format} format")

            # Generate report data based on type
            if config.report_type == ReportType.PERFORMANCE:
                data = await self._generate_performance_report(config)
            elif config.report_type == ReportType.USAGE:
                data = await self._generate_usage_report(config)
            elif config.report_type == ReportType.ERROR_ANALYSIS:
                data = await self._generate_error_report(config)
            elif config.report_type == ReportType.USER_ACTIVITY:
                data = await self._generate_user_activity_report(config)
            elif config.report_type == ReportType.SYSTEM_HEALTH:
                data = await self._generate_system_health_report(config)
            else:
                raise ValueError(f"Unsupported report type: {config.report_type}")

            # Format the report
            formatted_report = await self._format_report(data, config)

            self.logger.info(f"Successfully generated {config.report_type} report")
            return formatted_report

        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise

    async def _generate_performance_report(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate performance analytics report."""
        query_metrics = self.query_analyzer.get_query_metrics(config.time_range_hours)
        system_metrics = self.performance_analyzer.get_system_performance(config.time_range_hours)
        trends = self.performance_analyzer.analyze_performance_trends(config.time_range_hours // 24 or 1)
        slow_queries = self.query_analyzer.get_slow_queries()

        return {
            'report_type': 'performance',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'time_range_hours': config.time_range_hours,
            'summary': {
                'total_queries': query_metrics.total_queries,
                'average_response_time': query_metrics.average_response_time,
                'success_rate': query_metrics.success_rate,
                'system_cpu_usage': system_metrics.cpu_usage,
                'system_memory_usage': system_metrics.memory_usage,
                'cache_hit_rate': system_metrics.cache_hit_rate,
            },
            'query_metrics': asdict(query_metrics),
            'system_metrics': asdict(system_metrics),
            'performance_trends': trends,
            'slow_queries': slow_queries[:10],  # Top 10 slowest
            'recommendations': self._generate_performance_recommendations(query_metrics, system_metrics),
        }

    async def _generate_usage_report(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate usage analytics report."""
        query_patterns = self.query_analyzer.analyze_query_patterns(config.time_range_hours)
        recent_metrics = self.metrics_collector.get_recent_metrics(config.time_range_hours)

        # Analyze usage patterns from metrics
        usage_by_hour = {}
        usage_by_framework = {}
        user_activity = {}

        for metric in recent_metrics:
            if metric.metric_name == "user_activity":
                user_id = metric.tags.get("user_id", "unknown")
                user_activity[user_id] = user_activity.get(user_id, 0) + 1

            if metric.metric_name == "query_execution":
                framework = metric.tags.get("framework", "unknown")
                usage_by_framework[framework] = usage_by_framework.get(framework, 0) + 1

                hour = metric.timestamp.hour
                usage_by_hour[hour] = usage_by_hour.get(hour, 0) + 1

        return {
            'report_type': 'usage',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'time_range_hours': config.time_range_hours,
            'summary': {
                'total_requests': len(recent_metrics),
                'unique_users': len(user_activity),
                'most_active_hour': max(usage_by_hour.items(), key=lambda x: x[1])[0] if usage_by_hour else None,
                'most_used_framework': max(usage_by_framework.items(), key=lambda x: x[1])[0]
                if usage_by_framework
                else None,
            },
            'usage_patterns': {
                'hourly_distribution': usage_by_hour,
                'framework_usage': usage_by_framework,
                'user_activity': dict(sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:20]),
            },
            'query_patterns': query_patterns,
        }

    async def _generate_error_report(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate error analysis report."""
        error_analysis = self.performance_analyzer.get_error_analysis(config.time_range_hours)

        return {
            'report_type': 'error_analysis',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'time_range_hours': config.time_range_hours,
            'summary': {
                'total_errors': error_analysis.get('total_errors', 0),
                'error_rate_per_hour': error_analysis.get('error_rate_per_hour', 0),
                'most_common_category': max(error_analysis.get('by_category', {}).items(), key=lambda x: x[1])[0]
                if error_analysis.get('by_category')
                else None,
            },
            'error_analysis': error_analysis,
            'recommendations': self._generate_error_recommendations(error_analysis),
        }

    async def _generate_user_activity_report(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate user activity report."""
        recent_metrics = self.metrics_collector.get_recent_metrics(config.time_range_hours)

        user_metrics = {}
        for metric in recent_metrics:
            if metric.metric_name == "user_activity":
                user_id = metric.tags.get("user_id", "unknown")
                action = metric.tags.get("action", "unknown")

                if user_id not in user_metrics:
                    user_metrics[user_id] = {
                        'total_actions': 0,
                        'actions_by_type': {},
                        'last_activity': metric.timestamp,
                    }

                user_metrics[user_id]['total_actions'] += 1
                user_metrics[user_id]['actions_by_type'][action] = (
                    user_metrics[user_id]['actions_by_type'].get(action, 0) + 1
                )

                if metric.timestamp > user_metrics[user_id]['last_activity']:
                    user_metrics[user_id]['last_activity'] = metric.timestamp

        # Convert timestamps to ISO format
        for user_data in user_metrics.values():
            user_data['last_activity'] = user_data['last_activity'].isoformat()

        return {
            'report_type': 'user_activity',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'time_range_hours': config.time_range_hours,
            'summary': {
                'total_active_users': len(user_metrics),
                'total_user_actions': sum(data['total_actions'] for data in user_metrics.values()),
                'most_active_user': max(user_metrics.items(), key=lambda x: x[1]['total_actions'])[0]
                if user_metrics
                else None,
            },
            'user_metrics': user_metrics,
        }

    async def _generate_system_health_report(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate system health report."""
        system_metrics = self.performance_analyzer.get_system_performance(config.time_range_hours)
        error_analysis = self.performance_analyzer.get_error_analysis(config.time_range_hours)
        recent_metrics = self.metrics_collector.get_recent_metrics(config.time_range_hours)

        # Calculate health score (0-100)
        health_score = self._calculate_health_score(system_metrics, error_analysis)

        return {
            'report_type': 'system_health',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'time_range_hours': config.time_range_hours,
            'health_score': health_score,
            'status': 'healthy' if health_score > 80 else 'warning' if health_score > 60 else 'critical',
            'system_metrics': asdict(system_metrics),
            'error_summary': {
                'total_errors': error_analysis.get('total_errors', 0),
                'error_rate': error_analysis.get('error_rate_per_hour', 0),
            },
            'alerts': self._generate_health_alerts(system_metrics, error_analysis),
            'recommendations': self._generate_health_recommendations(system_metrics, error_analysis),
        }

    async def _format_report(self, data: Dict[str, Any], config: ReportConfig) -> Dict[str, Any]:
        """Format report data according to requested format."""
        if config.format == ReportFormat.JSON:
            return {
                'format': 'json',
                'data': data,
                'metadata': {'generated_at': datetime.now(timezone.utc).isoformat(), 'config': asdict(config)},
            }

        elif config.format == ReportFormat.CSV:
            return {
                'format': 'csv',
                'data': self._convert_to_csv(data),
                'metadata': {'generated_at': datetime.now(timezone.utc).isoformat(), 'config': asdict(config)},
            }

        elif config.format == ReportFormat.HTML:
            return {
                'format': 'html',
                'data': self._convert_to_html(data, config),
                'metadata': {'generated_at': datetime.now(timezone.utc).isoformat(), 'config': asdict(config)},
            }

        else:
            # Default to JSON for unsupported formats
            return await self._format_report(
                data, ReportConfig(config.report_type, ReportFormat.JSON, config.time_range_hours)
            )

    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert report data to CSV format."""
        import csv

        output = io.StringIO()

        # Write summary data
        writer = csv.writer(output)
        writer.writerow(['Metric', 'Value'])

        if 'summary' in data:
            for key, value in data['summary'].items():
                writer.writerow([key, value])

        return output.getvalue()

    def _convert_to_html(self, data: Dict[str, Any], config: ReportConfig) -> str:
        """Convert report data to HTML format."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{config.report_type.title()} Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; margin-bottom: 20px; }}
                .metric {{ margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{config.report_type.title()} Report</h1>
                <p>Generated: {data.get('generated_at', 'Unknown')}</p>
                <p>Time Range: {config.time_range_hours} hours</p>
            </div>
        """

        if 'summary' in data:
            html_content += '<div class="summary"><h2>Summary</h2>'
            for key, value in data['summary'].items():
                html_content += f'<div class="metric"><strong>{key}:</strong> {value}</div>'
            html_content += '</div>'

        html_content += '</body></html>'
        return html_content

    def _generate_performance_recommendations(self, query_metrics, system_metrics) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        if query_metrics.average_response_time > 5000:
            recommendations.append("Consider optimizing queries - average response time is high")

        if query_metrics.success_rate < 95:
            recommendations.append("Investigate query failures - success rate is below optimal")

        if system_metrics.cpu_usage > 80:
            recommendations.append("CPU usage is high - consider scaling or optimization")

        if system_metrics.memory_usage > 80:
            recommendations.append("Memory usage is high - check for memory leaks")

        if system_metrics.cache_hit_rate < 80:
            recommendations.append("Cache hit rate is low - review caching strategy")

        return recommendations

    def _generate_error_recommendations(self, error_analysis: Dict[str, Any]) -> List[str]:
        """Generate error-based recommendations."""
        recommendations = []

        if error_analysis.get('error_rate_per_hour', 0) > 10:
            recommendations.append("High error rate detected - investigate root causes")

        by_severity = error_analysis.get('by_severity', {})
        if by_severity.get('critical', 0) > 0:
            recommendations.append("Critical errors detected - immediate attention required")

        if by_severity.get('high', 0) > 5:
            recommendations.append("Multiple high-severity errors - prioritize fixes")

        return recommendations

    def _calculate_health_score(self, system_metrics, error_analysis) -> float:
        """Calculate overall system health score (0-100)."""
        score = 100.0

        # Deduct for high CPU usage
        if system_metrics.cpu_usage > 90:
            score -= 20
        elif system_metrics.cpu_usage > 80:
            score -= 10

        # Deduct for high memory usage
        if system_metrics.memory_usage > 90:
            score -= 20
        elif system_metrics.memory_usage > 80:
            score -= 10

        # Deduct for low cache hit rate
        if system_metrics.cache_hit_rate < 70:
            score -= 15
        elif system_metrics.cache_hit_rate < 80:
            score -= 5

        # Deduct for errors
        error_rate = error_analysis.get('error_rate_per_hour', 0)
        if error_rate > 20:
            score -= 30
        elif error_rate > 10:
            score -= 15
        elif error_rate > 5:
            score -= 5

        return max(0, min(100, score))

    def _generate_health_alerts(self, system_metrics, error_analysis) -> List[Dict[str, Any]]:
        """Generate health alerts based on metrics."""
        alerts = []

        if system_metrics.cpu_usage > 90:
            alerts.append(
                {
                    'level': 'critical',
                    'message': f'CPU usage at {system_metrics.cpu_usage}% - immediate action required',
                }
            )

        if system_metrics.memory_usage > 90:
            alerts.append(
                {
                    'level': 'critical',
                    'message': f'Memory usage at {system_metrics.memory_usage}% - system may become unstable',
                }
            )

        error_rate = error_analysis.get('error_rate_per_hour', 0)
        if error_rate > 20:
            alerts.append({'level': 'warning', 'message': f'High error rate: {error_rate} errors per hour'})

        return alerts

    def _generate_health_recommendations(self, system_metrics, error_analysis) -> List[str]:
        """Generate health-based recommendations."""
        recommendations = []

        if system_metrics.cpu_usage > 80:
            recommendations.append("Monitor CPU usage and consider scaling")

        if system_metrics.memory_usage > 80:
            recommendations.append("Monitor memory usage and check for leaks")

        if error_analysis.get('total_errors', 0) > 0:
            recommendations.append("Review error logs and implement fixes")

        if system_metrics.cache_hit_rate < 80:
            recommendations.append("Optimize caching strategy to improve performance")

        return recommendations


class ExportManager:
    """Manages report exports and scheduling."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.report_generator = ReportGenerator()

    async def export_report(self, config: ReportConfig, output_path: str) -> str:
        """Export report to file."""
        try:
            report = await self.report_generator.generate_report(config)

            with open(output_path, 'w') as f:
                if config.format == ReportFormat.JSON:
                    json.dump(report, f, indent=2, default=str)
                else:
                    f.write(str(report.get('data', '')))

            self.logger.info(f"Report exported to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
            raise

    async def schedule_report(self, config: ReportConfig, schedule: str, output_dir: str) -> str:
        """Schedule recurring report generation."""
        # This would integrate with a job scheduler like Celery or APScheduler
        # For now, return a placeholder
        self.logger.info(f"Report scheduled: {config.report_type} - {schedule}")
        return f"scheduled_{config.report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
