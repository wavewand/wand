"""
Analytics and Reporting System

Provides comprehensive analytics, metrics collection, reporting, and business
intelligence capabilities for the MCP platform.
"""

from .aggregators import DataAggregator, MetricsAggregator, TimeSeriesAggregator
from .collectors import MetricsCollector, PerformanceAnalyzer, QueryAnalyzer
from .insights import AnomalyDetector, InsightEngine, TrendAnalyzer
from .reporters import DashboardData, ExportManager, ReportGenerator

__all__ = [
    # Collectors
    'MetricsCollector',
    'QueryAnalyzer',
    'PerformanceAnalyzer',
    # Reporters
    'ReportGenerator',
    'DashboardData',
    'ExportManager',
    # Aggregators
    'DataAggregator',
    'TimeSeriesAggregator',
    'MetricsAggregator',
    # Insights
    'InsightEngine',
    'TrendAnalyzer',
    'AnomalyDetector',
]
