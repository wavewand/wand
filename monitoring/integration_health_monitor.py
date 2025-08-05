"""
Enhanced Integration Health Monitoring System for Wand
Provides real-time health scoring, configuration validation, and performance metrics
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels for integrations"""

    HEALTHY = "healthy"  # 90-100% health score
    DEGRADED = "degraded"  # 70-89% health score
    PARTIAL = "partial"  # 40-69% health score
    UNHEALTHY = "unhealthy"  # 10-39% health score
    CRITICAL = "critical"  # 0-9% health score
    UNKNOWN = "unknown"  # No data available


@dataclass
class HealthMetrics:
    """Comprehensive health metrics for an integration"""

    # Core Health Indicators
    status: HealthStatus
    score: float  # 0-100
    last_check: datetime
    check_interval: int = 60  # seconds

    # Configuration Health
    config_valid: bool = False
    config_complete: float = 0.0  # percentage of required config present
    credentials_valid: bool = False

    # Connectivity Health
    connection_status: str = "unknown"  # connected, disconnected, timeout, error
    last_successful_connection: Optional[datetime] = None
    connection_latency: float = 0.0  # milliseconds

    # Performance Health
    success_rate: float = 0.0  # percentage of successful operations
    average_response_time: float = 0.0  # milliseconds
    error_rate: float = 0.0  # percentage of failed operations

    # Usage Health
    requests_per_minute: float = 0.0
    cache_hit_rate: float = 0.0
    rate_limit_usage: float = 0.0  # percentage of rate limit used

    # Service-Specific Health
    quota_usage: float = 0.0  # percentage of API quota used
    service_status: str = "unknown"  # service-reported status

    # Historical Data (last 24h)
    uptime_percentage: float = 100.0
    incident_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        result = asdict(self)
        # Convert datetime objects to ISO strings
        if self.last_check:
            result['last_check'] = self.last_check.isoformat()
        if self.last_successful_connection:
            result['last_successful_connection'] = self.last_successful_connection.isoformat()
        # Convert enum to string
        result['status'] = self.status.value
        return result


class IntegrationHealthMonitor:
    """Advanced health monitoring system for Wand integrations"""

    def __init__(self):
        self.health_cache: Dict[str, HealthMetrics] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.integration_configs: Dict[str, Dict[str, Any]] = {}
        self.health_history: Dict[str, List[HealthMetrics]] = {}
        self.running = False

        # Configuration for health scoring weights
        self.health_weights = {
            'config_health': 0.25,  # Configuration completeness and validity
            'connectivity': 0.25,  # Connection status and latency
            'performance': 0.30,  # Success rate and response time
            'usage_efficiency': 0.20,  # Cache hits, rate limiting, quotas
        }

    async def start_monitoring(self):
        """Start the health monitoring system"""
        self.running = True
        logger.info("🏥 Integration Health Monitor starting...")

        # Start background health collection
        asyncio.create_task(self._health_collection_loop())

    async def stop_monitoring(self):
        """Stop the health monitoring system"""
        self.running = False

        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.values():
            if not task.done():
                task.cancel()

        await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
        logger.info("🏥 Integration Health Monitor stopped")

    async def register_integration(self, integration_name: str, integration_instance: Any, config: Dict[str, Any]):
        """Register an integration for health monitoring"""
        self.integration_configs[integration_name] = config

        # Initialize health metrics
        initial_health = HealthMetrics(
            status=HealthStatus.UNKNOWN,
            score=0.0,
            last_check=datetime.now(timezone.utc),
            config_valid=self._validate_integration_config(integration_name, config),
            config_complete=self._calculate_config_completeness(integration_name, config),
        )

        self.health_cache[integration_name] = initial_health
        self.health_history[integration_name] = []

        # Start individual monitoring task
        task = asyncio.create_task(self._monitor_integration(integration_name, integration_instance))
        self.monitoring_tasks[integration_name] = task

        logger.info(f"🏥 Registered {integration_name} for health monitoring")

    async def get_integration_health(self, integration_name: str) -> Optional[HealthMetrics]:
        """Get current health metrics for an integration"""
        return self.health_cache.get(integration_name)

    async def get_all_health_metrics(self) -> Dict[str, HealthMetrics]:
        """Get health metrics for all registered integrations"""
        return self.health_cache.copy()

    async def get_health_summary(self) -> Dict[str, Any]:
        """Get system-wide health summary"""
        all_health = await self.get_all_health_metrics()

        if not all_health:
            return {
                "total_integrations": 0,
                "healthy_count": 0,
                "degraded_count": 0,
                "unhealthy_count": 0,
                "average_score": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        status_counts = {}
        total_score = 0.0

        for health in all_health.values():
            status = health.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            total_score += health.score

        return {
            "total_integrations": len(all_health),
            "healthy_count": status_counts.get("healthy", 0),
            "degraded_count": status_counts.get("degraded", 0) + status_counts.get("partial", 0),
            "unhealthy_count": status_counts.get("unhealthy", 0) + status_counts.get("critical", 0),
            "unknown_count": status_counts.get("unknown", 0),
            "average_score": total_score / len(all_health),
            "status_breakdown": status_counts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _health_collection_loop(self):
        """Main health collection loop"""
        while self.running:
            try:
                # Update system-wide health statistics
                await self._update_system_health()

                # Clean old history data (keep last 24h)
                await self._cleanup_old_health_data()

                # Wait before next collection cycle
                await asyncio.sleep(30)  # System health update every 30 seconds

            except Exception as e:
                logger.error(f"❌ Health collection loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _monitor_integration(self, integration_name: str, integration_instance: Any):
        """Monitor a specific integration's health"""
        while self.running:
            try:
                start_time = time.time()

                # Get integration metrics
                metrics = integration_instance.get_metrics()

                # Perform health check
                health_check_result = await integration_instance.health_check()

                # Test connection with timeout
                connection_start = time.time()
                connection_result = await asyncio.wait_for(
                    integration_instance.test_connection(), timeout=10.0  # 10 second timeout
                )
                connection_time = (time.time() - connection_start) * 1000  # ms

                # Calculate health score
                health_score = await self._calculate_health_score(
                    integration_name, metrics, health_check_result, connection_result, connection_time
                )

                # Update health metrics
                health_metrics = HealthMetrics(
                    status=self._score_to_status(health_score),
                    score=health_score,
                    last_check=datetime.now(timezone.utc),
                    config_valid=self._validate_integration_config(
                        integration_name, self.integration_configs[integration_name]
                    ),
                    config_complete=self._calculate_config_completeness(
                        integration_name, self.integration_configs[integration_name]
                    ),
                    credentials_valid=connection_result.get('success', False),
                    connection_status="connected" if connection_result.get('success') else "disconnected",
                    last_successful_connection=datetime.now(timezone.utc)
                    if connection_result.get('success')
                    else self.health_cache[integration_name].last_successful_connection,
                    connection_latency=connection_time,
                    success_rate=metrics.get('success_rate', 0.0) * 100,
                    average_response_time=metrics.get('average_response_time', 0.0) * 1000,  # Convert to ms
                    error_rate=(1.0 - metrics.get('success_rate', 0.0)) * 100,
                    requests_per_minute=self._calculate_rpm(metrics),
                    cache_hit_rate=metrics.get('cache_hit_rate', 0.0) * 100,
                    rate_limit_usage=self._calculate_rate_limit_usage(integration_name, metrics),
                    uptime_percentage=self._calculate_uptime(integration_name),
                )

                # Store current health
                self.health_cache[integration_name] = health_metrics

                # Store in history
                self.health_history[integration_name].append(health_metrics)

                # Log significant health changes
                await self._log_health_changes(integration_name, health_metrics)

                # Wait for next check (vary based on health status)
                check_interval = self._get_check_interval(health_metrics.status)
                await asyncio.sleep(check_interval)

            except asyncio.TimeoutError:
                # Connection timeout
                await self._handle_connection_timeout(integration_name)
                await asyncio.sleep(120)  # Wait 2 minutes after timeout

            except Exception as e:
                logger.error(f"❌ Health monitoring error for {integration_name}: {e}")
                await self._handle_monitoring_error(integration_name, str(e))
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _calculate_health_score(
        self, integration_name: str, metrics: Dict, health_result: Dict, connection_result: Dict, connection_time: float
    ) -> float:
        """Calculate comprehensive health score (0-100)"""

        # Configuration Health (0-25 points)
        config_score = 0.0
        if self._validate_integration_config(integration_name, self.integration_configs[integration_name]):
            config_score += 15.0
        config_score += (
            self._calculate_config_completeness(integration_name, self.integration_configs[integration_name]) * 0.10
        )

        # Connectivity Health (0-25 points)
        connectivity_score = 0.0
        if connection_result.get('success', False):
            connectivity_score += 20.0
            # Bonus for fast response (under 1 second)
            if connection_time < 1000:
                connectivity_score += 5.0
            # Penalty for slow response (over 5 seconds)
            elif connection_time > 5000:
                connectivity_score -= 5.0

        # Performance Health (0-30 points)
        performance_score = 0.0
        success_rate = metrics.get('success_rate', 0.0)
        performance_score += success_rate * 20  # Up to 20 points for success rate

        avg_response = metrics.get('average_response_time', 0.0)
        if avg_response > 0:
            # Bonus for fast responses (under 500ms)
            if avg_response < 0.5:
                performance_score += 10.0
            # Good responses (under 2s)
            elif avg_response < 2.0:
                performance_score += 5.0
            # Penalty for slow responses (over 5s)
            elif avg_response > 5.0:
                performance_score -= 5.0

        # Usage Efficiency Health (0-20 points)
        efficiency_score = 0.0
        cache_hit_rate = metrics.get('cache_hit_rate', 0.0)
        efficiency_score += cache_hit_rate * 10  # Up to 10 points for cache efficiency

        rate_limit_usage = self._calculate_rate_limit_usage(integration_name, metrics)
        if rate_limit_usage < 0.8:  # Less than 80% of rate limit
            efficiency_score += 10.0
        elif rate_limit_usage < 0.95:  # Less than 95% of rate limit
            efficiency_score += 5.0

        total_score = config_score + connectivity_score + performance_score + efficiency_score
        return min(100.0, max(0.0, total_score))

    def _score_to_status(self, score: float) -> HealthStatus:
        """Convert numeric health score to status enum"""
        if score >= 90:
            return HealthStatus.HEALTHY
        elif score >= 70:
            return HealthStatus.DEGRADED
        elif score >= 40:
            return HealthStatus.PARTIAL
        elif score >= 10:
            return HealthStatus.UNHEALTHY
        elif score > 0:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.UNKNOWN

    def _validate_integration_config(self, integration_name: str, config: Dict[str, Any]) -> bool:
        """Validate if integration configuration is complete and valid"""
        required_fields = self._get_required_config_fields(integration_name)

        for field in required_fields:
            if field not in config or not config[field]:
                return False

        return True

    def _calculate_config_completeness(self, integration_name: str, config: Dict[str, Any]) -> float:
        """Calculate percentage of configuration completeness"""
        required_fields = self._get_required_config_fields(integration_name)
        optional_fields = self._get_optional_config_fields(integration_name)

        all_fields = required_fields + optional_fields
        if not all_fields:
            return 100.0

        configured_count = 0
        for field in all_fields:
            if field in config and config[field]:
                configured_count += 1

        return (configured_count / len(all_fields)) * 100.0

    def _get_required_config_fields(self, integration_name: str) -> List[str]:
        """Get required configuration fields for an integration"""
        field_map = {
            'slack': ['token'],
            'github': ['github_token'],
            'jenkins': ['url', 'username', 'token'],
            'youtrack': ['url', 'token'],
            'postgresql': ['host', 'user', 'password'],
            'aws': ['access_key_id', 'secret_access_key'],
            'openai': ['api_key'],
            'anthropic': ['api_key'],
        }
        return field_map.get(integration_name, [])

    def _get_optional_config_fields(self, integration_name: str) -> List[str]:
        """Get optional configuration fields for an integration"""
        field_map = {
            'slack': ['default_channel', 'app_token'],
            'github': ['gitlab_token', 'default_branch'],
            'jenkins': ['default_pipeline', 'timeout'],
            'youtrack': ['default_project'],
            'postgresql': ['port', 'default_database', 'ssl_mode'],
            'aws': ['region', 'session_token'],
        }
        return field_map.get(integration_name, [])

    def _calculate_rpm(self, metrics: Dict) -> float:
        """Calculate requests per minute from metrics"""
        total_requests = metrics.get('requests_total', 0)
        # Estimate RPM based on usage patterns (simplified)
        return total_requests * 0.1  # Rough estimate

    def _calculate_rate_limit_usage(self, integration_name: str, metrics: Dict) -> float:
        """Calculate rate limit usage percentage"""
        rate_limit_hits = metrics.get('rate_limit_hits', 0)
        total_requests = metrics.get('requests_total', 0)

        if total_requests == 0:
            return 0.0

        return rate_limit_hits / total_requests

    def _calculate_uptime(self, integration_name: str) -> float:
        """Calculate uptime percentage for last 24 hours"""
        history = self.health_history.get(integration_name, [])
        if not history:
            return 100.0

        # Simple uptime calculation based on successful health checks
        recent_history = [h for h in history if h.last_check > datetime.now(timezone.utc) - timedelta(hours=24)]
        if not recent_history:
            return 100.0

        healthy_checks = len([h for h in recent_history if h.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]])
        return (healthy_checks / len(recent_history)) * 100.0

    def _get_check_interval(self, status: HealthStatus) -> int:
        """Get health check interval based on current status"""
        intervals = {
            HealthStatus.HEALTHY: 300,  # 5 minutes
            HealthStatus.DEGRADED: 120,  # 2 minutes
            HealthStatus.PARTIAL: 60,  # 1 minute
            HealthStatus.UNHEALTHY: 30,  # 30 seconds
            HealthStatus.CRITICAL: 15,  # 15 seconds
            HealthStatus.UNKNOWN: 60,  # 1 minute
        }
        return intervals.get(status, 60)

    async def _handle_connection_timeout(self, integration_name: str):
        """Handle connection timeout for an integration"""
        timeout_health = HealthMetrics(
            status=HealthStatus.CRITICAL,
            score=5.0,  # Very low score for timeout
            last_check=datetime.now(timezone.utc),
            connection_status="timeout",
            connection_latency=10000.0,  # 10 second timeout
            config_valid=self.health_cache[integration_name].config_valid,
            config_complete=self.health_cache[integration_name].config_complete,
        )

        self.health_cache[integration_name] = timeout_health
        logger.warning(f"⏰ Connection timeout for {integration_name}")

    async def _handle_monitoring_error(self, integration_name: str, error: str):
        """Handle monitoring errors for an integration"""
        error_health = HealthMetrics(
            status=HealthStatus.UNKNOWN,
            score=0.0,
            last_check=datetime.now(timezone.utc),
            connection_status="error",
            config_valid=False,
        )

        self.health_cache[integration_name] = error_health
        logger.error(f"❌ Monitoring error for {integration_name}: {error}")

    async def _log_health_changes(self, integration_name: str, new_health: HealthMetrics):
        """Log significant health status changes"""
        previous_health = self.health_cache.get(integration_name)

        if previous_health and previous_health.status != new_health.status:
            logger.info(
                f"🏥 {integration_name} health changed: {previous_health.status.value} → {new_health.status.value} (score: {new_health.score:.1f})"
            )

    async def _update_system_health(self):
        """Update system-wide health statistics"""
        # This could update system-level metrics, trigger alerts, etc.
        pass

    async def _cleanup_old_health_data(self):
        """Clean up old health history data"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)

        for integration_name in self.health_history:
            self.health_history[integration_name] = [
                h for h in self.health_history[integration_name] if h.last_check > cutoff_time
            ]


# Global health monitor instance
health_monitor = IntegrationHealthMonitor()


async def get_health_monitor() -> IntegrationHealthMonitor:
    """Get the global health monitor instance"""
    return health_monitor
