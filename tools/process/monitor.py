"""
Process Monitor

Provides real-time process monitoring, alerting, and resource usage tracking.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .manager import ProcessManager
from .models import ProcessFilter, ProcessInfo, ProcessStatus

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ProcessAlert:
    """Process monitoring alert"""

    level: AlertLevel
    message: str
    process_info: ProcessInfo
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    alert_type: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'level': self.level.value,
            'message': self.message,
            'process': self.process_info.to_dict(),
            'threshold_value': self.threshold_value,
            'actual_value': self.actual_value,
            'alert_type': self.alert_type,
            'timestamp': self.timestamp,
        }


@dataclass
class MonitoringRule:
    """Process monitoring rule"""

    name: str
    description: str
    filter_criteria: ProcessFilter
    check_interval: float = 30.0  # seconds
    enabled: bool = True

    # Thresholds
    max_cpu_percent: Optional[float] = None
    max_memory_mb: Optional[float] = None
    max_memory_percent: Optional[float] = None
    min_processes: Optional[int] = None
    max_processes: Optional[int] = None

    # Process lifecycle alerts
    alert_on_new_process: bool = False
    alert_on_process_exit: bool = False
    alert_on_status_change: bool = False

    # Advanced conditions
    custom_condition: Optional[Callable[[List[ProcessInfo]], List[ProcessAlert]]] = None

    def check_processes(
        self, processes: List[ProcessInfo], previous_processes: List[ProcessInfo]
    ) -> List[ProcessAlert]:
        """Check processes against this rule and generate alerts"""
        alerts = []

        # Filter processes that match criteria
        matching_processes = [p for p in processes if self.filter_criteria.matches(p)]
        previous_matching = [p for p in previous_processes if self.filter_criteria.matches(p)]

        # Check process count thresholds
        if self.min_processes is not None and len(matching_processes) < self.min_processes:
            alerts.append(
                ProcessAlert(
                    level=AlertLevel.WARNING,
                    message=f"Too few processes matching '{self.name}': {len(matching_processes)} < {self.min_processes}",
                    process_info=ProcessInfo(pid=0, name="rule_check"),
                    threshold_value=self.min_processes,
                    actual_value=len(matching_processes),
                    alert_type="process_count_low",
                )
            )

        if self.max_processes is not None and len(matching_processes) > self.max_processes:
            alerts.append(
                ProcessAlert(
                    level=AlertLevel.WARNING,
                    message=f"Too many processes matching '{self.name}': {len(matching_processes)} > {self.max_processes}",
                    process_info=ProcessInfo(pid=0, name="rule_check"),
                    threshold_value=self.max_processes,
                    actual_value=len(matching_processes),
                    alert_type="process_count_high",
                )
            )

        # Check individual process thresholds
        for process in matching_processes:
            # CPU threshold
            if self.max_cpu_percent is not None and process.cpu_percent > self.max_cpu_percent:
                alerts.append(
                    ProcessAlert(
                        level=AlertLevel.WARNING,
                        message=f"High CPU usage: {process.name} (PID {process.pid}) using {process.cpu_percent:.1f}%",
                        process_info=process,
                        threshold_value=self.max_cpu_percent,
                        actual_value=process.cpu_percent,
                        alert_type="high_cpu",
                    )
                )

            # Memory MB threshold
            if self.max_memory_mb is not None and process.memory_mb > self.max_memory_mb:
                alerts.append(
                    ProcessAlert(
                        level=AlertLevel.WARNING,
                        message=f"High memory usage: {process.name} (PID {process.pid}) using {process.memory_mb:.1f} MB",
                        process_info=process,
                        threshold_value=self.max_memory_mb,
                        actual_value=process.memory_mb,
                        alert_type="high_memory_mb",
                    )
                )

            # Memory percent threshold
            if self.max_memory_percent is not None and process.memory_percent > self.max_memory_percent:
                alerts.append(
                    ProcessAlert(
                        level=AlertLevel.WARNING,
                        message=f"High memory percentage: {process.name} (PID {process.pid}) using {process.memory_percent:.1f}%",
                        process_info=process,
                        threshold_value=self.max_memory_percent,
                        actual_value=process.memory_percent,
                        alert_type="high_memory_percent",
                    )
                )

        # Check for new/exited processes
        if self.alert_on_new_process or self.alert_on_process_exit:
            current_pids = {p.pid for p in matching_processes}
            previous_pids = {p.pid for p in previous_matching}

            # New processes
            if self.alert_on_new_process:
                new_pids = current_pids - previous_pids
                for pid in new_pids:
                    process = next((p for p in matching_processes if p.pid == pid), None)
                    if process:
                        alerts.append(
                            ProcessAlert(
                                level=AlertLevel.INFO,
                                message=f"New process started: {process.name} (PID {process.pid})",
                                process_info=process,
                                alert_type="process_started",
                            )
                        )

            # Exited processes
            if self.alert_on_process_exit:
                exited_pids = previous_pids - current_pids
                for pid in exited_pids:
                    process = next((p for p in previous_matching if p.pid == pid), None)
                    if process:
                        alerts.append(
                            ProcessAlert(
                                level=AlertLevel.INFO,
                                message=f"Process exited: {process.name} (PID {pid})",
                                process_info=process,
                                alert_type="process_exited",
                            )
                        )

        # Custom condition check
        if self.custom_condition:
            try:
                custom_alerts = self.custom_condition(matching_processes)
                if custom_alerts:
                    alerts.extend(custom_alerts)
            except Exception as e:
                logger.error(f"Error in custom condition for rule '{self.name}': {e}")

        return alerts


class ProcessMonitor:
    """Real-time process monitoring system"""

    def __init__(self, process_manager: ProcessManager):
        self.process_manager = process_manager
        self.rules: Dict[str, MonitoringRule] = {}
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.alert_handlers: List[Callable[[ProcessAlert], None]] = []

        # State tracking
        self.previous_processes: Dict[str, List[ProcessInfo]] = {}
        self.alert_history: List[ProcessAlert] = []
        self.max_alert_history = 1000

        # Statistics
        self.stats = {'checks_performed': 0, 'alerts_generated': 0, 'last_check_time': 0.0, 'monitoring_duration': 0.0}

    def add_rule(self, rule: MonitoringRule):
        """Add a monitoring rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added monitoring rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a monitoring rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed monitoring rule: {rule_name}")
            return True
        return False

    def add_alert_handler(self, handler: Callable[[ProcessAlert], None]):
        """Add an alert handler function"""
        self.alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable[[ProcessAlert], None]):
        """Remove an alert handler function"""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)

    async def start_monitoring(self):
        """Start the monitoring loop"""
        if self.is_running:
            logger.warning("Monitor is already running")
            return

        self.is_running = True
        self.stats['monitoring_duration'] = time.time()
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Process monitoring started")

    async def stop_monitoring(self):
        """Stop the monitoring loop"""
        if not self.is_running:
            logger.warning("Monitor is not running")
            return

        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        duration = time.time() - self.stats['monitoring_duration']
        self.stats['monitoring_duration'] = duration
        logger.info(f"Process monitoring stopped after {duration:.1f} seconds")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.is_running:
                start_time = time.time()

                # Get current processes from all backends
                current_processes = {}
                for backend_name in self.process_manager.list_backends():
                    try:
                        processes = await self.process_manager.list_processes(backend_name)
                        current_processes[backend_name] = processes
                    except Exception as e:
                        logger.error(f"Error getting processes from backend {backend_name}: {e}")
                        current_processes[backend_name] = []

                # Check all rules
                all_alerts = []
                for rule_name, rule in self.rules.items():
                    if not rule.enabled:
                        continue

                    try:
                        # Combine processes from all backends for rule checking
                        all_current = []
                        all_previous = []

                        for backend_name, processes in current_processes.items():
                            all_current.extend(processes)
                            all_previous.extend(self.previous_processes.get(backend_name, []))

                        alerts = rule.check_processes(all_current, all_previous)
                        all_alerts.extend(alerts)

                    except Exception as e:
                        logger.error(f"Error checking rule '{rule_name}': {e}")

                # Process alerts
                for alert in all_alerts:
                    await self._handle_alert(alert)

                # Update statistics
                self.stats['checks_performed'] += 1
                self.stats['alerts_generated'] += len(all_alerts)
                self.stats['last_check_time'] = time.time()

                # Store current processes as previous for next iteration
                self.previous_processes = current_processes

                # Calculate sleep time (minimum interval across all rules)
                min_interval = min((rule.check_interval for rule in self.rules.values() if rule.enabled), default=30.0)
                elapsed = time.time() - start_time
                sleep_time = max(0, min_interval - elapsed)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            self.is_running = False

    async def _handle_alert(self, alert: ProcessAlert):
        """Handle a generated alert"""
        # Add to history
        self.alert_history.append(alert)

        # Trim history if needed
        if len(self.alert_history) > self.max_alert_history:
            self.alert_history = self.alert_history[-self.max_alert_history :]

        # Log alert
        log_func = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
        }.get(alert.level, logger.info)

        log_func(f"Process Alert [{alert.level.value.upper()}]: {alert.message}")

        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

    def get_recent_alerts(self, count: int = 50, level: Optional[AlertLevel] = None) -> List[ProcessAlert]:
        """Get recent alerts, optionally filtered by level"""
        alerts = self.alert_history

        if level:
            alerts = [a for a in alerts if a.level == level]

        return alerts[-count:] if count > 0 else alerts

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        stats = self.stats.copy()
        stats['is_running'] = self.is_running
        stats['active_rules'] = len([r for r in self.rules.values() if r.enabled])
        stats['total_rules'] = len(self.rules)
        stats['alert_handlers'] = len(self.alert_handlers)
        stats['alert_history_size'] = len(self.alert_history)

        if self.is_running and self.stats['monitoring_duration'] > 0:
            stats['uptime'] = time.time() - self.stats['monitoring_duration']
        else:
            stats['uptime'] = self.stats.get('monitoring_duration', 0)

        return stats

    def create_default_rules(self):
        """Create a set of default monitoring rules"""
        # High CPU usage rule
        self.add_rule(
            MonitoringRule(
                name="high_cpu_usage",
                description="Alert on processes using excessive CPU",
                filter_criteria=ProcessFilter(),
                max_cpu_percent=80.0,
                check_interval=30.0,
            )
        )

        # High memory usage rule
        self.add_rule(
            MonitoringRule(
                name="high_memory_usage",
                description="Alert on processes using excessive memory",
                filter_criteria=ProcessFilter(),
                max_memory_mb=1000.0,
                check_interval=30.0,
            )
        )

        # Zombie process detection
        self.add_rule(
            MonitoringRule(
                name="zombie_processes",
                description="Alert on zombie processes",
                filter_criteria=ProcessFilter(status=ProcessStatus.ZOMBIE),
                min_processes=0,  # Any zombie processes trigger alert
                check_interval=60.0,
                alert_on_new_process=True,
            )
        )

        # Process lifecycle monitoring for important services
        self.add_rule(
            MonitoringRule(
                name="important_services",
                description="Monitor important system services",
                filter_criteria=ProcessFilter(name_pattern="(systemd|init|sshd|networkd)"),
                alert_on_process_exit=True,
                check_interval=60.0,
            )
        )

        logger.info("Created default monitoring rules")


# Predefined alert handlers
def console_alert_handler(alert: ProcessAlert):
    """Simple console alert handler"""
    print(f"[{alert.timestamp:.0f}] {alert.level.value.upper()}: {alert.message}")


def log_alert_handler(alert: ProcessAlert):
    """Log-based alert handler"""
    logger.info(f"ALERT: {alert.to_dict()}")


async def webhook_alert_handler(alert: ProcessAlert, webhook_url: str):
    """Send alerts to webhook (example implementation)"""
    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            payload = {'alert': alert.to_dict(), 'timestamp': alert.timestamp}

            async with session.post(webhook_url, json=payload, timeout=10) as response:
                if response.status == 200:
                    logger.debug(f"Alert sent to webhook: {webhook_url}")
                else:
                    logger.warning(f"Webhook returned status {response.status}")

    except Exception as e:
        logger.error(f"Error sending alert to webhook: {e}")


def create_process_monitor(process_manager: ProcessManager, with_defaults: bool = True) -> ProcessMonitor:
    """Factory function to create a configured process monitor"""
    monitor = ProcessMonitor(process_manager)

    if with_defaults:
        monitor.create_default_rules()
        monitor.add_alert_handler(log_alert_handler)

    return monitor
