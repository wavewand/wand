"""
Enhanced Log Management and Rotation System

Provides comprehensive log management including:
- Automated log rotation with compression
- Log cleanup and retention policies
- Log aggregation and analysis
- Health monitoring and alerts
- Secure log archiving
"""

import asyncio
import gzip
import hashlib
import json
import logging
import os
import shutil
import tarfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import schedule


class RetentionPolicy(str, Enum):
    """Log retention policies."""

    HOURLY = "hourly"  # Keep hourly logs
    DAILY = "daily"  # Keep daily logs
    WEEKLY = "weekly"  # Keep weekly logs
    MONTHLY = "monthly"  # Keep monthly logs
    YEARLY = "yearly"  # Keep yearly logs


class CompressionFormat(str, Enum):
    """Supported compression formats."""

    GZIP = "gzip"
    BZIP2 = "bzip2"
    XZ = "xz"
    NONE = "none"


@dataclass
class LogRetentionRule:
    """Configuration for log retention rules."""

    category: str  # Log category pattern
    policy: RetentionPolicy  # Retention policy
    max_age_days: int  # Maximum age in days
    max_size_mb: int  # Maximum size in MB
    compression: CompressionFormat  # Compression format
    archive_location: Optional[Path] = None  # Archive directory


@dataclass
class LogManagementConfig:
    """Configuration for log management system."""

    # Basic settings
    enabled: bool = True
    check_interval_minutes: int = 60  # How often to check logs

    # Retention rules
    retention_rules: List[LogRetentionRule] = None

    # Compression settings
    compress_after_hours: int = 24  # Compress logs after 24 hours
    delete_after_compression: bool = True  # Delete original after compression

    # Archive settings
    enable_archiving: bool = True
    archive_base_path: Optional[Path] = None

    # Health monitoring
    enable_health_monitoring: bool = True
    max_log_file_size_mb: int = 100  # Alert when file exceeds this
    max_total_log_size_gb: int = 10  # Alert when total logs exceed this

    # Security
    enable_integrity_checks: bool = True
    verify_checksums: bool = True


class LogHealthMonitor:
    """Monitors log system health and performance."""

    def __init__(self, config: LogManagementConfig, log_directory: Path):
        self.config = config
        self.log_directory = log_directory
        self.health_logger = logging.getLogger("wand.log_health")

    def check_log_health(self) -> Dict[str, any]:
        """Check overall log system health."""
        health_status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "healthy",
            "issues": [],
            "metrics": {},
        }

        try:
            # Check disk space
            disk_usage = self._check_disk_usage()
            health_status["metrics"]["disk_usage"] = disk_usage

            # Check individual log files
            file_stats = self._check_log_files()
            health_status["metrics"]["file_stats"] = file_stats

            # Check total log size
            total_size = self._calculate_total_size()
            health_status["metrics"]["total_size_mb"] = total_size / (1024 * 1024)

            # Evaluate health status
            health_status = self._evaluate_health(health_status)

        except Exception as e:
            health_status["status"] = "error"
            health_status["issues"].append(f"Health check failed: {str(e)}")
            self.health_logger.error(f"Log health check failed: {e}")

        return health_status

    def _check_disk_usage(self) -> Dict[str, float]:
        """Check disk space usage for log directory."""
        stat = shutil.disk_usage(self.log_directory)

        return {
            "total_gb": stat.total / (1024**3),
            "used_gb": stat.used / (1024**3),
            "free_gb": stat.free / (1024**3),
            "used_percent": (stat.used / stat.total) * 100,
        }

    def _check_log_files(self) -> List[Dict[str, any]]:
        """Check individual log file statistics."""
        file_stats = []

        for log_file in self.log_directory.glob("*.log"):
            try:
                stat = log_file.stat()
                file_info = {
                    "file": log_file.name,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "age_hours": (time.time() - stat.st_mtime) / 3600,
                }
                file_stats.append(file_info)
            except Exception as e:
                self.health_logger.warning(f"Could not stat {log_file}: {e}")

        return file_stats

    def _calculate_total_size(self) -> int:
        """Calculate total size of all log files."""
        total_size = 0

        for log_file in self.log_directory.rglob("*"):
            if log_file.is_file():
                try:
                    total_size += log_file.stat().st_size
                except Exception:
                    continue

        return total_size

    def _evaluate_health(self, health_status: Dict) -> Dict:
        """Evaluate health status and add issues."""
        issues = health_status["issues"]

        # Check disk usage
        disk_usage = health_status["metrics"]["disk_usage"]
        if disk_usage["used_percent"] > 90:
            issues.append(f"High disk usage: {disk_usage['used_percent']:.1f}%")
            health_status["status"] = "warning"

        # Check total log size
        total_size_gb = health_status["metrics"]["total_size_mb"] / 1024
        if total_size_gb > self.config.max_total_log_size_gb:
            issues.append(f"Total log size exceeds limit: {total_size_gb:.1f}GB")
            health_status["status"] = "warning"

        # Check individual file sizes
        for file_stat in health_status["metrics"]["file_stats"]:
            if file_stat["size_mb"] > self.config.max_log_file_size_mb:
                issues.append(f"Large log file: {file_stat['file']} ({file_stat['size_mb']:.1f}MB)")
                if health_status["status"] == "healthy":
                    health_status["status"] = "warning"

        return health_status


class LogCompressor:
    """Handles log file compression and archiving."""

    def __init__(self, config: LogManagementConfig):
        self.config = config
        self.compression_logger = logging.getLogger("wand.log_compression")

    def compress_file(self, source_path: Path, compression: CompressionFormat = CompressionFormat.GZIP) -> Path:
        """Compress a single log file."""
        if compression == CompressionFormat.NONE:
            return source_path

        compressed_path = source_path.with_suffix(f"{source_path.suffix}.{compression.value}")

        try:
            if compression == CompressionFormat.GZIP:
                with open(source_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

            elif compression == CompressionFormat.BZIP2:
                import bz2

                with open(source_path, 'rb') as f_in:
                    with bz2.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

            elif compression == CompressionFormat.XZ:
                import lzma

                with open(source_path, 'rb') as f_in:
                    with lzma.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

            # Verify compressed file
            if compressed_path.exists() and compressed_path.stat().st_size > 0:
                self.compression_logger.info(
                    f"Compressed {source_path.name} -> {compressed_path.name} "
                    f"({source_path.stat().st_size} -> {compressed_path.stat().st_size} bytes)"
                )

                if self.config.delete_after_compression:
                    source_path.unlink()

                return compressed_path
            else:
                raise Exception("Compression failed - empty output file")

        except Exception as e:
            self.compression_logger.error(f"Failed to compress {source_path}: {e}")
            if compressed_path.exists():
                compressed_path.unlink()  # Clean up failed compression
            raise

    def create_archive(self, files: List[Path], archive_path: Path) -> Path:
        """Create a tar archive of multiple log files."""
        try:
            with tarfile.open(archive_path, 'w:gz') as tar:
                for file_path in files:
                    if file_path.exists():
                        tar.add(file_path, arcname=file_path.name)

            self.compression_logger.info(f"Created archive {archive_path} with {len(files)} files")
            return archive_path

        except Exception as e:
            self.compression_logger.error(f"Failed to create archive {archive_path}: {e}")
            raise


class LogRetentionManager:
    """Manages log retention policies and cleanup."""

    def __init__(self, config: LogManagementConfig, log_directory: Path):
        self.config = config
        self.log_directory = log_directory
        self.retention_logger = logging.getLogger("wand.log_retention")
        self.compressor = LogCompressor(config)

        # Default retention rules if none specified
        if not self.config.retention_rules:
            self.config.retention_rules = self._default_retention_rules()

    def _default_retention_rules(self) -> List[LogRetentionRule]:
        """Create default retention rules for different log categories."""
        return [
            LogRetentionRule(
                category="mcp_protocol*",
                policy=RetentionPolicy.DAILY,
                max_age_days=30,
                max_size_mb=500,
                compression=CompressionFormat.GZIP,
            ),
            LogRetentionRule(
                category="tool_execution*",
                policy=RetentionPolicy.DAILY,
                max_age_days=14,
                max_size_mb=200,
                compression=CompressionFormat.GZIP,
            ),
            LogRetentionRule(
                category="security*",
                policy=RetentionPolicy.MONTHLY,
                max_age_days=365,
                max_size_mb=1000,
                compression=CompressionFormat.GZIP,
            ),
            LogRetentionRule(
                category="audit*",
                policy=RetentionPolicy.MONTHLY,
                max_age_days=2555,  # 7 years for audit logs
                max_size_mb=2000,
                compression=CompressionFormat.GZIP,
            ),
            LogRetentionRule(
                category="*",  # Default rule
                policy=RetentionPolicy.WEEKLY,
                max_age_days=7,
                max_size_mb=100,
                compression=CompressionFormat.GZIP,
            ),
        ]

    def apply_retention_policies(self):
        """Apply retention policies to all log files."""
        self.retention_logger.info("Starting log retention policy application")

        processed_files = 0
        compressed_files = 0
        deleted_files = 0

        try:
            for log_file in self.log_directory.glob("*.log"):
                rule = self._find_matching_rule(log_file.name)
                if rule:
                    action = self._process_file_with_rule(log_file, rule)
                    processed_files += 1

                    if action == "compressed":
                        compressed_files += 1
                    elif action == "deleted":
                        deleted_files += 1

            # Process rotated files (*.log.1, *.log.2, etc.)
            for log_file in self.log_directory.glob("*.log.*"):
                if not any(log_file.name.endswith(f".{fmt.value}") for fmt in CompressionFormat):
                    rule = self._find_matching_rule(log_file.stem)
                    if rule:
                        action = self._process_file_with_rule(log_file, rule)
                        processed_files += 1

                        if action == "compressed":
                            compressed_files += 1
                        elif action == "deleted":
                            deleted_files += 1

            self.retention_logger.info(
                f"Retention policy completed: "
                f"processed={processed_files}, compressed={compressed_files}, deleted={deleted_files}"
            )

        except Exception as e:
            self.retention_logger.error(f"Retention policy application failed: {e}")

    def _find_matching_rule(self, filename: str) -> Optional[LogRetentionRule]:
        """Find the retention rule that matches a filename."""
        for rule in self.config.retention_rules:
            if self._matches_pattern(filename, rule.category):
                return rule
        return None

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches a pattern (supports * wildcards)."""
        import fnmatch

        return fnmatch.fnmatch(filename, pattern)

    def _process_file_with_rule(self, file_path: Path, rule: LogRetentionRule) -> str:
        """Process a file according to a retention rule."""
        try:
            file_stat = file_path.stat()
            file_age = (time.time() - file_stat.st_mtime) / 86400  # days
            file_size_mb = file_stat.st_size / (1024 * 1024)

            # Check if file should be deleted
            if file_age > rule.max_age_days:
                file_path.unlink()
                self.retention_logger.info(f"Deleted old file: {file_path.name} (age: {file_age:.1f} days)")
                return "deleted"

            # Check if file should be compressed
            if (
                file_age > (self.config.compress_after_hours / 24)
                and rule.compression != CompressionFormat.NONE
                and not self._is_already_compressed(file_path)
            ):
                self.compressor.compress_file(file_path, rule.compression)
                self.retention_logger.info(f"Compressed file: {file_path.name}")
                return "compressed"

            # Check if file exceeds size limit
            if file_size_mb > rule.max_size_mb:
                self.retention_logger.warning(
                    f"File {file_path.name} exceeds size limit: {file_size_mb:.1f}MB > {rule.max_size_mb}MB"
                )

            return "processed"

        except Exception as e:
            self.retention_logger.error(f"Failed to process {file_path}: {e}")
            return "error"

    def _is_already_compressed(self, file_path: Path) -> bool:
        """Check if file is already compressed."""
        compressed_extensions = {fmt.value for fmt in CompressionFormat if fmt != CompressionFormat.NONE}
        return any(str(file_path).endswith(f".{ext}") for ext in compressed_extensions)


class LogManagementSystem:
    """Main log management system coordinating all components."""

    def __init__(self, config: LogManagementConfig, log_directory: Path):
        self.config = config
        self.log_directory = log_directory
        self.health_monitor = LogHealthMonitor(config, log_directory)
        self.retention_manager = LogRetentionManager(config, log_directory)
        self.management_logger = logging.getLogger("wand.log_management")

        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the log management system."""
        if not self.config.enabled:
            self.management_logger.info("Log management system disabled")
            return

        self.management_logger.info("Starting log management system")

        # Schedule regular tasks
        schedule.every(self.config.check_interval_minutes).minutes.do(self._run_maintenance)

        # Schedule daily retention policy application
        schedule.every().day.at("02:00").do(self.retention_manager.apply_retention_policies)

        # Schedule health checks
        if self.config.enable_health_monitoring:
            schedule.every(30).minutes.do(self._run_health_check)

        # Start scheduler in background thread
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()

        self.management_logger.info("Log management system started")

    def stop(self):
        """Stop the log management system."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)

        self.management_logger.info("Log management system stopped")

    def _run_scheduler(self):
        """Run the scheduled tasks."""
        while self._running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.management_logger.error(f"Scheduler error: {e}")

    def _run_maintenance(self):
        """Run routine maintenance tasks."""
        try:
            self.management_logger.info("Running log maintenance")

            # Apply retention policies
            self.retention_manager.apply_retention_policies()

            # Run health check
            if self.config.enable_health_monitoring:
                self._run_health_check()

            self.management_logger.info("Log maintenance completed")

        except Exception as e:
            self.management_logger.error(f"Maintenance failed: {e}")

    def _run_health_check(self):
        """Run health check and log results."""
        try:
            health_status = self.health_monitor.check_log_health()

            if health_status["status"] == "healthy":
                self.management_logger.info("Log system health check: OK")
            else:
                self.management_logger.warning(f"Log system health issues detected: {health_status['issues']}")

        except Exception as e:
            self.management_logger.error(f"Health check failed: {e}")

    def get_health_status(self) -> Dict[str, any]:
        """Get current health status."""
        return self.health_monitor.check_log_health()

    def manual_cleanup(self) -> Dict[str, int]:
        """Manually trigger cleanup and return statistics."""
        self.management_logger.info("Manual cleanup triggered")

        try:
            self.retention_manager.apply_retention_policies()
            health_status = self.health_monitor.check_log_health()

            return {
                "status": "success",
                "total_size_mb": health_status["metrics"]["total_size_mb"],
                "file_count": len(health_status["metrics"]["file_stats"]),
            }

        except Exception as e:
            self.management_logger.error(f"Manual cleanup failed: {e}")
            return {"status": "error", "error": str(e)}
