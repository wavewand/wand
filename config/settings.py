"""
Application Settings

Defines structured settings classes with validation and type safety
for all application components.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .manager import ConfigManager, ConfigSchema, Environment


@dataclass
class APISettings:
    """API server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_credentials: bool = True
    cors_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 300  # 5 minutes

    @classmethod
    def get_schemas(cls) -> List[ConfigSchema]:
        """Get configuration schemas for API settings."""
        return [
            ConfigSchema("api_host", default_value="0.0.0.0", env_var="API_HOST"),
            ConfigSchema("api_port", data_type=int, default_value=8000, env_var="API_PORT"),
            ConfigSchema("api_debug", data_type=bool, default_value=False, env_var="DEBUG"),
            ConfigSchema("api_reload", data_type=bool, default_value=False, env_var="API_RELOAD"),
            ConfigSchema("api_workers", data_type=int, default_value=1, env_var="API_WORKERS"),
            ConfigSchema("api_cors_origins", data_type=list, default_value=["*"], env_var="CORS_ORIGINS"),
            ConfigSchema(
                "api_max_request_size", data_type=int, default_value=10 * 1024 * 1024, env_var="MAX_REQUEST_SIZE"
            ),
            ConfigSchema("api_request_timeout", data_type=int, default_value=43200, env_var="REQUEST_TIMEOUT"),
        ]


@dataclass
class DatabaseSettings:
    """Database connection settings."""

    url: str = "sqlite:///./mcp_platform.db"
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    migration_dir: str = "migrations"

    @classmethod
    def get_schemas(cls) -> List[ConfigSchema]:
        """Get configuration schemas for database settings."""
        return [
            ConfigSchema("database_url", default_value="sqlite:///./mcp_platform.db", env_var="DATABASE_URL"),
            ConfigSchema("database_pool_size", data_type=int, default_value=20, env_var="DB_POOL_SIZE"),
            ConfigSchema("database_max_overflow", data_type=int, default_value=30, env_var="DB_MAX_OVERFLOW"),
            ConfigSchema("database_pool_timeout", data_type=int, default_value=30, env_var="DB_POOL_TIMEOUT"),
            ConfigSchema("database_pool_recycle", data_type=int, default_value=3600, env_var="DB_POOL_RECYCLE"),
            ConfigSchema("database_echo", data_type=bool, default_value=False, env_var="DB_ECHO"),
            ConfigSchema("database_migration_dir", default_value="migrations", env_var="DB_MIGRATION_DIR"),
        ]


@dataclass
class CacheSettings:
    """Cache configuration settings."""

    enabled: bool = True
    backend: str = "memory"  # memory, redis, memcached
    redis_url: str = "redis://localhost:6379/0"
    default_ttl: int = 3600
    max_size: int = 1000
    key_prefix: str = "mcp:"

    @classmethod
    def get_schemas(cls) -> List[ConfigSchema]:
        """Get configuration schemas for cache settings."""
        return [
            ConfigSchema("cache_enabled", data_type=bool, default_value=True, env_var="CACHE_ENABLED"),
            ConfigSchema("cache_backend", default_value="memory", env_var="CACHE_BACKEND"),
            ConfigSchema("cache_redis_url", default_value="redis://localhost:6379/0", env_var="REDIS_URL"),
            ConfigSchema("cache_default_ttl", data_type=int, default_value=3600, env_var="CACHE_DEFAULT_TTL"),
            ConfigSchema("cache_max_size", data_type=int, default_value=1000, env_var="CACHE_MAX_SIZE"),
            ConfigSchema("cache_key_prefix", default_value="mcp:", env_var="CACHE_KEY_PREFIX"),
        ]


@dataclass
class SecuritySettings:
    """Security configuration settings."""

    jwt_secret: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire: int = 3600  # 1 hour
    jwt_refresh_token_expire: int = 86400 * 30  # 30 days
    api_key_length: int = 32
    password_min_length: int = 8
    password_require_special: bool = True
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_requests_per_hour: int = 1000
    rate_limit_requests_per_day: int = 10000
    cors_enabled: bool = True
    https_only: bool = False

    @classmethod
    def get_schemas(cls) -> List[ConfigSchema]:
        """Get configuration schemas for security settings."""
        return [
            ConfigSchema("security_jwt_secret", required=True, env_var="JWT_SECRET"),
            ConfigSchema("security_jwt_algorithm", default_value="HS256", env_var="JWT_ALGORITHM"),
            ConfigSchema(
                "security_jwt_access_token_expire", data_type=int, default_value=3600, env_var="JWT_ACCESS_EXPIRE"
            ),
            ConfigSchema(
                "security_jwt_refresh_token_expire",
                data_type=int,
                default_value=86400 * 30,
                env_var="JWT_REFRESH_EXPIRE",
            ),
            ConfigSchema("security_api_key_length", data_type=int, default_value=32, env_var="API_KEY_LENGTH"),
            ConfigSchema("security_password_min_length", data_type=int, default_value=8, env_var="PASSWORD_MIN_LENGTH"),
            ConfigSchema(
                "security_password_require_special",
                data_type=bool,
                default_value=True,
                env_var="PASSWORD_REQUIRE_SPECIAL",
            ),
            ConfigSchema(
                "security_rate_limit_enabled", data_type=bool, default_value=True, env_var="RATE_LIMIT_ENABLED"
            ),
            ConfigSchema(
                "security_rate_limit_per_minute", data_type=int, default_value=60, env_var="RATE_LIMIT_PER_MINUTE"
            ),
            ConfigSchema(
                "security_rate_limit_per_hour", data_type=int, default_value=1000, env_var="RATE_LIMIT_PER_HOUR"
            ),
            ConfigSchema(
                "security_rate_limit_per_day", data_type=int, default_value=10000, env_var="RATE_LIMIT_PER_DAY"
            ),
            ConfigSchema("security_cors_enabled", data_type=bool, default_value=True, env_var="CORS_ENABLED"),
            ConfigSchema("security_https_only", data_type=bool, default_value=False, env_var="HTTPS_ONLY"),
        ]


@dataclass
class MonitoringSettings:
    """Monitoring and observability settings."""

    enabled: bool = True
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    logging_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    metrics_port: int = 9090
    health_check_interval: int = 30
    jaeger_endpoint: Optional[str] = None
    prometheus_enabled: bool = False

    @classmethod
    def get_schemas(cls) -> List[ConfigSchema]:
        """Get configuration schemas for monitoring settings."""
        return [
            ConfigSchema("monitoring_enabled", data_type=bool, default_value=True, env_var="MONITORING_ENABLED"),
            ConfigSchema("monitoring_metrics_enabled", data_type=bool, default_value=True, env_var="METRICS_ENABLED"),
            ConfigSchema("monitoring_tracing_enabled", data_type=bool, default_value=False, env_var="TRACING_ENABLED"),
            ConfigSchema("monitoring_logging_level", default_value="INFO", env_var="LOG_LEVEL"),
            ConfigSchema("monitoring_log_format", default_value="json", env_var="LOG_FORMAT"),
            ConfigSchema("monitoring_log_file", required=False, env_var="LOG_FILE"),
            ConfigSchema("monitoring_metrics_port", data_type=int, default_value=9090, env_var="METRICS_PORT"),
            ConfigSchema(
                "monitoring_health_check_interval", data_type=int, default_value=30, env_var="HEALTH_CHECK_INTERVAL"
            ),
            ConfigSchema("monitoring_jaeger_endpoint", required=False, env_var="JAEGER_ENDPOINT"),
            ConfigSchema(
                "monitoring_prometheus_enabled", data_type=bool, default_value=False, env_var="PROMETHEUS_ENABLED"
            ),
        ]


@dataclass
class FrameworkSettings:
    """AI framework settings."""

    default_framework: str = "haystack"
    enabled_frameworks: List[str] = field(default_factory=lambda: ["haystack", "llamaindex"])
    framework_timeout: int = 43200  # 12 hours
    max_concurrent_requests: int = 10
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0

    # Framework-specific settings
    haystack_api_key: Optional[str] = None
    haystack_endpoint: Optional[str] = None
    llamaindex_api_key: Optional[str] = None
    llamaindex_endpoint: Optional[str] = None

    @classmethod
    def get_schemas(cls) -> List[ConfigSchema]:
        """Get configuration schemas for framework settings."""
        return [
            ConfigSchema("framework_default", default_value="haystack", env_var="DEFAULT_FRAMEWORK"),
            ConfigSchema(
                "framework_enabled",
                data_type=list,
                default_value=["haystack", "llamaindex"],
                env_var="ENABLED_FRAMEWORKS",
            ),
            ConfigSchema("framework_timeout", data_type=int, default_value=43200, env_var="FRAMEWORK_TIMEOUT"),
            ConfigSchema(
                "framework_max_concurrent", data_type=int, default_value=10, env_var="FRAMEWORK_MAX_CONCURRENT"
            ),
            ConfigSchema(
                "framework_circuit_breaker_enabled",
                data_type=bool,
                default_value=True,
                env_var="CIRCUIT_BREAKER_ENABLED",
            ),
            ConfigSchema(
                "framework_circuit_breaker_threshold",
                data_type=int,
                default_value=5,
                env_var="CIRCUIT_BREAKER_THRESHOLD",
            ),
            ConfigSchema(
                "framework_circuit_breaker_timeout",
                data_type=int,
                default_value=43200,
                env_var="CIRCUIT_BREAKER_TIMEOUT",
            ),
            ConfigSchema("framework_retry_enabled", data_type=bool, default_value=True, env_var="RETRY_ENABLED"),
            ConfigSchema("framework_retry_max_attempts", data_type=int, default_value=3, env_var="RETRY_MAX_ATTEMPTS"),
            ConfigSchema("framework_retry_base_delay", data_type=float, default_value=1.0, env_var="RETRY_BASE_DELAY"),
            ConfigSchema("haystack_api_key", required=False, env_var="HAYSTACK_API_KEY"),
            ConfigSchema("haystack_endpoint", required=False, env_var="HAYSTACK_ENDPOINT"),
            ConfigSchema("llamaindex_api_key", required=False, env_var="LLAMAINDEX_API_KEY"),
            ConfigSchema("llamaindex_endpoint", required=False, env_var="LLAMAINDEX_ENDPOINT"),
        ]


@dataclass
class Settings:
    """Main application settings container."""

    environment: Environment = Environment.DEVELOPMENT
    api: APISettings = field(default_factory=APISettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = field(default_factory=MonitoringSettings)
    frameworks: FrameworkSettings = field(default_factory=FrameworkSettings)

    @classmethod
    def from_config_manager(cls, config_manager: ConfigManager) -> 'Settings':
        """Create settings from configuration manager."""
        config = config_manager.get_all()

        return cls(
            environment=config_manager.environment,
            api=APISettings(
                host=config.get("api_host", "0.0.0.0"),
                port=config.get("api_port", 8000),
                debug=config.get("api_debug", False),
                reload=config.get("api_reload", False),
                workers=config.get("api_workers", 1),
                cors_origins=config.get("api_cors_origins", ["*"]),
                max_request_size=config.get("api_max_request_size", 10 * 1024 * 1024),
                request_timeout=config.get("api_request_timeout", 300),
            ),
            database=DatabaseSettings(
                url=config.get("database_url", "sqlite:///./mcp_platform.db"),
                pool_size=config.get("database_pool_size", 20),
                max_overflow=config.get("database_max_overflow", 30),
                pool_timeout=config.get("database_pool_timeout", 30),
                pool_recycle=config.get("database_pool_recycle", 3600),
                echo=config.get("database_echo", False),
                migration_dir=config.get("database_migration_dir", "migrations"),
            ),
            cache=CacheSettings(
                enabled=config.get("cache_enabled", True),
                backend=config.get("cache_backend", "memory"),
                redis_url=config.get("cache_redis_url", "redis://localhost:6379/0"),
                default_ttl=config.get("cache_default_ttl", 3600),
                max_size=config.get("cache_max_size", 1000),
                key_prefix=config.get("cache_key_prefix", "mcp:"),
            ),
            security=SecuritySettings(
                jwt_secret=config.get("security_jwt_secret", "your-secret-key-change-in-production"),
                jwt_algorithm=config.get("security_jwt_algorithm", "HS256"),
                jwt_access_token_expire=config.get("security_jwt_access_token_expire", 3600),
                jwt_refresh_token_expire=config.get("security_jwt_refresh_token_expire", 86400 * 30),
                api_key_length=config.get("security_api_key_length", 32),
                password_min_length=config.get("security_password_min_length", 8),
                password_require_special=config.get("security_password_require_special", True),
                rate_limit_enabled=config.get("security_rate_limit_enabled", True),
                rate_limit_requests_per_minute=config.get("security_rate_limit_per_minute", 60),
                rate_limit_requests_per_hour=config.get("security_rate_limit_per_hour", 1000),
                rate_limit_requests_per_day=config.get("security_rate_limit_per_day", 10000),
                cors_enabled=config.get("security_cors_enabled", True),
                https_only=config.get("security_https_only", False),
            ),
            monitoring=MonitoringSettings(
                enabled=config.get("monitoring_enabled", True),
                metrics_enabled=config.get("monitoring_metrics_enabled", True),
                tracing_enabled=config.get("monitoring_tracing_enabled", False),
                logging_level=config.get("monitoring_logging_level", "INFO"),
                log_format=config.get("monitoring_log_format", "json"),
                log_file=config.get("monitoring_log_file"),
                metrics_port=config.get("monitoring_metrics_port", 9090),
                health_check_interval=config.get("monitoring_health_check_interval", 30),
                jaeger_endpoint=config.get("monitoring_jaeger_endpoint"),
                prometheus_enabled=config.get("monitoring_prometheus_enabled", False),
            ),
            frameworks=FrameworkSettings(
                default_framework=config.get("framework_default", "haystack"),
                enabled_frameworks=config.get("framework_enabled", ["haystack", "llamaindex"]),
                framework_timeout=config.get("framework_timeout", 300),
                max_concurrent_requests=config.get("framework_max_concurrent", 10),
                circuit_breaker_enabled=config.get("framework_circuit_breaker_enabled", True),
                circuit_breaker_failure_threshold=config.get("framework_circuit_breaker_threshold", 5),
                circuit_breaker_recovery_timeout=config.get("framework_circuit_breaker_timeout", 60),
                retry_enabled=config.get("framework_retry_enabled", True),
                retry_max_attempts=config.get("framework_retry_max_attempts", 3),
                retry_base_delay=config.get("framework_retry_base_delay", 1.0),
                haystack_api_key=config.get("haystack_api_key"),
                haystack_endpoint=config.get("haystack_endpoint"),
                llamaindex_api_key=config.get("llamaindex_api_key"),
                llamaindex_endpoint=config.get("llamaindex_endpoint"),
            ),
        )

    @classmethod
    def get_all_schemas(cls) -> List[ConfigSchema]:
        """Get all configuration schemas."""
        schemas = []
        schemas.extend(APISettings.get_schemas())
        schemas.extend(DatabaseSettings.get_schemas())
        schemas.extend(CacheSettings.get_schemas())
        schemas.extend(SecuritySettings.get_schemas())
        schemas.extend(MonitoringSettings.get_schemas())
        schemas.extend(FrameworkSettings.get_schemas())
        return schemas


# Global settings instance
_settings: Optional[Settings] = None


def initialize_settings(config_manager: ConfigManager) -> Settings:
    """Initialize global settings from configuration manager."""
    global _settings

    # Register all schemas
    schemas = Settings.get_all_schemas()
    config_manager.register_schemas(schemas)

    # Load configuration
    config_manager.load_config()

    # Create settings
    _settings = Settings.from_config_manager(config_manager)

    return _settings


def get_settings() -> Settings:
    """Get global settings instance."""
    if _settings is None:
        raise RuntimeError("Settings not initialized. Call initialize_settings first.")
    return _settings


def reload_settings(config_manager: ConfigManager) -> Settings:
    """Reload settings from configuration manager."""
    return initialize_settings(config_manager)
