"""
Configuration Management Package

Provides comprehensive configuration management with environment-specific settings,
hot-reloading, validation, and secrets management.
"""

from .manager import (
    ConfigManager,
    ConfigSchema,
    ConfigValidationError,
    EnvironmentConfig,
    config_manager,
    get_config,
    reload_config,
    watch_config,
)
from .secrets import (
    EnvironmentSecretProvider,
    FileSecretProvider,
    SecretProvider,
    SecretsManager,
    get_secret,
    secrets_manager,
)
from .settings import (
    APISettings,
    CacheSettings,
    DatabaseSettings,
    FrameworkSettings,
    MonitoringSettings,
    SecuritySettings,
    Settings,
    get_settings,
)

__all__ = [
    'ConfigManager',
    'EnvironmentConfig',
    'ConfigSchema',
    'ConfigValidationError',
    'config_manager',
    'get_config',
    'reload_config',
    'watch_config',
    'APISettings',
    'DatabaseSettings',
    'CacheSettings',
    'SecuritySettings',
    'MonitoringSettings',
    'FrameworkSettings',
    'Settings',
    'get_settings',
    'SecretsManager',
    'SecretProvider',
    'EnvironmentSecretProvider',
    'FileSecretProvider',
    'secrets_manager',
    'get_secret',
]
