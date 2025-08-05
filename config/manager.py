"""
Configuration Manager

Provides centralized configuration management with environment-specific settings,
validation, hot-reloading, and change notifications.
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from utils.error_handling import ErrorCategory, ErrorSeverity, MCPError


class Environment(str, Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigFormat(str, Enum):
    """Configuration file formats."""

    JSON = "json"
    YAML = "yaml"
    ENV = "env"


class ConfigValidationError(MCPError):
    """Configuration validation error."""

    def __init__(self, message: str, config_key: str = None, config_value: Any = None):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION_ERROR,
            severity=ErrorSeverity.HIGH,
            details={"config_key": config_key, "config_value": str(config_value)},
        )


@dataclass
class ConfigSchema:
    """Configuration schema definition."""

    key: str
    required: bool = True
    data_type: type = str
    default_value: Any = None
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""
    env_var: Optional[str] = None

    def validate(self, value: Any) -> Any:
        """Validate configuration value."""
        if value is None:
            if self.required:
                raise ConfigValidationError(f"Required configuration key '{self.key}' is missing")
            return self.default_value

        # Type conversion
        try:
            if self.data_type == bool and isinstance(value, str):
                value = value.lower() in ("true", "1", "yes", "on")
            elif self.data_type == int:
                value = int(value)
            elif self.data_type == float:
                value = float(value)
            elif self.data_type == list and isinstance(value, str):
                value = [item.strip() for item in value.split(",")]
            else:
                value = self.data_type(value)
        except (ValueError, TypeError) as e:
            raise ConfigValidationError(
                f"Invalid type for '{self.key}': expected {self.data_type.__name__}, got {type(value).__name__}"
            )

        # Custom validation
        if self.validator and not self.validator(value):
            raise ConfigValidationError(f"Validation failed for '{self.key}' with value '{value}'")

        return value


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""

    environment: Environment
    config_files: List[str] = field(default_factory=list)
    config_data: Dict[str, Any] = field(default_factory=dict)
    last_modified: Optional[float] = None
    checksum: Optional[str] = None


class ConfigFileWatcher(FileSystemEventHandler):
    """File system watcher for configuration files."""

    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            file_path = event.src_path
            if any(file_path.endswith(ext) for ext in ['.json', '.yaml', '.yml', '.env']):
                self.logger.info(f"Configuration file modified: {file_path}")
                asyncio.create_task(self.config_manager.reload_config())


class ConfigManager:
    """Centralized configuration management."""

    def __init__(self, base_path: str = ".", environment: Environment = None):
        self.base_path = Path(base_path)
        self.environment = environment or self._detect_environment()
        self.logger = logging.getLogger(__name__)

        # Configuration storage
        self.configs: Dict[Environment, EnvironmentConfig] = {}
        self.schemas: Dict[str, ConfigSchema] = {}
        self.change_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # File watching
        self.observer = None
        self.watch_enabled = False

        # Thread safety
        self.lock = threading.RLock()

        self.logger.info(f"ConfigManager initialized for environment: {self.environment.value}")

    def _detect_environment(self) -> Environment:
        """Detect environment from environment variables."""
        env_name = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()

        try:
            return Environment(env_name)
        except ValueError:
            self.logger.warning(f"Unknown environment '{env_name}', defaulting to development")
            return Environment.DEVELOPMENT

    def register_schema(self, schema: ConfigSchema):
        """Register configuration schema."""
        with self.lock:
            self.schemas[schema.key] = schema
            self.logger.debug(f"Registered schema for '{schema.key}'")

    def register_schemas(self, schemas: List[ConfigSchema]):
        """Register multiple configuration schemas."""
        for schema in schemas:
            self.register_schema(schema)

    def add_config_file(self, file_path: str, environment: Environment = None):
        """Add configuration file for specific environment."""
        environment = environment or self.environment
        file_path = str(self.base_path / file_path)

        with self.lock:
            if environment not in self.configs:
                self.configs[environment] = EnvironmentConfig(environment)

            if file_path not in self.configs[environment].config_files:
                self.configs[environment].config_files.append(file_path)
                self.logger.info(f"Added config file: {file_path} for environment: {environment.value}")

    def load_config(self, environment: Environment = None) -> Dict[str, Any]:
        """Load configuration for specified environment."""
        environment = environment or self.environment

        with self.lock:
            if environment not in self.configs:
                raise ConfigValidationError(f"No configuration defined for environment: {environment.value}")

            env_config = self.configs[environment]
            merged_config = {}

            # Load from files
            for file_path in env_config.config_files:
                file_config = self._load_config_file(file_path)
                merged_config.update(file_config)

            # Override with environment variables
            env_overrides = self._load_env_overrides()
            merged_config.update(env_overrides)

            # Validate against schemas
            validated_config = self._validate_config(merged_config)

            # Update environment config
            env_config.config_data = validated_config
            env_config.last_modified = time.time()
            env_config.checksum = self._calculate_checksum(validated_config)

            self.logger.info(f"Loaded configuration for environment: {environment.value}")
            return validated_config

    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        if not os.path.exists(file_path):
            self.logger.warning(f"Configuration file not found: {file_path}")
            return {}

        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    return json.load(f)
                elif file_path.endswith(('.yaml', '.yml')):
                    return yaml.safe_load(f) or {}
                elif file_path.endswith('.env'):
                    return self._parse_env_file(f.read())
                else:
                    self.logger.warning(f"Unsupported config file format: {file_path}")
                    return {}

        except Exception as e:
            self.logger.error(f"Error loading config file {file_path}: {e}")
            raise ConfigValidationError(f"Failed to load config file: {file_path}")

    def _parse_env_file(self, content: str) -> Dict[str, Any]:
        """Parse .env file content."""
        config = {}

        for line in content.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    config[key] = value

        return config

    def _load_env_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}

        for key, schema in self.schemas.items():
            if schema.env_var:
                env_value = os.getenv(schema.env_var)
                if env_value is not None:
                    overrides[key] = env_value

        return overrides

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against registered schemas."""
        validated = {}

        for key, schema in self.schemas.items():
            value = config.get(key)
            validated[key] = schema.validate(value)

        # Include non-schema keys as-is
        for key, value in config.items():
            if key not in self.schemas:
                validated[key] = value

        return validated

    def _calculate_checksum(self, config: Dict[str, Any]) -> str:
        """Calculate checksum for configuration data."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def get(self, key: str, default: Any = None, environment: Environment = None) -> Any:
        """Get configuration value."""
        environment = environment or self.environment

        with self.lock:
            if environment in self.configs:
                return self.configs[environment].config_data.get(key, default)

            # Try to load config if not already loaded
            try:
                config = self.load_config(environment)
                return config.get(key, default)
            except Exception:
                return default

    def set(self, key: str, value: Any, environment: Environment = None):
        """Set configuration value (runtime only)."""
        environment = environment or self.environment

        with self.lock:
            if environment not in self.configs:
                self.configs[environment] = EnvironmentConfig(environment)

            # Validate if schema exists
            if key in self.schemas:
                value = self.schemas[key].validate(value)

            self.configs[environment].config_data[key] = value
            self.logger.debug(f"Set config {key} = {value} for environment {environment.value}")

            # Notify change callbacks
            self._notify_changes({key: value})

    def update(self, updates: Dict[str, Any], environment: Environment = None):
        """Update multiple configuration values."""
        for key, value in updates.items():
            self.set(key, value, environment)

    def get_all(self, environment: Environment = None) -> Dict[str, Any]:
        """Get all configuration values."""
        environment = environment or self.environment

        with self.lock:
            if environment in self.configs:
                return self.configs[environment].config_data.copy()

            # Try to load config if not already loaded
            try:
                return self.load_config(environment)
            except Exception:
                return {}

    async def reload_config(self, environment: Environment = None):
        """Reload configuration from files."""
        environment = environment or self.environment

        try:
            old_config = self.get_all(environment)
            new_config = self.load_config(environment)

            # Check for changes
            if old_config != new_config:
                changes = {k: v for k, v in new_config.items() if old_config.get(k) != v}

                if changes:
                    self.logger.info(f"Configuration reloaded with {len(changes)} changes")
                    self._notify_changes(changes)
                else:
                    self.logger.debug("Configuration reloaded with no changes")

        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
            raise

    def start_watching(self):
        """Start watching configuration files for changes."""
        if self.watch_enabled:
            return

        try:
            self.observer = Observer()
            event_handler = ConfigFileWatcher(self)

            # Watch the base directory
            self.observer.schedule(event_handler, str(self.base_path), recursive=True)
            self.observer.start()

            self.watch_enabled = True
            self.logger.info("Started watching configuration files for changes")

        except Exception as e:
            self.logger.error(f"Failed to start file watching: {e}")

    def stop_watching(self):
        """Stop watching configuration files."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.watch_enabled = False
            self.logger.info("Stopped watching configuration files")

    def add_change_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for configuration changes."""
        self.change_callbacks.append(callback)
        self.logger.debug("Added configuration change callback")

    def remove_change_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove configuration change callback."""
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
            self.logger.debug("Removed configuration change callback")

    def _notify_changes(self, changes: Dict[str, Any]):
        """Notify all registered callbacks of configuration changes."""
        for callback in self.change_callbacks:
            try:
                callback(changes)
            except Exception as e:
                self.logger.error(f"Error in configuration change callback: {e}")

    def export_config(self, format: ConfigFormat = ConfigFormat.JSON, environment: Environment = None) -> str:
        """Export configuration in specified format."""
        config = self.get_all(environment)

        if format == ConfigFormat.JSON:
            return json.dumps(config, indent=2, default=str)
        elif format == ConfigFormat.YAML:
            return yaml.dump(config, default_flow_style=False)
        elif format == ConfigFormat.ENV:
            lines = []
            for key, value in config.items():
                lines.append(f"{key}={value}")
            return '\n'.join(lines)
        else:
            raise ConfigValidationError(f"Unsupported export format: {format.value}")

    def get_schema_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about registered schemas."""
        return {
            key: {
                "required": schema.required,
                "type": schema.data_type.__name__,
                "default": schema.default_value,
                "description": schema.description,
                "env_var": schema.env_var,
            }
            for key, schema in self.schemas.items()
        }

    def validate_current_config(self) -> Dict[str, List[str]]:
        """Validate current configuration and return any errors."""
        errors = {}

        for environment, env_config in self.configs.items():
            env_errors = []

            for key, schema in self.schemas.items():
                try:
                    value = env_config.config_data.get(key)
                    schema.validate(value)
                except ConfigValidationError as e:
                    env_errors.append(str(e))

            if env_errors:
                errors[environment.value] = env_errors

        return errors


# Global configuration manager instance
config_manager = None


def initialize_config_manager(base_path: str = ".", environment: Environment = None):
    """Initialize global configuration manager."""
    global config_manager
    config_manager = ConfigManager(base_path, environment)
    return config_manager


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value from global manager."""
    if config_manager is None:
        raise ConfigValidationError("Configuration manager not initialized")
    return config_manager.get(key, default)


async def reload_config():
    """Reload configuration from global manager."""
    if config_manager is None:
        raise ConfigValidationError("Configuration manager not initialized")
    await config_manager.reload_config()


def watch_config():
    """Start watching configuration files."""
    if config_manager is None:
        raise ConfigValidationError("Configuration manager not initialized")
    config_manager.start_watching()
