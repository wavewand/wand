"""
Secrets Management

Provides secure handling of sensitive configuration data including API keys,
passwords, and other secrets with multiple provider backends.
"""

import base64
import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from utils.error_handling import ErrorCategory, ErrorSeverity, MCPError


class SecretsError(MCPError):
    """Secrets management error."""

    def __init__(self, message: str, secret_key: str = None):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION_ERROR,
            severity=ErrorSeverity.HIGH,
            details={"secret_key": secret_key},
        )


@dataclass
class Secret:
    """Represents a secret value with metadata."""

    key: str
    value: str
    encrypted: bool = False
    source: str = "unknown"
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SecretProvider(ABC):
    """Abstract base class for secret providers."""

    @abstractmethod
    async def get_secret(self, key: str) -> Optional[Secret]:
        """Get secret by key."""
        pass

    @abstractmethod
    async def set_secret(self, secret: Secret) -> bool:
        """Set secret value."""
        pass

    @abstractmethod
    async def delete_secret(self, key: str) -> bool:
        """Delete secret by key."""
        pass

    @abstractmethod
    async def list_secrets(self) -> List[str]:
        """List all secret keys."""
        pass


class EnvironmentSecretProvider(SecretProvider):
    """Environment variables secret provider."""

    def __init__(self, prefix: str = "MCP_SECRET_"):
        self.prefix = prefix
        self.logger = logging.getLogger(__name__)

    async def get_secret(self, key: str) -> Optional[Secret]:
        """Get secret from environment variables."""
        env_key = f"{self.prefix}{key.upper()}"
        value = os.getenv(env_key)

        if value is not None:
            return Secret(key=key, value=value, source="environment", metadata={"env_key": env_key})

        return None

    async def set_secret(self, secret: Secret) -> bool:
        """Set secret in environment (runtime only)."""
        env_key = f"{self.prefix}{secret.key.upper()}"
        os.environ[env_key] = secret.value
        self.logger.info(f"Set secret {secret.key} in environment")
        return True

    async def delete_secret(self, key: str) -> bool:
        """Delete secret from environment."""
        env_key = f"{self.prefix}{key.upper()}"
        if env_key in os.environ:
            del os.environ[env_key]
            self.logger.info(f"Deleted secret {key} from environment")
            return True
        return False

    async def list_secrets(self) -> List[str]:
        """List all secret keys from environment."""
        keys = []
        for env_key in os.environ:
            if env_key.startswith(self.prefix):
                key = env_key[len(self.prefix) :].lower()
                keys.append(key)
        return keys


class FileSecretProvider(SecretProvider):
    """File-based secret provider with encryption."""

    def __init__(self, secrets_file: str, encryption_key: Optional[str] = None):
        self.secrets_file = Path(secrets_file)
        self.logger = logging.getLogger(__name__)

        # Initialize encryption
        if encryption_key:
            self.cipher = self._create_cipher(encryption_key)
        else:
            self.cipher = None

        # Ensure secrets file exists
        if not self.secrets_file.exists():
            self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_secrets({})

    def _create_cipher(self, password: str) -> Fernet:
        """Create encryption cipher from password."""
        password_bytes = password.encode()
        salt = b'salt_'  # In production, use a proper random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return Fernet(key)

    def _encrypt_value(self, value: str) -> str:
        """Encrypt secret value."""
        if self.cipher:
            encrypted = self.cipher.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        return value

    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt secret value."""
        if self.cipher:
            try:
                encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
                decrypted = self.cipher.decrypt(encrypted_bytes)
                return decrypted.decode()
            except Exception as e:
                raise SecretsError(f"Failed to decrypt secret: {e}")
        return encrypted_value

    def _load_secrets(self) -> Dict[str, Dict[str, Any]]:
        """Load secrets from file."""
        try:
            if self.secrets_file.exists():
                with open(self.secrets_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load secrets file: {e}")
            return {}

    def _save_secrets(self, secrets: Dict[str, Dict[str, Any]]):
        """Save secrets to file."""
        try:
            with open(self.secrets_file, 'w') as f:
                json.dump(secrets, f, indent=2)

            # Set restrictive permissions
            os.chmod(self.secrets_file, 0o600)

        except Exception as e:
            self.logger.error(f"Failed to save secrets file: {e}")
            raise SecretsError(f"Failed to save secrets: {e}")

    async def get_secret(self, key: str) -> Optional[Secret]:
        """Get secret from file."""
        secrets = self._load_secrets()

        if key in secrets:
            secret_data = secrets[key]
            value = secret_data.get("value", "")

            # Decrypt if encrypted
            if secret_data.get("encrypted", False):
                value = self._decrypt_value(value)

            return Secret(
                key=key,
                value=value,
                encrypted=secret_data.get("encrypted", False),
                source="file",
                created_at=secret_data.get("created_at"),
                expires_at=secret_data.get("expires_at"),
                metadata=secret_data.get("metadata", {}),
            )

        return None

    async def set_secret(self, secret: Secret) -> bool:
        """Set secret in file."""
        secrets = self._load_secrets()

        value = secret.value
        encrypted = False

        # Encrypt if cipher is available
        if self.cipher:
            value = self._encrypt_value(value)
            encrypted = True

        secrets[secret.key] = {
            "value": value,
            "encrypted": encrypted,
            "created_at": secret.created_at,
            "expires_at": secret.expires_at,
            "metadata": secret.metadata or {},
        }

        self._save_secrets(secrets)
        self.logger.info(f"Set secret {secret.key} in file")
        return True

    async def delete_secret(self, key: str) -> bool:
        """Delete secret from file."""
        secrets = self._load_secrets()

        if key in secrets:
            del secrets[key]
            self._save_secrets(secrets)
            self.logger.info(f"Deleted secret {key} from file")
            return True

        return False

    async def list_secrets(self) -> List[str]:
        """List all secret keys from file."""
        secrets = self._load_secrets()
        return list(secrets.keys())


class SecretsManager:
    """Main secrets manager with multiple providers."""

    def __init__(self):
        self.providers: Dict[str, SecretProvider] = {}
        self.provider_priority: List[str] = []
        self.logger = logging.getLogger(__name__)

        # Cache for frequently accessed secrets
        self._cache: Dict[str, Secret] = {}
        self._cache_enabled = True

    def add_provider(self, name: str, provider: SecretProvider, priority: int = 0):
        """Add secret provider with priority."""
        self.providers[name] = provider

        # Insert provider in priority order (higher priority = earlier in list)
        if name not in self.provider_priority:
            inserted = False
            for i, existing_name in enumerate(self.provider_priority):
                if priority > getattr(self.providers[existing_name], 'priority', 0):
                    self.provider_priority.insert(i, name)
                    inserted = True
                    break

            if not inserted:
                self.provider_priority.append(name)

        setattr(provider, 'priority', priority)
        self.logger.info(f"Added secret provider: {name} with priority {priority}")

    def remove_provider(self, name: str):
        """Remove secret provider."""
        if name in self.providers:
            del self.providers[name]
            self.provider_priority.remove(name)
            self.logger.info(f"Removed secret provider: {name}")

    async def get_secret(self, key: str, provider_name: str = None) -> Optional[str]:
        """Get secret value by key."""
        # Check cache first
        if self._cache_enabled and key in self._cache:
            return self._cache[key].value

        # Use specific provider if specified
        if provider_name:
            if provider_name in self.providers:
                secret = await self.providers[provider_name].get_secret(key)
                if secret:
                    # Cache the secret
                    if self._cache_enabled:
                        self._cache[key] = secret
                    return secret.value
            return None

        # Try providers in priority order
        for provider_name in self.provider_priority:
            provider = self.providers[provider_name]
            try:
                secret = await provider.get_secret(key)
                if secret:
                    # Cache the secret
                    if self._cache_enabled:
                        self._cache[key] = secret

                    self.logger.debug(f"Retrieved secret {key} from provider {provider_name}")
                    return secret.value
            except Exception as e:
                self.logger.warning(f"Provider {provider_name} failed to get secret {key}: {e}")
                continue

        self.logger.warning(f"Secret {key} not found in any provider")
        return None

    async def set_secret(
        self, key: str, value: str, provider_name: str = None, expires_at: str = None, metadata: Dict[str, Any] = None
    ) -> bool:
        """Set secret value."""
        secret = Secret(
            key=key, value=value, source=provider_name or "unknown", expires_at=expires_at, metadata=metadata
        )

        # Use specific provider if specified
        if provider_name:
            if provider_name in self.providers:
                success = await self.providers[provider_name].set_secret(secret)
                if success and self._cache_enabled:
                    self._cache[key] = secret
                return success
            return False

        # Try to set in the highest priority provider
        if self.provider_priority:
            provider_name = self.provider_priority[0]
            provider = self.providers[provider_name]
            try:
                success = await provider.set_secret(secret)
                if success and self._cache_enabled:
                    self._cache[key] = secret
                self.logger.info(f"Set secret {key} using provider {provider_name}")
                return success
            except Exception as e:
                self.logger.error(f"Failed to set secret {key} using provider {provider_name}: {e}")

        return False

    async def delete_secret(self, key: str, provider_name: str = None) -> bool:
        """Delete secret by key."""
        success = False

        # Remove from cache
        if key in self._cache:
            del self._cache[key]

        # Use specific provider if specified
        if provider_name:
            if provider_name in self.providers:
                return await self.providers[provider_name].delete_secret(key)
            return False

        # Try to delete from all providers
        for provider_name in self.provider_priority:
            provider = self.providers[provider_name]
            try:
                result = await provider.delete_secret(key)
                if result:
                    success = True
                    self.logger.info(f"Deleted secret {key} from provider {provider_name}")
            except Exception as e:
                self.logger.warning(f"Provider {provider_name} failed to delete secret {key}: {e}")

        return success

    async def list_secrets(self, provider_name: str = None) -> List[str]:
        """List all secret keys."""
        if provider_name:
            if provider_name in self.providers:
                return await self.providers[provider_name].list_secrets()
            return []

        # Combine keys from all providers
        all_keys = set()
        for provider_name in self.provider_priority:
            provider = self.providers[provider_name]
            try:
                keys = await provider.list_secrets()
                all_keys.update(keys)
            except Exception as e:
                self.logger.warning(f"Provider {provider_name} failed to list secrets: {e}")

        return list(all_keys)

    def clear_cache(self):
        """Clear secret cache."""
        self._cache.clear()
        self.logger.info("Cleared secret cache")

    def enable_cache(self, enabled: bool = True):
        """Enable or disable secret caching."""
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()
        self.logger.info(f"Secret caching {'enabled' if enabled else 'disabled'}")

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about registered providers."""
        return {
            "providers": list(self.providers.keys()),
            "priority_order": self.provider_priority,
            "cache_enabled": self._cache_enabled,
            "cached_secrets": len(self._cache),
        }


# Global secrets manager instance
secrets_manager = SecretsManager()


def initialize_secrets_manager(
    use_environment: bool = True, use_file: bool = False, file_path: str = "secrets.json", encryption_key: str = None
):
    """Initialize global secrets manager with default providers."""
    if use_environment:
        env_provider = EnvironmentSecretProvider()
        secrets_manager.add_provider("environment", env_provider, priority=100)

    if use_file:
        file_provider = FileSecretProvider(file_path, encryption_key)
        secrets_manager.add_provider("file", file_provider, priority=50)


async def get_secret(key: str, provider_name: str = None) -> Optional[str]:
    """Get secret from global secrets manager."""
    return await secrets_manager.get_secret(key, provider_name)
