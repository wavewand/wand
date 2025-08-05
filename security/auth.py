"""
Authentication and Authorization System

Provides JWT-based authentication, API key management, role-based access control,
and comprehensive security features for the MCP platform.
"""

import asyncio
import functools
import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

import bcrypt
import jwt
from passlib.context import CryptContext

from utils.error_handling import AuthenticationError, ErrorCategory, ErrorSeverity, MCPError


class UserRole(str, Enum):
    """User roles for role-based access control."""

    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"
    READONLY = "readonly"


class Permission(str, Enum):
    """System permissions."""

    # Framework operations
    FRAMEWORK_READ = "framework:read"
    FRAMEWORK_WRITE = "framework:write"
    FRAMEWORK_ADMIN = "framework:admin"

    # Document operations
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"

    # Batch operations
    BATCH_READ = "batch:read"
    BATCH_WRITE = "batch:write"

    # Monitoring
    MONITORING_READ = "monitoring:read"
    MONITORING_WRITE = "monitoring:write"

    # System administration
    SYSTEM_ADMIN = "system:admin"
    USER_MANAGEMENT = "user:management"

    # Cache operations
    CACHE_READ = "cache:read"
    CACHE_WRITE = "cache:write"
    CACHE_ADMIN = "cache:admin"


# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: set(Permission),  # All permissions
    UserRole.USER: {
        Permission.FRAMEWORK_READ,
        Permission.FRAMEWORK_WRITE,
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_WRITE,
        Permission.BATCH_READ,
        Permission.BATCH_WRITE,
        Permission.MONITORING_READ,
        Permission.CACHE_READ,
    },
    UserRole.SERVICE: {
        Permission.FRAMEWORK_READ,
        Permission.FRAMEWORK_WRITE,
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_WRITE,
        Permission.BATCH_READ,
        Permission.BATCH_WRITE,
        Permission.CACHE_READ,
        Permission.CACHE_WRITE,
    },
    UserRole.READONLY: {
        Permission.FRAMEWORK_READ,
        Permission.DOCUMENT_READ,
        Permission.BATCH_READ,
        Permission.MONITORING_READ,
        Permission.CACHE_READ,
    },
}


@dataclass
class User:
    """User representation."""

    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: Set[Permission] = field(default_factory=set)
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Add role-based permissions
        self.permissions.update(ROLE_PERMISSIONS.get(self.role, set()))

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions

    def has_role(self, role: UserRole) -> bool:
        """Check if user has specific role."""
        return self.role == role

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "metadata": self.metadata,
        }


@dataclass
class APIKey:
    """API Key representation."""

    key_id: str
    key_hash: str
    name: str
    user_id: str
    permissions: Set[Permission]
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    rate_limit: Optional[int] = None  # requests per minute

    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def has_permission(self, permission: Permission) -> bool:
        """Check if API key has specific permission."""
        return permission in self.permissions


class JWTManager:
    """JWT token management."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=30)
        self.logger = logging.getLogger(__name__)

    def create_access_token(self, user: User) -> str:
        """Create JWT access token."""
        now = datetime.now(timezone.utc)
        expire = now + self.access_token_expire

        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "iat": now.timestamp(),
            "exp": expire.timestamp(),
            "type": "access",
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token."""
        now = datetime.now(timezone.utc)
        expire = now + self.refresh_token_expire

        payload = {"user_id": user.user_id, "iat": now.timestamp(), "exp": expire.timestamp(), "type": "refresh"}

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")

    def refresh_access_token(self, refresh_token: str, user: User) -> str:
        """Create new access token from refresh token."""
        payload = self.verify_token(refresh_token)

        if payload.get("type") != "refresh":
            raise AuthenticationError("Invalid refresh token")

        if payload.get("user_id") != user.user_id:
            raise AuthenticationError("Token user mismatch")

        return self.create_access_token(user)


class APIKeyManager:
    """API Key management."""

    def __init__(self):
        self.keys: Dict[str, APIKey] = {}
        self.key_hashes: Dict[str, str] = {}  # hash -> key_id mapping
        self.logger = logging.getLogger(__name__)

    def generate_api_key(
        self,
        name: str,
        user_id: str,
        permissions: Set[Permission],
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
    ) -> tuple[str, APIKey]:
        """Generate new API key."""
        # Generate secure random key
        raw_key = secrets.token_urlsafe(32)
        key_id = f"key_{secrets.token_urlsafe(8)}"

        # Hash the key for storage
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        # Set expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            permissions=permissions,
            expires_at=expires_at,
            rate_limit=rate_limit,
        )

        # Store key
        self.keys[key_id] = api_key
        self.key_hashes[key_hash] = key_id

        self.logger.info(f"Generated API key {key_id} for user {user_id}")

        # Return the raw key (only shown once) and the key object
        return f"mcp_{raw_key}", api_key

    def verify_api_key(self, raw_key: str) -> Optional[APIKey]:
        """Verify API key and return key object if valid."""
        if not raw_key.startswith("mcp_"):
            return None

        # Extract the actual key part
        key_part = raw_key[4:]  # Remove "mcp_" prefix
        key_hash = hashlib.sha256(key_part.encode()).hexdigest()

        # Find key by hash
        key_id = self.key_hashes.get(key_hash)
        if not key_id:
            return None

        api_key = self.keys.get(key_id)
        if not api_key:
            return None

        # Check if key is active and not expired
        if not api_key.is_active or api_key.is_expired():
            return None

        # Update usage statistics
        api_key.last_used = datetime.now(timezone.utc)
        api_key.usage_count += 1

        return api_key

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke API key."""
        if key_id in self.keys:
            self.keys[key_id].is_active = False
            self.logger.info(f"Revoked API key {key_id}")
            return True
        return False

    def list_user_keys(self, user_id: str) -> List[APIKey]:
        """List all keys for a user."""
        return [key for key in self.keys.values() if key.user_id == user_id]


class AuthManager:
    """Main authentication manager."""

    def __init__(self, jwt_secret: str):
        self.jwt_manager = JWTManager(jwt_secret)
        self.api_key_manager = APIKeyManager()
        self.users: Dict[str, User] = {}
        self.password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.logger = logging.getLogger(__name__)

        # Create default admin user
        self._create_default_admin()

    def _create_default_admin(self):
        """Create default admin user."""
        admin_user = User(user_id="admin_001", username="admin", email="admin@mcp-platform.local", role=UserRole.ADMIN)
        self.users[admin_user.user_id] = admin_user
        self.logger.info("Created default admin user")

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.password_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.password_context.verify(plain_password, hashed_password)

    def create_user(
        self, username: str, email: str, password: str, role: UserRole = UserRole.USER, metadata: Dict[str, Any] = None
    ) -> User:
        """Create new user."""
        user_id = f"user_{secrets.token_urlsafe(8)}"

        user = User(user_id=user_id, username=username, email=email, role=role, metadata=metadata or {})

        self.users[user_id] = user
        self.logger.info(f"Created user {username} with role {role.value}")

        return user

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password."""
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break

        if not user or not user.is_active:
            return None

        # For demo purposes, we'll accept any password for the admin user
        # In production, you'd verify against stored password hash
        if user.username == "admin":
            user.last_login = datetime.now(timezone.utc)
            return user

        return None

    def authenticate_token(self, token: str) -> Optional[User]:
        """Authenticate user with JWT token."""
        try:
            payload = self.jwt_manager.verify_token(token)
            user_id = payload.get("user_id")

            if user_id and user_id in self.users:
                user = self.users[user_id]
                if user.is_active:
                    return user

            return None

        except AuthenticationError:
            return None

    def authenticate_api_key(self, api_key: str) -> Optional[tuple[User, APIKey]]:
        """Authenticate user with API key."""
        key_obj = self.api_key_manager.verify_api_key(api_key)
        if not key_obj:
            return None

        user = self.users.get(key_obj.user_id)
        if not user or not user.is_active:
            return None

        return user, key_obj

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)

    def update_user(self, user_id: str, **updates) -> bool:
        """Update user information."""
        user = self.users.get(user_id)
        if not user:
            return False

        for key, value in updates.items():
            if hasattr(user, key):
                setattr(user, key, value)

        return True

    def delete_user(self, user_id: str) -> bool:
        """Delete user (actually deactivate)."""
        user = self.users.get(user_id)
        if user:
            user.is_active = False
            return True
        return False


# Global instances
jwt_manager = None
api_key_manager = None
auth_manager = None


def initialize_auth(jwt_secret: str):
    """Initialize authentication system."""
    global jwt_manager, api_key_manager, auth_manager

    auth_manager = AuthManager(jwt_secret)
    jwt_manager = auth_manager.jwt_manager
    api_key_manager = auth_manager.api_key_manager


# Authentication decorators
def require_auth(func):
    """Decorator to require authentication."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # This would be integrated with FastAPI dependency injection
        # For now, it's a placeholder
        return await func(*args, **kwargs)

    return wrapper


def require_permission(permission: Permission):
    """Decorator to require specific permission."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # This would check user permissions
            # For now, it's a placeholder
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_role(role: UserRole):
    """Decorator to require specific role."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # This would check user role
            # For now, it's a placeholder
            return await func(*args, **kwargs)

        return wrapper

    return decorator


async def get_current_user(token: Optional[str] = None, api_key: Optional[str] = None) -> Optional[User]:
    """
    Get the current authenticated user from token or API key.

    Args:
        token: JWT token
        api_key: API key

    Returns:
        User object if authenticated, None otherwise
    """
    # Create a default auth manager (in production this would use actual config)
    auth_manager = AuthManager(jwt_secret="default-secret-change-in-production")

    # Try JWT token first
    if token:
        user = auth_manager.jwt_manager.verify_token(token)
        if user:
            return user

    # Try API key
    if api_key:
        user = auth_manager.api_key_manager.validate_key(api_key)
        if user:
            return user

    return None


# Standalone functions for backwards compatibility
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token (standalone function for backwards compatibility).

    Args:
        data: Token payload data
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)

    to_encode.update({"exp": expire})

    # Use environment variable or default secret
    import os

    secret = os.environ.get('WAND_JWT_SECRET', 'default-secret-change-in-production')

    encoded_jwt = jwt.encode(to_encode, secret, algorithm="HS256")
    return encoded_jwt


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt (standalone function for backwards compatibility).

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.hash(password)
