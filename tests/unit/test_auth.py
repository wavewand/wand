"""
Unit Tests for Authentication and Authorization System

Tests JWT management, API keys, user management, and role-based access control.
"""

import secrets
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import jwt
import pytest

from security.auth import ROLE_PERMISSIONS, APIKey, APIKeyManager, AuthManager, JWTManager, Permission, User, UserRole
from utils.error_handling import AuthenticationError


class TestUser:
    """Test User model."""

    def test_user_creation(self):
        """Test user creation with role-based permissions."""
        user = User(user_id="test_123", username="testuser", email="test@example.com", role=UserRole.USER)

        assert user.user_id == "test_123"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.USER
        assert user.is_active is True
        assert isinstance(user.created_at, datetime)

        # Check role-based permissions
        expected_permissions = ROLE_PERMISSIONS[UserRole.USER]
        assert user.permissions == expected_permissions

    def test_user_permissions(self):
        """Test user permission checking."""
        user = User(user_id="admin_123", username="admin", email="admin@example.com", role=UserRole.ADMIN)

        # Admin should have all permissions
        assert user.has_permission(Permission.FRAMEWORK_READ)
        assert user.has_permission(Permission.SYSTEM_ADMIN)
        assert user.has_permission(Permission.USER_MANAGEMENT)

        # Test role checking
        assert user.has_role(UserRole.ADMIN)
        assert not user.has_role(UserRole.USER)

    def test_user_to_dict(self):
        """Test user serialization."""
        user = User(
            user_id="test_123",
            username="testuser",
            email="test@example.com",
            role=UserRole.USER,
            metadata={"source": "test"},
        )

        user_dict = user.to_dict()

        assert user_dict["user_id"] == "test_123"
        assert user_dict["username"] == "testuser"
        assert user_dict["role"] == "user"
        assert isinstance(user_dict["permissions"], list)
        assert user_dict["metadata"]["source"] == "test"


class TestAPIKey:
    """Test APIKey model."""

    def test_api_key_creation(self):
        """Test API key creation."""
        permissions = {Permission.FRAMEWORK_READ, Permission.DOCUMENT_READ}

        api_key = APIKey(
            key_id="key_123",
            key_hash="hash_value",
            name="Test Key",
            user_id="user_123",
            permissions=permissions,
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )

        assert api_key.key_id == "key_123"
        assert api_key.name == "Test Key"
        assert api_key.user_id == "user_123"
        assert api_key.permissions == permissions
        assert api_key.is_active is True
        assert not api_key.is_expired()

    def test_api_key_expiration(self):
        """Test API key expiration checking."""
        # Expired key
        expired_key = APIKey(
            key_id="expired_123",
            key_hash="hash",
            name="Expired Key",
            user_id="user_123",
            permissions=set(),
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )

        assert expired_key.is_expired()

        # Non-expiring key
        non_expiring_key = APIKey(
            key_id="permanent_123", key_hash="hash", name="Permanent Key", user_id="user_123", permissions=set()
        )

        assert not non_expiring_key.is_expired()

    def test_api_key_permissions(self):
        """Test API key permission checking."""
        permissions = {Permission.FRAMEWORK_READ, Permission.DOCUMENT_READ}

        api_key = APIKey(
            key_id="key_123", key_hash="hash", name="Test Key", user_id="user_123", permissions=permissions
        )

        assert api_key.has_permission(Permission.FRAMEWORK_READ)
        assert api_key.has_permission(Permission.DOCUMENT_READ)
        assert not api_key.has_permission(Permission.SYSTEM_ADMIN)


class TestJWTManager:
    """Test JWT token management."""

    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager for testing."""
        return JWTManager("test-secret-key")

    @pytest.fixture
    def test_user(self):
        """Create test user."""
        return User(user_id="test_123", username="testuser", email="test@example.com", role=UserRole.USER)

    def test_create_access_token(self, jwt_manager, test_user):
        """Test access token creation."""
        token = jwt_manager.create_access_token(test_user)

        assert isinstance(token, str)
        assert len(token) > 0

        # Decode and verify token
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])

        assert payload["user_id"] == "test_123"
        assert payload["username"] == "testuser"
        assert payload["role"] == "user"
        assert payload["type"] == "access"
        assert "permissions" in payload
        assert "exp" in payload
        assert "iat" in payload

    def test_create_refresh_token(self, jwt_manager, test_user):
        """Test refresh token creation."""
        token = jwt_manager.create_refresh_token(test_user)

        assert isinstance(token, str)
        assert len(token) > 0

        # Decode and verify token
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])

        assert payload["user_id"] == "test_123"
        assert payload["type"] == "refresh"
        assert "exp" in payload
        assert "iat" in payload
        # Refresh tokens should not contain sensitive data
        assert "permissions" not in payload

    def test_verify_token(self, jwt_manager, test_user):
        """Test token verification."""
        token = jwt_manager.create_access_token(test_user)
        payload = jwt_manager.verify_token(token)

        assert payload["user_id"] == "test_123"
        assert payload["username"] == "testuser"
        assert payload["type"] == "access"

    def test_verify_invalid_token(self, jwt_manager):
        """Test verification of invalid token."""
        with pytest.raises(AuthenticationError) as exc_info:
            jwt_manager.verify_token("invalid.token.here")

        assert "Invalid token" in str(exc_info.value)

    def test_verify_expired_token(self, jwt_manager, test_user):
        """Test verification of expired token."""
        # Create JWT manager with very short expiration
        jwt_manager.access_token_expire = timedelta(microseconds=1)
        token = jwt_manager.create_access_token(test_user)

        # Wait for token to expire
        import time

        time.sleep(0.001)

        with pytest.raises(AuthenticationError) as exc_info:
            jwt_manager.verify_token(token)

        assert "Token has expired" in str(exc_info.value)

    def test_refresh_access_token(self, jwt_manager, test_user):
        """Test access token refresh."""
        refresh_token = jwt_manager.create_refresh_token(test_user)
        new_access_token = jwt_manager.refresh_access_token(refresh_token, test_user)

        assert isinstance(new_access_token, str)

        # Verify new token
        payload = jwt_manager.verify_token(new_access_token)
        assert payload["user_id"] == "test_123"
        assert payload["type"] == "access"

    def test_refresh_with_invalid_token(self, jwt_manager, test_user):
        """Test refresh with invalid refresh token."""
        access_token = jwt_manager.create_access_token(test_user)

        with pytest.raises(AuthenticationError) as exc_info:
            jwt_manager.refresh_access_token(access_token, test_user)

        assert "Invalid refresh token" in str(exc_info.value)


class TestAPIKeyManager:
    """Test API key management."""

    @pytest.fixture
    def api_key_manager(self):
        """Create API key manager for testing."""
        return APIKeyManager()

    def test_generate_api_key(self, api_key_manager):
        """Test API key generation."""
        permissions = {Permission.FRAMEWORK_READ, Permission.DOCUMENT_READ}

        raw_key, api_key = api_key_manager.generate_api_key(
            name="Test Key", user_id="user_123", permissions=permissions, expires_in_days=30, rate_limit=100
        )

        # Check raw key format
        assert raw_key.startswith("mcp_")
        assert len(raw_key) > 20

        # Check API key object
        assert api_key.name == "Test Key"
        assert api_key.user_id == "user_123"
        assert api_key.permissions == permissions
        assert api_key.rate_limit == 100
        assert api_key.is_active is True
        assert not api_key.is_expired()
        assert api_key.usage_count == 0

    def test_verify_api_key(self, api_key_manager):
        """Test API key verification."""
        permissions = {Permission.FRAMEWORK_READ}

        raw_key, _ = api_key_manager.generate_api_key(name="Test Key", user_id="user_123", permissions=permissions)

        # Verify the key
        verified_key = api_key_manager.verify_api_key(raw_key)

        assert verified_key is not None
        assert verified_key.name == "Test Key"
        assert verified_key.user_id == "user_123"
        assert verified_key.usage_count == 1  # Should increment
        assert verified_key.last_used is not None

    def test_verify_invalid_api_key(self, api_key_manager):
        """Test verification of invalid API key."""
        # Test with invalid format
        assert api_key_manager.verify_api_key("invalid_key") is None

        # Test with valid format but non-existent key
        fake_key = "mcp_" + secrets.token_urlsafe(32)
        assert api_key_manager.verify_api_key(fake_key) is None

    def test_verify_inactive_api_key(self, api_key_manager):
        """Test verification of inactive API key."""
        permissions = {Permission.FRAMEWORK_READ}

        raw_key, api_key = api_key_manager.generate_api_key(
            name="Test Key", user_id="user_123", permissions=permissions
        )

        # Deactivate the key
        api_key.is_active = False

        # Should not verify inactive key
        assert api_key_manager.verify_api_key(raw_key) is None

    def test_verify_expired_api_key(self, api_key_manager):
        """Test verification of expired API key."""
        permissions = {Permission.FRAMEWORK_READ}

        raw_key, api_key = api_key_manager.generate_api_key(
            name="Test Key", user_id="user_123", permissions=permissions, expires_in_days=1
        )

        # Force expiration
        api_key.expires_at = datetime.now(timezone.utc) - timedelta(days=1)

        # Should not verify expired key
        assert api_key_manager.verify_api_key(raw_key) is None

    def test_revoke_api_key(self, api_key_manager):
        """Test API key revocation."""
        permissions = {Permission.FRAMEWORK_READ}

        raw_key, api_key = api_key_manager.generate_api_key(
            name="Test Key", user_id="user_123", permissions=permissions
        )

        # Revoke the key
        result = api_key_manager.revoke_api_key(api_key.key_id)
        assert result is True
        assert api_key.is_active is False

        # Should not verify revoked key
        assert api_key_manager.verify_api_key(raw_key) is None

    def test_list_user_keys(self, api_key_manager):
        """Test listing user's API keys."""
        permissions = {Permission.FRAMEWORK_READ}

        # Create multiple keys for same user
        for i in range(3):
            api_key_manager.generate_api_key(name=f"Key {i}", user_id="user_123", permissions=permissions)

        # Create key for different user
        api_key_manager.generate_api_key(name="Other Key", user_id="user_456", permissions=permissions)

        # List keys for user_123
        user_keys = api_key_manager.list_user_keys("user_123")
        assert len(user_keys) == 3

        for key in user_keys:
            assert key.user_id == "user_123"


class TestAuthManager:
    """Test authentication manager."""

    @pytest.fixture
    def auth_manager(self):
        """Create auth manager for testing."""
        return AuthManager("test-jwt-secret")

    def test_auth_manager_initialization(self, auth_manager):
        """Test auth manager initialization."""
        assert auth_manager.jwt_manager is not None
        assert auth_manager.api_key_manager is not None
        assert len(auth_manager.users) == 1  # Default admin user

        # Check default admin user
        admin_user = list(auth_manager.users.values())[0]
        assert admin_user.username == "admin"
        assert admin_user.role == UserRole.ADMIN

    def test_create_user(self, auth_manager):
        """Test user creation."""
        user = auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            role=UserRole.USER,
            metadata={"source": "test"},
        )

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.USER
        assert user.metadata["source"] == "test"
        assert user.user_id in auth_manager.users

    def test_authenticate_admin_user(self, auth_manager):
        """Test admin user authentication."""
        user = auth_manager.authenticate_user("admin", "any_password")

        assert user is not None
        assert user.username == "admin"
        assert user.role == UserRole.ADMIN
        assert user.last_login is not None

    def test_authenticate_invalid_user(self, auth_manager):
        """Test authentication with invalid user."""
        user = auth_manager.authenticate_user("nonexistent", "password")
        assert user is None

    def test_authenticate_token(self, auth_manager):
        """Test token-based authentication."""
        # Create a user first
        test_user = auth_manager.create_user(username="testuser", email="test@example.com", password="password123")

        # Create token
        token = auth_manager.jwt_manager.create_access_token(test_user)

        # Authenticate with token
        authenticated_user = auth_manager.authenticate_token(token)

        assert authenticated_user is not None
        assert authenticated_user.user_id == test_user.user_id
        assert authenticated_user.username == "testuser"

    def test_authenticate_invalid_token(self, auth_manager):
        """Test authentication with invalid token."""
        user = auth_manager.authenticate_token("invalid.token.here")
        assert user is None

    def test_authenticate_api_key(self, auth_manager):
        """Test API key authentication."""
        # Create a user first
        test_user = auth_manager.create_user(username="testuser", email="test@example.com", password="password123")

        # Generate API key
        permissions = {Permission.FRAMEWORK_READ}
        raw_key, _ = auth_manager.api_key_manager.generate_api_key(
            name="Test Key", user_id=test_user.user_id, permissions=permissions
        )

        # Authenticate with API key
        result = auth_manager.authenticate_api_key(raw_key)

        assert result is not None
        user, api_key = result
        assert user.user_id == test_user.user_id
        assert api_key.name == "Test Key"

    def test_authenticate_invalid_api_key(self, auth_manager):
        """Test authentication with invalid API key."""
        result = auth_manager.authenticate_api_key("invalid_key")
        assert result is None

    def test_get_user(self, auth_manager):
        """Test user retrieval."""
        test_user = auth_manager.create_user(username="testuser", email="test@example.com", password="password123")

        retrieved_user = auth_manager.get_user(test_user.user_id)
        assert retrieved_user is not None
        assert retrieved_user.user_id == test_user.user_id

        # Test non-existent user
        assert auth_manager.get_user("nonexistent") is None

    def test_update_user(self, auth_manager):
        """Test user updates."""
        test_user = auth_manager.create_user(username="testuser", email="test@example.com", password="password123")

        # Update user
        result = auth_manager.update_user(test_user.user_id, email="newemail@example.com", role=UserRole.ADMIN)

        assert result is True

        updated_user = auth_manager.get_user(test_user.user_id)
        assert updated_user.email == "newemail@example.com"
        assert updated_user.role == UserRole.ADMIN

    def test_delete_user(self, auth_manager):
        """Test user deletion (deactivation)."""
        test_user = auth_manager.create_user(username="testuser", email="test@example.com", password="password123")

        # Delete user
        result = auth_manager.delete_user(test_user.user_id)
        assert result is True

        # User should be deactivated, not removed
        deactivated_user = auth_manager.get_user(test_user.user_id)
        assert deactivated_user is not None
        assert deactivated_user.is_active is False
