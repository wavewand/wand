"""
Security Package

Provides authentication, authorization, rate limiting, and security utilities
for the multi-framework AI platform.
"""

from .auth import (
    APIKeyManager,
    AuthManager,
    JWTManager,
    Permission,
    User,
    UserRole,
    api_key_manager,
    auth_manager,
    jwt_manager,
    require_auth,
    require_permission,
    require_role,
)
from .rate_limiter import RateLimitConfig, RateLimiter, RateLimitExceeded, rate_limit, rate_limiter
from .validator import (
    InputValidator,
    SecurityValidator,
    ValidationRule,
    input_validator,
    sanitize_input,
    validate_input,
)

__all__ = [
    'AuthManager',
    'JWTManager',
    'APIKeyManager',
    'User',
    'UserRole',
    'Permission',
    'auth_manager',
    'jwt_manager',
    'api_key_manager',
    'require_auth',
    'require_permission',
    'require_role',
    'RateLimiter',
    'RateLimitConfig',
    'RateLimitExceeded',
    'rate_limiter',
    'rate_limit',
    'InputValidator',
    'SecurityValidator',
    'ValidationRule',
    'input_validator',
    'validate_input',
    'sanitize_input',
]
