"""
SaaS Integration Layer for Wand Backend

Provides communication bridge between Python Wand backend and Go microservices.
Handles subscription limits, user authentication, and cross-service coordination.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class SubscriptionTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class UserContext:
    """User context for SaaS operations"""

    user_id: str
    subscription_tier: SubscriptionTier
    monthly_api_calls_used: int
    monthly_api_calls_limit: int
    monthly_storage_used: int
    monthly_storage_limit: int
    permissions: List[str]
    is_enterprise: bool = False

    @property
    def api_calls_remaining(self) -> int:
        if self.monthly_api_calls_limit == 0:  # Unlimited for free self-hosted
            return float('inf')
        return max(0, self.monthly_api_calls_limit - self.monthly_api_calls_used)

    @property
    def storage_remaining(self) -> int:
        if self.monthly_storage_limit == 0:  # Unlimited for free self-hosted
            return float('inf')
        return max(0, self.monthly_storage_limit - self.monthly_storage_used)


class SaaSBridge:
    """
    Bridge service for communication with Go SaaS microservices.
    Handles user context, subscription limits, and audit logging.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = self.config.get("saas_enabled", os.getenv("SAAS_ENABLED", "false").lower() == "true")
        self.service_token = self.config.get("service_token", os.getenv("WAND_SERVICE_TOKEN"))

        # SaaS service URLs
        self.billing_service_url = self.config.get(
            "billing_service_url", os.getenv("BILLING_SERVICE_URL", "http://localhost:8081")
        )
        self.enterprise_service_url = self.config.get(
            "enterprise_service_url", os.getenv("ENTERPRISE_SERVICE_URL", "http://localhost:8083")
        )
        self.auth_service_url = self.config.get(
            "auth_service_url", os.getenv("AUTH_SERVICE_URL", "http://localhost:8080")
        )

        # HTTP client for service communication
        self.client = (
            httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                headers={
                    "Authorization": f"Bearer {self.service_token}",
                    "X-Service-Name": "wand-backend",
                    "User-Agent": "wand-backend/1.0.0",
                },
            )
            if self.enabled
            else None
        )

        logger.info(f"SaaS Bridge initialized - Enabled: {self.enabled}")

    async def get_user_context(self, user_id: str) -> Optional[UserContext]:
        """Get user context including subscription and limits"""
        if not self.enabled or not user_id:
            # For self-hosted/free tier, return unlimited context
            return UserContext(
                user_id=user_id or "anonymous",
                subscription_tier=SubscriptionTier.FREE,
                monthly_api_calls_used=0,
                monthly_api_calls_limit=0,  # Unlimited
                monthly_storage_used=0,
                monthly_storage_limit=0,  # Unlimited
                permissions=["integrations:read", "integrations:write", "api:use"],
            )

        try:
            # Get subscription limits from billing service
            response = await self.client.get(f"{self.billing_service_url}/api/v1/users/{user_id}/limits")

            if response.status_code == 200:
                data = response.json()
                return UserContext(
                    user_id=user_id,
                    subscription_tier=SubscriptionTier(data.get("plan_id", "free")),
                    monthly_api_calls_used=data.get("monthly_api_calls_used", 0),
                    monthly_api_calls_limit=data.get("monthly_api_calls_limit", 0),
                    monthly_storage_used=data.get("monthly_storage_used", 0),
                    monthly_storage_limit=data.get("monthly_storage_limit", 0),
                    permissions=data.get("features", []),
                    is_enterprise=data.get("plan_id") == "enterprise",
                )
            else:
                logger.warning(f"Failed to get user context for {user_id}: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return None

    async def check_rate_limits(self, user_context: UserContext, operation: str = "api_call") -> bool:
        """Check if user is within rate limits"""
        if not self.enabled or user_context.subscription_tier == SubscriptionTier.FREE:
            return True  # No limits for self-hosted free tier

        # Check API call limits
        if user_context.api_calls_remaining <= 0:
            logger.warning(f"User {user_context.user_id} exceeded API call limit")
            return False

        return True

    async def record_usage(
        self, user_id: str, operation: str = "api_call", storage_used: int = 0, integration_name: str = None
    ):
        """Record usage for billing and analytics"""
        if not self.enabled or not user_id:
            return

        try:
            usage_data = {
                "api_calls_count": 1 if operation == "api_call" else 0,
                "storage_used": storage_used,
                "integration_usage": {integration_name: 1} if integration_name else {},
            }

            response = await self.client.post(
                f"{self.billing_service_url}/api/v1/users/{user_id}/usage", json=usage_data
            )

            if response.status_code not in [200, 201]:
                logger.warning(f"Failed to record usage for {user_id}: {response.status_code}")

        except Exception as e:
            logger.error(f"Error recording usage: {e}")

    async def log_audit_event(
        self,
        user_id: str,
        action: str,
        resource: str,
        details: Dict[str, Any] = None,
        success: bool = True,
        error_message: str = None,
        ip_address: str = None,
    ):
        """Log audit event for enterprise users"""
        if not self.enabled:
            return

        try:
            audit_data = {
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "details": details or {},
                "success": success,
                "error_message": error_message,
                "ip_address": ip_address,
            }

            response = await self.client.post(f"{self.enterprise_service_url}/api/v1/audit/log", json=audit_data)

            if response.status_code not in [200, 201]:
                logger.warning(f"Failed to log audit event: {response.status_code}")

        except Exception as e:
            logger.error(f"Error logging audit event: {e}")

    async def validate_integration_access(self, user_context: UserContext, integration_name: str) -> bool:
        """Check if user has access to a specific integration"""
        if not self.enabled:
            return True  # All integrations available for self-hosted

        # For SaaS, all tiers have access to all integrations
        # Future: Could implement tier-based integration restrictions
        return True

    async def get_integration_config_for_user(self, user_id: str, integration_name: str) -> Dict[str, Any]:
        """Get integration configuration with user-specific overrides"""
        if not self.enabled:
            return {}

        try:
            # Get user-specific integration settings (future feature)
            # For now, return empty config to use defaults
            return {}

        except Exception as e:
            logger.error(f"Error getting integration config: {e}")
            return {}

    async def report_integration_health(self, integration_name: str, health_data: Dict[str, Any]):
        """Report integration health to monitoring service"""
        if not self.enabled:
            return

        try:
            # Add timestamp and additional metadata
            enriched_health_data = {
                **health_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "wand-backend",
                "integration": integration_name,
            }

            # This would be sent to a monitoring service
            # For now, just log it
            logger.debug(f"Health report for {integration_name}: {enriched_health_data}")

        except Exception as e:
            logger.error(f"Error reporting health: {e}")

    async def close(self):
        """Clean up resources"""
        if self.client:
            await self.client.aclose()


# Enhanced integration base class with SaaS support


class SaaSAwareIntegration:
    """
    Mixin class to add SaaS awareness to integrations.
    Provides user context, rate limiting, and audit logging.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saas_bridge = SaaSBridge(self.config.get("saas", {}))

    async def execute_with_saas_context(
        self, operation: str, user_id: str = None, ip_address: str = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute integration operation with SaaS context and controls.
        This wraps the standard execute_operation with SaaS-specific logic.
        """

        # Get user context
        user_context = await self.saas_bridge.get_user_context(user_id) if user_id else None

        # Check rate limits
        if user_context and not await self.saas_bridge.check_rate_limits(user_context, operation):
            return {
                "success": False,
                "error": "Rate limit exceeded",
                "error_code": "RATE_LIMIT_EXCEEDED",
                "integration": self.name,
                "user_limits": {
                    "api_calls_remaining": user_context.api_calls_remaining,
                    "subscription_tier": user_context.subscription_tier.value,
                },
            }

        # Check integration access
        if user_context and not await self.saas_bridge.validate_integration_access(user_context, self.name):
            return {
                "success": False,
                "error": "Integration not available in your subscription tier",
                "error_code": "INTEGRATION_ACCESS_DENIED",
                "integration": self.name,
                "subscription_tier": user_context.subscription_tier.value,
            }

        # Log audit event (start)
        if user_context and user_context.is_enterprise:
            await self.saas_bridge.log_audit_event(
                user_id=user_id,
                action=f"integration_{operation}",
                resource=f"integration:{self.name}",
                details={"operation": operation, "parameters": kwargs},
                ip_address=ip_address,
            )

        # Execute the actual operation
        start_time = time.time()
        try:
            # Call the original execute_operation method
            result = await self.execute_operation(operation, **kwargs)

            # Record successful usage
            if user_id:
                await self.saas_bridge.record_usage(user_id=user_id, operation="api_call", integration_name=self.name)

            # Log successful audit event
            if user_context and user_context.is_enterprise:
                await self.saas_bridge.log_audit_event(
                    user_id=user_id,
                    action=f"integration_{operation}",
                    resource=f"integration:{self.name}",
                    details={"operation": operation, "response_time": time.time() - start_time},
                    success=True,
                    ip_address=ip_address,
                )

            # Add SaaS metadata to response
            if isinstance(result, dict):
                result.update(
                    {
                        "saas_metadata": {
                            "user_id": user_id,
                            "subscription_tier": user_context.subscription_tier.value if user_context else "free",
                            "api_calls_remaining": user_context.api_calls_remaining if user_context else float('inf'),
                            "rate_limited": False,
                        }
                    }
                )

            return result

        except Exception as e:
            # Log failed audit event
            if user_context and user_context.is_enterprise:
                await self.saas_bridge.log_audit_event(
                    user_id=user_id,
                    action=f"integration_{operation}",
                    resource=f"integration:{self.name}",
                    details={"operation": operation, "error": str(e)},
                    success=False,
                    error_message=str(e),
                    ip_address=ip_address,
                )

            # Re-raise the exception to maintain original behavior
            raise

    async def get_saas_enhanced_health_check(self) -> Dict[str, Any]:
        """Enhanced health check with SaaS metadata"""
        # Get standard health check
        health_data = await self.health_check()

        # Add SaaS-specific metadata
        saas_metadata = {
            "saas_enabled": self.saas_bridge.enabled,
            "supports_user_context": True,
            "supports_audit_logging": True,
            "supports_rate_limiting": True,
        }

        enhanced_health = {**health_data, "saas_metadata": saas_metadata}

        # Report to monitoring
        await self.saas_bridge.report_integration_health(self.name, enhanced_health)

        return enhanced_health

    async def cleanup(self):
        """Cleanup SaaS resources"""
        if hasattr(super(), 'cleanup'):
            await super().cleanup()
        if hasattr(self, 'saas_bridge'):
            await self.saas_bridge.close()


# Global SaaS bridge instance
_saas_bridge = None


async def get_saas_bridge() -> SaaSBridge:
    """Get global SaaS bridge instance"""
    global _saas_bridge
    if _saas_bridge is None:
        _saas_bridge = SaaSBridge()
    return _saas_bridge


async def initialize_saas_integration():
    """Initialize SaaS integration on startup"""
    bridge = await get_saas_bridge()
    logger.info("SaaS integration initialized")
    return bridge


# Decorator for adding SaaS context to API endpoints


def with_saas_context(f):
    """Decorator to add SaaS context to API endpoints"""

    async def wrapper(*args, **kwargs):
        # Extract user_id and IP from request context
        user_id = kwargs.get('user_id') or getattr(args[0] if args else None, 'user_id', None)
        ip_address = kwargs.get('ip_address') or getattr(args[0] if args else None, 'ip_address', None)

        # Get SaaS bridge
        bridge = await get_saas_bridge()

        # Add context to kwargs
        kwargs['saas_bridge'] = bridge
        kwargs['user_context'] = await bridge.get_user_context(user_id) if user_id else None

        return await f(*args, **kwargs)

    return wrapper
