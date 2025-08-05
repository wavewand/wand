"""
SaaS-specific API endpoints for Wand Backend

Provides endpoints that integrate with the Go microservices for
user context, subscription management, and enterprise features.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from integrations import get_integration_manager

from .saas_integration import SaaSBridge, UserContext, get_saas_bridge, with_saas_context

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# Create SaaS router
saas_router = APIRouter(prefix="/api/v1/saas", tags=["saas"])


async def get_user_context_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    saas_bridge: SaaSBridge = Depends(get_saas_bridge),
) -> Optional[UserContext]:
    """Extract user context from JWT token or headers"""

    # If SaaS is not enabled, return None (self-hosted mode)
    if not saas_bridge.enabled:
        return None

    # Use X-User-ID header if provided (from Gateway service)
    if x_user_id:
        return await saas_bridge.get_user_context(x_user_id)

    # If we have a token, validate it with auth service
    if credentials:
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{saas_bridge.auth_service_url}/api/v1/auth/token/validate",
                    headers={"Authorization": f"Bearer {credentials.credentials}"},
                )

                if response.status_code == 200:
                    user_data = response.json()
                    if user_data.get("valid"):
                        user_id = user_data["user"]["id"]
                        return await saas_bridge.get_user_context(user_id)
        except Exception as e:
            logger.warning(f"Token validation failed: {e}")

    return None


@saas_router.get("/user/context")
async def get_user_context_endpoint(user_context: UserContext = Depends(get_user_context_from_token)):
    """Get current user's subscription context and limits"""
    if not user_context:
        return {
            "subscription_tier": "free",
            "saas_enabled": False,
            "limits": {"api_calls": "unlimited", "storage": "unlimited"},
        }

    return {
        "user_id": user_context.user_id,
        "subscription_tier": user_context.subscription_tier.value,
        "saas_enabled": True,
        "limits": {
            "api_calls": {
                "used": user_context.monthly_api_calls_used,
                "limit": user_context.monthly_api_calls_limit,
                "remaining": user_context.api_calls_remaining,
            },
            "storage": {
                "used": user_context.monthly_storage_used,
                "limit": user_context.monthly_storage_limit,
                "remaining": user_context.storage_remaining,
            },
        },
        "permissions": user_context.permissions,
        "is_enterprise": user_context.is_enterprise,
    }


@saas_router.get("/integrations/available")
async def get_available_integrations(user_context: UserContext = Depends(get_user_context_from_token)):
    """Get list of available integrations for the user's subscription tier"""
    integration_manager = get_integration_manager()
    all_integrations = integration_manager.get_all_integrations()

    available_integrations = []

    for integration_name, integration in all_integrations.items():
        # Get health status
        try:
            health_data = await integration.health_check()
        except Exception as e:
            health_data = {"status": "unhealthy", "error": str(e)}

        integration_info = {
            "name": integration_name,
            "enabled": integration.enabled,
            "status": health_data.get("status", "unknown"),
            "healthy": health_data.get("status") == "healthy",
            "metrics": health_data.get("metrics", {}),
            "tier_available": True,  # All tiers have access to all integrations
            "description": getattr(integration, 'description', f"{integration_name} integration"),
            "category": getattr(integration, 'category', 'general'),
        }

        # Add subscription-specific metadata
        if user_context:
            integration_info.update(
                {
                    "user_has_access": True,  # All users have access currently
                    "usage_counted": user_context.subscription_tier != "free",
                }
            )

        available_integrations.append(integration_info)

    return {
        "integrations": available_integrations,
        "total_count": len(available_integrations),
        "user_tier": user_context.subscription_tier.value if user_context else "free",
    }


@saas_router.post("/integrations/{integration_name}/execute")
async def execute_integration_with_saas(
    integration_name: str,
    request: Request,
    operation_data: Dict[str, Any],
    user_context: UserContext = Depends(get_user_context_from_token),
    saas_bridge: SaaSBridge = Depends(get_saas_bridge),
):
    """Execute integration with SaaS context and rate limiting"""

    # Get integration
    integration_manager = get_integration_manager()
    integration = integration_manager.get_integration(integration_name)

    if not integration:
        raise HTTPException(status_code=404, detail=f"Integration '{integration_name}' not found")

    # Check if user has access (currently all users have access to all integrations)
    if user_context and not await saas_bridge.validate_integration_access(user_context, integration_name):
        raise HTTPException(
            status_code=403, detail=f"Integration '{integration_name}' not available in your subscription tier"
        )

    # Check rate limits
    if user_context and not await saas_bridge.check_rate_limits(user_context):
        raise HTTPException(status_code=429, detail="API rate limit exceeded", headers={"Retry-After": "3600"})

    # Extract operation and parameters
    operation = operation_data.get("operation", "execute")
    parameters = operation_data.get("parameters", {})

    # Execute with SaaS context if the integration supports it
    try:
        if hasattr(integration, 'execute_with_saas_context'):
            # Use SaaS-aware execution
            result = await integration.execute_with_saas_context(
                operation=operation,
                user_id=user_context.user_id if user_context else None,
                ip_address=request.client.host,
                **parameters,
            )
        else:
            # Fallback to standard execution
            result = await integration.execute_operation(operation, **parameters)

            # Record usage manually
            if user_context:
                await saas_bridge.record_usage(
                    user_id=user_context.user_id, operation="api_call", integration_name=integration_name
                )

        return result

    except Exception as e:
        logger.error(f"Integration execution failed: {e}")

        # Log audit event for enterprise users
        if user_context and user_context.is_enterprise:
            await saas_bridge.log_audit_event(
                user_id=user_context.user_id,
                action=f"integration_execute_failed",
                resource=f"integration:{integration_name}",
                details={"operation": operation, "error": str(e)},
                success=False,
                error_message=str(e),
                ip_address=request.client.host,
            )

        raise HTTPException(status_code=500, detail=f"Integration execution failed: {str(e)}")


@saas_router.get("/integrations/{integration_name}/health")
async def get_integration_health_with_saas(
    integration_name: str, user_context: UserContext = Depends(get_user_context_from_token)
):
    """Get integration health with SaaS-enhanced metadata"""

    integration_manager = get_integration_manager()
    integration = integration_manager.get_integration(integration_name)

    if not integration:
        raise HTTPException(status_code=404, detail=f"Integration '{integration_name}' not found")

    try:
        # Use SaaS-enhanced health check if available
        if hasattr(integration, 'get_saas_enhanced_health_check'):
            health_data = await integration.get_saas_enhanced_health_check()
        else:
            health_data = await integration.health_check()

        # Add user-specific context
        if user_context:
            health_data["user_access"] = {
                "has_access": True,  # All users currently have access
                "subscription_tier": user_context.subscription_tier.value,
                "usage_counted": user_context.subscription_tier != "free",
            }

        return health_data

    except Exception as e:
        logger.error(f"Health check failed for {integration_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@saas_router.get("/integrations/health")
async def get_all_integrations_health_with_saas(user_context: UserContext = Depends(get_user_context_from_token)):
    """Get health status of all integrations with SaaS metadata"""

    integration_manager = get_integration_manager()
    all_integrations = integration_manager.get_all_integrations()

    health_summary = {
        "overall_status": "healthy",
        "integrations": {},
        "stats": {
            "total_integrations": len(all_integrations),
            "healthy_integrations": 0,
            "unhealthy_integrations": 0,
            "disabled_integrations": 0,
        },
        "user_context": {
            "subscription_tier": user_context.subscription_tier.value if user_context else "free",
            "saas_enabled": user_context is not None,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

    for integration_name, integration in all_integrations.items():
        try:
            if hasattr(integration, 'get_saas_enhanced_health_check'):
                health_data = await integration.get_saas_enhanced_health_check()
            else:
                health_data = await integration.health_check()

            status = health_data.get("status", "unknown")
            if status == "healthy":
                health_summary["stats"]["healthy_integrations"] += 1
            elif status == "disabled":
                health_summary["stats"]["disabled_integrations"] += 1
            else:
                health_summary["stats"]["unhealthy_integrations"] += 1
                health_summary["overall_status"] = "degraded"

            health_summary["integrations"][integration_name] = health_data

        except Exception as e:
            logger.error(f"Health check failed for {integration_name}: {e}")
            health_summary["integrations"][integration_name] = {"status": "unhealthy", "error": str(e)}
            health_summary["stats"]["unhealthy_integrations"] += 1
            health_summary["overall_status"] = "degraded"

    return health_summary


@saas_router.post("/audit/integration-usage")
async def log_integration_usage_audit(
    usage_data: Dict[str, Any],
    request: Request,
    user_context: UserContext = Depends(get_user_context_from_token),
    saas_bridge: SaaSBridge = Depends(get_saas_bridge),
):
    """Log integration usage for audit purposes (Enterprise only)"""

    if not user_context or not user_context.is_enterprise:
        raise HTTPException(status_code=403, detail="Enterprise subscription required")

    # Log comprehensive audit event
    await saas_bridge.log_audit_event(
        user_id=user_context.user_id,
        action="integration_usage_audit",
        resource="integration_usage",
        details=usage_data,
        success=True,
        ip_address=request.client.host,
    )

    return {"message": "Usage audit logged successfully"}


@saas_router.get("/subscription/limits/{user_id}")
async def get_subscription_limits_for_user(
    user_id: str,
    requesting_user: UserContext = Depends(get_user_context_from_token),
    saas_bridge: SaaSBridge = Depends(get_saas_bridge),
):
    """Get subscription limits for a specific user (internal service endpoint)"""

    # This endpoint is typically called by Go services with service token
    # For now, allow any authenticated user to check their own limits
    if requesting_user and requesting_user.user_id != user_id:
        # Only allow if user is admin or it's a service call
        # For now, restrict to own user only
        raise HTTPException(status_code=403, detail="Access denied")

    user_context = await saas_bridge.get_user_context(user_id)
    if not user_context:
        raise HTTPException(status_code=404, detail="User not found or not in SaaS system")

    return {
        "user_id": user_id,
        "plan_id": user_context.subscription_tier.value,
        "monthly_api_calls_limit": user_context.monthly_api_calls_limit,
        "monthly_api_calls_used": user_context.monthly_api_calls_used,
        "monthly_storage_limit": user_context.monthly_storage_limit,
        "monthly_storage_used": user_context.monthly_storage_used,
        "features": user_context.permissions,
        "integrations_allowed": ["*"],  # All integrations allowed for all tiers
        "current_period_start": datetime.utcnow().replace(day=1).isoformat(),
        "current_period_end": datetime.utcnow().replace(day=28).isoformat(),  # Simplified
        "status": "active",
    }


@saas_router.post("/monitoring/health/{integration_name}")
async def receive_health_report(
    integration_name: str,
    health_data: Dict[str, Any],
    x_service_name: Optional[str] = Header(None, alias="X-Service-Name"),
):
    """Receive health reports from integrations (internal endpoint)"""

    # This endpoint receives health reports from integrations
    # In a full implementation, this would forward to monitoring systems
    logger.info(f"Health report received for {integration_name} from {x_service_name}: {health_data}")

    return {"message": "Health report received"}


# Add the SaaS router to the main application


def include_saas_endpoints(app):
    """Include SaaS endpoints in the FastAPI application"""
    app.include_router(saas_router)
