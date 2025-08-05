"""
Health Monitoring API Endpoints for Wand Integration Health System
Provides REST endpoints for accessing integration health data and controlling monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ..integrations_config import integrations
from ..monitoring.integration_health_monitor import HealthMetrics, HealthStatus, get_health_monitor

logger = logging.getLogger(__name__)

# Create health monitoring router
health_router = APIRouter(prefix="/api/v1", tags=["health"])


@health_router.get("/health/summary")
async def get_health_summary():
    """
    Get system-wide health summary with statistics for all integrations
    """
    try:
        monitor = await get_health_monitor()
        summary = await monitor.get_health_summary()

        return {"status": "success", "data": summary, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Failed to get health summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get health summary: {str(e)}")


@health_router.get("/integrations/health")
async def get_all_integrations_health():
    """
    Get health metrics for all registered integrations
    """
    try:
        monitor = await get_health_monitor()
        all_health = await monitor.get_all_health_metrics()

        # Convert HealthMetrics objects to dictionaries
        health_data = {
            integration_name: health_metrics.to_dict() for integration_name, health_metrics in all_health.items()
        }

        return {
            "status": "success",
            "data": health_data,
            "total_integrations": len(health_data),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get all integrations health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get integrations health: {str(e)}")


@health_router.get("/integrations/{integration_name}/health")
async def get_integration_health(integration_name: str):
    """
    Get detailed health metrics for a specific integration
    """
    try:
        monitor = await get_health_monitor()
        health_metrics = await monitor.get_integration_health(integration_name)

        if not health_metrics:
            raise HTTPException(status_code=404, detail=f"Integration '{integration_name}' not found or not monitored")

        return {
            "status": "success",
            "integration": integration_name,
            "health": health_metrics.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get health for {integration_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get integration health: {str(e)}")


@health_router.post("/integrations/{integration_name}/health/refresh")
async def refresh_integration_health(integration_name: str, background_tasks: BackgroundTasks):
    """
    Force refresh health check for a specific integration
    """
    try:
        monitor = await get_health_monitor()

        # Check if integration exists in monitoring
        health_metrics = await monitor.get_integration_health(integration_name)
        if not health_metrics:
            raise HTTPException(status_code=404, detail=f"Integration '{integration_name}' not found or not monitored")

        # Schedule immediate health check (background task)
        background_tasks.add_task(force_health_check, integration_name)

        return {
            "status": "success",
            "message": f"Health refresh initiated for {integration_name}",
            "integration": integration_name,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to refresh health for {integration_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh integration health: {str(e)}")


@health_router.post("/integrations/health/refresh")
async def refresh_all_integrations_health(background_tasks: BackgroundTasks):
    """
    Force refresh health checks for all integrations
    """
    try:
        monitor = await get_health_monitor()
        all_health = await monitor.get_all_health_metrics()

        integration_names = list(all_health.keys())

        # Schedule health checks for all integrations (background tasks)
        for integration_name in integration_names:
            background_tasks.add_task(force_health_check, integration_name)

        return {
            "status": "success",
            "message": f"Health refresh initiated for {len(integration_names)} integrations",
            "integrations": integration_names,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to refresh all integrations health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh all integrations health: {str(e)}")


@health_router.get("/integrations/{integration_name}/health/history")
async def get_integration_health_history(integration_name: str, hours: int = 24, limit: Optional[int] = None):
    """
    Get historical health data for a specific integration
    """
    try:
        monitor = await get_health_monitor()

        # Check if integration exists
        if integration_name not in monitor.health_history:
            raise HTTPException(
                status_code=404, detail=f"Integration '{integration_name}' not found or no history available"
            )

        history = monitor.health_history[integration_name]

        # Filter by time range
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        filtered_history = [h for h in history if h.last_check > cutoff_time]

        # Apply limit if specified
        if limit:
            filtered_history = filtered_history[-limit:]

        # Convert to dictionaries
        history_data = [h.to_dict() for h in filtered_history]

        return {
            "status": "success",
            "integration": integration_name,
            "history": history_data,
            "time_range_hours": hours,
            "total_records": len(history_data),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get health history for {integration_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get health history: {str(e)}")


@health_router.get("/health/status")
async def get_health_monitoring_status():
    """
    Get status of the health monitoring system itself
    """
    try:
        monitor = await get_health_monitor()

        return {
            "status": "success",
            "monitoring_active": monitor.running,
            "registered_integrations": len(monitor.health_cache),
            "active_tasks": len(monitor.monitoring_tasks),
            "health_weights": monitor.health_weights,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get health monitoring status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring status: {str(e)}")


@health_router.post("/health/monitoring/start")
async def start_health_monitoring():
    """
    Start the health monitoring system
    """
    try:
        monitor = await get_health_monitor()

        if monitor.running:
            return {
                "status": "info",
                "message": "Health monitoring is already running",
                "timestamp": datetime.utcnow().isoformat(),
            }

        await monitor.start_monitoring()

        return {
            "status": "success",
            "message": "Health monitoring started successfully",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to start health monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start health monitoring: {str(e)}")


@health_router.post("/health/monitoring/stop")
async def stop_health_monitoring():
    """
    Stop the health monitoring system
    """
    try:
        monitor = await get_health_monitor()

        if not monitor.running:
            return {
                "status": "info",
                "message": "Health monitoring is not running",
                "timestamp": datetime.utcnow().isoformat(),
            }

        await monitor.stop_monitoring()

        return {
            "status": "success",
            "message": "Health monitoring stopped successfully",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to stop health monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop health monitoring: {str(e)}")


@health_router.get("/health/metrics/aggregated")
async def get_aggregated_health_metrics(time_range: str = "24h", group_by: str = "status"):
    """
    Get aggregated health metrics across all integrations
    """
    try:
        monitor = await get_health_monitor()
        all_health = await monitor.get_all_health_metrics()

        if not all_health:
            return {
                "status": "success",
                "data": {},
                "message": "No health data available",
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Aggregate by status
        if group_by == "status":
            aggregated = {}
            for integration_name, health in all_health.items():
                status = health.status.value
                if status not in aggregated:
                    aggregated[status] = {"count": 0, "average_score": 0, "integrations": []}

                aggregated[status]["count"] += 1
                aggregated[status]["integrations"].append(integration_name)
                # Update running average
                current_avg = aggregated[status]["average_score"]
                count = aggregated[status]["count"]
                aggregated[status]["average_score"] = (current_avg * (count - 1) + health.score) / count

        # Aggregate by performance tiers
        elif group_by == "performance":
            aggregated = {
                "high_performance": {"count": 0, "integrations": []},
                "medium_performance": {"count": 0, "integrations": []},
                "low_performance": {"count": 0, "integrations": []},
            }

            for integration_name, health in all_health.items():
                if health.success_rate >= 95 and health.average_response_time <= 1000:
                    tier = "high_performance"
                elif health.success_rate >= 80 and health.average_response_time <= 3000:
                    tier = "medium_performance"
                else:
                    tier = "low_performance"

                aggregated[tier]["count"] += 1
                aggregated[tier]["integrations"].append(integration_name)

        else:
            raise HTTPException(status_code=400, detail=f"Invalid group_by parameter: {group_by}")

        return {
            "status": "success",
            "data": aggregated,
            "group_by": group_by,
            "time_range": time_range,
            "total_integrations": len(all_health),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get aggregated health metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get aggregated metrics: {str(e)}")


# Background task functions


async def force_health_check(integration_name: str):
    """
    Force an immediate health check for a specific integration
    """
    try:
        monitor = await get_health_monitor()

        # Get the integration instance (this would need to be implemented)
        # For now, we'll just trigger a refresh of the health cache

        logger.info(f"🔄 Forcing health check for {integration_name}")

        # This is a simplified version - in a real implementation,
        # you would get the actual integration instance and run its health check

        # For demonstration, we'll just update the last check time
        current_health = await monitor.get_integration_health(integration_name)
        if current_health:
            current_health.last_check = datetime.utcnow()
            monitor.health_cache[integration_name] = current_health

    except Exception as e:
        logger.error(f"❌ Failed to force health check for {integration_name}: {e}")


# Export the router
__all__ = ['health_router']
