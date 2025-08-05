"""
HR and operations integrations for Wand
"""

import base64
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class WorkdayIntegration(BaseIntegration):
    """Workday HR management system integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "tenant_url": os.getenv("WORKDAY_TENANT_URL", ""),
            "username": os.getenv("WORKDAY_USERNAME", ""),
            "password": os.getenv("WORKDAY_PASSWORD", ""),
            "client_id": os.getenv("WORKDAY_CLIENT_ID", ""),
            "client_secret": os.getenv("WORKDAY_CLIENT_SECRET", ""),
            "api_version": "v1",
        }
        super().__init__("workday", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Workday integration"""
        if not self.config["tenant_url"]:
            logger.warning("⚠️  Workday tenant URL not configured")
        logger.info("✅ Workday integration initialized")

    async def cleanup(self):
        """Cleanup Workday resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Workday API health"""
        if not self.config["tenant_url"]:
            return {"status": "unhealthy", "error": "Tenant URL not configured"}

        try:
            # Basic connectivity check
            async with aiohttp.ClientSession() as session:
                test_url = f"{self.config['tenant_url']}/ccx/api/{self.config['api_version']}"
                async with session.get(test_url, timeout=10) as response:
                    if response.status in [200, 401, 403]:  # API exists, auth may be required
                        return {
                            "status": "healthy" if response.status == 200 else "partial",
                            "tenant_url": self.config["tenant_url"],
                            "api_accessible": True,
                        }
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Workday operations"""

        if operation == "get_workers":
            return await self._get_workers(**kwargs)
        elif operation == "get_worker":
            return await self._get_worker(**kwargs)
        elif operation == "get_organizations":
            return await self._get_organizations(**kwargs)
        elif operation == "get_time_off":
            return await self._get_time_off(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _get_workers(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Get workers from Workday"""
        if not self.config["tenant_url"]:
            return {"success": False, "error": "Tenant URL not configured"}

        # Note: Real Workday integration would require proper OAuth2 authentication
        # This is a simplified structure for the integration framework
        return {
            "success": True,
            "workers": [],
            "total": 0,
            "note": "Configure WORKDAY_TENANT_URL and authentication for real data",
        }


class BambooHRIntegration(BaseIntegration):
    """BambooHR integration for HR information systems"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_key": os.getenv("BAMBOOHR_API_KEY", ""),
            "subdomain": os.getenv("BAMBOOHR_SUBDOMAIN", ""),
            "base_url": "https://api.bamboohr.com/api/gateway.php",
        }
        super().__init__("bamboohr", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize BambooHR integration"""
        if not all([self.config["api_key"], self.config["subdomain"]]):
            logger.warning("⚠️  BambooHR API key or subdomain not configured")
        logger.info("✅ BambooHR integration initialized")

    async def cleanup(self):
        """Cleanup BambooHR resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check BambooHR API health"""
        if not all([self.config["api_key"], self.config["subdomain"]]):
            return {"status": "unhealthy", "error": "API key or subdomain not configured"}

        try:
            headers = await self._get_auth_headers()
            url = f"{self.config['base_url']}/{self.config['subdomain']}/v1/meta/users"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return {"status": "healthy", "subdomain": self.config["subdomain"], "api_accessible": True}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for BambooHR API"""
        credentials = base64.b64encode(f"{self.config['api_key']}:x".encode()).decode()
        return {"Authorization": f"Basic {credentials}", "Accept": "application/json"}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute BambooHR operations"""

        if operation == "get_employees":
            return await self._get_employees(**kwargs)
        elif operation == "get_employee":
            return await self._get_employee(**kwargs)
        elif operation == "get_time_off_requests":
            return await self._get_time_off_requests(**kwargs)
        elif operation == "update_employee":
            return await self._update_employee(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _get_employees(self, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get employees from BambooHR"""
        if not all([self.config["api_key"], self.config["subdomain"]]):
            return {"success": False, "error": "API credentials not configured"}

        headers = await self._get_auth_headers()
        field_list = ",".join(fields) if fields else "firstName,lastName,workEmail,jobTitle"
        url = f"{self.config['base_url']}/{self.config['subdomain']}/v1/employees/directory"

        try:
            params = {"fields": field_list}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        employees = data.get("employees", [])

                        employee_list = []
                        for emp in employees:
                            employee_list.append(
                                {
                                    "id": emp.get("id"),
                                    "first_name": emp.get("firstName"),
                                    "last_name": emp.get("lastName"),
                                    "email": emp.get("workEmail"),
                                    "job_title": emp.get("jobTitle"),
                                    "department": emp.get("department"),
                                    "location": emp.get("location"),
                                }
                            )

                        return {"success": True, "employees": employee_list, "total": len(employee_list)}
                    else:
                        return {"success": False, "error": f"API returned {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


class TogglIntegration(BaseIntegration):
    """Toggl time tracking integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_token": os.getenv("TOGGL_API_TOKEN", ""),
            "workspace_id": os.getenv("TOGGL_WORKSPACE_ID", ""),
            "base_url": "https://api.track.toggl.com/api/v9",
        }
        super().__init__("toggl", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Toggl integration"""
        if not self.config["api_token"]:
            logger.warning("⚠️  Toggl API token not configured")
        logger.info("✅ Toggl integration initialized")

    async def cleanup(self):
        """Cleanup Toggl resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Toggl API health"""
        if not self.config["api_token"]:
            return {"status": "unhealthy", "error": "API token not configured"}

        try:
            headers = await self._get_auth_headers()
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/me", headers=headers) as response:
                    if response.status == 200:
                        user_data = await response.json()
                        return {"status": "healthy", "user": user_data.get("fullname"), "email": user_data.get("email")}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Toggl API"""
        credentials = base64.b64encode(f"{self.config['api_token']}:api_token".encode()).decode()
        return {"Authorization": f"Basic {credentials}", "Content-Type": "application/json"}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Toggl operations"""

        if operation == "start_timer":
            return await self._start_timer(**kwargs)
        elif operation == "stop_timer":
            return await self._stop_timer(**kwargs)
        elif operation == "get_time_entries":
            return await self._get_time_entries(**kwargs)
        elif operation == "create_project":
            return await self._create_project(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _start_timer(
        self, description: str, project_id: Optional[int] = None, tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Start a time tracking timer"""
        if not self.config["api_token"]:
            return {"success": False, "error": "API token not configured"}

        headers = await self._get_auth_headers()

        timer_data = {"description": description, "created_with": "Wand Integration System"}

        if project_id:
            timer_data["project_id"] = project_id
        if tags:
            timer_data["tags"] = tags

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/workspaces/{self.config['workspace_id']}/time_entries",
                    headers=headers,
                    json=timer_data,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "time_entry_id": result.get("id"),
                            "description": description,
                            "start_time": result.get("start"),
                            "running": result.get("duration", 0) < 0,
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error}

        except Exception as e:
            return {"success": False, "error": str(e)}


class HarvestIntegration(BaseIntegration):
    """Harvest time tracking and invoicing integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "access_token": os.getenv("HARVEST_ACCESS_TOKEN", ""),
            "account_id": os.getenv("HARVEST_ACCOUNT_ID", ""),
            "base_url": "https://api.harvestapp.com/v2",
        }
        super().__init__("harvest", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Harvest integration"""
        if not all([self.config["access_token"], self.config["account_id"]]):
            logger.warning("⚠️  Harvest access token or account ID not configured")
        logger.info("✅ Harvest integration initialized")

    async def cleanup(self):
        """Cleanup Harvest resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Harvest API health"""
        if not all([self.config["access_token"], self.config["account_id"]]):
            return {"status": "unhealthy", "error": "Access token or account ID not configured"}

        try:
            headers = await self._get_auth_headers()
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/users/me", headers=headers) as response:
                    if response.status == 200:
                        user_data = await response.json()
                        return {
                            "status": "healthy",
                            "user": f"{user_data.get('first_name')} {user_data.get('last_name')}",
                            "email": user_data.get("email"),
                        }
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Harvest API"""
        return {
            "Authorization": f"Bearer {self.config['access_token']}",
            "Harvest-Account-Id": self.config["account_id"],
            "Content-Type": "application/json",
        }

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Harvest operations"""

        if operation == "create_time_entry":
            return await self._create_time_entry(**kwargs)
        elif operation == "get_time_entries":
            return await self._get_time_entries(**kwargs)
        elif operation == "get_projects":
            return await self._get_projects(**kwargs)
        elif operation == "create_invoice":
            return await self._create_invoice(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _create_time_entry(
        self, project_id: int, task_id: int, spent_date: str, hours: float, notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a time entry in Harvest"""
        if not all([self.config["access_token"], self.config["account_id"]]):
            return {"success": False, "error": "Access token or account ID not configured"}

        headers = await self._get_auth_headers()

        time_entry_data = {"project_id": project_id, "task_id": task_id, "spent_date": spent_date, "hours": hours}

        if notes:
            time_entry_data["notes"] = notes

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/time_entries", headers=headers, json=time_entry_data
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        return {
                            "success": True,
                            "time_entry_id": result.get("id"),
                            "project_id": project_id,
                            "hours": hours,
                            "spent_date": spent_date,
                            "notes": notes,
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_projects(self, is_active: bool = True) -> Dict[str, Any]:
        """Get projects from Harvest"""
        if not all([self.config["access_token"], self.config["account_id"]]):
            return {"success": False, "error": "Access token or account ID not configured"}

        headers = await self._get_auth_headers()
        params = {"is_active": str(is_active).lower()}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}/projects", headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        projects = []

                        for project in data.get("projects", []):
                            projects.append(
                                {
                                    "id": project.get("id"),
                                    "name": project.get("name"),
                                    "code": project.get("code"),
                                    "is_active": project.get("is_active"),
                                    "client_name": project.get("client", {}).get("name"),
                                    "budget": project.get("budget"),
                                    "budget_by": project.get("budget_by"),
                                }
                            )

                        return {"success": True, "projects": projects, "total": len(projects)}
                    else:
                        return {"success": False, "error": f"API returned {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}
