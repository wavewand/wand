"""
Security tools integrations for Wand
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class VaultIntegration(BaseIntegration):
    """HashiCorp Vault secret management integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "vault_url": os.getenv("VAULT_URL", "http://localhost:8200"),
            "vault_token": os.getenv("VAULT_TOKEN", ""),
            "mount_path": "secret",
            "api_version": "v1",
        }
        super().__init__("vault", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Vault integration"""
        if not self.config["vault_token"]:
            logger.warning("⚠️  Vault token not configured")
        logger.info("✅ Vault integration initialized")

    async def cleanup(self):
        """Cleanup Vault resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Vault health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['vault_url']}/v1/sys/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        return {
                            "status": "healthy",
                            "vault_version": health_data.get("version", "unknown"),
                            "sealed": health_data.get("sealed", True),
                        }
                    else:
                        return {"status": "unhealthy", "error": f"Health check returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Vault operations"""

        if operation == "read_secret":
            return await self._read_secret(**kwargs)
        elif operation == "write_secret":
            return await self._write_secret(**kwargs)
        elif operation == "delete_secret":
            return await self._delete_secret(**kwargs)
        elif operation == "list_secrets":
            return await self._list_secrets(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _read_secret(self, path: str) -> Dict[str, Any]:
        """Read secret from Vault"""
        if not self.config["vault_token"]:
            return {"success": False, "error": "Vault token not configured"}

        headers = {"X-Vault-Token": self.config["vault_token"]}
        url = f"{self.config['vault_url']}/{self.config['api_version']}/{self.config['mount_path']}/data/{path}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "path": path,
                            "data": result.get("data", {}).get("data", {}),
                            "metadata": result.get("data", {}).get("metadata", {}),
                        }
                    elif response.status == 404:
                        return {"success": False, "error": "Secret not found"}
                    else:
                        return {"success": False, "error": f"Vault API returned {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _write_secret(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Write secret to Vault"""
        if not self.config["vault_token"]:
            return {"success": False, "error": "Vault token not configured"}

        headers = {"X-Vault-Token": self.config["vault_token"], "Content-Type": "application/json"}

        url = f"{self.config['vault_url']}/{self.config['api_version']}/{self.config['mount_path']}/data/{path}"
        payload = {"data": data}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status in [200, 204]:
                        return {"success": True, "path": path, "keys_written": list(data.keys())}
                    else:
                        error_data = await response.json()
                        return {"success": False, "error": error_data.get("errors", ["Unknown error"])}

        except Exception as e:
            return {"success": False, "error": str(e)}


class OnePasswordIntegration(BaseIntegration):
    """1Password secret management integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "service_account_token": os.getenv("ONEPASSWORD_SERVICE_ACCOUNT", ""),
            "base_url": "https://my.1password.com/api/v1",
        }
        super().__init__("onepassword", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize 1Password integration"""
        if not self.config["service_account_token"]:
            logger.warning("⚠️  1Password service account token not configured")
        logger.info("✅ 1Password integration initialized")

    async def cleanup(self):
        """Cleanup 1Password resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check 1Password API health"""
        if not self.config["service_account_token"]:
            return {"status": "unhealthy", "error": "Service account token not configured"}

        # 1Password Connect API health check would go here
        # For now, return basic status
        return {"status": "healthy", "configured": True}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute 1Password operations"""

        if operation == "get_item":
            return await self._get_item(**kwargs)
        elif operation == "list_vaults":
            return await self._list_vaults(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}


class OktaIntegration(BaseIntegration):
    """Okta identity management integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "domain": os.getenv("OKTA_DOMAIN", ""),
            "api_token": os.getenv("OKTA_API_TOKEN", ""),
            "base_url": None,  # Will be constructed from domain
        }
        super().__init__("okta", {**default_config, **(config or {})})

        if self.config["domain"]:
            self.config["base_url"] = f"https://{self.config['domain']}.okta.com/api/v1"

    async def initialize(self):
        """Initialize Okta integration"""
        if not self.config["domain"] or not self.config["api_token"]:
            logger.warning("⚠️  Okta domain or API token not configured")
        logger.info("✅ Okta integration initialized")

    async def cleanup(self):
        """Cleanup Okta resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Okta API health"""
        if not self.config["api_token"] or not self.config["base_url"]:
            return {"status": "unhealthy", "error": "API token or domain not configured"}

        try:
            headers = {"Authorization": f"SSWS {self.config['api_token']}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/org", headers=headers) as response:
                    if response.status == 200:
                        org_data = await response.json()
                        return {
                            "status": "healthy",
                            "org_id": org_data.get("id"),
                            "subdomain": org_data.get("subdomain"),
                        }
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Okta operations"""

        if operation == "list_users":
            return await self._list_users(**kwargs)
        elif operation == "create_user":
            return await self._create_user(**kwargs)
        elif operation == "deactivate_user":
            return await self._deactivate_user(**kwargs)
        elif operation == "list_groups":
            return await self._list_groups(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _list_users(self, limit: int = 10, filter_expr: Optional[str] = None) -> Dict[str, Any]:
        """List Okta users"""
        if not self.config["api_token"] or not self.config["base_url"]:
            return {"success": False, "error": "API token or domain not configured"}

        headers = {"Authorization": f"SSWS {self.config['api_token']}"}
        params = {"limit": limit}

        if filter_expr:
            params["filter"] = filter_expr

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/users", headers=headers, params=params) as response:
                    if response.status == 200:
                        users = await response.json()

                        user_list = []
                        for user in users:
                            user_list.append(
                                {
                                    "id": user["id"],
                                    "email": user["profile"]["email"],
                                    "first_name": user["profile"]["firstName"],
                                    "last_name": user["profile"]["lastName"],
                                    "status": user["status"],
                                    "created": user["created"],
                                    "last_login": user.get("lastLogin"),
                                }
                            )

                        return {"success": True, "users": user_list, "total": len(user_list)}
                    else:
                        error = await response.json()
                        return {"success": False, "error": error}

        except Exception as e:
            return {"success": False, "error": str(e)}


class Auth0Integration(BaseIntegration):
    """Auth0 identity management integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "domain": os.getenv("AUTH0_DOMAIN", ""),
            "client_id": os.getenv("AUTH0_CLIENT_ID", ""),
            "client_secret": os.getenv("AUTH0_CLIENT_SECRET", ""),
            "audience": None,  # Management API audience
            "base_url": None,  # Will be constructed from domain
        }
        super().__init__("auth0", {**default_config, **(config or {})})

        if self.config["domain"]:
            self.config["base_url"] = f"https://{self.config['domain']}.auth0.com/api/v2"
            self.config["audience"] = f"https://{self.config['domain']}.auth0.com/api/v2/"

        self.management_token = None

    async def initialize(self):
        """Initialize Auth0 integration"""
        if not all([self.config["domain"], self.config["client_id"], self.config["client_secret"]]):
            logger.warning("⚠️  Auth0 credentials not fully configured")
        else:
            await self._get_management_token()
            logger.info("✅ Auth0 integration initialized")

    async def cleanup(self):
        """Cleanup Auth0 resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Auth0 API health"""
        if not self.management_token:
            return {"status": "unhealthy", "error": "Not authenticated"}

        try:
            headers = {"Authorization": f"Bearer {self.management_token}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/stats/daily", headers=headers) as response:
                    if response.status == 200:
                        return {"status": "healthy", "domain": self.config["domain"]}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _get_management_token(self):
        """Get Auth0 Management API token"""
        token_url = f"https://{self.config['domain']}.auth0.com/oauth/token"

        data = {
            "client_id": self.config["client_id"],
            "client_secret": self.config["client_secret"],
            "audience": self.config["audience"],
            "grant_type": "client_credentials",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(token_url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.management_token = result["access_token"]
                    else:
                        error = await response.json()
                        logger.error(f"Auth0 token request failed: {error}")
        except Exception as e:
            logger.error(f"Auth0 token request error: {e}")

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Auth0 operations"""

        if operation == "list_users":
            return await self._list_users(**kwargs)
        elif operation == "create_user":
            return await self._create_user(**kwargs)
        elif operation == "update_user":
            return await self._update_user(**kwargs)
        elif operation == "get_logs":
            return await self._get_logs(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _list_users(self, per_page: int = 10, page: int = 0) -> Dict[str, Any]:
        """List Auth0 users"""
        if not self.management_token:
            return {"success": False, "error": "Not authenticated"}

        headers = {"Authorization": f"Bearer {self.management_token}"}
        params = {"per_page": per_page, "page": page}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/users", headers=headers, params=params) as response:
                    if response.status == 200:
                        users = await response.json()

                        user_list = []
                        for user in users:
                            user_list.append(
                                {
                                    "user_id": user["user_id"],
                                    "email": user.get("email"),
                                    "name": user.get("name"),
                                    "nickname": user.get("nickname"),
                                    "created_at": user["created_at"],
                                    "last_login": user.get("last_login"),
                                    "logins_count": user.get("logins_count", 0),
                                }
                            )

                        return {"success": True, "users": user_list, "total": len(user_list), "page": page}
                    else:
                        error = await response.json()
                        return {"success": False, "error": error}

        except Exception as e:
            return {"success": False, "error": str(e)}
