"""
CRM and sales integrations for Wand
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class SalesforceIntegration(BaseIntegration):
    """Salesforce CRM integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "client_id": os.getenv("SALESFORCE_CLIENT_ID", ""),
            "client_secret": os.getenv("SALESFORCE_CLIENT_SECRET", ""),
            "username": os.getenv("SALESFORCE_USERNAME", ""),
            "password": os.getenv("SALESFORCE_PASSWORD", ""),
            "security_token": os.getenv("SALESFORCE_SECURITY_TOKEN", ""),
            "domain": os.getenv("SALESFORCE_DOMAIN", "login"),  # or "test" for sandbox
            "version": "v58.0",
        }
        super().__init__("salesforce", {**default_config, **(config or {})})
        self.access_token = None
        self.instance_url = None

    async def initialize(self):
        """Initialize Salesforce integration"""
        if not all(
            [self.config["client_id"], self.config["client_secret"], self.config["username"], self.config["password"]]
        ):
            logger.warning("⚠️  Salesforce credentials not fully configured")
        else:
            await self._authenticate()
            logger.info("✅ Salesforce integration initialized")

    async def cleanup(self):
        """Cleanup Salesforce resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Salesforce connection health"""
        if not self.access_token:
            return {"status": "unhealthy", "error": "Not authenticated"}

        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.instance_url}/services/data/{self.config['version']}/", headers=headers
                ) as response:
                    if response.status == 200:
                        return {"status": "healthy", "instance_url": self.instance_url}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _authenticate(self):
        """Authenticate with Salesforce"""
        auth_url = f"https://{self.config['domain']}.salesforce.com/services/oauth2/token"

        data = {
            "grant_type": "password",
            "client_id": self.config["client_id"],
            "client_secret": self.config["client_secret"],
            "username": self.config["username"],
            "password": self.config["password"] + self.config["security_token"],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(auth_url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.access_token = result["access_token"]
                        self.instance_url = result["instance_url"]
                    else:
                        error = await response.json()
                        logger.error(f"Salesforce authentication failed: {error}")
        except Exception as e:
            logger.error(f"Salesforce authentication error: {e}")

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Salesforce operations"""

        if operation == "create_lead":
            return await self._create_lead(**kwargs)
        elif operation == "update_opportunity":
            return await self._update_opportunity(**kwargs)
        elif operation == "query_records":
            return await self._query_records(**kwargs)
        elif operation == "create_account":
            return await self._create_account(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _create_lead(
        self, first_name: str, last_name: str, company: str, email: Optional[str] = None, phone: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a lead in Salesforce"""
        if not self.access_token:
            return {"success": False, "error": "Not authenticated"}

        headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}

        data = {"FirstName": first_name, "LastName": last_name, "Company": company}

        if email:
            data["Email"] = email
        if phone:
            data["Phone"] = phone

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.instance_url}/services/data/{self.config['version']}/sobjects/Lead/",
                    headers=headers,
                    json=data,
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        return {
                            "success": True,
                            "lead_id": result["id"],
                            "first_name": first_name,
                            "last_name": last_name,
                            "company": company,
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error}

        except Exception as e:
            return {"success": False, "error": str(e)}


class HubSpotIntegration(BaseIntegration):
    """HubSpot CRM integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {"api_key": os.getenv("HUBSPOT_API_KEY", ""), "base_url": "https://api.hubapi.com"}
        super().__init__("hubspot", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize HubSpot integration"""
        if not self.config["api_key"]:
            logger.warning("⚠️  HubSpot API key not configured")
        logger.info("✅ HubSpot integration initialized")

    async def cleanup(self):
        """Cleanup HubSpot resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check HubSpot API health"""
        if not self.config["api_key"]:
            return {"status": "unhealthy", "error": "API key not configured"}

        try:
            headers = {"Authorization": f"Bearer {self.config['api_key']}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}/contacts/v1/lists/all/contacts/all",
                    headers=headers,
                    params={"count": 1},
                ) as response:
                    if response.status == 200:
                        return {"status": "healthy"}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute HubSpot operations"""

        if operation == "create_contact":
            return await self._create_contact(**kwargs)
        elif operation == "create_deal":
            return await self._create_deal(**kwargs)
        elif operation == "get_contacts":
            return await self._get_contacts(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}


class StripeIntegration(BaseIntegration):
    """Stripe payment processing integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "secret_key": os.getenv("STRIPE_SECRET_KEY", ""),
            "publishable_key": os.getenv("STRIPE_PUBLISHABLE_KEY", ""),
            "base_url": "https://api.stripe.com/v1",
        }
        super().__init__("stripe", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Stripe integration"""
        if not self.config["secret_key"]:
            logger.warning("⚠️  Stripe secret key not configured")
        logger.info("✅ Stripe integration initialized")

    async def cleanup(self):
        """Cleanup Stripe resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Stripe API health"""
        if not self.config["secret_key"]:
            return {"status": "unhealthy", "error": "Secret key not configured"}

        try:
            headers = {"Authorization": f"Bearer {self.config['secret_key']}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/balance", headers=headers) as response:
                    if response.status == 200:
                        return {"status": "healthy"}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Stripe operations"""

        if operation == "create_payment_intent":
            return await self._create_payment_intent(**kwargs)
        elif operation == "create_customer":
            return await self._create_customer(**kwargs)
        elif operation == "list_charges":
            return await self._list_charges(**kwargs)
        elif operation == "get_balance":
            return await self._get_balance(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _create_payment_intent(
        self, amount: int, currency: str = "usd", customer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a payment intent"""
        if not self.config["secret_key"]:
            return {"success": False, "error": "Secret key not configured"}

        headers = {
            "Authorization": f"Bearer {self.config['secret_key']}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {"amount": amount, "currency": currency}

        if customer_id:
            data["customer"] = customer_id

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/payment_intents", headers=headers, data=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "payment_intent_id": result["id"],
                            "amount": amount,
                            "currency": currency,
                            "status": result["status"],
                            "client_secret": result["client_secret"],
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("error", {}).get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}


class PipedriveIntegration(BaseIntegration):
    """Pipedrive CRM integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_token": os.getenv("PIPEDRIVE_API_TOKEN", ""),
            "company_domain": os.getenv("PIPEDRIVE_COMPANY_DOMAIN", ""),
            "base_url": "https://api.pipedrive.com/v1",
        }
        super().__init__("pipedrive", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Pipedrive integration"""
        if not self.config["api_token"]:
            logger.warning("⚠️  Pipedrive API token not configured")
        logger.info("✅ Pipedrive integration initialized")

    async def cleanup(self):
        """Cleanup Pipedrive resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Pipedrive API health"""
        if not self.config["api_token"]:
            return {"status": "unhealthy", "error": "API token not configured"}

        try:
            params = {"api_token": self.config["api_token"]}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/users/me", params=params) as response:
                    if response.status == 200:
                        return {"status": "healthy"}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Pipedrive operations"""

        if operation == "create_deal":
            return await self._create_deal(**kwargs)
        elif operation == "create_person":
            return await self._create_person(**kwargs)
        elif operation == "get_deals":
            return await self._get_deals(**kwargs)
        elif operation == "update_deal":
            return await self._update_deal(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _create_deal(
        self, title: str, value: Optional[float] = None, currency: str = "USD", person_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a deal in Pipedrive"""
        if not self.config["api_token"]:
            return {"success": False, "error": "API token not configured"}

        data = {"title": title, "api_token": self.config["api_token"]}

        if value:
            data["value"] = value
            data["currency"] = currency
        if person_id:
            data["person_id"] = person_id

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.config['base_url']}/deals", data=data) as response:
                    if response.status == 201:
                        result = await response.json()
                        return {
                            "success": True,
                            "deal_id": result["data"]["id"],
                            "title": title,
                            "value": value,
                            "currency": currency,
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("error", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}
