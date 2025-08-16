"""
Enterprise Identity Management integrations for Wand
Includes ServiceNow, SailPoint, Microsoft Entra, and Britive integrations
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class ServiceNowIntegration(BaseIntegration):
    """ServiceNow IT Service Management integration using PySNC"""

    REQUIRED_CONFIG_KEYS = ["instance_url", "username", "password"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "instance_url": os.getenv("SERVICENOW_INSTANCE_URL", ""),
            "username": os.getenv("SERVICENOW_USERNAME", ""),
            "password": os.getenv("SERVICENOW_PASSWORD", ""),
            "api_version": "v1",
            "timeout": 30,
        }
        super().__init__("servicenow", {**default_config, **(config or {})})
        self.client = None

    async def initialize(self):
        """Initialize ServiceNow integration"""
        try:
            # Import PySNC here to avoid import errors if not installed
            from pysnc import ServiceNowClient

            if not all([self.config["instance_url"], self.config["username"], self.config["password"]]):
                logger.warning("⚠️  ServiceNow credentials not fully configured")
                self.enabled = False
                return

            self.client = ServiceNowClient(
                self.config["instance_url"], (self.config["username"], self.config["password"])
            )
            logger.info("✅ ServiceNow integration initialized")
        except ImportError:
            logger.error("❌ PySNC library not installed. Run: pip install pysnc")
            self.enabled = False
        except Exception as e:
            logger.error(f"❌ ServiceNow initialization failed: {e}")
            self.enabled = False

    async def cleanup(self):
        """Cleanup ServiceNow resources"""
        self.client = None

    async def health_check(self) -> Dict[str, Any]:
        """Check ServiceNow connection health"""
        base_health = await super().health_check()

        if not self.client:
            base_health.update({"status": "unhealthy", "error": "Client not initialized"})
            return base_health

        try:
            # Test connection by querying sys_user table with limit 1
            gr = self.client.GlideRecord('sys_user')
            gr.set_limit(1)
            gr.query()

            if gr.get_row_count() >= 0:  # Even 0 records means connection is working
                base_health.update(
                    {"status": "healthy", "instance_url": self.config["instance_url"], "connection": "active"}
                )
            else:
                base_health.update({"status": "unhealthy", "error": "Unable to query sys_user table"})

        except Exception as e:
            base_health.update({"status": "unhealthy", "error": str(e)})

        return base_health

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute ServiceNow operations"""
        if not self.client:
            return {"success": False, "error": "ServiceNow client not initialized"}

        try:
            if operation == "create_incident":
                return await self._create_incident(**kwargs)
            elif operation == "query_records":
                return await self._query_records(**kwargs)
            elif operation == "update_record":
                return await self._update_record(**kwargs)
            elif operation == "get_user":
                return await self._get_user(**kwargs)
            elif operation == "create_user":
                return await self._create_user(**kwargs)
            elif operation == "list_tables":
                return await self._list_tables(**kwargs)
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _create_incident(self, **kwargs) -> Dict[str, Any]:
        """Create a new incident in ServiceNow"""
        required_fields = kwargs.get('required_fields', ['short_description'])

        # Validate required fields
        for field in required_fields:
            if field not in kwargs:
                return {"success": False, "error": f"Missing required field: {field}"}

        try:
            gr = self.client.GlideRecord('incident')
            gr.initialize()

            # Set incident fields
            for field, value in kwargs.items():
                if field not in ['required_fields']:
                    setattr(gr, field, value)

            # Set default priority if not specified
            if not hasattr(gr, 'priority') or not gr.priority:
                gr.priority = kwargs.get('priority', '3')  # Default to medium priority

            sys_id = gr.insert()

            return {
                "success": True,
                "incident_id": sys_id,
                "number": gr.number,
                "message": "Incident created successfully",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to create incident: {str(e)}"}

    async def _query_records(self, **kwargs) -> Dict[str, Any]:
        """Query records from ServiceNow table"""
        table = kwargs.get('table', 'incident')
        limit = kwargs.get('limit', 10)
        query_filter = kwargs.get('query_filter', '')

        try:
            gr = self.client.GlideRecord(table)

            if query_filter:
                gr.add_encoded_query(query_filter)

            gr.set_limit(limit)
            gr.query()

            records = []
            for record in gr:
                record_data = {}
                # Get all fields for the record
                for field in gr.get_fields():
                    record_data[field] = getattr(record, field, '')
                records.append(record_data)

            return {
                "success": True,
                "table": table,
                "records": records,
                "count": len(records),
                "message": f"Retrieved {len(records)} records from {table}",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to query records: {str(e)}"}

    async def _update_record(self, **kwargs) -> Dict[str, Any]:
        """Update a record in ServiceNow"""
        table = kwargs.get('table', 'incident')
        sys_id = kwargs.get('sys_id')

        if not sys_id:
            return {"success": False, "error": "sys_id is required for update operations"}

        try:
            gr = self.client.GlideRecord(table)
            if not gr.get(sys_id):
                return {"success": False, "error": f"Record with sys_id {sys_id} not found"}

            # Update fields
            for field, value in kwargs.items():
                if field not in ['table', 'sys_id']:
                    setattr(gr, field, value)

            gr.update()

            return {"success": True, "sys_id": sys_id, "table": table, "message": "Record updated successfully"}
        except Exception as e:
            return {"success": False, "error": f"Failed to update record: {str(e)}"}

    async def _get_user(self, **kwargs) -> Dict[str, Any]:
        """Get user information from ServiceNow"""
        user_id = kwargs.get('user_id')
        username = kwargs.get('username')
        email = kwargs.get('email')

        if not any([user_id, username, email]):
            return {"success": False, "error": "user_id, username, or email is required"}

        try:
            gr = self.client.GlideRecord('sys_user')

            if user_id:
                gr.add_query('sys_id', user_id)
            elif username:
                gr.add_query('user_name', username)
            elif email:
                gr.add_query('email', email)

            gr.query()

            if gr.next():
                user_data = {
                    'sys_id': gr.sys_id,
                    'user_name': gr.user_name,
                    'first_name': gr.first_name,
                    'last_name': gr.last_name,
                    'email': gr.email,
                    'active': gr.active,
                    'department': gr.department.get_display_value() if hasattr(gr, 'department') else '',
                    'title': gr.title,
                    'phone': gr.phone,
                }

                return {"success": True, "user": user_data, "message": "User found successfully"}
            else:
                return {"success": False, "error": "User not found"}

        except Exception as e:
            return {"success": False, "error": f"Failed to get user: {str(e)}"}

    async def _create_user(self, **kwargs) -> Dict[str, Any]:
        """Create a new user in ServiceNow"""
        required_fields = ['user_name', 'first_name', 'last_name', 'email']

        # Validate required fields
        for field in required_fields:
            if field not in kwargs:
                return {"success": False, "error": f"Missing required field: {field}"}

        try:
            gr = self.client.GlideRecord('sys_user')
            gr.initialize()

            # Set user fields
            for field, value in kwargs.items():
                setattr(gr, field, value)

            # Set default active status if not specified
            if not hasattr(gr, 'active') or gr.active == '':
                gr.active = 'true'

            sys_id = gr.insert()

            return {
                "success": True,
                "user_id": sys_id,
                "username": gr.user_name,
                "message": "User created successfully",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to create user: {str(e)}"}

    async def _list_tables(self, **kwargs) -> Dict[str, Any]:
        """List available tables in ServiceNow"""
        try:
            gr = self.client.GlideRecord('sys_db_object')
            gr.add_query('super_class', 'CONTAINS', 'task')  # Focus on task-related tables
            gr.set_limit(kwargs.get('limit', 50))
            gr.query()

            tables = []
            for table in gr:
                table_info = {
                    'name': table.name,
                    'label': table.label,
                    'is_extendable': table.is_extendable,
                    'super_class': table.super_class,
                }
                tables.append(table_info)

            return {
                "success": True,
                "tables": tables,
                "count": len(tables),
                "message": f"Retrieved {len(tables)} tables",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to list tables: {str(e)}"}


class SailPointIntegration(BaseIntegration):
    """SailPoint IdentityNow integration using official Python SDK"""

    REQUIRED_CONFIG_KEYS = ["base_url", "client_id", "client_secret"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "base_url": os.getenv("SAILPOINT_BASE_URL", ""),  # https://tenant.api.identitynow.com
            "client_id": os.getenv("SAILPOINT_CLIENT_ID", ""),
            "client_secret": os.getenv("SAILPOINT_CLIENT_SECRET", ""),
            "timeout": 30,
        }
        super().__init__("sailpoint", {**default_config, **(config or {})})
        self.client = None
        self.access_token = None

    async def initialize(self):
        """Initialize SailPoint integration"""
        try:
            if not all([self.config["base_url"], self.config["client_id"], self.config["client_secret"]]):
                logger.warning("⚠️  SailPoint credentials not fully configured")
                self.enabled = False
                return

            await self._authenticate()
            logger.info("✅ SailPoint integration initialized")
        except Exception as e:
            logger.error(f"❌ SailPoint initialization failed: {e}")
            self.enabled = False

    async def cleanup(self):
        """Cleanup SailPoint resources"""
        self.client = None
        self.access_token = None

    async def _authenticate(self):
        """Authenticate with SailPoint using OAuth2"""
        auth_url = f"{self.config['base_url']}/oauth/token"

        data = {
            "grant_type": "client_credentials",
            "client_id": self.config["client_id"],
            "client_secret": self.config["client_secret"],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(auth_url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.access_token = result["access_token"]
                    else:
                        error = await response.text()
                        logger.error(f"SailPoint authentication failed: {error}")
                        raise Exception(f"Authentication failed: {error}")
        except Exception as e:
            logger.error(f"SailPoint authentication error: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check SailPoint connection health"""
        base_health = await super().health_check()

        if not self.access_token:
            base_health.update({"status": "unhealthy", "error": "Not authenticated"})
            return base_health

        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}/v3/public-identities", headers=headers, params={"limit": 1}
                ) as response:
                    if response.status == 200:
                        base_health.update(
                            {"status": "healthy", "base_url": self.config["base_url"], "connection": "active"}
                        )
                    else:
                        base_health.update({"status": "unhealthy", "error": f"API returned {response.status}"})
        except Exception as e:
            base_health.update({"status": "unhealthy", "error": str(e)})

        return base_health

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute SailPoint operations"""
        if not self.access_token:
            return {"success": False, "error": "SailPoint not authenticated"}

        try:
            if operation == "get_identities":
                return await self._get_identities(**kwargs)
            elif operation == "get_identity":
                return await self._get_identity(**kwargs)
            elif operation == "request_access":
                return await self._request_access(**kwargs)
            elif operation == "get_accounts":
                return await self._get_accounts(**kwargs)
            elif operation == "launch_campaign":
                return await self._launch_campaign(**kwargs)
            elif operation == "get_access_profiles":
                return await self._get_access_profiles(**kwargs)
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_identities(self, **kwargs) -> Dict[str, Any]:
        """Get list of identities from SailPoint"""
        limit = kwargs.get('limit', 10)
        filters = kwargs.get('filters', '')

        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            params = {"limit": limit}
            if filters:
                params["filters"] = filters

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}/v3/public-identities", headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        identities = await response.json()
                        return {
                            "success": True,
                            "identities": identities,
                            "count": len(identities),
                            "message": f"Retrieved {len(identities)} identities",
                        }
                    else:
                        error = await response.text()
                        return {"success": False, "error": f"Failed to get identities: {error}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to get identities: {str(e)}"}

    async def _get_identity(self, **kwargs) -> Dict[str, Any]:
        """Get specific identity by ID"""
        identity_id = kwargs.get('identity_id')

        if not identity_id:
            return {"success": False, "error": "identity_id is required"}

        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}/v3/public-identities/{identity_id}", headers=headers
                ) as response:
                    if response.status == 200:
                        identity = await response.json()
                        return {"success": True, "identity": identity, "message": "Identity retrieved successfully"}
                    else:
                        error = await response.text()
                        return {"success": False, "error": f"Failed to get identity: {error}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to get identity: {str(e)}"}

    async def _request_access(self, **kwargs) -> Dict[str, Any]:
        """Request access for an identity"""
        identity_id = kwargs.get('identity_id')
        access_profile_ids = kwargs.get('access_profile_ids', [])
        justification = kwargs.get('justification', 'Automated access request')

        if not identity_id:
            return {"success": False, "error": "identity_id is required"}

        if not access_profile_ids:
            return {"success": False, "error": "access_profile_ids is required"}

        try:
            headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}

            request_data = {
                "requestType": "GRANT_ACCESS",
                "identity": identity_id,
                "accessItems": [{"id": profile_id, "type": "ACCESS_PROFILE"} for profile_id in access_profile_ids],
                "justification": justification,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/v3/access-requests", headers=headers, json=request_data
                ) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        return {
                            "success": True,
                            "request_id": result.get("id"),
                            "status": result.get("status"),
                            "message": "Access request submitted successfully",
                        }
                    else:
                        error = await response.text()
                        return {"success": False, "error": f"Failed to request access: {error}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to request access: {str(e)}"}

    async def _get_accounts(self, **kwargs) -> Dict[str, Any]:
        """Get accounts from SailPoint"""
        limit = kwargs.get('limit', 10)
        filters = kwargs.get('filters', '')

        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            params = {"limit": limit}
            if filters:
                params["filters"] = filters

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}/v3/accounts", headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        accounts = await response.json()
                        return {
                            "success": True,
                            "accounts": accounts,
                            "count": len(accounts),
                            "message": f"Retrieved {len(accounts)} accounts",
                        }
                    else:
                        error = await response.text()
                        return {"success": False, "error": f"Failed to get accounts: {error}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to get accounts: {str(e)}"}

    async def _get_access_profiles(self, **kwargs) -> Dict[str, Any]:
        """Get access profiles from SailPoint"""
        limit = kwargs.get('limit', 10)
        filters = kwargs.get('filters', '')

        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            params = {"limit": limit}
            if filters:
                params["filters"] = filters

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}/v3/access-profiles", headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        profiles = await response.json()
                        return {
                            "success": True,
                            "access_profiles": profiles,
                            "count": len(profiles),
                            "message": f"Retrieved {len(profiles)} access profiles",
                        }
                    else:
                        error = await response.text()
                        return {"success": False, "error": f"Failed to get access profiles: {error}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to get access profiles: {str(e)}"}

    async def _launch_campaign(self, **kwargs) -> Dict[str, Any]:
        """Launch an access certification campaign"""
        campaign_name = kwargs.get('campaign_name')
        description = kwargs.get('description', '')
        identities = kwargs.get('identities', [])

        if not campaign_name:
            return {"success": False, "error": "campaign_name is required"}

        try:
            headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}

            campaign_data = {
                "name": campaign_name,
                "description": description,
                "type": "IDENTITY",
                "identities": identities,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/v3/campaigns", headers=headers, json=campaign_data
                ) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        return {
                            "success": True,
                            "campaign_id": result.get("id"),
                            "status": result.get("status"),
                            "message": "Campaign launched successfully",
                        }
                    else:
                        error = await response.text()
                        return {"success": False, "error": f"Failed to launch campaign: {error}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to launch campaign: {str(e)}"}


class MicrosoftEntraIntegration(BaseIntegration):
    """Microsoft Entra (Azure AD) integration using MSAL and Graph SDK"""

    REQUIRED_CONFIG_KEYS = ["tenant_id", "client_id", "client_secret"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "tenant_id": os.getenv("AZURE_TENANT_ID", ""),
            "client_id": os.getenv("AZURE_CLIENT_ID", ""),
            "client_secret": os.getenv("AZURE_CLIENT_SECRET", ""),
            "graph_endpoint": "https://graph.microsoft.com/v1.0",
            "timeout": 30,
        }
        super().__init__("microsoft_entra", {**default_config, **(config or {})})
        self.graph_client = None
        self.access_token = None

    async def initialize(self):
        """Initialize Microsoft Entra integration"""
        try:
            # Import libraries here to avoid import errors if not installed
            from azure.identity.aio import ClientSecretCredential
            from msgraph import GraphServiceClient

            if not all([self.config["tenant_id"], self.config["client_id"], self.config["client_secret"]]):
                logger.warning("⚠️  Microsoft Entra credentials not fully configured")
                self.enabled = False
                return

            # Initialize credentials and Graph client
            credential = ClientSecretCredential(
                tenant_id=self.config["tenant_id"],
                client_id=self.config["client_id"],
                client_secret=self.config["client_secret"],
            )

            self.graph_client = GraphServiceClient(
                credentials=credential, scopes=["https://graph.microsoft.com/.default"]
            )

            logger.info("✅ Microsoft Entra integration initialized")
        except ImportError as e:
            logger.error(f"❌ Required libraries not installed. Run: pip install azure-identity msgraph-sdk")
            self.enabled = False
        except Exception as e:
            logger.error(f"❌ Microsoft Entra initialization failed: {e}")
            self.enabled = False

    async def cleanup(self):
        """Cleanup Microsoft Entra resources"""
        self.graph_client = None
        self.access_token = None

    async def health_check(self) -> Dict[str, Any]:
        """Check Microsoft Entra connection health"""
        base_health = await super().health_check()

        if not self.graph_client:
            base_health.update({"status": "unhealthy", "error": "Graph client not initialized"})
            return base_health

        try:
            # Test connection by getting organization info
            organization = await self.graph_client.organization.get()
            if organization and organization.value:
                base_health.update(
                    {
                        "status": "healthy",
                        "tenant_id": self.config["tenant_id"],
                        "organization": organization.value[0].display_name if organization.value else "Unknown",
                        "connection": "active",
                    }
                )
            else:
                base_health.update({"status": "unhealthy", "error": "Unable to retrieve organization info"})
        except Exception as e:
            base_health.update({"status": "unhealthy", "error": str(e)})

        return base_health

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Microsoft Entra operations"""
        if not self.graph_client:
            return {"success": False, "error": "Microsoft Entra client not initialized"}

        try:
            if operation == "get_users":
                return await self._get_users(**kwargs)
            elif operation == "get_user":
                return await self._get_user(**kwargs)
            elif operation == "create_user":
                return await self._create_user(**kwargs)
            elif operation == "assign_role":
                return await self._assign_role(**kwargs)
            elif operation == "get_groups":
                return await self._get_groups(**kwargs)
            elif operation == "get_group_members":
                return await self._get_group_members(**kwargs)
            elif operation == "add_user_to_group":
                return await self._add_user_to_group(**kwargs)
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_users(self, **kwargs) -> Dict[str, Any]:
        """Get list of users from Azure AD"""
        limit = kwargs.get('limit', 10)
        filter_query = kwargs.get('filter', '')

        try:
            users_request = self.graph_client.users.get()
            if filter_query:
                users_request.filter = filter_query
            users_request.top = limit

            users_response = await users_request
            users = users_response.value if users_response else []

            users_data = []
            for user in users:
                user_data = {
                    'id': user.id,
                    'userPrincipalName': user.user_principal_name,
                    'displayName': user.display_name,
                    'givenName': user.given_name,
                    'surname': user.surname,
                    'mail': user.mail,
                    'accountEnabled': user.account_enabled,
                    'jobTitle': user.job_title,
                    'department': user.department,
                }
                users_data.append(user_data)

            return {
                "success": True,
                "users": users_data,
                "count": len(users_data),
                "message": f"Retrieved {len(users_data)} users",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get users: {str(e)}"}

    async def _get_user(self, **kwargs) -> Dict[str, Any]:
        """Get specific user by ID or UPN"""
        user_id = kwargs.get('user_id')
        upn = kwargs.get('user_principal_name')

        if not user_id and not upn:
            return {"success": False, "error": "user_id or user_principal_name is required"}

        try:
            identifier = user_id or upn
            user = await self.graph_client.users.by_user_id(identifier).get()

            if user:
                user_data = {
                    'id': user.id,
                    'userPrincipalName': user.user_principal_name,
                    'displayName': user.display_name,
                    'givenName': user.given_name,
                    'surname': user.surname,
                    'mail': user.mail,
                    'accountEnabled': user.account_enabled,
                    'jobTitle': user.job_title,
                    'department': user.department,
                    'officeLocation': user.office_location,
                    'mobilePhone': user.mobile_phone,
                }

                return {"success": True, "user": user_data, "message": "User retrieved successfully"}
            else:
                return {"success": False, "error": "User not found"}
        except Exception as e:
            return {"success": False, "error": f"Failed to get user: {str(e)}"}

    async def _create_user(self, **kwargs) -> Dict[str, Any]:
        """Create a new user in Azure AD"""
        required_fields = ['userPrincipalName', 'displayName', 'passwordProfile']

        # Validate required fields
        for field in required_fields:
            if field not in kwargs:
                return {"success": False, "error": f"Missing required field: {field}"}

        try:
            from msgraph.generated.models.password_profile import PasswordProfile
            from msgraph.generated.models.user import User

            user = User()
            user.user_principal_name = kwargs['userPrincipalName']
            user.display_name = kwargs['displayName']
            user.account_enabled = kwargs.get('accountEnabled', True)

            # Set password profile
            password_profile = PasswordProfile()
            if isinstance(kwargs['passwordProfile'], dict):
                password_profile.password = kwargs['passwordProfile'].get('password', 'TempPass123!')
                password_profile.force_change_password_next_sign_in = kwargs['passwordProfile'].get(
                    'forceChangePasswordNextSignIn', True
                )
            else:
                password_profile.password = 'TempPass123!'
                password_profile.force_change_password_next_sign_in = True

            user.password_profile = password_profile

            # Set optional fields
            if 'givenName' in kwargs:
                user.given_name = kwargs['givenName']
            if 'surname' in kwargs:
                user.surname = kwargs['surname']
            if 'jobTitle' in kwargs:
                user.job_title = kwargs['jobTitle']
            if 'department' in kwargs:
                user.department = kwargs['department']
            if 'mail' in kwargs:
                user.mail = kwargs['mail']

            created_user = await self.graph_client.users.post(user)

            return {
                "success": True,
                "user_id": created_user.id,
                "userPrincipalName": created_user.user_principal_name,
                "message": "User created successfully",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to create user: {str(e)}"}

    async def _get_groups(self, **kwargs) -> Dict[str, Any]:
        """Get list of groups from Azure AD"""
        limit = kwargs.get('limit', 10)
        filter_query = kwargs.get('filter', '')

        try:
            groups_request = self.graph_client.groups.get()
            if filter_query:
                groups_request.filter = filter_query
            groups_request.top = limit

            groups_response = await groups_request
            groups = groups_response.value if groups_response else []

            groups_data = []
            for group in groups:
                group_data = {
                    'id': group.id,
                    'displayName': group.display_name,
                    'description': group.description,
                    'groupTypes': group.group_types,
                    'mail': group.mail,
                    'mailEnabled': group.mail_enabled,
                    'securityEnabled': group.security_enabled,
                }
                groups_data.append(group_data)

            return {
                "success": True,
                "groups": groups_data,
                "count": len(groups_data),
                "message": f"Retrieved {len(groups_data)} groups",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get groups: {str(e)}"}

    async def _assign_role(self, **kwargs) -> Dict[str, Any]:
        """Assign directory role to a user"""
        user_id = kwargs.get('user_id')
        role_id = kwargs.get('role_id')

        if not user_id or not role_id:
            return {"success": False, "error": "user_id and role_id are required"}

        try:
            from msgraph.generated.models.directory_role_member_reference import DirectoryRoleMemberReference

            reference = DirectoryRoleMemberReference()
            reference.odata_id = f"https://graph.microsoft.com/v1.0/users/{user_id}"

            await self.graph_client.directory_roles.by_directory_role_id(role_id).members.ref.post(reference)

            return {"success": True, "user_id": user_id, "role_id": role_id, "message": "Role assigned successfully"}
        except Exception as e:
            return {"success": False, "error": f"Failed to assign role: {str(e)}"}

    async def _get_group_members(self, **kwargs) -> Dict[str, Any]:
        """Get members of a specific group"""
        group_id = kwargs.get('group_id')

        if not group_id:
            return {"success": False, "error": "group_id is required"}

        try:
            members = await self.graph_client.groups.by_group_id(group_id).members.get()

            members_data = []
            if members and members.value:
                for member in members.value:
                    member_data = {
                        'id': member.id,
                        'displayName': getattr(member, 'display_name', ''),
                        'userPrincipalName': getattr(member, 'user_principal_name', ''),
                        'odata_type': member.odata_type,
                    }
                    members_data.append(member_data)

            return {
                "success": True,
                "group_id": group_id,
                "members": members_data,
                "count": len(members_data),
                "message": f"Retrieved {len(members_data)} group members",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get group members: {str(e)}"}

    async def _add_user_to_group(self, **kwargs) -> Dict[str, Any]:
        """Add user to a group"""
        user_id = kwargs.get('user_id')
        group_id = kwargs.get('group_id')

        if not user_id or not group_id:
            return {"success": False, "error": "user_id and group_id are required"}

        try:
            from msgraph.generated.models.directory_object_reference import DirectoryObjectReference

            reference = DirectoryObjectReference()
            reference.odata_id = f"https://graph.microsoft.com/v1.0/users/{user_id}"

            await self.graph_client.groups.by_group_id(group_id).members.ref.post(reference)

            return {
                "success": True,
                "user_id": user_id,
                "group_id": group_id,
                "message": "User added to group successfully",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to add user to group: {str(e)}"}


class BritiveIntegration(BaseIntegration):
    """Britive cloud privileged access management integration"""

    REQUIRED_CONFIG_KEYS = ["tenant", "api_token"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "tenant": os.getenv("BRITIVE_TENANT", ""),
            "api_token": os.getenv("BRITIVE_API_TOKEN", ""),
            "base_url": os.getenv("BRITIVE_BASE_URL", ""),  # Optional, will be constructed from tenant
            "timeout": 30,
        }
        super().__init__("britive", {**default_config, **(config or {})})
        self.client = None

    async def initialize(self):
        """Initialize Britive integration"""
        try:
            # Import Britive SDK here to avoid import errors if not installed
            from britive import Britive

            if not all([self.config["tenant"], self.config["api_token"]]):
                logger.warning("⚠️  Britive credentials not fully configured")
                self.enabled = False
                return

            # Initialize Britive client
            self.client = Britive(
                tenant=self.config["tenant"], token=self.config["api_token"], base_url=self.config.get("base_url")
            )

            logger.info("✅ Britive integration initialized")
        except ImportError:
            logger.error("❌ Britive library not installed. Run: pip install britive")
            self.enabled = False
        except Exception as e:
            logger.error(f"❌ Britive initialization failed: {e}")
            self.enabled = False

    async def cleanup(self):
        """Cleanup Britive resources"""
        self.client = None

    async def health_check(self) -> Dict[str, Any]:
        """Check Britive connection health"""
        base_health = await super().health_check()

        if not self.client:
            base_health.update({"status": "unhealthy", "error": "Client not initialized"})
            return base_health

        try:
            # Test connection by getting user profiles
            profiles = self.client.profiles.list(limit=1)
            base_health.update(
                {
                    "status": "healthy",
                    "tenant": self.config["tenant"],
                    "connection": "active",
                    "profiles_accessible": len(profiles) >= 0,
                }
            )
        except Exception as e:
            base_health.update({"status": "unhealthy", "error": str(e)})

        return base_health

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Britive operations"""
        if not self.client:
            return {"success": False, "error": "Britive client not initialized"}

        try:
            if operation == "list_profiles":
                return await self._list_profiles(**kwargs)
            elif operation == "request_access":
                return await self._request_access(**kwargs)
            elif operation == "checkout_secret":
                return await self._checkout_secret(**kwargs)
            elif operation == "approve_request":
                return await self._approve_request(**kwargs)
            elif operation == "get_my_requests":
                return await self._get_my_requests(**kwargs)
            elif operation == "get_user_permissions":
                return await self._get_user_permissions(**kwargs)
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _list_profiles(self, **kwargs) -> Dict[str, Any]:
        """List available Britive profiles"""
        limit = kwargs.get('limit', 10)
        application_id = kwargs.get('application_id')

        try:
            if application_id:
                profiles = self.client.profiles.list(application_id=application_id, limit=limit)
            else:
                profiles = self.client.profiles.list(limit=limit)

            profiles_data = []
            for profile in profiles:
                profile_data = {
                    'id': profile.get('id'),
                    'name': profile.get('name'),
                    'description': profile.get('description'),
                    'applicationId': profile.get('applicationId'),
                    'status': profile.get('status'),
                    'type': profile.get('type'),
                }
                profiles_data.append(profile_data)

            return {
                "success": True,
                "profiles": profiles_data,
                "count": len(profiles_data),
                "message": f"Retrieved {len(profiles_data)} profiles",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to list profiles: {str(e)}"}

    async def _request_access(self, **kwargs) -> Dict[str, Any]:
        """Request access to a Britive profile"""
        profile_id = kwargs.get('profile_id')
        justification = kwargs.get('justification', 'Automated access request')
        duration = kwargs.get('duration', 60)  # minutes

        if not profile_id:
            return {"success": False, "error": "profile_id is required"}

        try:
            request_data = {'profileId': profile_id, 'justification': justification, 'duration': duration}

            request = self.client.access_requests.create(**request_data)

            return {
                "success": True,
                "request_id": request.get('id'),
                "status": request.get('status'),
                "profile_id": profile_id,
                "message": "Access request submitted successfully",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to request access: {str(e)}"}

    async def _checkout_secret(self, **kwargs) -> Dict[str, Any]:
        """Checkout a secret from Britive"""
        secret_id = kwargs.get('secret_id')
        profile_id = kwargs.get('profile_id')

        if not secret_id:
            return {"success": False, "error": "secret_id is required"}

        try:
            secret = self.client.secrets.checkout(secret_id, profile_id=profile_id)

            # Don't return the actual secret value in logs for security
            return {
                "success": True,
                "secret_id": secret_id,
                "checkout_time": secret.get('checkoutTime'),
                "expires_at": secret.get('expiresAt'),
                "message": "Secret checked out successfully",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to checkout secret: {str(e)}"}

    async def _approve_request(self, **kwargs) -> Dict[str, Any]:
        """Approve an access request"""
        request_id = kwargs.get('request_id')
        comment = kwargs.get('comment', 'Automated approval')

        if not request_id:
            return {"success": False, "error": "request_id is required"}

        try:
            approval = self.client.access_requests.approve(request_id, comment=comment)

            return {
                "success": True,
                "request_id": request_id,
                "status": approval.get('status'),
                "message": "Request approved successfully",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to approve request: {str(e)}"}

    async def _get_my_requests(self, **kwargs) -> Dict[str, Any]:
        """Get current user's access requests"""
        status = kwargs.get('status', 'pending')
        limit = kwargs.get('limit', 10)

        try:
            requests = self.client.access_requests.list(status=status, limit=limit)

            requests_data = []
            for request in requests:
                request_data = {
                    'id': request.get('id'),
                    'profileId': request.get('profileId'),
                    'status': request.get('status'),
                    'justification': request.get('justification'),
                    'requestedAt': request.get('requestedAt'),
                    'expiresAt': request.get('expiresAt'),
                }
                requests_data.append(request_data)

            return {
                "success": True,
                "requests": requests_data,
                "count": len(requests_data),
                "message": f"Retrieved {len(requests_data)} requests",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get requests: {str(e)}"}

    async def _get_user_permissions(self, **kwargs) -> Dict[str, Any]:
        """Get user permissions in Britive"""
        user_id = kwargs.get('user_id')

        try:
            if user_id:
                permissions = self.client.users.permissions(user_id)
            else:
                permissions = self.client.my.permissions()

            return {
                "success": True,
                "permissions": permissions,
                "count": len(permissions) if isinstance(permissions, list) else 0,
                "message": "Permissions retrieved successfully",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get permissions: {str(e)}"}
