"""
Project management integrations for Wand
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


class JiraIntegration(BaseIntegration):
    """Atlassian Jira project management integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "server_url": os.getenv("JIRA_SERVER_URL", ""),
            "username": os.getenv("JIRA_USERNAME", ""),
            "api_token": os.getenv("JIRA_API_TOKEN", ""),
            "project_key": os.getenv("JIRA_PROJECT_KEY", ""),
            "api_version": "3",
        }
        super().__init__("jira", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Jira integration"""
        if not all([self.config["server_url"], self.config["username"], self.config["api_token"]]):
            logger.warning("⚠️  Jira credentials not fully configured")
        logger.info("✅ Jira integration initialized")

    async def cleanup(self):
        """Cleanup Jira resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Jira API health"""
        if not all([self.config["server_url"], self.config["username"], self.config["api_token"]]):
            return {"status": "unhealthy", "error": "Credentials not configured"}

        try:
            headers = await self._get_auth_headers()
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['server_url']}/rest/api/{self.config['api_version']}/myself", headers=headers
                ) as response:
                    if response.status == 200:
                        user_data = await response.json()
                        return {
                            "status": "healthy",
                            "user": user_data.get("displayName"),
                            "server": self.config["server_url"],
                        }
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Jira API"""
        credentials = base64.b64encode(f"{self.config['username']}:{self.config['api_token']}".encode()).decode()

        return {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Jira operations"""

        if operation == "create_issue":
            return await self._create_issue(**kwargs)
        elif operation == "get_issue":
            return await self._get_issue(**kwargs)
        elif operation == "search_issues":
            return await self._search_issues(**kwargs)
        elif operation == "update_issue":
            return await self._update_issue(**kwargs)
        elif operation == "get_projects":
            return await self._get_projects(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _create_issue(
        self,
        summary: str,
        issue_type: str = "Task",
        description: Optional[str] = None,
        project_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a Jira issue"""
        if not all([self.config["server_url"], self.config["username"], self.config["api_token"]]):
            return {"success": False, "error": "Credentials not configured"}

        project = project_key or self.config["project_key"]
        if not project:
            return {"success": False, "error": "Project key required"}

        headers = await self._get_auth_headers()

        issue_data = {"fields": {"project": {"key": project}, "summary": summary, "issuetype": {"name": issue_type}}}

        if description:
            issue_data["fields"]["description"] = {
                "type": "doc",
                "version": 1,
                "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}],
            }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['server_url']}/rest/api/{self.config['api_version']}/issue",
                    headers=headers,
                    json=issue_data,
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        return {
                            "success": True,
                            "issue_key": result.get("key"),
                            "issue_id": result.get("id"),
                            "summary": summary,
                            "url": f"{self.config['server_url']}/browse/{result.get('key')}",
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("errorMessages", ["Unknown error"])}

        except Exception as e:
            return {"success": False, "error": str(e)}


class AsanaIntegration(BaseIntegration):
    """Asana task management integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "access_token": os.getenv("ASANA_ACCESS_TOKEN", ""),
            "workspace_gid": os.getenv("ASANA_WORKSPACE_GID", ""),
            "base_url": "https://app.asana.com/api/1.0",
        }
        super().__init__("asana", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Asana integration"""
        if not self.config["access_token"]:
            logger.warning("⚠️  Asana access token not configured")
        logger.info("✅ Asana integration initialized")

    async def cleanup(self):
        """Cleanup Asana resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Asana API health"""
        if not self.config["access_token"]:
            return {"status": "unhealthy", "error": "Access token not configured"}

        try:
            headers = {"Authorization": f"Bearer {self.config['access_token']}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/users/me", headers=headers) as response:
                    if response.status == 200:
                        user_data = await response.json()
                        return {
                            "status": "healthy",
                            "user": user_data.get("data", {}).get("name"),
                            "email": user_data.get("data", {}).get("email"),
                        }
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Asana operations"""

        if operation == "create_task":
            return await self._create_task(**kwargs)
        elif operation == "get_tasks":
            return await self._get_tasks(**kwargs)
        elif operation == "update_task":
            return await self._update_task(**kwargs)
        elif operation == "get_projects":
            return await self._get_projects(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _create_task(
        self, name: str, notes: Optional[str] = None, project_gid: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a task in Asana"""
        if not self.config["access_token"]:
            return {"success": False, "error": "Access token not configured"}

        headers = {"Authorization": f"Bearer {self.config['access_token']}", "Content-Type": "application/json"}

        task_data = {"data": {"name": name}}

        if notes:
            task_data["data"]["notes"] = notes
        if project_gid:
            task_data["data"]["projects"] = [project_gid]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/tasks", headers=headers, json=task_data
                ) as response:
                    if response.status == 201:
                        result = await response.json()
                        task = result.get("data", {})
                        return {
                            "success": True,
                            "task_gid": task.get("gid"),
                            "name": name,
                            "permalink_url": task.get("permalink_url"),
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("errors", ["Unknown error"])}

        except Exception as e:
            return {"success": False, "error": str(e)}


class TrelloIntegration(BaseIntegration):
    """Trello board management integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_key": os.getenv("TRELLO_API_KEY", ""),
            "api_token": os.getenv("TRELLO_API_TOKEN", ""),
            "base_url": "https://api.trello.com/1",
        }
        super().__init__("trello", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Trello integration"""
        if not all([self.config["api_key"], self.config["api_token"]]):
            logger.warning("⚠️  Trello API credentials not configured")
        logger.info("✅ Trello integration initialized")

    async def cleanup(self):
        """Cleanup Trello resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Trello API health"""
        if not all([self.config["api_key"], self.config["api_token"]]):
            return {"status": "unhealthy", "error": "API credentials not configured"}

        try:
            params = {"key": self.config["api_key"], "token": self.config["api_token"]}

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/members/me", params=params) as response:
                    if response.status == 200:
                        user_data = await response.json()
                        return {
                            "status": "healthy",
                            "username": user_data.get("username"),
                            "full_name": user_data.get("fullName"),
                        }
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Trello operations"""

        if operation == "create_card":
            return await self._create_card(**kwargs)
        elif operation == "get_boards":
            return await self._get_boards(**kwargs)
        elif operation == "get_lists":
            return await self._get_lists(**kwargs)
        elif operation == "move_card":
            return await self._move_card(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _create_card(self, name: str, list_id: str, desc: Optional[str] = None) -> Dict[str, Any]:
        """Create a card in Trello"""
        if not all([self.config["api_key"], self.config["api_token"]]):
            return {"success": False, "error": "API credentials not configured"}

        params = {"key": self.config["api_key"], "token": self.config["api_token"], "name": name, "idList": list_id}

        if desc:
            params["desc"] = desc

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.config['base_url']}/cards", params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "card_id": result.get("id"),
                            "name": name,
                            "url": result.get("url"),
                            "short_url": result.get("shortUrl"),
                        }
                    else:
                        return {"success": False, "error": f"API returned {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


class LinearIntegration(BaseIntegration):
    """Linear issue tracking integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {"api_key": os.getenv("LINEAR_API_KEY", ""), "base_url": "https://api.linear.app/graphql"}
        super().__init__("linear", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Linear integration"""
        if not self.config["api_key"]:
            logger.warning("⚠️  Linear API key not configured")
        logger.info("✅ Linear integration initialized")

    async def cleanup(self):
        """Cleanup Linear resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Linear API health"""
        if not self.config["api_key"]:
            return {"status": "unhealthy", "error": "API key not configured"}

        try:
            headers = {"Authorization": self.config["api_key"], "Content-Type": "application/json"}

            query = {"query": "{ viewer { id name email } }"}

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["base_url"], headers=headers, json=query) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "errors" not in result:
                            viewer = result.get("data", {}).get("viewer", {})
                            return {"status": "healthy", "user": viewer.get("name"), "email": viewer.get("email")}
                        else:
                            return {"status": "unhealthy", "error": result.get("errors")}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Linear operations"""

        if operation == "create_issue":
            return await self._create_issue(**kwargs)
        elif operation == "get_issues":
            return await self._get_issues(**kwargs)
        elif operation == "update_issue":
            return await self._update_issue(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _create_issue(
        self, title: str, description: Optional[str] = None, team_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create an issue in Linear"""
        if not self.config["api_key"]:
            return {"success": False, "error": "API key not configured"}

        headers = {"Authorization": self.config["api_key"], "Content-Type": "application/json"}

        mutation = """
        mutation IssueCreate($input: IssueCreateInput!) {
            issueCreate(input: $input) {
                success
                issue {
                    id
                    identifier
                    title
                    url
                }
            }
        }
        """

        variables = {"input": {"title": title}}

        if description:
            variables["input"]["description"] = description
        if team_id:
            variables["input"]["teamId"] = team_id

        query = {"query": mutation, "variables": variables}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["base_url"], headers=headers, json=query) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "errors" not in result:
                            issue_data = result.get("data", {}).get("issueCreate", {})
                            if issue_data.get("success"):
                                issue = issue_data.get("issue", {})
                                return {
                                    "success": True,
                                    "issue_id": issue.get("id"),
                                    "identifier": issue.get("identifier"),
                                    "title": title,
                                    "url": issue.get("url"),
                                }
                            else:
                                return {"success": False, "error": "Issue creation failed"}
                        else:
                            return {"success": False, "error": result.get("errors")}
                    else:
                        return {"success": False, "error": f"API returned {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


class MondayIntegration(BaseIntegration):
    """Monday.com work management integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {"api_token": os.getenv("MONDAY_API_TOKEN", ""), "base_url": "https://api.monday.com/v2"}
        super().__init__("monday", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Monday.com integration"""
        if not self.config["api_token"]:
            logger.warning("⚠️  Monday.com API token not configured")
        logger.info("✅ Monday.com integration initialized")

    async def cleanup(self):
        """Cleanup Monday.com resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Monday.com API health"""
        if not self.config["api_token"]:
            return {"status": "unhealthy", "error": "API token not configured"}

        try:
            headers = {"Authorization": self.config["api_token"], "Content-Type": "application/json"}

            query = {"query": "{ me { id name email } }"}

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["base_url"], headers=headers, json=query) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "errors" not in result:
                            user = result.get("data", {}).get("me", {})
                            return {"status": "healthy", "user": user.get("name"), "email": user.get("email")}
                        else:
                            return {"status": "unhealthy", "error": result.get("errors")}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Monday.com operations"""

        if operation == "create_item":
            return await self._create_item(**kwargs)
        elif operation == "get_boards":
            return await self._get_boards(**kwargs)
        elif operation == "update_item":
            return await self._update_item(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _create_item(
        self, board_id: str, item_name: str, column_values: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create an item in Monday.com"""
        if not self.config["api_token"]:
            return {"success": False, "error": "API token not configured"}

        headers = {"Authorization": self.config["api_token"], "Content-Type": "application/json"}

        mutation = """
        mutation ($boardId: Int!, $itemName: String!, $columnValues: JSON) {
            create_item (board_id: $boardId, item_name: $itemName, column_values: $columnValues) {
                id
                name
                url
            }
        }
        """

        variables = {"boardId": int(board_id), "itemName": item_name}

        if column_values:
            variables["columnValues"] = json.dumps(column_values)

        query = {"query": mutation, "variables": variables}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["base_url"], headers=headers, json=query) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "errors" not in result:
                            item = result.get("data", {}).get("create_item", {})
                            return {
                                "success": True,
                                "item_id": item.get("id"),
                                "name": item_name,
                                "url": item.get("url"),
                            }
                        else:
                            return {"success": False, "error": result.get("errors")}
                    else:
                        return {"success": False, "error": f"API returned {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}
