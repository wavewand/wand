"""
Communication integrations for Wand
"""

import logging
import os
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import aiohttp

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class DiscordIntegration(BaseIntegration):
    """Discord bot and webhook integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "bot_token": os.getenv("DISCORD_BOT_TOKEN", ""),
            "webhook_url": os.getenv("DISCORD_WEBHOOK_URL", ""),
            "base_url": "https://discord.com/api/v10",
            "default_channel": None,
        }
        super().__init__("discord", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Discord integration"""
        if not self.config["bot_token"] and not self.config["webhook_url"]:
            logger.warning("⚠️  Discord bot token or webhook URL not configured")
        logger.info("✅ Discord integration initialized")

    async def cleanup(self):
        """Cleanup Discord resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Discord integration health"""
        if self.config["bot_token"]:
            try:
                headers = {"Authorization": f"Bot {self.config['bot_token']}"}
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.config['base_url']}/users/@me", headers=headers) as response:
                        if response.status == 200:
                            return {"status": "healthy", "mode": "bot"}
                        else:
                            return {"status": "unhealthy", "error": f"Bot API returned {response.status}"}
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        elif self.config["webhook_url"]:
            return {"status": "healthy", "mode": "webhook"}
        else:
            return {"status": "unhealthy", "error": "No token or webhook configured"}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Discord operations"""

        if operation == "send_message":
            return await self._send_message(**kwargs)
        elif operation == "send_webhook":
            return await self._send_webhook(**kwargs)
        elif operation == "create_channel":
            return await self._create_channel(**kwargs)
        elif operation == "get_guilds":
            return await self._get_guilds(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _send_message(self, channel_id: str, content: str, embed: Optional[Dict] = None) -> Dict[str, Any]:
        """Send message to Discord channel"""
        if not self.config["bot_token"]:
            return {"success": False, "error": "Bot token not configured"}

        headers = {"Authorization": f"Bot {self.config['bot_token']}", "Content-Type": "application/json"}

        payload = {"content": content}
        if embed:
            payload["embeds"] = [embed]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/channels/{channel_id}/messages", headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "message_id": result["id"],
                            "channel_id": channel_id,
                            "content": content,
                            "timestamp": result["timestamp"],
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_webhook(
        self,
        content: str,
        username: Optional[str] = None,
        avatar_url: Optional[str] = None,
        embeds: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Send message via Discord webhook"""
        if not self.config["webhook_url"]:
            return {"success": False, "error": "Webhook URL not configured"}

        payload = {"content": content}
        if username:
            payload["username"] = username
        if avatar_url:
            payload["avatar_url"] = avatar_url
        if embeds:
            payload["embeds"] = embeds

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config["webhook_url"], json=payload) as response:
                    if response.status == 204:
                        return {"success": True, "content": content, "webhook": "sent"}
                    else:
                        error = await response.text()
                        return {"success": False, "error": error}

        except Exception as e:
            return {"success": False, "error": str(e)}


class TelegramIntegration(BaseIntegration):
    """Telegram bot integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {"bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""), "base_url": "https://api.telegram.org/bot"}
        super().__init__("telegram", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Telegram integration"""
        if not self.config["bot_token"]:
            logger.warning("⚠️  Telegram bot token not configured")
        logger.info("✅ Telegram integration initialized")

    async def cleanup(self):
        """Cleanup Telegram resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Telegram bot health"""
        if not self.config["bot_token"]:
            return {"status": "unhealthy", "error": "Bot token not configured"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}{self.config['bot_token']}/getMe") as response:
                    if response.status == 200:
                        bot_info = await response.json()
                        return {
                            "status": "healthy",
                            "bot_username": bot_info["result"]["username"],
                            "bot_name": bot_info["result"]["first_name"],
                        }
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Telegram operations"""

        if operation == "send_message":
            return await self._send_message(**kwargs)
        elif operation == "send_photo":
            return await self._send_photo(**kwargs)
        elif operation == "get_updates":
            return await self._get_updates(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _send_message(self, chat_id: str, text: str, parse_mode: str = "Markdown") -> Dict[str, Any]:
        """Send message to Telegram chat"""
        if not self.config["bot_token"]:
            return {"success": False, "error": "Bot token not configured"}

        payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}{self.config['bot_token']}/sendMessage", json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "message_id": result["result"]["message_id"],
                            "chat_id": chat_id,
                            "text": text,
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("description", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}


class EmailIntegration(BaseIntegration):
    """Email SMTP/IMAP integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "smtp_host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "smtp_username": os.getenv("SMTP_USERNAME", ""),
            "smtp_password": os.getenv("SMTP_PASSWORD", ""),
            "from_email": os.getenv("FROM_EMAIL", ""),
            "use_tls": True,
        }
        super().__init__("email", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize email integration"""
        if not self.config["smtp_username"] or not self.config["smtp_password"]:
            logger.warning("⚠️  SMTP credentials not configured")
        logger.info("✅ Email integration initialized")

    async def cleanup(self):
        """Cleanup email resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check email configuration health"""
        if not self.config["smtp_username"] or not self.config["smtp_password"]:
            return {"status": "unhealthy", "error": "SMTP credentials not configured"}

        try:
            # Test SMTP connection
            import asyncio

            loop = asyncio.get_event_loop()

            def test_smtp():
                server = smtplib.SMTP(self.config["smtp_host"], self.config["smtp_port"])
                if self.config["use_tls"]:
                    server.starttls()
                server.login(self.config["smtp_username"], self.config["smtp_password"])
                server.quit()
                return True

            await loop.run_in_executor(None, test_smtp)
            return {"status": "healthy", "smtp_host": self.config["smtp_host"]}

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute email operations"""

        if operation == "send":
            return await self._send_email(**kwargs)
        elif operation == "send_html":
            return await self._send_html_email(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _send_email(
        self, to_email: str, subject: str, body: str, from_email: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send plain text email"""
        if not self.config["smtp_username"] or not self.config["smtp_password"]:
            return {"success": False, "error": "SMTP credentials not configured"}

        from_email = from_email or self.config["from_email"] or self.config["smtp_username"]

        try:
            import asyncio

            loop = asyncio.get_event_loop()

            def send_smtp():
                # Create message
                msg = MIMEText(body)
                msg['Subject'] = subject
                msg['From'] = from_email
                msg['To'] = to_email

                # Send email
                server = smtplib.SMTP(self.config["smtp_host"], self.config["smtp_port"])
                if self.config["use_tls"]:
                    server.starttls()
                server.login(self.config["smtp_username"], self.config["smtp_password"])
                server.send_message(msg)
                server.quit()

                return True

            await loop.run_in_executor(None, send_smtp)

            return {
                "success": True,
                "to_email": to_email,
                "from_email": from_email,
                "subject": subject,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


class CalendarIntegration(BaseIntegration):
    """Google Calendar integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {"credentials_file": os.getenv("GOOGLE_CREDENTIALS_FILE", ""), "calendar_id": "primary"}
        super().__init__("calendar", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize calendar integration"""
        if not self.config["credentials_file"]:
            logger.warning("⚠️  Google credentials file not configured")
        logger.info("✅ Calendar integration initialized")

    async def cleanup(self):
        """Cleanup calendar resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check calendar integration health"""
        if not self.config["credentials_file"]:
            return {"status": "unhealthy", "error": "Google credentials not configured"}
        return {"status": "healthy"}  # Simplified for demo

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute calendar operations"""

        if operation == "create_event":
            return await self._create_event(**kwargs)
        elif operation == "list_events":
            return await self._list_events(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _create_event(
        self, title: str, start_time: str, end_time: str, description: str = "", location: str = ""
    ) -> Dict[str, Any]:
        """Create calendar event"""
        # Simplified implementation - would use Google Calendar API
        return {
            "success": True,
            "event_id": f"event_{int(datetime.now().timestamp())}",
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
            "description": description,
            "location": location,
        }


class NotionIntegration(BaseIntegration):
    """Notion API integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_token": os.getenv("NOTION_TOKEN", ""),
            "base_url": "https://api.notion.com/v1",
            "version": "2022-06-28",
        }
        super().__init__("notion", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Notion integration"""
        if not self.config["api_token"]:
            logger.warning("⚠️  Notion API token not configured")
        logger.info("✅ Notion integration initialized")

    async def cleanup(self):
        """Cleanup Notion resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Notion API health"""
        if not self.config["api_token"]:
            return {"status": "unhealthy", "error": "API token not configured"}

        try:
            headers = {"Authorization": f"Bearer {self.config['api_token']}", "Notion-Version": self.config["version"]}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/users/me", headers=headers) as response:
                    if response.status == 200:
                        return {"status": "healthy"}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Notion operations"""

        if operation == "create_page":
            return await self._create_page(**kwargs)
        elif operation == "update_page":
            return await self._update_page(**kwargs)
        elif operation == "query_database":
            return await self._query_database(**kwargs)
        elif operation == "create_database":
            return await self._create_database(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _create_page(self, parent_id: str, title: str, content: List[Dict] = None) -> Dict[str, Any]:
        """Create new Notion page"""
        if not self.config["api_token"]:
            return {"success": False, "error": "API token not configured"}

        headers = {
            "Authorization": f"Bearer {self.config['api_token']}",
            "Content-Type": "application/json",
            "Notion-Version": self.config["version"],
        }

        payload = {"parent": {"page_id": parent_id}, "properties": {"title": {"title": [{"text": {"content": title}}]}}}

        if content:
            payload["children"] = content

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.config['base_url']}/pages", headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "page_id": result["id"],
                            "title": title,
                            "url": result["url"],
                            "created_time": result["created_time"],
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _query_database(
        self,
        database_id: str,
        filter_conditions: Optional[Dict] = None,
        sorts: Optional[List[Dict]] = None,
        page_size: int = 100,
    ) -> Dict[str, Any]:
        """Query Notion database"""
        if not self.config["api_token"]:
            return {"success": False, "error": "API token not configured"}

        headers = {
            "Authorization": f"Bearer {self.config['api_token']}",
            "Content-Type": "application/json",
            "Notion-Version": self.config["version"],
        }

        payload = {"page_size": page_size}
        if filter_conditions:
            payload["filter"] = filter_conditions
        if sorts:
            payload["sorts"] = sorts

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/databases/{database_id}/query", headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        pages = []
                        for page in result.get("results", []):
                            pages.append(
                                {
                                    "id": page["id"],
                                    "url": page["url"],
                                    "created_time": page["created_time"],
                                    "last_edited_time": page["last_edited_time"],
                                    "properties": page["properties"],
                                }
                            )

                        return {
                            "success": True,
                            "results": pages,
                            "has_more": result.get("has_more", False),
                            "total_results": len(pages),
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}
