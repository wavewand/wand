"""
Microsoft Teams communication integration for Wand
Uses webhook-based approach for simple message sending
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class MicrosoftTeamsIntegration(BaseIntegration):
    """Microsoft Teams integration using incoming webhooks"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "default_webhook_url": os.getenv("TEAMS_WEBHOOK_URL", ""),
            "timeout": 30,
            "rate_limit_per_second": 4,  # Teams allows max 4 requests per second
        }
        super().__init__("microsoft_teams", {**default_config, **(config or {})})
        self.webhook_urls = {}  # Store multiple webhook URLs for different channels

    async def initialize(self):
        """Initialize Microsoft Teams integration"""
        try:
            # Load webhook URLs from config or environment
            if self.config.get("default_webhook_url"):
                self.webhook_urls["default"] = self.config["default_webhook_url"]

            # Load additional webhooks from environment variables
            # Format: TEAMS_WEBHOOK_CHANNEL_NAME=url
            for key, value in os.environ.items():
                if key.startswith("TEAMS_WEBHOOK_") and key != "TEAMS_WEBHOOK_URL":
                    channel_name = key.replace("TEAMS_WEBHOOK_", "").lower()
                    self.webhook_urls[channel_name] = value

            if not self.webhook_urls:
                logger.warning("⚠️  No Teams webhook URLs configured")
                self.enabled = False
                return

            logger.info(f"✅ Microsoft Teams integration initialized with {len(self.webhook_urls)} webhook(s)")
        except Exception as e:
            logger.error(f"❌ Microsoft Teams initialization failed: {e}")
            self.enabled = False

    async def cleanup(self):
        """Cleanup Microsoft Teams resources"""
        self.webhook_urls.clear()

    async def health_check(self) -> Dict[str, Any]:
        """Check Microsoft Teams webhook health"""
        base_health = await super().health_check()

        if not self.webhook_urls:
            base_health.update({"status": "unhealthy", "error": "No webhook URLs configured"})
            return base_health

        # Test default webhook if available
        default_url = self.webhook_urls.get("default")
        if default_url:
            try:
                # Send a minimal test message to verify webhook is accessible
                test_payload = {
                    "text": "Health check - please ignore",
                    "@type": "MessageCard",
                    "@context": "https://schema.org/extensions",
                    "summary": "Health Check",
                    "themeColor": "28a745",
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        default_url, json=test_payload, timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            base_health.update(
                                {
                                    "status": "healthy",
                                    "webhooks_configured": len(self.webhook_urls),
                                    "default_webhook": "accessible",
                                }
                            )
                        else:
                            base_health.update(
                                {
                                    "status": "degraded",
                                    "error": f"Default webhook returned {response.status}",
                                    "webhooks_configured": len(self.webhook_urls),
                                }
                            )
            except Exception as e:
                base_health.update(
                    {
                        "status": "degraded",
                        "error": f"Webhook test failed: {str(e)}",
                        "webhooks_configured": len(self.webhook_urls),
                    }
                )
        else:
            base_health.update(
                {
                    "status": "healthy",
                    "webhooks_configured": len(self.webhook_urls),
                    "note": "No default webhook to test",
                }
            )

        return base_health

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Microsoft Teams operations"""
        if not self.webhook_urls:
            return {"success": False, "error": "No webhook URLs configured"}

        try:
            if operation == "send_message":
                return await self._send_message(**kwargs)
            elif operation == "send_card":
                return await self._send_card(**kwargs)
            elif operation == "send_notification":
                return await self._send_notification(**kwargs)
            elif operation == "list_webhooks":
                return await self._list_webhooks(**kwargs)
            elif operation == "add_webhook":
                return await self._add_webhook(**kwargs)
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _send_message(self, **kwargs) -> Dict[str, Any]:
        """Send a simple text message to Teams channel"""
        message = kwargs.get('message', kwargs.get('text', ''))
        channel = kwargs.get('channel', 'default')
        webhook_url = kwargs.get('webhook_url')

        if not message:
            return {"success": False, "error": "message or text is required"}

        # Determine which webhook to use
        if webhook_url:
            url = webhook_url
        elif channel in self.webhook_urls:
            url = self.webhook_urls[channel]
        else:
            return {"success": False, "error": f"No webhook URL found for channel '{channel}'"}

        try:
            payload = {"text": message}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=self.config.get("timeout", 30))
                ) as response:
                    if response.status == 200:
                        return {
                            "success": True,
                            "channel": channel,
                            "message": "Message sent successfully",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"Teams API returned {response.status}: {error_text}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to send message: {str(e)}"}

    async def _send_card(self, **kwargs) -> Dict[str, Any]:
        """Send an adaptive card or message card to Teams"""
        card_data = kwargs.get('card')
        title = kwargs.get('title', '')
        summary = kwargs.get('summary', title or 'Notification')
        channel = kwargs.get('channel', 'default')
        webhook_url = kwargs.get('webhook_url')
        theme_color = kwargs.get('theme_color', '0078d4')  # Microsoft blue

        # Determine which webhook to use
        if webhook_url:
            url = webhook_url
        elif channel in self.webhook_urls:
            url = self.webhook_urls[channel]
        else:
            return {"success": False, "error": f"No webhook URL found for channel '{channel}'"}

        try:
            if card_data:
                # Use provided card data
                if isinstance(card_data, dict):
                    payload = card_data
                else:
                    # Assume it's a JSON string
                    import json

                    payload = json.loads(card_data)
            else:
                # Create a simple message card
                sections = kwargs.get('sections', [])
                facts = kwargs.get('facts', [])
                actions = kwargs.get('actions', [])

                payload = {
                    "@type": "MessageCard",
                    "@context": "https://schema.org/extensions",
                    "summary": summary,
                    "themeColor": theme_color,
                }

                if title:
                    payload["title"] = title

                if sections:
                    payload["sections"] = sections
                elif facts:
                    payload["sections"] = [{"facts": facts}]

                if actions:
                    payload["potentialAction"] = actions

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=self.config.get("timeout", 30))
                ) as response:
                    if response.status == 200:
                        return {
                            "success": True,
                            "channel": channel,
                            "message": "Card sent successfully",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"Teams API returned {response.status}: {error_text}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to send card: {str(e)}"}

    async def _send_notification(self, **kwargs) -> Dict[str, Any]:
        """Send a formatted notification with status and details"""
        title = kwargs.get('title', 'Notification')
        message = kwargs.get('message', '')
        status = kwargs.get('status', 'info')  # success, warning, error, info
        details = kwargs.get('details', [])
        channel = kwargs.get('channel', 'default')
        webhook_url = kwargs.get('webhook_url')
        include_timestamp = kwargs.get('include_timestamp', True)

        # Status color mapping
        status_colors = {
            'success': '28a745',  # Green
            'warning': 'ffc107',  # Yellow
            'error': 'dc3545',  # Red
            'info': '17a2b8',  # Blue
            'default': '6c757d',  # Gray
        }

        theme_color = status_colors.get(status, status_colors['default'])

        # Determine which webhook to use
        if webhook_url:
            url = webhook_url
        elif channel in self.webhook_urls:
            url = self.webhook_urls[channel]
        else:
            return {"success": False, "error": f"No webhook URL found for channel '{channel}'"}

        try:
            # Build the message card
            payload = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": title,
                "title": title,
                "themeColor": theme_color,
            }

            sections = []

            if message:
                sections.append({"text": message})

            # Add details as facts
            if details:
                facts = []
                for detail in details:
                    if isinstance(detail, dict) and 'name' in detail and 'value' in detail:
                        facts.append(detail)
                    elif isinstance(detail, str):
                        facts.append({"name": "Detail", "value": detail})

                if facts:
                    sections.append({"facts": facts})

            # Add timestamp if requested
            if include_timestamp:
                timestamp_fact = {
                    "name": "Timestamp",
                    "value": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                if sections and "facts" in sections[-1]:
                    sections[-1]["facts"].append(timestamp_fact)
                else:
                    sections.append({"facts": [timestamp_fact]})

            if sections:
                payload["sections"] = sections

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=self.config.get("timeout", 30))
                ) as response:
                    if response.status == 200:
                        return {
                            "success": True,
                            "channel": channel,
                            "status": status,
                            "message": "Notification sent successfully",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"Teams API returned {response.status}: {error_text}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to send notification: {str(e)}"}

    async def _list_webhooks(self, **kwargs) -> Dict[str, Any]:
        """List configured webhook channels"""
        try:
            webhook_list = []
            for channel_name, url in self.webhook_urls.items():
                # Don't expose the full URL for security, just show partial
                masked_url = url[:30] + "..." if len(url) > 30 else url
                webhook_list.append({"channel": channel_name, "url_preview": masked_url, "configured": True})

            return {
                "success": True,
                "webhooks": webhook_list,
                "count": len(webhook_list),
                "message": f"Found {len(webhook_list)} configured webhook(s)",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to list webhooks: {str(e)}"}

    async def _add_webhook(self, **kwargs) -> Dict[str, Any]:
        """Add a new webhook URL for a channel"""
        channel = kwargs.get('channel')
        webhook_url = kwargs.get('webhook_url')

        if not channel:
            return {"success": False, "error": "channel name is required"}

        if not webhook_url:
            return {"success": False, "error": "webhook_url is required"}

        if not webhook_url.startswith('https://'):
            return {"success": False, "error": "webhook_url must be a valid HTTPS URL"}

        try:
            # Test the webhook before adding it
            test_payload = {
                "text": f"Webhook test for channel '{channel}' - configuration successful",
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": "Webhook Configuration Test",
                "themeColor": "28a745",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url, json=test_payload, timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        # Test successful, add to our webhook list
                        self.webhook_urls[channel] = webhook_url

                        return {
                            "success": True,
                            "channel": channel,
                            "message": "Webhook added and tested successfully",
                            "test_sent": True,
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Webhook test failed with status {response.status}: {error_text}",
                        }
        except Exception as e:
            return {"success": False, "error": f"Failed to add webhook: {str(e)}"}


class TeamsMessageBuilder:
    """Helper class to build Teams message cards"""

    def __init__(self):
        self.card = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": "",
            "themeColor": "0078d4",
        }
        self.sections = []
        self.actions = []

    def set_title(self, title: str):
        """Set the card title"""
        self.card["title"] = title
        if not self.card["summary"]:
            self.card["summary"] = title
        return self

    def set_summary(self, summary: str):
        """Set the card summary"""
        self.card["summary"] = summary
        return self

    def set_theme_color(self, color: str):
        """Set the theme color (hex color without #)"""
        self.card["themeColor"] = color.lstrip('#')
        return self

    def add_section(
        self,
        text: str = None,
        facts: List[Dict[str, str]] = None,
        activity_title: str = None,
        activity_subtitle: str = None,
        activity_image: str = None,
    ):
        """Add a section to the card"""
        section = {}

        if activity_title:
            section["activityTitle"] = activity_title
        if activity_subtitle:
            section["activitySubtitle"] = activity_subtitle
        if activity_image:
            section["activityImage"] = activity_image
        if text:
            section["text"] = text
        if facts:
            section["facts"] = facts

        self.sections.append(section)
        return self

    def add_fact(self, name: str, value: str, section_index: int = -1):
        """Add a fact to a specific section (default: last section)"""
        if not self.sections:
            self.add_section()

        section = self.sections[section_index]
        if "facts" not in section:
            section["facts"] = []

        section["facts"].append({"name": name, "value": value})
        return self

    def add_action(self, name: str, target: str, action_type: str = "OpenUri"):
        """Add an action button"""
        action = {"@type": action_type, "name": name}

        if action_type == "OpenUri":
            action["targets"] = [{"os": "default", "uri": target}]
        else:
            action["target"] = target

        self.actions.append(action)
        return self

    def build(self) -> Dict[str, Any]:
        """Build and return the complete message card"""
        if self.sections:
            self.card["sections"] = self.sections
        if self.actions:
            self.card["potentialAction"] = self.actions

        return self.card


# Convenience functions for common use cases
def create_simple_message(text: str) -> Dict[str, Any]:
    """Create a simple text message"""
    return {"text": text}


def create_alert_card(title: str, message: str, alert_type: str = "info") -> Dict[str, Any]:
    """Create an alert-style message card"""
    colors = {"success": "28a745", "info": "17a2b8", "warning": "ffc107", "error": "dc3545"}

    builder = TeamsMessageBuilder()
    builder.set_title(f"🔔 {title}")
    builder.set_theme_color(colors.get(alert_type, colors["info"]))
    builder.add_section(text=message)
    builder.add_fact("Alert Type", alert_type.title())
    builder.add_fact("Timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

    return builder.build()


def create_status_card(service: str, status: str, details: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Create a service status message card"""
    status_colors = {"healthy": "28a745", "degraded": "ffc107", "unhealthy": "dc3545", "unknown": "6c757d"}

    status_emojis = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌", "unknown": "❓"}

    emoji = status_emojis.get(status.lower(), "")
    color = status_colors.get(status.lower(), status_colors["unknown"])

    builder = TeamsMessageBuilder()
    builder.set_title(f"{emoji} {service} Status")
    builder.set_theme_color(color)
    builder.add_fact("Service", service)
    builder.add_fact("Status", status.title())
    builder.add_fact("Checked", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

    if details:
        for detail in details:
            if isinstance(detail, dict) and "name" in detail and "value" in detail:
                builder.add_fact(detail["name"], detail["value"])

    return builder.build()
