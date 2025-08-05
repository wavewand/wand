"""
Gaming and streaming integrations for Wand
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class SteamIntegration(BaseIntegration):
    """Steam gaming platform integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_key": os.getenv("STEAM_API_KEY", ""),
            "base_url": "https://api.steampowered.com",
            "store_url": "https://store.steampowered.com/api",
        }
        super().__init__("steam", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Steam integration"""
        if not self.config["api_key"]:
            logger.warning("⚠️  Steam API key not configured")
        logger.info("✅ Steam integration initialized")

    async def cleanup(self):
        """Cleanup Steam resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Steam API health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/ISteamWebAPIUtil/GetServerInfo/v1/") as response:
                    if response.status == 200:
                        return {"status": "healthy", "steam_api": "operational"}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Steam operations"""

        if operation == "get_player_summaries":
            return await self._get_player_summaries(**kwargs)
        elif operation == "get_owned_games":
            return await self._get_owned_games(**kwargs)
        elif operation == "get_friends_list":
            return await self._get_friends_list(**kwargs)
        elif operation == "get_app_details":
            return await self._get_app_details(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _get_player_summaries(self, steam_ids: List[str]) -> Dict[str, Any]:
        """Get Steam player summaries"""
        if not self.config["api_key"]:
            return {"success": False, "error": "Steam API key not configured"}

        if not steam_ids:
            return {"success": False, "error": "Steam IDs required"}

        try:
            params = {"key": self.config["api_key"], "steamids": ",".join(steam_ids)}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}/ISteamUser/GetPlayerSummaries/v2/", params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        players = []

                        for player in data.get("response", {}).get("players", []):
                            players.append(
                                {
                                    "steam_id": player.get("steamid"),
                                    "persona_name": player.get("personaname"),
                                    "profile_url": player.get("profileurl"),
                                    "avatar": player.get("avatarfull"),
                                    "persona_state": player.get("personastate"),
                                    "real_name": player.get("realname"),
                                    "country_code": player.get("loccountrycode"),
                                    "time_created": player.get("timecreated"),
                                }
                            )

                        return {"success": True, "players": players, "total": len(players)}
                    else:
                        return {"success": False, "error": f"Steam API returned {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


class TwitchIntegration(BaseIntegration):
    """Twitch streaming platform integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "client_id": os.getenv("TWITCH_CLIENT_ID", ""),
            "client_secret": os.getenv("TWITCH_CLIENT_SECRET", ""),
            "base_url": "https://api.twitch.tv/helix",
        }
        super().__init__("twitch", {**default_config, **(config or {})})
        self.access_token = None

    async def initialize(self):
        """Initialize Twitch integration"""
        if not self.config["client_id"] or not self.config["client_secret"]:
            logger.warning("⚠️  Twitch client credentials not configured")
        else:
            await self._get_app_access_token()
            logger.info("✅ Twitch integration initialized")

    async def cleanup(self):
        """Cleanup Twitch resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Twitch API health"""
        if not self.access_token:
            return {"status": "unhealthy", "error": "Not authenticated"}

        try:
            headers = {"Client-ID": self.config["client_id"], "Authorization": f"Bearer {self.access_token}"}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}/games/top", headers=headers, params={"first": 1}
                ) as response:
                    if response.status == 200:
                        return {"status": "healthy", "twitch_api": "operational"}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _get_app_access_token(self):
        """Get Twitch app access token"""
        data = {
            "client_id": self.config["client_id"],
            "client_secret": self.config["client_secret"],
            "grant_type": "client_credentials",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://id.twitch.tv/oauth2/token", data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.access_token = result["access_token"]
                    else:
                        logger.error(f"Twitch token request failed: {response.status}")
        except Exception as e:
            logger.error(f"Twitch token request error: {e}")

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Twitch operations"""

        if operation == "get_streams":
            return await self._get_streams(**kwargs)
        elif operation == "get_users":
            return await self._get_users(**kwargs)
        elif operation == "get_games":
            return await self._get_games(**kwargs)
        elif operation == "search_channels":
            return await self._search_channels(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _get_streams(
        self, game_id: Optional[str] = None, user_login: Optional[str] = None, first: int = 20
    ) -> Dict[str, Any]:
        """Get live streams"""
        if not self.access_token:
            return {"success": False, "error": "Not authenticated"}

        headers = {"Client-ID": self.config["client_id"], "Authorization": f"Bearer {self.access_token}"}

        params = {"first": first}
        if game_id:
            params["game_id"] = game_id
        if user_login:
            params["user_login"] = user_login

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}/streams", headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        streams = []

                        for stream in data.get("data", []):
                            streams.append(
                                {
                                    "id": stream.get("id"),
                                    "user_id": stream.get("user_id"),
                                    "user_login": stream.get("user_login"),
                                    "user_name": stream.get("user_name"),
                                    "game_id": stream.get("game_id"),
                                    "game_name": stream.get("game_name"),
                                    "title": stream.get("title"),
                                    "viewer_count": stream.get("viewer_count"),
                                    "started_at": stream.get("started_at"),
                                    "language": stream.get("language"),
                                    "thumbnail_url": stream.get("thumbnail_url"),
                                }
                            )

                        return {"success": True, "streams": streams, "total": len(streams)}
                    else:
                        return {"success": False, "error": f"Twitch API returned {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


class DiscordBotIntegration(BaseIntegration):
    """Discord bot integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {"bot_token": os.getenv("DISCORD_BOT_TOKEN", ""), "base_url": "https://discord.com/api/v10"}
        super().__init__("discord_bot", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Discord bot integration"""
        if not self.config["bot_token"]:
            logger.warning("⚠️  Discord bot token not configured")
        logger.info("✅ Discord bot integration initialized")

    async def cleanup(self):
        """Cleanup Discord bot resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Discord bot health"""
        if not self.config["bot_token"]:
            return {"status": "unhealthy", "error": "Bot token not configured"}

        try:
            headers = {"Authorization": f"Bot {self.config['bot_token']}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/users/@me", headers=headers) as response:
                    if response.status == 200:
                        bot_data = await response.json()
                        return {
                            "status": "healthy",
                            "bot_username": bot_data.get("username"),
                            "bot_id": bot_data.get("id"),
                        }
                    else:
                        return {"status": "unhealthy", "error": f"Discord API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Discord bot operations"""

        if operation == "send_message":
            return await self._send_message(**kwargs)
        elif operation == "get_guild_info":
            return await self._get_guild_info(**kwargs)
        elif operation == "create_channel":
            return await self._create_channel(**kwargs)
        elif operation == "get_bot_guilds":
            return await self._get_bot_guilds(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _send_message(self, channel_id: str, content: str, embeds: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Send message to Discord channel"""
        if not self.config["bot_token"]:
            return {"success": False, "error": "Bot token not configured"}

        headers = {"Authorization": f"Bot {self.config['bot_token']}", "Content-Type": "application/json"}

        data = {"content": content}
        if embeds:
            data["embeds"] = embeds

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/channels/{channel_id}/messages", headers=headers, json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "message_id": result.get("id"),
                            "channel_id": channel_id,
                            "content": content,
                            "timestamp": result.get("timestamp"),
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error}

        except Exception as e:
            return {"success": False, "error": str(e)}
