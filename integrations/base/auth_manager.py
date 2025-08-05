"""
Authentication manager for Wand integrations
"""

import base64
import hashlib
import hmac
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import aiohttp
import jwt


class AuthManager:
    """
    Handles authentication for various integration types:
    - API keys
    - OAuth tokens
    - JWT tokens
    - Basic auth
    - Custom authentication schemes
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.auth_type = config.get("type", "api_key")
        self.tokens = {}  # Cache for tokens
        self.token_expiry = {}  # Track token expiration

    async def get_auth_headers(self, service: str) -> Dict[str, str]:
        """Get authentication headers for a service"""
        headers = {}

        if self.auth_type == "api_key":
            api_key = self.config.get("api_key") or os.getenv(f"{service.upper()}_API_KEY")
            if api_key:
                key_header = self.config.get("key_header", "Authorization")
                key_prefix = self.config.get("key_prefix", "Bearer ")
                headers[key_header] = f"{key_prefix}{api_key}"

        elif self.auth_type == "bearer":
            token = await self.get_bearer_token(service)
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif self.auth_type == "basic":
            username = self.config.get("username") or os.getenv(f"{service.upper()}_USERNAME")
            password = self.config.get("password") or os.getenv(f"{service.upper()}_PASSWORD")
            if username and password:
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {credentials}"

        elif self.auth_type == "jwt":
            token = await self.generate_jwt_token(service)
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif self.auth_type == "oauth":
            token = await self.get_oauth_token(service)
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif self.auth_type == "custom":
            headers.update(await self.get_custom_auth(service))

        return headers

    async def get_bearer_token(self, service: str) -> Optional[str]:
        """Get bearer token for service"""
        token_key = f"{service}_bearer_token"

        # Check cache
        if token_key in self.tokens:
            if token_key not in self.token_expiry or datetime.now(timezone.utc) < self.token_expiry[token_key]:
                return self.tokens[token_key]

        # Get new token
        token = self.config.get("token") or os.getenv(f"{service.upper()}_TOKEN")
        if token:
            self.tokens[token_key] = token
            # Set expiry if provided
            expires_in = self.config.get("expires_in", 3600)  # Default 1 hour
            self.token_expiry[token_key] = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

        return token

    async def generate_jwt_token(self, service: str) -> Optional[str]:
        """Generate JWT token for service"""
        secret = self.config.get("jwt_secret") or os.getenv(f"{service.upper()}_JWT_SECRET")
        if not secret:
            return None

        payload = {
            "service": service,
            "iat": int(time.time()),
            "exp": int(time.time()) + self.config.get("jwt_expires", 3600),
        }

        # Add custom claims
        custom_claims = self.config.get("jwt_claims", {})
        payload.update(custom_claims)

        algorithm = self.config.get("jwt_algorithm", "HS256")
        return jwt.encode(payload, secret, algorithm=algorithm)

    async def get_oauth_token(self, service: str) -> Optional[str]:
        """Get OAuth token for service"""
        token_key = f"{service}_oauth_token"

        # Check cache
        if token_key in self.tokens:
            if token_key not in self.token_expiry or datetime.now(timezone.utc) < self.token_expiry[token_key]:
                return self.tokens[token_key]

        # Refresh token if available
        refresh_token = self.config.get("refresh_token") or os.getenv(f"{service.upper()}_REFRESH_TOKEN")
        if refresh_token:
            return await self.refresh_oauth_token(service, refresh_token)

        # Return existing token if available
        return self.config.get("access_token") or os.getenv(f"{service.upper()}_ACCESS_TOKEN")

    async def refresh_oauth_token(self, service: str, refresh_token: str) -> Optional[str]:
        """Refresh OAuth token"""
        token_url = self.config.get("token_url")
        client_id = self.config.get("client_id") or os.getenv(f"{service.upper()}_CLIENT_ID")
        client_secret = self.config.get("client_secret") or os.getenv(f"{service.upper()}_CLIENT_SECRET")

        if not all([token_url, client_id, client_secret]):
            return None

        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(token_url, data=data) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        access_token = token_data.get("access_token")

                        if access_token:
                            token_key = f"{service}_oauth_token"
                            self.tokens[token_key] = access_token

                            # Set expiry
                            expires_in = token_data.get("expires_in", 3600)
                            self.token_expiry[token_key] = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

                            return access_token
            except Exception:
                pass

        return None

    async def get_custom_auth(self, service: str) -> Dict[str, str]:
        """Get custom authentication headers"""
        headers = {}

        # Service-specific custom auth
        if service == "github":
            token = os.getenv("GITHUB_TOKEN")
            if token:
                headers["Authorization"] = f"token {token}"
                headers["Accept"] = "application/vnd.github.v3+json"

        elif service == "slack":
            token = os.getenv("SLACK_BOT_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif service == "discord":
            token = os.getenv("DISCORD_BOT_TOKEN")
            if token:
                headers["Authorization"] = f"Bot {token}"

        elif service == "stripe":
            api_key = os.getenv("STRIPE_SECRET_KEY")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        elif service == "huggingface":
            token = os.getenv("HUGGINGFACE_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif service == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        elif service == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                headers["x-api-key"] = api_key

        # Add any additional custom headers from config
        custom_headers = self.config.get("custom_headers", {})
        headers.update(custom_headers)

        return headers

    def generate_signature(self, method: str, url: str, body: str = "") -> str:
        """Generate signature for webhook verification"""
        secret = self.config.get("webhook_secret", "")
        if not secret:
            return ""

        message = f"{method.upper()}{url}{body}"
        signature = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()

        return signature

    def verify_webhook_signature(self, signature: str, method: str, url: str, body: str = "") -> bool:
        """Verify webhook signature"""
        expected_signature = self.generate_signature(method, url, body)
        return hmac.compare_digest(signature, expected_signature)

    def is_token_expired(self, service: str, token_type: str = "bearer") -> bool:
        """Check if token is expired"""
        token_key = f"{service}_{token_type}_token"
        if token_key not in self.token_expiry:
            return False

        return datetime.now(timezone.utc) >= self.token_expiry[token_key]

    def clear_token_cache(self, service: str = None):
        """Clear token cache for a service or all services"""
        if service:
            keys_to_remove = [k for k in self.tokens.keys() if k.startswith(service)]
            for key in keys_to_remove:
                self.tokens.pop(key, None)
                self.token_expiry.pop(key, None)
        else:
            self.tokens.clear()
            self.token_expiry.clear()

    def get_auth_status(self) -> Dict[str, Any]:
        """Get authentication status summary"""
        return {
            "auth_type": self.auth_type,
            "cached_tokens": len(self.tokens),
            "expired_tokens": sum(
                1
                for service in self.token_expiry.keys()
                if self.is_token_expired(service.split("_")[0], service.split("_")[1])
            ),
            "config_keys": list(self.config.keys()),
        }
