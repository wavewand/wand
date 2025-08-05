"""
🪄 Wand Integration System
==========================

Comprehensive integration layer for the Wand magical toolkit.
Provides seamless access to 50+ external services and tools.

Categories:
- Multimedia: Video, audio, image processing
- AI/ML: HuggingFace, OpenAI, Anthropic, and more
- Data & Analytics: Databases, visualization, processing
- Productivity: Communication, storage, documentation
- DevTools: Containers, monitoring, testing
- Enterprise: CRM, project management, HR
- Security: Vault, authentication, compliance
- Specialized: Gaming, IoT, blockchain

Architecture:
- Base classes provide common functionality
- Each category has dedicated modules
- Configuration management handles all settings
- Error handling and retry logic built-in
- Rate limiting and caching for performance
"""

from .ai_ml import *
from .base.auth_manager import AuthManager
from .base.cache_manager import CacheManager
from .base.error_handler import ErrorHandler
from .base.integration_base import BaseIntegration
from .base.rate_limiter import RateLimiter
from .data_analytics import *
from .devtools import *
from .enterprise import *

# Import framework integrations
from .frameworks import *
from .haystack import *

# Import legacy integrations (existing 9 integrations)
from .legacy import (
    api_integration,
    aws_integration,
    bambu_integration,
    git_integration,
    jenkins_integration,
    postgres_integration,
    slack_integration,
    web_integration,
    youtrack_integration,
)
from .llamaindex import *

# Import new integration categories
from .multimedia import *
from .productivity import *
from .security import *
from .specialized import *

# Integration registry
INTEGRATION_REGISTRY = {}


def register_integration(name: str, integration_class, category: str = "general"):
    """Register an integration in the global registry"""
    INTEGRATION_REGISTRY[name] = {"class": integration_class, "category": category, "instance": None}


def get_integration(name: str):
    """Get an integration instance by name"""
    if name not in INTEGRATION_REGISTRY:
        raise ValueError(f"Integration '{name}' not found")

    entry = INTEGRATION_REGISTRY[name]
    if entry["instance"] is None:
        entry["instance"] = entry["class"]()

    return entry["instance"]


def list_integrations(category: str = None):
    """List all available integrations, optionally filtered by category"""
    if category:
        return {k: v for k, v in INTEGRATION_REGISTRY.items() if v["category"] == category}
    return INTEGRATION_REGISTRY


def get_integration_categories():
    """Get all available integration categories"""
    return list(set(entry["category"] for entry in INTEGRATION_REGISTRY.values()))


# Register legacy integrations
register_integration("slack", type(slack_integration), "communication")
register_integration("git", type(git_integration), "devtools")
register_integration("jenkins", type(jenkins_integration), "devtools")
register_integration("youtrack", type(youtrack_integration), "project_management")
register_integration("postgres", type(postgres_integration), "data_analytics")
register_integration("aws", type(aws_integration), "cloud")
register_integration("bambu", type(bambu_integration), "hardware")
register_integration("websearch", type(web_integration), "productivity")
register_integration("api", type(api_integration), "general")

__all__ = [
    "BaseIntegration",
    "AuthManager",
    "RateLimiter",
    "CacheManager",
    "ErrorHandler",
    "INTEGRATION_REGISTRY",
    "register_integration",
    "get_integration",
    "list_integrations",
    "get_integration_categories",
]
