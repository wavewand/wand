"""
Comprehensive configuration for all Wand integrations
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Extend existing integrations config
from integrations_config import (
    APIConfig,
    AWSConfig,
    BambuConfig,
    GitConfig,
    JenkinsConfig,
    PostgresConfig,
    SlackConfig,
    WebConfig,
    YouTrackConfig,
)


@dataclass
class MultimediaConfig:
    """Configuration for multimedia processing integrations"""

    # FFmpeg
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    ffmpeg_timeout: int = 300
    max_video_size_gb: float = 2.0

    # OpenCV
    opencv_cascade_path: str = ""
    opencv_confidence_threshold: float = 0.5

    # Whisper
    whisper_model: str = "base"  # tiny, base, small, medium, large
    whisper_device: str = "cpu"
    whisper_use_api: bool = False
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # ElevenLabs
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_default_voice: str = "21m00Tcm4TlvDq8ikWAM"

    # Audio processing
    audio_temp_dir: str = "/tmp/audio"
    audio_max_duration_minutes: int = 60

    # Image processing
    image_max_size: tuple = (4096, 4096)
    image_default_quality: int = 85

    # OCR
    tesseract_languages: List[str] = field(default_factory=lambda: ["eng"])
    ocr_confidence_threshold: int = 60

    # QR codes
    qr_default_size: int = 10
    qr_error_correction: str = "M"

    def __post_init__(self):
        os.makedirs(self.audio_temp_dir, exist_ok=True)


@dataclass
class AIMLConfig:
    """Configuration for AI/ML platform integrations"""

    # HuggingFace
    huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN", "")
    hf_default_model: str = "gpt2"
    hf_max_tokens: int = 100
    hf_temperature: float = 0.7

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_default_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 100
    openai_temperature: float = 0.7

    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_default_model: str = "claude-3-sonnet-20240229"
    anthropic_max_tokens: int = 100

    # Cohere
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    cohere_default_model: str = "command"

    # Replicate
    replicate_api_token: str = os.getenv("REPLICATE_API_TOKEN", "")

    # Stability AI
    stability_api_key: str = os.getenv("STABILITY_API_KEY", "")
    stability_default_engine: str = "stable-diffusion-xl-1024-v1-0"
    stability_default_steps: int = 30
    stability_cfg_scale: float = 7.0

    # DeepL
    deepl_api_key: str = os.getenv("DEEPL_API_KEY", "")
    deepl_default_target_lang: str = "EN"
    deepl_preserve_formatting: bool = True


@dataclass
class ProductivityConfig:
    """Configuration for productivity integrations"""

    # Discord
    discord_bot_token: str = os.getenv("DISCORD_BOT_TOKEN", "")
    discord_webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    discord_default_channel: Optional[str] = None

    # Telegram
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

    # Email
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    from_email: str = os.getenv("FROM_EMAIL", "")
    email_use_tls: bool = True

    # Calendar
    google_credentials_file: str = os.getenv("GOOGLE_CREDENTIALS_FILE", "")
    calendar_id: str = "primary"

    # Notion
    notion_token: str = os.getenv("NOTION_TOKEN", "")
    notion_version: str = "2022-06-28"

    # File Storage
    gdrive_credentials: str = os.getenv("GDRIVE_CREDENTIALS_FILE", "")
    dropbox_access_token: str = os.getenv("DROPBOX_ACCESS_TOKEN", "")
    onedrive_client_id: str = os.getenv("ONEDRIVE_CLIENT_ID", "")

    # FTP
    ftp_default_host: str = os.getenv("FTP_HOST", "")
    ftp_default_user: str = os.getenv("FTP_USER", "")
    ftp_default_password: str = os.getenv("FTP_PASSWORD", "")


@dataclass
class DevToolsConfig:
    """Configuration for developer tools integrations"""

    # Docker
    docker_host: str = os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    docker_api_version: str = "1.41"
    docker_timeout: int = 30

    # Kubernetes
    kubeconfig_path: str = os.getenv("KUBECONFIG", "~/.kube/config")
    k8s_default_namespace: str = "default"
    k8s_timeout: int = 60

    # Terraform
    terraform_binary: str = "terraform"
    terraform_working_dir: str = "."
    terraform_auto_approve: bool = False
    terraform_timeout: int = 300

    # Monitoring
    prometheus_url: str = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    datadog_api_key: str = os.getenv("DATADOG_API_KEY", "")
    datadog_app_key: str = os.getenv("DATADOG_APP_KEY", "")
    sentry_dsn: str = os.getenv("SENTRY_DSN", "")

    # Testing
    selenium_webdriver: str = "chrome"  # chrome, firefox, safari
    selenium_headless: bool = True
    playwright_browser: str = "chromium"  # chromium, firefox, webkit
    postman_api_key: str = os.getenv("POSTMAN_API_KEY", "")


@dataclass
class EnterpriseConfig:
    """Configuration for enterprise integrations"""

    # CRM & Sales
    salesforce_client_id: str = os.getenv("SALESFORCE_CLIENT_ID", "")
    salesforce_client_secret: str = os.getenv("SALESFORCE_CLIENT_SECRET", "")
    hubspot_api_key: str = os.getenv("HUBSPOT_API_KEY", "")
    pipedrive_api_token: str = os.getenv("PIPEDRIVE_API_TOKEN", "")
    stripe_secret_key: str = os.getenv("STRIPE_SECRET_KEY", "")

    # Project Management
    jira_url: str = os.getenv("JIRA_URL", "")
    jira_username: str = os.getenv("JIRA_USERNAME", "")
    jira_api_token: str = os.getenv("JIRA_API_TOKEN", "")
    asana_access_token: str = os.getenv("ASANA_ACCESS_TOKEN", "")
    trello_api_key: str = os.getenv("TRELLO_API_KEY", "")
    trello_token: str = os.getenv("TRELLO_TOKEN", "")

    # HR & Operations
    workday_client_id: str = os.getenv("WORKDAY_CLIENT_ID", "")
    bamboohr_api_key: str = os.getenv("BAMBOOHR_API_KEY", "")
    toggl_api_token: str = os.getenv("TOGGL_API_TOKEN", "")


@dataclass
class SecurityConfig:
    """Configuration for security integrations"""

    # Secret Management
    vault_url: str = os.getenv("VAULT_URL", "")
    vault_token: str = os.getenv("VAULT_TOKEN", "")
    onepassword_service_account: str = os.getenv("ONEPASSWORD_SERVICE_ACCOUNT", "")

    # Identity & Access
    okta_domain: str = os.getenv("OKTA_DOMAIN", "")
    okta_api_token: str = os.getenv("OKTA_API_TOKEN", "")
    auth0_domain: str = os.getenv("AUTH0_DOMAIN", "")
    auth0_client_id: str = os.getenv("AUTH0_CLIENT_ID", "")
    auth0_client_secret: str = os.getenv("AUTH0_CLIENT_SECRET", "")

    # Security Scanning
    veracode_api_id: str = os.getenv("VERACODE_API_ID", "")
    veracode_api_key: str = os.getenv("VERACODE_API_KEY", "")
    snyk_token: str = os.getenv("SNYK_TOKEN", "")


@dataclass
class SpecializedConfig:
    """Configuration for specialized integrations"""

    # Gaming
    steam_api_key: str = os.getenv("STEAM_API_KEY", "")
    minecraft_server_host: str = os.getenv("MINECRAFT_SERVER_HOST", "")
    minecraft_rcon_password: str = os.getenv("MINECRAFT_RCON_PASSWORD", "")

    # IoT & Hardware
    homeassistant_url: str = os.getenv("HOMEASSISTANT_URL", "")
    homeassistant_token: str = os.getenv("HOMEASSISTANT_TOKEN", "")
    philips_hue_bridge_ip: str = os.getenv("PHILIPS_HUE_BRIDGE_IP", "")
    philips_hue_username: str = os.getenv("PHILIPS_HUE_USERNAME", "")

    # Blockchain
    ethereum_node_url: str = os.getenv("ETHEREUM_NODE_URL", "")
    bitcoin_rpc_url: str = os.getenv("BITCOIN_RPC_URL", "")
    opensea_api_key: str = os.getenv("OPENSEA_API_KEY", "")


@dataclass
class WandIntegrationsConfig:
    """Master configuration for all Wand integrations"""

    # Legacy integrations (existing)
    slack: SlackConfig = field(default_factory=SlackConfig)
    git: GitConfig = field(default_factory=GitConfig)
    jenkins: JenkinsConfig = field(default_factory=JenkinsConfig)
    youtrack: YouTrackConfig = field(default_factory=YouTrackConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    bambu: BambuConfig = field(default_factory=BambuConfig)
    web: WebConfig = field(default_factory=WebConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # New integration categories
    multimedia: MultimediaConfig = field(default_factory=MultimediaConfig)
    ai_ml: AIMLConfig = field(default_factory=AIMLConfig)
    productivity: ProductivityConfig = field(default_factory=ProductivityConfig)
    devtools: DevToolsConfig = field(default_factory=DevToolsConfig)
    enterprise: EnterpriseConfig = field(default_factory=EnterpriseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    specialized: SpecializedConfig = field(default_factory=SpecializedConfig)

    # Global settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    rate_limiting_enabled: bool = True
    default_timeout: int = 30
    retry_attempts: int = 3
    log_level: str = "INFO"

    def get_integration_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all integrations"""
        status = {}

        # Check each integration category
        categories = {
            "legacy": {
                "slack": bool(self.slack.token),
                "git": bool(self.git.github_token or self.git.gitlab_token),
                "jenkins": bool(self.jenkins.token),
                "youtrack": bool(self.youtrack.token),
                "postgres": bool(self.postgres.password),
                "aws": bool(self.aws.access_key_id),
                "bambu": bool(self.bambu.api_key),
                "web": True,
                "api": True,
            },
            "multimedia": {
                "ffmpeg": True,  # Assume available
                "opencv": True,
                "whisper": bool(self.multimedia.openai_api_key) if self.multimedia.whisper_use_api else True,
                "elevenlabs": bool(self.multimedia.elevenlabs_api_key),
                "audio": True,
                "images": True,
                "ocr": True,  # Assume Tesseract available
                "qr": True,
            },
            "ai_ml": {
                "huggingface": bool(self.ai_ml.huggingface_token),
                "openai": bool(self.ai_ml.openai_api_key),
                "anthropic": bool(self.ai_ml.anthropic_api_key),
                "cohere": bool(self.ai_ml.cohere_api_key),
                "replicate": bool(self.ai_ml.replicate_api_token),
                "stability": bool(self.ai_ml.stability_api_key),
                "deepl": bool(self.ai_ml.deepl_api_key),
            },
            "productivity": {
                "discord": bool(self.productivity.discord_bot_token or self.productivity.discord_webhook_url),
                "telegram": bool(self.productivity.telegram_bot_token),
                "email": bool(self.productivity.smtp_username and self.productivity.smtp_password),
                "calendar": bool(self.productivity.google_credentials_file),
                "notion": bool(self.productivity.notion_token),
            },
            "devtools": {
                "docker": True,  # Assume Docker available
                "kubernetes": True,  # Assume kubectl available
                "terraform": True,  # Assume Terraform available
                "prometheus": bool(self.devtools.prometheus_url),
                "datadog": bool(self.devtools.datadog_api_key),
                "sentry": bool(self.devtools.sentry_dsn),
            },
        }

        # Calculate totals
        for category, integrations in categories.items():
            configured_count = sum(1 for configured in integrations.values() if configured)
            total_count = len(integrations)

            status[category] = {
                "configured": configured_count,
                "total": total_count,
                "percentage": (configured_count / total_count) * 100,
                "integrations": integrations,
            }

        return status

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get safe configuration summary without sensitive data"""
        integration_status = self.get_integration_status()

        total_configured = sum(cat["configured"] for cat in integration_status.values())
        total_available = sum(cat["total"] for cat in integration_status.values())

        return {
            "summary": {
                "total_integrations": total_available,
                "configured_integrations": total_configured,
                "configuration_percentage": (total_configured / total_available) * 100,
                "categories": len(integration_status),
            },
            "categories": integration_status,
            "global_settings": {
                "cache_enabled": self.cache_enabled,
                "rate_limiting_enabled": self.rate_limiting_enabled,
                "default_timeout": self.default_timeout,
                "retry_attempts": self.retry_attempts,
                "log_level": self.log_level,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def save_config_summary(self, filepath: str = "wand_integrations_status.json"):
        """Save configuration summary to file"""
        import json

        summary = self.get_configuration_summary()

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

    def load_env_file(self, filepath: str = ".env"):
        """Load environment variables from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

    def validate_integration_config(self, integration_name: str, config: Dict[str, Any]) -> bool:
        """Validate configuration for a specific integration

        Args:
            integration_name: Name of the integration
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid
        """
        # Basic validation - can be extended per integration
        if not config:
            return False

        # Check for required fields based on integration type
        required_fields = {
            'ffmpeg': ['ffmpeg_path'],
            'opencv': [],
            'whisper': ['whisper_model'],
            'elevenlabs': ['elevenlabs_api_key'],
            'slack': ['token'],
            'github': ['github_token'],
            'openai': ['openai_api_key'],
            'anthropic': ['anthropic_api_key'],
        }

        if integration_name in required_fields:
            for field in required_fields[integration_name]:
                if field not in config or not config[field]:
                    return False

        return True


# Global configuration instance
wand_config = WandIntegrationsConfig()
