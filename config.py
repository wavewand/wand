"""
Configuration management for MCP Python server with config.json support
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ServerConfig:
    """Server configuration"""
    http_port: str = "8080"
    metrics_port: str = "9090"
    log_level: str = "info"
    version: str = "1.0.0"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "postgres://mcp:mcp-password@localhost/mcp?sslmode=disable"
    max_connections: int = 10
    max_idle_conns: int = 5
    conn_max_lifetime: int = 3600

@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_auth: bool = True
    jwt_secret: str = "your-secret-key"
    enable_rate_limit: bool = True
    rate_limit: int = 100
    token_expiry: int = 3600

@dataclass
class SlackConfig:
    """Slack integration configuration"""
    enabled: bool = False
    bot_token: str = ""
    app_token: str = ""
    default_channel: str = "#general"

@dataclass
class GitConfig:
    """Git integration configuration"""
    enabled: bool = False
    github_token: str = ""
    gitlab_token: str = ""
    default_branch: str = "main"
    commit_author: str = "MCP Bot"
    commit_email: str = "mcp-bot@example.com"

@dataclass
class JenkinsConfig:
    """Jenkins integration configuration"""
    enabled: bool = False
    url: str = ""
    username: str = ""
    token: str = ""
    default_pipeline: str = "build"

@dataclass
class YouTrackConfig:
    """YouTrack integration configuration"""
    enabled: bool = False
    url: str = ""
    token: str = ""
    default_project: str = ""

@dataclass
class AWSConfig:
    """AWS integration configuration"""
    enabled: bool = False
    region: str = "us-east-1"
    access_key_id: str = ""
    secret_access_key: str = ""
    default_bucket: str = ""

@dataclass
class PostgresConfig:
    """PostgreSQL integration configuration"""
    enabled: bool = False
    host: str = "localhost"
    port: int = 5432
    database: str = "mcp_data"
    username: str = "mcp"
    password: str = ""
    ssl_mode: str = "disable"

@dataclass
class PrinterConfig:
    """Individual printer configuration"""
    model: str = ""
    ip_address: str = ""
    access_code: str = ""

@dataclass
class BambuConfig:
    """Bambu 3D printer configuration"""
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://api.bambulab.com"
    printers: Dict[str, PrinterConfig] = field(default_factory=dict)

@dataclass
class WebConfig:
    """Web integration configuration"""
    enabled: bool = False
    search_api_key: str = ""
    proxy_url: str = ""
    timeout: int = 30

@dataclass
class IntegrationsConfig:
    """All integrations configuration"""
    slack: SlackConfig = field(default_factory=SlackConfig)
    git: GitConfig = field(default_factory=GitConfig)
    jenkins: JenkinsConfig = field(default_factory=JenkinsConfig)
    youtrack: YouTrackConfig = field(default_factory=YouTrackConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    bambu: BambuConfig = field(default_factory=BambuConfig)
    web: WebConfig = field(default_factory=WebConfig)

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    prometheus_endpoint: str = "/metrics"
    collect_interval: int = 15
    retention_days: int = 7

@dataclass
class DistributedConfig:
    """Distributed system configuration"""
    max_concurrent_tasks: int = 5
    task_queue_size: int = 100
    message_queue_size: int = 100
    worker_pool_size: int = 10
    heartbeat_interval: int = 30
    task_timeout: int = 300

@dataclass
class Config:
    """Main configuration structure"""
    server: ServerConfig = field(default_factory=ServerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    integrations: IntegrationsConfig = field(default_factory=IntegrationsConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    def apply_env_overrides(self):
        """Apply environment variable overrides to the configuration"""
        # Server overrides
        if port := os.getenv("HTTP_PORT"):
            self.server.http_port = port
        if port := os.getenv("METRICS_PORT"):
            self.server.metrics_port = port
        if level := os.getenv("LOG_LEVEL"):
            self.server.log_level = level

        # Database overrides
        if url := os.getenv("DATABASE_URL"):
            self.database.url = url

        # Security overrides
        if secret := os.getenv("JWT_SECRET"):
            self.security.jwt_secret = secret

        # Integration overrides
        if token := os.getenv("SLACK_BOT_TOKEN"):
            self.integrations.slack.bot_token = token
            self.integrations.slack.enabled = bool(token)
        if token := os.getenv("SLACK_APP_TOKEN"):
            self.integrations.slack.app_token = token
        
        if token := os.getenv("GITHUB_TOKEN"):
            self.integrations.git.github_token = token
            self.integrations.git.enabled = bool(token)
        
        if url := os.getenv("JENKINS_URL"):
            self.integrations.jenkins.url = url
            self.integrations.jenkins.enabled = bool(url)
        if user := os.getenv("JENKINS_USER"):
            self.integrations.jenkins.username = user
        if token := os.getenv("JENKINS_TOKEN"):
            self.integrations.jenkins.token = token
        
        if url := os.getenv("YOUTRACK_URL"):
            self.integrations.youtrack.url = url
            self.integrations.youtrack.enabled = bool(url)
        if token := os.getenv("YOUTRACK_TOKEN"):
            self.integrations.youtrack.token = token
        
        if key := os.getenv("AWS_ACCESS_KEY_ID"):
            self.integrations.aws.access_key_id = key
            self.integrations.aws.enabled = bool(key)
        if key := os.getenv("AWS_SECRET_ACCESS_KEY"):
            self.integrations.aws.secret_access_key = key
        if region := os.getenv("AWS_REGION"):
            self.integrations.aws.region = region
        
        if key := os.getenv("BAMBU_API_KEY"):
            self.integrations.bambu.api_key = key
            self.integrations.bambu.enabled = bool(key)
        
        if key := os.getenv("SEARCH_API_KEY"):
            self.integrations.web.search_api_key = key
            self.integrations.web.enabled = bool(key)

    def validate(self) -> List[str]:
        """Validate the configuration and return any errors"""
        errors = []

        # Validate server config
        if not self.server.http_port:
            self.server.http_port = "8080"
        if not self.server.metrics_port:
            self.server.metrics_port = "9090"
        if not self.server.log_level:
            self.server.log_level = "info"

        # Validate database config
        if not self.database.url:
            errors.append("Database URL is required")
        if self.database.max_connections == 0:
            self.database.max_connections = 10
        if self.database.max_idle_conns == 0:
            self.database.max_idle_conns = 5

        # Validate security config
        if self.security.enable_auth and not self.security.jwt_secret:
            errors.append("JWT secret is required when auth is enabled")
        if self.security.rate_limit == 0:
            self.security.rate_limit = 100
        if self.security.token_expiry == 0:
            self.security.token_expiry = 3600

        # Validate distributed config
        if self.distributed.max_concurrent_tasks == 0:
            self.distributed.max_concurrent_tasks = 5
        if self.distributed.task_queue_size == 0:
            self.distributed.task_queue_size = 100
        if self.distributed.message_queue_size == 0:
            self.distributed.message_queue_size = 100
        if self.distributed.worker_pool_size == 0:
            self.distributed.worker_pool_size = 10
        if self.distributed.heartbeat_interval == 0:
            self.distributed.heartbeat_interval = 30
        if self.distributed.task_timeout == 0:
            self.distributed.task_timeout = 300

        return errors

def load_config(filename: str = "config.json") -> Config:
    """Load configuration from a JSON file"""
    config = Config()
    
    # Try to load from file
    if Path(filename).exists():
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            # Update config with loaded data
            if 'server' in data:
                config.server = ServerConfig(**data['server'])
            if 'database' in data:
                config.database = DatabaseConfig(**data['database'])
            if 'security' in data:
                config.security = SecurityConfig(**data['security'])
            if 'monitoring' in data:
                config.monitoring = MonitoringConfig(**data['monitoring'])
            if 'distributed' in data:
                config.distributed = DistributedConfig(**data['distributed'])
            
            # Handle integrations separately due to nested structure
            if 'integrations' in data:
                int_data = data['integrations']
                if 'slack' in int_data:
                    config.integrations.slack = SlackConfig(**int_data['slack'])
                if 'git' in int_data:
                    config.integrations.git = GitConfig(**int_data['git'])
                if 'jenkins' in int_data:
                    config.integrations.jenkins = JenkinsConfig(**int_data['jenkins'])
                if 'youtrack' in int_data:
                    config.integrations.youtrack = YouTrackConfig(**int_data['youtrack'])
                if 'aws' in int_data:
                    config.integrations.aws = AWSConfig(**int_data['aws'])
                if 'postgres' in int_data:
                    config.integrations.postgres = PostgresConfig(**int_data['postgres'])
                if 'bambu' in int_data:
                    bambu_data = int_data['bambu']
                    # Convert printer configs
                    if 'printers' in bambu_data:
                        printers = {}
                        for name, printer_data in bambu_data['printers'].items():
                            printers[name] = PrinterConfig(**printer_data)
                        bambu_data['printers'] = printers
                    config.integrations.bambu = BambuConfig(**bambu_data)
                if 'web' in int_data:
                    config.integrations.web = WebConfig(**int_data['web'])
                    
            logger.info(f"Loaded configuration from {filename}")
        except Exception as e:
            logger.error(f"Failed to load config from {filename}: {e}")
    else:
        logger.info(f"Config file {filename} not found, using defaults")
    
    # Apply environment variable overrides
    config.apply_env_overrides()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
    
    return config

def save_config(config: Config, filename: str = "config.json"):
    """Save configuration to a JSON file"""
    # Convert to dict, handling special cases
    data = asdict(config)
    
    # Convert PrinterConfig objects to dicts
    if 'integrations' in data and 'bambu' in data['integrations']:
        if 'printers' in data['integrations']['bambu']:
            printers = {}
            for name, printer in data['integrations']['bambu']['printers'].items():
                if isinstance(printer, PrinterConfig):
                    printers[name] = asdict(printer)
                else:
                    printers[name] = printer
            data['integrations']['bambu']['printers'] = printers
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved configuration to {filename}")

def get_default_config() -> Config:
    """Get a default configuration"""
    return Config()

# Example usage
if __name__ == "__main__":
    # Load config
    config = load_config()
    
    # Print some values
    print(f"Server port: {config.server.http_port}")
    print(f"Database URL: {config.database.url}")
    print(f"Slack enabled: {config.integrations.slack.enabled}")
    
    # Save default config
    default_config = get_default_config()
    save_config(default_config, "config.default.json")