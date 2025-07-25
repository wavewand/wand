"""
Integration configurations for distributed MCP server
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class SlackConfig:
    token: str = os.getenv("SLACK_BOT_TOKEN", "")
    app_token: str = os.getenv("SLACK_APP_TOKEN", "")
    default_channel: str = "#dev-updates"
    enable_threads: bool = True
    
@dataclass
class GitConfig:
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    gitlab_token: str = os.getenv("GITLAB_TOKEN", "")
    default_branch: str = "main"
    auto_commit: bool = False
    commit_author: str = "MCP Agent"
    commit_email: str = "mcp@example.com"
    
@dataclass
class JenkinsConfig:
    url: str = os.getenv("JENKINS_URL", "https://jenkins.example.com")
    username: str = os.getenv("JENKINS_USER", "")
    token: str = os.getenv("JENKINS_TOKEN", "")
    default_pipeline: str = "ci-cd"
    timeout: int = 300  # seconds
    
@dataclass
class YouTrackConfig:
    url: str = os.getenv("YOUTRACK_URL", "https://youtrack.example.com")
    token: str = os.getenv("YOUTRACK_TOKEN", "")
    default_project: str = "DEV"
    
@dataclass
class PostgresConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "")
    default_database: str = os.getenv("POSTGRES_DB", "production")
    ssl_mode: str = "require"
    pool_size: int = 10
    
@dataclass
class AWSConfig:
    access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    region: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    services: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.services is None:
            self.services = {
                "ec2": {"enabled": True},
                "s3": {"enabled": True, "default_bucket": "mcp-artifacts"},
                "lambda": {"enabled": True},
                "rds": {"enabled": True},
                "ecs": {"enabled": True},
                "cloudformation": {"enabled": True}
            }
            
@dataclass
class BambuConfig:
    api_key: str = os.getenv("BAMBU_API_KEY", "")
    cloud_url: str = "https://api.bambulab.com"
    printers: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.printers is None:
            self.printers = {
                "X1-Carbon-01": {
                    "ip": "192.168.1.100",
                    "access_code": os.getenv("BAMBU_X1_CODE", ""),
                    "model": "X1 Carbon"
                },
                "P1S-01": {
                    "ip": "192.168.1.101",
                    "access_code": os.getenv("BAMBU_P1S_CODE", ""),
                    "model": "P1S"
                }
            }
            
@dataclass
class WebConfig:
    search_api_key: str = os.getenv("SEARCH_API_KEY", "")
    search_engine: str = "duckduckgo"  # or "google", "bing"
    proxy: Optional[str] = os.getenv("HTTP_PROXY", None)
    timeout: int = 30
    max_results: int = 10
    
@dataclass
class APIConfig:
    default_timeout: int = 30
    max_retries: int = 3
    rate_limit: Dict[str, int] = None  # requests per minute per service
    
    def __post_init__(self):
        if self.rate_limit is None:
            self.rate_limit = {
                "default": 60,
                "openai": 50,
                "anthropic": 40,
                "github": 100,
                "aws": 200
            }

class IntegrationsManager:
    """Manages all integration configurations"""
    
    def __init__(self):
        self.slack = SlackConfig()
        self.git = GitConfig()
        self.jenkins = JenkinsConfig()
        self.youtrack = YouTrackConfig()
        self.postgres = PostgresConfig()
        self.aws = AWSConfig()
        self.bambu = BambuConfig()
        self.web = WebConfig()
        self.api = APIConfig()
        
    def validate_config(self) -> Dict[str, bool]:
        """Validate that required configurations are present"""
        validations = {
            "slack": bool(self.slack.token),
            "git": bool(self.git.github_token or self.git.gitlab_token),
            "jenkins": bool(self.jenkins.url and self.jenkins.token),
            "youtrack": bool(self.youtrack.url and self.youtrack.token),
            "postgres": bool(self.postgres.host and self.postgres.user),
            "aws": bool(self.aws.access_key_id and self.aws.secret_access_key),
            "bambu": bool(self.bambu.api_key),
            "web": True,  # Web search can work without API key
            "api": True   # Generic API calls always available
        }
        return validations
        
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of all configurations (without sensitive data)"""
        return {
            "slack": {
                "configured": bool(self.slack.token),
                "default_channel": self.slack.default_channel
            },
            "git": {
                "github_configured": bool(self.git.github_token),
                "gitlab_configured": bool(self.git.gitlab_token),
                "default_branch": self.git.default_branch
            },
            "jenkins": {
                "configured": bool(self.jenkins.token),
                "url": self.jenkins.url,
                "default_pipeline": self.jenkins.default_pipeline
            },
            "youtrack": {
                "configured": bool(self.youtrack.token),
                "url": self.youtrack.url,
                "default_project": self.youtrack.default_project
            },
            "postgres": {
                "configured": bool(self.postgres.password),
                "host": self.postgres.host,
                "database": self.postgres.default_database
            },
            "aws": {
                "configured": bool(self.aws.access_key_id),
                "region": self.aws.region,
                "enabled_services": list(self.aws.services.keys())
            },
            "bambu": {
                "configured": bool(self.bambu.api_key),
                "printers": list(self.bambu.printers.keys())
            },
            "web": {
                "search_engine": self.web.search_engine,
                "has_api_key": bool(self.web.search_api_key)
            }
        }
        
    def save_config(self, filepath: str = "integrations.json"):
        """Save configuration to file (excluding sensitive data)"""
        config = self.get_config_summary()
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
            
    def load_env_file(self, filepath: str = ".env"):
        """Load environment variables from a file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value

# Global instance
integrations = IntegrationsManager()