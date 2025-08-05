"""
Configuration management for MCP Python server with config.json support
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    jwt_secret: str = ""  # Must be set via environment variable
    enable_rate_limit: bool = True
    rate_limit: int = 100
    token_expiry: int = 3600

    def __post_init__(self):
        """Validate security configuration"""
        import os
        import secrets

        # Get JWT secret from environment or generate one
        if not self.jwt_secret:
            self.jwt_secret = os.environ.get('WAND_JWT_SECRET', '')

        # If still no secret, generate a secure one and warn
        if not self.jwt_secret or self.jwt_secret == "your-secret-key":
            self.jwt_secret = secrets.token_urlsafe(32)
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("JWT secret was not set or using default value!")
            logger.warning("Generated new JWT secret. Set WAND_JWT_SECRET environment variable for production.")
            logger.warning("This generated secret will change on restart, invalidating existing tokens.")


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
class ExecutionSecurityConfig:
    """Security configuration for command execution"""

    command_validation: bool = True
    path_restrictions: List[str] = field(default_factory=lambda: ["/workspace", "/tmp"])
    user_isolation: bool = True
    allowed_commands: List[str] = field(
        default_factory=lambda: [
            "git",
            "npm",
            "yarn",
            "python",
            "python3",
            "pip",
            "pip3",
            "node",
            "docker",
            "ls",
            "cat",
            "grep",
            "find",
            "head",
            "tail",
            "echo",
            "which",
            "whoami",
            "pwd",
            "mkdir",
            "touch",
            "cp",
            "mv",
        ]
    )
    blocked_commands: List[str] = field(
        default_factory=lambda: [
            "rm",
            "dd",
            "mkfs",
            "fdisk",
            "mount",
            "umount",
            "su",
            "sudo",
            "passwd",
            "useradd",
            "userdel",
            "usermod",
            "systemctl",
            "service",
            "kill",
            "killall",
        ]
    )
    max_memory: str = "1GB"
    max_cpu: str = "1.0"
    max_processes: int = 50
    max_execution_time: int = 300


@dataclass
class HostAgentConfig:
    """Host Agent execution backend configuration"""

    url: str = "http://host-agent:8001"
    auth_token: str = ""
    timeout: int = 30
    retry_attempts: int = 3
    health_check_interval: int = 60


@dataclass
class DockerSocketConfig:
    """Docker Socket execution backend configuration"""

    socket_path: str = "/var/run/docker.sock"
    default_image: str = "ubuntu:22.04"
    network: str = "bridge"
    working_dir: str = "/workspace"
    auto_remove: bool = True
    memory_limit: str = "512m"
    cpu_limit: str = "0.5"
    disk_limit: str = "1g"


@dataclass
class SSHRemoteConfig:
    """SSH Remote execution backend configuration"""

    host: str = "localhost"
    port: int = 22
    username: str = "mcp-agent"
    auth_method: str = "key"
    key_file: str = "/app/secrets/mcp_key"
    password: str = ""
    known_hosts_file: str = "/app/config/known_hosts"
    timeout: int = 30
    keepalive: bool = True


@dataclass
class VolumeMountConfig:
    """Volume Mount execution backend configuration"""

    host_binary_path: str = "/host"
    chroot_jail: str = "/workspace"
    bind_mounts: List[Dict[str, str]] = field(
        default_factory=lambda: [
            {"host": "/workspace", "container": "/workspace", "mode": "rw"},
            {"host": "/tmp", "container": "/tmp", "mode": "rw"},
        ]
    )


@dataclass
class PrivilegedConfig:
    """Privileged execution backend configuration (DANGEROUS - development only)"""

    host_root: str = "/host"
    namespace_isolation: bool = False
    capabilities: List[str] = field(default_factory=lambda: ["ALL"])
    allow_dangerous_commands: bool = True
    mount_host_filesystem: bool = True


@dataclass
class ExecutionConfig:
    """System command execution configuration"""

    mode: str = "native"  # native, host_agent, docker_socket, ssh_remote, volume_mount, privileged
    default_timeout: int = 30
    max_concurrent: int = 10
    working_directory: str = "/workspace"
    audit_logging: bool = True

    # Security settings
    security: ExecutionSecurityConfig = field(default_factory=ExecutionSecurityConfig)

    # Backend-specific configurations
    host_agent: HostAgentConfig = field(default_factory=HostAgentConfig)
    docker_socket: DockerSocketConfig = field(default_factory=DockerSocketConfig)
    ssh_remote: SSHRemoteConfig = field(default_factory=SSHRemoteConfig)
    volume_mount: VolumeMountConfig = field(default_factory=VolumeMountConfig)
    privileged: PrivilegedConfig = field(default_factory=PrivilegedConfig)


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
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

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

        # Execution overrides
        if mode := os.getenv("EXECUTION_MODE"):
            self.execution.mode = mode
        if timeout := os.getenv("EXECUTION_TIMEOUT"):
            self.execution.default_timeout = int(timeout)
        if concurrent := os.getenv("EXECUTION_MAX_CONCURRENT"):
            self.execution.max_concurrent = int(concurrent)
        if workdir := os.getenv("EXECUTION_WORKING_DIR"):
            self.execution.working_directory = workdir

        # Host Agent overrides
        if url := os.getenv("HOST_AGENT_URL"):
            self.execution.host_agent.url = url
        if token := os.getenv("HOST_AGENT_TOKEN"):
            self.execution.host_agent.auth_token = token

        # SSH Remote overrides
        if host := os.getenv("SSH_HOST"):
            self.execution.ssh_remote.host = host
        if user := os.getenv("SSH_USERNAME"):
            self.execution.ssh_remote.username = user
        if keyfile := os.getenv("SSH_KEY_FILE"):
            self.execution.ssh_remote.key_file = keyfile

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

        # Validate execution config
        valid_modes = ["native", "host_agent", "docker_socket", "ssh_remote", "volume_mount", "privileged"]
        if self.execution.mode not in valid_modes:
            errors.append(f"Invalid execution mode '{self.execution.mode}'. Must be one of: {', '.join(valid_modes)}")

        if self.execution.mode == "host_agent" and not self.execution.host_agent.auth_token:
            errors.append("Host Agent auth token is required when using host_agent execution mode")

        if self.execution.mode == "ssh_remote":
            if not self.execution.ssh_remote.host:
                errors.append("SSH host is required when using ssh_remote execution mode")
            if not self.execution.ssh_remote.username:
                errors.append("SSH username is required when using ssh_remote execution mode")

        if self.execution.default_timeout <= 0:
            self.execution.default_timeout = 30
        if self.execution.max_concurrent <= 0:
            self.execution.max_concurrent = 10

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
            if 'execution' in data:
                exec_data = data['execution']
                config.execution = ExecutionConfig()

                # Basic execution settings
                for key in ['mode', 'default_timeout', 'max_concurrent', 'working_directory', 'audit_logging']:
                    if key in exec_data:
                        setattr(config.execution, key, exec_data[key])

                # Security settings
                if 'security' in exec_data:
                    config.execution.security = ExecutionSecurityConfig(**exec_data['security'])

                # Backend-specific settings
                if 'host_agent' in exec_data:
                    config.execution.host_agent = HostAgentConfig(**exec_data['host_agent'])
                if 'docker_socket' in exec_data:
                    config.execution.docker_socket = DockerSocketConfig(**exec_data['docker_socket'])
                if 'ssh_remote' in exec_data:
                    config.execution.ssh_remote = SSHRemoteConfig(**exec_data['ssh_remote'])
                if 'volume_mount' in exec_data:
                    config.execution.volume_mount = VolumeMountConfig(**exec_data['volume_mount'])
                if 'privileged' in exec_data:
                    config.execution.privileged = PrivilegedConfig(**exec_data['privileged'])

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
