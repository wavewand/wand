import pytest
import os
import json
import tempfile
from unittest.mock import patch
from config import Config, ServerConfig, IntegrationsConfig, SlackConfig, load_config, save_config

class TestConfig:
    """Test configuration management"""
    
    def test_default_config(self):
        """Test loading default configuration"""
        config = Config()
        
        # Check server defaults
        assert config.server.http_port == "8080"
        assert config.server.metrics_port == "9090"
        assert config.server.log_level == "info"
        
        # Check database defaults
        assert config.database.url == "postgres://mcp:mcp-password@localhost/mcp?sslmode=disable"
        assert config.database.max_connections == 10
        
        # Check monitoring defaults
        assert config.monitoring.enabled == True
        assert config.monitoring.prometheus_endpoint == "/metrics"
    
    def test_load_from_json(self):
        """Test loading configuration from JSON file"""
        test_config = {
            "server": {
                "http_port": "9000",
                "metrics_port": "9090"
            },
            "integrations": {
                "slack": {
                    "bot_token": "test-token",
                    "default_channel": "#test"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            
            assert config.server.http_port == "9000"  # JSON loads as string
            assert config.integrations.slack.bot_token == "test-token"
            assert config.integrations.slack.default_channel == "#test"
        finally:
            os.unlink(temp_path)
    
    def test_environment_override(self):
        """Test environment variable overrides"""
        with patch.dict(os.environ, {
            'HTTP_PORT': '8888',
            'SLACK_BOT_TOKEN': 'env-token',
            'DATABASE_URL': 'postgres://test:secret123@localhost/testdb',
            'AWS_REGION': 'eu-west-1'
        }):
            config = Config()
            config.apply_env_overrides()
            
            assert config.server.http_port == '8888'
            assert config.integrations.slack.bot_token == 'env-token'
            assert config.database.url == 'postgres://test:secret123@localhost/testdb'
            assert config.integrations.aws.region == 'eu-west-1'
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = Config()
        
        # Valid config should pass (returns empty error list)
        assert config.validate() == []
        
        # Test missing database URL
        config.database.url = ""
        errors = config.validate()
        assert "Database URL is required" in errors
        
        # Test missing JWT secret when auth is enabled
        config = Config()
        config.security.jwt_secret = ""
        config.security.enable_auth = True
        errors = config.validate()
        assert "JWT secret is required when auth is enabled" in errors
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        os.unlink(temp_path)  # Delete so we can test creation
        
        # Create and save config
        config = Config()
        config.server.http_port = "9999"
        config.integrations.slack.bot_token = "test-save-token"
        save_config(config, temp_path)
        
        try:
            assert os.path.exists(temp_path)
            
            # Load and verify
            loaded_config = load_config(temp_path)
            assert loaded_config.server.http_port == "9999"
            assert loaded_config.integrations.slack.bot_token == "test-save-token"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_integration_config_defaults(self):
        """Test integration configuration defaults"""
        config = IntegrationsConfig()
        
        # Slack defaults
        assert config.slack.default_channel == "#general"
        assert config.slack.bot_token == ""
        
        # Git defaults  
        assert config.git.default_branch == "main"
        assert config.git.commit_author == "MCP Bot"
        
        # AWS defaults
        assert config.aws.region == "us-east-1"
        assert config.aws.access_key_id == ""
        
        # Bambu defaults
        assert config.bambu.api_key == ""
        assert config.bambu.base_url == "https://api.bambulab.com"
    
    def test_distributed_config(self):
        """Test distributed configuration"""
        config = Config()
        
        assert config.distributed.task_queue_size == 100
        assert config.distributed.heartbeat_interval == 30
        assert config.distributed.task_timeout == 300
        assert config.distributed.max_concurrent_tasks == 5
        assert config.distributed.worker_pool_size == 10
    
    def test_security_config(self):
        """Test security configuration"""
        config = Config()
        
        assert config.security.jwt_secret == "your-secret-key"  # Default value
        assert config.security.token_expiry == 3600
        assert config.security.enable_auth == True
        assert config.security.enable_rate_limit == True
        assert config.security.rate_limit == 100
    
    def test_partial_config_load(self):
        """Test loading partial configuration"""
        test_config = {
            "server": {
                "http_port": "9999"
            }
            # Other sections missing
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            
            # Changed value
            assert config.server.http_port == "9999"  # JSON loads as string
            
            # Defaults should still be present
            assert config.server.metrics_port == "9090"
            assert config.database.url == "postgres://mcp:mcp-password@localhost/mcp?sslmode=disable"
            assert config.integrations.slack.bot_token == ""
        finally:
            os.unlink(temp_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])