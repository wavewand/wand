# 🪄 Wand Integration System

## Overview

The Wand Integration System is a comprehensive magical toolkit providing 55+ integrations across 7 categories. Each integration follows enterprise-grade patterns with authentication management, rate limiting, caching, and error handling.

## Architecture

```
🪄 Wand Integration System
├── 🎯 Core MCP Framework (FastMCP)
├── 🔧 Base Infrastructure
│   ├── BaseIntegration (enterprise patterns)
│   ├── AuthManager (multi-type auth)
│   ├── RateLimiter (token bucket + circuit breaker)
│   ├── CacheManager (multi-level caching)
│   └── ErrorHandler (retry strategies)
├── 🎬 Multimedia (7 integrations)
├── 🤖 AI/ML (7 integrations)
├── 🌐 Productivity (5 integrations)
├── 🛠 DevTools (3 integrations)
├── 🏢 Enterprise (17 integrations)
├── 🔒 Security (6 integrations)
├── 🎮 Specialized (12 integrations)
└── 🔄 Legacy (9 integrations)
```

## Integration Categories

### 🎬 Multimedia Integrations

**Video & Audio Processing**
- `ffmpeg` - Video processing, conversion, streaming
- `opencv` - Computer vision, image processing
- `whisper` - Speech-to-text transcription
- `elevenlabs` - Text-to-speech synthesis

**Image Processing**
- `image` - Image manipulation, filters, effects
- `ocr` - Optical character recognition
- `qr` - QR code generation and scanning

### 🤖 AI/ML Integrations

**Large Language Models**
- `openai` - OpenAI GPT models, embeddings
- `anthropic` - Claude models, message handling
- `cohere` - Cohere language models
- `replicate` - AI model hosting platform

**Specialized AI**
- `huggingface` - Model inference, datasets
- `stability` - Stable Diffusion image generation
- `deepl` - Professional translation services

### 🌐 Productivity Integrations

**Communication**
- `discord` - Discord messaging, server management
- `telegram` - Telegram bot operations
- `email` - Email sending, IMAP/SMTP handling

**Workflow**
- `calendar` - Google Calendar, CalDAV integration
- `notion` - Notion database and page management

### 🛠 DevTools Integrations

**Infrastructure**
- `docker` - Container management, image building
- `kubernetes` - Cluster management, deployments
- `terraform` - Infrastructure as code

### 🏢 Enterprise Integrations

**Identity & Access Management**
- `servicenow` - IT Service Management, incident tracking, user management
- `sailpoint` - Identity Security Cloud, access governance, certification campaigns
- `entra` - Microsoft Entra (Azure AD) user/group/role management
- `britive` - Privileged access management, just-in-time access, secret checkout

**Communication & Collaboration**
- `teams` - Microsoft Teams messaging, cards, notifications via webhooks

**CRM & Sales**
- `salesforce` - Lead management, opportunities
- `hubspot` - Contact management, deals
- `pipedrive` - Sales pipeline management
- `stripe` - Payment processing, subscriptions

**Project Management**
- `jira` - Issue tracking, project management
- `asana` - Task management, team coordination
- `trello` - Kanban boards, card management
- `linear` - Issue tracking for software teams
- `monday` - Work management platform

**HR & Operations**
- `workday` - HR management system
- `bamboohr` - HR information system
- `toggl` - Time tracking and reporting
- `harvest` - Time tracking and invoicing

### 🔒 Security Integrations

**Secret Management**
- `vault` - HashiCorp Vault secret storage
- `onepassword` - 1Password secret management

**Identity Management**
- `okta` - Identity and access management
- `auth0` - Authentication and authorization

**Security Scanning**
- `snyk` - Vulnerability scanning
- `sonarqube` - Code quality and security analysis

### 🎮 Specialized Integrations

**Gaming & Streaming**
- `steam` - Steam platform integration
- `twitch` - Twitch streaming platform
- `discord_bot` - Discord bot development

**IoT & Hardware**
- `arduino` - Arduino microcontroller programming
- `raspberrypi` - Raspberry Pi GPIO control
- `esp` - ESP32/ESP8266 integration
- `mqtt` - MQTT message broker

**Blockchain & Web3**
- `ethereum` - Ethereum blockchain operations
- `bitcoin` - Bitcoin blockchain queries
- `web3` - Multi-chain Web3 operations
- `nft` - NFT and digital collectibles

## MCP Tool Usage

All integrations are available as single-word MCP tools:

```bash
# Enterprise
salesforce create_lead --first_name="John" --last_name="Doe" --company="Acme Corp"
stripe create_payment_intent --amount=2000 --currency="usd"

# Security
vault read_secret --path="secret/app/config"
snyk test_project --project_path="." --severity_threshold="high"

# Gaming & Blockchain
steam get_player_summaries --steam_ids="76561198000000000"
ethereum get_balance --address="0x..." --block="latest"

# IoT & Hardware
arduino send_command --command="digitalWrite(13, HIGH)"
mqtt publish --topic="sensors/temperature" --payload="23.5"
```

## Configuration

### Environment Variables

Each integration uses environment variables for configuration. See `integrations/config/integration_configs.py` for the complete list.

**Key Configuration Patterns:**

```bash
# API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export HUBSPOT_API_KEY="pat-..."

# OAuth Credentials
export SALESFORCE_CLIENT_ID="..."
export SALESFORCE_CLIENT_SECRET="..."
export GOOGLE_CLIENT_ID="..."

# Server URLs
export VAULT_URL="https://vault.company.com"
export SONARQUBE_URL="https://sonar.company.com"
export ELASTICSEARCH_URL="http://localhost:9200"
```

### Configuration Management

The Wand system includes comprehensive configuration management:

```python
from integrations.config.integration_configs import wand_config

# Get configuration summary
status = wand_config.get_configuration_summary()
print(f"Configured: {status['summary']['configured_integrations']}")

# Validate specific integration
is_valid = wand_config.validate_integration_config("openai")

# Save configuration summary
wand_config.save_config_summary("config_status.json")
```

## Installation & Setup

### 1. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# For production (excludes dev tools)
pip install --no-dev -r requirements.txt
```

### 2. Configure Integrations

Create a `.env` file with your API keys and configuration:

```bash
# Copy example configuration
cp .env.example .env

# Edit with your credentials
nano .env
```

### 3. Start the Server

```bash
# HTTP mode (for web interfaces)
python wand.py http

# MCP stdio mode (for MCP clients)
python wand.py stdio
```

### 4. Test Integrations

```bash
# Test system structure
python test_wand_simple.py

# Full integration test (requires dependencies)
python test_wand_integrations.py
```

## Docker Deployment

### Workspace Mode (Recommended)

```bash
# Build and start
docker-compose -f docker-compose.workspace.yml up --build

# Access at http://localhost:8001
```

### Configuration Volume

The Docker setup mounts configuration as a volume:

```yaml
volumes:
  - ./config:/app/config
  - ./logs:/app/logs
  - ${WORKSPACE_DIR:-./workspace}:/workspace
```

## Enterprise Features

### Authentication Management

Supports multiple authentication types:
- API Keys
- OAuth 2.0 / OAuth 1.0
- JWT tokens
- Basic authentication

### Rate Limiting

Token bucket algorithm with circuit breaker pattern:
- Configurable rates per integration
- Automatic backoff on rate limit exceeded
- Circuit breaker for failing services

### Caching System

Multi-level caching with LRU eviction:
- Memory cache for hot data
- Disk cache for persistence
- Configurable TTL per integration

### Error Handling

Comprehensive error handling with retry strategies:
- Exponential backoff
- Jitter for thundering herd prevention
- Circuit breaker integration
- Detailed error classification

## Health Monitoring

Each integration provides health checks:

```python
# Check individual integration
health = await salesforce_integration.health_check()
print(health['status'])  # 'healthy', 'unhealthy', 'degraded'

# System-wide health
status = wand_config.get_system_health()
```

## Development

### Adding New Integrations

1. Create integration class inheriting from `BaseIntegration`
2. Implement required methods: `initialize()`, `cleanup()`, `health_check()`
3. Add operations in `_execute_operation_impl()`
4. Register in appropriate category `__init__.py`
5. Add MCP tool in `wand.py`
6. Update configuration in `integration_configs.py`

### Testing

```bash
# Run integration tests
pytest tests/integration/

# Test specific category
pytest tests/integration/test_multimedia.py

# Test with coverage
pytest --cov=integrations tests/
```

## Security Considerations

- **API Keys**: Store in environment variables, never in code
- **Rate Limiting**: Respect service limits to avoid blocking
- **Data Handling**: Sensitive data is not logged or cached
- **Network Security**: HTTPS/TLS for all external communications
- **Input Validation**: All inputs are validated and sanitized

## Support & Documentation

- **Configuration**: See `integrations/config/integration_configs.py`
- **Examples**: Check `examples/` directory
- **API Reference**: Generated from docstrings in each integration
- **Troubleshooting**: See `TROUBLESHOOTING.md`

---

## Quick Start Examples

### CRM Operations
```bash
# Create lead in Salesforce
salesforce create_lead --first_name="Alice" --last_name="Smith" --company="Tech Corp" --email="alice@techcorp.com"

# Create contact in HubSpot
hubspot create_contact --email="bob@company.com" --first_name="Bob" --last_name="Johnson"
```

### AI/ML Operations
```bash
# Generate text with OpenAI
openai complete --prompt="Write a Python function to calculate fibonacci" --max_tokens=150

# Translate with DeepL
deepl translate --text="Hello world" --target_lang="ES"
```

### Security Operations
```bash
# Read secret from Vault
vault read_secret --path="secret/database/password"

# Scan project with Snyk
snyk test_project --project_path="." --severity_threshold="medium"
```

### Multimedia Operations
```bash
# Convert video with FFmpeg
ffmpeg convert --input="video.mp4" --output="video.webm" --format="webm"

# Generate QR code
qr generate --data="https://example.com" --output_path="qr.png"
```

**🌟 The Wand Integration System transforms your development workflow with magical automation across 50+ services and platforms!**
